"""Janus — Building Footprint Extraction via SAM 3.

A Gradio app for extracting GIS-ready building footprints from aerial imagery
using Meta's Segment Anything Model 3 with text prompts.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import gradio as gr
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from affine import Affine
from PIL import Image
from pyproj import CRS
from rasterio.features import shapes
from rasterio.plot import show
from shapely.geometry import shape

# ---------------------------------------------------------------------------
# Global model state (loaded once, reused across requests)
# ---------------------------------------------------------------------------
MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

matplotlib.use("Agg")


def load_model():
    """Load SAM 3 model and processor (cached globally)."""
    global MODEL, PROCESSOR
    if MODEL is None:
        from transformers import Sam3Model, Sam3Processor

        MODEL = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
        PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
    return MODEL, PROCESSOR


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

WORLD_EXT_MAP = {
    ".png": ".pgw",
    ".jpg": ".jgw",
    ".jpeg": ".jgw",
    ".tif": ".tfw",
    ".tiff": ".tfw",
}


def _parse_world_file(world_path: str) -> Affine:
    """Parse a 6-line world file into a rasterio Affine transform.

    World file format:
      line 0: x pixel size (a)
      line 1: rotation (d)
      line 2: rotation (b)
      line 3: y pixel size, negative (e)
      line 4: x origin - upper left pixel center (c)
      line 5: y origin - upper left pixel center (f)

    Affine(a, b, c, d, e, f)
    """
    lines = Path(world_path).read_text().strip().splitlines()
    if len(lines) < 6:
        raise gr.Error(f"World file must have 6 lines, got {len(lines)}")
    a = float(lines[0])  # x pixel size
    d = float(lines[1])  # rotation
    b = float(lines[2])  # rotation
    e = float(lines[3])  # y pixel size (negative)
    c = float(lines[4])  # x origin
    f = float(lines[5])  # y origin
    return Affine(a, b, c, d, e, f)


def ingest(image_path: str, world_path: str | None, crs: str) -> str:
    """Convert raw image + optional world file to GeoTIFF using rasterio."""
    tmp = tempfile.mkdtemp(prefix="janus_")
    geotiff = os.path.join(tmp, "input.tif")

    # Check if input is already a GeoTIFF
    if image_path.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(image_path) as src:
                if src.crs is not None and src.transform != Affine.identity():
                    # Already georeferenced — copy with target CRS
                    import shutil
                    shutil.copy2(image_path, geotiff)
                    return geotiff
        except Exception:
            pass

    # Need a world file for raw images
    if world_path is None:
        # Try to find it automatically next to the image
        img_p = Path(image_path)
        expected_ext = WORLD_EXT_MAP.get(img_p.suffix.lower(), ".pgw")
        auto_wf = img_p.with_suffix(expected_ext)
        if auto_wf.exists():
            world_path = str(auto_wf)
        else:
            raise gr.Error(
                f"No world file provided and none found next to image. "
                f"Expected: {auto_wf.name}"
            )

    # Parse world file
    transform = _parse_world_file(world_path)
    target_crs = CRS.from_user_input(crs)

    # Read image with PIL and write as GeoTIFF
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)  # (H, W, 3)

    height, width, bands = img_array.shape

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": bands,
        "crs": target_crs,
        "transform": transform,
    }

    with rasterio.open(geotiff, "w", **profile) as dst:
        for band in range(bands):
            dst.write(img_array[:, :, band], band + 1)

    return geotiff


def segment(geotiff: str, text_prompt: str, confidence: float) -> tuple[str, int]:
    """Run SAM 3 text-prompted segmentation. Returns (mask_path, count)."""
    model, processor = load_model()

    image = Image.open(geotiff).convert("RGB")
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=confidence,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"].cpu().numpy()
    count = len(masks)

    # Write labeled mask raster preserving georeferencing
    mask_path = geotiff.replace("input.tif", "masks.tif")
    with rasterio.open(geotiff) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="int32", nodata=0)
        labeled = np.zeros((src.height, src.width), dtype=np.int32)
        for i, mask in enumerate(masks):
            labeled[mask > 0] = i + 1
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(labeled, 1)

    return mask_path, count


def vectorize(mask_path: str) -> gpd.GeoDataFrame:
    """Convert raster mask to GeoDataFrame of polygons."""
    with rasterio.open(mask_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

        polys, vals = [], []
        for geom, val in shapes(data, mask=(data > 0), transform=transform):
            polys.append(shape(geom))
            vals.append(int(val))

    return gpd.GeoDataFrame({"building_id": vals, "geometry": polys}, crs=crs)


def filter_buildings(gdf: gpd.GeoDataFrame, min_area: float) -> gpd.GeoDataFrame:
    """Filter polygons by area and compactness."""
    if len(gdf) == 0:
        return gdf

    if gdf.crs and gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = gdf_proj.geometry.area
    else:
        gdf["area_m2"] = gdf.geometry.area

    gdf = gdf[gdf["area_m2"] >= min_area].copy()

    def compactness(geom):
        if geom.is_empty or geom.length == 0:
            return 0.0
        return (4.0 * math.pi * geom.area) / (geom.length**2)

    gdf["compactness"] = gdf.geometry.apply(compactness)
    gdf = gdf[gdf["compactness"] >= 0.15].copy()
    return gdf


def export_files(gdf: gpd.GeoDataFrame, tmp_dir: str) -> list[str]:
    """Export to GeoPackage, GeoJSON, and WKT. Return list of file paths."""
    paths = []
    gpkg = os.path.join(tmp_dir, "buildings.gpkg")
    gdf.to_file(gpkg, driver="GPKG")
    paths.append(gpkg)

    geojson = os.path.join(tmp_dir, "buildings.geojson")
    gdf.to_file(geojson, driver="GeoJSON")
    paths.append(geojson)

    wkt_path = os.path.join(tmp_dir, "buildings.wkt")
    with open(wkt_path, "w") as f:
        for idx, row in enumerate(gdf.itertuples(), start=1):
            f.write(f"{idx}|{row.geometry.wkt}\n")
    paths.append(wkt_path)

    return paths


def make_overlay(geotiff: str, gdf: gpd.GeoDataFrame) -> str:
    """Create a comparison overlay image. Returns path to PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    fig.patch.set_facecolor("#fafafa")

    with rasterio.open(geotiff) as src:
        show(src, ax=axes[0])
        show(src, ax=axes[1])

    axes[0].set_title("Input", fontsize=14, fontweight=600, color="#18181b", pad=12)
    axes[0].tick_params(labelsize=7, colors="#71717a")

    if len(gdf) > 0:
        gdf.plot(
            ax=axes[1],
            edgecolor="#10b981",
            facecolor="#10b981",
            alpha=0.35,
            linewidth=1.5,
        )
    axes[1].set_title(
        f"{len(gdf)} Buildings Extracted",
        fontsize=14,
        fontweight=600,
        color="#18181b",
        pad=12,
    )
    axes[1].tick_params(labelsize=7, colors="#71717a")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("#e4e4e7")

    plt.tight_layout(pad=2)
    out = geotiff.replace("input.tif", "comparison.png")
    plt.savefig(out, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def run_pipeline(
    image_file,
    world_file,
    crs: str,
    text_prompt: str,
    confidence: float,
    min_area: float,
    progress=gr.Progress(track_tqdm=True),
):
    """Run the full Janus pipeline end to end."""
    if image_file is None:
        raise gr.Error("Please upload an aerial image.")

    progress(0.05, desc="Ingesting image")
    geotiff = ingest(image_file, world_file, crs)

    progress(0.15, desc="Loading SAM 3 model")
    load_model()

    progress(0.25, desc=f'Segmenting: "{text_prompt}"')
    mask_path, raw_count = segment(geotiff, text_prompt, confidence)

    progress(0.65, desc="Vectorizing masks")
    gdf = vectorize(mask_path)

    progress(0.75, desc="Filtering polygons")
    gdf = filter_buildings(gdf, min_area)
    filtered_count = len(gdf)

    progress(0.85, desc="Exporting files")
    tmp_dir = os.path.dirname(geotiff)
    export_paths = export_files(gdf, tmp_dir)

    progress(0.95, desc="Generating overlay")
    overlay = make_overlay(geotiff, gdf)

    stats = (
        f"**{filtered_count}** buildings extracted "
        f"({raw_count} raw segments, {raw_count - filtered_count} filtered)\n\n"
        f"Min area: {min_area} m\u00b2 | CRS: {crs} | Prompt: \"{text_prompt}\""
    )

    return overlay, stats, export_paths


# ---------------------------------------------------------------------------
# Gradio UI (taste-skill: no emojis, no Inter, no purple, no centered hero)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --body-background-fill: #fafafa !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e4e4e7 !important;
    --block-label-text-color: #3f3f46 !important;
    --block-title-text-color: #18181b !important;
    --button-primary-background-fill: #18181b !important;
    --button-primary-text-color: #fafafa !important;
    --button-primary-background-fill-hover: #27272a !important;
    --input-background-fill: #ffffff !important;
    --border-color-primary: #e4e4e7 !important;
}

* {
    font-family: 'Outfit', system-ui, -apple-system, sans-serif !important;
}

code, pre, .code, [class*="mono"] {
    font-family: 'JetBrains Mono', monospace !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: #fafafa !important;
}

.gr-button-primary {
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    padding: 12px 32px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

.gr-button-primary:active {
    transform: translateY(0) scale(0.98) !important;
}

.gr-panel, .gr-box, .gr-form {
    border-radius: 20px !important;
    border: 1px solid #e4e4e7 !important;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.04) !important;
}

.gr-input, .gr-textbox textarea, select {
    border-radius: 12px !important;
    border: 1px solid #e4e4e7 !important;
    font-size: 14px !important;
    transition: border-color 0.2s ease !important;
}

.gr-input:focus, .gr-textbox textarea:focus {
    border-color: #18181b !important;
    box-shadow: 0 0 0 3px rgba(24,24,27,0.06) !important;
}

h1 {
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.1 !important;
    color: #18181b !important;
}

h2, h3, .gr-block-label {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #3f3f46 !important;
}

.markdown-text p {
    color: #52525b !important;
    line-height: 1.6 !important;
    max-width: 65ch !important;
}

footer { display: none !important; }

.file-preview {
    border-radius: 16px !important;
}
"""

HEADER_MD = """
# Janus
### GIS-ready building footprint extraction from aerial imagery

Upload a high-resolution aerial image with its world file, and Janus will detect
every building using SAM 3 text prompts \u2014 returning clean, georeferenced vector
polygons ready for GIS workflows.
"""

gpu_status = (
    f"Running on **{torch.cuda.get_device_name(0)}** "
    f"({torch.cuda.get_device_properties(0).total_memory / (1024**3):.0f} GB)"
    if torch.cuda.is_available()
    else "No GPU detected \u2014 inference will be slow"
)

with gr.Blocks(css=CUSTOM_CSS, title="Janus \u2014 Building Footprints") as demo:
    # Header (left-aligned per taste-skill rule 3: anti-center bias)
    gr.Markdown(HEADER_MD)
    gr.Markdown(f"*{gpu_status}*")

    with gr.Row(equal_height=False):
        # Left column: inputs (wider)
        with gr.Column(scale=3):
            with gr.Group():
                image_input = gr.File(
                    label="Aerial Image",
                    file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    type="filepath",
                )
                world_input = gr.File(
                    label="World File (optional if GeoTIFF)",
                    file_types=[".pgw", ".jgw", ".tfw"],
                    type="filepath",
                )

            with gr.Row():
                crs_input = gr.Textbox(
                    value="EPSG:4326",
                    label="CRS",
                    info="Coordinate reference system",
                    scale=1,
                )
                prompt_input = gr.Textbox(
                    value="building",
                    label="Text Prompt",
                    info="SAM 3 concept to detect",
                    scale=2,
                )

            with gr.Row():
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                )
                area_slider = gr.Slider(
                    minimum=5,
                    maximum=200,
                    value=20,
                    step=5,
                    label="Min Area (m\u00b2)",
                )

            run_btn = gr.Button(
                "Extract Footprints",
                variant="primary",
                size="lg",
            )

        # Right column: outputs (narrower)
        with gr.Column(scale=4):
            output_image = gr.Image(
                label="Result",
                type="filepath",
                show_download_button=True,
            )
            stats_output = gr.Markdown(label="Summary")
            file_output = gr.Files(label="Download GIS Files")

    # Wire it up
    run_btn.click(
        fn=run_pipeline,
        inputs=[
            image_input,
            world_input,
            crs_input,
            prompt_input,
            confidence_slider,
            area_slider,
        ],
        outputs=[output_image, stats_output, file_output],
    )

if __name__ == "__main__":
    demo.launch()
