"""Janus — Feature Extraction via SAM 3 with Tiling + Live Dashboard.

Upload large GeoTIFFs, select feature types, and watch as SAM 3 processes
tile by tile with live progress and confidence tracking.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import gradio as gr
import geopandas as gpd
import spaces
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
from rasterio.windows import Window
from shapely.geometry import shape

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
MODEL = None
PROCESSOR = None


def get_device():
    """Detect GPU at runtime, not import time."""
    import subprocess
    print(f"[GPU DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[GPU DEBUG] torch.version.cuda = {torch.version.cuda}")
    print(f"[GPU DEBUG] torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}")
    try:
        print(f"[GPU DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}")
    except Exception as e:
        print(f"[GPU DEBUG] torch.cuda.device_count() ERROR: {e}")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        print(f"[GPU DEBUG] nvidia-smi output:\n{result.stdout[:500]}")
        if result.stderr:
            print(f"[GPU DEBUG] nvidia-smi stderr: {result.stderr[:200]}")
    except Exception as e:
        print(f"[GPU DEBUG] nvidia-smi ERROR: {e}")
    try:
        import os
        print(f"[GPU DEBUG] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
        print(f"[GPU DEBUG] NVIDIA_VISIBLE_DEVICES = {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')}")
    except Exception as e:
        print(f"[GPU DEBUG] env check ERROR: {e}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[GPU DEBUG] Using GPU: {name}")
        return "cuda"
    print("[GPU DEBUG] Falling back to CPU")
    return "cpu"

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Feature presets
# ---------------------------------------------------------------------------
FEATURE_PRESETS = {
    "Building": {
        "prompt": "building",
        "min_area": 20.0, "max_area": 50000.0,
        "min_compactness": 0.25, "min_rectangularity": 0.5,
        "color": "#10b981",
    },
    "Rooftop": {
        "prompt": "rooftop",
        "min_area": 20.0, "max_area": 50000.0,
        "min_compactness": 0.25, "min_rectangularity": 0.5,
        "color": "#34d399",
    },
    "Road": {
        "prompt": "road",
        "min_area": 10.0, "max_area": 500000.0,
        "min_compactness": 0.0, "min_rectangularity": 0.0,
        "color": "#f59e0b",
    },
    "Waterbody": {
        "prompt": "water",
        "min_area": 50.0, "max_area": 5000000.0,
        "min_compactness": 0.0, "min_rectangularity": 0.0,
        "color": "#3b82f6",
    },
    "Vegetation": {
        "prompt": "tree",
        "min_area": 30.0, "max_area": 5000000.0,
        "min_compactness": 0.0, "min_rectangularity": 0.0,
        "color": "#22c55e",
    },
    "Parking Lot": {
        "prompt": "parking lot",
        "min_area": 100.0, "max_area": 200000.0,
        "min_compactness": 0.2, "min_rectangularity": 0.4,
        "color": "#8b5cf6",
    },
}

WORLD_EXT_MAP = {
    ".png": ".pgw", ".jpg": ".jgw", ".jpeg": ".jgw",
    ".tif": ".tfw", ".tiff": ".tfw",
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        from transformers import Sam3Model, Sam3Processor
        device = get_device()
        print(f"[MODEL] Loading SAM 3 on {device}...")
        MODEL = Sam3Model.from_pretrained("facebook/sam3").to(device)
        PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
        print(f"[MODEL] SAM 3 loaded on {device}")
    return MODEL, PROCESSOR

# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _parse_world_file(world_path: str) -> Affine:
    lines = Path(world_path).read_text().strip().splitlines()
    if len(lines) < 6:
        raise gr.Error(f"World file must have 6 lines, got {len(lines)}")
    return Affine(
        float(lines[0]), float(lines[2]), float(lines[4]),
        float(lines[1]), float(lines[3]), float(lines[5]),
    )


def ingest(image_path: str, world_path: str | None, crs: str) -> str:
    tmp = tempfile.mkdtemp(prefix="janus_")
    geotiff = os.path.join(tmp, "input.tif")

    if image_path.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(image_path) as src:
                if src.crs is not None and src.transform != Affine.identity():
                    import shutil
                    shutil.copy2(image_path, geotiff)
                    return geotiff
        except Exception:
            pass

    if world_path is None:
        img_p = Path(image_path)
        expected_ext = WORLD_EXT_MAP.get(img_p.suffix.lower(), ".pgw")
        auto_wf = img_p.with_suffix(expected_ext)
        if auto_wf.exists():
            world_path = str(auto_wf)
        else:
            raise gr.Error(f"No world file found. Expected: {auto_wf.name}")

    transform = _parse_world_file(world_path)
    target_crs = CRS.from_user_input(crs)
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w, bands = img_array.shape
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": bands,
        "crs": target_crs, "transform": transform,
    }
    with rasterio.open(geotiff, "w", **profile) as dst:
        for b in range(bands):
            dst.write(img_array[:, :, b], b + 1)
    return geotiff

# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def compute_tile_windows(img_w: int, img_h: int, tile_size: int, overlap: int) -> list[Window]:
    step = tile_size - overlap
    windows = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            tw = min(tile_size, img_w - x)
            th = min(tile_size, img_h - y)
            if tw < tile_size // 4 or th < tile_size // 4:
                continue
            windows.append(Window(x, y, tw, th))
    return windows

# ---------------------------------------------------------------------------
# Per-tile segmentation
# ---------------------------------------------------------------------------

@spaces.GPU
def segment_tile(tile_rgb: np.ndarray, prompt: str, confidence: float):
    model, processor = load_model()
    device = get_device()
    image = Image.fromarray(tile_rgb)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=confidence, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return results["masks"].cpu().numpy(), results["scores"].cpu().numpy()

# ---------------------------------------------------------------------------
# Shape metrics
# ---------------------------------------------------------------------------

def _compactness(geom):
    if geom.is_empty or geom.length == 0:
        return 0.0
    return (4.0 * math.pi * geom.area) / (geom.length ** 2)

def _rectangularity(geom):
    if geom.is_empty:
        return 0.0
    mrr = geom.minimum_rotated_rectangle
    return geom.area / mrr.area if mrr.area > 0 else 0.0

# ---------------------------------------------------------------------------
# Dedup + filter
# ---------------------------------------------------------------------------

def merge_and_deduplicate(features: list[dict], crs_obj, iou_thresh: float = 0.5) -> gpd.GeoDataFrame:
    if not features:
        return gpd.GeoDataFrame(columns=["feature_id", "feature_type", "confidence", "geometry"])
    gdf = gpd.GeoDataFrame(features, crs=crs_obj)
    if len(gdf) == 0:
        return gdf
    sindex = gdf.sindex
    drop = set()
    for idx, row in gdf.iterrows():
        if idx in drop:
            continue
        for cand_idx in sindex.intersection(row.geometry.bounds):
            if cand_idx <= idx or cand_idx in drop:
                continue
            cand = gdf.loc[cand_idx]
            if row["feature_type"] != cand["feature_type"]:
                continue
            if not row.geometry.intersects(cand.geometry):
                continue
            try:
                inter = row.geometry.intersection(cand.geometry).area
                union = row.geometry.union(cand.geometry).area
                if union > 0 and inter / union >= iou_thresh:
                    if row["confidence"] >= cand["confidence"]:
                        drop.add(cand_idx)
                    else:
                        drop.add(idx)
                        break
            except Exception:
                continue
    gdf = gdf.drop(index=drop).reset_index(drop=True)
    gdf["feature_id"] = range(1, len(gdf) + 1)
    return gdf


def filter_features(gdf: gpd.GeoDataFrame, preset: dict) -> gpd.GeoDataFrame:
    if len(gdf) == 0:
        return gdf
    if gdf.crs and gdf.crs.is_geographic:
        proj = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_m2"] = proj.geometry.area
        pg = proj.geometry
    else:
        gdf["area_m2"] = gdf.geometry.area
        pg = gdf.geometry
    gdf = gdf[(gdf["area_m2"] >= preset["min_area"]) & (gdf["area_m2"] <= preset["max_area"])].copy()
    if len(gdf) == 0:
        return gdf
    if gdf.crs and gdf.crs.is_geographic:
        pg = gdf.to_crs(gdf.estimate_utm_crs()).geometry
    else:
        pg = gdf.geometry
    gdf["compactness"] = pg.apply(_compactness)
    if preset["min_compactness"] > 0:
        gdf = gdf[gdf["compactness"] >= preset["min_compactness"]].copy()
    if len(gdf) == 0:
        return gdf
    if preset["min_rectangularity"] > 0:
        if gdf.crs and gdf.crs.is_geographic:
            pg = gdf.to_crs(gdf.estimate_utm_crs()).geometry
        else:
            pg = gdf.geometry
        gdf["rectangularity"] = pg.apply(_rectangularity)
        gdf = gdf[gdf["rectangularity"] >= preset["min_rectangularity"]].copy()
    return gdf

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_all(features_by_type: dict[str, gpd.GeoDataFrame], tmp_dir: str) -> list[str]:
    paths = []
    for feat_type, gdf in features_by_type.items():
        if len(gdf) == 0:
            continue
        name = feat_type.lower().replace(" ", "_")
        gpkg = os.path.join(tmp_dir, f"{name}.gpkg")
        gdf.to_file(gpkg, driver="GPKG")
        paths.append(gpkg)
        gj = os.path.join(tmp_dir, f"{name}.geojson")
        gdf.to_file(gj, driver="GeoJSON")
        paths.append(gj)
        wkt = os.path.join(tmp_dir, f"{name}.wkt")
        with open(wkt, "w") as f:
            for idx, row in enumerate(gdf.itertuples(), start=1):
                c = getattr(row, "confidence", 0.0)
                f.write(f"{idx}|{c:.3f}|{row.geometry.wkt}\n")
        paths.append(wkt)
    return paths

# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def make_overlay(
    geotiff: str,
    features_by_type: dict[str, gpd.GeoDataFrame],
    scanned_bounds: tuple | None = None,
) -> str:
    """Render before/after overlay with optional scan-progress box.

    Args:
        geotiff: Path to the source GeoTIFF.
        features_by_type: Dict of feature_type -> GeoDataFrame.
        scanned_bounds: (left, bottom, right, top) in CRS coordinates showing
                        the area processed so far. Drawn as a dashed box.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    fig.patch.set_facecolor("#fafafa")
    with rasterio.open(geotiff) as src:
        max_dim = 2000
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_w = int(src.width * scale)
        out_h = int(src.height * scale)
        data = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    rgb = np.transpose(data[:3], (1, 2, 0))

    # Left panel: input image
    axes[0].imshow(rgb, extent=extent)
    axes[0].set_title("Input", fontsize=14, fontweight=600, color="#18181b", pad=12)
    axes[0].tick_params(labelsize=7, colors="#71717a")

    # Right panel: image + detected features
    axes[1].imshow(rgb, extent=extent)

    # Draw scan progress box
    if scanned_bounds is not None:
        from matplotlib.patches import Rectangle
        left, bottom, right, top = scanned_bounds
        rect = Rectangle(
            (left, bottom), right - left, top - bottom,
            linewidth=1.5, edgecolor="#71717a", facecolor="none",
            linestyle="--", alpha=0.6,
        )
        axes[1].add_patch(rect)

    # Plot features color-coded by type
    total = 0
    legend_items = []
    for feat_type, gdf in features_by_type.items():
        if len(gdf) == 0:
            continue
        color = FEATURE_PRESETS.get(feat_type, {}).get("color", "#ef4444")
        gdf.plot(ax=axes[1], edgecolor=color, facecolor=color, alpha=0.35, linewidth=1)
        total += len(gdf)
        legend_items.append(f"{feat_type}: {len(gdf)}")

    title = f"{total} Features"
    if legend_items:
        title += f"  ({', '.join(legend_items)})"
    axes[1].set_title(title, fontsize=12, fontweight=600, color="#18181b", pad=12)
    axes[1].tick_params(labelsize=7, colors="#71717a")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color("#e4e4e7")

    plt.tight_layout(pad=2)
    out = os.path.join(os.path.dirname(geotiff), "overlay.png")
    plt.savefig(out, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    return out

# ---------------------------------------------------------------------------
# Dashboard formatting
# ---------------------------------------------------------------------------

class ConfidenceTracker:
    """Track running confidence stats per feature type."""
    def __init__(self):
        self.scores: dict[str, list[float]] = defaultdict(list)

    def add(self, feat_type: str, score: float):
        self.scores[feat_type].append(score)

    def count(self, feat_type: str) -> int:
        return len(self.scores[feat_type])

    def total(self) -> int:
        return sum(len(v) for v in self.scores.values())

    def stats(self, feat_type: str) -> dict:
        s = self.scores[feat_type]
        if not s:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "low": 0}
        return {
            "count": len(s),
            "avg": sum(s) / len(s),
            "min": min(s),
            "max": max(s),
            "low": sum(1 for x in s if x < 0.5),
        }


def format_dashboard(
    image_name: str,
    img_w: int,
    img_h: int,
    tile_idx: int,
    total_tiles: int,
    elapsed: float,
    tracker: ConfidenceTracker,
    feature_types: list[str],
    status: str = "Processing",
) -> str:
    pct = (tile_idx / total_tiles * 100) if total_tiles > 0 else 0
    elapsed_str = _fmt_time(elapsed)

    if tile_idx > 0 and tile_idx < total_tiles:
        rate = elapsed / tile_idx
        remaining = rate * (total_tiles - tile_idx)
        remaining_str = _fmt_time(remaining)
    else:
        remaining_str = "--"

    md = f"**{status}: {image_name}** ({img_w:,} x {img_h:,} px)\n\n"
    md += f"Tile {tile_idx} / {total_tiles} | {pct:.0f}% | "
    md += f"Elapsed: {elapsed_str} | Remaining: ~{remaining_str}\n\n"

    md += "| Feature | Count | Avg Conf | Min Conf | Max Conf | Low (<0.5) |\n"
    md += "|---------|-------|----------|----------|----------|------------|\n"

    for ft in feature_types:
        st = tracker.stats(ft)
        if st["count"] > 0:
            md += (
                f"| {ft} | {st['count']} | {st['avg']:.2f} | "
                f"{st['min']:.2f} | {st['max']:.2f} | {st['low']} |\n"
            )
        else:
            md += f"| {ft} | 0 | -- | -- | -- | -- |\n"

    return md


def format_final_summary(
    tracker: ConfidenceTracker,
    feature_types: list[str],
    final_counts: dict[str, int],
    elapsed: float,
) -> str:
    md = f"**Extraction Complete** | Total time: {_fmt_time(elapsed)}\n\n"

    md += "| Feature | Raw | Final | Avg Conf | Min | Max | Low (<0.5) |\n"
    md += "|---------|-----|-------|----------|-----|-----|------------|\n"

    for ft in feature_types:
        st = tracker.stats(ft)
        final = final_counts.get(ft, 0)
        if st["count"] > 0:
            md += (
                f"| {ft} | {st['count']} | {final} | "
                f"{st['avg']:.2f} | {st['min']:.2f} | {st['max']:.2f} | {st['low']} |\n"
            )
        else:
            md += f"| {ft} | 0 | 0 | -- | -- | -- | -- |\n"

    md += "\n*Adjust the confidence threshold and re-run to improve results.*"
    return md


def _fmt_time(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Main pipeline (generator for live updates)
# ---------------------------------------------------------------------------

PREVIEW_INTERVAL = 10  # update overlay every N tiles
CHECKPOINT_INTERVAL = 100  # save downloadable files every N tiles


def run_pipeline(
    image_file,
    world_file,
    crs: str,
    feature_types: list[str],
    confidence: float,
    tile_size: int,
):
    """Generator: yields (overlay_image, dashboard_md, files) after each tile."""
    if image_file is None:
        raise gr.Error("Please upload an aerial image.")
    if not feature_types:
        raise gr.Error("Select at least one feature type.")

    start_time = time.time()
    tracker = ConfidenceTracker()
    all_features: list[dict] = []
    tmp_dir = tempfile.mkdtemp(prefix="janus_")

    # -- Ingest --
    yield None, "**Ingesting image...**", None
    geotiff = ingest(image_file, world_file, crs)

    # Model loads on first segment_tile call (inside @spaces.GPU context)
    yield None, "**Starting extraction...** (model loads on first tile)", None

    # -- Compute tiles --
    with rasterio.open(geotiff) as src:
        img_w, img_h = src.width, src.height
        crs_obj = src.crs
        transform = src.transform

    image_name = Path(image_file).name
    overlap = tile_size // 8
    windows = compute_tile_windows(img_w, img_h, tile_size, overlap)
    total_tiles = len(windows)

    dash = format_dashboard(image_name, img_w, img_h, 0, total_tiles, 0, tracker, feature_types, "Starting")
    yield None, dash, None

    # -- Process tiles --
    # Track the scanned region for progress visualization
    scan_left = scan_bottom = float("inf")
    scan_right = scan_top = float("-inf")

    with rasterio.open(geotiff) as src:
        for tile_idx, window in enumerate(windows):
            tile_data = src.read([1, 2, 3], window=window)
            tile_rgb = np.transpose(tile_data, (1, 2, 0))
            tile_transform = rasterio.windows.transform(window, transform)

            # Update scanned bounds
            tile_bounds = rasterio.windows.bounds(window, transform)
            scan_left = min(scan_left, tile_bounds[0])
            scan_bottom = min(scan_bottom, tile_bounds[1])
            scan_right = max(scan_right, tile_bounds[2])
            scan_top = max(scan_top, tile_bounds[3])

            for ft in feature_types:
                preset = FEATURE_PRESETS[ft]
                masks, scores = segment_tile(tile_rgb, preset["prompt"], confidence)

                if len(masks) == 0:
                    continue

                labeled = np.zeros((tile_rgb.shape[0], tile_rgb.shape[1]), dtype=np.int32)
                for i, mask in enumerate(masks):
                    labeled[mask > 0] = i + 1

                for geom, val in shapes(labeled, mask=(labeled > 0), transform=tile_transform):
                    val = int(val)
                    score = float(scores[val - 1]) if val - 1 < len(scores) else 0.0
                    tracker.add(ft, score)
                    all_features.append({
                        "geometry": shape(geom),
                        "feature_type": ft,
                        "confidence": round(score, 3),
                    })

            elapsed = time.time() - start_time
            dash = format_dashboard(
                image_name, img_w, img_h,
                tile_idx + 1, total_tiles,
                elapsed, tracker, feature_types,
            )

            # Checkpoint: save downloadable files every N tiles
            # Update overlay + save files every PREVIEW_INTERVAL tiles
            if (tile_idx + 1) % PREVIEW_INTERVAL == 0 or tile_idx == total_tiles - 1 or tile_idx == 0:
                temp_by_type = {}
                for ft in feature_types:
                    ft_feats = [f for f in all_features if f["feature_type"] == ft]
                    if ft_feats:
                        temp_by_type[ft] = gpd.GeoDataFrame(ft_feats, crs=crs_obj)

                # Save files every update so they're always downloadable
                checkpoint_files = None
                if temp_by_type:
                    checkpoint_files = export_all(temp_by_type, tmp_dir)

                scanned = (scan_left, scan_bottom, scan_right, scan_top)
                overlay = make_overlay(geotiff, temp_by_type, scanned_bounds=scanned)
                yield overlay, dash, checkpoint_files
            else:
                yield gr.update(), dash, gr.update()

    # -- Deduplicate + filter --
    elapsed = time.time() - start_time
    yield gr.update(), format_dashboard(
        image_name, img_w, img_h, total_tiles, total_tiles,
        elapsed, tracker, feature_types, "Deduplicating & filtering",
    ), None

    final_by_type: dict[str, gpd.GeoDataFrame] = {}
    final_counts: dict[str, int] = {}

    for ft in feature_types:
        ft_feats = [f for f in all_features if f["feature_type"] == ft]
        if not ft_feats:
            final_counts[ft] = 0
            continue
        gdf = merge_and_deduplicate(ft_feats, crs_obj)
        gdf = filter_features(gdf, FEATURE_PRESETS[ft])
        final_by_type[ft] = gdf
        final_counts[ft] = len(gdf)

    # -- Export --
    export_paths = export_all(final_by_type, tmp_dir)

    # -- Final overlay --
    final_overlay = make_overlay(geotiff, final_by_type)

    elapsed = time.time() - start_time
    final_dash = format_final_summary(tracker, feature_types, final_counts, elapsed)

    yield final_overlay, final_dash, export_paths


# ---------------------------------------------------------------------------
# Gradio UI
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

* { font-family: 'Outfit', system-ui, -apple-system, sans-serif !important; }
code, pre, .code, [class*="mono"] { font-family: 'JetBrains Mono', monospace !important; }

.gradio-container { max-width: 1400px !important; margin: 0 auto !important; background: #fafafa !important; }

.gr-button-primary {
    border-radius: 12px !important; font-weight: 600 !important;
    letter-spacing: -0.01em !important; padding: 12px 32px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
.gr-button-primary:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; }
.gr-button-primary:active { transform: translateY(0) scale(0.98) !important; }

.gr-panel, .gr-box, .gr-form {
    border-radius: 20px !important; border: 1px solid #e4e4e7 !important;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.04) !important;
}

.gr-input, .gr-textbox textarea, select {
    border-radius: 12px !important; border: 1px solid #e4e4e7 !important;
    font-size: 14px !important; transition: border-color 0.2s ease !important;
}
.gr-input:focus, .gr-textbox textarea:focus {
    border-color: #18181b !important; box-shadow: 0 0 0 3px rgba(24,24,27,0.06) !important;
}

h1 { font-size: 2.25rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; line-height: 1.1 !important; color: #18181b !important; }
h2, h3, .gr-block-label { font-weight: 600 !important; letter-spacing: -0.02em !important; color: #3f3f46 !important; }
.markdown-text p { color: #52525b !important; line-height: 1.6 !important; max-width: 65ch !important; }
footer { display: none !important; }
"""

HEADER_MD = """
# Janus [Auto-Synced]
### GIS-ready feature extraction from aerial imagery

Upload a GeoTIFF (any size), select feature types to extract, and watch SAM 3 process
tile by tile with live confidence tracking.
"""

def _gpu_status():
    try:
        if torch.cuda.is_available():
            return (
                f"Running on **{torch.cuda.get_device_name(0)}** "
                f"({torch.cuda.get_device_properties(0).total_memory / (1024**3):.0f} GB)"
            )
    except Exception:
        pass
    return "No GPU detected — inference will be slow"

gpu_status = _gpu_status()

with gr.Blocks(css=CUSTOM_CSS, title="Janus — Feature Extraction") as demo:
    gr.Markdown(HEADER_MD)
    gr.Markdown(f"*{gpu_status}*")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Group():
                image_input = gr.File(
                    label="Aerial Image",
                    file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                    type="filepath",
                )
                world_input = gr.File(
                    label="World File (optional for GeoTIFF)",
                    file_types=[".pgw", ".jgw", ".tfw"],
                    type="filepath",
                )

            crs_input = gr.Textbox(value="EPSG:4326", label="CRS")

            feature_checks = gr.CheckboxGroup(
                choices=list(FEATURE_PRESETS.keys()),
                value=["Building"],
                label="Feature Types",
                info="Select one or more feature types to extract",
            )

            with gr.Row():
                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.5, step=0.05,
                    label="Confidence Threshold",
                )
                tile_size_dropdown = gr.Dropdown(
                    choices=[256, 512, 1024, 2048],
                    value=256,
                    label="Tile Size (px)",
                    info="Larger = fewer tiles but more GPU memory",
                )

            run_btn = gr.Button("Extract Features", variant="primary", size="lg")

        with gr.Column(scale=5):
            dashboard = gr.Markdown(value="*Upload an image and click Extract Features to begin.*")
            output_image = gr.Image(label="Live Preview", type="filepath", show_download_button=True)
            file_output = gr.Files(label="Download GIS Files")

    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, world_input, crs_input, feature_checks, confidence_slider, tile_size_dropdown],
        outputs=[output_image, dashboard, file_output],
    )

if __name__ == "__main__":
    demo.launch()
