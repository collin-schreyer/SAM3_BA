import os
from gradio_client import Client, handle_file

# Make sure you have your Hugging Face token exported as an environment variable, 
# or paste it below if the space is private.
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def test_building_detection(image_path: str, prompt: str):
    """
    Connects to the hosted SAM3_BA agent and extracts features
    using the specified prompt (e.g. 'Building' or 'Rooftop').
    """
    if not os.path.exists(image_path):
        print(f"Error: Could not find test image at {image_path}")
        return

    print(f"Connecting to Hugging Face Space 'collzibip/SAM3_BA'...")
    client = Client("collzibip/SAM3_BA", hf_token=HF_TOKEN)
    
    print(f"[*] Sending '{image_path}' to the agent with feature preset: {prompt}")
    
    try:
        # Call the default /predict endpoint
        result = client.predict(
            image_file=handle_file(image_path),
            world_file=None,             
            crs="EPSG:4326",
            feature_types=[prompt],       # E.g. 'Building' or 'Rooftop'
            confidence=0.5,              
            tile_size=256,               
            api_name="/predict"          
        )
        
        output_image_path, dashboard_text, exported_gis_files = result
        
        print("\n=== Agent Result Dashbaord ===")
        print(dashboard_text)
        print(f"\n[+] Extraction Complete! Downloadable GIS files saved to:")
        print(exported_gis_files)
        
    except Exception as e:
        print(f"\n[!] Error connecting to the agent: {e}")

if __name__ == "__main__":
    # Example usage: Replace 'my_test_image.tif' with your actual aerial image
    TEST_IMAGE = "my_test_image.tif" 
    
    # Run the test checking for typical Buildings
    # test_building_detection(TEST_IMAGE, "Building")
    
    # Run the test specifically looking for Rooftops
    # test_building_detection(TEST_IMAGE, "Rooftop")
    
    print("Edit test_agent.py with your test image path and uncomment the runs to begin testing!")
