import os
import sys
import numpy as np

# Ensure Python can find your 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.pipeline import MatchingPipeline

# Set up the paths
PROCESSED_DIR = r"D:\LunaMatch\LunaMatch\data\processed"
# We will use your Nadir tile_0 as the simulated "uploaded image"
MOCK_UPLOAD_PATH = os.path.join(PROCESSED_DIR, "ch2_tmc_ncn_20260813T0627378557_d_img_d18_tile_0.npy")

def run_test():
    print("1. Simulating user upload...")
    # The backend will load the image into a numpy array
    mock_upload_array = np.load(MOCK_UPLOAD_PATH)
    
    print("2. Initializing your MatchingPipeline API...")
    # The backend instantiates your class pointing to the database
    pipeline = MatchingPipeline(PROCESSED_DIR)
    
    print("3. Calling process_upload()...")
    # The backend passes the array to your function
    response = pipeline.process_upload(mock_upload_array)
    
    # 4. Displaying the dictionary your code returns
    print("\n--- API RESPONSE DICTIONARY ---")
    if response["success"]:
        print("Status: SUCCESS")
        print(f"Matched Database File: {response['matched_file']}")
        print(f"Valid Coordinates Found: {response['match_count']}")
        
        # Verify the arrays are formatted correctly for Vivek
        print(f"pts_src array shape: {response['pts_src'].shape}")
        print(f"pts_ref array shape: {response['pts_ref'].shape}")
        print("\nTest passed! The backend team can now safely integrate this.")
    else:
        print("Status: FAILED")
        print(f"Error Message: {response['error']}")

if __name__ == "__main__":
    run_test()