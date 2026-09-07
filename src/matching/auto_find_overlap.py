import os
import glob
import numpy as np
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ROOT_DIR)

from src.core.matcher import LunarMatcher
from src.core.geometric_verification import GeometricVerifier

PROCESSED_BASE = os.path.join(ROOT_DIR, "data", "processed2")

#Directory selection
NADIR_DIR_NAME = "ch2_tmc_ncn_20260813T0627378557_d_img_d18"
NADIR_DIR = os.path.join(PROCESSED_BASE, NADIR_DIR_NAME)

AFT_DIR_NAME = "ch2_tmc_nca_20260813T0627378526_d_img_d18"
AFT_DIR = os.path.join(PROCESSED_BASE, AFT_DIR_NAME)

try:
    NADIR_TARGET = glob.glob(os.path.join(NADIR_DIR, "ch2_tmc_ncn_20260813T0627378557_d_img_d18_tile_241.npy"))[0]
except IndexError:
    print(f"Error: Could not find tile_10.npy in {NADIR_DIR}")
    sys.exit(1)

def find_true_overlap():
    print("Initializing SuperPoint, LightGlue, and RANSAC...")
    #alter max_keypoints according to needs
    matcher = LunarMatcher(max_keypoints=2048)
    verifier = GeometricVerifier() # Uses the new class defaults
    
    nadir_tile = np.load(NADIR_TARGET)
    core_timestamp = os.path.basename(NADIR_TARGET).split('_')[3][:15] 
    
    # Target the AFT Dir
    aft_files = glob.glob(os.path.join(AFT_DIR, f"*_nca_{core_timestamp}*.npy"))
    
    if not aft_files:
        print(f"No Aft tiles found in {AFT_DIR}. Please check if batch_process completed successfully.")
        return

    best_inlier_count = 0
    best_file = None
    best_coords = None
    
    print(f"\nSweeping {len(aft_files)} Aft tiles (Filtering by RANSAC Inliers)...")
    for aft_path in aft_files:
        tile_num = int(os.path.basename(aft_path).split('_tile_')[1].replace('.npy', ''))
        
        aft_tile = np.load(aft_path)
        
        # 1. AI extracts raw points
        results = matcher.match_image_pair(nadir_tile, aft_tile)
        pts_src, pts_ref = results["pts_src"], results["pts_ref"]
        raw_match_count = len(pts_src)
        
        inlier_count = 0
        rmse_val = 0
        # 2. RANSAC immediately tests the geometry
        if raw_match_count >= 50:
            verify_result = verifier.verify_and_align(pts_src, pts_ref)
            if verify_result["success"]:
                inlier_count = verify_result["inlier_count"]
                rmse_val = verify_result["rmse"]
        
        print(f"  -> tile_{tile_num}: {raw_match_count} raw AI matches | {inlier_count} RANSAC INLIERS | RMSE: {rmse_val}")
        
        # 3. We pick the winner based on INLIERS, not raw matches
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_file = aft_path
            best_coords = results

    if best_inlier_count >= 30: 
        print(f"\nOVERLAP FOUND! Matches With: {os.path.basename(best_file)}")
        # Save results in the Nadir directory for easy reference
        save_path = os.path.join(NADIR_DIR, "final_matched_coordinates.npz")
        np.savez(save_path, pts_src=best_coords["pts_src"], pts_ref=best_coords["pts_ref"])
        print(f"Exported verified coordinates to: {save_path}")
    else:
        print("\nFAILED: No tile passed geometric verification.")

if __name__ == "__main__":
    find_true_overlap()


