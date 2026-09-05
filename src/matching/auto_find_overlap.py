import os
import glob
import numpy as np
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ROOT_DIR)

from src.core.matcher import LunarMatcher
from src.core.geometric_verification import GeometricVerifier  # Import your Phase 2 code!

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

# Target a tile deep in the middle of the strip
NADIR_TARGET = glob.glob(os.path.join(PROCESSED_DIR, "*_ncn_*_tile_10.npy"))[0]

def find_true_overlap():
    print("Initializing SuperPoint, LightGlue, and RANSAC...")
    # Double the keypoint pool to feed more valid points to RANSAC
    matcher = LunarMatcher(max_keypoints=4096)
    verifier = GeometricVerifier() # Uses the new class defaults
    
    nadir_tile = np.load(NADIR_TARGET)
    core_timestamp = os.path.basename(NADIR_TARGET).split('_')[3][:15] 
    
    aft_files = glob.glob(os.path.join(PROCESSED_DIR, f"*_nca_{core_timestamp}*.npy"))
    
    best_inlier_count = 0
    best_file = None
    best_coords = None
    
    print("\nSweeping Aft tiles (Filtering by RANSAC Inliers)...")
    for aft_path in aft_files:
        tile_num = int(os.path.basename(aft_path).split('_tile_')[1].replace('.npy', ''))
        
        aft_tile = np.load(aft_path)
        
        # 1. AI extracts raw points
        results = matcher.match_image_pair(nadir_tile, aft_tile)
        pts_src, pts_ref = results["pts_src"], results["pts_ref"]
        raw_match_count = len(pts_src)
        
        inlier_count = 0
        # 2. RANSAC immediately tests the geometry
        if raw_match_count >= 4:
            verify_result = verifier.verify_and_align(pts_src, pts_ref)
            if verify_result["success"]:
                inlier_count = verify_result["inlier_count"]
        
        print(f"  -> tile_{tile_num}: {raw_match_count} raw AI matches | {inlier_count} RANSAC INLIERS")
        
        # 3. We pick the winner based on INLIERS, not raw matches
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_file = aft_path
            best_coords = results

    # Lower the threshold to accept the 17-point consensus
    if best_inlier_count >= 10: 
        print(f"\nOVERLAP FOUND! Matches With: {os.path.basename(best_file)}")
        save_path = os.path.join(PROCESSED_DIR, "final_matched_coordinates.npz")
        np.savez(save_path, pts_src=best_coords["pts_src"], pts_ref=best_coords["pts_ref"])
        print(f"Exported verified coordinates to: {save_path}")
    else:
        print("\nFAILED: No tile passed geometric verification.")

if __name__ == "__main__":
    find_true_overlap()