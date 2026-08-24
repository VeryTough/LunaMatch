import os
import glob
import numpy as np
import sys

# Ensure Python can find your 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.matcher import LunarMatcher

PROCESSED_DIR = r"D:\LunaMatch\LunaMatch\data\processed"

# The Nadir tile we want to find a match for
NADIR_TARGET = os.path.join(PROCESSED_DIR, "ch2_tmc_ncn_20260813T0627378557_d_img_d18_tile_2.npy")

def find_true_overlap():
    print("Initializing SuperPoint and LightGlue...")
    matcher = LunarMatcher(max_keypoints=1024)
    
    print(f"\nLoading Target: {os.path.basename(NADIR_TARGET)}")
    nadir_tile = np.load(NADIR_TARGET)
    
    # Grab every Aft tile we generated
    aft_files = glob.glob(os.path.join(PROCESSED_DIR, "*_nca_*.npy"))
    
    best_match_count = 0
    best_aft_file = None
    best_coordinates = None
    
    print("\nSweeping through Aft tiles to find the physical overlap...")
    for aft_path in aft_files:
        aft_tile = np.load(aft_path)
        
        # Feed the pair to the AI
        results = matcher.match_image_pair(nadir_tile, aft_tile)
        match_count = len(results["pts_src"])
        
        print(f"  -> {os.path.basename(aft_path)}: {match_count} matches")
        
        # Keep track of the highest spike in matches
        if match_count > best_match_count:
            best_match_count = match_count
            best_aft_file = aft_path
            best_coordinates = results

    print("\n" + "="*50)
    # A threshold of 20+ usually means a true physical overlap, not just noise
    if best_match_count > 20: 
        print(f"OVERLAP FOUND!")
        print(f"Target Nadir: tile_0")
        print(f"Matches With: {os.path.basename(best_aft_file)}")
        print(f"Total Valid Coordinates: {best_match_count}")
        
        # This is where we save the exact dictionary output for Vivek
        save_path = os.path.join(PROCESSED_DIR, "final_matched_coordinates.npz")
        np.savez(save_path, 
                 pts_src=best_coordinates["pts_src"], 
                 pts_ref=best_coordinates["pts_ref"])
        print(f"Exported coordinates for RANSAC to: {save_path}")
        
    else:
        print("FAILED: No tile contained enough overlapping geometry.")

if __name__ == "__main__":
    find_true_overlap()