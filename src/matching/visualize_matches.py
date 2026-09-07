import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ROOT_DIR)
from src.core.geometric_verification import GeometricVerifier


# --- CONFIGURATION ---
PROCESSED_BASE = os.path.join(ROOT_DIR, "data", "processed2")

# The Nadir directory and specific tile you tested
NADIR_DIR = os.path.join(PROCESSED_BASE, "ch2_tmc_ncn_20260813T0627378557_d_img_d18")
NADIR_TILE_PATH = os.path.join(NADIR_DIR, "ch2_tmc_ncn_20260813T0627378557_d_img_d18_tile_10.npy")

# The specific Aft tile that triggered the "OVERLAP FOUND" message
AFT_TILE_PATH = os.path.join(PROCESSED_BASE, "ch2_tmc_nca_20260813T0627378526_d_img_d18", "ch2_tmc_nca_20260813T0627378526_d_img_d18_tile_413.npy")
COORDS_PATH = os.path.join(NADIR_DIR, "final_matched_coordinates.npz")


def plot_verified_matches():
    if not os.path.exists(COORDS_PATH):
        print(f"Error: Coordinates file not found at {COORDS_PATH}")
        return

    print("Loading image data and coordinates...")
    nadir_img = np.load(NADIR_TILE_PATH)
    aft_img = np.load(AFT_TILE_PATH)
    
    # Load the raw matches
    match_data = np.load(COORDS_PATH)
    raw_pts_nadir = match_data['pts_src']
    raw_pts_aft = match_data['pts_ref']
    
    print(f"Loaded {len(raw_pts_nadir)} raw AI matches. Filtering for inliers...")
    
    # Run RANSAC using YOUR class
    verifier = GeometricVerifier()
    verify_result = verifier.verify_and_align(raw_pts_nadir, raw_pts_aft)
    
    if not verify_result["success"]:
        print("RANSAC failed on these points. No inliers to plot.")
        return
        
    # Now this will work!
    inlier_mask = verify_result["inliers"].flatten()
    
    # Apply the mask to keep ONLY the points where RANSAC returned 1 (True)
    inlier_pts_nadir = raw_pts_nadir[inlier_mask == 1]
    inlier_pts_aft = raw_pts_aft[inlier_mask == 1]
    
    inlier_count = len(inlier_pts_nadir)
    print(f"Plotting {inlier_count} verified RANSAC inliers.")
    
    # --- Plotting ---
    h1, w1 = nadir_img.shape[:2]
    combined_img = np.concatenate((nadir_img, aft_img), axis=1)
    
    plt.figure(figsize=(18, 9))
    plt.imshow(combined_img, cmap='gray')
    plt.title(f"Strict Geometric Verification: {inlier_count} RANSAC Inliers", fontsize=16, pad=15)
    
    for i in range(inlier_count):
        x1, y1 = inlier_pts_nadir[i]
        x2, y2 = inlier_pts_aft[i]
        
        x2_shifted = x2 + w1
        
        # Connecting line (Lime Green)
        plt.plot([x1, x2_shifted], [y1, y2], color='lime', linewidth=1.2, alpha=0.6)
        
        # Keypoints (Red with white borders)
        plt.scatter(x1, y1, color='red', s=20, edgecolors='white', linewidths=0.5, zorder=5)
        plt.scatter(x2_shifted, y2, color='red', s=20, edgecolors='white', linewidths=0.5, zorder=5)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_verified_matches()