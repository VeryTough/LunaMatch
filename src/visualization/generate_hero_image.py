import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import CheckButtons

# --- BULLETPROOF DYNAMIC PATHS ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
<<<<<<< HEAD
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed2")
=======
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4

def find_one(pattern):
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    return matches[0]

<<<<<<< HEAD
NADIR_IMG_PATH = find_one(os.path.join(PROCESSED_DIR, "ch2_tmc_ncn_20260813T0627378557_d_img_d18","ch2_tmc_ncn_20260813T0627378557_d_img_d18_tile_241.npy"))
AFT_IMG_PATH = find_one(os.path.join(PROCESSED_DIR, "ch2_tmc_nca_20260813T0627378526_d_img_d18", "ch2_tmc_nca_20260813T0627378526_d_img_d18_tile_211.npy"))
COORDS_PATH = os.path.join(PROCESSED_DIR, "ch2_tmc_ncn_20260813T0627378557_d_img_d18", "final_matched_coordinates.npz")
=======
NADIR_IMG_PATH = find_one(os.path.join(PROCESSED_DIR, "*_ncn_*_tile_10.npy"))
AFT_IMG_PATH = find_one(os.path.join(PROCESSED_DIR, "*_nca_*_tile_24.npy"))
COORDS_PATH = os.path.join(PROCESSED_DIR, "final_matched_coordinates.npz")
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4

OUTPUT_PATH = "LunaMatch_Interactive_MatchMap.png"
EPIPOLAR_THRESH = 1.0  

# Clean, muted palette
BG = "#0c0e12"
NADIR_ACCENT = "#6fa8dc"   
AFT_ACCENT = "#e8a262"     
LINE_COLOR = "#8f98a8"     

def load_gray_u8(path):
    raw = np.load(path)
    return cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def compute_inliers(pts_src, pts_ref, thresh):
    F, mask = cv2.findFundamentalMat(
        pts_src.astype(np.float64), pts_ref.astype(np.float64),
        cv2.USAC_MAGSAC, thresh, 0.99, 2000
    )
    if F is None or mask is None:
        raise RuntimeError("Fundamental matrix fit failed -- check input points.")
    return np.where(mask.ravel() == 1)[0]

def create_interactive_match_map():
    print("Loading imagery and filtering for true 3D correspondences...")
    nadir_img = load_gray_u8(NADIR_IMG_PATH)
    aft_img = load_gray_u8(AFT_IMG_PATH)
    
    data = np.load(COORDS_PATH)
    raw_src, raw_ref = data["pts_src"], data["pts_ref"]
    
    inlier_idx = compute_inliers(raw_src, raw_ref, EPIPOLAR_THRESH)
    pts_src, pts_ref = raw_src[inlier_idx], raw_ref[inlier_idx]

    # Set up the figure with extra space at the bottom for the toggle switch
    fig, (ax_nadir, ax_aft) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor=BG)
    fig.subplots_adjust(wspace=0.04, bottom=0.15)

    for ax, img, title, accent in [
        (ax_nadir, nadir_img, "NADIR — Reference View", NADIR_ACCENT),
        (ax_aft, aft_img, "AFT — 25° Baseline View", AFT_ACCENT),
    ]:
        ax.imshow(img, cmap='gray')
        ax.set_title(title, color=accent, fontsize=13, fontweight='medium', loc='left', pad=8)
        ax.axis('off')
        ax.set_facecolor(BG)

    # Store references to lines and points so the toggle function can control them
    connection_lines = []
    for i in range(len(pts_src)):
        conn = ConnectionPatch(
            xyA=pts_src[i], xyB=pts_ref[i], coordsA="data", coordsB="data",
            axesA=ax_nadir, axesB=ax_aft,
            color=LINE_COLOR, linewidth=0.5, alpha=0.35, zorder=2
        )
        fig.add_artist(conn)
        connection_lines.append(conn)

    scatter_nadir = ax_nadir.scatter(pts_src[:, 0], pts_src[:, 1], s=10, color=NADIR_ACCENT, alpha=0.85, zorder=3)
    scatter_aft = ax_aft.scatter(pts_ref[:, 0], pts_ref[:, 1], s=10, color=AFT_ACCENT, alpha=0.85, zorder=3)

    fig.suptitle("LunaMatch — Verified Sub-Pixel Correspondences", color="#e8e8e8",
                 fontsize=18, fontweight='medium', y=0.96)

    # --- ADD INTERACTIVE TOGGLE WIDGET ---
    # Position for the checkbox box [left, bottom, width, height]
    ax_box = fig.add_axes([0.40, 0.03, 0.20, 0.08])
    ax_box.set_facecolor(BG)
    
    checkbox = CheckButtons(
        ax=ax_box,
        labels=[' Show Overlay (Points & Lines)'],
        actives=[True]
    )
    
    # Style the checkbox text to match the dark theme
    for text in checkbox.labels:
        text.set_color('#e8e8e8')
        text.set_fontsize(11)

    # Define the toggle event handler
    def toggle_visibility(label):
        is_visible = not scatter_nadir.get_visible()
        
        # Toggle scatter points
        scatter_nadir.set_visible(is_visible)
        scatter_aft.set_visible(is_visible)
        
        # Toggle all connecting lines
        for conn in connection_lines:
            conn.set_visible(is_visible)
            
        fig.canvas.draw_idle()

    checkbox.on_clicked(toggle_visibility)

    print("SUCCESS! Interactive map loaded. Use the toggle box at the bottom of the window.")
    plt.show()

if __name__ == "__main__":
    create_interactive_match_map()