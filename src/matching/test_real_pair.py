import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure Python can find your 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.matcher import LunarMatcher

# 1. Load the real-world tiles from your processed folder
# Update these strings with the exact filenames generated in your data/processed/ folder
# Ensure both tiles end in "_tile_0.npy" so you are looking at the exact same slice!
NADIR_PATH = r"D:\LunaMatch\LunaMatch\data\processed\ch2_tmc_ncn_20260813T0627378557_d_img_d18_tile_0.npy"
AFT_PATH = r"D:\LunaMatch\LunaMatch\data\processed\ch2_tmc_nca_20260813T1023298778_d_img_d18_tile_26.npy"

print("Loading data...")
nadir_tile = np.load(NADIR_PATH)
aft_tile = np.load(AFT_PATH)

# 2. Initialize your AI Matcher
print("Initializing SuperPoint and LightGlue...")
matcher = LunarMatcher(max_keypoints=1024)

# 3. Execute the Handoff Interface
print("Matching real Nadir and Aft views...")
results = matcher.match_image_pair(nadir_tile, aft_tile)

pts_src = results["pts_src"]
pts_ref = results["pts_ref"]

print(f"Success! Found {len(pts_src)} real cross-camera matches.")

# 4. Visualize the final output
# 4. Visualize the final output (Raw Comparison + Matched View)
# Create OpenCV KeyPoints and DMatches for drawing
kp1 = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in pts_src]
kp2 = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in pts_ref]
matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(pts_src))]

# Generate the image with match lines
vis_matches = cv2.drawMatches(nadir_tile, kp1, aft_tile, kp2, matches, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                              matchColor=(0, 255, 0))

# Generate the clean side-by-side image
vis_raw = np.hstack((nadir_tile, aft_tile))

# Set up a plot with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

# Top Plot: Clean side-by-side
ax1.imshow(vis_raw, cmap='gray')
ax1.set_title("Raw Camera Views (Nadir | Aft) - Notice the perspective and lighting differences")
ax1.axis('off')

# Bottom Plot: The AI Matches
ax2.imshow(vis_matches, cmap='gray')
ax2.set_title(f"SuperPoint + LightGlue Connections: {len(pts_src)} matches")
ax2.axis('off')

plt.tight_layout()
plt.show()