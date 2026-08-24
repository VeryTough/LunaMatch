import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure Python can find your 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.preprocessing import load_chandrayaan_pds4

# 1. Load the image directly from the XML file into RAM
xml_path = r"D:\LunaMatch\LunaMatch\data\raw\chandrayaan2_samples\ch2_tmc_nca_20260813T0627378526_d_img_d18.xml"
base_img = load_chandrayaan_pds4(xml_path)

# 2. Extract a 1024x1024 patch from the massive array
# Ensure these coordinates actually point to a bright crater, not black space!
crop = base_img[1000:2024, 100:1124] 

# 3. Create the "Live Camera" view (Synthetic Warp + Shadow Shift)
h, w = crop.shape
pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2 = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.0], [w*0.2, h*0.9], [w*0.8, h*1.0]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(crop, matrix, (w, h))

# Simulating harsh lunar lighting change
gradient = np.tile(np.linspace(0.2, 1.8, w), (h, 1))
shifted_illumination = np.clip(warped * gradient, 0, 255).astype(np.uint8)

# 4. Phase 1: Initialize Classical SIFT
sift = cv2.SIFT_create(nfeatures=1000)
kp1, des1 = sift.detectAndCompute(crop, None)
kp2, des2 = sift.detectAndCompute(shifted_illumination, None)

# 5. Match using FLANN and Lowe's Ratio Test
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 6. Visualize the output
vis = cv2.drawMatches(crop, kp1, shifted_illumination, kp2, good_matches, None, 
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 7))
plt.imshow(vis, cmap='gray')
plt.title(f"SIFT Baseline: {len(good_matches)} matches found")
plt.axis('off')
plt.show()

# ... (Keep all the warping and illumination shift code exactly the same) ...

# 3.5 PHASE 2: Apply CLAHE Preprocessing
print("Applying CLAHE Illumination Normalization...")
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Apply CLAHE to both the original crop and the harshly lit warped image
crop_preprocessed = clahe.apply(crop)
shifted_preprocessed = clahe.apply(shifted_illumination)

# 4. Initialize Classical SIFT (Run on the PREPROCESSED images now)
sift = cv2.SIFT_create(nfeatures=1000)
kp1, des1 = sift.detectAndCompute(crop_preprocessed, None)
kp2, des2 = sift.detectAndCompute(shifted_preprocessed, None)

# ... (Keep the FLANN matching and Lowe's ratio test the same) ...

# 6. Visualize the output
vis = cv2.drawMatches(crop_preprocessed, kp1, shifted_preprocessed, kp2, good_matches, None, 
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 7))
plt.imshow(vis, cmap='gray')
plt.title(f"Phase 2 (CLAHE + SIFT): {len(good_matches)} matches found")
plt.axis('off')
plt.show()