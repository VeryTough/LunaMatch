import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from lightglue.utils import rbd

# Ensure Python can find your 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.preprocessing import load_chandrayaan_pds4
from lightglue import LightGlue, SuperPoint

# 1. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# 2. Load Models
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

# 3. Load and Prepare Image Data (Same exact setup as before)
xml_path = r"D:\LunaMatch\LunaMatch\data\raw\chandrayaan2_samples\ch2_tmc_nca_20260813T0627378526_d_img_d18.xml"
base_img = load_chandrayaan_pds4(xml_path)
crop = base_img[1000:2024, 100:1124] 

h, w = crop.shape
pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts2 = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.0], [w*0.2, h*0.9], [w*0.8, h*1.0]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(crop, matrix, (w, h))

gradient = np.tile(np.linspace(0.2, 1.8, w), (h, 1))
shifted_illumination = np.clip(warped * gradient, 0, 255).astype(np.uint8)

# 4. Helper Function to Convert NumPy to PyTorch for LightGlue
def numpy_to_torch(img):
    # LightGlue expects shape (1, 1, H, W) and float32 [0, 1]
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)

image0 = numpy_to_torch(crop)
image1 = numpy_to_torch(shifted_illumination)

# 5. Extract and Match!
with torch.no_grad():
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({'image0': feats0, 'image1': feats1})

# 6. Parse Results back to NumPy for OpenCV Visualization
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
# rbd removes the batch dimension

matches = matches01['matches']  # (M, 2) tensor of indices
pts0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
pts1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

# 7. Convert points to cv2.KeyPoint objects to use drawMatches
kp1_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in pts0]
kp2_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in pts1]
good_matches_cv = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(pts0))]

vis = cv2.drawMatches(crop, kp1_cv, shifted_illumination, kp2_cv, good_matches_cv, None, 
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                      matchColor=(0, 255, 0)) # Force green lines for AI matches!

plt.figure(figsize=(15, 7))
plt.imshow(vis, cmap='gray')
plt.title(f"Phase 3 (SuperPoint + LightGlue): {len(pts0)} matches found")
plt.axis('off')
plt.show()