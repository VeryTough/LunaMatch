# LunaMatch: Chandrayaan-2 Orbital Image Registration

LunaMatch is a high-precision computer vision pipeline designed to accurately register Chandrayaan-2 Terrain Mapping Camera (TMC) imagery. It specifically matches Nadir (0°) and Aft (-25°) viewing angles by accounting for 3D lunar parallax.

## The Challenge
Matching orbital imagery of the Moon is difficult due to significant 3D parallax. Craters exhibit elevation relief, meaning a standard 2D Homography matrix fails to accurately warp the images. 

## Our Approach
1. **Feature Extraction:** Utilizes PyTorch-accelerated SuperPoint and LightGlue to extract and match thousands of deep-learned keypoints across image tiles.
2. **Epipolar Geometry:** Replaces standard flat-plane RANSAC with a 3D-aware Fundamental Matrix verifier to mathematically account for crater depth and camera tilt.
3. **Adaptive Tiling:** Processes the massive orbital strips in localized chunks to guarantee sub-pixel alignment.

## Final Hackathon Metrics (Phase 2 Verification)
* **Inlier Ratio:** 73.53% (Target: >65%)
* **Registration RMSE:** 0.6243 pixels (Target: <0.8 pixels)
* **Geometric Consensus:** Verified 3D stereoscopic lock using OpenCV `USAC_MAGSAC`.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the tile sweeper: `python src/matching/auto_find_overlap.py`
3. Verify metrics: `python tests/test_ransac.py`