import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.geometric_verification import GeometricVerifier

NPZ_PATH = r"data\processed\final_matched_coordinates.npz"

def test_pipeline():
    data = np.load(NPZ_PATH)
    pts_src = data["pts_src"]
    pts_ref = data["pts_ref"]
    
    verifier = GeometricVerifier()
    result = verifier.verify_and_align(pts_src, pts_ref)

    if result["success"]:
        # Sync the terminal printout with the optimized point pool
        total_eval = result.get("total_evaluated", len(pts_src))
        
        print(f"Loading Phase 1 coordinates from: {NPZ_PATH}")
        print(f"Total AI matches evaluated: {total_eval}")
        print("Executing RANSAC Geometric Verification...\n")
        print("====== PHASE 2: SUCCESS ======")
        print(f"Inlier Count      : {result['inlier_count']} / {total_eval}")
        print(f"Inlier Ratio      : {result['inlier_ratio'] * 100:.2f}%")
        print(f"Registration RMSE : {result['rmse']:.4f} pixels")
        print("==============================")
    else:
        print(f"RANSAC Failed: {result['error']}")

if __name__ == "__main__":
    test_pipeline()