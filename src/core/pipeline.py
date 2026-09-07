import os
import glob
import numpy as np
from src.core.matcher import LunarMatcher

class MatchingPipeline:
    def __init__(self, database_dir):
        """
        Initializes the AI once so the backend doesn't have to reload 
        the neural networks on every single web request.
        """
        self.matcher = LunarMatcher(max_keypoints=1024)
        self.database_dir = database_dir

    def process_upload(self, query_img_array: np.ndarray) -> dict:
        """
        The backend team will call this function and pass the uploaded image as a NumPy array.
        """
        # Find all candidate Aft tiles in the database
        candidate_files = glob.glob(os.path.join(self.database_dir, "*_nca_*.npy"))
        
        best_count = 0
        best_file = None
        best_coords = None
        
        # Sweep the database
        for aft_path in candidate_files:
            aft_tile = np.load(aft_path)
            results = self.matcher.match_image_pair(query_img_array, aft_tile)
            match_count = len(results["pts_src"])
            
            if match_count > best_count:
                best_count = match_count
                best_file = aft_path
                best_coords = results

        # Return a clean dictionary to the backend
        if best_count > 20:
            return {
                "success": True,
                "matched_file": os.path.basename(best_file),
                "match_count": best_count,
                "pts_src": best_coords["pts_src"],
                "pts_ref": best_coords["pts_ref"]
            }
        else:
            return {
                "success": False,
                "error": "No overlapping terrain found in database."
            }