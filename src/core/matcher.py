import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

class LunarMatcher:
    def __init__(self, max_keypoints=1024):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the models once when the class is initialized
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """Converts an 8-bit OpenCV image to the tensor LightGlue expects."""
        tensor = torch.from_numpy(img_np).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def match_image_pair(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        """
        Takes two real 8-bit grayscale images (e.g., Nadir and Aft views).
        Returns the exact dictionary interface required for P2 (RANSAC).
        """
        t1 = self._to_tensor(img1)
        t2 = self._to_tensor(img2)

        with torch.no_grad():
            feats1 = self.extractor.extract(t1)
            feats2 = self.extractor.extract(t2)
            matches01 = self.matcher({'image0': feats1, 'image1': feats2})

        # Remove batch dimensions and move to CPU
        feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]
        matches = matches01['matches']  
        
        pts1 = feats1['keypoints'][matches[..., 0]].cpu().numpy()
        pts2 = feats2['keypoints'][matches[..., 1]].cpu().numpy()
        scores = matches01['scores'].cpu().numpy() if 'scores' in matches01 else np.ones(len(pts1))

        return {
            "pts_src": pts1,
            "pts_ref": pts2,
            "confidences": scores
        }