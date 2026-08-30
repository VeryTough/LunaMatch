import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import cv2

class LunarMatcher:
    def __init__(self, max_keypoints=4096):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Dropping thresholds to practically zero to flood the pipeline with raw matches
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints, detection_threshold=0.0001).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint', filter_threshold=0.001).eval().to(self.device)

    def _apply_clahe(self, img_np: np.ndarray) -> np.ndarray:
        if len(img_np.shape) == 3: img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        if img_np.dtype != np.uint8: img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_np)

    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(self._apply_clahe(img_np)).float() / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

    def match_image_pair(self, img1: np.ndarray, img2: np.ndarray) -> dict:
        t1, t2 = self._to_tensor(img1), self._to_tensor(img2)
        with torch.no_grad():
            feats1, feats2 = self.extractor.extract(t1), self.extractor.extract(t2)
            matches01 = self.matcher({'image0': feats1, 'image1': feats2})
        feats1, feats2, matches01 = [rbd(x) for x in [feats1, feats2, matches01]]
        matches = matches01['matches']
        return {
            "pts_src": feats1['keypoints'][matches[..., 0]].cpu().numpy(),
            "pts_ref": feats2['keypoints'][matches[..., 1]].cpu().numpy(),
            "confidences": matches01.get('scores', torch.ones(len(matches))).cpu().numpy()
        }