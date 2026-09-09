import cv2
import numpy as np
import tifffile as tiff  # Requires: pip install tifffile
import pds4_tools

class LunarPreprocessor:
    def __init__(self, clip_limit: float = 3.0, grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def load_and_crop_tiff(self, filepath: str, crop_size: int = 1024) -> list[np.ndarray]:
        """Reads heavy Chandrayaan-2 imagery and outputs manageable square tiles."""

        if filepath.endswith(('.tif', '.tiff')):
            img = tiff.imread(filepath)
        else:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


        return self.crop_into_tiles(img)

        h, w = img.shape[:2]
        return [img[y:y+crop_size, x:x+crop_size] 
                for y in range(0, h - crop_size, crop_size) 
                for x in range(0, w - crop_size, crop_size)]


    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Normalizes lunar contrast, bypassing if disabled in YAML."""
        clahe_config = self.profile.get('clahe', {})
        
        # Bypass enhancement entirely if disabled for this sensor
        if not clahe_config.get('enabled', True):
            return img
            
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        normalized = self.clahe.apply(img)
                
        # Conditionally apply Bilateral Filter
        bf_config = self.profile.get('bilateral_filter', {'enabled': False})
        if bf_config.get('enabled', False):
            return cv2.bilateralFilter(
                normalized, 
                d=bf_config.get('d', 3), 
                sigmaColor=bf_config.get('sigma_color', 15), 
                sigmaSpace=bf_config.get('sigma_space', 15)
            )
        return normalized
        
    def is_usable_terrain(self, tile: np.ndarray) -> bool:
        """Discards tiles that are entirely shadow or featureless flat dust."""
        sf_config = self.profile.get('shadow_filter', {})
        if not sf_config.get('enabled', False):
            return True
            
        # 1. Reject if the tile is predominantly pitch-black
        dark_ratio = np.sum(tile < sf_config['dark_threshold']) / tile.size
        if dark_ratio > sf_config['max_dark_ratio']:
            return False
            
        # 2. Reject if the tile lacks geometric texture (flat noise)
        if np.std(tile) < sf_config['min_std_dev']:
            return False
            
        return True

def load_chandrayaan_pds4(xml_filepath: str, profile: dict) -> np.ndarray:
    """
    Reads a Chandrayaan-2 PDS4 image and converts it to a standard OpenCV 8-bit array
    using dynamic percentile clipping from the YAML profile.
    """
    structures = pds4_tools.read(xml_filepath, quiet=True)
    image_data = structures[0].data
    
    # Extract clipping percentiles from the YAML profile
    percentiles = tuple(profile['clip_percentiles'])
    p_low, p_high = np.percentile(image_data, percentiles)
    
    clipped_data = np.clip(image_data, p_low, p_high)
    normalized = cv2.normalize(clipped_data, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized.astype(np.uint8)




