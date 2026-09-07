<<<<<<< HEAD
import os
import cv2
import yaml
import numpy as np
import tifffile as tiff
import pds4_tools

class PreprocessRouter:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)['sensors']

    def get_profile(self, filename: str) -> dict:
        """Returns the correct preprocessing parameters based on the filename."""
        if any(tag in filename for tag in self.config['tmc2']['identifiers']):
            return self.config['tmc2']
        
        if filename.startswith(tuple(self.config['lroc_nac']['identifiers'])):
            return self.config['lroc_nac']
            
        raise ValueError(f"No sensor profile matches filename: {filename}")

class LunarPreprocessor:
    def __init__(self, profile: dict):
        self.profile = profile
        
        # Dynamically configure CLAHE from the YAML profile
        limit = self.profile['clahe']['clip_limit']
        grid = tuple(self.profile['clahe']['grid_size'])
        self.clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)

    def crop_into_tiles(self, img: np.ndarray) -> list[np.ndarray]:
        """Slices any 2D image array into a grid of square tiles."""
        crop_size = self.profile['crop_size']
        h, w = img.shape[:2]
        return [img[y:y+crop_size, x:x+crop_size] 
                for y in range(0, h - crop_size, crop_size) 
                for x in range(0, w - crop_size, crop_size)]

    def load_and_crop_tiff(self, filepath: str) -> list[np.ndarray]:
        """Reads standard TIFF/PNG imagery and outputs manageable square tiles."""
=======
import cv2
import numpy as np
import tifffile as tiff  # Requires: pip install tifffile
import pds4_tools

class LunarPreprocessor:
    def __init__(self, clip_limit: float = 3.0, grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def load_and_crop_tiff(self, filepath: str, crop_size: int = 1024) -> list[np.ndarray]:
        """Reads heavy Chandrayaan-2 imagery and outputs manageable square tiles."""
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4
        if filepath.endswith(('.tif', '.tiff')):
            img = tiff.imread(filepath)
        else:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

<<<<<<< HEAD
        return self.crop_into_tiles(img)
=======
        h, w = img.shape[:2]
        return [img[y:y+crop_size, x:x+crop_size] 
                for y in range(0, h - crop_size, crop_size) 
                for x in range(0, w - crop_size, crop_size)]
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Normalizes lunar contrast without destroying shadow geometry."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        normalized = self.clahe.apply(img)
<<<<<<< HEAD
        
        # Conditionally apply Bilateral Filter based on YAML profile
        bf_config = self.profile['bilateral_filter']
        if bf_config['enabled']:
            return cv2.bilateralFilter(
                normalized, 
                d=bf_config['d'], 
                sigmaColor=bf_config['sigma_color'], 
                sigmaSpace=bf_config['sigma_space']
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
=======
        # Gentle bilateral filter for sensor noise
        return cv2.bilateralFilter(normalized, d=5, sigmaColor=50, sigmaSpace=50)



def load_chandrayaan_pds4(xml_filepath: str) -> np.ndarray:
    """
    Reads a Chandrayaan-2 PDS4 image and converts it to a standard OpenCV 8-bit array.
    """
    # 1. Read the PDS4 data using the XML label
    structures = pds4_tools.read(xml_filepath)
    
    # 2. Extract the actual image array (usually the first array in the structure)
    # The structure will automatically find and read the paired .img file
    image_data = structures[0].data
    
    # 3. TMC-2 data is often 16-bit or 32-bit float. We must normalize it to 8-bit for OpenCV.
    # cv2.normalize scales the darkest pixels to 0 and the brightest to 255.
    normalized = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
    image_8bit = normalized.astype(np.uint8)
    
    return image_8bit
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4
