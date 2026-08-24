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

        h, w = img.shape[:2]
        return [img[y:y+crop_size, x:x+crop_size] 
                for y in range(0, h - crop_size, crop_size) 
                for x in range(0, w - crop_size, crop_size)]

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Normalizes lunar contrast without destroying shadow geometry."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        normalized = self.clahe.apply(img)
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