import os
import glob
import cv2
import numpy as np
from src.core.preprocessing import load_chandrayaan_pds4, LunarPreprocessor, PreprocessRouter
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def batch_process():
    # 1. Initialize the YAML config router instead of the preprocessor
    router = PreprocessRouter(os.path.join(ROOT_DIR, "config.yaml"))
    
    # Process BOTH Nadir and Aft in one run
    all_files = glob.glob(os.path.join(RAW_DIR, "*_ncn_*.xml")) + \
                glob.glob(os.path.join(RAW_DIR, "*_nca_*.xml"))
    
    if not all_files:
        print(f"No XML files found in {RAW_DIR}. Check your path!")
        return

    for xml_path in all_files:
        base_name = os.path.basename(xml_path).replace('.xml', '')
        print(f"Processing: {base_name}")
        
        # 2. Get the specific profile for this file (TMC-2)
        try:
            profile = router.get_profile(base_name)
        except ValueError as e:
            print(f"Skipping {base_name}: {e}")
            continue
            
        # 3. NOW instantiate the preprocessor with the correct profile
        preprocessor = LunarPreprocessor(profile)
    
    # Process BOTH Nadir and Aft in one run
    all_files = glob.glob(os.path.join(RAW_DIR, "*_ncn_*.xml")) + \
                glob.glob(os.path.join(RAW_DIR, "*_nca_*.xml"))
    
    if not all_files:
        print(f"No XML files found in {RAW_DIR}. Check your path!")
        return

    for xml_path in all_files:
        base_name = os.path.basename(xml_path).replace('.xml', '')
        print(f"Processing: {base_name}")
        
        # Create a dedicated output folder for this specific XML file
        xml_output_dir = os.path.join(PROCESSED_DIR, base_name)
        os.makedirs(xml_output_dir, exist_ok=True)
        
        # 1. Load PDS4 data into an 8-bit numpy array
        img_array = load_chandrayaan_pds4(xml_path, profile)
        
        # 2. Crop logic
        crop_size = 1024
        h, w = img_array.shape[:2]
        
        # Generates exact 1024x1024 squares across both X and Y axes
        tiles = [img_array[y:y+crop_size, x:x+crop_size] 
                 for y in range(0, h - crop_size, crop_size) 
                 for x in range(0, w - crop_size, crop_size)]
        
        # 3. Enhance and save each tile
        for tile_id, tile in enumerate(tiles):
            # Apply CLAHE and bilateral filtering
            enhanced_tile = preprocessor.apply_clahe(tile)
            
            # Save into the newly created subfolder
            npy_path = os.path.join(xml_output_dir, f"{base_name}_tile_{tile_id}.npy")
            png_path = os.path.join(xml_output_dir, f"{base_name}_tile_{tile_id}.png")
            
            # Save the enhanced output
            np.save(npy_path, enhanced_tile)
            cv2.imwrite(png_path, enhanced_tile)
            
        print(f"  -> Generated and enhanced {len(tiles)} tiles in {xml_output_dir}")

if __name__ == "__main__":
    batch_process()
