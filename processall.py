import os
import glob
import cv2
import numpy as np
<<<<<<< HEAD
from src.core.preprocessing import load_chandrayaan_pds4, LunarPreprocessor
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed2")
=======
from src.core.preprocessing import load_chandrayaan_pds4

RAW_DIR = r"D:\LunaMatch\LunaMatch\data\raw\chandrayaan2_samples" 
PROCESSED_DIR = r"D:\LunaMatch\LunaMatch\data\processed"
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4

os.makedirs(PROCESSED_DIR, exist_ok=True)

def batch_process():
<<<<<<< HEAD
    # Initialize the preprocessor to access CLAHE
    preprocessor = LunarPreprocessor()
    
    # Process BOTH Nadir and Aft in one run
=======
    # Let's process BOTH Nadir and Aft in one run so we have all tiles
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4
    all_files = glob.glob(os.path.join(RAW_DIR, "*_ncn_*.xml")) + \
                glob.glob(os.path.join(RAW_DIR, "*_nca_*.xml"))
    
    if not all_files:
        print(f"No XML files found in {RAW_DIR}. Check your path!")
        return

    for xml_path in all_files:
<<<<<<< HEAD
        base_name = os.path.basename(xml_path).replace('.xml', '')
        print(f"Processing: {base_name}")
        
        # Create a dedicated output folder for this specific XML file
        xml_output_dir = os.path.join(PROCESSED_DIR, base_name)
        os.makedirs(xml_output_dir, exist_ok=True)
        
        # 1. Load PDS4 data into an 8-bit numpy array
        img_array = load_chandrayaan_pds4(xml_path)
        
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
=======
        print(f"Processing: {os.path.basename(xml_path)}")
        img_array = load_chandrayaan_pds4(xml_path)
        
        h, w = img_array.shape
        tile_height = 1024
        
        # FIX: Slice through the ENTIRE height of the image, stepping every 4000 pixels
        tile_id = 0
        for start_y in range(0, h - tile_height, 4000):
            end_y = start_y + tile_height
            
            tile = img_array[start_y:end_y, 0:w]
            base_name = os.path.basename(xml_path).replace('.xml', '')
            
            npy_path = os.path.join(PROCESSED_DIR, f"{base_name}_tile_{tile_id}.npy")
            png_path = os.path.join(PROCESSED_DIR, f"{base_name}_tile_{tile_id}.png")
            
            np.save(npy_path, tile)
            cv2.imwrite(png_path, tile)
            
            tile_id += 1
            
        print(f"  -> Generated {tile_id} tiles across the entire strip.")
>>>>>>> df0b1380b057c4ffcef330e219f6a26c64d5d6c4

if __name__ == "__main__":
    batch_process()