import os
import glob
import cv2
import numpy as np
from src.core.preprocessing import load_chandrayaan_pds4

RAW_DIR = r"D:\LunaMatch\LunaMatch\data\raw\chandrayaan2_samples" 
PROCESSED_DIR = r"D:\LunaMatch\LunaMatch\data\processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def batch_process():
    # Let's process BOTH Nadir and Aft in one run so we have all tiles
    all_files = glob.glob(os.path.join(RAW_DIR, "*_ncn_*.xml")) + \
                glob.glob(os.path.join(RAW_DIR, "*_nca_*.xml"))
    
    if not all_files:
        print(f"No XML files found in {RAW_DIR}. Check your path!")
        return

    for xml_path in all_files:
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

if __name__ == "__main__":
    batch_process()