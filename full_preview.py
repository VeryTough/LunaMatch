import os
import cv2
import numpy as np
import pds4_tools

# --- CONFIGURATION ---
# Replace with the path to your raw .xml file
XML_FILEPATH = r"D:\LunaMatch\LunaMatch-main\data\raw\ch2_tmc_ncn_20260813T0627378557_d_img_d18.xml"

# Replace with where you want the full PNG saved
OUTPUT_PNG_PATH = r"D:\LunaMatch\LunaMatch\data\processed2\full\ch2_tmc_ncn_20260813T0627378557.png"

# Orbital strips are often 40,000+ pixels tall. 
# Set this below 1.0 (e.g., 0.25) if your standard image viewer crashes trying to open it.
SCALE_FACTOR = 0.25 

def export_full_strip():
    if not os.path.exists(XML_FILEPATH):
        print(f"Error: Could not find XML file at {XML_FILEPATH}")
        return

    print(f"Loading full PDS4 dataset from: {os.path.basename(XML_FILEPATH)}")
    
    # 1. Read the PDS4 data silently
    structures = pds4_tools.read(XML_FILEPATH, quiet=True)
    image_data = structures[0].data
    
    original_h, original_w = image_data.shape
    print(f"Original array size: {original_w}x{original_h} pixels")
    
    # 2. Apply Percentile Clipping (Removes cosmic rays and dead pixels)
    print("Applying percentile clipping and 8-bit normalization...")
    p_low, p_high = np.percentile(image_data, (1, 99))
    clipped_data = np.clip(image_data, p_low, p_high)
    
    # 3. Normalize to standard 0-255 image format
    normalized = cv2.normalize(clipped_data, None, 0, 255, cv2.NORM_MINMAX)
    image_8bit = normalized.astype(np.uint8)
    
    # 4. Optional Rescaling for easier viewing
    if SCALE_FACTOR != 1.0:
        print(f"Downscaling image by {SCALE_FACTOR * 100}% for safer viewing...")
        new_w = int(original_w * SCALE_FACTOR)
        new_h = int(original_h * SCALE_FACTOR)
        image_8bit = cv2.resize(image_8bit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 5. Export to PNG
    print(f"Saving PNG to: {OUTPUT_PNG_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PNG_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PNG_PATH, image_8bit)
    
    print("Export complete.")

if __name__ == "__main__":
    export_full_strip()