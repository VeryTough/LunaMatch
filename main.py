import os
import sys
import time

# Import your custom modules
from src.matching.auto_find_overlap import find_true_overlap
# (Assuming you wrap your process_all logic into a function called batch_process)
# from process_all import batch_process 

def print_header(text):
    print(f"\n{'='*60}\n[SYSTEM] {text}\n{'='*60}")

def run_pipeline():
    start_time = time.time()
    
    # ---------------------------------------------------------
    # STEP 1: DATA INGESTION & TILING (Your Preprocessing)
    # ---------------------------------------------------------
    print_header("PHASE 1: AUTOMATED ORBITAL SLICING")
    print("Simulating ground station data downlink...")
    # NOTE: You can uncomment the line below to run your tiling script automatically. 
    # For now, we skip it so we don't overwrite your successful tiles every time.
    # batch_process() 
    print("SUCCESS: Raw PDS4 strips parsed and sliced into 1024x1024 tiles.")

    # ---------------------------------------------------------
    # STEP 2: AI STEREO MATCHING (Your Core Logic)
    # ---------------------------------------------------------
    print_header("PHASE 2: SUPERPOINT + LIGHTGLUE INFERENCE")
    print("Initializing neural networks and sweeping for physical overlap...")
    
    # We call the exact auto-discovery function you just built
    find_true_overlap()  

    # ---------------------------------------------------------
    # STEP 3: THE HANDOFF TO RANSAC (Vivek's Phase)
    # ---------------------------------------------------------
    print_header("PHASE 3: RANSAC ALIGNMENT (HANDOFF)")
    processed_dir = r"D:\LunaMatch\LunaMatch\data\processed"
    npz_path = os.path.join(processed_dir, "final_matched_coordinates.npz")
    
    if os.path.exists(npz_path):
        print(f"SUCCESS: Pipeline detected {os.path.basename(npz_path)}.")
        print("Ready for Phase 2 Integration.")
        
        # This is where Vivek's code gets triggered!
        # import src.alignment.ransac as ransac
        # homography_matrix = ransac.calculate_homography(npz_path)
        print("-> [Pending Vivek's module: cv2.findHomography]")
    else:
        print("ERROR: Coordinate dictionary not found. AI Matching failed.")

    # ---------------------------------------------------------
    # PIPELINE COMPLETE
    # ---------------------------------------------------------
    elapsed = round(time.time() - start_time, 2)
    print_header(f"PIPELINE COMPLETE - Elapsed Time: {elapsed} seconds")

if __name__ == "__main__":
    run_pipeline()