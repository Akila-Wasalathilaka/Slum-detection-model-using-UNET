# File: preprocess_data.py
import os
import glob
from tqdm import tqdm
import cv2
import numpy as np

# --- Configuration ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data_preprocessed")

SETS = ["train", "val", "test"]

CLASS_COLORS = [
    (0, 128, 0), (128, 128, 128), (255, 0, 0), (0, 0, 128), 
    (165, 42, 42), (0, 255, 255), (128, 0, 128)
]

def convert_rgb_to_class_mask(mask_rgb, color_map):
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for i, color in enumerate(color_map):
        condition = np.all(mask_rgb == color, axis=-1)
        class_mask[condition] = i
    return class_mask

def main():
    print("Starting one-time dataset pre-processing...")
    for s in SETS:
        print(f"\nProcessing set: {s}")
        img_dir = os.path.join(DATA_DIR, s, "images")
        mask_dir = os.path.join(DATA_DIR, s, "masks")
        
        output_img_dir = os.path.join(PREPROCESSED_DIR, s, "images")
        output_mask_dir = os.path.join(PREPROCESSED_DIR, s, "masks")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        
        mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
        
        for mask_path in tqdm(mask_paths, desc=f"Converting {s} masks"):
            base_name_png = os.path.basename(mask_path)
            base_name_tif = base_name_png.replace(".png", ".tif")
            
            mask_rgb = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            class_mask = convert_rgb_to_class_mask(mask_rgb, CLASS_COLORS)
            cv2.imwrite(os.path.join(output_mask_dir, base_name_png), class_mask)
            
            img_path = os.path.join(img_dir, base_name_tif)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                cv2.imwrite(os.path.join(output_img_dir, base_name_tif), img)

    print("\nPre-processing complete!")
    print(f"Optimized dataset is now in '{PREPROCESSED_DIR}' folder.")

if __name__ == "__main__":
    main()