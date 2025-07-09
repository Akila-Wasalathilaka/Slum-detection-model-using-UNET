# File: preprocess_binary.py
# Binary preprocessing for slum detection - converts multi-class to binary
import os
import glob
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DIR = BASE_DIR / "data_preprocessed"

SETS = ["train", "val", "test"]

# Original expected class mapping (from documentation)
EXPECTED_CLASS_COLORS = [
    (0, 128, 0),      # 0: vegetation
    (128, 128, 128),  # 1: built-up
    (255, 0, 0),      # 2: informal settlements (SLUM - our target class)
    (0, 0, 128),      # 3: impervious surfaces
    (165, 42, 42),    # 4: barren
    (0, 255, 255),    # 5: water
    (128, 0, 128)     # 6: unlabelled
]

# Actual colors found in the dataset - needs to be mapped to classes
ACTUAL_COLORS_FOUND = [
    (0, 0, 0),        # Black
    (40, 120, 240),   # Blue-ish
    (80, 140, 50),    # Green-ish  
    (100, 100, 150),  # Purple-ish
    (200, 160, 40),   # Yellow-ish
    (200, 200, 200),  # Light gray
    (250, 235, 185)   # Beige-ish
]

# We need to map actual colors to class IDs
# This mapping should be based on visual inspection or dataset documentation
# For now, let's create a reasonable mapping based on color similarity
COLOR_TO_CLASS_MAPPING = {
    (0, 0, 0): 6,        # Black -> unlabelled
    (40, 120, 240): 2,   # Blue -> informal settlements (slums) - ASSUMPTION
    (80, 140, 50): 0,    # Green -> vegetation
    (100, 100, 150): 3,  # Purple -> impervious surfaces 
    (200, 160, 40): 4,   # Yellow -> barren
    (200, 200, 200): 1,  # Gray -> built-up
    (250, 235, 185): 5   # Beige -> water
}

SLUM_CLASS_ID = 2  # informal settlements

def discover_dataset_colors(data_dir, max_files_per_set=100):
    """Discover all unique colors present in the dataset"""
    print("Discovering actual colors in the dataset...")
    
    all_colors = set()
    color_counts = {}
    
    for dataset_split in SETS:
        mask_dir = data_dir / dataset_split / "masks"
        if not mask_dir.exists():
            print(f"Warning: {mask_dir} does not exist")
            continue
            
        mask_files = list(mask_dir.glob("*.png"))[:max_files_per_set]
        
        for mask_file in tqdm(mask_files, desc=f"Analyzing {dataset_split}"):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
            unique_colors = np.unique(mask_rgb.reshape(-1, mask_rgb.shape[-1]), axis=0)
            for color in unique_colors:
                color_tuple = tuple(color)
                all_colors.add(color_tuple)
                if color_tuple not in color_counts:
                    color_counts[color_tuple] = 0
                color_counts[color_tuple] += 1
    
    print(f"\nFound {len(all_colors)} unique colors:")
    for color in sorted(all_colors):
        print(f"  {color} (appears in {color_counts[color]} files)")
    
    return all_colors, color_counts

def convert_rgb_to_class_mask(mask_rgb, color_mapping=None):
    """Convert RGB mask to class indices using actual color mapping."""
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    if color_mapping is None:
        color_mapping = COLOR_TO_CLASS_MAPPING
    
    # Apply the color mapping
    for rgb_color, class_id in color_mapping.items():
        condition = np.all(mask_rgb == rgb_color, axis=-1)
        class_mask[condition] = class_id
        
        # Debug: show how many pixels were mapped
        pixel_count = np.sum(condition)
        if pixel_count > 0:
            class_names = ['vegetation', 'built-up', 'informal settlements', 
                          'impervious surfaces', 'barren', 'water', 'unlabelled']
            print(f"    {rgb_color} -> Class {class_id} ({class_names[class_id]}): {pixel_count} pixels")
    
    return class_mask

def convert_to_binary_mask(class_mask, slum_class_id):
    """Convert multi-class mask to binary slum vs non-slum."""
    binary_mask = (class_mask == slum_class_id).astype(np.uint8)
    return binary_mask

def analyze_class_distribution(mask_paths, slum_class_id):
    """Analyze the distribution of classes in the dataset."""
    print("Analyzing class distribution...")
    
    total_pixels = 0
    slum_pixels = 0
    class_counts = {i: 0 for i in range(7)}  # 7 classes
    
    for mask_path in tqdm(mask_paths, desc="Analyzing"):
        mask_rgb = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)
        class_mask = convert_rgb_to_class_mask(mask_rgb)
        
        # Count pixels for each class
        unique, counts = np.unique(class_mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < 7:
                class_counts[cls] += count
        
        total_pixels += class_mask.size
        slum_pixels += np.sum(class_mask == slum_class_id)
    
    print("\nClass Distribution:")
    print("-" * 50)
    class_names = ['vegetation', 'built-up', 'informal settlements', 
                   'impervious surfaces', 'barren', 'water', 'unlabelled']
    
    for i, (cls_name, count) in enumerate(zip(class_names, class_counts.values())):
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"{i}: {cls_name:20} - {count:10,} pixels ({percentage:5.2f}%)")
    
    print("-" * 50)
    print(f"Binary Distribution:")
    print(f"Slum pixels:     {slum_pixels:10,} ({(slum_pixels/total_pixels)*100:5.2f}%)")
    print(f"Non-slum pixels: {total_pixels-slum_pixels:10,} ({((total_pixels-slum_pixels)/total_pixels)*100:5.2f}%)")
    print(f"Total pixels:    {total_pixels:10,}")
    
    return class_counts, slum_pixels, total_pixels

def main():
    print("Starting binary slum detection preprocessing...")
    print(f"Target class: {SLUM_CLASS_ID} (informal settlements)")
    
    print("\nActual color to class mapping:")
    class_names = ['vegetation', 'built-up', 'informal settlements', 
                   'impervious surfaces', 'barren', 'water', 'unlabelled']
    for color, class_id in COLOR_TO_CLASS_MAPPING.items():
        print(f"  {color} -> Class {class_id} ({class_names[class_id]})")
    
    # Create output directories
    for s in SETS:
        output_img_dir = PREPROCESSED_DIR / s / "images"
        output_mask_dir = PREPROCESSED_DIR / s / "masks"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    overall_stats = {
        'total_images': 0,
        'total_pixels': 0,
        'slum_pixels': 0,
        'class_distribution': {i: 0 for i in range(7)}
    }
    
    for s in SETS:
        print(f"\n{'='*60}")
        print(f"Processing set: {s.upper()}")
        print(f"{'='*60}")
        
        img_dir = DATA_DIR / s / "images"
        mask_dir = DATA_DIR / s / "masks"
        
        if not img_dir.exists() or not mask_dir.exists():
            print(f"Warning: Directories not found for {s} set")
            continue
        
        output_img_dir = PREPROCESSED_DIR / s / "images"
        output_mask_dir = PREPROCESSED_DIR / s / "masks"
        
        # Get all mask files
        mask_paths = list(mask_dir.glob("*.png"))
        print(f"Found {len(mask_paths)} mask files")
        
        if len(mask_paths) == 0:
            print(f"No mask files found in {mask_dir}")
            continue
        
        # Analyze class distribution for this set
        class_counts, slum_pixels, total_pixels = analyze_class_distribution(
            mask_paths, SLUM_CLASS_ID
        )
        
        # Update overall stats
        overall_stats['total_images'] += len(mask_paths)
        overall_stats['total_pixels'] += total_pixels
        overall_stats['slum_pixels'] += slum_pixels
        for i, count in class_counts.items():
            overall_stats['class_distribution'][i] += count
        
        # Process masks
        successful_conversions = 0
        
        for mask_path in tqdm(mask_paths, desc=f"Converting {s} masks"):
            try:
                base_name_png = mask_path.name
                base_name_tif = base_name_png.replace(".png", ".tif")
                
                # Load and convert mask
                mask_rgb = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)
                class_mask = convert_rgb_to_class_mask(mask_rgb)
                
                # Save the class mask (for reference)
                class_mask_path = output_mask_dir / base_name_png
                cv2.imwrite(str(class_mask_path), class_mask)
                
                # Copy corresponding image
                img_path = img_dir / base_name_tif
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        output_img_path = output_img_dir / base_name_tif
                        cv2.imwrite(str(output_img_path), img)
                        successful_conversions += 1
                    else:
                        print(f"Warning: Could not load image {img_path}")
                else:
                    print(f"Warning: Image not found for mask {mask_path}")
                    
            except Exception as e:
                print(f"Error processing {mask_path}: {str(e)}")
                continue
        
        print(f"Successfully processed {successful_conversions}/{len(mask_paths)} image-mask pairs")
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print("OVERALL DATASET STATISTICS")
    print(f"{'='*60}")
    
    print(f"Total images processed: {overall_stats['total_images']}")
    print(f"Total pixels: {overall_stats['total_pixels']:,}")
    
    print(f"\nOverall Class Distribution:")
    print("-" * 50)
    class_names = ['vegetation', 'built-up', 'informal settlements', 
                   'impervious surfaces', 'barren', 'water', 'unlabelled']
    
    for i, (cls_name, count) in enumerate(zip(class_names, overall_stats['class_distribution'].values())):
        percentage = (count / overall_stats['total_pixels']) * 100 if overall_stats['total_pixels'] > 0 else 0
        print(f"{i}: {cls_name:20} - {count:12,} pixels ({percentage:5.2f}%)")
    
    print("-" * 50)
    print(f"Binary Classification Summary:")
    print(f"Slum pixels:     {overall_stats['slum_pixels']:12,} ({(overall_stats['slum_pixels']/overall_stats['total_pixels'])*100:5.2f}%)")
    print(f"Non-slum pixels: {overall_stats['total_pixels']-overall_stats['slum_pixels']:12,} ({((overall_stats['total_pixels']-overall_stats['slum_pixels'])/overall_stats['total_pixels'])*100:5.2f}%)")
    
    # Calculate class imbalance ratio
    slum_ratio = overall_stats['slum_pixels'] / overall_stats['total_pixels']
    imbalance_ratio = (1 - slum_ratio) / slum_ratio if slum_ratio > 0 else float('inf')
    print(f"Class imbalance ratio (non-slum:slum): {imbalance_ratio:.2f}:1")
    
    print(f"\n[OK] Pre-processing complete!")
    print(f"Processed dataset is now in '{PREPROCESSED_DIR}' folder.")
    print("\nNote: The binary_slum_detection.py script will automatically convert")
    print("the class masks to binary format during training.")

if __name__ == "__main__":
    main()
