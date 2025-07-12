import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import torch

def analyze_dataset():
    """Analyze the dataset to understand image/mask characteristics"""
    
    # Paths
    data_root = "data"
    splits = ["train", "val", "test"]
    
    print("=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    
    for split in splits:
        print(f"\n--- {split.upper()} SET ---")
        
        img_dir = os.path.join(data_root, split, "images")
        mask_dir = os.path.join(data_root, split, "masks")
        
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        print(f"Images: {len(img_files)}")
        print(f"Masks: {len(mask_files)}")
        
        # Sample a few images to check dimensions and content
        sample_files = img_files[:3]
        
        for i, img_file in enumerate(sample_files):
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file.replace('.tif', '.png'))
            
            try:
                # Load image
                img = Image.open(img_path)
                mask = Image.open(mask_path)
                
                print(f"\nSample {i+1}: {img_file}")
                print(f"  Image size: {img.size}, mode: {img.mode}")
                print(f"  Mask size: {mask.size}, mode: {mask.mode}")
                
                # Convert to numpy for analysis
                img_np = np.array(img)
                mask_np = np.array(mask)
                
                print(f"  Image shape: {img_np.shape}, dtype: {img_np.dtype}")
                print(f"  Image range: [{img_np.min()}, {img_np.max()}]")
                print(f"  Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
                
                # Analyze mask values
                unique_values = np.unique(mask_np)
                print(f"  Unique mask values: {unique_values}")
                
                # Count pixels per class
                value_counts = Counter(mask_np.flatten())
                print(f"  Mask value distribution: {dict(value_counts)}")
                
                if i == 0:  # Save first sample for visualization
                    sample_img = img_np
                    sample_mask = mask_np
                    
            except Exception as e:
                print(f"  Error loading {img_file}: {e}")
    
    # Create a visualization of the first sample
    try:
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(sample_img)
        plt.title("Original RGB Image")
        plt.axis('off')
        
        # Original mask
        plt.subplot(1, 3, 2)
        plt.imshow(sample_mask, cmap='viridis')
        plt.title("Original Multi-class Mask")
        plt.colorbar()
        plt.axis('off')
        
        # Binary mask (assuming class 2 is slum)
        binary_mask = (sample_mask == 2).astype(np.uint8)
        plt.subplot(1, 3, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Binary Slum Mask (Class 2)")
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nSample visualization saved as 'dataset_sample.png'")
        
        # Calculate class statistics for the entire training set
        print("\n" + "=" * 30)
        print("TRAINING SET CLASS ANALYSIS")
        print("=" * 30)
        
        train_mask_dir = os.path.join(data_root, "train", "masks")
        all_class_counts = Counter()
        
        for mask_file in os.listdir(train_mask_dir)[:100]:  # Sample first 100 for speed
            if mask_file.endswith('.png'):
                mask_path = os.path.join(train_mask_dir, mask_file)
                mask = np.array(Image.open(mask_path))
                file_counts = Counter(mask.flatten())
                all_class_counts.update(file_counts)
        
        total_pixels = sum(all_class_counts.values())
        print(f"Analyzed {len([f for f in os.listdir(train_mask_dir) if f.endswith('.png')][:100])} training masks")
        print(f"Total pixels analyzed: {total_pixels:,}")
        
        for class_id in sorted(all_class_counts.keys()):
            count = all_class_counts[class_id]
            percentage = (count / total_pixels) * 100
            print(f"Class {class_id}: {count:,} pixels ({percentage:.2f}%)")
            
    except Exception as e:
        print(f"Error in visualization: {e}")

if __name__ == "__main__":
    analyze_dataset()
