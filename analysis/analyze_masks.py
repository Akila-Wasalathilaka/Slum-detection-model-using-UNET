import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def decode_mask_values():
    """Analyze mask encoding to understand the class mapping"""
    
    # Sample a few masks to understand the encoding
    mask_path = "data/train/masks/jp22_1.1_1.png"
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    
    print("Mask Analysis:")
    print(f"Shape: {mask_np.shape}")
    print(f"Dtype: {mask_np.dtype}")
    print(f"Unique values per channel:")
    
    for i, channel in enumerate(['R', 'G', 'B']):
        unique_vals = np.unique(mask_np[:, :, i])
        print(f"  {channel}: {unique_vals}")
    
    # Check if all channels are identical
    r_channel = mask_np[:, :, 0]
    g_channel = mask_np[:, :, 1] 
    b_channel = mask_np[:, :, 2]
    
    print(f"\nChannel comparison:")
    print(f"R==G: {np.array_equal(r_channel, g_channel)}")
    print(f"G==B: {np.array_equal(g_channel, b_channel)}")
    print(f"R==B: {np.array_equal(r_channel, b_channel)}")
    
    # Take just one channel (since they seem identical)
    mask_single = r_channel
    unique_classes = np.unique(mask_single)
    print(f"\nUnique class values: {unique_classes}")
    
    # Map to class labels based on typical CVAT export
    # Common mapping: 
    # 40 (dark) -> Background/Other
    # 120 (medium) -> Built-up/Vegetation  
    # 240 (bright) -> Target class (Slums)
    
    class_mapping = {
        40: 0,   # Background/Other classes
        120: 0,  # Non-slum classes  
        240: 1   # Slum class (target)
    }
    
    print(f"\nProposed class mapping for binary segmentation:")
    for original, binary in class_mapping.items():
        pixel_count = np.sum(mask_single == original)
        percentage = (pixel_count / mask_single.size) * 100
        class_name = "SLUM" if binary == 1 else "NON-SLUM"
        print(f"  {original} -> {binary} ({class_name}): {pixel_count} pixels ({percentage:.1f}%)")
    
    # Create binary mask
    binary_mask = np.zeros_like(mask_single)
    for original, binary in class_mapping.items():
        binary_mask[mask_single == original] = binary
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img_path = "data/train/images/jp22_1.1_1.tif"
    img = Image.open(img_path)
    axes[0].imshow(img)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')
    
    # Original mask (single channel)
    im1 = axes[1].imshow(mask_single, cmap='viridis')
    axes[1].set_title("Original Mask Values")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Binary mask
    im2 = axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title("Binary Slum Mask")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('mask_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return class_mapping

def analyze_class_distribution(class_mapping, num_samples=500):
    """Analyze class distribution across the dataset"""
    
    print(f"\n{'='*50}")
    print("CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*50}")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n--- {split.upper()} SET ---")
        mask_dir = f"data/{split}/masks"
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        # Sample masks for analysis
        sample_size = min(num_samples, len(mask_files))
        sample_files = mask_files[:sample_size]
        
        total_pixels = 0
        slum_pixels = 0
        
        for mask_file in sample_files:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = np.array(Image.open(mask_path))
            
            # Take first channel only
            mask_single = mask[:, :, 0]
            
            # Convert to binary
            binary_mask = np.zeros_like(mask_single)
            for original, binary in class_mapping.items():
                binary_mask[mask_single == original] = binary
            
            total_pixels += binary_mask.size
            slum_pixels += np.sum(binary_mask == 1)
        
        slum_percentage = (slum_pixels / total_pixels) * 100
        non_slum_percentage = 100 - slum_percentage
        
        print(f"Analyzed {sample_size} masks:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Slum pixels: {slum_pixels:,} ({slum_percentage:.2f}%)")
        print(f"  Non-slum pixels: {total_pixels - slum_pixels:,} ({non_slum_percentage:.2f}%)")
        print(f"  Class imbalance ratio: 1:{(total_pixels - slum_pixels)/slum_pixels:.1f}")

if __name__ == "__main__":
    class_mapping = decode_mask_values()
    analyze_class_distribution(class_mapping)
