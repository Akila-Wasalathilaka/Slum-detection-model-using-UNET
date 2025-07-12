import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def deep_mask_analysis():
    """Deep analysis to understand the actual mask content and variation"""
    
    print("=" * 60)
    print("DEEP MASK CONTENT ANALYSIS")
    print("=" * 60)
    
    # Let's check if masks actually have spatial variation
    # Maybe they're not uniform as initially thought
    
    def analyze_mask_file(mask_path, img_path=None):
        """Analyze individual mask file in detail"""
        mask = np.array(Image.open(mask_path))
        
        print(f"\nAnalyzing: {os.path.basename(mask_path)}")
        print(f"  Shape: {mask.shape}")
        print(f"  Dtype: {mask.dtype}")
        
        # Check each channel separately
        for ch in range(3):
            channel_data = mask[:, :, ch]
            unique_vals = np.unique(channel_data)
            print(f"  Channel {ch} ({'RGB'[ch]}): unique values = {unique_vals}")
            
            if len(unique_vals) > 1:
                print(f"    Value distribution:")
                for val in unique_vals:
                    count = np.sum(channel_data == val)
                    pct = (count / channel_data.size) * 100
                    print(f"      {val}: {count} pixels ({pct:.1f}%)")
            else:
                print(f"    Uniform channel with value {unique_vals[0]}")
        
        # Check if there's spatial variation by looking at different regions
        h, w = mask.shape[:2]
        regions = [
            ("Top-left", mask[:h//2, :w//2]),
            ("Top-right", mask[:h//2, w//2:]),
            ("Bottom-left", mask[h//2:, :w//2]),
            ("Bottom-right", mask[h//2:, w//2:])
        ]
        
        print(f"  Regional analysis:")
        has_variation = False
        for region_name, region in regions:
            region_unique = []
            for ch in range(3):
                ch_unique = np.unique(region[:, :, ch])
                region_unique.append(len(ch_unique))
            
            if any(x > 1 for x in region_unique):
                has_variation = True
                print(f"    {region_name}: Variation detected!")
            else:
                print(f"    {region_name}: Uniform")
        
        if not has_variation:
            print(f"  ⚠️  This mask appears to be completely uniform!")
        
        return mask, has_variation
    
    # Analyze multiple files from different sets
    sample_files = [
        ("data/train/masks/jp22_1.1_1.png", "data/train/images/jp22_1.1_1.tif"),
        ("data/train/masks/jp22_1.1_10.png", "data/train/images/jp22_1.1_10.tif"),
        ("data/train/masks/jp22_1.1_50.png", "data/train/images/jp22_1.1_50.tif"),
        ("data/val/masks/jp22_1.10_1.png", "data/val/images/jp22_1.10_1.tif"),
        ("data/test/masks/jp22_1.0_1.png", "data/test/images/jp22_1.0_1.tif"),
    ]
    
    analyzed_masks = []
    variations_found = []
    
    for mask_path, img_path in sample_files:
        if os.path.exists(mask_path):
            mask, has_var = analyze_mask_file(mask_path, img_path if os.path.exists(img_path) else None)
            analyzed_masks.append((mask_path, mask))
            variations_found.append(has_var)
    
    print(f"\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    
    if any(variations_found):
        print("✅ Spatial variation found in some masks!")
    else:
        print("❌ No spatial variation found - all masks appear uniform!")
        print("   This suggests either:")
        print("   1. The masks are placeholder/dummy files")
        print("   2. The entire tiles belong to a single class")
        print("   3. There's an issue with mask generation/export")
    
    return analyzed_masks

def check_different_mask_files():
    """Check if different mask files have different content"""
    
    print(f"\n" + "=" * 60)
    print("COMPARING DIFFERENT MASK FILES")
    print("=" * 60)
    
    # Get a sample of mask files
    train_mask_dir = "data/train/masks"
    mask_files = [f for f in os.listdir(train_mask_dir) if f.endswith('.png')]
    
    # Sample files from different parts of the dataset
    sample_indices = [0, 10, 50, 100, 500, 1000, 2000, 5000]
    sample_files = [mask_files[i] for i in sample_indices if i < len(mask_files)]
    
    print(f"Comparing {len(sample_files)} mask files...")
    
    reference_mask = None
    all_identical = True
    
    for i, mask_file in enumerate(sample_files):
        mask_path = os.path.join(train_mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        if reference_mask is None:
            reference_mask = mask
            print(f"Reference: {mask_file}")
        else:
            if np.array_equal(mask, reference_mask):
                print(f"✅ {mask_file}: IDENTICAL to reference")
            else:
                print(f"❌ {mask_file}: DIFFERENT from reference")
                all_identical = False
                
                # Show the differences
                diff = np.any(mask != reference_mask, axis=2)
                diff_pixels = np.sum(diff)
                total_pixels = diff.size
                print(f"   Different pixels: {diff_pixels}/{total_pixels} ({(diff_pixels/total_pixels)*100:.2f}%)")
    
    if all_identical:
        print(f"\n🚨 ALL MASKS ARE IDENTICAL!")
        print(f"   This is highly unusual for a segmentation dataset.")
        print(f"   Possible issues:")
        print(f"   1. Masks were not properly exported from annotation tool")
        print(f"   2. All tiles happen to belong to the same class")
        print(f"   3. Dataset preprocessing error")
    else:
        print(f"\n✅ Masks have variation - this is expected for segmentation!")

def visualize_problematic_masks():
    """Create visualization to understand the mask content better"""
    
    print(f"\n" + "=" * 60)
    print("CREATING DETAILED VISUALIZATIONS")
    print("=" * 60)
    
    # Load a few samples for detailed visualization
    sample_pairs = [
        ("data/train/images/jp22_1.1_1.tif", "data/train/masks/jp22_1.1_1.png"),
        ("data/train/images/jp22_1.1_10.tif", "data/train/masks/jp22_1.1_10.png"),
        ("data/val/images/jp22_1.10_1.tif", "data/val/masks/jp22_1.10_1.png"),
    ]
    
    fig, axes = plt.subplots(len(sample_pairs), 5, figsize=(25, 5*len(sample_pairs)))
    if len(sample_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img_path, mask_path) in enumerate(sample_pairs):
        if os.path.exists(img_path) and os.path.exists(mask_path):
            # Load image and mask
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            # Original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Image {i+1}\n{os.path.basename(img_path)}")
            axes[i, 0].axis('off')
            
            # RGB mask
            axes[i, 1].imshow(mask)
            axes[i, 1].set_title(f"RGB Mask {i+1}")
            axes[i, 1].axis('off')
            
            # Individual channels
            for ch in range(3):
                channel_data = mask[:, :, ch]
                im = axes[i, 2+ch].imshow(channel_data, cmap='gray', vmin=0, vmax=255)
                axes[i, 2+ch].set_title(f"{'RGB'[ch]} Channel\nRange: [{channel_data.min()}, {channel_data.max()}]")
                axes[i, 2+ch].axis('off')
                plt.colorbar(im, ax=axes[i, 2+ch], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('mask_problem_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Problem analysis visualization saved as 'mask_problem_analysis.png'")

if __name__ == "__main__":
    analyzed_masks = deep_mask_analysis()
    check_different_mask_files()
    visualize_problematic_masks()
