import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def decode_rgb_mask():
    """Analyze RGB mask encoding to understand the class mapping"""
    
    # Sample a few masks to understand the encoding
    mask_path = "data/train/masks/jp22_1.1_1.png"
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    
    print("RGB Mask Analysis:")
    print(f"Shape: {mask_np.shape}")
    print(f"Dtype: {mask_np.dtype}")
    
    # Analyze RGB combinations
    print(f"\nRGB value analysis:")
    h, w, c = mask_np.shape
    
    # Get unique RGB combinations
    rgb_values = mask_np.reshape(-1, 3)
    unique_rgb = np.unique(rgb_values, axis=0)
    
    print(f"Unique RGB combinations:")
    for i, rgb in enumerate(unique_rgb):
        count = np.sum(np.all(rgb_values == rgb, axis=1))
        percentage = (count / len(rgb_values)) * 100
        print(f"  RGB({rgb[0]}, {rgb[1]}, {rgb[2]}): {count} pixels ({percentage:.1f}%)")
    
    # Analyze multiple samples to understand the pattern
    print(f"\nAnalyzing multiple samples...")
    
    sample_files = ['jp22_1.1_1.png', 'jp22_1.1_2.png', 'jp22_1.1_3.png', 'jp22_1.1_4.png', 'jp22_1.1_5.png']
    all_rgb_combinations = set()
    
    for sample_file in sample_files:
        try:
            sample_path = f"data/train/masks/{sample_file}"
            if os.path.exists(sample_path):
                sample_mask = np.array(Image.open(sample_path))
                sample_rgb = sample_mask.reshape(-1, 3)
                sample_unique = np.unique(sample_rgb, axis=0)
                
                print(f"\n{sample_file}:")
                for rgb in sample_unique:
                    count = np.sum(np.all(sample_rgb == rgb, axis=1))
                    percentage = (count / len(sample_rgb)) * 100
                    print(f"  RGB({rgb[0]}, {rgb[1]}, {rgb[2]}): {count} pixels ({percentage:.1f}%)")
                    all_rgb_combinations.add(tuple(rgb))
        except Exception as e:
            print(f"Error with {sample_file}: {e}")
    
    print(f"\nAll unique RGB combinations found across samples:")
    for rgb in sorted(all_rgb_combinations):
        print(f"  RGB{rgb}")
    
    # Based on typical CVAT annotation patterns, let's try to map these
    # Usually masks have values like:
    # (0,0,0) - background/unlabelled
    # Different colors for different classes
    
    # Let's assume the pattern and create a mapping
    # We need to identify which RGB combination represents slums
    
    # For now, let's create a visualization to understand better
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Load and visualize first few samples
    for i, sample_file in enumerate(sample_files[:6]):
        try:
            row = i // 3
            col = i % 3
            
            # Load image and mask
            img_path = f"data/train/images/{sample_file.replace('.png', '.tif')}"
            mask_path = f"data/train/masks/{sample_file}"
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = Image.open(img_path)
                mask = np.array(Image.open(mask_path))
                
                # Show image
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Sample {i+1}: {sample_file}")
                axes[row, col].axis('off')
                
                # Add text showing RGB values
                unique_rgb = np.unique(mask.reshape(-1, 3), axis=0)
                rgb_text = "\n".join([f"RGB{tuple(rgb)}" for rgb in unique_rgb])
                axes[row, col].text(0.02, 0.98, rgb_text, transform=axes[row, col].transAxes, 
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Error visualizing {sample_file}: {e}")
    
    plt.tight_layout()
    plt.savefig('rgb_mask_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return list(all_rgb_combinations)

def create_class_mapping(rgb_combinations):
    """Create a mapping from RGB values to binary classes"""
    
    print(f"\n{'='*50}")
    print("CREATING CLASS MAPPING")
    print(f"{'='*50}")
    
    # Based on the analysis, let's create a mapping
    # We need to identify which RGB represents informal settlements (slums)
    
    # Common patterns in CVAT exports:
    # - Often one channel per class
    # - Or specific RGB combinations for each class
    
    # Let's assume that one of the combinations represents slums
    # We'll need to examine the data more carefully or ask the user
    
    print("Found RGB combinations:")
    for i, rgb in enumerate(sorted(rgb_combinations)):
        print(f"  {i}: RGB{rgb}")
    
    # For now, let's create a provisional mapping
    # We'll assume the middle value or a specific pattern represents slums
    
    # Provisional mapping (this might need adjustment based on your specific dataset)
    class_mapping = {}
    
    if len(rgb_combinations) >= 3:
        # If we have multiple combinations, we need to identify the slum class
        # This requires domain knowledge or manual inspection
        sorted_combos = sorted(rgb_combinations)
        
        # Let's create a mapping where we assume each combination is a different class
        # and arbitrarily assign one as the slum class (you may need to adjust this)
        for i, rgb in enumerate(sorted_combos):
            # Assume the combination with highest total value might be slums
            # Or the one with specific pattern - this is a guess
            if i == len(sorted_combos) - 1:  # Last one as slum class
                class_mapping[rgb] = 1  # Slum
            else:
                class_mapping[rgb] = 0  # Non-slum
    
    print(f"\nProvisional class mapping:")
    for rgb, class_id in class_mapping.items():
        class_name = "SLUM" if class_id == 1 else "NON-SLUM"
        print(f"  RGB{rgb} -> {class_id} ({class_name})")
    
    print(f"\n⚠️  WARNING: This mapping is provisional!")
    print(f"   You may need to adjust it based on your specific annotation scheme.")
    print(f"   Please verify which RGB combination represents informal settlements.")
    
    return class_mapping

if __name__ == "__main__":
    rgb_combinations = decode_rgb_mask()
    class_mapping = create_class_mapping(rgb_combinations)
