import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_masks_and_images():
    """Visualize PNG masks alongside their corresponding satellite images"""
    
    # Look for tile images that might show informal settlements
    image_dir = "data/train/images"
    mask_dir = "data/train/masks"
    
    # Get some tile files that have corresponding masks
    tile_masks = [f for f in os.listdir(mask_dir) if f.startswith('tile_')]
    print(f"Found {len(tile_masks)} tile mask files")
    
    # Sample some masks to visualize
    sample_masks = tile_masks[:12]  # First 12 for visualization
    
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()
    
    slum_samples = []
    
    for i, mask_file in enumerate(sample_masks):
        if i >= len(axes) // 2:
            break
            
        # Load mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Find corresponding image
        img_file = mask_file.replace('.png', '.tif')
        img_path = os.path.join(image_dir, img_file)
        
        if os.path.exists(img_path):
            # Load image
            image = np.array(Image.open(img_path))
            
            # Show image
            axes[i*2].imshow(image)
            axes[i*2].set_title(f'Image: {img_file}')
            axes[i*2].axis('off')
            
            # Show mask
            axes[i*2+1].imshow(mask)
            axes[i*2+1].set_title(f'Mask: {mask_file}')
            axes[i*2+1].axis('off')
            
            # Check if this mask contains slums (RGB 250,235,185)
            slum_rgb = np.array([250, 235, 185])
            has_slums = np.any(np.all(mask == slum_rgb, axis=2))
            
            if has_slums:
                slum_samples.append((img_file, mask_file))
                axes[i*2].set_title(f'Image: {img_file} ✅ HAS SLUMS', color='red', fontweight='bold')
                axes[i*2+1].set_title(f'Mask: {mask_file} ✅ HAS SLUMS', color='red', fontweight='bold')
            
            # Show RGB values in mask
            unique_rgb = np.unique(mask.reshape(-1, 3), axis=0)
            print(f"\n{mask_file}:")
            print(f"  Has slums: {'YES' if has_slums else 'NO'}")
            print(f"  RGB values: {[tuple(rgb) for rgb in unique_rgb]}")
    
    plt.tight_layout()
    plt.savefig('mask_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nFound {len(slum_samples)} samples with slums:")
    for img_file, mask_file in slum_samples:
        print(f"  {img_file} -> {mask_file}")
    
    return slum_samples

def analyze_specific_masks(mask_files):
    """Analyze specific mask files in detail"""
    
    mask_dir = "data/train/masks"
    image_dir = "data/train/images"
    
    print("Detailed mask analysis:")
    print("=" * 50)
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            
            print(f"\n{mask_file}:")
            print(f"  Shape: {mask.shape}")
            
            # Get unique RGB combinations
            rgb_pixels = mask.reshape(-1, 3)
            unique_rgb, counts = np.unique(rgb_pixels, axis=0, return_counts=True)
            
            total_pixels = mask.shape[0] * mask.shape[1]
            
            print(f"  RGB color distribution:")
            for rgb, count in zip(unique_rgb, counts):
                percentage = (count / total_pixels) * 100
                color_name = get_color_name(tuple(rgb))
                print(f"    RGB{tuple(rgb)} ({color_name}): {count:,} pixels ({percentage:.1f}%)")
            
            # Check for slum class
            slum_rgb = (250, 235, 185)
            slum_mask = np.all(mask == slum_rgb, axis=2)
            slum_pixels = np.sum(slum_mask)
            
            if slum_pixels > 0:
                print(f"  🏚️  SLUM DETECTED: {slum_pixels:,} pixels ({(slum_pixels/total_pixels)*100:.1f}%)")
            else:
                print(f"  ✅ No slums in this tile")

def get_color_name(rgb):
    """Get descriptive name for RGB color"""
    r, g, b = rgb
    
    if rgb == (250, 235, 185):
        return "SLUMS/Informal settlements"
    elif rgb == (80, 140, 50):
        return "Vegetation"
    elif rgb == (40, 120, 240):
        return "Water"
    elif rgb == (200, 160, 40):
        return "Built-up/Impervious"
    elif rgb == (200, 200, 200):
        return "Built-up/Roads"
    elif rgb == (100, 100, 150):
        return "Water/Other"
    elif rgb == (0, 0, 0):
        return "Background/Unlabelled"
    else:
        return "Unknown class"

def find_slum_rich_samples():
    """Find samples with high proportion of slum pixels"""
    
    mask_dir = "data/train/masks"
    image_dir = "data/train/images"
    
    # Check all mask files
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    slum_samples = []
    slum_rgb = np.array([250, 235, 185])
    
    print("Scanning for slum-rich samples...")
    
    for mask_file in mask_files[:100]:  # Check first 100 for speed
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Check for slums
        slum_mask = np.all(mask == slum_rgb, axis=2)
        slum_pixels = np.sum(slum_mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        slum_percentage = (slum_pixels / total_pixels) * 100
        
        if slum_pixels > 0:
            slum_samples.append((mask_file, slum_pixels, slum_percentage))
    
    # Sort by slum percentage
    slum_samples.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nFound {len(slum_samples)} samples with slums:")
    print("Top 10 slum-rich samples:")
    print("-" * 50)
    
    for i, (mask_file, pixels, percentage) in enumerate(slum_samples[:10]):
        img_file = mask_file.replace('.png', '.tif')
        print(f"{i+1:2d}. {img_file:<25} | {pixels:>6,} pixels ({percentage:>5.1f}%)")
    
    return slum_samples

if __name__ == "__main__":
    print("Analyzing PNG masks for informal settlements...")
    
    # First, find slum-rich samples
    slum_samples = find_slum_rich_samples()
    
    if slum_samples:
        # Visualize some examples
        print("\nCreating visualizations...")
        visualize_masks_and_images()
        
        # Analyze top slum samples in detail
        top_masks = [sample[0] for sample in slum_samples[:5]]
        analyze_specific_masks(top_masks)
    else:
        print("No slum samples found in the first 100 masks checked.")
        print("Let's analyze a random sample of masks...")
        sample_masks = [f for f in os.listdir("data/train/masks") if f.endswith('.png')][:10]
        analyze_specific_masks(sample_masks)
