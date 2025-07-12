import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def find_tile_masks_with_slums():
    """Find tile_* masks that contain slums"""
    
    mask_dir = "data/train/masks"
    image_dir = "data/train/images"
    
    # Get tile mask files
    tile_masks = [f for f in os.listdir(mask_dir) if f.startswith('tile_') and f.endswith('.png')]
    
    print(f"Found {len(tile_masks)} tile mask files")
    
    slum_samples = []
    slum_rgb = np.array([250, 235, 185])
    
    print("Scanning tile masks for slums...")
    
    for mask_file in tile_masks:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Check for slums
        slum_mask = np.all(mask == slum_rgb, axis=2)
        slum_pixels = np.sum(slum_mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        if slum_pixels > 0:
            slum_percentage = (slum_pixels / total_pixels) * 100
            slum_samples.append((mask_file, slum_pixels, slum_percentage))
            
            # Also get unique RGB values
            unique_rgb = np.unique(mask.reshape(-1, 3), axis=0)
            print(f"✅ {mask_file}: {slum_pixels:,} slum pixels ({slum_percentage:.1f}%)")
            print(f"   RGB values: {[tuple(rgb) for rgb in unique_rgb]}")
    
    if not slum_samples:
        print("❌ No slums found in tile masks. Let me check what RGB values are present...")
        
        # Check first few tile masks to see what RGB values they contain
        sample_tiles = tile_masks[:10]
        for mask_file in sample_tiles:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = np.array(Image.open(mask_path))
            unique_rgb = np.unique(mask.reshape(-1, 3), axis=0)
            print(f"\n{mask_file}:")
            for rgb in unique_rgb:
                rgb_tuple = tuple(rgb)
                color_name = get_color_name(rgb_tuple)
                print(f"  RGB{rgb_tuple} - {color_name}")
    
    return slum_samples

def get_color_name(rgb):
    """Get descriptive name for RGB color"""
    r, g, b = rgb
    
    color_map = {
        (250, 235, 185): "🏚️  SLUMS/Informal settlements",
        (80, 140, 50): "🌿 Vegetation", 
        (40, 120, 240): "💧 Water",
        (200, 160, 40): "🏢 Built-up/Impervious",
        (200, 200, 200): "🛣️  Built-up/Roads",
        (100, 100, 150): "💧 Water/Other",
        (0, 0, 0): "⚫ Background/Unlabelled"
    }
    
    return color_map.get(rgb, f"❓ Unknown class")

def visualize_tile_masks():
    """Visualize tile masks and their corresponding images"""
    
    mask_dir = "data/train/masks" 
    image_dir = "data/train/images"
    
    # Get some tile masks
    tile_masks = [f for f in os.listdir(mask_dir) if f.startswith('tile_')][:12]
    
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, mask_file in enumerate(tile_masks):
        if i >= len(axes) // 2:
            break
            
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Find corresponding image
        img_file = mask_file.replace('.png', '.tif')
        img_path = os.path.join(image_dir, img_file)
        
        if os.path.exists(img_path):
            image = np.array(Image.open(img_path))
            
            # Show image
            axes[i*2].imshow(image)
            axes[i*2].set_title(f'Satellite: {img_file}', fontsize=10)
            axes[i*2].axis('off')
            
            # Show mask with different classes colored
            axes[i*2+1].imshow(mask)
            axes[i*2+1].set_title(f'Mask: {mask_file}', fontsize=10)
            axes[i*2+1].axis('off')
            
            # Check for slums
            slum_rgb = np.array([250, 235, 185])
            has_slums = np.any(np.all(mask == slum_rgb, axis=2))
            
            if has_slums:
                axes[i*2].set_title(f'🏚️  {img_file}', color='red', fontweight='bold', fontsize=10)
                axes[i*2+1].set_title(f'🏚️  {mask_file}', color='red', fontweight='bold', fontsize=10)
            
            # Add border colors based on content
            unique_rgb = np.unique(mask.reshape(-1, 3), axis=0)
            if len(unique_rgb) > 1:
                # Multi-class mask
                for ax in [axes[i*2], axes[i*2+1]]:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('tile_masks_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'tile_masks_visualization.png'")

def check_specific_tiles():
    """Check if we can find masks for the informal settlement images shown"""
    
    # The images shown were tile_1.87_* 
    # Let's see if we have any similar tiles in our dataset
    
    mask_dir = "data/train/masks"
    image_dir = "data/train/images"
    
    # Look for any tile masks that might contain similar patterns
    tile_masks = [f for f in os.listdir(mask_dir) if f.startswith('tile_')]
    
    print("Checking for masks that might correspond to informal settlement areas...")
    
    # Check different tile patterns
    patterns_to_check = ['tile_1.8', 'tile_1.9', 'tile_2.', 'tile_3.', 'tile_4.', 'tile_5.', 'tile_6.']
    
    for pattern in patterns_to_check:
        matching_tiles = [f for f in tile_masks if pattern in f]
        if matching_tiles:
            print(f"\nFound {len(matching_tiles)} masks with pattern '{pattern}':")
            for tile in matching_tiles[:5]:  # Show first 5
                print(f"  {tile}")
                
                # Check if corresponding image exists
                img_file = tile.replace('.png', '.tif')
                img_path = os.path.join(image_dir, img_file)
                if os.path.exists(img_path):
                    print(f"    ✅ Has corresponding image: {img_file}")
                else:
                    print(f"    ❌ No corresponding image found")

if __name__ == "__main__":
    print("Searching for PNG masks with informal settlements...")
    print("=" * 60)
    
    # Find slum samples
    slum_samples = find_tile_masks_with_slums()
    
    # Visualize tile masks
    print(f"\nCreating visualization...")
    visualize_tile_masks()
    
    # Check for specific patterns
    print(f"\n" + "=" * 60)
    check_specific_tiles()
