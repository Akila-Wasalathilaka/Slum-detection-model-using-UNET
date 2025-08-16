"""
Investigate mask format and values to understand why model predictions are poor.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def investigate_masks():
    """Check the actual mask values and format."""
    
    project_root = Path(__file__).parent
    train_masks_dir = project_root / "data/train/masks"
    
    # Get some tile_ mask files
    mask_files = list(train_masks_dir.glob("tile_*.png"))[:5]
    
    print(f"Investigating {len(mask_files)} mask files...")
    
    for i, mask_file in enumerate(mask_files):
        print(f"\n--- Mask {i+1}: {mask_file.name} ---")
        
        # Load mask
        mask = Image.open(mask_file)
        print(f"Original mask mode: {mask.mode}")
        print(f"Original mask size: {mask.size}")
        
        # Convert to different formats and check
        mask_l = mask.convert('L')
        mask_np = np.array(mask_l)
        
        print(f"Mask array shape: {mask_np.shape}")
        print(f"Mask array dtype: {mask_np.dtype}")
        print(f"Unique values: {sorted(np.unique(mask_np))}")
        print(f"Value counts:")
        unique, counts = np.unique(mask_np, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  {val}: {count} pixels ({count/mask_np.size*100:.1f}%)")
        
        # Check if it's RGB and has slum color
        if mask.mode == 'RGB':
            mask_rgb = np.array(mask)
            print(f"RGB mask shape: {mask_rgb.shape}")
            
            # Check for slum color (250, 235, 185)
            slum_color = np.array([250, 235, 185])
            slum_pixels = np.all(mask_rgb == slum_color, axis=2)
            slum_count = np.sum(slum_pixels)
            print(f"Slum color (250,235,185) pixels: {slum_count}")
            
            # Check other common colors
            for color_name, color_val in [("white", [255, 255, 255]), ("black", [0, 0, 0]), ("red", [255, 0, 0])]:
                color_pixels = np.all(mask_rgb == color_val, axis=2)
                color_count = np.sum(color_pixels)
                print(f"{color_name} ({color_val}) pixels: {color_count}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original mask
        axes[0].imshow(mask)
        axes[0].set_title(f'Original Mask ({mask.mode})')
        axes[0].axis('off')
        
        # Grayscale version
        axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Grayscale (values: {mask_np.min()}-{mask_np.max()})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'mask_investigation_{i+1}_{mask_file.stem}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: mask_investigation_{i+1}_{mask_file.stem}.png")

if __name__ == "__main__":
    investigate_masks()
