"""
Test the SlumDataset's mask conversion process directly to debug the issue.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.dataset import SlumDataset

def test_mask_conversion():
    """Test the dataset's mask conversion process."""
    
    # Create dataset with different configurations
    print("🧪 Testing SlumDataset mask conversion...")
    
    dataset = SlumDataset(
        images_dir=project_root / "data/train/images",
        masks_dir=project_root / "data/train/masks",
        transform=None,
        use_tile_masks_only=True,
        slum_rgb=(250, 235, 185)  # Explicitly set the slum color
    )
    
    print(f"Dataset loaded {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("❌ No images loaded!")
        return
    
    # Test first 3 samples
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i+1} ---")
        
        # Get the data
        image, mask = dataset[i]
        image_path = dataset.image_paths[i]
        mask_path = dataset.mask_paths[i]
        
        print(f"Image: {Path(image_path).name}")
        print(f"Mask: {Path(mask_path).name}")
        
        # Check the processed data
        print(f"Image tensor shape: {image.shape}")
        print(f"Image tensor range: {image.min():.3f} - {image.max():.3f}")
        print(f"Mask tensor shape: {mask.shape}")
        print(f"Mask tensor dtype: {mask.dtype}")
        print(f"Mask unique values: {torch.unique(mask)}")
        print(f"Mask sum (slum pixels): {mask.sum()}")
        
        # Load the original mask to compare
        original_mask = dataset._load_binary_mask(mask_path)
        print(f"Original binary mask shape: {original_mask.shape}")
        print(f"Original binary mask sum: {original_mask.sum()}")
        print(f"Original binary mask unique values: {np.unique(original_mask)}")
        
        # Visualize the conversion
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Image processing
        if image.shape[0] == 3:  # If CHW format
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = image.numpy()
        
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Processed Image')
        axes[0, 0].axis('off')
        
        # Load original RGB mask for display
        original_rgb_mask = Image.open(mask_path)
        axes[0, 1].imshow(original_rgb_mask)
        axes[0, 1].set_title('Original RGB Mask')
        axes[0, 1].axis('off')
        
        # Show original binary mask
        axes[0, 2].imshow(original_mask, cmap='Reds', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Binary Mask\n{original_mask.sum()} slum pixels')
        axes[0, 2].axis('off')
        
        # Row 2: Final processed data
        if len(mask.shape) == 3:
            mask_display = mask[0].numpy() if mask.shape[0] == 1 else mask.numpy()
        else:
            mask_display = mask.numpy()
            
        axes[1, 0].imshow(mask_display, cmap='Reds', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Final Mask Tensor\n{mask.sum()} slum pixels')
        axes[1, 0].axis('off')
        
        # Show difference between original and final
        difference = original_mask - mask_display
        axes[1, 1].imshow(difference, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 1].set_title(f'Difference\n(red=lost, blue=gained)')
        axes[1, 1].axis('off')
        
        # Show overlay
        axes[1, 2].imshow(image_np)
        axes[1, 2].imshow(mask_display, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[1, 2].set_title('Image + Mask Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'mask_conversion_test_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: mask_conversion_test_{i+1}.png")

if __name__ == "__main__":
    import torch
    test_mask_conversion()
