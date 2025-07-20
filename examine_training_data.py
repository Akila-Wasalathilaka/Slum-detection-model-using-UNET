"""
Examine the training data to understand what type of slums the model learned to detect.
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

def examine_training_slums():
    """Look at the actual training data to understand what slums look like."""
    
    # Load some training data
    train_images_dir = project_root / "data/train/images"
    train_masks_dir = project_root / "data/train/masks"
    
    # Get tile images that have significant slum content
    dataset = SlumDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=None,
        use_tile_masks_only=True,
        min_slum_percentage=0.3  # Only images with >30% slums
    )
    
    print(f"Found {len(dataset)} training images with >30% slum content")
    
    if len(dataset) == 0:
        print("❌ No slum-rich training images found!")
        return
    
    # Examine first 3 high-slum-content images
    for i in range(min(3, len(dataset))):
        image, mask = dataset[i]
        image_path = dataset.image_paths[i]
        mask_path = dataset.mask_paths[i]
        
        print(f"\n--- Training Sample {i+1}: {Path(image_path).name} ---")
        
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        original_image_np = np.array(original_image)
        
        # Load original mask 
        original_mask = dataset._load_binary_mask(mask_path)
        
        slum_percentage = (original_mask.sum() / original_mask.size) * 100
        print(f"Slum percentage: {slum_percentage:.1f}%")
        print(f"Slum pixels: {original_mask.sum()}")
        print(f"Total pixels: {original_mask.size}")
        
        # Create detailed visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Original data
        axes[0, 0].imshow(original_image_np)
        axes[0, 0].set_title(f'Training Image\n{Path(image_path).name}')
        axes[0, 0].axis('off')
        
        # Show mask
        axes[0, 1].imshow(original_mask, cmap='Reds', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Ground Truth Mask\n{slum_percentage:.1f}% slums')
        axes[0, 1].axis('off')
        
        # Show overlay
        axes[0, 2].imshow(original_image_np)
        axes[0, 2].imshow(original_mask, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[0, 2].set_title('Image + Mask Overlay')
        axes[0, 2].axis('off')
        
        # Row 2: Detailed analysis
        # Zoom into slum areas
        slum_indices = np.where(original_mask > 0)
        if len(slum_indices[0]) > 0:
            min_row, max_row = slum_indices[0].min(), slum_indices[0].max()
            min_col, max_col = slum_indices[1].min(), slum_indices[1].max()
            
            # Add some padding
            padding = 10
            min_row = max(0, min_row - padding)
            max_row = min(original_image_np.shape[0], max_row + padding)
            min_col = max(0, min_col - padding)
            max_col = min(original_image_np.shape[1], max_col + padding)
            
            # Zoom into slum area
            zoom_image = original_image_np[min_row:max_row, min_col:max_col]
            zoom_mask = original_mask[min_row:max_row, min_col:max_col]
            
            axes[1, 0].imshow(zoom_image)
            axes[1, 0].set_title('Zoomed: Slum Area')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(zoom_mask, cmap='Reds', vmin=0, vmax=1)
            axes[1, 1].set_title('Zoomed: Slum Mask')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(zoom_image)
            axes[1, 2].imshow(zoom_mask, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
            axes[1, 2].set_title('Zoomed: Combined')
            axes[1, 2].axis('off')
        else:
            # If no slums found (shouldn't happen with our filter)
            axes[1, 0].text(0.5, 0.5, 'No slums found', ha='center', va='center')
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'training_slum_analysis_{i+1}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: training_slum_analysis_{i+1}.png")

def load_and_compare_colombo():
    """Load a Colombo tile for comparison."""
    
    tiles_dir = Path("colombo/tiles")
    if not tiles_dir.exists():
        print("❌ Colombo tiles directory not found")
        return
    
    tile_files = list(tiles_dir.glob("*.png"))
    if not tile_files:
        print("❌ No Colombo tiles found")
        return
    
    print(f"\n🏙️ Loading Colombo tile for comparison...")
    
    # Load first tile
    colombo_tile = Image.open(tile_files[0]).convert('RGB')
    colombo_np = np.array(colombo_tile)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(colombo_np)
    axes[0].set_title(f'Colombo Tile\n{tile_files[0].name}')
    axes[0].axis('off')
    
    # Add text describing what we see
    axes[1].text(0.1, 0.9, 'Colombo Characteristics:', fontsize=14, weight='bold', transform=axes[1].transAxes)
    axes[1].text(0.1, 0.8, '• Regular street grid pattern', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.7, '• Mixed building sizes', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.6, '• Some dense housing areas', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.5, '• Concrete/paved roads', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.4, '• More organized layout', fontsize=12, transform=axes[1].transAxes)
    
    axes[1].text(0.1, 0.2, 'Model Training Data:', fontsize=14, weight='bold', transform=axes[1].transAxes)
    axes[1].text(0.1, 0.1, '• Specific slum RGB signature (250,235,185)', fontsize=12, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.0, '• May be different architectural style', fontsize=12, transform=axes[1].transAxes)
    
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('colombo_vs_training_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Saved: colombo_vs_training_comparison.png")

if __name__ == "__main__":
    print("🔍 EXAMINING TRAINING DATA TO UNDERSTAND SLUM DETECTION")
    print("=" * 60)
    examine_training_slums()
    load_and_compare_colombo()
    print("\n📊 Analysis complete! Check the generated images to compare training vs Colombo data.")
