"""
Analyze pixel distribution differences between training data and Colombo data.
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
from utils.transforms import get_test_transforms

def analyze_pixel_distributions():
    """Compare pixel value distributions between training and Colombo data."""
    
    print("🔍 ANALYZING PIXEL DISTRIBUTIONS")
    print("=" * 50)
    
    # Load training data with slums
    train_images_dir = project_root / "data/train/images"
    train_masks_dir = project_root / "data/train/masks"
    
    dataset = SlumDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=None,
        use_tile_masks_only=True,
        min_slum_percentage=0.3
    )
    
    print(f"Training dataset: {len(dataset)} high-slum images")
    
    # Sample training data
    train_pixels = []
    train_slum_pixels = []
    train_non_slum_pixels = []
    
    for i in range(min(10, len(dataset))):
        image, mask = dataset[i]
        image_path = dataset.image_paths[i]
        mask_path = dataset.mask_paths[i]
        
        # Load original image and mask
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        mask_np = dataset._load_binary_mask(mask_path)
        
        # Collect pixel values
        train_pixels.extend(img_np.flatten())
        
        # Separate slum vs non-slum pixels
        slum_mask = mask_np.flatten() > 0
        train_slum_pixels.extend(img_np.flatten()[slum_mask])
        train_non_slum_pixels.extend(img_np.flatten()[~slum_mask])
    
    # Load Colombo data
    tiles_dir = Path("colombo/tiles")
    colombo_pixels = []
    
    if tiles_dir.exists():
        tile_files = list(tiles_dir.glob("*.png"))[:10]  # First 10 tiles
        
        for tile_file in tile_files:
            img = Image.open(tile_file).convert('RGB')
            img_np = np.array(img)
            colombo_pixels.extend(img_np.flatten())
    
    # Convert to arrays
    train_pixels = np.array(train_pixels)
    train_slum_pixels = np.array(train_slum_pixels)
    train_non_slum_pixels = np.array(train_non_slum_pixels)
    colombo_pixels = np.array(colombo_pixels)
    
    print(f"Training pixels: {len(train_pixels):,}")
    print(f"Training slum pixels: {len(train_slum_pixels):,}")
    print(f"Training non-slum pixels: {len(train_non_slum_pixels):,}")
    print(f"Colombo pixels: {len(colombo_pixels):,}")
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall distribution comparison
    axes[0, 0].hist(train_pixels, bins=50, alpha=0.7, label='Training', density=True)
    axes[0, 0].hist(colombo_pixels, bins=50, alpha=0.7, label='Colombo', density=True)
    axes[0, 0].set_title('Overall Pixel Value Distribution')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training: Slum vs Non-Slum
    axes[0, 1].hist(train_slum_pixels, bins=50, alpha=0.7, label='Slum Areas', density=True)
    axes[0, 1].hist(train_non_slum_pixels, bins=50, alpha=0.7, label='Non-Slum Areas', density=True)
    axes[0, 1].set_title('Training Data: Slum vs Non-Slum Pixels')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Statistics comparison
    stats_text = f"""
    STATISTICS COMPARISON:
    
    Training Data:
    • Mean: {train_pixels.mean():.1f}
    • Std: {train_pixels.std():.1f}
    • Min: {train_pixels.min()}
    • Max: {train_pixels.max()}
    
    Training Slum Areas:
    • Mean: {train_slum_pixels.mean():.1f}
    • Std: {train_slum_pixels.std():.1f}
    
    Training Non-Slum Areas:
    • Mean: {train_non_slum_pixels.mean():.1f}
    • Std: {train_non_slum_pixels.std():.1f}
    
    Colombo Data:
    • Mean: {colombo_pixels.mean():.1f}
    • Std: {colombo_pixels.std():.1f}
    • Min: {colombo_pixels.min()}
    • Max: {colombo_pixels.max()}
    """
    
    axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 0].axis('off')
    
    # Domain shift visualization
    domain_shift_text = f"""
    DOMAIN SHIFT ANALYSIS:
    
    Mean Difference:
    • Training vs Colombo: {abs(train_pixels.mean() - colombo_pixels.mean()):.1f}
    
    Standard Deviation Difference:
    • Training vs Colombo: {abs(train_pixels.std() - colombo_pixels.std()):.1f}
    
    POTENTIAL ISSUES:
    
    1. Different Image Sources:
       • Training data may be from different satellite/sensor
       • Different time periods, lighting conditions
       • Different preprocessing pipelines
    
    2. Geographic Differences:
       • Different architectural styles
       • Different materials used in construction
       • Different urban planning patterns
    
    3. Color/Brightness Shift:
       • Model trained on specific RGB range
       • Colombo data may have different exposure/contrast
    
    RECOMMENDATIONS:
    
    1. Apply histogram matching to Colombo data
    2. Retrain model with more diverse data
    3. Use domain adaptation techniques
    4. Apply color normalization/standardization
    """
    
    axes[1, 1].text(0.05, 0.95, domain_shift_text, transform=axes[1, 1].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('pixel_distribution_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Saved: pixel_distribution_analysis.png")
    
    return {
        'train_mean': train_pixels.mean(),
        'train_std': train_pixels.std(),
        'colombo_mean': colombo_pixels.mean(),
        'colombo_std': colombo_pixels.std(),
        'slum_mean': train_slum_pixels.mean(),
        'slum_std': train_slum_pixels.std()
    }

def test_histogram_matching():
    """Test if histogram matching improves predictions."""
    
    print("\n🔧 TESTING HISTOGRAM MATCHING")
    print("=" * 40)
    
    from skimage import exposure
    import torch
    from utils.checkpoint import load_checkpoint
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _, _, _ = load_checkpoint('models/best_model.pth', device)
    model.eval()
    
    # Get transforms
    transforms = get_test_transforms()
    
    # Load training reference
    train_images_dir = project_root / "data/train/images"
    train_files = list(train_images_dir.glob("*.tif"))
    if train_files:
        ref_img = Image.open(train_files[0]).convert('RGB')
        ref_img_np = np.array(ref_img)
    else:
        print("❌ No training images found for reference")
        return
    
    # Load Colombo tile
    tiles_dir = Path("colombo/tiles")
    tile_files = list(tiles_dir.glob("*.png"))
    if not tile_files:
        print("❌ No Colombo tiles found")
        return
    
    colombo_img = Image.open(tile_files[0]).convert('RGB')
    colombo_np = np.array(colombo_img)
    
    # Apply histogram matching
    matched_np = exposure.match_histograms(colombo_np, ref_img_np, channel_axis=2)
    matched_img = Image.fromarray(matched_np.astype(np.uint8))
    
    # Test predictions on original vs matched
    def predict_on_image(img):
        img_tensor = transforms(image=np.array(img))['image'].unsqueeze(0)
        with torch.no_grad():
            pred = model(img_tensor)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
        return pred_prob
    
    original_pred = predict_on_image(colombo_img)
    matched_pred = predict_on_image(matched_img)
    
    print(f"Original Colombo prediction:")
    print(f"  Mean: {original_pred.mean():.4f}")
    print(f"  Max: {original_pred.max():.4f}")
    print(f"  Slum pixels (>0.5): {(original_pred > 0.5).sum()}")
    
    print(f"\nHistogram-matched prediction:")
    print(f"  Mean: {matched_pred.mean():.4f}")
    print(f"  Max: {matched_pred.max():.4f}")
    print(f"  Slum pixels (>0.5): {(matched_pred > 0.5).sum()}")
    
    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Images
    axes[0, 0].imshow(ref_img_np)
    axes[0, 0].set_title('Training Reference')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(colombo_np)
    axes[0, 1].set_title('Original Colombo')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(matched_np.astype(np.uint8))
    axes[0, 2].set_title('Histogram Matched')
    axes[0, 2].axis('off')
    
    # Row 2: Predictions
    axes[1, 0].imshow(np.zeros_like(original_pred), cmap='gray')
    axes[1, 0].set_title('Reference (no prediction)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(original_pred, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Original Prediction\nMax: {original_pred.max():.3f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(matched_pred, cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Matched Prediction\nMax: {matched_pred.max():.3f}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('histogram_matching_test.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Saved: histogram_matching_test.png")
    
    improvement = matched_pred.max() / original_pred.max() if original_pred.max() > 0 else float('inf')
    print(f"\nImprovement factor: {improvement:.2f}x")
    
    return improvement > 2.0  # Return True if significant improvement

if __name__ == "__main__":
    stats = analyze_pixel_distributions()
    
    # Test if histogram matching helps
    try:
        improved = test_histogram_matching()
        if improved:
            print("\n✅ Histogram matching shows significant improvement!")
            print("💡 SOLUTION: Apply histogram matching to Colombo data before prediction")
        else:
            print("\n❌ Histogram matching doesn't solve the issue")
            print("💡 The problem may be deeper - different slum characteristics")
    except Exception as e:
        print(f"\n❌ Histogram matching test failed: {e}")
        print("Install scikit-image: pip install scikit-image")
