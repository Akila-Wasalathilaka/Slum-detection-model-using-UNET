"""
Test model on Colombo tiles using the exact same approach that worked on training data.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.checkpoint import load_checkpoint
from models.unet import create_model

def get_correct_transforms():
    """Get the exact transforms that worked on training data."""
    return A.Compose([
        A.Resize(120, 120),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def test_colombo_tiles_directly():
    """Test model on Colombo tiles using exact same approach as successful training test."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model (same as successful test)
    model = create_model('unet')
    checkpoint = load_checkpoint('experiments/development_20250713_175410/checkpoints/best_checkpoint.pth', model, device=device)
    model.eval()
    
    # Get Colombo tiles
    tiles_dir = Path("colombo/tiles")
    tile_files = list(tiles_dir.glob("*.png"))[:5]  # Test first 5 tiles
    
    print(f"Testing {len(tile_files)} Colombo tiles with PROVEN approach...")
    
    transforms = get_correct_transforms()
    
    for i, tile_file in enumerate(tile_files):
        print(f"\n--- Colombo Tile {i+1}: {tile_file.name} ---")
        
        # Load exactly like successful test
        image = Image.open(tile_file).convert('RGB')
        image_np = np.array(image)
        
        print(f"Original image shape: {image_np.shape}")
        print(f"Original image range: {image_np.min()} - {image_np.max()}")
        
        # Apply exact same transforms as successful test
        transformed = transforms(image=image_np)
        image_tensor = transformed['image']
        
        print(f"Transformed image shape: {image_tensor.shape}")
        print(f"Transformed image range: {image_tensor.min():.3f} - {image_tensor.max():.3f}")
        
        # Inference (exact same as successful test)
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(device)
            prediction = model(image_batch)
            prob_map = torch.sigmoid(prediction).cpu().numpy()[0, 0]
        
        # Statistics (exact same as successful test)
        max_prob = prob_map.max()
        mean_prob = prob_map.mean()
        pred_pixels_05 = (prob_map > 0.5).sum()
        pred_pixels_03 = (prob_map > 0.3).sum()
        pred_pixels_01 = (prob_map > 0.1).sum()
        
        print(f"Max prediction probability: {max_prob:.4f}")
        print(f"Mean prediction probability: {mean_prob:.4f}")
        print(f"Predicted slum pixels (>0.5): {pred_pixels_05}")
        print(f"Predicted slum pixels (>0.3): {pred_pixels_03}")
        print(f"Predicted slum pixels (>0.1): {pred_pixels_01}")
        
        if max_prob > 0.5:
            print("✅ Strong slum detection!")
        elif max_prob > 0.3:
            print("🟡 Moderate slum detection")
        else:
            print("❌ Weak/no slum detection")
        
        # Create visualization (same as successful test)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Denormalize for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_display = image_tensor.permute(1, 2, 0).numpy()
        image_display = (image_display * std + mean)
        image_display = np.clip(image_display, 0, 1)
        
        # Row 1
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Tile')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(image_display)
        axes[0, 1].set_title('Normalized for Model')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(prob_map, cmap='Reds', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Predictions\nMax: {max_prob:.3f}')
        axes[0, 2].axis('off')
        
        # Row 2: Different thresholds
        axes[1, 0].imshow(prob_map > 0.5, cmap='Reds')
        axes[1, 0].set_title(f'Threshold > 0.5\n{pred_pixels_05} pixels')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(prob_map > 0.3, cmap='Reds')
        axes[1, 1].set_title(f'Threshold > 0.3\n{pred_pixels_03} pixels')
        axes[1, 1].axis('off')
        
        # Overlay
        axes[1, 2].imshow(image_np)
        axes[1, 2].imshow(prob_map, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[1, 2].set_title('Original + Predictions')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'colombo_tile_test_{i+1}_{tile_file.stem}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: colombo_tile_test_{i+1}_{tile_file.stem}.png")

if __name__ == "__main__":
    test_colombo_tiles_directly()
