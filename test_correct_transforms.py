"""
Test model with correct normalization (same as training).
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

from utils.dataset import SlumDataset
from utils.checkpoint import load_checkpoint
from models.unet import create_model

def get_test_transforms_correct():
    """Get the correct test transforms (same as training)."""
    return A.Compose([
        A.Resize(120, 120),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def test_with_correct_transforms():
    """Test model with the exact same transforms used during training."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = create_model('unet')
    checkpoint = load_checkpoint('experiments/development_20250713_175410/checkpoints/best_checkpoint.pth', model, device=device)
    model.eval()
    
    # Create dataset with correct transforms
    dataset = SlumDataset(
        images_dir=project_root / "data/train/images",
        masks_dir=project_root / "data/train/masks",
        transform=get_test_transforms_correct(),  # Use correct transforms!
        use_tile_masks_only=True,
        slum_rgb=(250, 235, 185),
        image_size=(120, 120)
    )
    
    print(f"Dataset loaded {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("❌ No images loaded!")
        return
    
    # Test on first 5 images
    print("\n🧪 Testing model with CORRECT normalization...")
    
    total_correct_predictions = 0
    
    for i in range(min(5, len(dataset))):
        image, mask = dataset[i]
        
        # Get image info
        image_path = dataset.image_paths[i]
        image_name = Path(image_path).name
        
        print(f"\n--- Image {i+1}: {image_name} ---")
        print(f"Image tensor shape: {image.shape}")
        print(f"Image tensor range: {image.min():.3f} - {image.max():.3f}")
        print(f"Mask tensor shape: {mask.shape}")
        print(f"True slum pixels: {mask.sum()}")
        
        # Run prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            prediction = model(image_tensor)
            prob_map = torch.sigmoid(prediction).cpu().numpy()[0, 0]
        
        # Calculate statistics
        true_slum_pixels = (mask.numpy() > 0.5).sum()
        pred_slum_pixels = (prob_map > 0.5).sum()
        max_prob = prob_map.max()
        mean_prob = prob_map.mean()
        
        print(f"Max prediction probability: {max_prob:.4f}")
        print(f"Mean prediction probability: {mean_prob:.4f}")
        print(f"Predicted slum pixels (>0.5): {pred_slum_pixels}")
        print(f"Predicted slum pixels (>0.3): {(prob_map > 0.3).sum()}")
        print(f"Predicted slum pixels (>0.1): {(prob_map > 0.1).sum()}")
        
        # Calculate accuracy metrics
        if max_prob > 0.5:
            total_correct_predictions += 1
            print("✅ Model detected slums!")
        else:
            print("❌ Model failed to detect slums")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Denormalize image for display
        image_display = image.permute(1, 2, 0).numpy()
        # Reverse normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_display = (image_display * std + mean)
        image_display = np.clip(image_display, 0, 1)
        
        # Row 1
        axes[0, 0].imshow(image_display)
        axes[0, 0].set_title('Normalized Image\n(Training format)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask.numpy(), cmap='Reds', alpha=0.8)
        axes[0, 1].set_title(f'Ground Truth\n{true_slum_pixels} slum pixels')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(prob_map, cmap='Reds', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Prediction\nMax: {max_prob:.3f}')
        axes[0, 2].axis('off')
        
        # Row 2: Different thresholds
        axes[1, 0].imshow(prob_map > 0.5, cmap='Reds')
        axes[1, 0].set_title(f'Threshold > 0.5\n{pred_slum_pixels} pixels')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(prob_map > 0.3, cmap='Reds')
        axes[1, 1].set_title(f'Threshold > 0.3\n{(prob_map > 0.3).sum()} pixels')
        axes[1, 1].axis('off')
        
        # Overlay
        axes[1, 2].imshow(image_display)
        axes[1, 2].imshow(prob_map, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[1, 2].set_title('Image + Prediction')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'correct_transforms_test_{i+1}_{image_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: correct_transforms_test_{i+1}_{image_name}.png")
    
    print(f"\n🎯 Summary: {total_correct_predictions}/5 images had good predictions (max prob > 0.5)")

if __name__ == "__main__":
    test_with_correct_transforms()
