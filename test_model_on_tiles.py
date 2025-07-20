"""
Test the model on tile_ images to verify it works correctly.
"""

import torch
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
from utils.checkpoint import load_checkpoint
from models.unet import create_model

def test_on_tiles():
    """Test model on tile_ images that it was actually trained on."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = create_model('unet')
    checkpoint = load_checkpoint('experiments/development_20250713_175410/checkpoints/best_checkpoint.pth', model, device=device)
    model.eval()
    
    # Create dataset (this will only load tile_ images)
    train_dataset = SlumDataset(
        images_dir=project_root / "data/train/images",
        masks_dir=project_root / "data/train/masks",
        transform=None,  # Use None for now
        use_tile_masks_only=True  # This is the key parameter
    )
    
    print(f"Dataset loaded {len(train_dataset)} tile_ images")
    
    if len(train_dataset) == 0:
        print("❌ No images loaded! Dataset filtering issue.")
        return
    
    # Test on first 5 images
    print("\n🧪 Testing model predictions on tile_ images...")
    
    for i in range(min(5, len(train_dataset))):
        image, mask = train_dataset[i]
        
        # Get image info
        image_path = train_dataset.image_paths[i]
        image_name = Path(image_path).name
        
        print(f"\n--- Image {i+1}: {image_name} ---")
        
        # Run prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            prediction = model(image_tensor)
            prob_map = torch.sigmoid(prediction).cpu().numpy()[0, 0]
        
        # Calculate statistics
        true_slum_pixels = (mask.numpy() > 0.5).sum()
        pred_slum_pixels = (prob_map > 0.5).sum()
        
        print(f"Ground truth slum pixels: {true_slum_pixels}")
        print(f"Predicted slum pixels: {pred_slum_pixels}")
        print(f"Max prediction probability: {prob_map.max():.4f}")
        print(f"Mean prediction probability: {prob_map.mean():.4f}")
        print(f"Prediction threshold 0.5 pixels: {(prob_map > 0.5).sum()}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(mask.numpy(), cmap='Reds', alpha=0.7)
        axes[1].set_title(f'Ground Truth\n{true_slum_pixels} slum pixels')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(prob_map, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f'Prediction\nMax: {prob_map.max():.3f}, Mean: {prob_map.mean():.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'tile_test_{i+1}_{image_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: tile_test_{i+1}_{image_name}.png")

if __name__ == "__main__":
    test_on_tiles()
