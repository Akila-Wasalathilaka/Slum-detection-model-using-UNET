"""
Debug script to check data types in the training pipeline.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import get_data_config, get_model_config
from utils.dataset import SlumDataset
from utils.transforms import get_train_transforms
from models.losses import create_loss

def debug_data_types():
    """Debug data types in the pipeline."""
    print("🔍 DEBUGGING DATA TYPES")
    print("=" * 30)
    
    # Load configs
    data_config = get_data_config('standard')
    model_config = get_model_config('balanced')
    
    # Get paths
    paths = data_config.get_paths()
    
    # Create dataset
    train_transforms = get_train_transforms(data_config)
    dataset = SlumDataset(
        images_dir=paths['train_images'],
        masks_dir=paths['train_masks'],
        transform=train_transforms,
        slum_rgb=data_config.slum_rgb,
        image_size=data_config.image_size,
        use_tile_masks_only=data_config.use_tile_masks_only,
        min_slum_pixels=data_config.min_slum_pixels,
        cache_masks=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"\nSample data types:")
        print(f"  Image: {image.dtype}, shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Mask: {mask.dtype}, shape: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Test with batch dimension
        image_batch = image.unsqueeze(0)
        mask_batch = mask.unsqueeze(0)
        
        if len(mask_batch.shape) == 3:
            mask_batch = mask_batch.unsqueeze(1)
        
        print(f"\nBatch data types:")
        print(f"  Image batch: {image_batch.dtype}, shape: {image_batch.shape}")
        print(f"  Mask batch: {mask_batch.dtype}, shape: {mask_batch.shape}")
        
        # Test loss function
        try:
            criterion = create_loss('combined')
            
            # Create dummy prediction with same shape as mask
            pred = torch.randn_like(mask_batch)
            
            print(f"\nTesting loss function:")
            print(f"  Prediction: {pred.dtype}, shape: {pred.shape}")
            print(f"  Target: {mask_batch.dtype}, shape: {mask_batch.shape}")
            
            # Ensure types are correct
            mask_batch = mask_batch.float()
            
            loss = criterion(pred, mask_batch)
            print(f"  Loss: {loss.item():.4f} - SUCCESS!")
            
        except Exception as e:
            print(f"  Loss computation failed: {e}")
            print(f"  Pred dtype: {pred.dtype}")
            print(f"  Target dtype: {mask_batch.dtype}")
    
    else:
        print("❌ No samples in dataset!")

if __name__ == "__main__":
    debug_data_types()
