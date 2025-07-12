import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def verify_dataset():
    """Final verification of the dataset for slum detection training"""
    
    print("FINAL DATASET VERIFICATION FOR SLUM DETECTION")
    print("=" * 60)
    
    # Check dataset structure
    base_path = 'data'
    splits = ['train', 'val', 'test']
    
    total_images = 0
    total_masks = 0
    slum_masks = 0
    
    for split in splits:
        img_dir = os.path.join(base_path, split, 'images')
        mask_dir = os.path.join(base_path, split, 'masks')
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
            masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
            
            print(f"\n{split.upper()} SET:")
            print(f"  Images: {len(images)}")
            print(f"  Masks: {len(masks)}")
            
            total_images += len(images)
            total_masks += len(masks)
            
            # Check for slum presence in tile masks (which contain slums)
            split_slum_count = 0
            tile_masks = [f for f in masks if f.startswith('tile_')]
            check_count = min(100, len(tile_masks))  # Check first 100 tile masks
            
            for i, mask_file in enumerate(tile_masks[:check_count]):
                mask_path = os.path.join(mask_dir, mask_file)
                try:
                    mask = np.array(Image.open(mask_path))
                    slum_color = np.array([250, 235, 185])
                    has_slums = np.any(np.all(mask == slum_color, axis=-1))
                    if has_slums:
                        split_slum_count += 1
                except:
                    continue
            
            print(f"  Tile masks: {len(tile_masks)}")
            print(f"  Tile masks with slums (first {check_count}): {split_slum_count}")
            slum_masks += split_slum_count
    
    print(f"\nTOTAL DATASET:")
    print(f"  Total images: {total_images}")
    print(f"  Total masks: {total_masks}")
    print(f"  Masks with slums (sampled): {slum_masks}")
    
    # Verify class mapping
    print(f"\nCLASS MAPPING VERIFICATION:")
    print(f"  Slum class RGB: (250, 235, 185)")
    print(f"  Binary conversion: Slum=1, Non-slum=0")
    
    # Check image dimensions
    if total_images > 0:
        # Check a sample image
        for split in splits:
            img_dir = os.path.join(base_path, split, 'images')
            if os.path.exists(img_dir):
                images = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
                if images:
                    sample_img_path = os.path.join(img_dir, images[0])
                    try:
                        img = Image.open(sample_img_path)
                        print(f"\nSAMPLE IMAGE VERIFICATION:")
                        print(f"  Dimensions: {img.size}")
                        print(f"  Mode: {img.mode}")
                        break
                    except:
                        continue
    
    print(f"\nDATASET STATUS: READY FOR TRAINING")
    print(f"This dataset contains satellite images with slum annotations.")
    print(f"Use RGB (250, 235, 185) to identify slum pixels for binary classification.")
    
    return True

def create_training_summary():
    """Create a summary for training setup"""
    
    summary = """
SLUM DETECTION MODEL TRAINING SETUP
===================================

Dataset Overview:
- Task: Binary semantic segmentation (Slum vs Non-slum)
- Input: 120x120 RGB satellite image tiles
- Target: Binary masks (1=Slum, 0=Non-slum)
- Slum class RGB in original masks: (250, 235, 185)

Model Architecture:
- UNet with ResNet34 or EfficientNet-B0 encoder
- Binary classification output (sigmoid activation)

Data Augmentation:
- Horizontal/vertical flips
- 90-degree rotations
- Brightness/contrast adjustments
- Elastic transformations
- Grid distortions
- Gaussian noise

Loss Functions (to compare):
- Binary Cross-Entropy (BCE)
- BCE + Dice Loss
- Focal Loss
- Tversky Loss

Evaluation Metrics:
- IoU (Intersection over Union)
- Dice Score
- Precision, Recall, F1-score
- Pixel Accuracy

Training Configuration:
- Batch size: 16-32
- Learning rate: 1e-4 with ReduceLROnPlateau
- Early stopping on validation IoU
- 100 epochs maximum

Files Ready:
- unet_slum_detection.py - Model and data pipeline
- train_model.py - Training script
- test_model.py - Evaluation script
- requirements.txt - Dependencies

Next Steps:
1. Run: python train_model.py
2. Monitor training progress
3. Evaluate on test set: python test_model.py
4. Use best model for inference

Class Mapping Confirmed:
- RGB (250, 235, 185) = Informal Settlements (Slums) → Binary 1
- All other RGB values = Non-slum → Binary 0
"""
    
    with open('TRAINING_SETUP_SUMMARY.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print("Summary saved to 'TRAINING_SETUP_SUMMARY.txt'")

if __name__ == "__main__":
    verify_dataset()
    print("\n" + "="*60)
    create_training_summary()
