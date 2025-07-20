#!/usr/bin/env python3
"""
Model Diagnostic Script
======================

Test the trained model on actual training data to verify it's working correctly
and diagnose the Colombo prediction issue.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import create_model
from config import get_model_config, get_data_config
from utils.checkpoint import load_checkpoint
from utils.transforms import get_test_transforms

def test_model_on_training_data():
    """Test model on some training samples to verify it works."""
    print("🔍 MODEL DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model_config = get_model_config('balanced')
    data_config = get_data_config('standard')
    
    checkpoint_path = project_root / "experiments/development_20250713_175410/checkpoints/best_checkpoint.pth"
    
    # Create and load model
    model = create_model(
        architecture=model_config.architecture,
        encoder=model_config.encoder,
        pretrained=False,
        num_classes=model_config.num_classes
    )
    
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    
    # Get transforms
    test_transforms = get_test_transforms(data_config)
    print("✅ Transforms loaded")
    
    # Test on training images
    train_images_dir = project_root / "data/train/images"
    train_masks_dir = project_root / "data/train/masks"
    
    # Get some training files
    image_files = list(train_images_dir.glob("*.tif"))[:5]  # Test 5 images
    
    print(f"\n🧪 Testing on {len(image_files)} training samples...")
    
    results = []
    
    for i, image_file in enumerate(image_files):
        print(f"\n--- Sample {i+1}: {image_file.name} ---")
        
        # Load image
        image = Image.open(image_file).convert('RGB')
        image_np = np.array(image)
        print(f"Image shape: {image_np.shape}")
        
        # Load corresponding mask
        mask_file = train_masks_dir / image_file.name
        if mask_file.exists():
            mask = Image.open(mask_file).convert('L')
            mask_np = np.array(mask)
            unique_values = np.unique(mask_np)
            print(f"Mask unique values: {unique_values}")
            
            # Check if mask contains slums
            has_slums = np.any(mask_np > 128)  # Assuming white pixels are slums
            print(f"Contains slums: {has_slums}")
        else:
            print("⚠️ No corresponding mask found")
            has_slums = False
            mask_np = None
        
        # Apply transforms
        if test_transforms:
            transformed = test_transforms(image=image_np)
            image_tensor = transformed['image']
        else:
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(device)
        print(f"Input tensor shape: {image_tensor.shape}")
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predictions = torch.sigmoid(outputs)
            
            print(f"Output shape: {outputs.shape}")
            print(f"Prediction shape: {predictions.shape}")
            
            # Get statistics
            pred_np = predictions.cpu().numpy().squeeze()
            avg_prob = pred_np.mean()
            max_prob = pred_np.max()
            min_prob = pred_np.min()
            
            print(f"Prediction stats: avg={avg_prob:.4f}, max={max_prob:.4f}, min={min_prob:.4f}")
            
            # Count high probability pixels
            high_prob_pixels = np.sum(pred_np > 0.5)
            total_pixels = pred_np.size
            high_prob_percentage = (high_prob_pixels / total_pixels) * 100
            
            print(f"High probability pixels (>0.5): {high_prob_pixels}/{total_pixels} ({high_prob_percentage:.2f}%)")
            
            results.append({
                'file': image_file.name,
                'has_slums_gt': has_slums,
                'avg_prediction': avg_prob,
                'max_prediction': max_prob,
                'high_prob_percentage': high_prob_percentage,
                'prediction_map': pred_np,
                'original_image': image_np,
                'ground_truth': mask_np
            })
    
    # Create visualization
    print(f"\n🎨 Creating diagnostic visualization...")
    
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['original_image'])
        axes[i, 0].set_title(f"Original\n{result['file']}")
        axes[i, 0].axis('off')
        
        # Ground truth
        if result['ground_truth'] is not None:
            axes[i, 1].imshow(result['ground_truth'], cmap='gray')
            axes[i, 1].set_title(f"Ground Truth\nSlums: {result['has_slums_gt']}")
        else:
            axes[i, 1].text(0.5, 0.5, 'No GT', ha='center', va='center')
            axes[i, 1].set_title("No Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction
        im = axes[i, 2].imshow(result['prediction_map'], cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction\nAvg: {result['avg_prediction']:.3f}")
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Binary prediction
        binary_pred = (result['prediction_map'] > 0.5).astype(np.uint8)
        axes[i, 3].imshow(binary_pred, cmap='gray')
        axes[i, 3].set_title(f"Binary (>0.5)\n{result['high_prob_percentage']:.1f}% slum")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save diagnostic plot
    diagnostic_path = Path("model_diagnostic_results.png")
    plt.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Diagnostic visualization saved: {diagnostic_path}")
    
    # Print summary
    print(f"\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    for result in results:
        print(f"📄 {result['file']}:")
        print(f"   GT has slums: {result['has_slums_gt']}")
        print(f"   Avg prediction: {result['avg_prediction']:.4f}")
        print(f"   Max prediction: {result['max_prediction']:.4f}")
        print(f"   Slum area detected: {result['high_prob_percentage']:.2f}%")
        print()
    
    # Test on Colombo tile for comparison
    print(f"\n🌏 TESTING ON COLOMBO TILE:")
    print("=" * 50)
    
    colombo_tiles_dir = Path("tiles")
    if colombo_tiles_dir.exists():
        colombo_files = list(colombo_tiles_dir.glob("*.png"))[:3]
        
        for tile_file in colombo_files:
            print(f"\n--- Colombo Tile: {tile_file.name} ---")
            
            # Load image
            image = Image.open(tile_file).convert('RGB')
            image_np = np.array(image)
            
            # Apply transforms
            if test_transforms:
                transformed = test_transforms(image=image_np)
                image_tensor = transformed['image']
            else:
                image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(image_tensor)
                predictions = torch.sigmoid(outputs)
                
                pred_np = predictions.cpu().numpy().squeeze()
                avg_prob = pred_np.mean()
                max_prob = pred_np.max()
                
                print(f"Colombo prediction: avg={avg_prob:.4f}, max={max_prob:.4f}")
    
    return results

if __name__ == "__main__":
    results = test_model_on_training_data()
