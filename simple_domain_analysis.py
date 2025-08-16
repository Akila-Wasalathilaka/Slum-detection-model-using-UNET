"""
Simple analysis to identify why the model fails on Colombo data.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.transforms import get_test_transforms
from utils.checkpoint import load_checkpoint

def analyze_simple_comparison():
    """Simple comparison between training and Colombo data."""
    
    print("🔍 SIMPLE DOMAIN ANALYSIS")
    print("=" * 40)
    
    # Load some training images
    train_dir = Path("data/train/images")
    train_files = list(train_dir.glob("*.tif"))[:5]
    
    # Load some Colombo tiles
    colombo_dir = Path("colombo/tiles")
    colombo_files = list(colombo_dir.glob("*.png"))[:5] if colombo_dir.exists() else []
    
    if not train_files:
        print("❌ No training files found")
        return
        
    if not colombo_files:
        print("❌ No Colombo files found")
        return
    
    # Collect pixel statistics
    train_stats = []
    colombo_stats = []
    
    print("📊 Training images:")
    for f in train_files:
        img = np.array(Image.open(f).convert('RGB'))
        stats = {
            'mean': img.mean(),
            'std': img.std(),
            'min': img.min(),
            'max': img.max(),
            'shape': img.shape
        }
        train_stats.append(stats)
        print(f"  {f.name}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")
    
    print("\n📊 Colombo tiles:")
    for f in colombo_files:
        img = np.array(Image.open(f).convert('RGB'))
        stats = {
            'mean': img.mean(),
            'std': img.std(),
            'min': img.min(),
            'max': img.max(),
            'shape': img.shape
        }
        colombo_stats.append(stats)
        print(f"  {f.name}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")
    
    # Calculate averages
    train_mean = np.mean([s['mean'] for s in train_stats])
    train_std_avg = np.mean([s['std'] for s in train_stats])
    colombo_mean = np.mean([s['mean'] for s in colombo_stats])
    colombo_std_avg = np.mean([s['std'] for s in colombo_stats])
    
    print(f"\n📈 SUMMARY:")
    print(f"Training avg: mean={train_mean:.1f}, std={train_std_avg:.1f}")
    print(f"Colombo avg: mean={colombo_mean:.1f}, std={colombo_std_avg:.1f}")
    print(f"Mean difference: {abs(train_mean - colombo_mean):.1f}")
    print(f"Std difference: {abs(train_std_avg - colombo_std_avg):.1f}")
    
    return train_mean, colombo_mean

def test_simple_predictions():
    """Test model predictions on training vs Colombo images."""
    
    print("\n🤖 TESTING MODEL PREDICTIONS")
    print("=" * 40)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, _, _, _, _ = load_checkpoint('models/best_model.pth', device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get transforms
    transforms = get_test_transforms()
    
    def predict_image(img_path):
        """Get prediction for a single image."""
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # Apply transforms
        transformed = transforms(image=img_array)['image']
        img_tensor = transformed.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_tensor)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
        
        return pred_prob, img_array
    
    # Test on training images
    train_dir = Path("data/train/images")
    train_files = list(train_dir.glob("*.tif"))[:3]
    
    print("🎯 Training image predictions:")
    train_preds = []
    for f in train_files:
        pred, img = predict_image(f)
        train_preds.append(pred)
        print(f"  {f.name}: max={pred.max():.4f}, mean={pred.mean():.4f}, slum_pixels={np.sum(pred > 0.5)}")
    
    # Test on Colombo tiles
    colombo_dir = Path("colombo/tiles")
    colombo_files = list(colombo_dir.glob("*.png"))[:3] if colombo_dir.exists() else []
    
    if colombo_files:
        print("\n🏙️ Colombo tile predictions:")
        colombo_preds = []
        for f in colombo_files:
            pred, img = predict_image(f)
            colombo_preds.append(pred)
            print(f"  {f.name}: max={pred.max():.4f}, mean={pred.mean():.4f}, slum_pixels={np.sum(pred > 0.5)}")
        
        # Compare averages
        train_max_avg = np.mean([p.max() for p in train_preds])
        colombo_max_avg = np.mean([p.max() for p in colombo_preds])
        
        print(f"\n📊 PREDICTION COMPARISON:")
        print(f"Training avg max: {train_max_avg:.4f}")
        print(f"Colombo avg max: {colombo_max_avg:.4f}")
        print(f"Ratio (Colombo/Training): {colombo_max_avg/train_max_avg:.4f}")
        
        if colombo_max_avg < 0.1 and train_max_avg > 0.5:
            print("\n🚨 DIAGNOSIS: SEVERE DOMAIN SHIFT DETECTED")
            print("   - Model works on training data but fails on Colombo")
            print("   - This suggests the model learned very specific features")
            print("   - Colombo tiles look too different from training data")
        
        return True
    else:
        print("❌ No Colombo tiles found for comparison")
        return False

def create_side_by_side_visualization():
    """Create a visual comparison of training vs Colombo images."""
    
    print("\n🖼️ CREATING VISUAL COMPARISON")
    print("=" * 40)
    
    # Load examples
    train_dir = Path("data/train/images")
    train_files = list(train_dir.glob("*.tif"))[:2]
    
    colombo_dir = Path("colombo/tiles")
    colombo_files = list(colombo_dir.glob("*.png"))[:2] if colombo_dir.exists() else []
    
    if not train_files or not colombo_files:
        print("❌ Missing files for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Training images
    for i, f in enumerate(train_files):
        img = Image.open(f).convert('RGB')
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Training: {f.name}')
        axes[0, i].axis('off')
    
    # Colombo images
    for i, f in enumerate(colombo_files):
        img = Image.open(f).convert('RGB')
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Colombo: {f.name}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_vs_colombo_visual.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Saved: training_vs_colombo_visual.png")

if __name__ == "__main__":
    analyze_simple_comparison()
    test_simple_predictions()
    create_side_by_side_visualization()
    
    print("\n💡 NEXT STEPS:")
    print("1. Check training_vs_colombo_visual.png to see visual differences")
    print("2. If severe domain shift detected, consider:")
    print("   - Histogram matching/color normalization")
    print("   - Retraining with more diverse data")
    print("   - Domain adaptation techniques")
    print("   - Fine-tuning on Colombo-like data")
