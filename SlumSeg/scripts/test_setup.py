#!/usr/bin/env python3
"""
Quick test script to verify SlumSeg setup.
"""

import sys
import os
from pathlib import Path

# Set random seeds for reproducibility
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import segmentation_models_pytorch as smp
        print(f"✅ SMP: {smp.__version__}")
    except ImportError:
        print("❌ segmentation-models-pytorch not found")
        return False
    
    try:
        import albumentations as A
        print(f"✅ Albumentations: {A.__version__}")
    except ImportError:
        print("❌ albumentations not found")
        return False
    
    try:
        import rasterio
        print(f"✅ Rasterio: {rasterio.__version__}")
    except ImportError:
        print("❌ rasterio not found")
        return False
    
    return True


def test_data_structure():
    """Test if data structure is correct."""
    print("\n📁 Testing data structure...")
    
    data_root = Path("../data")
    if not data_root.exists():
        print(f"❌ Data directory not found: {data_root.absolute()}")
        return False
    
    # Check splits
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = data_root / split
        if split_dir.exists():
            images_dir = split_dir / 'images'
            masks_dir = split_dir / 'masks'
            
            if images_dir.exists():
                image_count = len(list(images_dir.glob("*.tif")))
                print(f"✅ {split}/images: {image_count} files")
            else:
                print(f"⚠️  {split}/images: directory not found")
            
            if masks_dir.exists():
                mask_count = len(list(masks_dir.glob("*.tif")))
                print(f"✅ {split}/masks: {mask_count} files")
            else:
                print(f"⚠️  {split}/masks: directory not found")
        else:
            print(f"⚠️  {split}: directory not found")
    
    return True


def test_config():
    """Test if configuration is valid."""
    print("\n⚙️  Testing configuration...")
    
    try:
        import yaml
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        data_root = config['data']['root']
        if Path(data_root).exists():
            print(f"✅ Config data path exists: {data_root}")
        else:
            print(f"❌ Config data path not found: {data_root}")
            return False
        
        print(f"✅ Model: {config['model']['arch']} + {config['model']['encoder']}")
        print(f"✅ Batch size: {config['train']['batch_size']}")
        print(f"✅ Learning rate: {config['train']['lr']}")
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    return True


def test_model_creation():
    """Test if model can be created."""
    print("\n🤖 Testing model creation...")
    
    try:
        from slumseg.models.factory import make_model
        
        model = make_model(
            arch="unet",
            encoder="resnet34",
            classes=1,
            in_channels=3,
            pretrained=False  # Skip pretrained for speed
        )
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            output = model(dummy_input)
        
        print(f"✅ Model created successfully")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    return True


def main():
    print("🔬 SlumSeg Setup Test")
    print("====================")
    
    all_good = True
    
    all_good &= test_imports()
    all_good &= test_data_structure()
    all_good &= test_config()
    all_good &= test_model_creation()
    
    print("\n" + "="*40)
    if all_good:
        print("🎉 All tests passed! SlumSeg is ready to go!")
        print("\n📝 Next steps:")
        print("   1. Run: python scripts/analyze_dataset.py --config configs/default.yaml --out outputs/charts")
        print("   2. Run: python scripts/train.py --config configs/default.yaml")
        print("   3. Run: python scripts/evaluate.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt --tiles . --charts outputs/charts")
        print("   4. Run: python scripts/infer.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt --images ../data/val/images --out outputs/predictions --num 20")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 To fix:")
        print("   1. Install requirements: pip install -r requirements.txt")
        print("   2. Check data directory structure")
        print("   3. Verify configuration in configs/default.yaml")


if __name__ == "__main__":
    main()
