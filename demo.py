"""
Quick demo script to test the ultra-accurate slum detection pipeline.
"""
import sys
import os
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported."""
    try:
        print("🔍 Testing imports...")
        
        # Test core config
        from config import config
        print("✅ Config imported successfully")
        
        # Test model
        from model import UltraAccurateUNet
        print("✅ UltraAccurateUNet imported successfully")
        
        # Test losses
        from losses import UltraSlumDetectionLoss
        print("✅ UltraSlumDetectionLoss imported successfully")
        
        # Test trainer
        from trainer import UltraAccurateTrainer
        print("✅ UltraAccurateTrainer imported successfully")
        
        # Test inference
        from inference import UltraAccurateInference
        print("✅ UltraAccurateInference imported successfully")
        
        # Test utils
        from utils import setup_ultra_logging
        print("✅ Utils imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test creating the ultra-accurate model."""
    try:
        print("\n🏗️ Testing model creation...")
        
        import torch
        from model import UltraAccurateUNet
        
        # Create model
        model = UltraAccurateUNet()
        print("✅ Model created successfully")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 384, 384).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {str(e)}")
        traceback.print_exc()
        return False

def test_loss_function():
    """Test the ultra-accurate loss function."""
    try:
        print("\n🎯 Testing loss function...")
        
        import torch
        from losses import UltraSlumDetectionLoss
        
        # Create loss function
        loss_fn = UltraSlumDetectionLoss()
        print("✅ Loss function created successfully")
        
        # Create dummy tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred = torch.randn(2, 1, 384, 384).to(device)
        target = torch.randint(0, 2, (2, 1, 384, 384)).float().to(device)
        
        # Calculate loss
        loss = loss_fn(pred, target)
        
        print(f"✅ Loss calculation successful! Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function error: {str(e)}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration settings."""
    try:
        print("\n⚙️ Testing configuration...")
        
        from config import config
        
        print(f"✅ Primary size: {config.PRIMARY_SIZE}")
        print(f"✅ Batch size: {config.BATCH_SIZE}")
        print(f"✅ Epochs: {config.NUM_EPOCHS}")
        print(f"✅ Learning rate: {config.LEARNING_RATE}")
        print(f"✅ Model save directory: {config.MODEL_SAVE_DIR}")
        print(f"✅ Results directory: {config.RESULTS_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        traceback.print_exc()
        return False

def test_data_paths():
    """Test if data paths exist."""
    try:
        print("\n📁 Testing data paths...")
        
        from config import config
        
        # Check if data directories exist
        if os.path.exists(config.DATA_ROOT):
            print(f"✅ Data root exists: {config.DATA_ROOT}")
        else:
            print(f"⚠️ Data root not found: {config.DATA_ROOT}")
        
        # Check test images
        test_img_dir = os.path.join(config.DATA_ROOT, 'test', 'images')
        if os.path.exists(test_img_dir):
            test_files = list(Path(test_img_dir).glob('*.tif'))
            print(f"✅ Test images directory exists with {len(test_files)} images")
        else:
            print(f"⚠️ Test images directory not found: {test_img_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data paths error: {str(e)}")
        traceback.print_exc()
        return False

def create_demo_directories():
    """Create demo directories."""
    try:
        print("\n📂 Creating demo directories...")
        
        from config import config
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, 'maps'), exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, 'visualizations'), exist_ok=True)
        
        print("✅ Demo directories created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Directory creation error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print("🚀 Ultra-Accurate Slum Detection Demo")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Data Paths Test", test_data_paths),
        ("Directory Creation Test", create_demo_directories),
        ("Model Creation Test", test_model_creation),
        ("Loss Function Test", test_loss_function),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ultra-accurate pipeline is ready!")
        print("\nNext steps:")
        print("1. Run training: python main.py --mode train")
        print("2. Run evaluation: python main.py --mode evaluate")
        print("3. Create demo maps: python main.py --mode demo")
        print("4. Run inference: python main.py --mode inference --image_path <path>")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
