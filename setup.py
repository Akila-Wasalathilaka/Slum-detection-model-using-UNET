#!/usr/bin/env python3
"""
Setup script for Production Slum Detection Pipeline
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {command}")
        print(f"Exception: {e}")
        return False

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    # Check if pip is available
    if not run_command("pip --version"):
        print("pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install requirements.")
        return False
    
    print("Requirements installed successfully!")
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    base_dir = Path.cwd()
    directories = [
        base_dir / "data_preprocessed" / "train" / "images",
        base_dir / "data_preprocessed" / "train" / "masks",
        base_dir / "data_preprocessed" / "val" / "images",
        base_dir / "data_preprocessed" / "val" / "masks",
        base_dir / "data_preprocessed" / "test" / "images",
        base_dir / "data_preprocessed" / "test" / "masks",
        base_dir / "results_production",
        base_dir / "models_production",
        base_dir / "models_production" / "checkpoints"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Directories created successfully!")
    return True

def check_cuda():
    """Check CUDA availability."""
    print("Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available!")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Training will use CPU.")
    except ImportError:
        print("PyTorch is not installed. Cannot check CUDA.")
    
    return True

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    required_modules = [
        'torch',
        'torchvision',
        'numpy',
        'cv2',
        'albumentations',
        'segmentation_models_pytorch',
        'tqdm'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("All imports successful!")
    return True

def main():
    """Main setup function."""
    print("=== Production Slum Detection Pipeline Setup ===")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required.")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    print("\n=== Setup Complete ===")
    print("You can now run the pipeline with:")
    print("  python main.py --mode train      # For training")
    print("  python main.py --mode infer      # For inference")
    print("  python main.py --mode both       # For both")
    print("\nOr use the combined version:")
    print("  python combined_pipeline.py --mode train")
    
    print("\nMake sure to place your data in the data_preprocessed directory:")
    print("  data_preprocessed/train/images/   # Training images (.tif)")
    print("  data_preprocessed/train/masks/    # Training masks (.png)")
    print("  data_preprocessed/val/images/     # Validation images")
    print("  data_preprocessed/val/masks/      # Validation masks")
    print("  data_preprocessed/test/images/    # Test images")
    print("  data_preprocessed/test/masks/     # Test masks")

if __name__ == "__main__":
    main()
