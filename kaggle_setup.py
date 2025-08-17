#!/usr/bin/env python3
"""
Kaggle Environment Setup
=======================
Setup script for Kaggle environment
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages for Kaggle"""
    packages = [
        'segmentation-models-pytorch',
        'albumentations',
        'dataclasses-json'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def setup_kaggle_paths():
    """Setup paths for Kaggle environment"""
    # Create necessary directories
    os.makedirs('/kaggle/working/experiments', exist_ok=True)
    os.makedirs('/kaggle/working/data', exist_ok=True)
    
    print("✅ Kaggle directories created")

def main():
    """Main setup function"""
    print("🔧 Setting up Kaggle environment...")
    
    install_requirements()
    setup_kaggle_paths()
    
    print("\n✅ Kaggle setup complete!")
    print("\nNext steps:")
    print("1. Upload your dataset to /kaggle/input/")
    print("2. Run: python kaggle_train_pipeline.py")
    print("3. Run: python kaggle_analysis_pipeline.py")

if __name__ == "__main__":
    main()