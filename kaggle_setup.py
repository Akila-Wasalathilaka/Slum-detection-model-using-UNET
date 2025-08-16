#!/usr/bin/env python3
"""
Kaggle Setup Script for Slum Detection
=====================================
Run this first in Kaggle to clone repo and install dependencies
"""

import subprocess
import sys
import os

def run_command(command):
    """Run shell command and print output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def main():
    print("🚀 KAGGLE SLUM DETECTION SETUP")
    print("=" * 50)
    
    # Clone repository
    print("📥 Cloning repository...")
    if not run_command("git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git"):
        print("❌ Failed to clone repository")
        return
    
    # Change to project directory
    os.chdir("Slum-detection-model-using-UNET")
    
    # Install additional packages not in Kaggle by default
    print("📦 Installing additional packages...")
    packages = [
        "albumentations",
        "segmentation-models-pytorch", 
        "timm",
        "efficientnet-pytorch"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        run_command(f"pip install {package}")
    
    # Verify data structure
    print("📊 Verifying data structure...")
    if os.path.exists("data"):
        for split in ["train", "val", "test"]:
            img_path = f"data/{split}/images"
            mask_path = f"data/{split}/masks"
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img_count = len([f for f in os.listdir(img_path) if f.endswith('.tif')])
                mask_count = len([f for f in os.listdir(mask_path) if f.endswith('.png')])
                print(f"✅ {split.upper()}: {img_count} images, {mask_count} masks")
            else:
                print(f"❌ {split.upper()}: Missing directories")
    else:
        print("❌ Data directory not found!")
    
    print("\n✅ Setup complete! You can now run:")
    print("1. python comprehensive_analysis.py  # Analyze dataset")
    print("2. python advanced_training.py       # Train model")
    print("3. python create_charts.py          # Generate charts")
    print("4. python make_predictions.py       # Generate predictions")

if __name__ == "__main__":
    main()