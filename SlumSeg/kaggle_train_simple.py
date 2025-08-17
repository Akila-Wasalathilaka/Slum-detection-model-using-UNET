#!/usr/bin/env python3
"""Simple Kaggle Training Pipeline"""

import os
import subprocess
import yaml
from pathlib import Path

print("🚂 SlumSeg Training Pipeline")
print("=" * 30)

# Install dependencies
print("📦 Installing dependencies...")
subprocess.run("pip install -r requirements.txt --no-input --quiet", shell=True)

# Update config for current directory structure
config_path = "configs/default.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Set data path to current directory
config['data']['root'] = './data'
config['train']['batch_size'] = 8
config['train']['epochs'] = 15
config['train']['num_workers'] = 2

# Save updated config
kaggle_config = "configs/kaggle.yaml"
with open(kaggle_config, 'w') as f:
    yaml.dump(config, f)

print(f"✅ Config updated: {kaggle_config}")

# Check if data exists
data_path = Path("./data")
if data_path.exists():
    print(f"✅ Dataset found: {data_path.absolute()}")
    
    # Check structure
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if split_dir.exists():
            img_dir = split_dir / 'images'
            mask_dir = split_dir / 'masks'
            img_count = len(list(img_dir.glob('*.tif'))) if img_dir.exists() else 0
            mask_count = len(list(mask_dir.glob('*.tif'))) if mask_dir.exists() else 0
            print(f"📁 {split}: {img_count} images, {mask_count} masks")
else:
    print(f"❌ Dataset not found: {data_path.absolute()}")
    raise FileNotFoundError(f"Dataset not found: {data_path.absolute()}")

# Run training
print("\n🚂 Starting training...")
result = subprocess.run(f"python scripts/train.py --config {kaggle_config}", shell=True)

if result.returncode == 0:
    print("✅ Training completed!")
    
    # Package results
    import zipfile
    with zipfile.ZipFile('trained_model.zip', 'w') as z:
        ckpt_dir = Path('outputs/checkpoints')
        if ckpt_dir.exists():
            for f in ckpt_dir.glob('*.ckpt'):
                z.write(f, f'checkpoints/{f.name}')
        z.write(kaggle_config, 'config.yaml')
    
    print("🎉 Download trained_model.zip from Output panel")
else:
    print("❌ Training failed!")