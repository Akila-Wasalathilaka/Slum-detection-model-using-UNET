#!/usr/bin/env python3
"""Simple Kaggle Analysis Pipeline"""

import os
import subprocess
import yaml
from pathlib import Path

print("📊 SlumSeg Analysis Pipeline")
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
else:
    print(f"❌ Dataset not found: {data_path.absolute()}")

# Run dataset analysis
print("\n📊 Running dataset analysis...")
subprocess.run(f"python scripts/analyze_dataset.py --config {kaggle_config} --out outputs/charts", shell=True)

# Check for trained model
model_paths = [
    "/kaggle/input/trained-model/best.ckpt",
    "/kaggle/input/model/best.ckpt", 
    "outputs/checkpoints/best.ckpt"
]

model_path = None
for path in model_paths:
    if Path(path).exists():
        model_path = path
        break

if model_path:
    print(f"🤖 Model found: {model_path}")
    
    # Run evaluation
    print("📈 Running evaluation...")
    subprocess.run(f"python scripts/prep_dataset.py --config {kaggle_config} --out outputs/tiles", shell=True)
    subprocess.run(f"python scripts/evaluate.py --config {kaggle_config} --ckpt {model_path} --tiles outputs/tiles --charts outputs/charts", shell=True)
    
    # Generate predictions
    print("🔮 Generating predictions...")
    val_images = "outputs/tiles/val/images"
    if not Path(val_images).exists():
        val_images = "data/val/images"
    subprocess.run(f"python scripts/infer.py --config {kaggle_config} --ckpt {model_path} --images {val_images} --out outputs/predictions --num 20", shell=True)
else:
    print("⚠️ No trained model found - only dataset analysis generated")

# Package results
print("\n📦 Packaging results...")
import zipfile
with zipfile.ZipFile('analysis_results.zip', 'w') as z:
    # Add charts
    charts_dir = Path('outputs/charts')
    if charts_dir.exists():
        for f in charts_dir.glob('*.png'):
            z.write(f, f'charts/{f.name}')
    
    # Add predictions
    pred_dir = Path('outputs/predictions')
    if pred_dir.exists():
        for f in pred_dir.glob('*.png'):
            z.write(f, f'predictions/{f.name}')
    
    z.write(kaggle_config, 'config.yaml')

print("🎉 Download analysis_results.zip from Output panel")