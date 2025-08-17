#!/usr/bin/env python3
"""
Kaggle Training Pipeline - Train Model Only
Copy this code into Kaggle notebook cells to train the slum segmentation model
"""

# =============================================================================
# CELL 1: Environment Setup and Clone Repository
# =============================================================================

import os
import subprocess
import sys

def run_cmd(cmd, description=""):
    """Run command and show output"""
    print(f"🔄 {description}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        raise Exception(f"Command failed: {cmd}")
    print(f"✅ {description} completed")
    if result.stdout:
        print(result.stdout)
    return result.stdout

# Check environment
print("🚂 SlumSeg: Kaggle Training Pipeline")
print("=" * 40)
print(f"Python: {sys.version}")
run_cmd("nvidia-smi", "Checking GPU")

# Clone repository (replace with your actual repo URL)
REPO_URL = "https://github.com/YOUR-USERNAME/SlumSeg.git"  # UPDATE THIS!
print(f"\n📥 Cloning repository from {REPO_URL}")

if not os.path.exists("/kaggle/working/SlumSeg"):
    run_cmd(f"cd /kaggle/working && git clone {REPO_URL}", "Cloning SlumSeg repository")
else:
    print("✅ Repository already exists")

# Change to project directory
os.chdir("/kaggle/working/SlumSeg")
print(f"📁 Working directory: {os.getcwd()}")

# =============================================================================
# CELL 2: Install Dependencies
# =============================================================================

print("\n📦 Installing dependencies...")

# Install requirements
run_cmd("pip install -r requirements.txt --no-input --quiet", "Installing Python packages")

# Verify key imports
try:
    import torch
    import segmentation_models_pytorch as smp
    import albumentations as A
    import rasterio
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✅ SMP: {smp.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    raise

# =============================================================================
# CELL 3: Training Configuration
# =============================================================================

import yaml
from pathlib import Path

print("\n🔧 Setting up training configuration...")

# Create optimized training config for Kaggle T4
training_config = {
    'seed': 42,
    'project_name': 'SlumSeg_Kaggle',
    
    'data': {
        'root': '/kaggle/input/YOUR-DATASET-NAME',  # UPDATE THIS!
        'images_dir': 'images',
        'masks_dir': 'masks',
        'tile_size': 512,
        'tile_overlap': 64,
        'val_ratio': 0.15,
        'min_slum_px': 256,
        'cache_dir': '/kaggle/working/cache'
    },
    
    'augment': {
        'train': {
            'hflip': 0.5,
            'vflip': 0.3,
            'rotate_deg': 15,
            'brightness': 0.2,
            'contrast': 0.2,
            'blur': 0.1,
            'cutout_prob': 0.3,
            'cutout_max_holes': 8,
            'cutout_max_h_size': 32,
            'cutout_max_w_size': 32
        },
        'valid': {}
    },
    
    'model': {
        'arch': 'unet',
        'encoder': 'resnet34',  # Good balance of speed/accuracy
        'pretrained': True,
        'in_channels': 3,
        'classes': 1
    },
    
    'loss': {
        'name': 'bce_dice',
        'pos_weight': None,  # Auto-calculated
        'alpha': 0.7,
        'gamma': 2.0
    },
    
    'train': {
        'epochs': 20,  # Reasonable for Kaggle
        'batch_size': 12,  # Optimized for T4
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'scheduler': 'onecycle',
        'amp': True,
        'channels_last': True,
        'grad_ckpt': True,
        'compile_mode': 'reduce-overhead',
        'num_workers': 2,  # Kaggle limit
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
        'accumulate_grad_batches': 1,
        'early_stopping_patience': 5,
        'save_top_k': 2
    },
    
    'eval': {
        'threshold': 0.5,
        'tta': False
    },
    
    'hardware': {
        'cudnn_benchmark': True,
        'tf32': True,
        'deterministic': False
    }
}

# Save training config
kaggle_config_path = "configs/kaggle_train.yaml"
os.makedirs("configs", exist_ok=True)
with open(kaggle_config_path, 'w') as f:
    yaml.dump(training_config, f, default_flow_style=False)

print(f"✅ Created training config: {kaggle_config_path}")
print(f"📊 Dataset: {training_config['data']['root']}")
print(f"🚂 Model: {training_config['model']['arch']} + {training_config['model']['encoder']}")
print(f"⚙️  Training: {training_config['train']['epochs']} epochs, batch size {training_config['train']['batch_size']}")

# Create output directories
output_dirs = ["outputs/checkpoints", "outputs/logs", "/kaggle/working/cache"]
for dir_path in output_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"📁 Created: {dir_path}")

# =============================================================================
# CELL 4: Quick Dataset Check
# =============================================================================

print("\n🔍 Quick Dataset Verification")
print("=" * 30)

dataset_path = Path(training_config['data']['root'])
if dataset_path.exists():
    print(f"✅ Dataset found: {dataset_path}")
    
    # Check structure
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            img_dir = split_path / 'images'
            mask_dir = split_path / 'masks'
            
            img_count = len(list(img_dir.glob('*.tif'))) if img_dir.exists() else 0
            mask_count = len(list(mask_dir.glob('*.tif'))) if mask_dir.exists() else 0
            
            print(f"📁 {split}: {img_count} images, {mask_count} masks")
        else:
            print(f"⚠️  {split}: not found")
else:
    print(f"❌ Dataset not found: {dataset_path}")
    print("🔧 Please update the dataset path in the config above")
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

# =============================================================================
# CELL 5: Data Preprocessing
# =============================================================================

print("\n🔄 Data Preprocessing")
print("=" * 25)

# Run data preprocessing
run_cmd(
    f"python scripts/prep_dataset.py --config {kaggle_config_path} --out /kaggle/working/tiles",
    "Creating tiles and computing class weights"
)

# Verify preprocessing
tiles_dir = Path("/kaggle/working/tiles")
if tiles_dir.exists():
    for split in ["train", "val"]:
        split_dir = tiles_dir / split
        if split_dir.exists():
            img_count = len(list((split_dir / "images").glob("*.tif"))) if (split_dir / "images").exists() else 0
            mask_count = len(list((split_dir / "masks").glob("*.tif"))) if (split_dir / "masks").exists() else 0
            print(f"✅ Preprocessed {split}: {img_count} images, {mask_count} masks")

# =============================================================================
# CELL 6: Model Training
# =============================================================================

print("\n🚂 Model Training")
print("=" * 20)

# Start training
print("🔥 Starting training... This will take 15-30 minutes on T4")
run_cmd(
    f"python scripts/train.py --config {kaggle_config_path} --tiles /kaggle/working/tiles",
    "Training slum segmentation model"
)

# Check training results
checkpoints_dir = Path("outputs/checkpoints")
if checkpoints_dir.exists():
    ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
    print(f"✅ Training completed! Generated {len(ckpt_files)} checkpoints:")
    for ckpt in sorted(ckpt_files):
        size_mb = ckpt.stat().st_size / (1024*1024)
        print(f"   💾 {ckpt.name} ({size_mb:.1f} MB)")

# =============================================================================
# CELL 7: Quick Model Test
# =============================================================================

print("\n🧪 Quick Model Test")
print("=" * 20)

# Test the trained model
try:
    import torch
    from slumseg.models.factory import make_model
    
    # Load best checkpoint
    best_ckpt = checkpoints_dir / "best.ckpt"
    if not best_ckpt.exists():
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            best_ckpt = ckpt_files[0]
    
    if best_ckpt.exists():
        print(f"🔍 Testing model: {best_ckpt.name}")
        
        # Create model
        model = make_model(
            arch=training_config['model']['arch'],
            encoder=training_config['model']['encoder'],
            classes=training_config['model']['classes'],
            pretrained=False
        )
        
        # Load weights
        checkpoint = torch.load(best_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        with torch.inference_mode():
            dummy_input = torch.randn(1, 3, 512, 512).to(device)
            output = model(dummy_input)
            
        print(f"✅ Model test successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Show training metrics if available
        if 'epoch' in checkpoint:
            print(f"   Trained epochs: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            print(f"   Best metric: {checkpoint['best_metric']:.4f}")
            
    else:
        print("❌ No checkpoint found for testing")
        
except Exception as e:
    print(f"⚠️  Model test failed: {e}")

# =============================================================================
# CELL 8: Package Training Results
# =============================================================================

print("\n📦 Packaging Training Results")
print("=" * 30)

import zipfile
from datetime import datetime

# Create training results package
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"slumseg_trained_model_{timestamp}.zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add checkpoints
    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob('*.ckpt'):
            zipf.write(ckpt_file, f"checkpoints/{ckpt_file.name}")
            print(f"📁 Added checkpoint: {ckpt_file.name}")
    
    # Add config
    zipf.write(kaggle_config_path, "training_config.yaml")
    
    # Add logs if they exist
    logs_dir = Path("outputs/logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob('*'):
            if log_file.is_file():
                zipf.write(log_file, f"logs/{log_file.name}")
    
    # Create training summary
    summary = f"""SlumSeg Training Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Kaggle Environment: T4 GPU

Training Configuration:
- Model: {training_config['model']['arch']} + {training_config['model']['encoder']}
- Epochs: {training_config['train']['epochs']}
- Batch Size: {training_config['train']['batch_size']}
- Learning Rate: {training_config['train']['lr']}
- Dataset: {training_config['data']['root']}

Results:
- Checkpoints: {len(list(checkpoints_dir.glob('*.ckpt')))} files
- Model ready for inference and evaluation

Usage:
1. Download this package
2. Use the checkpoints for inference with scripts/infer.py
3. Run evaluation with scripts/evaluate.py
4. Best checkpoint is typically 'best.ckpt'

Next Steps:
- Run the evaluation pipeline to generate charts and predictions
- Use the trained model for slum detection on new satellite images
"""
    
    zipf.writestr("README.txt", summary)

print(f"✅ Created training package: {zip_filename}")
print(f"📊 Package size: {os.path.getsize(zip_filename) / (1024*1024):.1f} MB")

# =============================================================================
# CELL 9: Training Complete Summary
# =============================================================================

print("\n🎉 Training Pipeline Complete!")
print("=" * 40)

# Final statistics
total_checkpoints = len(list(checkpoints_dir.glob("*.ckpt")))

print(f"🚂 Training Summary:")
print(f"   • Model: {training_config['model']['arch']} + {training_config['model']['encoder']}")
print(f"   • Epochs trained: {training_config['train']['epochs']}")
print(f"   • Checkpoints saved: {total_checkpoints}")
print(f"   • Training package: {zip_filename}")

print(f"\n📥 Download Instructions:")
print(f"   1. Download {zip_filename} from Kaggle Output panel")
print(f"   2. Extract to get trained model checkpoints")
print(f"   3. Use 'best.ckpt' for inference")

print(f"\n🔄 Next Steps:")
print(f"   1. Run the evaluation pipeline to generate charts")
print(f"   2. Use scripts/infer.py to create prediction overlays")
print(f"   3. Deploy the model for slum detection")

print(f"\n📁 Files in package:")
print(f"   • checkpoints/ - Trained model weights")
print(f"   • training_config.yaml - Training configuration")
print(f"   • logs/ - Training logs (if available)")
print(f"   • README.txt - Usage instructions")

print(f"\n✨ Your slum segmentation model is trained and ready! ✨")
print(f"🏠 Happy slum detection!")