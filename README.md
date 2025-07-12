# 🏘️ Slum Detection Model

**Advanced Deep Learning Pipeline for Satellite Image Slum Detection**

A comprehensive PyTorch-based solution for detecting informal settlements (slums) from 120×120 RGB satellite image tiles using state-of-the-art semantic segmentation models.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Quick development training
python scripts/train.py --model balanced --training development --data standard

# Production training with high accuracy
python scripts/train.py --model accurate --training production --data heavy_augmentation
```

### 3. Monitor Training
Check `experiments/` directory for:
- Training logs and metrics
- Model checkpoints
- Visualization plots
- Configuration files

## 📁 Project Structure

```
slum-detection-model/
├── 📊 data/                    # Dataset (120x120 RGB tiles)
│   ├── train/images/          # Training satellite images
│   ├── train/masks/           # Training segmentation masks  
│   ├── val/images/            # Validation images
│   ├── val/masks/             # Validation masks
│   ├── test/images/           # Test images
│   └── test/masks/            # Test masks
│
├── 🏗️ models/                  # Model architectures
│   ├── __init__.py           # Model package
│   ├── unet.py               # UNet variants (ResNet, EfficientNet)
│   ├── losses.py             # Loss functions (Dice, Focal, Combined)
│   └── metrics.py            # Evaluation metrics (IoU, F1, etc.)
│
├── ⚙️ config/                  # Configuration management
│   ├── __init__.py           # Config package
│   ├── model_config.py       # Model hyperparameters
│   ├── training_config.py    # Training settings
│   └── data_config.py        # Data preprocessing config
│
├── 🛠️ utils/                   # Utilities and helpers
│   ├── __init__.py           # Utils package
│   ├── dataset.py            # Dataset class with filtering
│   ├── transforms.py         # Data augmentation pipeline
│   ├── visualization.py      # Training/result visualization
│   └── checkpoint.py         # Model checkpoint management
│
├── 🎯 scripts/                 # Main execution scripts
│   ├── train.py              # Training script with experiment management
│   ├── test.py               # Model evaluation and testing
│   ├── inference.py          # Single image prediction
│   └── export_model.py       # Model export (ONNX, TorchScript)
│
├── 🧪 experiments/             # Training experiments
│   ├── logs/                 # Training logs
│   ├── checkpoints/          # Model weights
│   ├── results/              # Test results and plots
│   └── configs/              # Experiment configurations
│
├── 📈 analysis/               # Dataset analysis scripts
│   ├── comprehensive_dataset_analysis.py
│   ├── FINAL_DATASET_ANALYSIS_REPORT.txt
│   └── [various analysis scripts...]
│
├── 📊 charts/                 # Model analysis and visualization
│   ├── model_analysis.py     # Comprehensive model analysis
│   ├── quick_analysis.py     # Fast post-training evaluation
│   ├── post_training_analysis.py # Automated analysis pipeline
│   ├── example_analysis.py   # Usage examples
│   └── README.md             # Analysis documentation
│
└── 📋 requirements.txt        # Python dependencies
```

## 🎯 Key Features

### 🏗️ **Advanced Model Architectures**
- **UNet**: Standard U-Net with multiple encoder options
- **UNet++**: Nested U-Net for improved feature representation  
- **DeepLabV3+**: Atrous convolutions for multi-scale context
- **Encoders**: ResNet, EfficientNet, MobileNet, DenseNet

### 🔥 **Sophisticated Loss Functions**
- **Combined Loss**: BCE + Dice + Focal for optimal training
- **Focal Loss**: Handles class imbalance (slum vs non-slum)
- **Tversky Loss**: Precision/recall balance control
- **Dice Loss**: Overlap optimization

### 📊 **Comprehensive Metrics**
- **IoU (Jaccard)**: Primary segmentation metric
- **Dice Score**: Overlap measurement
- **Precision/Recall**: Class-specific performance
- **F1 Score**: Balanced performance measure

### 🔄 **Advanced Data Augmentation**
- **Geometric**: Rotation, flipping, scaling, elastic transforms
- **Color**: Brightness, contrast, saturation adjustments  
- **Noise**: Gaussian noise, blur for robustness
- **Advanced**: Grid distortion, cutout, mixup

### ⚡ **Training Optimizations**
- **Mixed Precision**: Faster training with AMP
- **Learning Rate Scheduling**: Cosine annealing, plateau reduction
- **Early Stopping**: Prevent overfitting
- **Gradient Clipping**: Training stability

### 📊 **Model Analysis & Visualization**
- **Automatic Analysis**: Post-training evaluation with confusion matrices, ROC curves
- **Comprehensive Charts**: Threshold analysis, performance metrics, prediction samples
- **Multiple Formats**: Quick analysis (4 charts) or comprehensive (15+ charts)
- **Performance Metrics**: AUC-ROC, Precision-Recall, F1-Score optimization
- **Visual Debugging**: Sample predictions with ground truth comparison

## 🎛️ Configuration Presets

### Model Configurations
```python
# Fast inference
python scripts/train.py --model fast

# Balanced accuracy/speed  
python scripts/train.py --model balanced

# Highest accuracy
python scripts/train.py --model accurate

# Lightweight deployment
python scripts/train.py --model lightweight
```

### Training Configurations
```python
# Quick testing (10 epochs)
python scripts/train.py --training quick_test

# Development (50 epochs)
python scripts/train.py --training development

# Production training (150 epochs)
python scripts/train.py --training production

# High precision focus
python scripts/train.py --training high_precision

# High recall focus  
python scripts/train.py --training high_recall
```

### Data Configurations
```python
# Minimal augmentation
python scripts/train.py --data minimal

# Standard augmentation
python scripts/train.py --data standard

# Heavy augmentation
python scripts/train.py --data heavy_augmentation

# Production with TTA
python scripts/train.py --data production
```

## 📋 Dataset Requirements

### 🎯 **Class Mapping**
- **Slum Class**: RGB (250, 235, 185) → Binary 1
- **Non-Slum**: All other RGB values → Binary 0

### 📊 **Dataset Statistics** 
- **Total Images**: 8,910 masks analyzed
- **Slum Coverage**: 1,657 masks contain slums (18.6%)
- **Image Size**: 120×120 RGB tiles
- **Mask Format**: PNG with RGB encoding

### 🎨 **Data Quality**
- ✅ **Excellent Coverage**: Sufficient slum examples for training
- ✅ **Balanced Distribution**: 0-100% slum coverage range  
- ✅ **Clean Encoding**: Consistent RGB class mapping
- ✅ **Ready for Training**: No preprocessing required

## 🚀 Training Examples

### Basic Training
```bash
# Train with default settings
python scripts/train.py
```

### Custom Training
```bash
# High-accuracy model with heavy augmentation
python scripts/train.py \
  --model accurate \
  --training production \
  --data heavy_augmentation \
  --experiment "high_accuracy_v1"
```

### Resume Training
```bash
# The system automatically handles checkpointing
# Just rerun the same command to resume from last checkpoint
python scripts/train.py --experiment "my_experiment"
```

## 📈 Monitoring Training

### Real-time Monitoring
- **Console Logs**: Batch-level progress and metrics
- **Plots**: Automatically generated training curves
- **Checkpoints**: Best models saved automatically

### 4. Analyze Results
```bash
# Automatic analysis after training (built-in)
# Or run manual analysis:

# Quick analysis
python charts/post_training_analysis.py --auto-find

# Comprehensive analysis with all charts
python charts/post_training_analysis.py --checkpoint experiments/*/checkpoints/best_model.pth --analysis-type comprehensive

# Quick example
python charts/example_analysis.py
```

### Result Analysis
```bash
# Check experiment results
ls experiments/[experiment_name]/

# View generated charts
ls charts/analysis_*/
```

## 🏆 Expected Performance

Based on dataset analysis, expect:
- **IoU Score**: 0.75-0.85 for well-trained models
- **Dice Score**: 0.80-0.90 for optimal overlap
- **F1 Score**: 0.80-0.90 for balanced performance
- **Training Time**: 2-4 hours on modern GPU

## 🛠️ Customization

### Adding New Models
1. Implement in `models/unet.py`
2. Add configuration in `config/model_config.py`
3. Update factory function

### Custom Loss Functions  
1. Implement in `models/losses.py`
2. Add to `create_loss()` factory
3. Configure in training config

### New Augmentations
1. Add to `utils/transforms.py`
2. Configure in `config/data_config.py`
3. Test with visualization utilities

## 📝 License

This project is designed for satellite image analysis and slum detection research. Please ensure compliance with satellite imagery usage terms and local regulations.

---

**🎯 Ready to detect slums with state-of-the-art deep learning!**

For questions or issues, check the experiment logs or configuration files for detailed settings and results.
