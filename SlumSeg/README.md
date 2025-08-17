# SlumSeg: End-to-End Slum Detection Pipeline 🛰️

A production-ready slum segmentation pipeline designed for **Kaggle T4 GPU training** with **pure Python scripts** (no notebooks). Generates **20 prediction overlays** and **20 evaluation charts** for comprehensive analysis.

## 🚀 Quick Start (Kaggle T4)

```bash
# 1. Clone repo
cd /kaggle/working
git clone https://github.com/YOUR-ORG/SlumSeg.git
cd SlumSeg

# 2. Install dependencies
pip install -r requirements.txt --no-input

# 3. Dataset is already configured!
# Your data folder is already set in configs/default.yaml

# 4. Run complete pipeline
python scripts/analyze_dataset.py --config configs/default.yaml --out outputs/charts
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt --tiles . --charts outputs/charts
python scripts/infer.py --config configs/default.yaml --ckpt outputs/checkpoints/best.ckpt --images ../data/val/images --out outputs/predictions --num 20
```

## 🎯 Features

- **End-to-end pipeline**: Dataset analysis → Tiling → Training → Evaluation → Inference
- **Kaggle T4 optimized**: Mixed precision, channels-last, torch.compile, gradient checkpointing
- **Multiple architectures**: UNet, UNet++, DeepLabV3+ with various encoders
- **Smart augmentations**: Geo-realistic transforms for satellite imagery
- **Imbalance handling**: Weighted losses, hard example mining, oversampling
- **Comprehensive evaluation**: 20 evaluation charts + 20 prediction overlays
- **Region-based splits**: Avoid data leakage with geographic grouping

## 📁 Repository Structure

```
SlumSeg/
├── configs/                 # YAML configurations
├── slumseg/                # Core package
│   ├── data/               # Dataset, tiling, transforms
│   ├── models/             # Model factory, losses
│   └── utils/              # Metrics, visualization, optimization
├── scripts/                # Execution scripts
└── outputs/                # Results (charts, predictions, checkpoints)
```

## 🔧 Configuration

Edit `configs/default.yaml` to set your dataset path and hyperparameters:

```yaml
data:
  root: "e:/Slum-Detection-Using-Unet-Architecture/data"  # ✅ Already configured!
  tile_size: 512
  tile_overlap: 64
  
model:
  arch: "unet"
  encoder: "resnet34"
  
train:
  epochs: 25
  batch_size: 16
  lr: 3e-4
```

## 📊 Outputs

### 20 Evaluation Charts
1. Class pixel distribution
2. Slum coverage histogram
3. Per-region coverage boxplots
4. Train/val split statistics
5. Sample RGB + mask grid
6. Augmentation showcase
7. Learning rate schedule
8. Train/val loss curves
9. IoU progression
10. Dice score evolution
11. Precision-Recall curve
12. ROC curve
13. Confusion matrix
14. Threshold sensitivity
15. Calibration plot
16. Inference latency
17. Throughput analysis
18. Memory usage
19. Per-region performance
20. Error analysis mosaics

### 20 Prediction Overlays
Semi-transparent red overlays on RGB tiles showing detected slum areas.

## 🏗️ Technical Stack

- **PyTorch 2.x** with CUDA optimization
- **segmentation-models-pytorch** for architectures
- **Albumentations** for augmentations
- **Rasterio/GeoPandas** for geospatial data
- **Hydra** for configuration management

## 📈 Performance Optimizations

- Mixed precision training (AMP)
- Channels-last memory format
- PyTorch 2.0 compilation
- Gradient checkpointing
- Persistent data workers
- Memory pinning
- Optimized batch sizes for T4

## 🔄 Training Modes

### Single Model
```bash
python scripts/train.py --config configs/default.yaml
```

### 20-Model Ensemble (5-fold × 4 encoders)
```bash
python scripts/kfold_train.py --config configs/default.yaml --folds configs/folds.yaml
```

## 📏 Metrics

- **Pixel-wise**: IoU, Dice, Precision, Recall, AUROC, AUPRC
- **Regional**: Per-city/area performance
- **Qualitative**: Top FN/FP examples

## 🎨 Visualization

Red overlay generation with configurable transparency:
```python
overlay = 0.6*rgb + 0.4*red  # Only where mask=1
```

## ⚡ Speed Benchmarks

Optimized for T4 GPU with:
- Batch size 16-32 @ 512×512
- ~100ms inference per tile
- Memory efficient with gradient checkpointing

## 📦 Export Options

- PyTorch checkpoints
- ONNX models
- TorchScript for deployment

---

**Built for production slum detection from satellite imagery** 🌍
