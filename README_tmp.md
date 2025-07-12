# Production-Ready Slum Detection Pipeline

## Overview
This repository provides a state-of-the-art, production-level pipeline for mapping informal settlements (slums) in satellite imagery using deep learning. It features advanced architectures, robust data handling, and comprehensive evaluation/visualization tools.

## Quick Start

### 1. Installation
```bash
# Clone the repository
cd slum_detection
pip install -r requirements.txt
```

### 2. Data Preparation
Organize your data as follows:
```
data_preprocessed/
├── train/
│   ├── images/  # .tif images
│   └── masks/   # .png masks
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 3. Training
```bash
python run_production_pipeline.py --mode train
```

### 4. Inference
```bash
python run_production_pipeline.py --mode inference
```

### 5. Evaluation & Visualization
```bash
python run_production_pipeline.py --mode eval
# All results and visualizations will be saved in results_production/ or results_binary/
```

## Features
- Advanced architectures: UNet, UNet++, DeepLabV3+, FPN, SegFormer
- Robust loss functions: BCE, Dice, Focal, Tversky, OHEM, Asymmetric Focal Tversky
- Test-time augmentation (TTA) and model ensembling
- Adaptive post-processing (morphology, CRF)
- Oversampling, CutMix, MixUp, multi-scale training
- Comprehensive visualizations: confusion matrix, ROC/PR, error maps, per-image metrics, threshold curves, etc.

## Visualizations
- Confusion Matrix
- Classification Report Table
- ROC Curve (AUC-ROC)
- Precision-Recall (PR) Curve
- IoU/Dice Score Plots
- Pixel Accuracy Histogram
- Error Maps
- Probability Maps
- Per-Image Metrics Table
- Training History Curves
- Model Comparison Bar Charts
- Threshold Optimization Curve
- Post-Processing Effect Plots
- Uncertainty Maps

## Advanced Usage
- See `run_production_pipeline.py` for all CLI options (e.g., skipping preprocessing, custom thresholds, etc.)
- For custom evaluation or batch inference, see `improved_slum_detection_production.py`.

## Clean Codebase
- Only the main scripts and essential modules are kept.
- All legacy/old/duplicate files and markdowns have been removed for clarity.

## Support
- For issues, open a GitHub issue or discussion.

---
**Ready for production. Making a difference with AI-powered urban mapping!**
