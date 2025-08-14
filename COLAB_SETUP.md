# 🚀 Google Colab Setup Guide

## Quick Start (5 minutes)

### 1. Open Google Colab
- Go to [colab.research.google.com](https://colab.research.google.com)
- Create new notebook or upload `colab_setup.ipynb`
- Enable GPU: Runtime → Change runtime type → GPU → T4

### 2. Upload Your Dataset to Google Drive
- Upload your `data/` folder to Google Drive
- Path: `/MyDrive/data/`
- Structure: `train/`, `val/`, `test/` folders with `images/` and `masks/`

### 3. Run Training
```python
# Quick test (5 epochs, ~30 minutes)
!python scripts/train.py --model balanced --training development

# Full training (50 epochs, ~4 hours)
!python scripts/train.py --model balanced --training production
```

### 4. Access Results
Results automatically saved to Google Drive: `/MyDrive/slum_detection_results/`

## Expected Results
- **Development**: ~95% accuracy in 30 minutes
- **Production**: ~99% accuracy in 4 hours
- **GPU Memory**: ~2GB usage
- **Storage**: Results saved to Google Drive

## Colab Advantages
- **Free GPU**: T4 GPU with 15GB memory
- **Persistent Storage**: Google Drive integration
- **No Setup**: Pre-installed libraries
- **Easy Sharing**: Share notebooks with collaborators

## Troubleshooting

### Memory Issues
```python
# Reduce batch size
!python scripts/train.py --model lightweight --training development
```

### Session Timeout
```python
# Use development config for quick training
!python scripts/train.py --training development
```

### Data Path Issues
```python
# Check Google Drive mount
!ls /content/drive/MyDrive/
# Update data path in notebook
```

## Model Configurations
- `fast`: MobileNet (fastest)
- `balanced`: ResNet34 (recommended)
- `accurate`: EfficientNet (best quality)
- `lightweight`: EfficientNet-B0 (memory efficient)