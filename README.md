# Slum Detection Using U-Net Architecture

A clean, minimal implementation for multi-class slum detection using U-Net architecture.

## Dataset Analysis Results

Based on comprehensive analysis, the dataset contains:
- **7 classes** with the following distribution:
  - Class 0: Background (0.08%)
  - Class 105: Slum Type A (12.44%)
  - Class 109: Slum Main Type (31.89%) - Most common
  - Class 111: Slum Type B (18.75%)
  - Class 158: Slum Type C (15.84%)
  - Class 200: Slum Type D (9.15%)
  - Class 233: Slum Type E (11.84%)

- **Dataset splits:**
  - Train: 7,128 images
  - Validation: 891 images
  - Test: 891 images
  - Image size: 120x120x3

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Slum-Detection-Using-Unet-Architecture
pip install -r requirements.txt
```

### 2. Run Dataset Analysis
```bash
python quick_analysis.py
```
This will generate class distribution plots in the `analysis/` directory.

### 3. Train the Model
```bash
python kaggle_slum_detection.py
```

## Model Architecture

- **U-Net** with encoder-decoder structure
- **Input:** 128x128x3 RGB images (resized from 120x120)
- **Output:** 7-class segmentation masks
- **Features:**
  - Batch normalization
  - Skip connections
  - Class-weighted loss for imbalanced data
  - Data augmentation

## Training Configuration

- **Batch size:** 16
- **Epochs:** 20
- **Learning rate:** 1e-3 with ReduceLROnPlateau scheduler
- **Loss:** CrossEntropyLoss with class weights
- **Optimizer:** Adam
- **Augmentations:** Flip, rotation, brightness/contrast

## Files Structure

```
├── data/
│   ├── train/images/    # Training images (.tif)
│   ├── train/masks/     # Training masks (.png)
│   ├── val/images/      # Validation images
│   ├── val/masks/       # Validation masks
│   ├── test/images/     # Test images
│   └── test/masks/      # Test masks
├── analysis/            # Analysis results and plots
├── kaggle_slum_detection.py  # Main training script
├── quick_analysis.py    # Dataset analysis script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Output Files

After training:
- `best_slum_model.pth` - Best model weights
- `training_history.png` - Training curves
- `predictions.png` - Sample predictions

## Key Features

1. **Multi-class segmentation** (7 classes)
2. **Class-weighted training** for imbalanced data
3. **Memory efficient** implementation
4. **Kaggle-ready** - minimal dependencies
5. **Comprehensive analysis** with visualizations

## Usage in Kaggle

1. Upload this repository to Kaggle
2. Ensure data is in the correct structure
3. Run `python kaggle_slum_detection.py`
4. Model will train and save results automatically

## Performance Metrics

The model tracks:
- **Pixel-wise accuracy** across all classes
- **Class-weighted loss** for balanced training
- **Per-epoch validation** with early stopping capability

## Customization

Modify these parameters in `kaggle_slum_detection.py`:
- `IMG_SIZE`: Input image size (default: 128)
- `BATCH_SIZE`: Batch size (default: 16)
- `EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate (default: 1e-3)