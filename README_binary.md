# Binary Slum Detection with UNet

This repository contains an enhanced UNet-based semantic segmentation pipeline for detecting informal settlements (slums) from satellite imagery. The system converts multi-class annotations into binary classification (slum vs non-slum) and compares multiple loss functions and architectures.

## 🎯 Overview

### Key Features
- **Binary Classification**: Converts multi-class segmentation to slum vs non-slum
- **Multiple Architectures**: UNet with ResNet34, EfficientNet-B0 encoders
- **Advanced Loss Functions**: BCE, BCE+Dice, Focal Loss, Tversky Loss
- **Comprehensive Augmentation**: Geometric, color, noise, and weather effects
- **Post-processing**: Morphological operations to refine predictions
- **Extensive Evaluation**: IoU, Dice, F1-score, Precision, Recall, AUC metrics
- **Model Comparison**: Automated comparison across all configurations

### Dataset Specifications
- **Input**: 120×120 RGB satellite tiles from Pleiades-1A imagery
- **Original Classes**: 7 classes (vegetation, built-up, informal settlements, impervious surfaces, barren, water, unlabelled)
- **Target**: Binary classification (informal settlements = 1, all others = 0)
- **Split**: 7,128 train / 891 validation / 891 test images

## 🚀 Quick Start

### 1. Install Requirements (Optional)
If you don't have the required packages installed:
```bash
# Install requirements (optional - pipeline will skip if not needed)
pip install -r requirements.txt
```

**Required packages** (install manually if needed):
- torch, torchvision, segmentation-models-pytorch
- opencv-python, albumentations, scikit-image  
- numpy, pandas, matplotlib, seaborn, scikit-learn

### 2. Data Organization
Ensure your data is organized as:
```
data/
├── train/
│   ├── images/     # *.tif files
│   └── masks/      # *.png files (RGB multi-class masks)
├── val/
│   ├── images/     # *.tif files
│   └── masks/      # *.png files
└── test/
    ├── images/     # *.tif files
    └── masks/      # *.png files
```

### 3. Run Complete Pipeline
```bash
python run_binary_pipeline.py
```

This will:
1. Check data organization
2. Preprocess data (analyze class distribution)
3. Train multiple models with different configurations
4. Generate comprehensive evaluation reports

**Note**: The pipeline will skip package installation and run directly. Make sure you have the required packages installed manually if needed.

### 4. Run Individual Components

#### Data Preprocessing Only
```bash
python preprocess_binary.py
```

#### Training Only
```bash
python binary_slum_detection.py
```

## 📊 Model Configurations

### Encoders Tested
- **ResNet34**: Deep residual network, good balance of performance and speed
- **EfficientNet-B0**: Efficient architecture optimized for mobile deployment

### Loss Functions Compared
1. **Binary Cross Entropy (BCE)**: Standard binary classification loss
2. **BCE + Dice**: Combines BCE with Dice loss (0.5 weight each)
3. **Focal Loss**: Addresses class imbalance (α=0.75, γ=2.0)
4. **Tversky Loss**: Flexible loss for imbalanced data (α=0.7, β=0.3)

### Data Augmentation Pipeline
- **Geometric**: Flips, rotations, shifts, scaling, elastic deformation
- **Color**: Brightness/contrast, gamma, HSV shifts, RGB shifts
- **Noise/Blur**: Gaussian noise, Gaussian blur, motion blur
- **Weather**: Shadows, fog effects
- **Cutout**: Random rectangular patches

## 🔧 Configuration

Key parameters can be modified in `binary_slum_detection.py`:

```python
class Config:
    IMAGE_SIZE = 120          # Input image size
    BATCH_SIZE = 16           # Training batch size
    EPOCHS = 100              # Maximum training epochs
    LEARNING_RATE = 1e-4      # Initial learning rate
    PATIENCE = 15             # Early stopping patience
    SLUM_CLASS_ID = 2         # ID of slum class in original masks
    BINARY_THRESHOLD = 0.5    # Binary classification threshold
```

## 📈 Evaluation Metrics

### Primary Metrics
- **IoU (Intersection over Union)**: Jaccard index for spatial overlap
- **Dice Coefficient**: F1-score equivalent for segmentation
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Additional Metrics
- **Pixel Accuracy**: Overall pixel-wise classification accuracy
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

## 📁 Output Structure

After running the pipeline, you'll find:

```
results_binary/
├── experiment_results.json           # Detailed results for all models
├── model_comparison.png              # Visual comparison charts
├── unet_resnet34_bce_history.png     # Training history plots
├── unet_resnet34_bce_predictions.png # Prediction visualizations
├── unet_resnet34_focal_history.png
├── unet_resnet34_focal_predictions.png
└── ... (for all model configurations)

models_binary/
├── unet_resnet34_bce.pth            # Trained model weights
├── unet_resnet34_focal.pth
├── unet_efficientnet-b0_bce.pth
└── ... (for all model configurations)

data_preprocessed/
├── train/
├── val/
└── test/
```

## 🎛️ Post-processing

The system applies morphological post-processing to improve predictions:

1. **Small object removal**: Remove connected components < 50 pixels
2. **Morphological opening**: Remove noise with 3×3 kernel
3. **Morphological closing**: Fill holes with 5×5 kernel

This can be configured in the `apply_morphological_postprocessing()` function.

## 🧪 Experiment Results

The system automatically generates:

1. **Training History Plots**: Loss, IoU, Dice coefficient over epochs
2. **Prediction Visualizations**: Side-by-side comparison of input, ground truth, probability maps, and final predictions
3. **Model Comparison Charts**: Bar charts and scatter plots comparing all models
4. **Detailed Metrics Table**: Comprehensive performance comparison

### Sample Results Format
```json
{
  "Model": "unet_resnet34_focal",
  "Encoder": "resnet34",
  "Loss": "focal",
  "Val IoU": 0.7234,
  "Test IoU": 0.7156,
  "Test Dice": 0.8321,
  "Test F1": 0.8298,
  "Test Precision": 0.8567,
  "Test Recall": 0.8045,
  "Test Pixel Acc": 0.9234,
  "AUC-ROC": 0.9456,
  "AUC-PR": 0.8876
}
```

## 🔍 Class Imbalance Analysis

The preprocessing script provides detailed class distribution analysis:
- Pixel counts for each original class
- Binary classification distribution (slum vs non-slum)
- Class imbalance ratio calculation
- Per-dataset statistics

This information helps in:
- Understanding dataset characteristics
- Choosing appropriate loss functions
- Setting loss function parameters

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in config
   - Use smaller image size
   - Enable gradient checkpointing

2. **Slow Training**
   - Reduce `NUM_WORKERS` if system is slow
   - Disable some augmentations
   - Use smaller model (efficientnet-b0 vs resnet34)

3. **Poor Performance**
   - Check class distribution (extreme imbalance)
   - Adjust loss function parameters
   - Increase training epochs
   - Verify data quality

### GPU Requirements
- Recommended: NVIDIA GPU with ≥6GB VRAM
- Minimum: 4GB VRAM (with reduced batch size)
- CPU-only training is supported but significantly slower

## 📚 Dependencies

### Core Libraries
- PyTorch ≥1.13.0
- torchvision ≥0.14.0
- segmentation-models-pytorch ≥0.3.0

### Image Processing
- OpenCV ≥4.5.0
- albumentations ≥1.3.0
- scikit-image ≥0.18.0

### Scientific Computing
- NumPy ≥1.21.0
- scikit-learn ≥1.0.0
- pandas ≥1.3.0

### Visualization
- matplotlib ≥3.5.0
- seaborn ≥0.11.0

## 🤝 Contributing

To extend the pipeline:

1. **Add new encoders**: Modify the `ENCODERS` list in config
2. **Add new loss functions**: Implement in the loss functions section
3. **Add new metrics**: Extend the `MetricsTracker` class
4. **Add new augmentations**: Modify `get_training_augmentation()`

## 📄 License

This project is open source. Please check the original dataset license for data usage restrictions.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error messages in the console output
3. Verify data organization and requirements installation
4. Check CUDA/GPU compatibility if using GPU acceleration

## 🎯 Performance Benchmarks

Expected performance ranges (may vary with dataset):
- **IoU**: 0.65-0.80
- **F1-Score**: 0.75-0.85
- **Training Time**: 2-4 hours on modern GPU
- **Inference Speed**: ~100-200 images/second on GPU

The exact performance depends on:
- Dataset quality and class balance
- Model architecture choice
- Loss function selection
- Training hyperparameters
