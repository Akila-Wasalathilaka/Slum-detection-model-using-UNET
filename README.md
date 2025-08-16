# Advanced Slum Detection Using U-Net Architecture

A comprehensive, state-of-the-art implementation for multi-class slum detection using advanced U-Net architecture with EfficientNet backbone.

## 🚀 Kaggle Quick Start (One-Click Solution)

### Option 1: Complete Pipeline (Recommended)
```python
# In Kaggle Notebook, run this single command:
!git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
%cd Slum-detection-model-using-UNET
!python kaggle_complete_pipeline.py
```

### Option 2: Step-by-Step
```python
# 1. Clone and setup
!git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
%cd Slum-detection-model-using-UNET
!python kaggle_setup.py

# 2. Run analysis
!python quick_analysis.py

# 3. Train advanced model
!python advanced_training.py

# 4. Generate charts
!python create_charts.py

# 5. Make predictions
!python make_predictions.py --num 25
```

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

## 🏗️ Advanced Model Architecture

- **U-Net++** with EfficientNet-B4 encoder
- **Input:** 256x256x3 RGB images (upscaled for better accuracy)
- **Output:** 7-class segmentation masks
- **Advanced Features:**
  - Pre-trained EfficientNet-B4 backbone
  - Multi-head attention mechanisms
  - Deep supervision
  - Mixed precision training
  - Advanced data augmentations
  - Class-weighted Focal + Dice loss

## 🎯 Training Configuration

- **Architecture:** U-Net++ with EfficientNet-B4
- **Image size:** 256x256 (enhanced resolution)
- **Batch size:** 8 (optimized for larger images)
- **Epochs:** 50 (with early stopping)
- **Learning rate:** 1e-4 with CosineAnnealingWarmRestarts
- **Loss:** Combined Focal + Dice Loss with class weights
- **Optimizer:** AdamW with weight decay
- **Augmentations:** 15+ advanced augmentations including:
  - Geometric: Flip, rotation, elastic transform, grid distortion
  - Color: Brightness/contrast, CLAHE, HSV shifts
  - Noise: Gaussian noise, blur, motion blur
  - Weather: Random shadow, fog effects
  - Regularization: Coarse dropout

## 📁 Complete File Structure

```
├── data/
│   ├── train/images/    # Training images (.tif)
│   ├── train/masks/     # Training masks (.png)
│   ├── val/images/      # Validation images
│   ├── val/masks/       # Validation masks
│   ├── test/images/     # Test images
│   └── test/masks/      # Test masks
├── analysis/            # Dataset analysis results
│   ├── class_analysis.png
│   ├── sample_images.png
│   └── class_analysis_results.json
├── charts/              # Comprehensive analysis charts
│   ├── dataset_overview.png
│   ├── model_performance.png
│   └── training_analysis.png
├── predictions/         # Model predictions (25+ samples)
│   ├── prediction_01_*.png
│   ├── prediction_summary.png
│   └── predictions_data.json
├── kaggle_setup.py           # Kaggle setup script
├── kaggle_complete_pipeline.py  # One-click complete pipeline
├── advanced_training.py      # Advanced training script
├── create_charts.py          # Comprehensive chart generation
├── make_predictions.py       # Advanced prediction generator
├── quick_analysis.py         # Dataset analysis script
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 📊 Output Files

After running the complete pipeline:
- `best_advanced_slum_model.pth` - Best model weights
- `advanced_training_history.png` - Comprehensive training curves
- `advanced_training_history.json` - Training metrics data
- `analysis/` - Dataset analysis with class distribution
- `charts/` - Performance analysis and visualizations
- `predictions/` - 25+ diverse predictions with analysis

## 🌟 Key Features

1. **Advanced Architecture**: U-Net++ with EfficientNet-B4 backbone
2. **State-of-the-art Training**: Mixed precision, advanced augmentations
3. **Comprehensive Analysis**: 7-class multi-class segmentation
4. **Balanced Training**: Focal + Dice loss with class weights
5. **Extensive Evaluation**: 25+ diverse predictions with analysis
6. **Professional Visualizations**: Comprehensive charts and metrics
7. **Kaggle-Optimized**: One-click deployment and execution
8. **Production-Ready**: Complete pipeline with error handling

## 🎯 Performance Metrics

The advanced model tracks:
- **Pixel-wise accuracy** across all 7 classes
- **Per-class IoU, F1-score, Precision, Recall**
- **Confidence analysis** and uncertainty quantification
- **Class-weighted metrics** for imbalanced data handling
- **Comprehensive error analysis** by class and region

## ⚙️ Advanced Configuration

Modify parameters in `advanced_training.py`:
```python
CONFIG = {
    'IMG_SIZE': 256,              # Higher resolution
    'BATCH_SIZE': 8,              # Optimized for GPU memory
    'EPOCHS': 50,                 # Extended training
    'LEARNING_RATE': 1e-4,        # Fine-tuned learning rate
    'ENCODER': 'efficientnet-b4', # Advanced backbone
    'USE_MIXED_PRECISION': True,  # Memory optimization
    'EARLY_STOPPING_PATIENCE': 10,
    'REDUCE_LR_PATIENCE': 5,
}
```

## 🏆 Expected Performance

- **Validation Accuracy**: 85-95%+ (depending on data quality)
- **Training Time**: 1-3 hours on GPU
- **Memory Usage**: ~8GB GPU memory
- **Model Size**: ~100MB
- **Inference Speed**: ~50ms per image

## 📈 What You Get

1. **Trained Model**: State-of-the-art slum detection model
2. **Comprehensive Analysis**: Dataset statistics and class distribution
3. **Performance Charts**: Training curves, confusion matrices, metrics
4. **25+ Predictions**: Diverse samples with confidence analysis
5. **Production Pipeline**: Ready-to-deploy prediction system
6. **Documentation**: Complete analysis and performance reports

## 🔧 Troubleshooting

### Common Issues:
1. **GPU Memory Error**: Reduce `BATCH_SIZE` to 4 or 2
2. **Package Installation**: Run `!pip install --upgrade pip` first
3. **Data Not Found**: Ensure data folder structure is correct
4. **Training Too Slow**: Enable mixed precision training

### Performance Tips:
1. Use GPU runtime in Kaggle for faster training
2. Enable internet access for downloading pre-trained weights
3. Monitor GPU memory usage during training
4. Use early stopping to prevent overfitting

## 📞 Support

For issues or questions:
1. Check the generated logs in each output directory
2. Review the comprehensive analysis charts
3. Examine the prediction confidence scores
4. Adjust hyperparameters based on your specific dataset

---

**Ready to detect slums with state-of-the-art accuracy? Run the pipeline now! 🚀**