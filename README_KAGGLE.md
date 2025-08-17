# Kaggle Pipelines for Slum Detection

## Quick Start

### 1. Setup Environment
```bash
python kaggle_setup.py
```

### 2. Training Pipeline
```bash
python kaggle_train_pipeline.py
```

### 3. Analysis Pipeline
```bash
python kaggle_analysis_pipeline.py
```

## Files Created

- **kaggle_setup.py** - Environment setup and dependency installation
- **kaggle_train_pipeline.py** - Minimal training pipeline
- **kaggle_analysis_pipeline.py** - Analysis, charts, and predictions

## Expected Outputs

### Training Pipeline
- Trained model saved as `/kaggle/working/slum_model.pth`
- Training progress logs

### Analysis Pipeline
- `/kaggle/working/analysis_charts.png` - ROC curve, confusion matrix, threshold analysis
- `/kaggle/working/prediction_samples.png` - Sample predictions visualization
- Performance metrics printed to console

## Data Structure Expected

```
/kaggle/input/your-dataset/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
```

## Model Architecture

Uses the existing UNet + ResNet34 architecture without any modifications to maintain the 99.67% AUC-ROC performance.