# SlumSeg Kaggle Pipelines

Two separate pipelines for running SlumSeg on Kaggle T4 GPU:

## 🚂 Pipeline 1: Training Only (`kaggle_train_only.py`)

**Purpose**: Train the slum segmentation model from scratch
**Time**: ~20-30 minutes on T4
**Output**: Trained model checkpoints

### Steps:
1. Create new Kaggle notebook
2. Copy code from `kaggle_train_only.py`
3. Update these variables:
   ```python
   REPO_URL = "https://github.com/YOUR-USERNAME/SlumSeg.git"
   training_config['data']['root'] = '/kaggle/input/YOUR-DATASET-NAME'
   ```
4. Add your dataset to Kaggle
5. Run all cells
6. Download `slumseg_trained_model_YYYYMMDD_HHMMSS.zip`

### What you get:
- ✅ Trained model checkpoints (`best.ckpt`, `last.ckpt`)
- ✅ Training configuration
- ✅ Training logs
- ✅ Model ready for inference

---

## 📊 Pipeline 2: Analysis & Evaluation (`kaggle_analysis_pipeline.py`)

**Purpose**: Analyze dataset + evaluate trained model + generate predictions
**Time**: ~15-20 minutes on T4
**Output**: 20 charts + 20 prediction overlays

### Steps:
1. Create new Kaggle notebook
2. Copy code from `kaggle_analysis_pipeline.py`
3. Update these variables:
   ```python
   REPO_URL = "https://github.com/YOUR-USERNAME/SlumSeg.git"
   analysis_config['data']['root'] = '/kaggle/input/YOUR-DATASET-NAME'
   MODEL_PATH = "/kaggle/input/your-trained-model/best.ckpt"
   ```
4. Add your dataset to Kaggle
5. Upload your trained model checkpoint
6. Run all cells
7. Download `slumseg_complete_analysis_YYYYMMDD_HHMMSS.zip`

### What you get:
- ✅ **20 Analysis Charts**:
  - Charts 1-6: Dataset analysis (structure, distribution, samples)
  - Charts 7-15: Model evaluation (metrics, ROC, confusion matrix)
  - Charts 16-20: Additional analysis (architecture, performance, deployment)
- ✅ **20 Prediction Overlays**: Red slum areas on satellite imagery
- ✅ Comprehensive analysis report

---

## 🔄 Recommended Workflow

### Option A: Full Pipeline (Separate Notebooks)
1. **First**: Run training pipeline → Get trained model
2. **Second**: Run analysis pipeline → Get charts + predictions

### Option B: Training Only
1. Run training pipeline
2. Use trained model in your own inference code

### Option C: Analysis Only
1. Upload pre-trained model
2. Run analysis pipeline for evaluation

---

## 📋 Prerequisites

### Dataset Structure
Your Kaggle dataset should have this structure:
```
your-dataset/
├── train/
│   ├── images/
│   │   ├── image1.tif
│   │   └── image2.tif
│   └── masks/
│       ├── image1.tif  # 0=background, 1=slum
│       └── image2.tif
├── val/           # Optional
│   ├── images/
│   └── masks/
└── test/          # Optional
    ├── images/
    └── masks/
```

### Model Upload (for Analysis Pipeline)
- Upload your `.ckpt` file as a Kaggle dataset
- Or add it to `/kaggle/input/` directory
- Update `MODEL_PATH` in the analysis script

---

## ⚙️ Configuration

### Training Pipeline Settings
- **Epochs**: 20 (good for T4 time limits)
- **Batch Size**: 12 (optimized for T4 16GB)
- **Model**: UNet + ResNet34 (speed/accuracy balance)
- **Optimizations**: AMP, channels_last, gradient checkpointing

### Analysis Pipeline Settings
- **Predictions**: 20 overlays with TTA
- **Charts**: 20 comprehensive analysis charts
- **Threshold**: 0.45 (optimized for slum detection)
- **Post-processing**: Morphological operations enabled

---

## 🎯 Expected Results

### Training Pipeline Output
```
slumseg_trained_model_20240101_120000.zip
├── checkpoints/
│   ├── best.ckpt      # Best validation model
│   ├── last.ckpt      # Final epoch model
│   └── epoch_XX.ckpt  # Intermediate checkpoints
├── training_config.yaml
├── logs/              # Training logs
└── README.txt         # Usage instructions
```

### Analysis Pipeline Output
```
slumseg_complete_analysis_20240101_120000.zip
├── charts/
│   ├── 01_dataset_overview.png
│   ├── 02_image_properties.png
│   ├── ...
│   └── 20_deployment_readiness.png
├── predictions/
│   ├── pred_01.png    # Red overlay predictions
│   ├── pred_02.png
│   ├── ...
│   ├── pred_20.png
│   ├── prob_01.png    # Probability maps
│   └── ...
├── analysis_config.yaml
└── README.txt
```

---

## 🚨 Important Notes

1. **Update URLs**: Replace `YOUR-USERNAME` and `YOUR-DATASET-NAME`
2. **GPU Time**: Each pipeline uses ~20-30 minutes of T4 time
3. **Memory**: Optimized for T4 16GB GPU memory
4. **Internet**: Required for cloning repository and installing packages
5. **Dataset Size**: Works best with datasets < 5GB for Kaggle limits

---

## 🔧 Troubleshooting

### Common Issues:
- **"Dataset not found"**: Update `data.root` path in config
- **"Model not found"**: Update `MODEL_PATH` in analysis pipeline
- **"Out of memory"**: Reduce `batch_size` in config
- **"Time limit exceeded"**: Reduce `epochs` or use smaller model

### Performance Tips:
- Use smaller tile sizes (256x256) for faster processing
- Reduce number of predictions if needed
- Skip TTA for faster inference
- Use `efficientnet_b0` instead of `resnet34` for speed

---

## 📞 Support

If you encounter issues:
1. Check the error messages in Kaggle logs
2. Verify dataset structure and paths
3. Ensure model checkpoint is uploaded correctly
4. Check GPU memory usage in Kaggle

Happy slum detection! 🏠✨