# Colombo Slum Detection Analysis

This folder contains specialized scripts for analyzing satellite images of Colombo to detect informal settlements (slums) using the trained UNet model.

## 📁 Folder Structure

```
colombo/
├── 📄 README.md                    # This file
├── 🔧 process_colombo_image.py     # Image preprocessing and tiling
├── 🤖 detect_slums.py              # Slum detection inference
├── 🎨 visualize_results.py         # Results visualization
├── 🚀 run_pipeline.py              # Complete analysis pipeline
├── 📂 tiles/                       # Processed image tiles (120x120px)
├── 📂 predictions/                 # Detection results and analysis
├── 📂 visualizations/              # Charts and graphs
├── 📂 metadata/                    # Processing metadata
└── 📄 colombo_analysis_report.json # Final comprehensive report
```

## 🛠️ Scripts Overview

### 1. `process_colombo_image.py` - Image Preprocessing
Converts satellite images into model-compatible 120x120 pixel tiles.

**Features:**
- Loads high-resolution satellite images (PNG, JPG, TIFF)
- Splits into 120x120 pixel tiles (model requirement)
- Applies RGB preprocessing and normalization
- Saves tiles with spatial metadata
- Creates tile overview visualization

**Usage:**
```bash
python process_colombo_image.py --image satellite_image.jpg --output colombo --tile-size 120
```

### 2. `detect_slums.py` - Slum Detection
Performs slum detection on processed tiles using the trained model.

**Features:**
- Loads trained UNet model from checkpoint
- Processes tiles for slum detection
- Calculates probabilities and confidence levels
- Generates detection statistics and analysis
- Creates prediction visualizations

**Usage:**
```bash
python detect_slums.py --tiles-dir colombo/tiles --checkpoint ../experiments/development_20250713_175410/checkpoints/best_checkpoint.pth --output colombo/predictions
```

### 3. `visualize_results.py` - Results Visualization
Creates comprehensive visualizations of detection results.

**Features:**
- Spatial heatmap of slum probabilities
- Probability distribution analysis
- Confidence level breakdown
- Comprehensive summary dashboard
- Statistical charts and graphs

**Usage:**
```bash
python visualize_results.py --results colombo/predictions/colombo_slum_detection_results.json --output colombo/visualizations
```

### 4. `run_pipeline.py` - Complete Pipeline
Runs the entire analysis pipeline from raw image to final report.

**Features:**
- End-to-end processing automation
- Combines all analysis steps
- Generates comprehensive final report
- Provides executive summary
- Includes recommendations based on findings

**Usage:**
```bash
python run_pipeline.py --image satellite_image.jpg --checkpoint ../experiments/development_20250713_175410/checkpoints/best_checkpoint.pth --output colombo
```

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)
Run everything with a single command:
```bash
python run_pipeline.py --image your_satellite_image.jpg --checkpoint ../experiments/development_20250713_175410/checkpoints/best_checkpoint.pth
```

### Option 2: Step-by-Step Analysis
1. **Process the image:**
   ```bash
   python process_colombo_image.py --image your_satellite_image.jpg
   ```

2. **Detect slums:**
   ```bash
   python detect_slums.py --tiles-dir colombo/tiles --checkpoint ../experiments/development_20250713_175410/checkpoints/best_checkpoint.pth
   ```

3. **Create visualizations:**
   ```bash
   python visualize_results.py --results colombo/predictions/colombo_slum_detection_results.json
   ```

## 📊 Output Files

### Processing Outputs
- `tiles/*.png` - Individual 120x120 pixel tiles
- `metadata/colombo_tile_metadata.json` - Spatial metadata for tiles
- `tile_overview.png` - Grid visualization of all tiles

### Detection Outputs
- `predictions/colombo_slum_detection_results.json` - Complete results
- `predictions/detection_summary.json` - Executive summary
- `predictions/prediction_visualization.png` - Sample predictions
- `predictions/probability_distribution.png` - Probability histogram

### Visualization Outputs
- `visualizations/spatial_heatmap.png` - Geographic distribution
- `visualizations/probability_analysis.png` - Statistical analysis
- `visualizations/confidence_analysis.png` - Confidence breakdown
- `visualizations/comprehensive_summary.png` - Overall dashboard

### Final Report
- `colombo_analysis_report.json` - Comprehensive analysis report with:
  - Executive summary
  - Detailed findings
  - Methodology description
  - File locations
  - Recommendations

## 🎯 Model Requirements

The scripts automatically handle model requirements:
- **Input size:** All images are processed into 120x120 pixel tiles
- **RGB format:** 3-channel RGB images are required
- **Normalization:** Pixel values are normalized to [0,1] range
- **Format support:** PNG, JPG, JPEG, TIFF formats supported

## 📈 Analysis Features

### Detection Metrics
- **Slum probability** for each tile (0-1 scale)
- **Binary classification** (slum/non-slum) using threshold
- **Confidence levels** (high/medium/low) based on probability distance from threshold

### Spatial Analysis
- **Geographic distribution** of detected slums
- **Spatial clustering** analysis
- **Coverage statistics** (percentage of area with slums)

### Statistical Analysis
- **Probability distributions** and histograms
- **Confidence breakdowns** by detection quality
- **Threshold sensitivity** analysis
- **Summary statistics** (mean, min, max probabilities)

## 🔧 Advanced Options

### Tile Processing
- `--tile-size`: Tile dimensions (default: 120)
- `--overlap`: Pixel overlap between tiles (default: 0)

### Detection
- `--threshold`: Classification threshold (default: 0.5)
- Higher threshold = fewer but more confident detections
- Lower threshold = more detections but potentially more false positives

### Output Control
- `--output`: Custom output directory
- `--quiet`: Reduce verbosity for automated processing

## 📋 Requirements

Make sure you have the trained model checkpoint available:
- Default location: `../experiments/development_20250713_175410/checkpoints/best_checkpoint.pth`
- Or specify custom path with `--checkpoint` parameter

All Python dependencies are included in the main project `requirements.txt`.

## 🎯 Use Cases

1. **Urban Planning:** Identify informal settlements for development planning
2. **Policy Making:** Quantify slum coverage for resource allocation
3. **Monitoring:** Track changes in informal settlements over time
4. **Research:** Academic studies on urban development patterns
5. **NGO Work:** Target interventions in specific areas

## ⚠️ Important Notes

- **Image Quality:** Higher resolution images provide better detection accuracy
- **Geographic Context:** Results are specific to satellite imagery similar to training data
- **Validation:** Consider field verification for critical planning decisions
- **Temporal Changes:** Model reflects conditions at time of training data collection

## 🆘 Troubleshooting

**Common Issues:**
1. **Memory errors:** Reduce tile overlap or process smaller image sections
2. **Model loading errors:** Verify checkpoint path and model compatibility
3. **Image format errors:** Convert to RGB format if needed
4. **No detections:** Adjust threshold or verify image quality

**Getting Help:**
- Check console output for detailed error messages
- Verify all file paths are correct
- Ensure sufficient disk space for outputs
- Check that all dependencies are installed
