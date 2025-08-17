#!/bin/bash

# SlumSeg Complete Pipeline Runner
# This script integrates with your existing data structure and runs the complete pipeline

set -e  # Exit on any error

echo "🏠 SlumSeg: End-to-End Slum Segmentation Pipeline"
echo "=================================================="

# Default parameters
CONFIG_FILE="configs/default.yaml"
DATA_ROOT="e:/Slum-Detection-Using-Unet-Architecture/data"
OUTPUT_DIR="outputs"
KAGGLE_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --kaggle)
            KAGGLE_MODE=true
            CONFIG_FILE="configs/t4_fast.yaml"
            DATA_ROOT="/kaggle/input/slum-detection"
            OUTPUT_DIR="/kaggle/working/outputs"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config FILE     Config file to use (default: configs/default.yaml)"
            echo "  --data-root PATH  Data root directory (default: your existing data folder)"
            echo "  --output DIR      Output directory (default: outputs)"
            echo "  --kaggle          Use Kaggle T4 optimized settings"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p "$OUTPUT_DIR"/{charts,predictions,checkpoints,tiles}

echo "📍 Configuration:"
echo "   Config file: $CONFIG_FILE"
echo "   Data root: $DATA_ROOT"
echo "   Output dir: $OUTPUT_DIR"
echo "   Kaggle mode: $KAGGLE_MODE"
echo ""

# Step 1: Environment setup
echo "🔧 Step 1: Environment Setup"
echo "-----------------------------"
if [ "$KAGGLE_MODE" = true ]; then
    echo "Setting up Kaggle T4 environment..."
    pip install -r requirements.txt --no-input --quiet
    nvidia-smi
else
    echo "Setting up local environment..."
    pip install -r requirements.txt
fi

# Check PyTorch installation
python -c "import torch; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda if torch.cuda.is_available() else \"CPU only\"}')"
echo ""

# Step 2: Dataset Analysis
echo "📊 Step 2: Dataset Analysis (Charts 1-5)"
echo "----------------------------------------"
echo "Analyzing your existing dataset structure..."

# Update config with correct data root
python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
config['data']['root'] = '$DATA_ROOT'
with open('temp_config.yaml', 'w') as f:
    yaml.dump(config, f)
"

python scripts/analyze_dataset.py --config temp_config.yaml --out "$OUTPUT_DIR/charts"
echo "✅ Dataset analysis complete! Check $OUTPUT_DIR/charts/ for initial charts."
echo ""

# Step 3: Data Preprocessing
echo "🔄 Step 3: Data Preprocessing"
echo "-----------------------------"
echo "Creating tiles and computing class weights..."
python scripts/prep_dataset.py --config temp_config.yaml --out "$OUTPUT_DIR/tiles"
echo "✅ Data preprocessing complete!"
echo ""

# Step 4: Model Training
echo "🚂 Step 4: Model Training"
echo "-------------------------"
echo "Training segmentation model..."
python scripts/train.py --config temp_config.yaml --tiles "$OUTPUT_DIR/tiles"
echo "✅ Model training complete!"
echo ""

# Step 5: Model Evaluation
echo "📈 Step 5: Model Evaluation (Charts 6-20)"
echo "-----------------------------------------"
echo "Evaluating model and generating comprehensive charts..."
python scripts/evaluate.py \
    --config temp_config.yaml \
    --ckpt "$OUTPUT_DIR/checkpoints/best.ckpt" \
    --tiles "$OUTPUT_DIR/tiles" \
    --charts "$OUTPUT_DIR/charts"
echo "✅ Model evaluation complete!"
echo ""

# Step 6: Inference and Overlays
echo "🔮 Step 6: Inference (20 Prediction Overlays)"
echo "---------------------------------------------"
echo "Generating red overlay predictions..."
python scripts/infer.py \
    --config temp_config.yaml \
    --ckpt "$OUTPUT_DIR/checkpoints/best.ckpt" \
    --images "$OUTPUT_DIR/tiles/val/images" \
    --out "$OUTPUT_DIR/predictions" \
    --num 20
echo "✅ Inference complete!"
echo ""

# Step 7: Results Summary
echo "🎉 Pipeline Complete!"
echo "===================="
echo ""
echo "📊 Generated Outputs:"
echo "   • 20 Analysis Charts: $OUTPUT_DIR/charts/"
echo "   • 20 Prediction Overlays: $OUTPUT_DIR/predictions/"
echo "   • Trained Model: $OUTPUT_DIR/checkpoints/best.ckpt"
echo ""

# Display sample results
if [ -d "$OUTPUT_DIR/charts" ]; then
    chart_count=$(ls "$OUTPUT_DIR/charts"/*.png 2>/dev/null | wc -l || echo 0)
    echo "   Charts generated: $chart_count"
fi

if [ -d "$OUTPUT_DIR/predictions" ]; then
    pred_count=$(ls "$OUTPUT_DIR/predictions"/*.png 2>/dev/null | wc -l || echo 0)
    echo "   Predictions generated: $pred_count"
fi

# Package results for Kaggle
if [ "$KAGGLE_MODE" = true ]; then
    echo ""
    echo "📦 Packaging results for download..."
    cd "$OUTPUT_DIR"
    zip -r slumseg_artifacts.zip charts predictions checkpoints
    echo "✅ Download slumseg_artifacts.zip from the Output panel!"
fi

# Cleanup
rm -f temp_config.yaml

echo ""
echo "🚀 SlumSeg pipeline completed successfully!"
echo "Your dataset has been analyzed, tiled, trained, and evaluated."
echo "Check the output directories for comprehensive results!"
