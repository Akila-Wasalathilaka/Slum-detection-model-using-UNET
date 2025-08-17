@echo off
REM Quick run script for SlumSeg pipeline on Windows
REM Usage: run.bat

echo 🏗️  SlumSeg Pipeline Runner
echo ==========================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

REM Install requirements
echo 📦 Installing requirements...
pip install -r requirements.txt

REM Create output directories
if not exist "outputs\charts" mkdir outputs\charts
if not exist "outputs\predictions" mkdir outputs\predictions
if not exist "outputs\checkpoints" mkdir outputs\checkpoints

echo 🔍 Step 1: Dataset Analysis
python scripts\analyze_dataset.py --config configs\default.yaml --out outputs\charts

echo 🏋️  Step 2: Training
python scripts\train.py --config configs\default.yaml

echo 📊 Step 3: Evaluation (20 charts)
python scripts\evaluate.py --config configs\default.yaml --ckpt outputs\checkpoints\best.ckpt --tiles . --charts outputs\charts

echo 🎯 Step 4: Inference (20 predictions)
python scripts\infer.py --config configs\default.yaml --ckpt outputs\checkpoints\best.ckpt --images ..\data\val\images --out outputs\predictions --num 20

echo ✅ Pipeline complete!
echo 📁 Results saved to outputs\
echo    - Charts: outputs\charts\
echo    - Predictions: outputs\predictions\  
echo    - Checkpoints: outputs\checkpoints\

pause
