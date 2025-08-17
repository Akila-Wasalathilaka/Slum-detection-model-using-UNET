#!/usr/bin/env python3
"""
SlumSeg Complete Pipeline Runner - Python Version
Integrates with your existing data structure and runs the complete pipeline
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description="Running command"):
    """Run a shell command and handle errors."""
    # Sanitize command for logging to prevent log injection
    safe_cmd = str(cmd).replace('\n', '').replace('\r', '')
    logger.info(f"{description}: {safe_cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Sanitize error output for logging
        safe_error = str(e).replace('\n', '').replace('\r', '')
        safe_stderr = str(e.stderr).replace('\n', '').replace('\r', '') if e.stderr else 'No error output'
        logger.error(f"Command failed: {safe_error}")
        logger.error(f"Error output: {safe_stderr}")
        raise


def setup_environment(kaggle_mode=False):
    """Setup the environment."""
    logger.info("🔧 Setting up environment...")
    
    if kaggle_mode:
        logger.info("Setting up Kaggle T4 environment...")
        run_command("pip install -r requirements.txt --no-input --quiet", "Installing dependencies")
        run_command("nvidia-smi", "Checking GPU")
    else:
        logger.info("Setting up local environment...")
        run_command("pip install -r requirements.txt", "Installing dependencies")
    
    # Check PyTorch
    try:
        import torch
        cuda_info = torch.version.cuda if torch.cuda.is_available() else "CPU only"
        logger.info(f"✅ PyTorch {torch.__version__} with CUDA {cuda_info}")
    except ImportError:
        logger.error("❌ PyTorch not found! Please install PyTorch first.")
        raise


def update_config(config_file, data_root, output_file="temp_config.yaml"):
    """Update config file with correct data root."""
    # Sanitize data_root for logging
    safe_data_root = str(data_root).replace('\n', '').replace('\r', '')
    logger.info(f"Updating config with data root: {safe_data_root}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        config['data']['root'] = str(data_root)
        
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return output_file
    except (FileNotFoundError, yaml.YAMLError, PermissionError) as e:
        logger.error(f"Error updating config: {e}")
        raise


def analyze_dataset(config_file, output_dir):
    """Run dataset analysis."""
    logger.info("📊 Step 2: Dataset Analysis (Charts 1-5)")
    logger.info("Analyzing your existing dataset structure...")
    
    charts_dir = Path(output_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = f"python scripts/analyze_dataset.py --config {config_file} --out {charts_dir}"
    run_command(cmd, "Analyzing dataset")
    
    logger.info(f"✅ Dataset analysis complete! Check {charts_dir} for initial charts.")


def prep_dataset(config_file, output_dir):
    """Preprocess dataset."""
    logger.info("🔄 Step 3: Data Preprocessing")
    logger.info("Creating tiles and computing class weights...")
    
    tiles_dir = Path(output_dir) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = f"python scripts/prep_dataset.py --config {config_file} --out {tiles_dir}"
    run_command(cmd, "Preprocessing data")
    
    logger.info("✅ Data preprocessing complete!")


def train_model(config_file, tiles_dir, output_dir):
    """Train the model."""
    logger.info("🚂 Step 4: Model Training")
    logger.info("Training segmentation model...")
    
    checkpoints_dir = Path(output_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = f"python scripts/train.py --config {config_file} --tiles {tiles_dir}"
    run_command(cmd, "Training model")
    
    logger.info("✅ Model training complete!")


def evaluate_model(config_file, tiles_dir, output_dir):
    """Evaluate the model."""
    logger.info("📈 Step 5: Model Evaluation (Charts 6-20)")
    logger.info("Evaluating model and generating comprehensive charts...")
    
    charts_dir = Path(output_dir) / "charts"
    checkpoints_dir = Path(output_dir) / "checkpoints"
    best_ckpt = checkpoints_dir / "best.ckpt"
    
    if not best_ckpt.exists():
        logger.warning(f"Best checkpoint not found at {best_ckpt}, looking for alternatives...")
        # Look for any .ckpt file
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            best_ckpt = ckpt_files[0]
            logger.info(f"Using checkpoint: {best_ckpt}")
        else:
            logger.error("No checkpoint files found!")
            raise FileNotFoundError("No model checkpoints found")
    
    cmd = f"python scripts/evaluate.py --config {config_file} --ckpt {best_ckpt} --tiles {tiles_dir} --charts {charts_dir}"
    run_command(cmd, "Evaluating model")
    
    logger.info("✅ Model evaluation complete!")


def generate_predictions(config_file, tiles_dir, output_dir, num_predictions=20):
    """Generate prediction overlays."""
    logger.info("🔮 Step 6: Inference (20 Prediction Overlays)")
    logger.info("Generating red overlay predictions...")
    
    predictions_dir = Path(output_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = Path(output_dir) / "checkpoints"
    best_ckpt = checkpoints_dir / "best.ckpt"
    
    if not best_ckpt.exists():
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            best_ckpt = ckpt_files[0]
        else:
            logger.error("No checkpoint files found!")
            raise FileNotFoundError("No model checkpoints found")
    
    val_images = Path(tiles_dir) / "val" / "images"
    if not val_images.exists():
        # Try test images as fallback
        val_images = Path(tiles_dir) / "test" / "images"
        if not val_images.exists():
            logger.error(f"No validation/test images found in {tiles_dir}")
            raise FileNotFoundError("No validation images found")
    
    cmd = f"python scripts/infer.py --config {config_file} --ckpt {best_ckpt} --images {val_images} --out {predictions_dir} --num {num_predictions}"
    run_command(cmd, "Generating predictions")
    
    logger.info("✅ Inference complete!")


def package_results(output_dir):
    """Package results for download."""
    logger.info("📦 Packaging results for download...")
    
    output_path = Path(output_dir)
    os.chdir(output_path)
    
    # Create zip file
    import zipfile
    with zipfile.ZipFile('slumseg_artifacts.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in ['charts', 'predictions', 'checkpoints']:
            folder_path = Path(folder)
            if folder_path.exists():
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path)
    
    logger.info("✅ Created slumseg_artifacts.zip for download!")


def main():
    parser = argparse.ArgumentParser(description='SlumSeg Complete Pipeline Runner')
    parser.add_argument('--config', default='configs/default.yaml', help='Config file to use')
    parser.add_argument('--data-root', default='e:/Slum-Detection-Using-Unet-Architecture/data', 
                       help='Data root directory')
    parser.add_argument('--output', default='outputs', help='Output directory')
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle T4 optimized settings')
    parser.add_argument('--skip-train', action='store_true', help='Skip training (use existing model)')
    parser.add_argument('--num-predictions', type=int, default=20, help='Number of predictions to generate')
    
    args = parser.parse_args()
    
    # Adjust settings for Kaggle
    if args.kaggle:
        args.config = 'configs/t4_fast.yaml'
        args.data_root = '/kaggle/input/slum-detection'
        args.output = '/kaggle/working/outputs'
    
    print("🏠 SlumSeg: End-to-End Slum Segmentation Pipeline")
    print("==================================================")
    print(f"📍 Configuration:")
    print(f"   Config file: {args.config}")
    print(f"   Data root: {args.data_root}")
    print(f"   Output dir: {args.output}")
    print(f"   Kaggle mode: {args.kaggle}")
    print()
    
    # Create output directories
    output_path = Path(args.output)
    for subdir in ['charts', 'predictions', 'checkpoints', 'tiles']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Environment setup
        setup_environment(args.kaggle)
        print()
        
        # Update config with correct data root
        temp_config = update_config(args.config, args.data_root)
        
        # Step 2: Dataset analysis
        analyze_dataset(temp_config, args.output)
        print()
        
        # Step 3: Data preprocessing
        prep_dataset(temp_config, args.output)
        print()
        
        # Step 4: Model training (optional skip)
        if not args.skip_train:
            train_model(temp_config, output_path / "tiles", args.output)
            print()
        else:
            logger.info("⏭️ Skipping training (using existing model)")
        
        # Step 5: Model evaluation
        evaluate_model(temp_config, output_path / "tiles", args.output)
        print()
        
        # Step 6: Generate predictions
        generate_predictions(temp_config, output_path / "tiles", args.output, args.num_predictions)
        print()
        
        # Step 7: Results summary
        print("🎉 Pipeline Complete!")
        print("====================")
        print()
        print("📊 Generated Outputs:")
        print(f"   • Charts: {output_path / 'charts'}")
        print(f"   • Predictions: {output_path / 'predictions'}")
        print(f"   • Model: {output_path / 'checkpoints'}")
        print()
        
        # Count generated files
        charts_dir = output_path / "charts"
        preds_dir = output_path / "predictions"
        
        if charts_dir.exists():
            chart_count = len(list(charts_dir.glob("*.png")))
            print(f"   Charts generated: {chart_count}")
        
        if preds_dir.exists():
            pred_count = len(list(preds_dir.glob("*.png")))
            print(f"   Predictions generated: {pred_count}")
        
        # Package for Kaggle
        if args.kaggle:
            print()
            package_results(args.output)
        
        print()
        print("🚀 SlumSeg pipeline completed successfully!")
        print("Your dataset has been analyzed, tiled, trained, and evaluated.")
        print("Check the output directories for comprehensive results!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        if os.path.exists("temp_config.yaml"):
            os.remove("temp_config.yaml")


if __name__ == "__main__":
    main()
