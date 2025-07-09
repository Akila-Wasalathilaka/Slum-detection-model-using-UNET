#!/usr/bin/env python3
# File: run_production_pipeline.py
# Production pipeline runner for improved slum detection

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'segmentation-models-pytorch',
        'albumentations', 'opencv-python', 'scikit-learn',
        'matplotlib', 'seaborn', 'numpy', 'pandas', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_data_structure():
    """Check if data structure is correct."""
    base_dir = Path(os.getcwd())
    required_dirs = [
        base_dir / "data_preprocessed" / "train" / "images",
        base_dir / "data_preprocessed" / "train" / "masks",
        base_dir / "data_preprocessed" / "val" / "images",
        base_dir / "data_preprocessed" / "val" / "masks",
        base_dir / "data_preprocessed" / "test" / "images",
        base_dir / "data_preprocessed" / "test" / "masks"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists() or len(list(dir_path.glob("*"))) == 0:
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        print("❌ Missing or empty data directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        print("\nPlease ensure your data is properly organized and preprocessed.")
        return False
    
    print("✅ Data structure is correct!")
    return True

def run_preprocessing():
    """Run data preprocessing if needed."""
    print("\n" + "="*60)
    print("🔄 RUNNING DATA PREPROCESSING")
    print("="*60)
    
    try:
        result = subprocess.run([
            sys.executable, "preprocess_binary.py"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Data preprocessing completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Data preprocessing failed: {e}")
        if e.stdout:
            print("Output:", e.stdout[-500:])
        if e.stderr:
            print("Error:", e.stderr[-500:])
        return False

def run_training():
    """Run the production training pipeline."""
    print("\n" + "="*60)
    print("🚀 STARTING PRODUCTION TRAINING")
    print("="*60)
    
    try:
        # Import and run training
        from improved_slum_detection_production import main_training
        main_training()
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_inference():
    """Run inference on test data."""
    print("\n" + "="*60)
    print("🔍 RUNNING INFERENCE")
    print("="*60)
    
    try:
        from improved_slum_detection_production import main_inference
        main_inference()
        print("✅ Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_evaluation():
    """Run model evaluation and generate reports."""
    print("\n" + "="*60)
    print("📊 RUNNING EVALUATION")
    print("="*60)
    
    try:
        # Create evaluation script inline
        eval_code = '''
import torch
import numpy as np
from pathlib import Path
from improved_slum_detection_production import ProductionInference, ProductionConfig
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

config = ProductionConfig()

# Load test data
test_images = list(config.TEST_IMG_DIR.glob("*.tif"))
test_masks = list(config.TEST_MASK_DIR.glob("*.png"))

if len(test_images) == 0:
    print("No test images found!")
    exit(1)

# Initialize inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = config.MODEL_DIR / 'best_production_model.pth'

if not model_path.exists():
    print("No trained model found!")
    exit(1)

inference = ProductionInference(str(model_path), device, use_tta=True)

# Evaluate on test set
all_true = []
all_pred = []

print(f"Evaluating on {len(test_images)} test images...")

for i, img_path in enumerate(test_images[:20]):  # Evaluate on first 20 images
    # Load ground truth
    mask_path = config.TEST_MASK_DIR / f"{img_path.stem}.png"
    if not mask_path.exists():
        continue
    
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt_binary = (gt_mask == config.SLUM_CLASS_ID).astype(np.uint8)
    
    # Get prediction
    _, pred_binary = inference.predict_single(str(img_path), threshold='balanced')
    
    # Flatten and add to lists
    all_true.extend(gt_binary.flatten())
    all_pred.extend(pred_binary.flatten())

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

accuracy = accuracy_score(all_true, all_pred)
precision = precision_score(all_true, all_pred, zero_division=0)
recall = recall_score(all_true, all_pred, zero_division=0)
f1 = f1_score(all_true, all_pred, zero_division=0)
iou = jaccard_score(all_true, all_pred, zero_division=0)

print(f"\\nProduction Model Evaluation Results:")
print(f"=====================================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")

# Save results
results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'iou': float(iou)
}

import json
with open(config.RESULTS_DIR / 'production_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\nResults saved to {config.RESULTS_DIR / 'production_evaluation.json'}")
'''
        
        # Execute evaluation
        exec(eval_code)
        print("✅ Evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the complete production pipeline."""
    parser = argparse.ArgumentParser(description='Production Slum Detection Pipeline')
    parser.add_argument('--mode', choices=['full', 'train', 'inference', 'eval'], 
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--skip-checks', action='store_true', 
                       help='Skip requirement and data checks')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing step')
    
    args = parser.parse_args()
    
    print("🏭 PRODUCTION SLUM DETECTION PIPELINE")
    print("="*50)
    
    # Run checks unless skipped
    if not args.skip_checks:
        print("🔍 Checking requirements and data structure...")
        
        if not check_requirements():
            return False
            
        if not check_data_structure():
            if not args.skip_preprocessing:
                print("🔄 Attempting to run preprocessing...")
                if not run_preprocessing():
                    return False
            else:
                print("❌ Data structure issues detected. Please fix manually.")
                return False
    
    success = True
    
    if args.mode in ['full', 'train']:
        success &= run_training()
    
    if args.mode in ['full', 'inference'] and success:
        success &= run_inference()
    
    if args.mode in ['full', 'eval'] and success:
        success &= run_evaluation()
    
    if success:
        print("\n" + "="*60)
        print("🎉 PRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check the following directories for results:")
        print(f"   - Models: models_production/")
        print(f"   - Results: results_production/")
        print(f"   - Inference: results_production/inference_results/")
    else:
        print("\n" + "="*60)
        print("❌ PIPELINE FAILED")
        print("="*60)
        print("Please check the error messages above and fix the issues.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
