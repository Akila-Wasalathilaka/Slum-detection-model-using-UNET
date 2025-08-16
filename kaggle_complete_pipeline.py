#!/usr/bin/env python3
"""
Complete Kaggle Pipeline for Slum Detection
==========================================
One-click solution to run the entire pipeline in Kaggle.
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run command with progress tracking"""
    print(f"\\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"\\n✅ {description} completed successfully in {end_time - start_time:.1f}s")
    else:
        print(f"\\n❌ {description} failed!")
        return False
    
    return True

def main():
    """Run complete pipeline"""
    print("🏘️ KAGGLE SLUM DETECTION COMPLETE PIPELINE")
    print("=" * 70)
    print("This will run the complete slum detection pipeline:")
    print("1. Setup and install dependencies")
    print("2. Comprehensive dataset analysis") 
    print("3. Advanced model training")
    print("4. Generate comprehensive charts")
    print("5. Create 25+ diverse predictions")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("data"):
        print("❌ Data directory not found!")
        print("Please ensure you're in the correct directory with the data folder.")
        return
    
    # Step 1: Install dependencies
    if not run_command("pip install albumentations segmentation-models-pytorch timm efficientnet-pytorch", 
                      "Installing dependencies"):
        return
    
    # Step 2: Dataset analysis
    if not run_command("python quick_analysis.py", 
                      "Running comprehensive dataset analysis"):
        return
    
    # Step 3: Advanced training
    print("\\n⚠️ Starting advanced training - this may take 1-3 hours depending on hardware")
    if not run_command("python advanced_training.py", 
                      "Training advanced slum detection model"):
        return
    
    # Step 4: Generate charts
    if not run_command("python create_charts.py --model best_advanced_slum_model.pth", 
                      "Generating comprehensive analysis charts"):
        return
    
    # Step 5: Generate predictions
    if not run_command("python make_predictions.py --model best_advanced_slum_model.pth --num 25", 
                      "Generating 25+ diverse predictions"):
        return
    
    # Final summary
    print("\\n" + "="*70)
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)
    print()
    print("📁 Generated Files:")
    print("├── analysis/")
    print("│   ├── class_analysis.png")
    print("│   ├── sample_images.png") 
    print("│   └── class_analysis_results.json")
    print("├── charts/")
    print("│   ├── dataset_overview.png")
    print("│   ├── model_performance.png")
    print("│   └── training_analysis.png")
    print("├── predictions/")
    print("│   ├── prediction_01_*.png (25+ files)")
    print("│   ├── prediction_summary.png")
    print("│   └── predictions_data.json")
    print("├── best_advanced_slum_model.pth")
    print("├── advanced_training_history.png")
    print("└── advanced_training_history.json")
    print()
    print("🏆 Your advanced slum detection model is ready!")
    print("📊 Check the charts/ directory for detailed analysis")
    print("🎯 Check the predictions/ directory for model predictions")

if __name__ == "__main__":
    main()