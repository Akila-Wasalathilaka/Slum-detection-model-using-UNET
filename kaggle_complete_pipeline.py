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
    print("🏘️ KAGGLE SLUM DETECTION TRAINING PIPELINE")
    print("=" * 70)
    print("Enhanced training with water discrimination!")
    print("1. Setup and install dependencies")
    print("2. Dataset analysis") 
    print("3. Enhanced model training")
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
    
    # Step 3: Enhanced training
    print("\\n⚠️ Starting enhanced training - this may take 1-3 hours depending on hardware")
    print("Features: Water discrimination, attention mechanisms, boundary loss")
    if not run_command("python advanced_training.py", 
                      "Training enhanced slum detection model"):
        return
    

    
    # Final summary
    print("\\n" + "="*70)
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print("="*70)
    print()
    print("📁 Generated Files:")
    print("├── analysis/")
    print("│   ├── class_analysis.png")
    print("│   └── class_analysis_results.json")
    print("├── best_advanced_slum_model.pth")
    print("├── advanced_training_history.png")
    print("└── advanced_training_history.json")
    print()
    print("🏆 Your enhanced slum detection model is ready!")
    print("🌊 Water discrimination: Applied to reduce misclassification")
    print()
    print("📋 Next Steps:")
    print("Run the analysis pipeline to generate charts and predictions:")
    print("!python kaggle_analysis_pipeline.py")

if __name__ == "__main__":
    main()