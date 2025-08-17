#!/usr/bin/env python3
"""
Kaggle Analysis Pipeline for Slum Detection
==========================================
Generates charts and predictions after training.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run shell command and print output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def main():
    print("📊 KAGGLE SLUM DETECTION ANALYSIS PIPELINE")
    print("=" * 70)
    print("Generate charts and predictions from trained model!")
    print("1. Generate comprehensive charts")
    print("2. Create enhanced predictions with analysis")
    print()
    
    # Check if model exists
    if not os.path.exists("best_advanced_slum_model.pth"):
        print("❌ Trained model not found!")
        print("Please run the training pipeline first:")
        print("!python kaggle_complete_pipeline.py")
        return
    
    # Step 1: Generate charts
    print("\n" + "="*60)
    print("🚀 Generating comprehensive analysis charts")
    print("="*60)
    if not run_command("python create_charts.py --model best_advanced_slum_model.pth"):
        print("⚠️ Chart generation failed, but continuing...")
    
    # Step 2: Generate enhanced predictions
    print("\n" + "="*60)
    print("🚀 Generating 25+ enhanced predictions with post-processing")
    print("="*60)
    if not run_command("python make_predictions.py --model best_advanced_slum_model.pth --num 25"):
        print("⚠️ Prediction generation failed...")
        return
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ANALYSIS PIPELINE COMPLETED!")
    print("="*70)
    print()
    print("📁 Generated Files:")
    print("├── charts/")
    print("│   ├── dataset_overview.png")
    print("│   ├── model_performance.png")
    print("│   └── training_analysis.png")
    print("├── predictions/")
    print("│   ├── prediction_01_*.png (25+ files)")
    print("│   ├── prediction_summary.png")
    print("│   └── predictions_data.json")
    print()
    print("📊 Check the charts/ directory for detailed analysis")
    print("🎯 Check the predictions/ directory for enhanced predictions")
    print("🌊 Water discrimination: Applied in predictions")
    print("⚡ Enhanced features: TTA, post-processing, quality analysis")

if __name__ == "__main__":
    main()