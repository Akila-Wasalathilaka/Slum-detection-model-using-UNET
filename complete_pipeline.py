"""
Complete Pipeline for World-Class Slum Detection
===============================================
From dataset analysis to deployment
"""

import subprocess
import sys
from pathlib import Path

def run_complete_pipeline():
    """Run the complete pipeline from scratch."""
    
    print("🌍 WORLD-CLASS SLUM DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Analyze dataset
    print("\n📊 Step 1: Analyzing dataset...")
    try:
        from dataset_analyzer import SlumDatasetAnalyzer
        analyzer = SlumDatasetAnalyzer()
        analysis_results = analyzer.analyze_complete_dataset()
        print("✅ Dataset analysis complete")
    except Exception as e:
        print(f"⚠️ Dataset analysis failed: {e}")
    
    # Step 2: Train world-class model
    print("\n🏋️ Step 2: Training world-class model...")
    try:
        from comprehensive_trainer import ComprehensiveTrainer
        trainer = ComprehensiveTrainer()
        best_iou = trainer.train(epochs=30, batch_size=4)
        print(f"✅ Training complete! Best IoU: {best_iou:.4f}")
    except Exception as e:
        print(f"⚠️ Training failed: {e}")
        print("Continuing with pre-trained weights...")
    
    # Step 3: Launch interface
    print("\n🚀 Step 3: Launching world-class interface...")
    try:
        from world_class_interface import create_world_class_app
        app = create_world_class_app()
        app.launch(share=True, server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"❌ Interface launch failed: {e}")

if __name__ == "__main__":
    run_complete_pipeline()