"""
Update README Images Script
==========================

This script automatically copies the latest analysis results to the images/ folder
for README documentation. Run this after generating new analysis results.
"""

import os
import shutil
import glob
from pathlib import Path

def update_readme_images():
    """Update images folder with latest analysis results."""
    
    print("🖼️  UPDATING README IMAGES")
    print("=" * 30)
    
    # Find latest analysis directory
    analysis_dirs = glob.glob("charts/analysis_*")
    if not analysis_dirs:
        print("❌ No analysis directories found! Please run analysis first.")
        return
    
    latest_analysis = max(analysis_dirs, key=os.path.getctime)
    print(f"📁 Latest analysis: {latest_analysis}")
    
    # Create images directory if it doesn't exist
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    # Image mappings: source -> destination
    image_mappings = {
        f"{latest_analysis}/predictions/prediction_samples.png": "images/prediction_samples.png",
        f"{latest_analysis}/confusion_matrices/confusion_matrix_t0.30.png": "images/confusion_matrix.png", 
        f"{latest_analysis}/roc_curves/roc_curve.png": "images/roc_curve.png",
        f"{latest_analysis}/performance_metrics/performance_summary.png": "images/performance_summary.png",
        f"{latest_analysis}/threshold_analysis/combined_metrics.png": "images/threshold_analysis.png"
    }
    
    # Copy images
    copied_count = 0
    for source, dest in image_mappings.items():
        if os.path.exists(source):
            shutil.copy2(source, dest)
            print(f"✅ Copied: {os.path.basename(dest)}")
            copied_count += 1
        else:
            print(f"⚠️  Missing: {source}")
    
    print(f"\n🎉 Updated {copied_count}/{len(image_mappings)} images for README!")
    print("📝 Images are now ready for documentation.")

if __name__ == "__main__":
    update_readme_images()
