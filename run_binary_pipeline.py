# File: run_binary_pipeline.py
# Complete pipeline runner for binary slum detection

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 characters
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error details:", e.stderr)
        return False

def check_data_exists():
    """Check if the required data directories exist."""
    base_dir = Path(os.getcwd())
    data_dir = base_dir / "data"
    
    required_dirs = [
        data_dir / "train" / "images",
        data_dir / "train" / "masks",
        data_dir / "val" / "images", 
        data_dir / "val" / "masks",
        data_dir / "test" / "images",
        data_dir / "test" / "masks"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        print("❌ Missing required data directories:")
        for missing in missing_dirs:
            print(f"   - {missing}")
        return False
    
    # Check if directories have files
    for set_name in ["train", "val", "test"]:
        img_dir = data_dir / set_name / "images"
        mask_dir = data_dir / set_name / "masks"
        
        img_files = list(img_dir.glob("*.tif"))
        mask_files = list(mask_dir.glob("*.png"))
        
        if len(img_files) == 0:
            print(f"❌ No .tif files found in {img_dir}")
            return False
        
        if len(mask_files) == 0:
            print(f"❌ No .png files found in {mask_dir}")
            return False
        
        print(f"✅ {set_name} set: {len(img_files)} images, {len(mask_files)} masks")
    
    return True

def main():
    """Run the complete binary slum detection pipeline."""
    
    print("🌟 BINARY SLUM DETECTION PIPELINE")
    print("=" * 80)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if data exists
    print("\n📁 Checking data directories...")
    if not check_data_exists():
        print("\n❌ Please ensure your data is properly organized in the 'data' folder")
        print("Expected structure:")
        print("data/")
        print("  train/")
        print("    images/  (*.tif files)")
        print("    masks/   (*.png files)")
        print("  val/")
        print("    images/  (*.tif files)")
        print("    masks/   (*.png files)")
        print("  test/")
        print("    images/  (*.tif files)")
        print("    masks/   (*.png files)")
        return
    
    # Note: Skipping requirements installation - running directly
    print("\n� Note: Skipping package installation - running directly")
    print("   Make sure you have the required packages installed:")
    print("   - torch, torchvision, segmentation-models-pytorch")
    print("   - opencv-python, albumentations, scikit-image") 
    print("   - numpy, pandas, matplotlib, seaborn, scikit-learn")
    
    # Step 1: Run preprocessing
    print("\n🔄 Running data preprocessing...")
    if not run_command("python preprocess_binary.py", "Data preprocessing"):
        print("❌ Preprocessing failed. Cannot continue.")
        return
    
    # Step 2: Run binary slum detection training
    print("\n🧠 Starting binary slum detection training...")
    if not run_command("python binary_slum_detection.py", "Binary slum detection training"):
        print("❌ Training failed.")
        return
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("📊 Check the 'results_binary' folder for:")
    print("   - Training history plots")
    print("   - Model comparison charts")
    print("   - Prediction visualizations")
    print("   - Experiment results (JSON)")
    print("🤖 Check the 'models_binary' folder for trained models")
    print("=" * 80)

if __name__ == "__main__":
    main()
