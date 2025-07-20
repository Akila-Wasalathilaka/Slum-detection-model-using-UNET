"""
Test what configs are being loaded and if they match the training config.
"""

from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.data_config import get_data_config

def compare_configs():
    """Compare the loaded config with the actual training config."""
    
    # Load the actual training config
    training_config_path = project_root / "experiments/development_20250713_175410/configs/data_config.json"
    with open(training_config_path, 'r') as f:
        training_config = json.load(f)
    
    # Load the config that Colombo script is using
    current_config = get_data_config('standard')
    
    print("🔍 Comparing configs...")
    print(f"\nTraining config normalize: {training_config.get('normalize')}")
    print(f"Current config normalize: {current_config.normalize}")
    
    print(f"\nTraining config mean: {training_config.get('mean')}")
    print(f"Current config mean: {current_config.mean}")
    
    print(f"\nTraining config std: {training_config.get('std')}")
    print(f"Current config std: {current_config.std}")
    
    print(f"\nTraining config use_tile_masks_only: {training_config.get('use_tile_masks_only')}")
    print(f"Current config use_tile_masks_only: {current_config.use_tile_masks_only}")
    
    # Check if they match
    matches = (
        training_config.get('normalize') == current_config.normalize and
        training_config.get('mean') == current_config.mean and
        training_config.get('std') == current_config.std
    )
    
    if matches:
        print("\n✅ Configs match - normalization should work correctly")
    else:
        print("\n❌ Configs don't match - this is the problem!")
        print("\nTo fix Colombo detection, update the data config to match training config:")
        print(f"   normalize: {training_config.get('normalize')}")
        print(f"   mean: {training_config.get('mean')}")
        print(f"   std: {training_config.get('std')}")

if __name__ == "__main__":
    compare_configs()
