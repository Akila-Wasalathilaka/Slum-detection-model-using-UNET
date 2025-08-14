# Google Colab optimized configuration
import os
from config.training_config import TrainingConfig
from config.data_config import DataConfig

# Colab-specific paths
COLAB_DRIVE_PATH = "/content/drive/MyDrive/data"
COLAB_WORKING = "/content"

def get_colab_training_config():
    """Optimized training config for Google Colab."""
    config = TrainingConfig()
    
    # Colab optimizations
    config.batch_size = 16  # Reduce if memory issues
    config.num_workers = 2  # Lower for Colab
    config.pin_memory = True  # Enable for Colab
    config.epochs = 20  # Reasonable for Colab time limits
    config.use_amp = True  # Enable mixed precision
    
    return config

def get_colab_data_config():
    """Data config for Colab environment."""
    config = DataConfig()
    
    # Update paths for Colab
    if os.path.exists(COLAB_DRIVE_PATH):
        config.data_root = COLAB_DRIVE_PATH
    
    return config