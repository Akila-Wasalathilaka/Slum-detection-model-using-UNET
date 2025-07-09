"""
Utility functions for production slum detection pipeline.
"""
import random
import numpy as np
import torch
import logging
import warnings
from pathlib import Path

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def suppress_warnings():
    """Suppress unnecessary warnings."""
    warnings.filterwarnings("ignore")
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    logging.getLogger().setLevel(logging.ERROR)

def setup_device():
    """Setup and configure device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def create_directories(config):
    """Create necessary directories."""
    for dir_path in [config.RESULTS_DIR, config.MODEL_DIR, config.CHECKPOINT_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)

def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def save_experiment_info(config, model, save_path):
    """Save experiment configuration and model info."""
    import json
    
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'config': {
            'image_size': config.PRIMARY_SIZE,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'loss_weights': config.LOSS_WEIGHTS,
            'encoders': config.ENCODERS,
            'encoder_weights': config.ENCODER_WEIGHTS
        },
        'model': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'ProductionUNet'
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
