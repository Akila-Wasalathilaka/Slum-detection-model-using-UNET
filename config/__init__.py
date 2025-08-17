"""
Configuration Package for Slum Detection
========================================

Centralized configuration management for all model parameters,
training settings, and data preprocessing options.
"""

from .model_config import *
from .training_config import *
from .data_config import *

__all__ = [
    'ModelConfig', 'TrainingConfig', 'DataConfig',
    'get_config', 'save_config', 'load_config'
]
