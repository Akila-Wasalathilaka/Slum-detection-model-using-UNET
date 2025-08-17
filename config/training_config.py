"""
Training Configuration for Slum Detection
=========================================

Comprehensive training settings including optimization, scheduling,
and experiment management for slum detection models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters and settings.
    """
    
    # Training Parameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizer Settings
    optimizer: str = "adam"  # adam, adamw, sgd
    momentum: float = 0.9    # for SGD
    betas: tuple = (0.9, 0.999)  # for Adam/AdamW
    
    # Learning Rate Scheduling
    scheduler: str = "cosine"  # cosine, plateau, step, exponential
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "T_max": 100,      # for cosine
        "eta_min": 1e-6,   # for cosine
        "patience": 10,    # for plateau
        "factor": 0.5,     # for plateau/step
        "gamma": 0.95      # for exponential
    })
    
    # Loss Function
    loss_type: str = "combined"  # bce, dice, focal, tversky, combined
    loss_params: Dict[str, Any] = field(default_factory=lambda: {
        "bce_weight": 0.5,
        "dice_weight": 0.4,
        "focal_weight": 0.1,
        "focal_alpha": 1.0,
        "focal_gamma": 2.0
    })
    
    # Class Balancing
    class_weights: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "pos_weight": 2.0  # Weight for positive class
    })
    use_weighted_sampling: bool = False
    
    # Regularization
    dropout_rate: float = 0.1
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    
    # Early Stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    monitor_metric: str = "val_dice"  # val_loss, val_iou, val_dice, val_f1
    mode: str = "max"  # max for metrics, min for loss
    
    # Validation
    val_split: float = 0.1  # Fraction of training data for validation
    val_frequency: int = 1  # Validate every N epochs
    
    # Checkpointing
    save_best_only: bool = True
    save_frequency: int = 10  # Save checkpoint every N epochs
    max_checkpoints: int = 5   # Keep only N best checkpoints
    
    # Mixed Precision Training
    use_amp: bool = True  # Automatic Mixed Precision
    grad_clip_norm: Optional[float] = 1.0
    
    # Logging and Monitoring
    log_frequency: int = 10  # Log every N batches
    plot_frequency: int = 5   # Plot results every N epochs
    
    # Experiment Settings
    experiment_name: str = "slum_detection"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    
    # Hardware Settings
    device: str = "auto"  # auto, cuda, cpu
    multi_gpu: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'betas': self.betas,
            'scheduler': self.scheduler,
            'scheduler_params': self.scheduler_params,
            'loss_type': self.loss_type,
            'loss_params': self.loss_params,
            'class_weights': self.class_weights,
            'use_weighted_sampling': self.use_weighted_sampling,
            'dropout_rate': self.dropout_rate,
            'use_mixup': self.use_mixup,
            'mixup_alpha': self.mixup_alpha,
            'use_cutmix': self.use_cutmix,
            'cutmix_alpha': self.cutmix_alpha,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'val_split': self.val_split,
            'val_frequency': self.val_frequency,
            'save_best_only': self.save_best_only,
            'save_frequency': self.save_frequency,
            'max_checkpoints': self.max_checkpoints,
            'use_amp': self.use_amp,
            'grad_clip_norm': self.grad_clip_norm,
            'log_frequency': self.log_frequency,
            'plot_frequency': self.plot_frequency,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'device': self.device,
            'multi_gpu': self.multi_gpu
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined training configurations for different scenarios
PRESET_TRAINING_CONFIGS = {
    "quick_test": TrainingConfig(
        epochs=10,
        batch_size=8,
        learning_rate=1e-3,
        early_stopping=False,
        save_frequency=5,
        experiment_name="quick_test"
    ),
    
    "development": TrainingConfig(
        epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        early_stopping=True,
        patience=10,
        use_amp=False,  # Disable AMP for debugging
        experiment_name="development"
    ),
    
    "production": TrainingConfig(
        epochs=150,
        batch_size=32,
        learning_rate=1e-4,
        early_stopping=True,
        patience=20,
        use_amp=True,
        experiment_name="production"
    ),
    
    "high_precision": TrainingConfig(
        epochs=200,
        batch_size=16,
        learning_rate=5e-5,
        loss_type="tversky",
        loss_params={
            "alpha": 0.3,  # Less penalty for FN
            "beta": 0.7    # More penalty for FP
        },
        early_stopping=True,
        patience=25,
        experiment_name="high_precision"
    ),
    
    "high_recall": TrainingConfig(
        epochs=200,
        batch_size=16,
        learning_rate=5e-5,
        loss_type="tversky",
        loss_params={
            "alpha": 0.7,  # More penalty for FN
            "beta": 0.3    # Less penalty for FP
        },
        early_stopping=True,
        patience=25,
        experiment_name="high_recall"
    ),
    
    "balanced": TrainingConfig(
        epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        loss_type="combined",
        loss_params={
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "focal_weight": 0.0
        },
        early_stopping=True,
        patience=15,
        experiment_name="balanced"
    )
}


def get_training_config(preset: str = "development") -> TrainingConfig:
    """
    Get a predefined training configuration.
    
    Args:
        preset: Configuration preset name
    
    Returns:
        TrainingConfig instance
    """
    if preset not in PRESET_TRAINING_CONFIGS:
        available = list(PRESET_TRAINING_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return PRESET_TRAINING_CONFIGS[preset]


def list_training_presets() -> List[str]:
    """List all available training configuration presets."""
    return list(PRESET_TRAINING_CONFIGS.keys())


def get_optimizer_options() -> Dict[str, Dict[str, Any]]:
    """Get available optimizer options and their parameters."""
    return {
        "adam": {
            "description": "Adaptive Moment Estimation",
            "params": ["learning_rate", "betas", "weight_decay"]
        },
        "adamw": {
            "description": "Adam with Weight Decay",
            "params": ["learning_rate", "betas", "weight_decay"]
        },
        "sgd": {
            "description": "Stochastic Gradient Descent",
            "params": ["learning_rate", "momentum", "weight_decay"]
        }
    }


def get_scheduler_options() -> Dict[str, Dict[str, Any]]:
    """Get available learning rate scheduler options."""
    return {
        "cosine": {
            "description": "Cosine Annealing",
            "params": ["T_max", "eta_min"]
        },
        "plateau": {
            "description": "Reduce on Plateau",
            "params": ["patience", "factor", "min_lr"]
        },
        "step": {
            "description": "Step Decay",
            "params": ["step_size", "gamma"]
        },
        "exponential": {
            "description": "Exponential Decay",
            "params": ["gamma"]
        }
    }


def print_training_config_info():
    """Print information about available training configurations."""
    print("🏋️  TRAINING CONFIGURATION OPTIONS")
    print("=" * 50)
    
    print("\n📋 Available Presets:")
    for name, config in PRESET_TRAINING_CONFIGS.items():
        print(f"  {name.upper()}:")
        print(f"    Epochs: {config.epochs}")
        print(f"    Batch Size: {config.batch_size}")
        print(f"    Learning Rate: {config.learning_rate}")
        print(f"    Loss Type: {config.loss_type}")
        print(f"    Early Stopping: {config.early_stopping}")
        print()
    
    print("⚙️  Available Optimizers:")
    optimizers = get_optimizer_options()
    for name, info in optimizers.items():
        print(f"  {name.upper()}: {info['description']}")
    print()
    
    print("📈 Available Schedulers:")
    schedulers = get_scheduler_options()
    for name, info in schedulers.items():
        print(f"  {name.upper()}: {info['description']}")


if __name__ == "__main__":
    # Test configuration
    print("Testing Training Configuration...")
    
    # Create and test config
    config = get_training_config("development")
    print(f"Loaded config: {config.experiment_name}")
    print(f"Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    
    # Test save/load
    config.save("test_training_config.json")
    loaded_config = TrainingConfig.load("test_training_config.json")
    print(f"Saved and loaded successfully: {loaded_config.optimizer}")
    
    # Clean up
    if os.path.exists("test_training_config.json"):
        os.remove("test_training_config.json")
    
    print("\n")
    print_training_config_info()
