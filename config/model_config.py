"""
Model Configuration for Slum Detection
======================================

Centralized configuration for model architectures, hyperparameters,
and training settings optimized for slum detection.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class ModelConfig:
    """
    Configuration class for model architecture and hyperparameters.
    """
    
    # Model Architecture
    architecture: str = "unet"  # unet, unet++, deeplabv3+
    encoder: str = "resnet34"   # resnet34, efficientnet-b0, etc.
    pretrained: bool = True     # Use ImageNet pretrained weights
    
    # Model Parameters
    in_channels: int = 3        # RGB input channels
    num_classes: int = 1        # Binary classification
    activation: Optional[str] = None  # Output activation (None for logits)
    
    # Advanced Features
    attention_type: Optional[str] = None  # scse, cbam, etc.
    deep_supervision: bool = False
    aux_params: Optional[Dict[str, Any]] = None
    
    # Input/Output
    input_size: tuple = (120, 120)  # Height, Width
    output_stride: int = 1          # Output downsampling factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'architecture': self.architecture,
            'encoder': self.encoder,
            'pretrained': self.pretrained,
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'activation': self.activation,
            'attention_type': self.attention_type,
            'deep_supervision': self.deep_supervision,
            'aux_params': self.aux_params,
            'input_size': self.input_size,
            'output_stride': self.output_stride
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined model configurations
PRESET_CONFIGS = {
    "fast": ModelConfig(
        architecture="unet",
        encoder="mobilenet_v2",
        pretrained=True,
        input_size=(120, 120)
    ),
    
    "balanced": ModelConfig(
        architecture="unet",
        encoder="resnet34",
        pretrained=True,
        input_size=(120, 120)
    ),
    
    "accurate": ModelConfig(
        architecture="unet++",
        encoder="efficientnet-b2",
        pretrained=True,
        input_size=(120, 120)
    ),
    
    "lightweight": ModelConfig(
        architecture="unet",
        encoder="efficientnet-b0",
        pretrained=True,
        input_size=(120, 120)
    ),
    
    "high_res": ModelConfig(
        architecture="deeplabv3+",
        encoder="resnet50",
        pretrained=True,
        input_size=(120, 120)
    )
}


def get_model_config(preset: str = "balanced") -> ModelConfig:
    """
    Get a predefined model configuration.
    
    Args:
        preset: Configuration preset name
    
    Returns:
        ModelConfig instance
    """
    if preset not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return PRESET_CONFIGS[preset]


def list_model_presets() -> List[str]:
    """List all available model configuration presets."""
    return list(PRESET_CONFIGS.keys())


def get_encoder_options() -> Dict[str, List[str]]:
    """Get available encoder options by category."""
    return {
        "resnet": [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ],
        "efficientnet": [
            "efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
            "efficientnet-b3", "efficientnet-b4", "efficientnet-b5"
        ],
        "mobilenet": [
            "mobilenet_v2"
        ],
        "densenet": [
            "densenet121", "densenet161", "densenet169", "densenet201"
        ],
        "resnext": [
            "resnext50_32x4d", "resnext101_32x8d"
        ],
        "se_models": [
            "se_resnet50", "se_resnet101", "se_resnext50_32x4d"
        ]
    }


def print_config_info():
    """Print information about available configurations."""
    print("🏗️  MODEL CONFIGURATION OPTIONS")
    print("=" * 50)
    
    print("\n📋 Available Presets:")
    for name, config in PRESET_CONFIGS.items():
        print(f"  {name.upper()}:")
        print(f"    Architecture: {config.architecture}")
        print(f"    Encoder: {config.encoder}")
        print(f"    Input Size: {config.input_size}")
        print()
    
    print("🔧 Available Encoders:")
    encoders = get_encoder_options()
    for category, encoder_list in encoders.items():
        print(f"  {category.upper()}: {', '.join(encoder_list)}")
    print()
    
    print("🏛️  Available Architectures:")
    print("  - unet: Standard U-Net architecture")
    print("  - unet++: Nested U-Net with skip connections")
    print("  - deeplabv3+: DeepLab with atrous convolutions")


if __name__ == "__main__":
    # Test configuration
    print("Testing Model Configuration...")
    
    # Create and test config
    config = get_model_config("balanced")
    print(f"Loaded config: {config.architecture} + {config.encoder}")
    
    # Test save/load
    config.save("test_config.json")
    loaded_config = ModelConfig.load("test_config.json")
    print(f"Saved and loaded successfully: {loaded_config.encoder}")
    
    # Clean up
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("\n")
    print_config_info()
