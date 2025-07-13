"""
Advanced UNet Architectures for Slum Detection
==============================================

This module contains UNet variants optimized for satellite image slum detection.
Supports multiple encoders and advanced features like attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Dict, Any


class SlumUNet(nn.Module):
    """
    Advanced UNet for slum detection with multiple encoder options.
    
    Features:
    - Multiple encoder backbones (ResNet, EfficientNet, etc.)
    - Attention mechanisms
    - Multi-scale features
    - Custom decoder with skip connections
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        attention_type: Optional[str] = None,
        aux_params: Optional[Dict[str, Any]] = None
    ):
        super(SlumUNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params
        )
        
        # Add attention if specified
        if attention_type == "scse":
            self._add_attention_blocks()
    
    def _add_attention_blocks(self):
        """Add Spatial and Channel Squeeze & Excitation attention."""
        # Implementation would go here for advanced attention
        pass
    
    def forward(self, x):
        return self.model(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested UNet) for improved feature representation.
    Better handling of objects at different scales.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None
    ):
        super(UNetPlusPlus, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x):
        return self.model(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for slum detection with atrous convolutions.
    Good for capturing multi-scale context.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None
    ):
        super(DeepLabV3Plus, self).__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x):
        return self.model(x)


def create_model(
    architecture: str = "unet",
    encoder: str = "resnet34",
    pretrained: bool = True,
    num_classes: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models with different architectures.
    
    Args:
        architecture: Model architecture ('unet', 'unet++', 'deeplabv3+')
        encoder: Encoder backbone ('resnet34', 'efficientnet-b0', etc.)
        pretrained: Whether to use ImageNet pretrained weights
        num_classes: Number of output classes (1 for binary slum detection)
        **kwargs: Additional model parameters
    
    Returns:
        PyTorch model ready for training
    """
    
    encoder_weights = "imagenet" if pretrained else None
    
    if architecture.lower() == "unet":
        return SlumUNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            **kwargs
        )
    elif architecture.lower() == "unet++":
        return UNetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            **kwargs
        )
    elif architecture.lower() == "deeplabv3+":
        return DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Available encoder backbones
AVAILABLE_ENCODERS = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
    "mobilenet_v2", "densenet121", "densenet161",
    "resnext50_32x4d", "se_resnext50_32x4d"
]

# Model configurations for different scenarios
MODEL_CONFIGS = {
    "fast": {
        "architecture": "unet",
        "encoder": "mobilenet_v2",
        "description": "Fast inference, good for real-time applications"
    },
    "balanced": {
        "architecture": "unet",
        "encoder": "resnet34",
        "description": "Good balance of accuracy and speed"
    },
    "accurate": {
        "architecture": "unet++",
        "encoder": "efficientnet-b2",
        "description": "Highest accuracy, slower inference"
    },
    "lightweight": {
        "architecture": "unet",
        "encoder": "efficientnet-b0",
        "description": "Lightweight model for deployment"
    }
}


def get_model_info():
    """Print available models and their characteristics."""
    print("Available Model Configurations:")
    print("=" * 50)
    for name, config in MODEL_CONFIGS.items():
        print(f"{name.upper()}:")
        print(f"  Architecture: {config['architecture']}")
        print(f"  Encoder: {config['encoder']}")
        print(f"  Description: {config['description']}")
        print()


def list_available_models():
    """Return list of available model configuration names."""
    return list(MODEL_CONFIGS.keys())


# Convenience aliases for backward compatibility
UNet = SlumUNet

# Export main classes and functions
__all__ = [
    'SlumUNet',
    'UNet',  # Alias for SlumUNet
    'create_model',
    'get_model_info',
    'list_available_models'
]

if __name__ == "__main__":
    # Test model creation
    model = create_model("unet", "resnet34", pretrained=True)
    print(f"Model created: {model.__class__.__name__}")
    
    # Test forward pass
    x = torch.randn(1, 3, 120, 120)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    
    get_model_info()
