"""
Stable UNet wrapper with 6-channel input
=======================================
Uses segmentation_models_pytorch.Unet with minimal customizations to ensure
compatibility and reliability in Colab and local environments.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class EnhancedUNet(nn.Module):
    """Thin wrapper around SMP Unet that accepts 6-channel input."""
    def __init__(self, encoder_name: str = "resnet34", in_channels: int = 6, classes: int = 1):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )

        # If encoder was initialized with 3 channels weights, adapt first conv for >3 channels
        if in_channels != 3:
            self._modify_first_conv(in_channels)

    def _modify_first_conv(self, in_channels: int):
        first_conv = self.unet.encoder.conv1
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            # Copy RGB weights; init extra channels
            new_conv.weight[:, :3] = first_conv.weight
            if in_channels > 3:
                nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu")
        self.unet.encoder.conv1 = new_conv

    def forward(self, x):
        return self.unet(x)

def create_enhanced_model(encoder: str = "resnet34", in_channels: int = 6):
    """Factory for the stable enhanced UNet."""
    return EnhancedUNet(encoder_name=encoder, in_channels=in_channels)