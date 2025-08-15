"""
Stable UNet wrapper with 6-channel input
=======================================
Uses segmentation_models_pytorch.Unet with minimal customizations to ensure
compatibility and reliability across environments.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class EnhancedUNet(nn.Module):
    """Thin wrapper around SMP Unet that accepts arbitrary input channels (default 6).

    When using pretrained encoder weights (imagenet), we first build a 3-channel model
    to load weights, then replace the first conv to match the requested in_channels.
    This avoids shape mismatches and preserves pretrained RGB filters.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 6,
        classes: int = 1,
        encoder_weights: str | None = "imagenet",
    ):
        super().__init__()

        # If requesting non-3 input channels with pretrained weights, construct a 3-channel model first
        base_in_channels = 3 if (in_channels != 3 and encoder_weights) else in_channels

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=base_in_channels,
            classes=classes,
        )

        # If the built model's first conv doesn't match requested in_channels, adapt it
        if base_in_channels != in_channels:
            self._modify_first_conv(in_channels)

    def _modify_first_conv(self, in_channels: int):
        # Locate the first convolution layer of the encoder (works for ResNet family)
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
            old_in = first_conv.weight.shape[1]

            if in_channels <= old_in:
                # Truncate to match requested channels
                new_conv.weight.copy_(first_conv.weight[:, :in_channels])
            else:
                # Copy pretrained RGB (or existing) channels and initialize the rest from RGB mean
                new_conv.weight[:, :old_in].copy_(first_conv.weight)

                # Use mean over available channels to initialize the new ones for stability
                mean_rgb = first_conv.weight[:, :min(3, old_in)].mean(dim=1, keepdim=True)
                repeat_count = in_channels - old_in
                new_conv.weight[:, old_in:].copy_(mean_rgb.expand(-1, repeat_count, -1, -1))

            # Copy bias if present
            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        self.unet.encoder.conv1 = new_conv

    def forward(self, x):
        return self.unet(x)

def create_enhanced_model(encoder: str = "resnet34", in_channels: int = 6):
    """Factory for the stable enhanced UNet."""
    return EnhancedUNet(encoder_name=encoder, in_channels=in_channels)