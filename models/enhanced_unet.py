"""
Enhanced UNet with ASPP, Attention, and Texture Channels
========================================================
Global-ready architecture for domain generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels * 5)
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)
        
    def forward(self, x):
        size = x.shape[-2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.conv5(self.pool(x)), size, mode='bilinear', align_corners=False)
        
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.conv_out(F.relu(self.bn(out)))
        return out

class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class BoundaryHead(nn.Module):
    """Boundary detection head"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        
    def forward(self, x):
        return self.conv(x)

class EnhancedUNet(nn.Module):
    """Enhanced UNet with global generalization features"""
    def __init__(self, encoder_name="resnet34", in_channels=6, classes=1):
        super().__init__()
        
        # Base UNet with texture channels
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,  # RGB + texture channels
            classes=classes
        )
        
        # Get encoder output channels
        encoder_channels = self.unet.encoder.out_channels
        
        # ASPP module
        self.aspp = ASPP(encoder_channels[-1])
        
        # Attention gates
        self.att1 = AttentionGate(256, encoder_channels[-2], 128)
        self.att2 = AttentionGate(128, encoder_channels[-3], 64)
        self.att3 = AttentionGate(64, encoder_channels[-4], 32)
        
        # Boundary head
        self.boundary_head = BoundaryHead(encoder_channels[-3])
        
        # Modify first conv for texture channels
        if in_channels != 3:
            self._modify_first_conv(in_channels)
    
    def _modify_first_conv(self, in_channels):
        """Modify first conv to accept texture channels"""
        first_conv = self.unet.encoder.conv1
        new_conv = nn.Conv2d(
            in_channels, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # Initialize new channels with small random weights
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            if in_channels > 3:
                nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')
        
        self.unet.encoder.conv1 = new_conv
    
    def forward(self, x):
        # Encoder
        features = self.unet.encoder(x)
        
        # ASPP on bottleneck
        features[-1] = self.aspp(features[-1])
        
        # Decoder with attention
        decoder_output = features[-1]
        
        # Apply attention gates to skip connections
        skip_connections = [
            features[-2],
            self.att1(decoder_output, features[-2]),
            self.att2(decoder_output, features[-3]),
            self.att3(decoder_output, features[-4])
        ]
        
        # Decode
        for i, (decoder_block, skip) in enumerate(zip(self.unet.decoder.blocks, skip_connections)):
            decoder_output = decoder_block(decoder_output, skip)
        
        # Main output
        main_output = self.unet.segmentation_head(decoder_output)
        
        # Boundary output
        boundary_output = self.boundary_head(features[-3])
        
        if self.training:
            return {
                'main': main_output,
                'boundary': boundary_output
            }
        else:
            return main_output

def create_enhanced_model(encoder="resnet34", in_channels=6):
    """Create enhanced model"""
    return EnhancedUNet(encoder, in_channels)