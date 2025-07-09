"""
Ultra-accurate model architectures for slum detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from config import config

class SpatialAttentionModule(nn.Module):
    """Spatial attention for enhanced feature focus."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv(x_cat)
        return x * self.sigmoid(x_cat)

class ChannelAttentionModule(nn.Module):
    """Channel attention for feature refinement."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class UltraAttentionGate(nn.Module):
    """Ultra-enhanced attention gate for precise feature selection."""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Additional spatial and channel attention
        self.spatial_att = SpatialAttentionModule()
        self.channel_att = ChannelAttentionModule(F_l)
    
    def forward(self, g, x):
        # Enhanced attention with spatial and channel components
        x_att = self.channel_att(x)
        x_att = self.spatial_att(x_att)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x_att)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x_att * psi

class PyramidPoolingModule(nn.Module):
    """Pyramid pooling for multi-scale context."""
    
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for size in sizes
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(sizes) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            # Handle single batch during evaluation
            if x.size(0) == 1 and self.training:
                # Skip pyramid pooling for single batch in training mode
                continue
            pyramids.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        
        if len(pyramids) == 1:
            # If only original tensor, skip bottleneck
            return x
            
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class UltraAccurateUNet(nn.Module):
    """Ultra-accurate UNet with enhanced attention and multi-scale processing."""
    
    def __init__(self, encoder_name: str = 'timm-efficientnet-b5', encoder_weights: str = 'imagenet'):
        super().__init__()
        
        # Enhanced backbone with pyramid pooling
        self.backbone = smp.UnetPlusPlus(  # UNet++ for better feature reuse
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=config.INPUT_CHANNELS,
            classes=config.OUTPUT_CHANNELS,
            activation=None,
            decoder_attention_type='scse',
            decoder_use_batchnorm=True,
            decoder_channels=(512, 256, 128, 64, 32),  # More channels for detail
        )
        
        # Pyramid pooling module
        self.ppm = PyramidPoolingModule(512, 128)
        
        # Enhanced auxiliary heads for deep supervision
        self.aux_head1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        
        self.aux_head2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        self.aux_head3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Boundary refinement module
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(config.OUTPUT_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced dropout for better generalization
        self.dropout = nn.Dropout2d(p=0.15)
        
        # Multi-scale fusion module
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),  # 3 scales
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Multi-scale input processing
        if hasattr(config, 'MULTI_SCALE_INFERENCE') and config.MULTI_SCALE_INFERENCE and not self.training:
            return self._multi_scale_forward(x)
        
        # Use the backbone directly for now to avoid decoder issues
        main_output = self.backbone(x)
        
        if config.ENABLE_BOUNDARY_REFINEMENT:
            boundary_refined = self.boundary_refine(main_output)
            main_output = main_output + boundary_refined
        
        # Skip deep supervision during training for now
        # if self.training:
        #     aux1 = F.interpolate(self.aux_head1(features[-1]), 
        #                        size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     aux2 = F.interpolate(self.aux_head2(features[-2]), 
        #                        size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     aux3 = F.interpolate(self.aux_head3(features[-3]), 
        #                        size=x.shape[-2:], mode='bilinear', align_corners=False)
        #     return main_output, aux1, aux2, aux3
        
        return main_output
    
    def _multi_scale_forward(self, x):
        """Multi-scale inference for ultra-accuracy."""
        original_size = x.shape[-2:]
        scale_outputs = []
        
        for scale in config.SCALE_FACTORS:
            if scale != 1.0:
                h, w = int(original_size[0] * scale), int(original_size[1] * scale)
                x_scaled = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Forward pass
            features = self.backbone.encoder(x_scaled)
            features[-1] = self.ppm(features[-1])
            decoder_output = self.backbone.decoder(*features)
            output = self.backbone.segmentation_head(decoder_output)
            
            # Boundary refinement
            if config.ENABLE_BOUNDARY_REFINEMENT:
                boundary_refined = self.boundary_refine(output)
                output = output + boundary_refined
            
            # Resize back to original size
            if scale != 1.0:
                output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
            
            scale_outputs.append(output)
        
        # Fuse multi-scale outputs
        if len(scale_outputs) > 1:
            fused_output = torch.stack(scale_outputs, dim=1)
            weights = self.scale_fusion(fused_output)
            fused_output = (fused_output * weights).sum(dim=1, keepdim=True)
            return fused_output
        else:
            return scale_outputs[0]
