"""
Global Domain Generalization Transforms
======================================
Aggressive augmentations for worldwide generalization
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import filters, feature, morphology

class TextureChannels:
    """Compute texture channels from RGB"""
    def __init__(self):
        pass
    
    def __call__(self, image, **kwargs):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
        
        # Local entropy
        entropy = filters.rank.entropy(gray, morphology.disk(3))
        entropy = (entropy / entropy.max() * 255).astype(np.uint8)
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = (np.abs(laplacian) / np.abs(laplacian).max() * 255).astype(np.uint8)
        
        # Stack with original RGB
        if len(image.shape) == 3:
            result = np.concatenate([image, grad_mag[..., None], entropy[..., None], laplacian[..., None]], axis=2)
        else:
            result = np.stack([image, grad_mag, entropy, laplacian], axis=2)
        
        return result

class GlobalAugmentation:
    """Aggressive augmentation for global generalization"""
    def __init__(self, image_size=120):
        self.transform = A.Compose([
            # Geometric - extreme variations
            A.RandomScale(scale_limit=0.5, p=0.7),  # 0.5x to 2.0x
            A.Rotate(limit=45, p=0.7),
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.GridDistortion(num_steps=8, distort_limit=0.3, p=0.3),
            A.Perspective(scale=(0.05, 0.15), p=0.3),
            
            # Color - simulate different regions/sensors
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8),
            A.RandomGamma(gamma_limit=(50, 150), p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.ChannelShuffle(p=0.2),
            
            # Weather/atmospheric
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                           num_flare_circles_lower=1, num_flare_circles_upper=3, p=0.1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.3),
            
            # Sensor/compression artifacts
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            
            # Cutout/occlusion
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Cutout(num_holes=5, max_h_size=12, max_w_size=12, p=0.2),
            
            # Final resize and normalize
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image, mask=None):
        if mask is not None:
            result = self.transform(image=image, mask=mask)
            return result['image'], result['mask']
        else:
            result = self.transform(image=image)
            return result['image']

class CLAHEPreprocessing:
    """CLAHE preprocessing for better contrast"""
    def __init__(self, clip_limit=2.0):
        self.clip_limit = clip_limit
    
    def __call__(self, image, **kwargs):
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result

class GrayWorldNormalization:
    """Gray world color constancy"""
    def __call__(self, image, **kwargs):
        image = image.astype(np.float32)
        
        # Compute channel means
        mean_r = np.mean(image[:,:,0])
        mean_g = np.mean(image[:,:,1])
        mean_b = np.mean(image[:,:,2])
        
        # Gray world assumption
        gray_mean = (mean_r + mean_g + mean_b) / 3
        
        # Normalize
        if mean_r > 0:
            image[:,:,0] *= gray_mean / mean_r
        if mean_g > 0:
            image[:,:,1] *= gray_mean / mean_g
        if mean_b > 0:
            image[:,:,2] *= gray_mean / mean_b
        
        return np.clip(image, 0, 255).astype(np.uint8)

def get_global_train_transforms(image_size=120):
    """Get training transforms for global generalization"""
    return A.Compose([
        CLAHEPreprocessing(),
        GrayWorldNormalization(),
        TextureChannels(),
        GlobalAugmentation(image_size),
        ToTensorV2()
    ])

def get_global_val_transforms(image_size=120):
    """Get validation transforms"""
    return A.Compose([
        CLAHEPreprocessing(),
        TextureChannels(),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5], 
                   std=[0.229, 0.224, 0.225, 0.25, 0.25, 0.25]),
        ToTensorV2()
    ])