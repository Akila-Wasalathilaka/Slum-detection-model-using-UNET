#!/usr/bin/env python3
"""
Global Slum Detection System
===========================
Domain-generalized slum detection that works anywhere on Earth without new labels.
Implements TTA, TENT adaptation, adaptive thresholding, and robust post-processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import albumentations as A
from skimage import filters, morphology, measure
from scipy import ndimage
import matplotlib.pyplot as plt
import copy

class GlobalSlumDetector:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        self.model = self._load_model(checkpoint_path)
        self.ema_model = self._create_ema_model()
        
    def _load_model(self, checkpoint_path: str):
        """Load model with enhanced architecture"""
        from models.enhanced_unet import create_enhanced_model
        model = create_enhanced_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def _create_ema_model(self):
        """Create EMA version for stable inference by deep-copying the loaded model"""
        ema_model = copy.deepcopy(self.model)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        return ema_model
    
    def predict_global(self, image_path: str, use_tta: bool = True, use_tent: bool = True, 
                      adaptive_threshold: bool = True) -> Dict:
        """Global prediction with all enhancements"""
        # Load and preprocess
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Multi-scale sliding window prediction
        predictions = self._multi_scale_predict(image, use_tta, use_tent)
        
        # Adaptive thresholding
        if adaptive_threshold:
            threshold = self._compute_adaptive_threshold(predictions['probability'])
        else:
            threshold = 0.3
            
        binary_mask = (predictions['probability'] > threshold).astype(np.uint8)
        
        # Post-processing
        cleaned_mask = self._post_process_mask(binary_mask, image)
        
        # Create red overlay
        overlay = self._create_red_overlay(image, cleaned_mask)
        
        return {
            'probability': predictions['probability'],
            'binary_mask': cleaned_mask,
            'overlay': overlay,
            'threshold': threshold,
            'confidence': predictions['confidence'],
            'uncertainty': predictions['uncertainty']
        }
    
    def _multi_scale_predict(self, image: np.ndarray, use_tta: bool, use_tent: bool) -> Dict:
        """Multi-scale prediction with TTA and TENT"""
        scales = [0.8, 1.0, 1.2]
        all_probs = []
        all_uncertainties = []
        
        for scale in scales:
            # Resize image
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(image, (new_w, new_h))
            
            # Sliding window prediction
            prob_map, uncertainty_map = self._sliding_window_predict(
                scaled_img, use_tta, use_tent
            )
            
            # Resize back
            prob_map = cv2.resize(prob_map, (w, h))
            uncertainty_map = cv2.resize(uncertainty_map, (w, h))
            
            all_probs.append(prob_map)
            all_uncertainties.append(uncertainty_map)
        
        # Ensemble predictions
        final_prob = np.mean(all_probs, axis=0)
        final_uncertainty = np.mean(all_uncertainties, axis=0)
        confidence = 1.0 - final_uncertainty
        
        return {
            'probability': final_prob,
            'confidence': confidence,
            'uncertainty': final_uncertainty
        }
    
    def _sliding_window_predict(self, image: np.ndarray, use_tta: bool, use_tent: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding window prediction with overlap"""
        tile_size = 120
        stride = 48  # 60% overlap
        
        h, w = image.shape[:2]
        prob_map = np.zeros((h, w), dtype=np.float32)
        uncertainty_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # TENT adaptation setup
        if use_tent:
            self._setup_tent_adaptation()
        
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y:y+tile_size, x:x+tile_size]
                
                if use_tta:
                    tile_prob, tile_uncertainty = self._predict_tile_tta(tile, use_tent)
                else:
                    tile_prob, tile_uncertainty = self._predict_tile_single(tile, use_tent)
                
                # Gaussian weighting for smooth blending
                weight = self._get_gaussian_weight(tile_size)
                
                prob_map[y:y+tile_size, x:x+tile_size] += tile_prob * weight
                uncertainty_map[y:y+tile_size, x:x+tile_size] += tile_uncertainty * weight
                count_map[y:y+tile_size, x:x+tile_size] += weight
        
        # Normalize by overlap count
        prob_map = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map!=0)
        uncertainty_map = np.divide(uncertainty_map, count_map, out=np.zeros_like(uncertainty_map), where=count_map!=0)
        
        return prob_map, uncertainty_map
    
    def _predict_tile_tta(self, tile: np.ndarray, use_tent: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Test-time augmentation prediction"""
        augmentations = [
            lambda x: x,  # Original
            lambda x: np.fliplr(x),  # Horizontal flip
            lambda x: np.flipud(x),  # Vertical flip
            lambda x: np.rot90(x, 1),  # 90° rotation
            lambda x: np.rot90(x, 2),  # 180° rotation
            lambda x: np.rot90(x, 3),  # 270° rotation
        ]
        
        reverse_augs = [
            lambda x: x,
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x),
            lambda x: np.rot90(x, -1),
            lambda x: np.rot90(x, -2),
            lambda x: np.rot90(x, -3),
        ]
        
        predictions = []
        
        for aug, rev_aug in zip(augmentations, reverse_augs):
            aug_tile = aug(tile)
            pred, _ = self._predict_tile_single(aug_tile, use_tent)
            pred = rev_aug(pred)
            predictions.append(pred)
        
        # Ensemble statistics
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def _predict_tile_single(self, tile: np.ndarray, use_tent: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Single tile prediction with optional TENT adaptation"""
        # Preprocess tile
        processed_tile = self._preprocess_tile(tile)
        
        with torch.no_grad():
            if use_tent:
                # TENT adaptation
                pred = self._tent_predict(processed_tile)
            else:
                pred = self.model(processed_tile)
            
            # MC Dropout for uncertainty
            uncertainties = []
            self.model.train()  # Enable dropout
            for _ in range(5):
                mc_pred = self.model(processed_tile)
                uncertainties.append(torch.sigmoid(mc_pred).cpu().numpy())
            self.model.eval()
            
            uncertainties = np.stack(uncertainties)
            uncertainty = np.std(uncertainties, axis=0).squeeze()
            
            pred = torch.sigmoid(pred).cpu().numpy().squeeze()
        
        return pred, uncertainty
    
    def _preprocess_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Enhanced preprocessing with texture channels"""
        # Normalize
        tile = tile.astype(np.float32) / 255.0
        
        # CLAHE enhancement
        lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply((lab[:,:,0] * 255).astype(np.uint8)) / 255.0
        tile = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Compute texture channels
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Local entropy
        entropy = filters.rank.entropy(gray, morphology.disk(3))
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Stack all channels
        all_channels = np.stack([
            tile[:,:,0], tile[:,:,1], tile[:,:,2],  # RGB
            grad_mag, entropy, np.abs(laplacian)    # Texture
        ], axis=0)
        
        return torch.FloatTensor(all_channels).unsqueeze(0).to(self.device)
    
    def _setup_tent_adaptation(self):
        """Setup TENT adaptation (only BN parameters)"""
        self.tent_optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if 'bn' in n or 'norm' in n],
            lr=1e-4
        )
    
    def _tent_predict(self, x: torch.Tensor) -> torch.Tensor:
        """TENT adaptation prediction"""
        # Forward pass
        self.model.train()
        pred = self.model(x)
        
        # Entropy minimization
        prob = torch.sigmoid(pred)
        entropy = -prob * torch.log(prob + 1e-8) - (1-prob) * torch.log(1-prob + 1e-8)
        loss = entropy.mean()
        
        # Update only BN parameters
        self.tent_optimizer.zero_grad()
        loss.backward()
        self.tent_optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            final_pred = self.model(x)
        
        return final_pred
    
    def _compute_adaptive_threshold(self, prob_map: np.ndarray) -> float:
        """Compute adaptive threshold using Otsu's method"""
        # Convert to uint8 for Otsu
        prob_uint8 = (prob_map * 255).astype(np.uint8)
        
        # Otsu thresholding
        otsu_thresh, _ = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh = otsu_thresh / 255.0
        
        # Clamp to reasonable range
        adaptive_thresh = np.clip(otsu_thresh, 0.2, 0.6)
        
        # Fallback to percentile if low entropy
        entropy = -np.sum(prob_map * np.log(prob_map + 1e-8))
        if entropy < 0.1:  # Low entropy, use percentile
            adaptive_thresh = np.percentile(prob_map, 92)
        
        return float(adaptive_thresh)
    
    def _post_process_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Enhanced post-processing with morphology and CRF-like filtering"""
        # Remove small components
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=50).astype(np.uint8)
        
        # Morphological operations
        kernel_size = max(3, min(7, mask.shape[0] // 40))
        kernel = morphology.disk(kernel_size)
        
        # Opening then closing
        mask = morphology.opening(mask, kernel)
        mask = morphology.closing(mask, kernel)
        
        # Guided filtering using RGB
        mask_float = mask.astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Simple guided filter approximation
        mean_I = cv2.boxFilter(gray, cv2.CV_32F, (5, 5))
        mean_p = cv2.boxFilter(mask_float, cv2.CV_32F, (5, 5))
        corr_Ip = cv2.boxFilter(gray * mask_float, cv2.CV_32F, (5, 5))
        cov_Ip = corr_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(gray * gray, cv2.CV_32F, (5, 5))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + 0.01)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (5, 5))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (5, 5))
        
        filtered_mask = mean_a * gray + mean_b
        filtered_mask = np.clip(filtered_mask, 0, 1)
        
        return (filtered_mask > 0.5).astype(np.uint8)
    
    def _create_red_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create accurate red overlay for slum areas"""
        overlay = image.copy()
        
        # Create red mask with transparency
        red_mask = np.zeros_like(image)
        red_mask[:,:,0] = 255  # Pure red
        
        # Apply mask with alpha blending
        alpha = 0.6
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        overlay = np.where(mask_3d, 
                          (1-alpha) * image + alpha * red_mask,
                          image).astype(np.uint8)
        
        return overlay
    
    def _get_gaussian_weight(self, size: int) -> np.ndarray:
        """Generate Gaussian weight for smooth blending"""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        weight = np.exp(-((x - center)**2 + (y - center)**2) / (2 * (size/4)**2))
        return weight

def main():
    """Demo usage"""
    detector = GlobalSlumDetector("path/to/checkpoint.pth")
    
    result = detector.predict_global(
        "satellite_image.jpg",
        use_tta=True,
        use_tent=True,
        adaptive_threshold=True
    )
    
    # Save results
    cv2.imwrite("overlay.jpg", cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
    cv2.imwrite("probability.jpg", (result['probability'] * 255).astype(np.uint8))
    
    print(f"Adaptive threshold: {result['threshold']:.3f}")
    print(f"Mean confidence: {result['confidence'].mean():.3f}")

if __name__ == "__main__":
    main()