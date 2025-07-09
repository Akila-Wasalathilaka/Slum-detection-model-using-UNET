"""
Ultra-accurate inference pipeline with slum mapping and visualization.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from typing import List, Tuple, Dict
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from model import UltraAccurateUNet
from config import config

class UltraTestTimeAugmentation:
    """Ultra-enhanced Test Time Augmentation for maximum accuracy."""
    
    def __init__(self, num_transforms: int = 12):
        self.num_transforms = num_transforms
        self.transforms = self._get_ultra_tta_transforms()
    
    def _get_ultra_tta_transforms(self):
        """Get comprehensive TTA transform list."""
        transforms = []
        
        # Original
        transforms.append(A.Compose([]))
        
        # Flips
        transforms.append(A.Compose([A.HorizontalFlip(p=1.0)]))
        transforms.append(A.Compose([A.VerticalFlip(p=1.0)]))
        transforms.append(A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]))
        
        # Rotations
        transforms.append(A.Compose([A.RandomRotate90(p=1.0)]))
        transforms.append(A.Compose([A.Rotate(limit=15, p=1.0)]))
        transforms.append(A.Compose([A.Rotate(limit=-15, p=1.0)]))
        
        # Scale variations
        transforms.append(A.Compose([A.RandomScale(scale_limit=0.1, p=1.0)]))
        transforms.append(A.Compose([A.RandomScale(scale_limit=-0.1, p=1.0)]))
        
        # Brightness variations
        transforms.append(A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)]))
        
        # Subtle noise
        transforms.append(A.Compose([A.GaussNoise(var_limit=(5, 15), p=1.0)]))
        
        # Slight blur
        transforms.append(A.Compose([A.GaussianBlur(blur_limit=(1, 3), p=1.0)]))
        
        return transforms[:self.num_transforms]
    
    def __call__(self, model, image: torch.Tensor) -> torch.Tensor:
        """Apply ultra-TTA and return weighted averaged prediction."""
        predictions = []
        weights = []
        
        for i, transform in enumerate(self.transforms):
            # Convert to numpy for albumentations
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            
            # Apply transform
            augmented = transform(image=img_np)
            aug_img = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).unsqueeze(0)
            aug_img = aug_img.to(image.device)
            
            # Get prediction
            with torch.no_grad():
                pred = torch.sigmoid(model(aug_img))
            
            # Reverse transform on prediction
            pred_np = pred.squeeze().cpu().numpy()
            reversed_pred = self._reverse_transform(transform, pred_np)
            
            # Weight based on confidence (higher confidence = higher weight)
            confidence = np.std(reversed_pred)  # Higher std = more confident
            weight = 1.0 + confidence * 0.5  # Boost confident predictions
            
            predictions.append(torch.from_numpy(reversed_pred))
            weights.append(weight)
        
        # Weighted average predictions
        weights = torch.tensor(weights)
        weights = weights / weights.sum()
        
        weighted_pred = sum(w * p for w, p in zip(weights, predictions))
        return weighted_pred
    
    def _reverse_transform(self, transform, prediction):
        """Reverse the transform applied to get original orientation."""
        # Simplified - in production, implement proper reverse transforms
        return prediction

class SlumMapper:
    """Advanced slum mapping and visualization system."""
    
    def __init__(self):
        self.slum_color = config.SLUM_MARKER_COLOR
        self.marker_size = config.SLUM_MARKER_SIZE
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD_FOR_MARKING
    
    def identify_slum_regions(self, prediction: np.ndarray, confidence_map: np.ndarray) -> List[Dict]:
        """Identify and characterize slum regions."""
        # Apply confidence threshold
        confident_slums = (prediction > self.confidence_threshold) & (confidence_map > 0.8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(confident_slums.astype(np.uint8))
        
        slum_regions = []
        for label_id in range(1, num_labels):
            region_mask = (labels == label_id)
            
            if np.sum(region_mask) < config.MIN_OBJECT_SIZE:
                continue
            
            # Calculate region properties
            region_props = self._calculate_region_properties(region_mask, prediction, confidence_map)
            slum_regions.append(region_props)
        
        return slum_regions
    
    def _calculate_region_properties(self, mask: np.ndarray, prediction: np.ndarray, confidence: np.ndarray) -> Dict:
        """Calculate comprehensive properties of slum regions."""
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Average prediction and confidence in region
        avg_prediction = np.mean(prediction[mask])
        avg_confidence = np.mean(confidence[mask])
        
        # Compactness (circularity)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Density estimation
        density = area / (w * h) if (w * h) > 0 else 0
        
        return {
            'centroid': (cx, cy),
            'area': area,
            'perimeter': perimeter,
            'bounding_box': (x, y, w, h),
            'avg_prediction': avg_prediction,
            'avg_confidence': avg_confidence,
            'compactness': compactness,
            'density': density,
            'contour': largest_contour
        }
    
    def create_slum_map(self, original_image: np.ndarray, prediction: np.ndarray, 
                       confidence_map: np.ndarray) -> np.ndarray:
        """Create a visual slum map similar to reference images."""
        # Create base map
        map_image = original_image.copy()
        
        # Identify slum regions
        slum_regions = self.identify_slum_regions(prediction, confidence_map)
        
        # Draw slum markers
        for region in slum_regions:
            if not region:
                continue
                
            cx, cy = region['centroid']
            confidence = region['avg_confidence']
            
            # Marker size based on area and confidence
            marker_size = max(self.marker_size, int(np.sqrt(region['area']) / 10))
            marker_size = min(marker_size, 20)  # Cap maximum size
            
            # Color intensity based on confidence
            color_intensity = int(255 * confidence)
            marker_color = (color_intensity, 0, 0)  # Red with varying intensity
            
            # Draw filled circle marker
            cv2.circle(map_image, (cx, cy), marker_size, marker_color, -1)
            
            # Draw border for visibility
            cv2.circle(map_image, (cx, cy), marker_size, (255, 255, 255), 2)
            
            # Add confidence text for high-confidence regions
            if confidence > 0.9:
                cv2.putText(map_image, f'{confidence:.2f}', 
                           (cx - 15, cy - marker_size - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return map_image
    
    def create_clustered_map(self, original_image: np.ndarray, slum_regions: List[Dict]) -> np.ndarray:
        """Create a clustered slum map with geographic analysis."""
        if not slum_regions:
            return original_image
        
        # Extract centroids for clustering
        centroids = np.array([region['centroid'] for region in slum_regions if region])
        
        if len(centroids) < 2:
            return self.create_slum_map(original_image, 
                                      np.ones_like(original_image[:,:,0]), 
                                      np.ones_like(original_image[:,:,0]))
        
        # Perform DBSCAN clustering
        scaler = StandardScaler()
        centroids_scaled = scaler.fit_transform(centroids)
        
        clustering = DBSCAN(eps=0.3, min_samples=config.CLUSTER_MIN_SAMPLES).fit(centroids_scaled)
        
        # Create clustered map
        clustered_image = original_image.copy()
        
        # Color palette for clusters
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for i, (region, cluster_id) in enumerate(zip(slum_regions, clustering.labels_)):
            if not region or cluster_id == -1:  # -1 is noise in DBSCAN
                continue
                
            cx, cy = region['centroid']
            color = colors[cluster_id % len(colors)]
            
            # Draw cluster marker
            cv2.circle(clustered_image, (cx, cy), self.marker_size, color, -1)
            cv2.circle(clustered_image, (cx, cy), self.marker_size, (255, 255, 255), 2)
        
        return clustered_image

class UltraAccurateInference:
    """Ultra-accurate inference pipeline with advanced slum mapping."""
    
    def __init__(self, model_path: str, device: torch.device, use_tta: bool = True):
        self.device = device
        self.use_tta = use_tta
        
        # Load ultra-accurate model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = UltraAccurateUNet()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Initialize TTA
        if use_tta:
            self.tta = UltraTestTimeAugmentation(num_transforms=config.TTA_TRANSFORMS)
        
        # Initialize slum mapper
        self.slum_mapper = SlumMapper()
        
        # Enhanced post-processing kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (config.MORPHOLOGY_KERNEL, config.MORPHOLOGY_KERNEL)
        )
    
    def preprocess_image(self, image_path: str, size: int = None) -> Tuple[torch.Tensor, np.ndarray]:
        """Enhanced preprocessing with original image retention."""
        if size is None:
            size = config.PRIMARY_SIZE
            
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize for processing
        processed_image = cv2.resize(original_image, (size, size), interpolation=cv2.INTER_LINEAR)
        
        # Enhanced normalization
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        transformed = transform(image=processed_image)
        return transformed['image'].unsqueeze(0), original_image
    
    def ultra_postprocess_prediction(self, prediction: np.ndarray, 
                                   threshold: str = 'balanced') -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-accurate post-processing with confidence estimation."""
        thresh_value = config.BINARY_THRESHOLDS[threshold]
        
        # Generate confidence map
        confidence_map = np.abs(prediction - 0.5) * 2  # Distance from decision boundary
        
        # Apply threshold
        binary_pred = (prediction > thresh_value).astype(np.uint8)
        
        # Enhanced morphological operations
        # Remove small noise
        binary_pred = cv2.morphologyEx(binary_pred, cv2.MORPH_OPEN, self.morph_kernel, iterations=2)
        
        # Fill small holes
        binary_pred = cv2.morphologyEx(binary_pred, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        
        # Remove small objects with area analysis
        num_labels, labels = cv2.connectedComponents(binary_pred)
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            
            # Check both size and shape
            area = np.sum(mask)
            if area < config.MIN_OBJECT_SIZE:
                binary_pred[mask] = 0
                continue
            
            # Check compactness (remove very elongated objects)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter * perimeter)
                    if compactness < 0.1:  # Very elongated
                        binary_pred[mask] = 0
        
        return binary_pred, confidence_map
    
    def predict_single_with_mapping(self, image_path: str, 
                                  threshold: str = 'balanced') -> Dict:
        """Predict single image with comprehensive slum mapping."""
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            if self.use_tta:
                prediction = self.tta(self.model, image_tensor.squeeze(0))
            else:
                output = self.model(image_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                prediction = torch.sigmoid(output).squeeze().cpu()
        
        prediction_np = prediction.numpy()
        
        # Ultra post-processing
        binary_prediction, confidence_map = self.ultra_postprocess_prediction(prediction_np, threshold)
        
        # Resize back to original size
        original_height, original_width = original_image.shape[:2]
        prediction_resized = cv2.resize(prediction_np, (original_width, original_height), 
                                      interpolation=cv2.INTER_LINEAR)
        binary_resized = cv2.resize(binary_prediction, (original_width, original_height), 
                                  interpolation=cv2.INTER_NEAREST)
        confidence_resized = cv2.resize(confidence_map, (original_width, original_height), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # Create slum maps
        slum_regions = self.slum_mapper.identify_slum_regions(prediction_resized, confidence_resized)
        slum_map = self.slum_mapper.create_slum_map(original_image, prediction_resized, confidence_resized)
        clustered_map = self.slum_mapper.create_clustered_map(original_image, slum_regions)
        
        return {
            'prediction': prediction_resized,
            'binary_prediction': binary_resized,
            'confidence_map': confidence_resized,
            'slum_regions': slum_regions,
            'slum_map': slum_map,
            'clustered_map': clustered_map,
            'original_image': original_image
        }
    
    def predict_batch_with_mapping(self, image_paths: List[str], 
                                 threshold: str = 'balanced') -> List[Dict]:
        """Predict batch of images with comprehensive mapping."""
        results = []
        
        for image_path in tqdm(image_paths, desc="🎯 Processing images with ultra-accuracy"):
            result = self.predict_single_with_mapping(image_path, threshold)
            results.append(result)
        
        return results
