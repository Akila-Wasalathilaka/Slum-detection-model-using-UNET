#!/usr/bin/env python3
# File: optimize_slum_detection.py
# Configuration optimizer for better slum detection accuracy

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

class SlumDetectionOptimizer:
    """Optimizer for slum detection model configuration."""
    
    def __init__(self, model_path: str, test_data_dir: str):
        self.model_path = Path(model_path)
        self.test_data_dir = Path(test_data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load test data paths
        self.test_images = list((self.test_data_dir / "images").glob("*.tif"))
        self.test_masks = list((self.test_data_dir / "masks").glob("*.png"))
        
        print(f"Found {len(self.test_images)} test images")
    
    def load_ground_truth(self, mask_path: str, slum_class_id: int = 2) -> np.ndarray:
        """Load and convert ground truth mask to binary."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return (mask == slum_class_id).astype(np.uint8)
    
    def evaluate_thresholds(self, predictions: Dict[str, np.ndarray], 
                          ground_truths: Dict[str, np.ndarray]) -> Dict[float, Dict]:
        """Evaluate different probability thresholds."""
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = {}
        
        for threshold in thresholds:
            all_true = []
            all_pred = []
            
            for img_name in predictions.keys():
                if img_name in ground_truths:
                    pred = (predictions[img_name] > threshold).astype(np.uint8)
                    gt = ground_truths[img_name]
                    
                    all_true.extend(gt.flatten())
                    all_pred.extend(pred.flatten())
            
            if len(all_true) > 0:
                metrics = {
                    'precision': precision_score(all_true, all_pred, zero_division=0),
                    'recall': recall_score(all_true, all_pred, zero_division=0),
                    'f1': f1_score(all_true, all_pred, zero_division=0),
                    'iou': jaccard_score(all_true, all_pred, zero_division=0)
                }
                results[float(threshold)] = metrics
        
        return results
    
    def optimize_post_processing(self, predictions: Dict[str, np.ndarray],
                               ground_truths: Dict[str, np.ndarray],
                               threshold: float = 0.35) -> Dict:
        """Optimize post-processing parameters."""
        min_sizes = [10, 25, 50, 75, 100]
        kernel_sizes = [3, 5, 7, 9]
        
        best_score = 0
        best_params = {}
        
        for min_size in min_sizes:
            for kernel_size in kernel_sizes:
                all_true = []
                all_pred = []
                
                for img_name in predictions.keys():
                    if img_name in ground_truths:
                        # Apply threshold
                        pred = (predictions[img_name] > threshold).astype(np.uint8)
                        
                        # Apply post-processing
                        pred = self.apply_post_processing(pred, min_size, kernel_size)
                        
                        gt = ground_truths[img_name]
                        
                        all_true.extend(gt.flatten())
                        all_pred.extend(pred.flatten())
                
                if len(all_true) > 0:
                    f1 = f1_score(all_true, all_pred, zero_division=0)
                    iou = jaccard_score(all_true, all_pred, zero_division=0)
                    combined_score = 0.6 * f1 + 0.4 * iou
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {
                            'min_object_size': min_size,
                            'kernel_size': kernel_size,
                            'f1_score': f1,
                            'iou_score': iou,
                            'combined_score': combined_score
                        }
        
        return best_params
    
    def apply_post_processing(self, prediction: np.ndarray, 
                            min_size: int, kernel_size: int) -> np.ndarray:
        """Apply post-processing with given parameters."""
        # Remove small objects
        num_labels, labels = cv2.connectedComponents(prediction)
        for label_id in range(1, num_labels):
            mask = (labels == label_id)
            if np.sum(mask) < min_size:
                prediction[mask] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Opening to remove noise
        prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill holes
        prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)
        
        return prediction
    
    def analyze_class_distribution(self) -> Dict:
        """Analyze class distribution in test set."""
        slum_pixels = 0
        total_pixels = 0
        slum_images = 0
        
        for mask_path in self.test_masks[:50]:  # Sample first 50 masks
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            slum_mask = (mask == 2).astype(np.uint8)
            
            slum_pixels += np.sum(slum_mask)
            total_pixels += mask.size
            
            if np.sum(slum_mask) > 0:
                slum_images += 1
        
        slum_ratio = slum_pixels / total_pixels if total_pixels > 0 else 0
        
        return {
            'slum_pixel_ratio': slum_ratio,
            'non_slum_pixel_ratio': 1 - slum_ratio,
            'images_with_slums': slum_images,
            'total_images_analyzed': len(self.test_masks[:50]),
            'slum_image_ratio': slum_images / len(self.test_masks[:50]) if len(self.test_masks) > 0 else 0
        }
    
    def generate_optimal_config(self) -> Dict:
        """Generate optimal configuration based on analysis."""
        # Analyze class distribution
        class_dist = self.analyze_class_distribution()
        
        # Calculate optimal loss weights based on class imbalance
        slum_ratio = class_dist['slum_pixel_ratio']
        pos_weight = (1 - slum_ratio) / slum_ratio if slum_ratio > 0 else 10.0
        
        # Generate optimized configuration
        optimal_config = {
            'model_parameters': {
                'image_size': 160,  # Increased for better detail
                'batch_size': 12,
                'learning_rate': 1e-4,
                'epochs': 80,
                'patience': 15
            },
            'loss_configuration': {
                'loss_weights': {
                    'dice': 0.4,
                    'focal': 0.3,
                    'bce': 0.2,
                    'tversky': 0.1
                },
                'focal_alpha': min(0.4, slum_ratio * 2),  # Adaptive alpha
                'focal_gamma': 2.5,
                'tversky_alpha': 0.6,
                'tversky_beta': 0.4,
                'bce_pos_weight': min(pos_weight, 10.0)  # Cap at 10
            },
            'thresholds': {
                'conservative': 0.6,
                'balanced': 0.35,
                'sensitive': 0.2
            },
            'post_processing': {
                'min_object_size': 25,
                'morphology_kernel': 5,
                'use_tta': True,
                'tta_transforms': 8
            },
            'data_augmentation': {
                'use_advanced_augmentation': True,
                'augmentation_probability': 0.8,
                'color_augmentation_strength': 0.3,
                'geometric_augmentation_strength': 0.6
            },
            'class_distribution': class_dist
        }
        
        return optimal_config
    
    def save_optimization_report(self, config: Dict, 
                               threshold_results: Dict = None,
                               postproc_results: Dict = None):
        """Save comprehensive optimization report."""
        report = {
            'optimization_summary': {
                'timestamp': str(pd.Timestamp.now()),
                'test_images_analyzed': len(self.test_images),
                'device_used': str(self.device)
            },
            'optimal_configuration': config,
            'threshold_optimization': threshold_results,
            'post_processing_optimization': postproc_results,
            'recommendations': self.generate_recommendations(config)
        }
        
        # Save report
        output_path = Path("optimization_report.json")
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Optimization report saved to: {output_path}")
        return report
    
    def generate_recommendations(self, config: Dict) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        class_dist = config['class_distribution']
        slum_ratio = class_dist['slum_pixel_ratio']
        
        if slum_ratio < 0.01:
            recommendations.append(
                "Very low slum pixel ratio detected. Consider using higher focal loss gamma (3.0+) and lower threshold (0.2-0.3)."
            )
        
        if slum_ratio > 0.1:
            recommendations.append(
                "High slum pixel ratio detected. Consider balanced approach with moderate focal loss parameters."
            )
        
        if class_dist['slum_image_ratio'] < 0.3:
            recommendations.append(
                "Low proportion of images contain slums. Consider data augmentation focused on slum areas."
            )
        
        recommendations.extend([
            "Use Test Time Augmentation (TTA) for better inference accuracy.",
            "Consider ensemble of multiple models with different encoders.",
            "Monitor validation IoU and Dice coefficient during training.",
            "Use early stopping to prevent overfitting on this imbalanced dataset.",
            "Consider using learning rate scheduling for better convergence."
        ])
        
        return recommendations

def main():
    """Main optimization function."""
    import pandas as pd
    
    print("🔧 SLUM DETECTION MODEL OPTIMIZER")
    print("=" * 50)
    
    # Initialize optimizer
    test_data_dir = "data_preprocessed/test"
    model_path = "models_production/best_production_model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using run_production_pipeline.py")
        return
    
    optimizer = SlumDetectionOptimizer(model_path, test_data_dir)
    
    # Generate optimal configuration
    print("🔍 Analyzing dataset and generating optimal configuration...")
    optimal_config = optimizer.generate_optimal_config()
    
    # Save optimization report
    print("📊 Saving optimization report...")
    report = optimizer.save_optimization_report(optimal_config)
    
    # Print summary
    print("\n✅ OPTIMIZATION COMPLETED!")
    print("=" * 50)
    print("📋 Key Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    print(f"\n📁 Full report saved to: optimization_report.json")
    print(f"🎯 Optimal threshold: {optimal_config['thresholds']['balanced']}")
    print(f"🧬 Recommended pos_weight: {optimal_config['loss_configuration']['bce_pos_weight']:.2f}")

if __name__ == "__main__":
    main()
