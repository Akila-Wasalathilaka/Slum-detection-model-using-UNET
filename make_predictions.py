#!/usr/bin/env python3
"""
Advanced Prediction Generator for Slum Detection
===============================================
Generates comprehensive predictions with visualizations and analysis.
Creates 20+ diverse prediction samples with detailed analysis.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class WaterDiscriminator:
    """Post-processing to reduce water misclassification"""
    
    def detect_water_regions(self, image):
        """Detect water-like regions using color and texture"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Water color ranges
        water_mask1 = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        water_mask2 = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 30, 100]))
        water_mask = cv2.bitwise_or(water_mask1, water_mask2)
        
        # Low texture (water is smooth)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
        low_texture = (np.abs(texture) < 20).astype(np.uint8) * 255
        
        water_mask = cv2.bitwise_and(water_mask, low_texture)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        
        return water_mask > 0
    
    def detect_slum_texture(self, image):
        """Detect slum-like high-frequency textures"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
        edges = cv2.Canny(gray, 50, 150)
        
        high_texture = np.abs(texture) > 30
        edge_density = cv2.dilate(edges, np.ones((3, 3)), iterations=1) > 0
        
        return high_texture & edge_density
    
    def post_process(self, image, prediction):
        """Apply water/slum discrimination"""
        water_regions = self.detect_water_regions(image)
        slum_texture = self.detect_slum_texture(image)
        
        corrected = prediction.copy()
        
        # Fix water misclassified as slum
        for slum_class in [1, 2, 3, 4, 5, 6]:
            slum_mask = (prediction == slum_class)
            water_slum_overlap = slum_mask & water_regions & (~slum_texture)
            corrected[water_slum_overlap] = 0  # Change to background
        
        return corrected

class AdvancedPredictor:
    def __init__(self, model_path, data_root="data", output_dir="predictions"):
        self.model_path = model_path
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize water discriminator
        self.water_discriminator = WaterDiscriminator()
        
        # Comprehensive class names based on analysis
        self.class_names = {
            0: "Background",
            1: "Mixed-Urban-Residential", 
            2: "Informal-Settlements",
            3: "Secondary-Slums",
            4: "Bright-Urban-Commercial",
            5: "Bright-Urban-Commercial-2",
            6: "Primary-Slums"
        }
        
        self.class_colors = {
            0: [47, 47, 47],      # Dark gray
            105: [255, 107, 107], # Red
            109: [78, 205, 196],  # Teal  
            111: [69, 183, 209],  # Blue
            158: [150, 206, 180], # Green
            200: [255, 234, 167], # Yellow
            233: [221, 160, 221]  # Plum
        }
        
        # Load model
        self.model, self.device = self.load_model()
        
        # Setup transforms with TTA support
        self.transform = A.Compose([
            A.Resize(224, 224),  # Match training resolution
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Test-time augmentation transforms
        self.tta_transforms = [
            A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
            A.Compose([A.Resize(224, 224), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
        ]
    
    def load_model(self):
        """Load the trained model"""
        print(f"🔄 Loading model from {self.model_path}")
        
        try:
            import segmentation_models_pytorch as smp
            from advanced_training import AdvancedUNet
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AdvancedUNet(encoder_name='efficientnet-b1', num_classes=7)
            
            if os.path.exists(self.model_path):
                model.load_state_dict(torch.load(self.model_path, map_location=device))
                print(f"✅ Model loaded successfully on {device}")
            else:
                print(f"⚠️ Model file not found: {self.model_path}")
                print("Using randomly initialized model for demonstration")
            
            model.to(device)
            model.eval()
            return model, device
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Creating dummy model for demonstration")
            
            # Create a simple dummy model for demonstration
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 7, 1)
            ).to(device)
            model.eval()
            return model, device
    
    def predict_single_image(self, image_path, mask_path=None):
        """Predict on a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Load ground truth if available
        ground_truth = None
        if mask_path and os.path.exists(mask_path):
            ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Map ground truth classes to 0-6 range
            class_mapping = {0: 0, 105: 1, 109: 2, 111: 3, 158: 4, 200: 5, 233: 6}
            mapped_gt = np.zeros_like(ground_truth)
            for original_val, new_val in class_mapping.items():
                mapped_gt[ground_truth == original_val] = new_val
            ground_truth = mapped_gt
        
        # Test-time augmentation for better predictions
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for tta_transform in self.tta_transforms:
                augmented = tta_transform(image=image)
                input_tensor = augmented['image'].unsqueeze(0).to(self.device)
                
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1).squeeze().cpu().numpy()
                conf = probabilities.max(dim=1)[0].squeeze().cpu().numpy()
                
                # Reverse augmentation for prediction
                if 'HorizontalFlip' in str(tta_transform):
                    pred = np.fliplr(pred)
                    conf = np.fliplr(conf)
                elif 'VerticalFlip' in str(tta_transform):
                    pred = np.flipud(pred)
                    conf = np.flipud(conf)
                
                predictions.append(pred)
                confidences.append(conf)
        
        # Ensemble predictions
        prediction = np.round(np.mean(predictions, axis=0)).astype(np.uint8)
        confidence = np.mean(confidences, axis=0)
        
        # Resize prediction back to original size
        prediction_resized = cv2.resize(prediction.astype(np.uint8), 
                                       (original_size[1], original_size[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        confidence_resized = cv2.resize(confidence, 
                                       (original_size[1], original_size[0]))
        
        # Apply water discrimination post-processing on resized prediction
        prediction_resized = self.water_discriminator.post_process(image, prediction_resized)
        
        # Resize ground truth to match prediction if needed
        if ground_truth is not None and ground_truth.shape != prediction_resized.shape:
            ground_truth = cv2.resize(ground_truth, 
                                    (original_size[1], original_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        return {
            'image': image,
            'prediction': prediction_resized,
            'confidence': confidence_resized,
            'ground_truth': ground_truth,
            'probabilities': probabilities.squeeze().cpu().numpy()
        }
    
    def create_colored_mask(self, mask):
        """Create colored visualization of mask"""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            colored[mask == class_id] = color
        
        return colored
    
    def analyze_prediction(self, result):
        """Analyze prediction results"""
        prediction = result['prediction']
        confidence = result['confidence']
        ground_truth = result['ground_truth']
        
        analysis = {
            'unique_classes': np.unique(prediction).tolist(),
            'class_distribution': {},
            'mean_confidence': float(np.mean(confidence)),
            'min_confidence': float(np.min(confidence)),
            'max_confidence': float(np.max(confidence)),
            'low_confidence_pixels': float(np.sum(confidence < 0.5) / confidence.size),
            'issues': []
        }
        
        # Detect potential issues
        for slum_class in [1, 2, 3, 4, 5, 6]:
            if slum_class in analysis['unique_classes']:
                slum_mask = prediction == slum_class
                slum_confidence = confidence[slum_mask]
                if len(slum_confidence) > 0 and np.mean(slum_confidence) < 0.6:
                    analysis['issues'].append(f"Low confidence in {self.class_names[slum_class]}")
        
        # Calculate class distribution
        total_pixels = prediction.size
        for class_id in analysis['unique_classes']:
            count = np.sum(prediction == class_id)
            analysis['class_distribution'][int(class_id)] = {
                'count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
        
        # Calculate accuracy if ground truth available
        if ground_truth is not None:
            # Ensure shapes match
            if ground_truth.shape != prediction.shape:
                ground_truth = cv2.resize(ground_truth, 
                                        (prediction.shape[1], prediction.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            
            # Only calculate if shapes are compatible
            if ground_truth.shape == prediction.shape:
                accuracy = np.mean(prediction == ground_truth)
                analysis['accuracy'] = float(accuracy)
                
                # Per-class accuracy
                analysis['class_accuracy'] = {}
                for class_id in np.unique(ground_truth):
                    mask = ground_truth == class_id
                    if mask.sum() > 0:
                        class_acc = np.mean(prediction[mask] == class_id)
                        analysis['class_accuracy'][int(class_id)] = float(class_acc)
        
        return analysis
    
    def create_prediction_visualization(self, result, analysis, title="Prediction"):
        """Create comprehensive prediction visualization"""
        image = result['image']
        prediction = result['prediction']
        confidence = result['confidence']
        ground_truth = result['ground_truth']
        
        # Determine subplot layout
        n_cols = 4 if ground_truth is not None else 3
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))
        
        if n_cols == 3:
            axes = axes.reshape(2, 3)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction
        pred_colored = self.create_colored_mask(prediction)
        axes[0, 1].imshow(pred_colored)
        axes[0, 1].set_title(f'Prediction\\n(Acc: {analysis.get("accuracy", "N/A"):.3f})', 
                            fontweight='bold')
        axes[0, 1].axis('off')
        
        # Confidence map
        im = axes[0, 2].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Confidence\\n(Mean: {analysis["mean_confidence"]:.3f})', 
                            fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Ground truth (if available)
        if ground_truth is not None:
            gt_colored = self.create_colored_mask(ground_truth)
            axes[0, 3].imshow(gt_colored)
            axes[0, 3].set_title('Ground Truth', fontweight='bold')
            axes[0, 3].axis('off')
        
        # Class distribution
        classes = list(analysis['class_distribution'].keys())
        percentages = [analysis['class_distribution'][c]['percentage'] for c in classes]
        colors = [np.array(self.class_colors.get(c, [128, 128, 128])) / 255.0 for c in classes]
        
        bars = axes[1, 0].bar(range(len(classes)), percentages, color=colors, alpha=0.8)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_title('Class Distribution')
        axes[1, 0].set_xticks(range(len(classes)))
        axes[1, 0].set_xticklabels([str(c) for c in classes])
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Confidence histogram
        axes[1, 1].hist(confidence.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Confidence Distribution')
        axes[1, 1].axvline(analysis['mean_confidence'], color='red', linestyle='--', 
                          label=f'Mean: {analysis["mean_confidence"]:.3f}')
        axes[1, 1].legend()
        
        # Analysis summary
        if n_cols == 4:
            axes[1, 2].axis('off')
            summary_text = f"""
ENHANCED ANALYSIS
{'='*20}

Classes Found: {len(analysis['unique_classes'])}
Mean Confidence: {analysis['mean_confidence']:.3f}
Low Confidence: {analysis['low_confidence_pixels']*100:.1f}%

Quality Issues:
{chr(10).join([f"• {issue}" for issue in analysis['issues']]) if analysis['issues'] else "• No issues detected"}

Dominant Classes:
"""
            # Add top 3 classes
            sorted_classes = sorted(analysis['class_distribution'].items(), 
                                  key=lambda x: x[1]['percentage'], reverse=True)[:3]
            for class_id, info in sorted_classes:
                class_name = self.class_names.get(class_id, f'Class {class_id}')
                summary_text += f"• {class_name}: {info['percentage']:.1f}%\\n"
            
            if 'accuracy' in analysis:
                summary_text += f"\\nOverall Accuracy: {analysis['accuracy']:.3f}"
            
            axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            # Legend
            axes[1, 3].axis('off')
            legend_elements = []
            for class_id, color in self.class_colors.items():
                if class_id in analysis['unique_classes']:
                    color_norm = np.array(color) / 255.0
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                                       facecolor=color_norm, 
                                                       label=f'{class_id}: {self.class_names[class_id]}'))
            
            axes[1, 3].legend(handles=legend_elements, loc='center', fontsize=10)
            axes[1, 3].set_title('Class Legend', fontweight='bold')
        else:
            # Combine analysis and legend in one subplot
            axes[1, 2].axis('off')
            
            # Create legend
            legend_elements = []
            for class_id, color in self.class_colors.items():
                if class_id in analysis['unique_classes']:
                    color_norm = np.array(color) / 255.0
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                                       facecolor=color_norm, 
                                                       label=f'{class_id}: {self.class_names[class_id]}'))
            
            axes[1, 2].legend(handles=legend_elements, loc='upper left', fontsize=9)
            
            # Add summary text
            summary_text = f"""
ENHANCED ANALYSIS
{'='*16}

Classes: {len(analysis['unique_classes'])}
Confidence: {analysis['mean_confidence']:.3f}
Low Conf: {analysis['low_confidence_pixels']*100:.1f}%
Issues: {len(analysis['issues'])}
"""
            if 'accuracy' in analysis:
                summary_text += f"Accuracy: {analysis['accuracy']:.3f}\n"
            
            summary_text += "\nEnhancements:\n• Water discrimination\n• Test-time augmentation\n• Post-processing"
            
            axes[1, 2].text(0.05, 0.6, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_diverse_predictions(self, num_predictions=25):
        """Generate diverse predictions from different splits and scenarios"""
        print(f"🎯 Generating {num_predictions} diverse predictions...")
        
        predictions = []
        
        # Collect images from all splits
        all_images = []
        
        for split in ['test', 'val', 'train']:
            img_dir = self.data_root / split / "images"
            mask_dir = self.data_root / split / "masks"
            
            if img_dir.exists():
                img_files = list(img_dir.glob("*.tif"))
                for img_path in img_files:
                    mask_path = mask_dir / (img_path.stem + ".png") if mask_dir.exists() else None
                    all_images.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'split': split
                    })
        
        if not all_images:
            print("❌ No images found!")
            return []
        
        # Select diverse samples
        np.random.seed(42)  # For reproducible selection
        selected_indices = np.random.choice(len(all_images), 
                                          min(num_predictions, len(all_images)), 
                                          replace=False)
        
        for i, idx in enumerate(tqdm(selected_indices, desc="Generating predictions")):
            sample = all_images[idx]
            
            try:
                # Generate prediction
                result = self.predict_single_image(sample['image_path'], sample['mask_path'])
                analysis = self.analyze_prediction(result)
                
                # Create visualization
                title = f"Prediction {i+1:02d} - {sample['split'].upper()} - {sample['image_path'].stem}"
                fig = self.create_prediction_visualization(result, analysis, title)
                
                # Save visualization
                output_path = self.output_dir / f"prediction_{i+1:02d}_{sample['split']}_{sample['image_path'].stem}.png"
                fig.savefig(output_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                
                # Store prediction data
                prediction_data = {
                    'id': i + 1,
                    'image_path': str(sample['image_path']),
                    'mask_path': str(sample['mask_path']) if sample['mask_path'] else None,
                    'split': sample['split'],
                    'analysis': analysis,
                    'output_path': str(output_path)
                }
                
                predictions.append(prediction_data)
                
            except Exception as e:
                print(f"⚠️ Error processing {sample['image_path']}: {e}")
                continue
        
        return predictions
    
    def create_prediction_summary(self, predictions):
        """Create comprehensive summary of all predictions"""
        print("📊 Creating prediction summary...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Prediction Summary Analysis', fontsize=16, fontweight='bold')
        
        # Extract metrics
        accuracies = [p['analysis'].get('accuracy', 0) for p in predictions if 'accuracy' in p['analysis']]
        confidences = [p['analysis']['mean_confidence'] for p in predictions]
        low_conf_percentages = [p['analysis']['low_confidence_pixels'] * 100 for p in predictions]
        
        # 1. Accuracy distribution
        if accuracies:
            axes[0, 0].hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Accuracy')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title(f'Accuracy Distribution\\n(Mean: {np.mean(accuracies):.3f})')
            axes[0, 0].axvline(np.mean(accuracies), color='red', linestyle='--')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Ground Truth\\nAvailable', ha='center', va='center',
                           transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_title('Accuracy Distribution')
        
        # 2. Confidence distribution
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_xlabel('Mean Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Confidence Distribution\\n(Mean: {np.mean(confidences):.3f})')
        axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--')
        
        # 3. Low confidence analysis
        axes[0, 2].hist(low_conf_percentages, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 2].set_xlabel('Low Confidence Pixels (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Low Confidence Analysis\\n(Mean: {np.mean(low_conf_percentages):.1f}%)')
        axes[0, 2].axvline(np.mean(low_conf_percentages), color='red', linestyle='--')
        
        # 4. Class frequency across all predictions
        all_classes = {}
        for pred in predictions:
            for class_id, info in pred['analysis']['class_distribution'].items():
                if class_id not in all_classes:
                    all_classes[class_id] = []
                all_classes[class_id].append(info['percentage'])
        
        classes = sorted(all_classes.keys())
        mean_percentages = [np.mean(all_classes[c]) for c in classes]
        colors = [np.array(self.class_colors.get(c, [128, 128, 128])) / 255.0 for c in classes]
        
        bars = axes[1, 0].bar(range(len(classes)), mean_percentages, color=colors, alpha=0.8)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Mean Percentage (%)')
        axes[1, 0].set_title('Average Class Distribution')
        axes[1, 0].set_xticks(range(len(classes)))
        axes[1, 0].set_xticklabels([str(c) for c in classes])
        
        # Add labels
        for bar, pct in zip(bars, mean_percentages):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 5. Performance by split
        split_performance = {}
        for pred in predictions:
            split = pred['split']
            if split not in split_performance:
                split_performance[split] = {'accuracies': [], 'confidences': []}
            
            if 'accuracy' in pred['analysis']:
                split_performance[split]['accuracies'].append(pred['analysis']['accuracy'])
            split_performance[split]['confidences'].append(pred['analysis']['mean_confidence'])
        
        splits = list(split_performance.keys())
        if splits and any(split_performance[s]['accuracies'] for s in splits):
            split_accs = [np.mean(split_performance[s]['accuracies']) if split_performance[s]['accuracies'] else 0 
                         for s in splits]
            split_confs = [np.mean(split_performance[s]['confidences']) for s in splits]
            
            x = np.arange(len(splits))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, split_accs, width, label='Accuracy', alpha=0.8)
            axes[1, 1].bar(x + width/2, split_confs, width, label='Confidence', alpha=0.8)
            axes[1, 1].set_xlabel('Dataset Split')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Performance by Split')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([s.upper() for s in splits])
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No Split\\nComparison\\nAvailable', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Performance by Split')
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        
        summary_text = f"""
PREDICTION SUMMARY
{'='*20}

Total Predictions: {len(predictions)}
Mean Accuracy: {np.mean(accuracies):.3f if accuracies else 'N/A'}
Mean Confidence: {np.mean(confidences):.3f}
Mean Low Confidence: {np.mean(low_conf_percentages):.1f}%

Classes Detected: {len(all_classes)}
Most Common Class: {classes[mean_percentages.index(max(mean_percentages))]}
Least Common Class: {classes[mean_percentages.index(min(mean_percentages))]}

Performance Grade:
{self.get_performance_grade(np.mean(accuracies) if accuracies else 0.5)}

Files Generated: {len(predictions)} images
Output Directory: {self.output_dir}
"""
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def get_performance_grade(self, accuracy):
        """Get performance grade"""
        if accuracy >= 0.95:
            return "🏆 EXCELLENT (A+)"
        elif accuracy >= 0.90:
            return "🥇 VERY GOOD (A)"
        elif accuracy >= 0.85:
            return "🥈 GOOD (B+)"
        elif accuracy >= 0.80:
            return "🥉 FAIR (B)"
        elif accuracy >= 0.70:
            return "📈 NEEDS IMPROVEMENT (C)"
        else:
            return "⚠️ POOR (D)"
    
    def save_predictions_json(self, predictions):
        """Save predictions data to JSON"""
        output_file = self.output_dir / 'predictions_data.json'
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        predictions_serializable = convert_numpy_types(predictions)
        
        with open(output_file, 'w') as f:
            json.dump(predictions_serializable, f, indent=2)
        
        print(f"💾 Predictions data saved to: {output_file}")
    
    def run_comprehensive_prediction(self, num_predictions=25):
        """Run enhanced prediction pipeline with water discrimination"""
        print("🚀 STARTING ENHANCED PREDICTION GENERATION")
        print("=" * 60)
        print("Features: Water discrimination, TTA, post-processing")
        
        # Generate predictions
        predictions = self.generate_diverse_predictions(num_predictions)
        
        if not predictions:
            print("❌ No predictions generated!")
            return
        
        # Create summary
        self.create_prediction_summary(predictions)
        
        # Save data
        self.save_predictions_json(predictions)
        
        print(f"\\n✅ ENHANCED PREDICTION GENERATION COMPLETE!")
        print(f"📊 Generated {len(predictions)} predictions with water discrimination")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"🖼️ Individual predictions: prediction_XX_*.png")
        print(f"📈 Summary chart: prediction_summary.png")
        print(f"💾 Data file: predictions_data.json")
        print(f"🌊 Water misclassification: Reduced through post-processing")
        
        return predictions

def main():
    """Main function for enhanced predictions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced predictions with water discrimination')
    parser.add_argument('--model', default='best_advanced_slum_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--data', default='data', help='Data directory')
    parser.add_argument('--output', default='predictions', help='Output directory')
    parser.add_argument('--num', type=int, default=25, help='Number of predictions to generate')
    
    args = parser.parse_args()
    
    predictor = AdvancedPredictor(
        model_path=args.model,
        data_root=args.data,
        output_dir=args.output
    )
    
    predictor.run_comprehensive_prediction(args.num)

if __name__ == "__main__":
    main()