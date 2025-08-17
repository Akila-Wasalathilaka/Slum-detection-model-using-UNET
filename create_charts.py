#!/usr/bin/env python3
"""
Comprehensive Chart Generation for Slum Detection
================================================
Creates detailed analysis charts and visualizations.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChartGenerator:
    def __init__(self, model_path=None, data_root="data", output_dir="charts"):
        self.model_path = model_path
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Class information
        self.class_names = {
            0: "Background",
            105: "Slum Type A", 
            109: "Slum Main Type",
            111: "Slum Type B",
            158: "Slum Type C",
            200: "Slum Type D",
            233: "Slum Type E"
        }
        
        self.class_colors = {
            0: '#2E2E2E',    # Dark gray for background
            105: '#FF6B6B',  # Red
            109: '#4ECDC4',  # Teal  
            111: '#45B7D1',  # Blue
            158: '#96CEB4',  # Green
            200: '#FFEAA7',  # Yellow
            233: '#DDA0DD'   # Plum
        }
    
    def load_model_if_available(self):
        """Load model if path is provided"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                import segmentation_models_pytorch as smp
                from advanced_training import AdvancedUNet
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = AdvancedUNet(encoder_name='efficientnet-b3', num_classes=7)
                model.load_state_dict(torch.load(self.model_path, map_location=device))
                model.to(device)
                model.eval()
                return model, device
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                return None, None
        return None, None
    
    def create_dataset_overview_chart(self):
        """Create comprehensive dataset overview"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Dataset size by split
        splits = ['train', 'val', 'test']
        image_counts = []
        mask_counts = []
        
        for split in splits:
            img_dir = self.data_root / split / "images"
            mask_dir = self.data_root / split / "masks"
            
            img_count = len(list(img_dir.glob("*.tif"))) if img_dir.exists() else 0
            mask_count = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
            
            image_counts.append(img_count)
            mask_counts.append(mask_count)
        
        x = np.arange(len(splits))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, image_counts, width, label='Images', alpha=0.8)
        bars2 = axes[0, 0].bar(x + width/2, mask_counts, width, label='Masks', alpha=0.8)
        
        axes[0, 0].set_xlabel('Dataset Split')
        axes[0, 0].set_ylabel('File Count')
        axes[0, 0].set_title('Dataset Size Distribution')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([s.upper() for s in splits])
        axes[0, 0].legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(image_counts + mask_counts)*0.01,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Class distribution analysis
        class_distribution = self.analyze_class_distribution()
        
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        colors = [self.class_colors.get(cls, '#888888') for cls in classes]
        
        bars = axes[0, 1].bar(classes, counts, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 1].set_xlabel('Class Value')
        axes[0, 1].set_ylabel('Pixel Count')
        axes[0, 1].set_title('Class Distribution (Log Scale)')
        axes[0, 1].set_yscale('log')
        
        # Add percentage labels
        total_pixels = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_pixels) * 100
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height * 1.1,
                           f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Class distribution pie chart
        labels = [f'{self.class_names[cls]}\\n({cls})' for cls in classes]
        percentages = [(count/total_pixels)*100 for count in counts]
        
        wedges, texts, autotexts = axes[0, 2].pie(percentages, labels=labels, autopct='%1.1f%%', 
                                                 colors=colors, startangle=90)
        axes[0, 2].set_title('Class Distribution Percentage')
        
        # 4. Sample images grid
        self.create_sample_grid(axes[1, 0])
        
        # 5. Class imbalance analysis
        self.create_imbalance_analysis(axes[1, 1], class_distribution)
        
        # 6. Dataset statistics
        axes[1, 2].axis('off')
        total_images = sum(image_counts)
        total_masks = sum(mask_counts)
        
        stats_text = f"""
DATASET STATISTICS
{'='*25}

Total Images: {total_images:,}
Total Masks: {total_masks:,}
Total Classes: {len(classes)}

Split Distribution:
• TRAIN: {image_counts[0]:,} ({(image_counts[0]/total_images)*100:.1f}%)
• VAL: {image_counts[1]:,} ({(image_counts[1]/total_images)*100:.1f}%)
• TEST: {image_counts[2]:,} ({(image_counts[2]/total_images)*100:.1f}%)

Most Common Class: {self.class_names[classes[counts.index(max(counts))]]}
Least Common Class: {self.class_names[classes[counts.index(min(counts))]]}

Imbalance Ratio: {max(counts)/min(counts):.1f}:1
"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_class_distribution(self):
        """Analyze class distribution across all masks"""
        print("🔍 Analyzing class distribution...")
        class_counts = {}
        
        for split in ['train', 'val', 'test']:
            mask_dir = self.data_root / split / "masks"
            if not mask_dir.exists():
                continue
            
            mask_files = list(mask_dir.glob("*.png"))
            for mask_path in tqdm(mask_files[:500], desc=f"Analyzing {split}"):  # Sample for speed
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                unique_vals, counts = np.unique(mask, return_counts=True)
                
                for val, count in zip(unique_vals, counts):
                    if val not in class_counts:
                        class_counts[val] = 0
                    class_counts[val] += int(count)
        
        return class_counts
    
    def create_sample_grid(self, ax):
        """Create a grid of sample images and masks"""
        ax.set_title('Sample Images and Masks')
        
        # Create a composite image showing samples
        samples = []
        train_img_dir = self.data_root / "train" / "images"
        train_mask_dir = self.data_root / "train" / "masks"
        
        if train_img_dir.exists() and train_mask_dir.exists():
            img_files = list(train_img_dir.glob("*.tif"))[:4]
            
            for img_path in img_files:
                mask_path = train_mask_dir / (img_path.stem + ".png")
                
                if mask_path.exists():
                    # Load and resize
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (64, 64))
                    
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (64, 64))
                    
                    # Create colored mask
                    colored_mask = np.zeros((64, 64, 3), dtype=np.uint8)
                    for class_val, color_hex in self.class_colors.items():
                        if class_val in np.unique(mask):
                            color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                            colored_mask[mask == class_val] = color_rgb
                    
                    # Combine image and mask
                    combined = np.hstack([img, colored_mask])
                    samples.append(combined)
            
            if samples:
                # Stack samples vertically
                grid = np.vstack(samples)
                ax.imshow(grid)
                ax.set_xticks([32, 96])
                ax.set_xticklabels(['Image', 'Mask'])
                ax.set_yticks([])
        
        ax.axis('off')
    
    def create_imbalance_analysis(self, ax, class_distribution):
        """Create class imbalance analysis"""
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        # Calculate imbalance ratios
        max_count = max(counts)
        ratios = [max_count / count for count in counts]
        
        colors = [self.class_colors.get(cls, '#888888') for cls in classes]
        bars = ax.bar(range(len(classes)), ratios, color=colors, alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Imbalance Ratio')
        ax.set_title('Class Imbalance Analysis')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f'{cls}' for cls in classes])
        ax.set_yscale('log')
        
        # Add ratio labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{ratio:.1f}:1', ha='center', va='bottom', fontsize=9)
    
    def create_model_performance_charts(self):
        """Create model performance analysis charts"""
        model, device = self.load_model_if_available()
        
        if model is None:
            print("⚠️ Model not available for performance analysis")
            return
        
        print("📊 Creating model performance charts...")
        
        # Load validation data for evaluation
        val_img_dir = self.data_root / "val" / "images"
        val_mask_dir = self.data_root / "val" / "masks"
        
        if not (val_img_dir.exists() and val_mask_dir.exists()):
            print("❌ Validation data not found")
            return
        
        # Evaluate model on validation set
        predictions, ground_truths = self.evaluate_model(model, device, val_img_dir, val_mask_dir)
        
        # Create performance charts
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        self.create_confusion_matrix(axes[0, 0], predictions, ground_truths)
        
        # 2. Per-class metrics
        self.create_class_metrics_chart(axes[0, 1], predictions, ground_truths)
        
        # 3. Prediction confidence distribution
        self.create_confidence_distribution(axes[0, 2], model, device, val_img_dir)
        
        # 4. Sample predictions
        self.create_prediction_samples(axes[1, 0], model, device, val_img_dir, val_mask_dir)
        
        # 5. Error analysis
        self.create_error_analysis(axes[1, 1], predictions, ground_truths)
        
        # 6. Performance summary
        self.create_performance_summary(axes[1, 2], predictions, ground_truths)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, model, device, img_dir, mask_dir, max_samples=200):
        """Evaluate model on validation set"""
        print("🔍 Evaluating model...")
        
        img_files = list(img_dir.glob("*.tif"))[:max_samples]
        all_predictions = []
        all_ground_truths = []
        
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        with torch.no_grad():
            for img_path in tqdm(img_files, desc="Evaluating"):
                mask_path = mask_dir / (img_path.stem + ".png")
                
                if mask_path.exists():
                    # Load and preprocess
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    # Apply transforms
                    augmented = transform(image=img, mask=mask)
                    img_tensor = augmented['image'].unsqueeze(0).to(device)
                    mask_tensor = augmented['mask']
                    
                    # Predict
                    output = model(img_tensor)
                    pred = output.argmax(dim=1).squeeze().cpu().numpy()
                    
                    all_predictions.append(pred.flatten())
                    all_ground_truths.append(mask_tensor.numpy().flatten())
        
        return np.concatenate(all_predictions), np.concatenate(all_ground_truths)
    
    def create_confusion_matrix(self, ax, predictions, ground_truths):
        """Create confusion matrix"""
        # Sample data for speed
        sample_size = min(100000, len(predictions))
        indices = np.random.choice(len(predictions), sample_size, replace=False)
        
        cm = confusion_matrix(ground_truths[indices], predictions[indices])
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix (Normalized)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add labels
        classes = sorted(list(set(ground_truths) | set(predictions)))
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=8)
    
    def create_class_metrics_chart(self, ax, predictions, ground_truths):
        """Create per-class metrics chart"""
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truths, predictions, average=None, zero_division=0
        )
        
        classes = sorted(list(set(ground_truths) | set(predictions)))
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precision[:len(classes)], width, label='Precision', alpha=0.8)
        ax.bar(x, recall[:len(classes)], width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1[:len(classes)], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1)
    
    def create_confidence_distribution(self, ax, model, device, img_dir):
        """Create prediction confidence distribution"""
        print("📊 Analyzing prediction confidence...")
        
        confidences = []
        img_files = list(img_dir.glob("*.tif"))[:50]  # Sample for speed
        
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        with torch.no_grad():
            for img_path in tqdm(img_files, desc="Confidence analysis"):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                augmented = transform(image=img)
                img_tensor = augmented['image'].unsqueeze(0).to(device)
                
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                max_probs = probs.max(dim=1)[0]
                
                confidences.extend(max_probs.cpu().numpy().flatten())
        
        ax.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Confidence Distribution')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(confidences):.3f}')
        ax.legend()
    
    def create_prediction_samples(self, ax, model, device, img_dir, mask_dir):
        """Create sample predictions visualization"""
        ax.set_title('Sample Predictions')
        ax.axis('off')
        
        # This would be implemented similar to the sample grid
        # but showing model predictions vs ground truth
        ax.text(0.5, 0.5, 'Sample Predictions\\n(Implementation depends on model availability)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    def create_error_analysis(self, ax, predictions, ground_truths):
        """Create error analysis chart"""
        # Calculate per-class error rates
        classes = sorted(list(set(ground_truths) | set(predictions)))
        error_rates = []
        
        for cls in classes:
            mask = ground_truths == cls
            if mask.sum() > 0:
                errors = (predictions[mask] != cls).sum()
                error_rate = errors / mask.sum()
                error_rates.append(error_rate)
            else:
                error_rates.append(0)
        
        colors = [self.class_colors.get(cls, '#888888') for cls in classes]
        bars = ax.bar(range(len(classes)), error_rates, color=colors, alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Error Rate')
        ax.set_title('Per-Class Error Analysis')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1)
        
        # Add percentage labels
        for bar, rate in zip(bars, error_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    def create_performance_summary(self, ax, predictions, ground_truths):
        """Create performance summary"""
        ax.axis('off')
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Calculate overall metrics
        accuracy = accuracy_score(ground_truths, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truths, predictions, average='weighted', zero_division=0
        )
        
        summary_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*30}

Overall Accuracy: {accuracy:.4f}
Weighted Precision: {precision:.4f}
Weighted Recall: {recall:.4f}
Weighted F1-Score: {f1:.4f}

Total Samples Evaluated: {len(predictions):,}
Number of Classes: {len(set(ground_truths))}

Performance Grade:
{self.get_performance_grade(accuracy)}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    def get_performance_grade(self, accuracy):
        """Get performance grade based on accuracy"""
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
    
    def create_training_analysis_charts(self):
        """Create training analysis charts if history is available"""
        history_file = "advanced_training_history.json"
        
        if not os.path.exists(history_file):
            print("⚠️ Training history not found")
            return
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate schedule
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training summary
        axes[1, 1].axis('off')
        
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        summary_text = f"""
TRAINING SUMMARY
{'='*20}

Total Epochs: {len(epochs)}
Best Validation Accuracy: {best_val_acc:.4f}
Best Epoch: {best_epoch}

Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}

Convergence: {'✅ Good' if final_val_loss < final_train_loss * 1.2 else '⚠️ Possible Overfitting'}
Stability: {'✅ Stable' if np.std(history['val_acc'][-5:]) < 0.01 else '⚠️ Unstable'}
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_charts(self):
        """Generate all available charts"""
        print("📊 GENERATING COMPREHENSIVE CHARTS")
        print("=" * 50)
        
        # Dataset overview
        print("1. Creating dataset overview charts...")
        self.create_dataset_overview_chart()
        
        # Training analysis
        print("2. Creating training analysis charts...")
        self.create_training_analysis_charts()
        
        # Model performance (if model available)
        print("3. Creating model performance charts...")
        self.create_model_performance_charts()
        
        print(f"\\n✅ All charts saved to: {self.output_dir}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive charts')
    parser.add_argument('--model', default='best_advanced_slum_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--data', default='data', help='Data directory')
    parser.add_argument('--output', default='charts', help='Output directory')
    
    args = parser.parse_args()
    
    generator = ChartGenerator(
        model_path=args.model,
        data_root=args.data,
        output_dir=args.output
    )
    
    generator.generate_all_charts()

if __name__ == "__main__":
    main()