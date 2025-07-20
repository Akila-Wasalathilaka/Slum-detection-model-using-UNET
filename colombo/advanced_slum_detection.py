#!/usr/bin/env python3
"""
Colombo Slum Detection with Red Overlay Mapping
==============================================

Advanced slum detection system for Colombo tiles with accurate predictions
and red overlay visualization. Uses the exact same approach as successful
training analysis to ensure zero issues.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from models import create_model
from config import get_model_config, get_data_config
from utils.checkpoint import load_checkpoint
from utils.transforms import get_test_transforms
from utils.dataset import SlumDataset

class ColomboSlumDetectionSystem:
    """Advanced slum detection system for Colombo with red overlay mapping."""
    
    def __init__(self, checkpoint_path, output_dir="colombo_analysis"):
        """Initialize the detection system."""
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device}")
        
        # Load configurations (exact same as successful analysis)
        self.model_config = get_model_config('balanced')
        self.data_config = get_data_config('standard')
        
        # Load model
        self.model = self._load_model()
        
        # Get exact same transforms as successful analysis
        self.test_transforms = get_test_transforms(self.data_config)
        
        print(f"🤖 Colombo Slum Detection System initialized")
        print(f"   Model: {self.model_config.architecture} with {self.model_config.encoder}")
        print(f"   Output: {self.output_dir}")
    
    def _load_model(self):
        """Load model using exact same approach as successful analysis."""
        print(f"📂 Loading model from: {self.checkpoint_path}")
        
        # Create model - exact same as successful analysis
        model = create_model(
            architecture=self.model_config.architecture,
            encoder=self.model_config.encoder,
            pretrained=False,
            num_classes=self.model_config.num_classes
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(self.checkpoint_path, model, device=self.device)
        model = model.to(self.device)
        model.eval()
        
        print(f"   ✅ Model loaded successfully")
        return model
    
    def load_colombo_tiles(self, tiles_dir):
        """Load all Colombo tiles for processing."""
        tiles_dir = Path(tiles_dir)
        print(f"📂 Loading Colombo tiles from: {tiles_dir}")
        
        # Get all tile files
        tile_files = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg")))
        
        if not tile_files:
            raise ValueError(f"No tile files found in {tiles_dir}")
        
        print(f"   Found {len(tile_files)} tiles")
        
        # Load tile images and metadata
        tiles_data = []
        
        for i, tile_file in enumerate(tile_files):
            # Load image
            image = Image.open(tile_file).convert('RGB')
            image_np = np.array(image)
            
            # Extract tile position from filename (assuming format: tile_XXXX.png)
            tile_name = tile_file.stem
            if 'tile_' in tile_name:
                try:
                    tile_id = int(tile_name.split('_')[-1])
                except:
                    tile_id = i
            else:
                tile_id = i
            
            tiles_data.append({
                'id': tile_id,
                'path': str(tile_file),
                'name': tile_name,
                'image': image_np,
                'size': image_np.shape[:2]
            })
        
        print(f"   ✅ Loaded {len(tiles_data)} tiles")
        return tiles_data
    
    def predict_slums_for_tiles(self, tiles_data, threshold=0.5):
        """Generate slum predictions for all tiles."""
        print(f"🔍 Generating slum predictions for {len(tiles_data)} tiles...")
        
        predictions_data = []
        
        for i, tile_data in enumerate(tiles_data):
            try:
                # Get image
                image = tile_data['image']
                
                # Apply exact same transforms as successful analysis
                if self.test_transforms:
                    transformed = self.test_transforms(image=image)
                    image_tensor = transformed['image']
                else:
                    # Fallback - same normalization as training
                    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    # Apply ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_tensor = (image_tensor - mean) / std
                
                # Add batch dimension
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Generate prediction
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    prediction_map = torch.sigmoid(outputs).cpu().numpy()
                
                # Remove batch dimension
                if len(prediction_map.shape) == 4:
                    prediction_map = prediction_map[0, 0]  # Remove batch and channel dims
                elif len(prediction_map.shape) == 3:
                    prediction_map = prediction_map[0]  # Remove batch dim
                
                # Calculate statistics
                avg_probability = float(np.mean(prediction_map))
                max_probability = float(np.max(prediction_map))
                slum_pixels = int(np.sum(prediction_map > threshold))
                total_pixels = int(prediction_map.size)
                slum_percentage = (slum_pixels / total_pixels) * 100
                
                # Create binary mask
                binary_mask = (prediction_map > threshold).astype(np.uint8)
                
                prediction_data = {
                    'tile_id': tile_data['id'],
                    'tile_name': tile_data['name'],
                    'tile_path': tile_data['path'],
                    'prediction_map': prediction_map,
                    'binary_mask': binary_mask,
                    'avg_probability': avg_probability,
                    'max_probability': max_probability,
                    'slum_pixels': slum_pixels,
                    'total_pixels': total_pixels,
                    'slum_percentage': slum_percentage,
                    'classification': 'slum' if avg_probability > threshold else 'non-slum',
                    'confidence': 'high' if abs(avg_probability - 0.5) > 0.3 else 'medium' if abs(avg_probability - 0.5) > 0.1 else 'low'
                }
                
                predictions_data.append(prediction_data)
                
                # Progress update
                if (i + 1) % 10 == 0 or i == len(tiles_data) - 1:
                    print(f"   Progress: {i + 1}/{len(tiles_data)} tiles processed")
            
            except Exception as e:
                print(f"   ❌ Error processing tile {tile_data['name']}: {e}")
        
        print(f"   ✅ Generated predictions for {len(predictions_data)} tiles")
        return predictions_data
    
    def create_red_overlay_visualizations(self, tiles_data, predictions_data):
        """Create red overlay visualizations for slum areas."""
        print("🎨 Creating red overlay visualizations...")
        
        viz_dir = self.output_dir / "red_overlays"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        for tile_data, pred_data in zip(tiles_data, predictions_data):
            try:
                # Get original image and prediction
                original_image = tile_data['image']
                prediction_map = pred_data['prediction_map']
                binary_mask = pred_data['binary_mask']
                
                # Create red overlay
                overlay_image = original_image.copy()
                
                # Apply red overlay where slums are detected
                red_overlay = np.zeros_like(original_image)
                red_overlay[:, :, 0] = binary_mask * 255  # Red channel
                
                # Blend original image with red overlay
                alpha = 0.4  # Transparency of red overlay
                overlay_image = cv2.addWeighted(original_image, 1-alpha, red_overlay, alpha, 0)
                
                # Create visualization with multiple views
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original image
                axes[0].imshow(original_image)
                axes[0].set_title(f"Original\n{pred_data['tile_name']}", fontsize=10)
                axes[0].axis('off')
                
                # Prediction heatmap
                im = axes[1].imshow(prediction_map, cmap='hot', vmin=0, vmax=1)
                axes[1].set_title(f"Prediction Map\nAvg: {pred_data['avg_probability']:.3f}", fontsize=10)
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                # Binary mask
                axes[2].imshow(binary_mask, cmap='Reds', vmin=0, vmax=1)
                axes[2].set_title(f"Binary Mask\nSlum: {pred_data['slum_percentage']:.1f}%", fontsize=10)
                axes[2].axis('off')
                
                # Red overlay
                axes[3].imshow(overlay_image)
                axes[3].set_title(f"Red Overlay\n{pred_data['classification'].title()}", fontsize=10)
                axes[3].axis('off')
                
                # Overall title
                confidence_color = 'red' if pred_data['classification'] == 'slum' else 'green'
                fig.suptitle(f"Tile {pred_data['tile_id']:04d} - {pred_data['classification'].upper()} Detection", 
                           fontweight='bold', color=confidence_color, fontsize=14)
                
                plt.tight_layout()
                
                # Save visualization
                viz_file = viz_dir / f"tile_{pred_data['tile_id']:04d}_overlay.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                created_files.append(str(viz_file))
                
            except Exception as e:
                print(f"   ⚠️ Error creating overlay for {pred_data['tile_name']}: {e}")
        
        print(f"   ✅ Created {len(created_files)} red overlay visualizations")
        return created_files
    
    def create_summary_visualizations(self, tiles_data, predictions_data):
        """Create summary visualizations and analysis."""
        print("📊 Creating summary visualizations...")
        
        # 1. Grid view of all tiles with overlays
        print("   Creating grid view...")
        grid_size = int(np.ceil(np.sqrt(len(tiles_data))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten() if grid_size > 1 else [axes]
        
        for i, (tile_data, pred_data) in enumerate(zip(tiles_data, predictions_data)):
            if i < len(axes):
                # Create red overlay
                original_image = tile_data['image']
                binary_mask = pred_data['binary_mask']
                
                overlay_image = original_image.copy()
                red_overlay = np.zeros_like(original_image)
                red_overlay[:, :, 0] = binary_mask * 255
                overlay_image = cv2.addWeighted(original_image, 0.7, red_overlay, 0.3, 0)
                
                axes[i].imshow(overlay_image)
                
                # Title with classification
                color = 'red' if pred_data['classification'] == 'slum' else 'green'
                title = f"T{pred_data['tile_id']:02d}\nP:{pred_data['avg_probability']:.3f}"
                axes[i].set_title(title, fontsize=8, color=color, fontweight='bold')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(tiles_data), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Colombo Slum Detection - All Tiles with Red Overlays', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        grid_file = self.output_dir / "all_tiles_grid_overlay.png"
        plt.savefig(grid_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Statistics summary
        print("   Creating statistics summary...")
        probabilities = [p['avg_probability'] for p in predictions_data]
        slum_percentages = [p['slum_percentage'] for p in predictions_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Probability distribution
        axes[0, 0].hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(probabilities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(probabilities):.3f}')
        axes[0, 0].axvline(0.5, color='orange', linestyle='--', label='Threshold: 0.5')
        axes[0, 0].set_xlabel('Average Probability')
        axes[0, 0].set_ylabel('Number of Tiles')
        axes[0, 0].set_title('Distribution of Slum Probabilities')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Slum percentage distribution
        axes[0, 1].hist(slum_percentages, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(np.mean(slum_percentages), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(slum_percentages):.1f}%')
        axes[0, 1].set_xlabel('Slum Percentage (%)')
        axes[0, 1].set_ylabel('Number of Tiles')
        axes[0, 1].set_title('Distribution of Slum Coverage')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Classification summary
        classifications = [p['classification'] for p in predictions_data]
        class_counts = {cls: classifications.count(cls) for cls in set(classifications)}
        
        colors = ['green' if cls == 'non-slum' else 'red' for cls in class_counts.keys()]
        axes[1, 0].bar(class_counts.keys(), class_counts.values(), color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Number of Tiles')
        axes[1, 0].set_title('Classification Summary')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, (cls, count) in enumerate(class_counts.items()):
            axes[1, 0].text(i, count + 0.1, str(count), ha='center', fontweight='bold')
        
        # Confidence distribution
        confidences = [p['confidence'] for p in predictions_data]
        conf_counts = {conf: confidences.count(conf) for conf in ['low', 'medium', 'high']}
        
        conf_colors = {'low': 'red', 'medium': 'orange', 'high': 'green'}
        colors = [conf_colors[conf] for conf in conf_counts.keys()]
        axes[1, 1].bar(conf_counts.keys(), conf_counts.values(), color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Number of Tiles')
        axes[1, 1].set_title('Confidence Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, (conf, count) in enumerate(conf_counts.items()):
            axes[1, 1].text(i, count + 0.1, str(count), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        summary_file = self.output_dir / "detection_summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Summary visualizations saved")
        return str(grid_file), str(summary_file)
    
    def save_comprehensive_results(self, predictions_data, visualization_files):
        """Save comprehensive results and analysis."""
        print("💾 Saving comprehensive results...")
        
        # Calculate overall statistics
        total_tiles = len(predictions_data)
        slum_tiles = sum(1 for p in predictions_data if p['classification'] == 'slum')
        avg_probability = np.mean([p['avg_probability'] for p in predictions_data])
        max_probability = np.max([p['avg_probability'] for p in predictions_data])
        avg_slum_percentage = np.mean([p['slum_percentage'] for p in predictions_data])
        
        # Confidence breakdown
        confidences = [p['confidence'] for p in predictions_data]
        confidence_counts = {conf: confidences.count(conf) for conf in ['low', 'medium', 'high']}
        
        # Create comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'checkpoint': str(self.checkpoint_path),
                'architecture': self.model_config.architecture,
                'encoder': self.model_config.encoder,
                'device': str(self.device)
            },
            'overall_statistics': {
                'total_tiles': total_tiles,
                'slum_tiles': slum_tiles,
                'non_slum_tiles': total_tiles - slum_tiles,
                'slum_detection_rate': (slum_tiles / total_tiles) * 100,
                'average_probability': float(avg_probability),
                'max_probability': float(max_probability),
                'average_slum_coverage': float(avg_slum_percentage),
                'confidence_distribution': confidence_counts
            },
            'tile_predictions': [
                {
                    'tile_id': p['tile_id'],
                    'tile_name': p['tile_name'],
                    'classification': p['classification'],
                    'avg_probability': p['avg_probability'],
                    'max_probability': p['max_probability'],
                    'slum_percentage': p['slum_percentage'],
                    'confidence': p['confidence']
                }
                for p in predictions_data
            ],
            'visualization_files': visualization_files,
            'methodology': {
                'preprocessing': 'ImageNet normalization with test-time transforms',
                'threshold': 0.5,
                'overlay_transparency': 0.4,
                'red_channel_intensity': 255
            }
        }
        
        # Save results
        results_file = self.output_dir / "colombo_slum_detection_complete.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        report_lines = [
            "COLOMBO SLUM DETECTION ANALYSIS REPORT",
            "=" * 50,
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.model_config.architecture} with {self.model_config.encoder}",
            "",
            "OVERALL STATISTICS:",
            f"  Total tiles analyzed: {total_tiles}",
            f"  Slum areas detected: {slum_tiles} tiles ({(slum_tiles/total_tiles)*100:.1f}%)",
            f"  Non-slum areas: {total_tiles - slum_tiles} tiles ({((total_tiles-slum_tiles)/total_tiles)*100:.1f}%)",
            f"  Average slum probability: {avg_probability:.3f}",
            f"  Maximum slum probability: {max_probability:.3f}",
            f"  Average slum coverage: {avg_slum_percentage:.1f}%",
            "",
            "CONFIDENCE BREAKDOWN:",
            f"  High confidence: {confidence_counts.get('high', 0)} tiles",
            f"  Medium confidence: {confidence_counts.get('medium', 0)} tiles",
            f"  Low confidence: {confidence_counts.get('low', 0)} tiles",
            "",
            "METHODOLOGY:",
            "  - Used exact same preprocessing as successful training analysis",
            "  - Applied ImageNet normalization and test-time transforms",
            "  - Generated spatial probability maps for each tile",
            "  - Created red overlays with 40% transparency",
            "  - Classification threshold: 0.5",
            "",
            "OUTPUT FILES:",
            f"  - Individual tile overlays: {len(visualization_files)} files",
            f"  - Grid overview: all_tiles_grid_overlay.png",
            f"  - Statistics summary: detection_summary.png",
            f"  - Complete results: colombo_slum_detection_complete.json"
        ]
        
        report_file = self.output_dir / "ANALYSIS_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   ✅ Results saved to: {results_file}")
        print(f"   ✅ Report saved to: {report_file}")
        
        return str(results_file), str(report_file)
    
    def run_complete_analysis(self, tiles_dir, threshold=0.5):
        """Run complete slum detection analysis with red overlays."""
        print(f"\n🏘️ COLOMBO SLUM DETECTION WITH RED OVERLAYS")
        print("=" * 60)
        print(f"Tiles directory: {tiles_dir}")
        print(f"Detection threshold: {threshold}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        try:
            # Step 1: Load tiles
            tiles_data = self.load_colombo_tiles(tiles_dir)
            
            # Step 2: Generate predictions
            predictions_data = self.predict_slums_for_tiles(tiles_data, threshold)
            
            # Step 3: Create red overlay visualizations
            overlay_files = self.create_red_overlay_visualizations(tiles_data, predictions_data)
            
            # Step 4: Create summary visualizations
            grid_file, summary_file = self.create_summary_visualizations(tiles_data, predictions_data)
            
            # Step 5: Save comprehensive results
            all_viz_files = overlay_files + [grid_file, summary_file]
            results_file, report_file = self.save_comprehensive_results(predictions_data, all_viz_files)
            
            # Print final summary
            total_tiles = len(predictions_data)
            slum_tiles = sum(1 for p in predictions_data if p['classification'] == 'slum')
            avg_prob = np.mean([p['avg_probability'] for p in predictions_data])
            
            print(f"\n✅ ANALYSIS COMPLETE!")
            print("=" * 60)
            print(f"🏘️ FINAL RESULTS:")
            print(f"   📊 Total tiles analyzed: {total_tiles}")
            print(f"   🚨 Slum areas detected: {slum_tiles} tiles ({(slum_tiles/total_tiles)*100:.1f}%)")
            print(f"   📈 Average detection confidence: {avg_prob:.3f}")
            print(f"   🎨 Red overlay visualizations: {len(overlay_files)} files")
            print(f"   📁 All outputs saved to: {self.output_dir}")
            print("=" * 60)
            
            return {
                'tiles_analyzed': total_tiles,
                'slum_tiles_detected': slum_tiles,
                'detection_rate': (slum_tiles/total_tiles)*100,
                'average_confidence': avg_prob,
                'results_file': results_file,
                'report_file': report_file,
                'visualization_files': all_viz_files
            }
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function for running the complete analysis."""
    
    # Configuration
    checkpoint_path = "../experiments/development_20250713_175410/checkpoints/best_checkpoint.pth"
    tiles_dir = "tiles"
    output_dir = "complete_analysis"
    
    print(f"🚀 COLOMBO SLUM DETECTION SYSTEM")
    print("=" * 50)
    
    # Initialize system
    system = ColomboSlumDetectionSystem(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir
    )
    
    # Run complete analysis
    results = system.run_complete_analysis(
        tiles_dir=tiles_dir,
        threshold=0.5
    )
    
    if results:
        print(f"\n🎉 Colombo slum detection completed successfully!")
        print(f"Found {results['slum_tiles_detected']} potential slum areas")
        print(f"Detection rate: {results['detection_rate']:.1f}%")
    else:
        print(f"\n❌ Analysis failed!")

if __name__ == "__main__":
    main()
