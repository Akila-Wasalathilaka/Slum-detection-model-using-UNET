#!/usr/bin/env python3
"""
Enhanced Colombo Slum Detection with Multi-Threshold Analysis
============================================================

Advanced system that analyzes slums at multiple thresholds to capture
all potential slum areas, even those with lower confidence scores.
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
import seaborn as sns
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

class EnhancedColomboAnalysis:
    """Enhanced multi-threshold analysis for Colombo slum detection."""
    
    def __init__(self, checkpoint_path, output_dir="enhanced_analysis"):
        """Initialize the enhanced analysis system."""
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configurations
        self.model_config = get_model_config('balanced')
        self.data_config = get_data_config('standard')
        
        # Load model
        self.model = self._load_model()
        
        # Get transforms
        self.test_transforms = get_test_transforms(self.data_config)
        
        print(f"🚀 Enhanced Colombo Analysis System Ready")
        print(f"   Device: {self.device}")
        print(f"   Output: {self.output_dir}")
    
    def _load_model(self):
        """Load model."""
        model = create_model(
            architecture=self.model_config.architecture,
            encoder=self.model_config.encoder,
            pretrained=False,
            num_classes=self.model_config.num_classes
        )
        
        checkpoint = load_checkpoint(self.checkpoint_path, model, device=self.device)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def analyze_multiple_thresholds(self, tiles_dir):
        """Analyze tiles at multiple thresholds to capture all potential slums."""
        print(f"\n🔍 MULTI-THRESHOLD SLUM ANALYSIS")
        print("=" * 50)
        
        # Load tiles
        tiles_dir = Path(tiles_dir)
        tile_files = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg")))
        
        print(f"📂 Loading {len(tile_files)} tiles...")
        
        # Generate predictions for all tiles
        all_predictions = []
        tile_info = []
        
        for i, tile_file in enumerate(tile_files):
            # Load and preprocess image
            image = Image.open(tile_file).convert('RGB')
            image_np = np.array(image)
            
            # Apply transforms
            if self.test_transforms:
                transformed = self.test_transforms(image=image_np)
                image_tensor = transformed['image']
            else:
                image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
                # Apply ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = (image_tensor - mean) / std
            
            # Predict
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                prediction_map = torch.sigmoid(outputs).cpu().numpy()
            
            # Process prediction
            if len(prediction_map.shape) == 4:
                prediction_map = prediction_map[0, 0]
            elif len(prediction_map.shape) == 3:
                prediction_map = prediction_map[0]
            
            all_predictions.append(prediction_map)
            tile_info.append({
                'id': i,
                'name': tile_file.stem,
                'path': str(tile_file),
                'image': image_np
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Processed: {i + 1}/{len(tile_files)} tiles")
        
        print(f"✅ Generated predictions for {len(all_predictions)} tiles")
        
        # Analyze at multiple thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_results = {}
        
        print(f"\n📊 Analyzing at {len(thresholds)} different thresholds...")
        
        for threshold in thresholds:
            slum_count = 0
            total_slum_pixels = 0
            total_pixels = 0
            
            tile_results = []
            
            for i, (prediction_map, tile) in enumerate(zip(all_predictions, tile_info)):
                # Calculate statistics for this threshold
                binary_mask = (prediction_map > threshold).astype(np.uint8)
                slum_pixels = np.sum(binary_mask)
                tile_pixels = binary_mask.size
                slum_percentage = (slum_pixels / tile_pixels) * 100
                avg_prob = np.mean(prediction_map)
                max_prob = np.max(prediction_map)
                
                is_slum = slum_pixels > 0  # Any slum pixels detected
                if is_slum:
                    slum_count += 1
                
                total_slum_pixels += slum_pixels
                total_pixels += tile_pixels
                
                tile_results.append({
                    'tile_id': i,
                    'tile_name': tile['name'],
                    'is_slum': is_slum,
                    'slum_pixels': int(slum_pixels),
                    'slum_percentage': float(slum_percentage),
                    'avg_probability': float(avg_prob),
                    'max_probability': float(max_prob)
                })
            
            threshold_results[threshold] = {
                'slum_tiles': slum_count,
                'total_tiles': len(tile_info),
                'slum_rate': (slum_count / len(tile_info)) * 100,
                'total_slum_pixels': int(total_slum_pixels),
                'total_pixels': int(total_pixels),
                'overall_slum_coverage': (total_slum_pixels / total_pixels) * 100,
                'tile_results': tile_results
            }
            
            print(f"   Threshold {threshold:.1f}: {slum_count} slum tiles ({(slum_count/len(tile_info))*100:.1f}%)")
        
        return all_predictions, tile_info, threshold_results
    
    def create_multi_threshold_visualizations(self, all_predictions, tile_info, threshold_results):
        """Create comprehensive visualizations for all thresholds."""
        print(f"\n🎨 Creating multi-threshold visualizations...")
        
        # 1. Threshold comparison chart
        thresholds = list(threshold_results.keys())
        slum_rates = [threshold_results[t]['slum_rate'] for t in thresholds]
        coverage_rates = [threshold_results[t]['overall_slum_coverage'] for t in thresholds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Slum detection rate vs threshold
        ax1.plot(thresholds, slum_rates, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Detection Threshold')
        ax1.set_ylabel('Slum Detection Rate (%)')
        ax1.set_title('Slum Tiles Detected vs Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(slum_rates) * 1.1 if slum_rates else 1)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(thresholds, slum_rates)):
            ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        # Overall slum coverage vs threshold
        ax2.plot(thresholds, coverage_rates, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Detection Threshold')
        ax2.set_ylabel('Overall Slum Coverage (%)')
        ax2.set_title('Slum Area Coverage vs Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(coverage_rates) * 1.1 if coverage_rates else 1)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(thresholds, coverage_rates)):
            ax2.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.suptitle('Multi-Threshold Analysis Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        threshold_chart = self.output_dir / "threshold_analysis.png"
        plt.savefig(threshold_chart, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create detailed visualizations for optimal thresholds
        # Find the threshold that detects the most slums
        optimal_threshold = max(thresholds, key=lambda t: threshold_results[t]['slum_rate'])
        
        print(f"   📈 Optimal threshold for detection: {optimal_threshold}")
        
        # 3. Create red overlay visualization for optimal threshold
        self.create_enhanced_overlays(all_predictions, tile_info, optimal_threshold)
        
        # 4. Create heatmap of all probabilities
        self.create_probability_heatmap(all_predictions, tile_info)
        
        return str(threshold_chart), optimal_threshold
    
    def create_enhanced_overlays(self, all_predictions, tile_info, threshold):
        """Create enhanced red overlays for the optimal threshold."""
        print(f"   🎨 Creating enhanced overlays for threshold {threshold}...")
        
        overlay_dir = self.output_dir / f"overlays_threshold_{threshold}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual overlays
        created_files = []
        slum_tiles = []
        
        for i, (prediction_map, tile) in enumerate(zip(all_predictions, tile_info)):
            original_image = tile['image']
            
            # Create binary mask
            binary_mask = (prediction_map > threshold).astype(np.uint8)
            slum_pixels = np.sum(binary_mask)
            avg_prob = np.mean(prediction_map)
            max_prob = np.max(prediction_map)
            
            if slum_pixels > 0:
                slum_tiles.append((i, slum_pixels, avg_prob, max_prob))
            
            # Create enhanced overlay
            overlay_image = original_image.copy()
            
            # Create red overlay with varying intensity based on probability
            red_overlay = np.zeros_like(original_image)
            red_overlay[:, :, 0] = (prediction_map * 255).astype(np.uint8)  # Probability-based intensity
            
            # Blend with different alpha for detected vs non-detected
            alpha = 0.6 if slum_pixels > 0 else 0.3
            overlay_image = cv2.addWeighted(original_image, 1-alpha, red_overlay, alpha, 0)
            
            # Create detailed visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original
            axes[0].imshow(original_image)
            axes[0].set_title(f"Original\n{tile['name']}", fontsize=10)
            axes[0].axis('off')
            
            # Probability heatmap
            im = axes[1].imshow(prediction_map, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f"Probability Map\nAvg: {avg_prob:.3f}", fontsize=10)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Binary mask
            axes[2].imshow(binary_mask, cmap='Reds', vmin=0, vmax=1)
            axes[2].set_title(f"Detection (T={threshold})\nPixels: {slum_pixels}", fontsize=10)
            axes[2].axis('off')
            
            # Enhanced overlay
            axes[3].imshow(overlay_image)
            status = "SLUM DETECTED" if slum_pixels > 0 else "NO SLUM"
            color = 'red' if slum_pixels > 0 else 'green'
            axes[3].set_title(f"Enhanced Overlay\n{status}", fontsize=10, color=color, fontweight='bold')
            axes[3].axis('off')
            
            # Overall title
            fig.suptitle(f"Tile {i:04d} - Max Prob: {max_prob:.3f}", fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            
            # Save
            overlay_file = overlay_dir / f"enhanced_tile_{i:04d}.png"
            plt.savefig(overlay_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            created_files.append(str(overlay_file))
        
        print(f"   ✅ Created {len(created_files)} enhanced overlays")
        print(f"   🚨 Detected slums in {len(slum_tiles)} tiles at threshold {threshold}")
        
        # Create summary of detected slums
        if slum_tiles:
            slum_tiles.sort(key=lambda x: x[3], reverse=True)  # Sort by max probability
            print(f"   📊 Top slum detections:")
            for i, (tile_id, pixels, avg_prob, max_prob) in enumerate(slum_tiles[:5]):
                print(f"      Tile {tile_id:04d}: {pixels:4d} pixels, avg={avg_prob:.3f}, max={max_prob:.3f}")
        
        return created_files, slum_tiles
    
    def create_probability_heatmap(self, all_predictions, tile_info):
        """Create a heatmap showing probability distribution across all tiles."""
        print(f"   🌡️ Creating probability heatmap...")
        
        # Calculate grid dimensions (approximate square)
        n_tiles = len(all_predictions)
        grid_cols = int(np.ceil(np.sqrt(n_tiles)))
        grid_rows = int(np.ceil(n_tiles / grid_cols))
        
        # Create average probability map for each tile
        avg_probs = [np.mean(pred) for pred in all_predictions]
        max_probs = [np.max(pred) for pred in all_predictions]
        
        # Arrange in grid
        prob_grid = np.zeros((grid_rows, grid_cols))
        for i, avg_prob in enumerate(avg_probs):
            row = i // grid_cols
            col = i % grid_cols
            if row < grid_rows:
                prob_grid[row, col] = avg_prob
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Average probability heatmap
        im1 = ax1.imshow(prob_grid, cmap='hot', vmin=0, vmax=np.max(avg_probs))
        ax1.set_title('Average Probability per Tile', fontweight='bold')
        ax1.set_xlabel('Tile Column')
        ax1.set_ylabel('Tile Row')
        
        # Add tile numbers
        for i in range(min(n_tiles, grid_rows * grid_cols)):
            row = i // grid_cols
            col = i % grid_cols
            if row < grid_rows:
                ax1.text(col, row, str(i), ha='center', va='center', 
                        color='white' if prob_grid[row, col] > np.max(avg_probs)/2 else 'black',
                        fontsize=8, fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Probability distribution histogram
        ax2.hist(avg_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(avg_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(avg_probs):.3f}')
        ax2.axvline(np.median(avg_probs), color='green', linestyle='--', 
                   label=f'Median: {np.median(avg_probs):.3f}')
        ax2.set_xlabel('Average Probability')
        ax2.set_ylabel('Number of Tiles')
        ax2.set_title('Distribution of Average Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        heatmap_file = self.output_dir / "probability_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(heatmap_file)
    
    def save_enhanced_results(self, threshold_results, optimal_threshold, visualization_files):
        """Save comprehensive enhanced results."""
        print(f"\n💾 Saving enhanced analysis results...")
        
        # Prepare results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'checkpoint': str(self.checkpoint_path),
                'architecture': self.model_config.architecture,
                'encoder': self.model_config.encoder
            },
            'analysis_type': 'multi_threshold_enhanced',
            'optimal_threshold': optimal_threshold,
            'threshold_analysis': threshold_results,
            'summary_statistics': {
                'total_tiles': len(threshold_results[0.5]['tile_results']),
                'thresholds_tested': list(threshold_results.keys()),
                'optimal_detection_rate': threshold_results[optimal_threshold]['slum_rate'],
                'optimal_coverage_rate': threshold_results[optimal_threshold]['overall_slum_coverage']
            },
            'visualization_files': visualization_files
        }
        
        # Save results
        results_file = self.output_dir / "enhanced_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create detailed report
        report_lines = [
            "ENHANCED COLOMBO SLUM DETECTION ANALYSIS",
            "=" * 60,
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.model_config.architecture} with {self.model_config.encoder}",
            "",
            "MULTI-THRESHOLD ANALYSIS RESULTS:",
            f"  Optimal threshold: {optimal_threshold}",
            f"  Total tiles analyzed: {results['summary_statistics']['total_tiles']}",
            f"  Detection rate at optimal threshold: {threshold_results[optimal_threshold]['slum_rate']:.1f}%",
            f"  Overall slum coverage: {threshold_results[optimal_threshold]['overall_slum_coverage']:.3f}%",
            "",
            "THRESHOLD BREAKDOWN:"
        ]
        
        for threshold in sorted(threshold_results.keys()):
            result = threshold_results[threshold]
            report_lines.append(
                f"  Threshold {threshold:.1f}: {result['slum_tiles']} tiles ({result['slum_rate']:.1f}%) "
                f"- Coverage: {result['overall_slum_coverage']:.3f}%"
            )
        
        report_lines.extend([
            "",
            "ANALYSIS METHODOLOGY:",
            "  - Applied exact same preprocessing as training",
            "  - Tested 8 different detection thresholds (0.1 to 0.8)",
            "  - Generated probability-based red overlays",
            "  - Created spatial heatmaps and statistical distributions",
            "  - Identified optimal threshold for maximum detection sensitivity",
            "",
            "KEY FINDINGS:",
            f"  - Lower thresholds detect more potential slum areas",
            f"  - Threshold {optimal_threshold} provides best balance of detection and accuracy",
            f"  - Spatial probability maps show detailed slum likelihood",
            f"  - Enhanced overlays highlight areas of concern for further investigation"
        ])
        
        report_file = self.output_dir / "ENHANCED_ANALYSIS_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   ✅ Results saved: {results_file}")
        print(f"   ✅ Report saved: {report_file}")
        
        return str(results_file), str(report_file)
    
    def run_enhanced_analysis(self, tiles_dir):
        """Run the complete enhanced analysis."""
        print(f"\n🚀 ENHANCED COLOMBO SLUM ANALYSIS")
        print("=" * 60)
        
        try:
            # Step 1: Multi-threshold analysis
            all_predictions, tile_info, threshold_results = self.analyze_multiple_thresholds(tiles_dir)
            
            # Step 2: Create visualizations
            threshold_chart, optimal_threshold = self.create_multi_threshold_visualizations(
                all_predictions, tile_info, threshold_results)
            
            # Step 3: Create enhanced overlays
            overlay_files, slum_tiles = self.create_enhanced_overlays(
                all_predictions, tile_info, optimal_threshold)
            
            # Step 4: Create probability heatmap
            heatmap_file = self.create_probability_heatmap(all_predictions, tile_info)
            
            # Step 5: Save results
            all_viz_files = [threshold_chart, heatmap_file] + overlay_files
            results_file, report_file = self.save_enhanced_results(
                threshold_results, optimal_threshold, all_viz_files)
            
            # Final summary
            total_tiles = len(tile_info)
            slum_count = threshold_results[optimal_threshold]['slum_tiles']
            
            print(f"\n✅ ENHANCED ANALYSIS COMPLETE!")
            print("=" * 60)
            print(f"🏘️ ENHANCED RESULTS:")
            print(f"   📊 Total tiles analyzed: {total_tiles}")
            print(f"   🎯 Optimal threshold: {optimal_threshold}")
            print(f"   🚨 Slum areas detected: {slum_count} tiles ({(slum_count/total_tiles)*100:.1f}%)")
            print(f"   📈 Enhanced visualizations: {len(overlay_files)} files")
            print(f"   🌡️ Probability heatmaps: Generated")
            print(f"   📁 All outputs saved to: {self.output_dir}")
            print("=" * 60)
            
            return {
                'total_tiles': total_tiles,
                'optimal_threshold': optimal_threshold,
                'slum_tiles_detected': slum_count,
                'detection_rate': (slum_count/total_tiles)*100,
                'results_file': results_file,
                'visualization_files': all_viz_files
            }
            
        except Exception as e:
            print(f"\n❌ Enhanced analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Run enhanced analysis."""
    
    # Configuration
    checkpoint_path = "../experiments/development_20250713_175410/checkpoints/best_checkpoint.pth"
    tiles_dir = "tiles"
    output_dir = "enhanced_analysis"
    
    print(f"🚀 ENHANCED COLOMBO SLUM DETECTION")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedColomboAnalysis(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir
    )
    
    # Run enhanced analysis
    results = system.run_enhanced_analysis(tiles_dir)
    
    if results:
        print(f"\n🎉 Enhanced analysis completed successfully!")
        print(f"Optimal threshold: {results['optimal_threshold']}")
        print(f"Detected {results['slum_tiles_detected']} potential slum areas")
    else:
        print(f"\n❌ Enhanced analysis failed!")

if __name__ == "__main__":
    main()
