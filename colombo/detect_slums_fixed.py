#!/usr/bin/env python3
"""
Colombo Slum Detection Inference Script (Fixed)
===============================================

Performs slum detection inference on processed Colombo satellite image tiles
using the EXACT same approach as the successful chart analysis.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules - EXACT same as successful analysis
from models import create_model
from config import get_model_config, get_data_config
from utils.checkpoint import load_checkpoint
from utils.transforms import get_test_transforms
from utils.dataset import SlumDataset

class ColomboSlumDetector:
    """Slum detection inference for Colombo using the proven approach."""
    
    def __init__(self, checkpoint_path, output_dir="predictions"):
        """Initialize detector using the exact same approach as charts analysis."""
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use exact same device setup as successful analysis
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device}")
        
        # Load exact same configurations as successful analysis
        self.model_config = get_model_config('balanced')
        self.data_config = get_data_config('standard')
        
        # Load model using exact same approach
        self.model = self._load_model()
        
        # Get exact same transforms as successful analysis
        self.test_transforms = get_test_transforms(self.data_config)
        
        print(f"🤖 Colombo Slum Detector initialized")
        print(f"   Model loaded from: {checkpoint_path}")
        print(f"   Output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load model using EXACT same approach as successful analysis."""
        print(f"📂 Loading model from: {self.checkpoint_path}")
        
        # Create model - exact same as charts/quick_analysis.py
        model = create_model(
            architecture=self.model_config.architecture,
            encoder=self.model_config.encoder,
            pretrained=False,
            num_classes=self.model_config.num_classes
        )
        
        # Load checkpoint - exact same approach
        checkpoint = load_checkpoint(self.checkpoint_path, model, device=self.device)
        model = model.to(self.device)
        model.eval()
        
        print(f"   ✅ Model loaded successfully")
        return model
    
    def predict_colombo_tiles(self, tiles_dir, batch_size=16):
        """Predict using EXACT same approach as successful chart analysis."""
        print(f"🔍 COLOMBO SLUM DETECTION")
        print("=" * 50)
        
        # Get all tile files
        tiles_dir = Path(tiles_dir)
        tile_files = sorted(list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg")))
        
        if not tile_files:
            print("❌ No tile files found!")
            return [], []
        
        print(f"📂 Found {len(tile_files)} tile files")
        
        # Process tiles one by one using exact same preprocessing
        all_predictions = []
        all_images = []
        
        print("🔄 Processing tiles...")
        
        for i, tile_file in enumerate(tile_files):
            try:
                # Load and preprocess exactly like successful analysis
                image = Image.open(tile_file).convert('RGB')
                image_np = np.array(image)
                
                # Apply exact same transforms as successful analysis
                if self.test_transforms:
                    transformed = self.test_transforms(image=image_np)
                    image_tensor = transformed['image']
                else:
                    # Fallback - same as successful analysis
                    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
                
                # Add batch dimension
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Inference - EXACT same as successful analysis
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    predictions = torch.sigmoid(outputs)
                    
                    # Store results
                    all_predictions.append(predictions.cpu().numpy())
                    all_images.append(image_np)
                
                # Progress update
                if (i + 1) % 10 == 0 or i == len(tile_files) - 1:
                    print(f"   Progress: {i + 1}/{len(tile_files)} tiles processed")
                    
            except Exception as e:
                print(f"   ❌ Error processing {tile_file}: {e}")
        
        if not all_predictions:
            print("❌ No successful predictions!")
            return [], []
        
        # Concatenate results - exact same as successful analysis
        predictions = np.concatenate(all_predictions, axis=0)
        
        # Remove channel dimension if present - exact same as successful analysis
        if len(predictions.shape) == 4:
            predictions = predictions.squeeze(1)
        
        print(f"✅ Predictions shape: {predictions.shape}")
        print(f"✅ Completed inference on {len(all_predictions)} tiles")
        
        return predictions, all_images, tile_files
    
    def analyze_predictions(self, predictions, threshold=0.5):
        """Analyze predictions using same approach as successful analysis."""
        print("📊 Analyzing predictions...")
        
        if len(predictions) == 0:
            print("❌ No predictions to analyze!")
            return {}
        
        # Calculate statistics - exact same approach
        binary_predictions = (predictions > threshold).astype(int)
        
        # Per-tile statistics (average probability per tile)
        if len(predictions.shape) == 3:  # [tiles, height, width]
            tile_probabilities = predictions.mean(axis=(1, 2))
            tile_slum_percentages = binary_predictions.mean(axis=(1, 2))
        else:  # [tiles, features] or [tiles]
            tile_probabilities = predictions.flatten() if len(predictions.shape) > 1 else predictions
            tile_slum_percentages = binary_predictions.flatten() if len(binary_predictions.shape) > 1 else binary_predictions
        
        # Overall statistics
        total_tiles = len(tile_probabilities)
        slum_tiles = np.sum(tile_slum_percentages > 0.1)  # Tiles with >10% slum area
        high_conf_slum_tiles = np.sum(tile_probabilities > 0.7)  # High confidence slum tiles
        
        avg_probability = np.mean(tile_probabilities)
        max_probability = np.max(tile_probabilities)
        min_probability = np.min(tile_probabilities)
        
        analysis = {
            'total_tiles': int(total_tiles),
            'slum_tiles': int(slum_tiles),
            'high_confidence_slum_tiles': int(high_conf_slum_tiles),
            'non_slum_tiles': int(total_tiles - slum_tiles),
            'slum_percentage': float((slum_tiles / total_tiles) * 100 if total_tiles > 0 else 0),
            'avg_probability': float(avg_probability),
            'max_probability': float(max_probability),
            'min_probability': float(min_probability),
            'threshold_used': threshold,
            'tile_probabilities': tile_probabilities.tolist(),
            'predictions_shape': list(predictions.shape)
        }
        
        # Print results
        print(f"   📊 Analysis Results:")
        print(f"      Total tiles: {total_tiles}")
        print(f"      Slum tiles (>10% slum area): {slum_tiles} ({analysis['slum_percentage']:.1f}%)")
        print(f"      High confidence slum tiles (>70%): {high_conf_slum_tiles}")
        print(f"      Average probability: {avg_probability:.3f}")
        print(f"      Probability range: [{min_probability:.3f}, {max_probability:.3f}]")
        
        return analysis
    
    def create_prediction_visualizations(self, predictions, images, tile_files, analysis):
        """Create visualizations using same approach as successful analysis."""
        print("🎨 Creating prediction visualizations...")
        
        if len(predictions) == 0:
            print("❌ No predictions to visualize!")
            return []
        
        created_files = []
        
        # 1. Sample predictions grid (like the successful analysis)
        try:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle('Colombo Slum Detection - Sample Predictions', fontsize=16, fontweight='bold')
            
            # Select diverse samples
            tile_probs = analysis.get('tile_probabilities', [])
            if len(tile_probs) >= 10:
                # Get high and low probability samples
                sorted_indices = np.argsort(tile_probs)
                sample_indices = list(sorted_indices[-5:]) + list(sorted_indices[:5])
            else:
                sample_indices = list(range(min(10, len(images))))
            
            for i, idx in enumerate(sample_indices):
                if i >= 10 or idx >= len(images):
                    break
                
                row = i // 5
                col = i % 5
                
                # Show original image
                axes[row, col].imshow(images[idx])
                
                # Get prediction for this tile
                if len(predictions.shape) == 3:  # Spatial predictions
                    tile_pred = predictions[idx]
                    avg_prob = tile_pred.mean()
                else:
                    avg_prob = tile_probs[idx] if idx < len(tile_probs) else 0
                
                # Title with probability
                color = 'red' if avg_prob > 0.5 else 'orange' if avg_prob > 0.2 else 'green'
                title = f"Tile {idx}\nProb: {avg_prob:.3f}"
                axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
                axes[row, col].axis('off')
            
            # Hide unused subplots
            for i in range(len(sample_indices), 10):
                row = i // 5
                col = i % 5
                axes[row, col].axis('off')
            
            plt.tight_layout()
            viz_path = self.output_dir / "prediction_samples.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(viz_path))
            
        except Exception as e:
            print(f"   ⚠️ Error creating sample predictions: {e}")
        
        # 2. Probability distribution (like successful analysis)
        try:
            tile_probs = analysis.get('tile_probabilities', [])
            if tile_probs:
                plt.figure(figsize=(12, 8))
                plt.hist(tile_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
                plt.axvline(x=np.mean(tile_probs), color='green', linestyle='-', linewidth=2, 
                           label=f'Mean ({np.mean(tile_probs):.3f})')
                plt.xlabel('Slum Probability')
                plt.ylabel('Number of Tiles')
                plt.title('Distribution of Slum Probabilities - Colombo Analysis')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                dist_path = self.output_dir / "probability_distribution.png"
                plt.savefig(dist_path, dpi=300, bbox_inches='tight')
                plt.close()
                created_files.append(str(dist_path))
                
        except Exception as e:
            print(f"   ⚠️ Error creating probability distribution: {e}")
        
        print(f"   ✅ Created {len(created_files)} visualizations")
        return created_files
    
    def save_results(self, predictions, analysis, tile_files, visualizations):
        """Save comprehensive results."""
        print("💾 Saving results...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': str(self.checkpoint_path),
            'analysis': analysis,
            'tile_files': [str(f) for f in tile_files],
            'predictions_shape': list(predictions.shape) if len(predictions) > 0 else [],
            'visualizations': visualizations,
            'model_config': self.model_config.__dict__ if hasattr(self.model_config, '__dict__') else str(self.model_config),
            'approach': 'Fixed - using exact same method as successful chart analysis'
        }
        
        # Save full results
        results_file = self.output_dir / "colombo_slum_detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Results saved: {results_file}")
        return str(results_file)
    
    def detect_slums_fixed(self, tiles_dir, threshold=0.5):
        """Complete slum detection using the PROVEN successful approach."""
        print(f"\n🏘️ COLOMBO SLUM DETECTION (FIXED)")
        print("=" * 50)
        print("Using EXACT same approach as successful chart analysis")
        
        # Run predictions using proven approach
        predictions, images, tile_files = self.predict_colombo_tiles(tiles_dir)
        
        if len(predictions) == 0:
            print("❌ No predictions generated!")
            return None
        
        # Analyze results
        analysis = self.analyze_predictions(predictions, threshold)
        
        # Create visualizations
        visualizations = self.create_prediction_visualizations(predictions, images, tile_files, analysis)
        
        # Save results
        results_file = self.save_results(predictions, analysis, tile_files, visualizations)
        
        print(f"\n✅ DETECTION COMPLETE!")
        print("=" * 50)
        print(f"🏘️ Colombo Slum Detection Summary:")
        print(f"   Total tiles analyzed: {analysis['total_tiles']}")
        print(f"   Slum tiles detected: {analysis['slum_tiles']} ({analysis['slum_percentage']:.1f}%)")
        print(f"   High confidence slums: {analysis['high_confidence_slum_tiles']} tiles")
        print(f"   Average probability: {analysis['avg_probability']:.3f}")
        print(f"   Max probability: {analysis['max_probability']:.3f}")
        print(f"   Results saved to: {self.output_dir}")
        
        return {
            'analysis': analysis,
            'predictions': predictions,
            'results_file': results_file,
            'visualizations': visualizations
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Fixed Colombo Slum Detection')
    parser.add_argument('--tiles-dir', required=True, help='Directory containing processed tiles')
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', default='predictions', help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    
    args = parser.parse_args()
    
    # Create detector
    detector = ColomboSlumDetector(
        checkpoint_path=args.checkpoint,
        output_dir=args.output
    )
    
    # Run detection
    try:
        results = detector.detect_slums_fixed(
            tiles_dir=args.tiles_dir,
            threshold=args.threshold
        )
        
        if results:
            print(f"\n🎉 Fixed Colombo slum detection completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
