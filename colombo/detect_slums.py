#!/usr/bin/env python3
"""
Colombo Slum Detection Inference Script
======================================

Performs slum detection inference on processed Colombo satellite image tiles.
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

# Import project modules
from models import create_model
from config import get_model_config
from utils.checkpoint import load_checkpoint
from utils.transforms import get_test_transforms
from config.data_config import get_data_config

class ColomboSlumDetector:
    """Slum detection inference for Colombo satellite image tiles."""
    
    def __init__(self, checkpoint_path, output_dir="colombo/predictions"):
        """
        Initialize the detector.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            output_dir: Directory to save predictions
        """
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Get transforms - use proper test transforms like the charts analysis
        data_config = get_data_config("standard")
        self.transform = get_test_transforms(data_config)
        
        print(f"🤖 Colombo Slum Detector initialized")
        print(f"   Model loaded from: {checkpoint_path}")
        print(f"   Output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load the trained model."""
        print(f"📂 Loading model from: {self.checkpoint_path}")
        
        # Get model configuration
        model_config = get_model_config("balanced")
        
        # Create model
        model = create_model(
            architecture=model_config.architecture,
            encoder=model_config.encoder,
            pretrained=False,
            num_classes=model_config.num_classes
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(self.checkpoint_path, model, device=self.device)
        model = model.to(self.device)
        model.eval()
        
        print(f"   ✅ Model loaded successfully")
        return model
    
    def load_tiles(self, tiles_dir, metadata_file=None):
        """Load processed tiles for inference."""
        tiles_dir = Path(tiles_dir)
        print(f"📂 Loading tiles from: {tiles_dir}")
        
        # Find all tile images
        tile_files = list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg"))
        tile_files.sort()
        
        print(f"   Found {len(tile_files)} tile files")
        
        # Load metadata if available
        metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"   ✅ Loaded metadata: {len(metadata.get('tiles', []))} tiles")
        
        return tile_files, metadata
    
    def predict_tile(self, tile_path):
        """Predict slum probability for a single tile."""
        # Load image file
        tile_image = Image.open(tile_path).convert('RGB')
        tile_image = np.array(tile_image)
        
        # Apply transforms or manual preprocessing
        if self.transform:
            transformed = self.transform(image=tile_image)
            tile_tensor = transformed['image']
        else:
            # Manual preprocessing
            tile_tensor = torch.from_numpy(tile_image.transpose(2, 0, 1)).float() / 255.0
        
        # Add batch dimension
        tile_tensor = tile_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(tile_tensor)
            probability = torch.sigmoid(output).cpu().numpy()
            
            # Handle different output shapes
            if len(probability.shape) == 4:  # Shape: [batch, channels, height, width]
                probability = probability[0, 0].mean()  # Average over spatial dimensions
            elif len(probability.shape) == 3:  # Shape: [batch, height, width]
                probability = probability[0].mean()  # Average over spatial dimensions
            elif len(probability.shape) == 2:  # Shape: [batch, features]
                probability = probability[0, 0]
            else:  # Shape: [batch]
                probability = probability[0]
            
            # Ensure it's a scalar
            if isinstance(probability, np.ndarray):
                probability = float(probability)
        
        return probability
    
    def predict_all_tiles(self, tile_files, threshold=0.5):
        """Predict slum probabilities for all tiles."""
        print(f"🔍 Running inference on {len(tile_files)} tiles...")
        
        predictions = []
        
        for i, tile_file in enumerate(tile_files):
            try:
                # Predict
                probability = self.predict_tile(tile_file)
                
                # Store result
                result = {
                    'tile_file': str(tile_file),
                    'tile_name': Path(tile_file).name,
                    'probability': float(probability),
                    'prediction': 'slum' if probability >= threshold else 'non-slum',
                    'confidence': 'high' if abs(probability - 0.5) > 0.3 else 'medium' if abs(probability - 0.5) > 0.1 else 'low'
                }
                predictions.append(result)
                
                # Progress update
                if (i + 1) % 50 == 0 or i == len(tile_files) - 1:
                    print(f"   Progress: {i + 1}/{len(tile_files)} tiles processed")
                
            except Exception as e:
                print(f"   ❌ Error processing {tile_file}: {e}")
        
        print(f"   ✅ Completed inference on {len(predictions)} tiles")
        return predictions
    
    def analyze_predictions(self, predictions, threshold=0.5):
        """Analyze prediction results."""
        print("📊 Analyzing predictions...")
        
        # Calculate statistics
        total_tiles = len(predictions)
        slum_tiles = sum(1 for p in predictions if p['prediction'] == 'slum')
        non_slum_tiles = total_tiles - slum_tiles
        
        probabilities = [p['probability'] for p in predictions]
        avg_probability = np.mean(probabilities)
        max_probability = np.max(probabilities)
        min_probability = np.min(probabilities)
        
        # Confidence analysis
        high_conf = sum(1 for p in predictions if p['confidence'] == 'high')
        medium_conf = sum(1 for p in predictions if p['confidence'] == 'medium')
        low_conf = sum(1 for p in predictions if p['confidence'] == 'low')
        
        analysis = {
            'total_tiles': total_tiles,
            'slum_tiles': slum_tiles,
            'non_slum_tiles': non_slum_tiles,
            'slum_percentage': (slum_tiles / total_tiles) * 100 if total_tiles > 0 else 0,
            'avg_probability': avg_probability,
            'max_probability': max_probability,
            'min_probability': min_probability,
            'confidence_distribution': {
                'high': high_conf,
                'medium': medium_conf,
                'low': low_conf
            },
            'threshold_used': threshold
        }
        
        # Print summary
        print(f"   📊 Analysis Results:")
        print(f"      Total tiles: {total_tiles}")
        print(f"      Slum tiles: {slum_tiles} ({analysis['slum_percentage']:.1f}%)")
        print(f"      Non-slum tiles: {non_slum_tiles}")
        print(f"      Average probability: {avg_probability:.3f}")
        print(f"      Probability range: [{min_probability:.3f}, {max_probability:.3f}]")
        print(f"      High confidence: {high_conf} tiles")
        print(f"      Medium confidence: {medium_conf} tiles")
        print(f"      Low confidence: {low_conf} tiles")
        
        return analysis
    
    def visualize_predictions(self, predictions, tile_files, num_samples=20):
        """Create visualization of predictions."""
        print("🎨 Creating prediction visualizations...")
        
        # Sort by probability for better visualization
        sorted_predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
        
        # Select samples to show
        high_prob_samples = sorted_predictions[:num_samples//2]
        low_prob_samples = sorted_predictions[-num_samples//2:]
        
        samples = high_prob_samples + low_prob_samples
        
        # Create visualization
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle('Colombo Slum Detection Predictions', fontsize=16, fontweight='bold')
        
        for i, sample in enumerate(samples[:20]):
            row = i // 5
            col = i % 5
            
            # Load and display tile
            tile_image = Image.open(sample['tile_file'])
            axes[row, col].imshow(tile_image)
            
            # Title with prediction info
            prob = sample['probability']
            pred = sample['prediction']
            conf = sample['confidence']
            
            color = 'red' if pred == 'slum' else 'green'
            title = f"{pred.upper()}\nP={prob:.3f} ({conf})"
            
            axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        viz_path = self.output_dir / "prediction_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create probability distribution plot
        probabilities = [p['probability'] for p in predictions]
        
        plt.figure(figsize=(12, 8))
        plt.hist(probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        plt.xlabel('Slum Probability')
        plt.ylabel('Number of Tiles')
        plt.title('Distribution of Slum Probabilities - Colombo Satellite Image')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        dist_path = self.output_dir / "probability_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualizations saved:")
        print(f"      Samples: {viz_path}")
        print(f"      Distribution: {dist_path}")
        
        return str(viz_path), str(dist_path)
    
    def save_results(self, predictions, analysis, tile_metadata=None):
        """Save all results to files."""
        print("💾 Saving results...")
        
        # Comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': str(self.checkpoint_path),
            'analysis': analysis,
            'predictions': predictions,
            'metadata': tile_metadata
        }
        
        # Save full results
        results_file = self.output_dir / "colombo_slum_detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = {
            'detection_summary': analysis,
            'high_probability_tiles': [p for p in predictions if p['probability'] >= 0.7],
            'low_probability_tiles': [p for p in predictions if p['probability'] <= 0.3]
        }
        
        summary_file = self.output_dir / "detection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Results saved:")
        print(f"      Full results: {results_file}")
        print(f"      Summary: {summary_file}")
        
        return str(results_file), str(summary_file)
    
    def detect_slums(self, tiles_dir, metadata_file=None, threshold=0.5):
        """Complete slum detection pipeline."""
        print(f"\n🏘️ COLOMBO SLUM DETECTION")
        print("=" * 50)
        
        # Load tiles
        tile_files, tile_metadata = self.load_tiles(tiles_dir, metadata_file)
        
        if not tile_files:
            print("❌ No tile files found!")
            return None
        
        # Run predictions
        predictions = self.predict_all_tiles(tile_files, threshold)
        
        # Analyze results
        analysis = self.analyze_predictions(predictions, threshold)
        
        # Create visualizations
        viz_path, dist_path = self.visualize_predictions(predictions, tile_files)
        
        # Save results
        results_file, summary_file = self.save_results(predictions, analysis, tile_metadata)
        
        print(f"\n✅ DETECTION COMPLETE!")
        print("=" * 50)
        print(f"🏘️ Slum Detection Summary:")
        print(f"   Total area analyzed: {analysis['total_tiles']} tiles")
        print(f"   Slum areas detected: {analysis['slum_tiles']} tiles ({analysis['slum_percentage']:.1f}%)")
        print(f"   Average slum probability: {analysis['avg_probability']:.3f}")
        print(f"   Results saved to: {self.output_dir}")
        
        return {
            'analysis': analysis,
            'predictions': predictions,
            'results_file': results_file,
            'summary_file': summary_file,
            'visualizations': [viz_path, dist_path]
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Detect slums in Colombo satellite image tiles')
    parser.add_argument('--tiles-dir', required=True, help='Directory containing processed tiles')
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--metadata', help='Path to tile metadata file')
    parser.add_argument('--output', default='colombo/predictions', help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for slum classification')
    
    args = parser.parse_args()
    
    # Create detector
    detector = ColomboSlumDetector(
        checkpoint_path=args.checkpoint,
        output_dir=args.output
    )
    
    # Run detection
    try:
        results = detector.detect_slums(
            tiles_dir=args.tiles_dir,
            metadata_file=args.metadata,
            threshold=args.threshold
        )
        
        if results:
            print(f"\n🎉 Slum detection completed successfully!")
            print(f"Found {results['analysis']['slum_tiles']} potential slum areas out of {results['analysis']['total_tiles']} tiles")
        
    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
