"""
Comprehensive solution for Colombo slum detection with domain adaptation.
This script addresses the domain shift between training and Colombo data.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.transforms import get_test_transforms
from utils.checkpoint import load_checkpoint

class DomainAdaptation:
    """Handles domain adaptation for cross-region slum detection."""
    
    def __init__(self, model_path):
        """Initialize with trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model architecture
        from models.unet import SlumUNet
        from config.data_config import DataConfig
        
        self.model = SlumUNet(encoder_name='resnet34', classes=1)
        
        # Load checkpoint
        checkpoint = load_checkpoint(model_path, self.model, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize config and transforms
        self.config = DataConfig()
        self.transforms = get_test_transforms(self.config)
        
        # Load reference training image for histogram matching
        self.reference_image = self._load_reference_image()
        
    def _load_reference_image(self):
        """Load a representative training image for histogram matching."""
        train_dir = Path("data/train/images")
        train_files = list(train_dir.glob("*.tif"))
        
        if train_files:
            ref_img = Image.open(train_files[0]).convert('RGB')
            return np.array(ref_img)
        return None
    
    def histogram_match(self, source_img, reference_img):
        """Apply histogram matching to reduce domain shift."""
        try:
            from skimage import exposure
            
            # Convert to numpy if PIL Image
            if hasattr(source_img, 'convert'):
                source_np = np.array(source_img)
            else:
                source_np = source_img
                
            # Apply histogram matching
            matched = exposure.match_histograms(source_np, reference_img, channel_axis=2)
            return matched.astype(np.uint8)
            
        except ImportError:
            print("⚠️ scikit-image not available, skipping histogram matching")
            return np.array(source_img) if hasattr(source_img, 'convert') else source_img
    
    def normalize_brightness(self, img_array, target_mean=98.3, target_std=12.3):
        """Normalize image brightness to match training data statistics."""
        current_mean = img_array.mean()
        current_std = img_array.std()
        
        # Standardize then rescale
        normalized = (img_array - current_mean) / current_std
        adjusted = (normalized * target_std) + target_mean
        
        # Clip to valid range
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(np.uint8)
    
    def enhance_contrast(self, img_array):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def predict_with_adaptations(self, img_path, use_histogram_matching=True, 
                                use_brightness_norm=True, use_contrast_enhancement=False):
        """
        Predict slums with various domain adaptation techniques.
        
        Returns:
            dict with predictions from different adaptation methods
        """
        # Load original image
        original_img = Image.open(img_path).convert('RGB')
        original_np = np.array(original_img)
        
        results = {}
        
        # 1. Original prediction (no adaptation)
        results['original'] = self._predict_single(original_np)
        
        # 2. Histogram matching
        if use_histogram_matching and self.reference_image is not None:
            matched_np = self.histogram_match(original_np, self.reference_image)
            results['histogram_matched'] = self._predict_single(matched_np)
        
        # 3. Brightness normalization
        if use_brightness_norm:
            norm_np = self.normalize_brightness(original_np)
            results['brightness_normalized'] = self._predict_single(norm_np)
        
        # 4. Contrast enhancement
        if use_contrast_enhancement:
            enhanced_np = self.enhance_contrast(original_np)
            results['contrast_enhanced'] = self._predict_single(enhanced_np)
        
        # 5. Combined approach
        if use_histogram_matching and use_brightness_norm and self.reference_image is not None:
            combined_np = self.histogram_match(original_np, self.reference_image)
            combined_np = self.normalize_brightness(combined_np)
            results['combined'] = self._predict_single(combined_np)
        
        return results, original_np
    
    def _predict_single(self, img_array):
        """Predict on a single image array."""
        # Apply transforms
        transformed = self.transforms(image=img_array)['image']
        img_tensor = transformed.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.model(img_tensor)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
        
        return pred_prob

def analyze_all_colombo_tiles():
    """Analyze all Colombo tiles with domain adaptation."""
    
    print("🚀 COMPREHENSIVE COLOMBO ANALYSIS WITH DOMAIN ADAPTATION")
    print("=" * 70)
    
    # Find the best model
    model_path = "experiments/development_20250713_175410/checkpoints/best_checkpoint.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Initialize domain adaptation
    try:
        adapter = DomainAdaptation(model_path)
        print("✅ Domain adaptation system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Get all Colombo tiles
    tiles_dir = Path("colombo/tiles")
    if not tiles_dir.exists():
        print("❌ Colombo tiles directory not found")
        return
    
    tile_files = list(tiles_dir.glob("*.png"))
    print(f"📂 Found {len(tile_files)} Colombo tiles")
    
    if len(tile_files) == 0:
        print("❌ No Colombo tiles found")
        return
    
    # Analyze subset of tiles
    sample_tiles = tile_files[:10]  # Analyze first 10 tiles
    
    all_results = []
    best_method = {'method': 'original', 'avg_max': 0}
    
    print("\n🔍 Testing different domain adaptation methods...")
    
    for i, tile_file in enumerate(sample_tiles):
        print(f"\nTile {i+1}/{len(sample_tiles)}: {tile_file.name}")
        
        try:
            results, original_img = adapter.predict_with_adaptations(
                tile_file, 
                use_histogram_matching=True,
                use_brightness_norm=True,
                use_contrast_enhancement=True
            )
            
            all_results.append({
                'file': tile_file.name,
                'results': results,
                'original_img': original_img
            })
            
            # Print results for this tile
            for method, pred in results.items():
                max_prob = pred.max()
                slum_pixels = (pred > 0.5).sum()
                print(f"  {method:20}: max={max_prob:.4f}, slum_pixels={slum_pixels}")
                
                # Track best method
                if max_prob > best_method['avg_max']:
                    best_method = {'method': method, 'avg_max': max_prob}
                    
        except Exception as e:
            print(f"  ❌ Error processing {tile_file.name}: {e}")
    
    # Calculate average performance for each method
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("=" * 50)
    
    if all_results:
        method_stats = {}
        
        # Get all methods
        all_methods = set()
        for result in all_results:
            all_methods.update(result['results'].keys())
        
        for method in all_methods:
            max_probs = []
            slum_counts = []
            
            for result in all_results:
                if method in result['results']:
                    pred = result['results'][method]
                    max_probs.append(pred.max())
                    slum_counts.append((pred > 0.5).sum())
            
            if max_probs:
                avg_max = np.mean(max_probs)
                avg_slum_pixels = np.mean(slum_counts)
                method_stats[method] = {
                    'avg_max': avg_max,
                    'avg_slum_pixels': avg_slum_pixels,
                    'total_tiles': len(max_probs)
                }
                
                print(f"{method:20}: avg_max={avg_max:.4f}, avg_slum_pixels={avg_slum_pixels:.1f}")
        
        # Find best performing method
        if method_stats:
            best_method_name = max(method_stats.keys(), key=lambda k: method_stats[k]['avg_max'])
            best_stats = method_stats[best_method_name]
            
            print(f"\n🏆 BEST METHOD: {best_method_name}")
            print(f"   Average max probability: {best_stats['avg_max']:.4f}")
            print(f"   Average slum pixels detected: {best_stats['avg_slum_pixels']:.1f}")
            
            # Create visualizations
            create_method_comparison_plots(all_results, method_stats)
            
            # Apply best method to all tiles
            if best_stats['avg_max'] > 0.1:  # If we found a decent method
                apply_best_method_to_all_tiles(adapter, tile_files, best_method_name)
            else:
                print("\n⚠️ No method achieved satisfactory performance")
                print("   Consider retraining the model with more diverse data")
    
    return all_results

def create_method_comparison_plots(all_results, method_stats):
    """Create visualization comparing different adaptation methods."""
    
    if not all_results:
        return
    
    # Create comparison plot
    methods = list(method_stats.keys())
    avg_maxes = [method_stats[m]['avg_max'] for m in methods]
    
    plt.figure(figsize=(12, 8))
    
    # Bar plot of average max probabilities
    plt.subplot(2, 2, 1)
    bars = plt.bar(methods, avg_maxes)
    plt.title('Average Max Probability by Method')
    plt.ylabel('Average Max Probability')
    plt.xticks(rotation=45, ha='right')
    
    # Highlight best method
    best_idx = np.argmax(avg_maxes)
    bars[best_idx].set_color('red')
    
    # Plot individual tile results
    plt.subplot(2, 2, 2)
    for i, result in enumerate(all_results[:5]):  # First 5 tiles
        max_probs = [result['results'][m].max() for m in methods if m in result['results']]
        plt.plot(methods[:len(max_probs)], max_probs, marker='o', label=f"Tile {i+1}")
    
    plt.title('Max Probability by Tile')
    plt.ylabel('Max Probability')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Show sample predictions from best method
    if len(all_results) >= 2:
        best_method = methods[best_idx]
        
        plt.subplot(2, 2, 3)
        sample_pred = all_results[0]['results'][best_method]
        plt.imshow(sample_pred, cmap='Reds', vmin=0, vmax=1)
        plt.title(f'Sample Prediction: {best_method}')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        sample_img = all_results[0]['original_img']
        plt.imshow(sample_img)
        plt.title('Original Tile')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('colombo/domain_adaptation_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Saved: colombo/domain_adaptation_comparison.png")

def apply_best_method_to_all_tiles(adapter, tile_files, best_method):
    """Apply the best domain adaptation method to all tiles and save results."""
    
    print(f"\n🎯 APPLYING BEST METHOD ({best_method}) TO ALL TILES")
    print("=" * 60)
    
    output_dir = Path("colombo/final_predictions")
    output_dir.mkdir(exist_ok=True)
    
    all_predictions = []
    high_confidence_tiles = []
    
    for i, tile_file in enumerate(tile_files):
        print(f"Processing {i+1}/{len(tile_files)}: {tile_file.name}")
        
        try:
            results, original_img = adapter.predict_with_adaptations(tile_file)
            
            if best_method in results:
                pred = results[best_method]
                max_prob = pred.max()
                slum_pixels = (pred > 0.5).sum()
                
                all_predictions.append({
                    'file': tile_file.name,
                    'max_prob': max_prob,
                    'slum_pixels': slum_pixels,
                    'prediction': pred,
                    'original': original_img
                })
                
                if max_prob > 0.3:  # High confidence threshold
                    high_confidence_tiles.append({
                        'file': tile_file.name,
                        'max_prob': max_prob,
                        'slum_pixels': slum_pixels,
                        'prediction': pred,
                        'original': original_img
                    })
                
                # Save individual prediction
                save_tile_prediction(tile_file.name, original_img, pred, output_dir)
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Create summary
    create_final_summary(all_predictions, high_confidence_tiles, output_dir)
    
    print(f"\n✅ PROCESSING COMPLETE")
    print(f"   Total tiles processed: {len(all_predictions)}")
    print(f"   High-confidence detections: {len(high_confidence_tiles)}")
    print(f"   Results saved to: {output_dir}")

def save_tile_prediction(filename, original_img, prediction, output_dir):
    """Save individual tile prediction with overlay."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Tile')
    axes[0].axis('off')
    
    # Prediction heatmap
    axes[1].imshow(prediction, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title(f'Prediction\nMax: {prediction.max():.3f}')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_img)
    axes[2].imshow(prediction, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save with descriptive filename
    max_prob = prediction.max()
    slum_pixels = (prediction > 0.5).sum()
    safe_filename = filename.replace('.png', f'_max{max_prob:.3f}_slums{slum_pixels}.png')
    
    plt.savefig(output_dir / safe_filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_final_summary(all_predictions, high_confidence_tiles, output_dir):
    """Create final summary report and visualizations."""
    
    if not all_predictions:
        return
    
    # Summary statistics
    max_probs = [p['max_prob'] for p in all_predictions]
    slum_counts = [p['slum_pixels'] for p in all_predictions]
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Max probability - Mean: {np.mean(max_probs):.4f}, Max: {np.max(max_probs):.4f}")
    print(f"   Slum pixels - Mean: {np.mean(slum_counts):.1f}, Max: {np.max(slum_counts)}")
    print(f"   Tiles with slums detected (>0.5): {sum(1 for c in slum_counts if c > 0)}")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Distribution of max probabilities
    axes[0, 0].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Max Probabilities')
    axes[0, 0].set_xlabel('Max Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
    axes[0, 0].legend()
    
    # Distribution of slum pixel counts
    axes[0, 1].hist(slum_counts, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Slum Pixel Counts')
    axes[0, 1].set_xlabel('Slum Pixels')
    axes[0, 1].set_ylabel('Count')
    
    # Scatter plot
    axes[0, 2].scatter(max_probs, slum_counts, alpha=0.6)
    axes[0, 2].set_title('Max Probability vs Slum Pixels')
    axes[0, 2].set_xlabel('Max Probability')
    axes[0, 2].set_ylabel('Slum Pixels')
    
    # Show top predictions
    if high_confidence_tiles:
        # Sort by max probability
        sorted_tiles = sorted(high_confidence_tiles, key=lambda x: x['max_prob'], reverse=True)
        
        for i, tile_data in enumerate(sorted_tiles[:3]):
            row = 1
            col = i
            
            axes[row, col].imshow(tile_data['original'])
            axes[row, col].imshow(tile_data['prediction'], cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            axes[row, col].set_title(f"Top {i+1}: {tile_data['file']}\nMax: {tile_data['max_prob']:.3f}")
            axes[row, col].axis('off')
    else:
        for i in range(3):
            axes[1, i].text(0.5, 0.5, 'No high-confidence\ndetections found', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save summary report
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write("COLOMBO SLUM DETECTION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total tiles analyzed: {len(all_predictions)}\n")
        f.write(f"High-confidence detections: {len(high_confidence_tiles)}\n")
        f.write(f"Average max probability: {np.mean(max_probs):.4f}\n")
        f.write(f"Maximum probability found: {np.max(max_probs):.4f}\n")
        f.write(f"Average slum pixels: {np.mean(slum_counts):.1f}\n")
        f.write(f"Tiles with slums detected (>0.5): {sum(1 for c in slum_counts if c > 0)}\n\n")
        
        f.write("HIGH-CONFIDENCE DETECTIONS:\n")
        f.write("-" * 30 + "\n")
        for tile in sorted(high_confidence_tiles, key=lambda x: x['max_prob'], reverse=True):
            f.write(f"{tile['file']:30} | Max: {tile['max_prob']:.4f} | Slum pixels: {tile['slum_pixels']}\n")
    
    print(f"Saved: {output_dir}/final_summary.png")
    print(f"Saved: {output_dir}/summary_report.txt")

if __name__ == "__main__":
    try:
        # Install required package if missing
        try:
            from skimage import exposure
        except ImportError:
            print("📦 Installing scikit-image for histogram matching...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
        
        analyze_all_colombo_tiles()
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
