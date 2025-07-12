"""
Inference Script for Slum Detection
===================================

Single image inference and batch prediction script for deployed
slum detection models with visualization capabilities.
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from models import create_model
from config import get_model_config, get_data_config
from utils.transforms import get_test_transforms
from utils.checkpoint import load_checkpoint
from utils.visualization import visualize_overlay


class SlumDetectionInference:
    """
    Inference class for slum detection from satellite imagery.
    """
    
    def __init__(self, checkpoint_path, model_config_name='balanced', 
                 data_config_name='standard', device='auto'):
        """
        Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            model_config_name: Model configuration preset name
            data_config_name: Data configuration preset name  
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        
        # Load configurations
        self.model_config = get_model_config(model_config_name)
        self.data_config = get_data_config(data_config_name)
        
        # Load model
        self.model = self._load_model()
        
        # Setup transforms
        self.transform = get_test_transforms(self.data_config)
        
        print(f"🚀 Slum Detection Inference Ready!")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.model_config.architecture} + {self.model_config.encoder}")
    
    def _setup_device(self, device):
        """Setup computation device."""
        if device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        print(f"📂 Loading model from: {self.checkpoint_path}")
        
        # Create model
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
        
        return model
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess image for inference.
        
        Args:
            image_path_or_array: Path to image file or numpy array
        
        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(image_path_or_array, (str, Path)):
            # Load image from file
            image = Image.open(image_path_or_array).convert('RGB')
            image = np.array(image)
        else:
            # Use provided array
            image = image_path_or_array.copy()
        
        # Ensure correct data type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image
    
    def predict(self, image_input, threshold=0.5, return_probability=False):
        """
        Predict slum areas in satellite image.
        
        Args:
            image_input: Image path, PIL Image, or numpy array
            threshold: Binary threshold for slum detection
            return_probability: Whether to return probability map
        
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_input)
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output)
        
        # Convert to numpy
        prob_map = probability.squeeze().cpu().numpy()
        
        # Create binary mask
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        # Calculate statistics
        total_pixels = prob_map.size
        slum_pixels = np.sum(binary_mask)
        slum_percentage = (slum_pixels / total_pixels) * 100
        confidence = np.mean(prob_map[binary_mask == 1]) if slum_pixels > 0 else 0.0
        
        results = {
            'binary_mask': binary_mask,
            'slum_percentage': slum_percentage,
            'slum_pixels': slum_pixels,
            'total_pixels': total_pixels,
            'confidence': confidence,
            'threshold': threshold,
            'original_image': original_image
        }
        
        if return_probability:
            results['probability_map'] = prob_map
        
        return results
    
    def predict_batch(self, image_list, threshold=0.5, batch_size=8):
        """
        Predict slum areas for multiple images.
        
        Args:
            image_list: List of image paths or arrays
            threshold: Binary threshold for slum detection
            batch_size: Batch size for processing
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i+batch_size]
            batch_tensors = []
            batch_originals = []
            
            # Preprocess batch
            for img in batch:
                tensor, original = self.preprocess_image(img)
                batch_tensors.append(tensor.squeeze(0))
                batch_originals.append(original)
            
            # Stack tensors
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Run batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.sigmoid(outputs)
            
            # Process results
            for j, (prob, original) in enumerate(zip(probabilities, batch_originals)):
                prob_map = prob.squeeze().cpu().numpy()
                binary_mask = (prob_map > threshold).astype(np.uint8)
                
                total_pixels = prob_map.size
                slum_pixels = np.sum(binary_mask)
                slum_percentage = (slum_pixels / total_pixels) * 100
                confidence = np.mean(prob_map[binary_mask == 1]) if slum_pixels > 0 else 0.0
                
                results.append({
                    'binary_mask': binary_mask,
                    'probability_map': prob_map,
                    'slum_percentage': slum_percentage,
                    'slum_pixels': slum_pixels,
                    'total_pixels': total_pixels,
                    'confidence': confidence,
                    'threshold': threshold,
                    'original_image': original,
                    'image_index': i + j
                })
        
        return results
    
    def visualize_prediction(self, results, save_path=None, show_probability=True):
        """
        Visualize prediction results.
        
        Args:
            results: Prediction results from predict()
            save_path: Path to save visualization
            show_probability: Whether to show probability map
        """
        original = results['original_image']
        binary_mask = results['binary_mask']
        
        if show_probability and 'probability_map' in results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Probability map
            im1 = axes[1].imshow(results['probability_map'], cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title('Slum Probability')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Binary prediction overlay
            axes[2].imshow(original)
            axes[2].imshow(binary_mask, cmap='Reds', alpha=0.6)
            axes[2].set_title(f'Slum Detection ({results["slum_percentage"]:.1f}%)')
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Original image
            axes[0].imshow(original)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Binary prediction overlay
            axes[1].imshow(original)
            axes[1].imshow(binary_mask, cmap='Reds', alpha=0.6)
            axes[1].set_title(f'Slum Detection ({results["slum_percentage"]:.1f}%)')
            axes[1].axis('off')
        
        # Add statistics text
        stats_text = f"""
        Slum Coverage: {results['slum_percentage']:.1f}%
        Confidence: {results['confidence']:.3f}
        Threshold: {results['threshold']:.2f}
        """
        
        fig.suptitle('Slum Detection Results', fontsize=16, fontweight='bold')
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved: {save_path}")
        
        plt.show()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Slum Detection Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input image path or directory')
    parser.add_argument('--output', default='inference_results', help='Output directory')
    parser.add_argument('--model', default='balanced', help='Model configuration preset')
    parser.add_argument('--data', default='standard', help='Data configuration preset')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for multiple images')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--save_masks', action='store_true', help='Save binary masks')
    args = parser.parse_args()
    
    print("🎯 SLUM DETECTION INFERENCE")
    print("=" * 40)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference pipeline
    detector = SlumDetectionInference(
        checkpoint_path=args.checkpoint,
        model_config_name=args.model,
        data_config_name=args.data
    )
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image inference
        print(f"🖼️  Processing single image: {input_path}")
        
        results = detector.predict(
            str(input_path), 
            threshold=args.threshold, 
            return_probability=True
        )
        
        print(f"📊 Results:")
        print(f"   Slum coverage: {results['slum_percentage']:.2f}%")
        print(f"   Confidence: {results['confidence']:.3f}")
        print(f"   Slum pixels: {results['slum_pixels']:,}")
        
        # Save results
        result_data = {
            'image_path': str(input_path),
            'slum_percentage': results['slum_percentage'],
            'slum_pixels': int(results['slum_pixels']),
            'total_pixels': int(results['total_pixels']),
            'confidence': float(results['confidence']),
            'threshold': results['threshold'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'inference_results.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Visualize if requested
        if args.visualize:
            detector.visualize_prediction(
                results, 
                save_path=output_dir / 'prediction_visualization.png'
            )
        
        # Save mask if requested
        if args.save_masks:
            mask_path = output_dir / f'{input_path.stem}_mask.png'
            cv2.imwrite(str(mask_path), results['binary_mask'] * 255)
            print(f"💾 Mask saved: {mask_path}")
    
    elif input_path.is_dir():
        # Batch inference
        print(f"📁 Processing directory: {input_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("❌ No image files found in directory!")
            return
        
        print(f"   Found {len(image_files)} images")
        
        # Process batch
        batch_results = detector.predict_batch(
            [str(f) for f in image_files],
            threshold=args.threshold,
            batch_size=args.batch_size
        )
        
        # Compile results
        summary_results = []
        for i, (img_file, result) in enumerate(zip(image_files, batch_results)):
            summary_results.append({
                'image_path': str(img_file),
                'image_name': img_file.name,
                'slum_percentage': result['slum_percentage'],
                'slum_pixels': int(result['slum_pixels']),
                'total_pixels': int(result['total_pixels']),
                'confidence': float(result['confidence']),
                'threshold': result['threshold']
            })
            
            # Save individual mask if requested
            if args.save_masks:
                mask_path = output_dir / 'masks' / f'{img_file.stem}_mask.png'
                mask_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(mask_path), result['binary_mask'] * 255)
        
        # Save batch results
        batch_summary = {
            'total_images': len(image_files),
            'avg_slum_percentage': np.mean([r['slum_percentage'] for r in summary_results]),
            'images_with_slums': sum(1 for r in summary_results if r['slum_percentage'] > 0),
            'results': summary_results,
            'threshold': args.threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'batch_inference_results.json', 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"📊 Batch Results:")
        print(f"   Total images: {batch_summary['total_images']}")
        print(f"   Images with slums: {batch_summary['images_with_slums']}")
        print(f"   Average slum coverage: {batch_summary['avg_slum_percentage']:.2f}%")
        
        # Create summary visualization
        if args.visualize and len(batch_results) > 0:
            # Select top slum images for visualization
            top_slum_indices = sorted(range(len(batch_results)), 
                                    key=lambda i: batch_results[i]['slum_percentage'], 
                                    reverse=True)[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, idx in enumerate(top_slum_indices):
                if i >= 6:
                    break
                
                result = batch_results[idx]
                axes[i].imshow(result['original_image'])
                axes[i].imshow(result['binary_mask'], cmap='Reds', alpha=0.6)
                axes[i].set_title(f'{image_files[idx].name}\n'
                                f'Slums: {result["slum_percentage"]:.1f}%')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(top_slum_indices), 6):
                axes[i].axis('off')
            
            plt.suptitle('Top Slum Detection Results', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'batch_visualization.png', dpi=300, bbox_inches='tight')
            print(f"📊 Batch visualization saved: {output_dir}/batch_visualization.png")
    
    else:
        print(f"❌ Input path not found: {input_path}")
        return
    
    print(f"\n✅ Inference completed!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
