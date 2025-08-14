import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import create_model
from config import get_model_config
from utils.transforms import get_val_transforms
from config.data_config import get_data_config

class AdvancedSlumDetector:
    def __init__(self, checkpoint_path, model_config="balanced"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = get_model_config(model_config)
        self.data_config = get_data_config("standard")
        
        # Load model
        self.model = create_model(
            architecture=self.model_config.architecture,
            encoder=self.model_config.encoder,
            pretrained=False,
            num_classes=self.model_config.num_classes
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_val_transforms(self.data_config)
    
    def predict_single(self, image_path, threshold=0.3):
        """Predict on single image."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image_array)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Calculate stats
        slum_mask = prob > threshold
        slum_percentage = (slum_mask.sum() / slum_mask.size) * 100
        mean_confidence = prob.mean()
        
        return {
            'prediction': prob,
            'binary_mask': slum_mask,
            'area_stats': {
                'slum_percentage': slum_percentage,
                'total_pixels': slum_mask.size,
                'slum_pixels': slum_mask.sum()
            },
            'confidence_stats': {
                'mean_confidence': mean_confidence,
                'max_confidence': prob.max(),
                'min_confidence': prob.min()
            }
        }
    
    def predict_batch(self, image_dir, output_dir, threshold=0.3, save_visualizations=True):
        """Process batch of images."""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        for img_path in image_files:
            try:
                result = self.predict_single(img_path, threshold)
                result['image_name'] = img_path.name
                results.append(result)
                
                if save_visualizations:
                    self._save_visualization(img_path, result, output_dir)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save batch results
        batch_results = {
            'summary_stats': {
                'total_images': len(results),
                'slum_detected': sum(1 for r in results if r['area_stats']['slum_percentage'] > 1.0)
            },
            'results': results
        }
        
        with open(output_dir / 'batch_results.json', 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        return batch_results
    
    def _save_visualization(self, image_path, result, output_dir):
        """Save prediction visualization."""
        # Load original image
        image = Image.open(image_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Prediction probability
        axes[1].imshow(result['prediction'], cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f"Confidence: {result['confidence_stats']['mean_confidence']:.3f}")
        axes[1].axis('off')
        
        # Binary mask
        axes[2].imshow(result['binary_mask'], cmap='gray')
        axes[2].set_title(f"Slum: {result['area_stats']['slum_percentage']:.1f}%")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{image_path.stem}_prediction.png", dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    
    args = parser.parse_args()
    
    detector = AdvancedSlumDetector(args.checkpoint)
    
    if args.batch:
        results = detector.predict_batch(args.input, args.output, args.threshold)
        print(f"Processed {results['summary_stats']['total_images']} images")
    else:
        result = detector.predict_single(args.input, args.threshold)
        print(f"Slum percentage: {result['area_stats']['slum_percentage']:.1f}%")