"""
Inference Script for Slum Detection
===================================

Generate predictions on a directory of images using a trained model.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import cv2
from PIL import Image
import glob
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import create_model
from config import get_model_config
from utils.checkpoint import load_checkpoint
from utils.transforms import get_val_transforms


def load_model_from_checkpoint(checkpoint_path, model_config, device):
    """Load model from checkpoint file."""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Create model
    model = create_model(
        architecture=model_config.architecture,
        encoder=model_config.encoder,
        pretrained=False,
        num_classes=model_config.num_classes
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_single_image(model, image_path, transform, device, threshold=0.5):
    """Predict on a single image."""
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    except:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    original_size = image.shape[:2]
    image = cv2.resize(image, (120, 120))
    
    # Apply transforms
    if transform:
        transformed = transform(image=image)
        image_tensor = transformed['image']
    else:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        prediction = prediction.cpu().numpy()[0, 0]  # Remove batch and channel dims
    
    # Resize prediction back to original size
    prediction_resized = cv2.resize(prediction, (original_size[1], original_size[0]))
    
    # Apply threshold
    binary_mask = (prediction_resized > threshold).astype(np.uint8)
    
    return {
        'prediction': prediction_resized,
        'binary_mask': binary_mask,
        'slum_percentage': np.mean(binary_mask),
        'slum_detected': np.sum(binary_mask) > 0
    }


def batch_predict(model, input_dir, output_dir, transform, device, threshold=0.5, save_visualizations=True):
    """Predict on all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_path / ext)))
        image_files.extend(glob.glob(str(input_path / ext.upper())))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"🔍 Found {len(image_files)} images to process")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        image_name = Path(image_path).stem
        print(f"Processing {i+1}/{len(image_files)}: {image_name}")
        
        try:
            # Predict
            result = predict_single_image(model, image_path, transform, device, threshold)
            
            # Save prediction
            pred_path = output_path / f"{image_name}_prediction.png"
            cv2.imwrite(str(pred_path), (result['prediction'] * 255).astype(np.uint8))
            
            # Save binary mask
            mask_path = output_path / f"{image_name}_mask.png"
            cv2.imwrite(str(mask_path), result['binary_mask'] * 255)
            
            # Create visualization if requested
            if save_visualizations:
                # Load original image for visualization
                orig_img = cv2.imread(image_path)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                # Create overlay
                mask_colored = np.zeros_like(orig_img)
                mask_colored[result['binary_mask'] == 1] = [255, 0, 0]  # Red for slums
                
                # Blend with original
                overlay = cv2.addWeighted(orig_img, 0.7, mask_colored, 0.3, 0)
                
                # Save visualization
                viz_path = output_path / f"{image_name}_overlay.png"
                cv2.imwrite(str(viz_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Store results
            results.append({
                'image_name': image_name,
                'image_path': str(image_path),
                'slum_percentage': float(result['slum_percentage']),
                'slum_detected': bool(result['slum_detected']),
                'prediction_path': str(pred_path),
                'mask_path': str(mask_path)
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_name}: {str(e)}")
            continue
    
    # Save summary
    summary = {
        'total_images': len(image_files),
        'successful_predictions': len(results),
        'slum_detected_count': sum(1 for r in results if r['slum_detected']),
        'average_slum_percentage': np.mean([r['slum_percentage'] for r in results]) if results else 0,
        'threshold_used': threshold,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    summary_path = output_path / "prediction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Batch prediction completed!")
    print(f"📊 Summary:")
    print(f"   Total images: {summary['total_images']}")
    print(f"   Successful predictions: {summary['successful_predictions']}")
    print(f"   Images with slums detected: {summary['slum_detected_count']}")
    print(f"   Average slum percentage: {summary['average_slum_percentage']:.2%}")
    print(f"📁 Results saved to: {output_path}")
    
    return summary


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Slum Detection Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input directory with images')
    parser.add_argument('--output', required=True, help='Output directory for predictions')
    parser.add_argument('--config', default='balanced', help='Model configuration preset')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    args = parser.parse_args()
    
    print("🔮 SLUM DETECTION INFERENCE")
    print("=" * 40)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Load model config
    model_config = get_model_config(args.config)
    
    # Load model
    model, checkpoint_info = load_model_from_checkpoint(args.checkpoint, model_config, device)
    
    # Create transforms (same as validation)
    from config import get_data_config
    data_config = get_data_config("standard")
    transforms = get_val_transforms(data_config)
    
    # Run batch prediction
    results = batch_predict(
        model=model,
        input_dir=args.input,
        output_dir=args.output,
        transform=transforms,
        device=device,
        threshold=args.threshold,
        save_visualizations=not args.no_viz
    )
    
    return results


if __name__ == "__main__":
    main()
