#!/usr/bin/env python3
"""
Inference script to generate prediction overlays.
Creates 20 red overlay predictions on test/validation images.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import yaml
import torch
import cv2
import rasterio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from slumseg.data.dataset import SlumDataset, get_valid_transforms, get_image_ids_from_dir
from slumseg.models.factory import make_model
from slumseg.utils.visualize import create_red_overlay, save_prediction_overlay
from slumseg.train_loop import predict_batch


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    
    # Create model
    model = make_model(
        arch=config['model']['arch'],
        encoder=config['model']['encoder'],
        classes=config['model']['classes'],
        in_channels=config['model']['in_channels'],
        pretrained=False  # We're loading weights
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply optimizations
    if config['train'].get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
    
    # Compile if requested
    if config['train'].get('compile_mode'):
        try:
            model = torch.compile(model, mode=config['train']['compile_mode'])
        except Exception as e:
            print(f"Failed to compile model: {e}")
    
    model = model.to(device)
    model.eval()
    
    return model


def postprocess_mask(mask: np.ndarray, config: dict) -> np.ndarray:
    """Apply post-processing to prediction mask."""
    
    if not config['infer'].get('postprocess', False):
        return mask
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Closing to fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def predict_on_image(
    model: torch.nn.Module,
    image_path: str,
    transform,
    device: torch.device,
    config: dict
) -> tuple:
    """Predict on a single image and return RGB image + prediction mask."""
    
    # Load image
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  # RGB channels
        image = np.transpose(image, (1, 2, 0))  # HWC format
        original_image = image.copy()
        
        # Normalize to 0-255 if needed
        if image.max() > 255:
            image = (image / image.max() * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Apply transforms
    if transform:
        transformed = transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension
    else:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(device)
    
    # Predict
    use_tta = config['infer'].get('tta', False)
    use_amp = config['train'].get('amp', False)
    
    with torch.no_grad():
        outputs = predict_batch(model, image_tensor, device, use_tta=use_tta, use_amp=use_amp)
        
        # Convert to probability
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        # Apply threshold
        threshold = config['infer'].get('threshold', 0.5)
        pred_mask = (probs > threshold).astype(np.uint8)
        
        # Post-process
        pred_mask = postprocess_mask(pred_mask, config)
    
    return original_image, pred_mask, probs


def main():
    parser = argparse.ArgumentParser(description='Generate prediction overlays')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--images', type=str, required=True, help='Images directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--num', type=int, default=20, help='Number of predictions to generate')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.ckpt, config, device)
    
    # Get image IDs
    image_ids = get_image_ids_from_dir(args.images)
    
    if not image_ids:
        print(f"No images found in {args.images}")
        return
    
    # Select random images
    np.random.seed(42)
    if len(image_ids) > args.num:
        selected_ids = np.random.choice(image_ids, args.num, replace=False)
    else:
        selected_ids = image_ids[:args.num]
    
    print(f"Generating predictions for {len(selected_ids)} images...")
    
    # Create transforms
    transform = get_valid_transforms()
    
    # Generate predictions
    alpha = config['infer'].get('overlay_alpha', 0.4)
    
    for i, img_id in enumerate(tqdm(selected_ids, desc="Predicting")):
        try:
            # Sanitize filename to prevent path traversal
            clean_id = os.path.basename(str(img_id).replace('..', ''))
            image_path = Path(args.images) / f"{clean_id}.tif"
            
            # Predict
            rgb_image, pred_mask, prob_map = predict_on_image(
                model, str(image_path), transform, device, config
            )
            
            # Create overlay
            overlay = create_red_overlay(rgb_image, pred_mask, alpha=alpha)
            
            # Sanitize output path
            safe_filename = f"pred_{i+1:02d}.png"
            output_path = output_dir / safe_filename
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Optionally save probability map
            prob_output_path = output_dir / f"prob_{i+1:02d}.png"
            prob_vis = (prob_map * 255).astype(np.uint8)
            cv2.imwrite(str(prob_output_path), prob_vis)
            
            print(f"Saved: {output_path.name} (Slum pixels: {pred_mask.sum()}/{pred_mask.size})")
            
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue
    
    print(f"\nGenerated {len(selected_ids)} prediction overlays in: {output_dir}")
    
    # Create summary
    summary_file = output_dir / "predictions_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Prediction Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Model: {config['model']['arch']} + {config['model']['encoder']}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Threshold: {config['infer'].get('threshold', 0.5)}\n")
        f.write(f"TTA: {config['infer'].get('tta', False)}\n")
        f.write(f"Post-processing: {config['infer'].get('postprocess', False)}\n")
        f.write(f"Overlay alpha: {alpha}\n")
        f.write(f"Total predictions: {len(selected_ids)}\n")
        f.write(f"\nImage IDs:\n")
        for i, img_id in enumerate(selected_ids):
            f.write(f"  pred_{i+1:02d}.png: {img_id}\n")


if __name__ == "__main__":
    main()
