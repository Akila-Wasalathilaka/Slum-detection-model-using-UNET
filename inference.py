# File: inference.py
# Inference script for binary slum detection models

import os
import glob
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Import functions from the main script
from binary_slum_detection import (
    apply_morphological_postprocessing,
    get_validation_augmentation,
    Config
)

class InferenceDataset(Dataset):
    """Dataset for inference on new images."""
    
    def __init__(self, image_paths: List[str], transform: Optional[A.Compose] = None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, image_path

def load_model(model_path: str, encoder_name: str = 'resnet34') -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    
    # Create model
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # Don't load pretrained weights
        in_channels=3,
        classes=1,
        activation=None,
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def predict_single_image(
    model: torch.nn.Module,
    image: np.ndarray,
    transform: A.Compose,
    device: torch.device,
    threshold: float = 0.5,
    apply_postprocessing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on a single image.
    
    Returns:
        probability_map: Float array with prediction probabilities
        binary_mask: Binary mask after thresholding and post-processing
    """
    
    model.eval()
    
    # Prepare image
    original_height, original_width = image.shape[:2]
    
    # Apply transform
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        with torch.cuda.amp.autocast():
            output = model(image_tensor)
            probability_map = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Resize back to original size if needed
    if probability_map.shape != (original_height, original_width):
        probability_map = cv2.resize(
            probability_map, 
            (original_width, original_height), 
            interpolation=cv2.INTER_LINEAR
        )
    
    # Apply threshold
    binary_mask = (probability_map > threshold).astype(np.uint8)
    
    # Apply post-processing
    if apply_postprocessing:
        binary_mask = apply_morphological_postprocessing(binary_mask)
    
    return probability_map, binary_mask

def batch_inference(
    model_path: str,
    image_paths: List[str],
    output_dir: str,
    encoder_name: str = 'resnet34',
    image_size: int = 120,
    batch_size: int = 8,
    threshold: float = 0.5,
    apply_postprocessing: bool = True,
    save_probability_maps: bool = False,
    device: Optional[str] = None
) -> None:
    """Run batch inference on multiple images."""
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_probability_maps:
        prob_dir = output_dir / "probability_maps"
        prob_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, encoder_name)
    model = model.to(device)
    model.eval()
    
    # Setup data loader
    transform = get_validation_augmentation(image_size)
    dataset = InferenceDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Processing {len(image_paths)} images...")
    
    with torch.no_grad():
        for batch_images, batch_paths in tqdm(dataloader, desc="Inference"):
            batch_images = batch_images.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(batch_images)
                probability_maps = torch.sigmoid(outputs).cpu().numpy()
            
            # Process each image in batch
            for i, image_path in enumerate(batch_paths):
                prob_map = probability_maps[i, 0]  # Remove channel dimension
                
                # Load original image for resizing
                original_image = cv2.imread(image_path)
                original_height, original_width = original_image.shape[:2]
                
                # Resize probability map to original size
                if prob_map.shape != (original_height, original_width):
                    prob_map = cv2.resize(
                        prob_map, 
                        (original_width, original_height), 
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Create binary mask
                binary_mask = (prob_map > threshold).astype(np.uint8)
                
                # Apply post-processing
                if apply_postprocessing:
                    binary_mask = apply_morphological_postprocessing(binary_mask)
                
                # Save results
                base_name = Path(image_path).stem
                
                # Save binary mask
                mask_path = output_dir / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), binary_mask * 255)
                
                # Save probability map if requested
                if save_probability_maps:
                    prob_path = prob_dir / f"{base_name}_prob.png"
                    prob_map_uint8 = (prob_map * 255).astype(np.uint8)
                    cv2.imwrite(str(prob_path), prob_map_uint8)
    
    print(f"Inference completed! Results saved to: {output_dir}")

def visualize_predictions(
    image_paths: List[str],
    mask_dir: str,
    output_path: str,
    num_samples: int = 6
) -> None:
    """Create visualization of predictions."""
    
    mask_dir = Path(mask_dir)
    
    # Select random samples
    if len(image_paths) > num_samples:
        import random
        image_paths = random.sample(image_paths, num_samples)
    
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(12, 4 * len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for i, image_path in enumerate(image_paths):
        base_name = Path(image_path).stem
        mask_path = mask_dir / f"{base_name}_mask.png"
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load prediction mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Plot mask overlay
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(mask, alpha=0.6, cmap='Reds')
        axes[i, 1].set_title(f'Slum Prediction {i+1}')
        axes[i, 1].axis('off')
        
        # Plot mask only
        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title(f'Binary Mask {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description='Binary Slum Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    parser.add_argument('--encoder', type=str, default='resnet34',
                      choices=['resnet34', 'efficientnet-b0', 'timm-efficientnet-b3'],
                      help='Model encoder architecture')
    parser.add_argument('--image_size', type=int, default=120,
                      help='Input image size')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Binary classification threshold')
    parser.add_argument('--no_postprocessing', action='store_true',
                      help='Disable morphological post-processing')
    parser.add_argument('--save_prob_maps', action='store_true',
                      help='Save probability maps in addition to binary masks')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                      help='Device to use for inference')
    parser.add_argument('--visualize', action='store_true',
                      help='Create prediction visualizations')
    parser.add_argument('--extensions', nargs='+', default=['.tif', '.png', '.jpg', '.jpeg'],
                      help='Image file extensions to process')
    
    args = parser.parse_args()
    
    # Find all images in input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    image_paths = []
    for ext in args.extensions:
        image_paths.extend(glob.glob(str(input_dir / f"*{ext}")))
        image_paths.extend(glob.glob(str(input_dir / f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"Error: No images found in '{input_dir}' with extensions {args.extensions}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Run inference
    batch_inference(
        model_path=args.model_path,
        image_paths=image_paths,
        output_dir=args.output_dir,
        encoder_name=args.encoder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        apply_postprocessing=not args.no_postprocessing,
        save_probability_maps=args.save_prob_maps,
        device=args.device
    )
    
    # Create visualization if requested
    if args.visualize:
        viz_path = Path(args.output_dir) / "predictions_visualization.png"
        visualize_predictions(image_paths, args.output_dir, str(viz_path))

if __name__ == "__main__":
    main()
