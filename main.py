"""
Ultra-accurate slum detection pipeline with advanced mapping capabilities.
"""
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse
import json
from datetime import datetime

# Import all modules
from config import config
from model import UltraAccurateUNet
from trainer import UltraAccurateTrainer
from inference import UltraAccurateInference
from dataset import get_ultra_data_loaders  # Changed from data_loader
from utils import (
    setup_ultra_logging, calculate_ultra_metrics, 
    save_training_plots, visualize_ultra_results
)

def setup_ultra_environment():
    """Setup ultra-accurate environment for slum detection."""
    # Create output directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, 'maps'), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, 'visualizations'), exist_ok=True)
    
    # Setup device with memory optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        
        print(f"💻 GPU: {torch.cuda.get_device_name()}")
        print(f"🔢 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
    
    return device

def train_ultra_model(device: torch.device) -> Dict:
    """Train the ultra-accurate slum detection model."""
    print("\n🎯 Starting Ultra-Accurate Slum Detection Training...")
    
    # Setup data loaders
    train_loader, val_loader, test_loader = get_ultra_data_loaders()
    
    # Initialize model
    model = UltraAccurateUNet().to(device)
    
    # Initialize trainer with correct arguments
    trainer = UltraAccurateTrainer(model, train_loader, val_loader, device)
    
    # Train model
    results = trainer.train()  # No parameters needed
    
    print("✅ Training completed successfully!")
    return results

def evaluate_ultra_model(device: torch.device, model_path: str = None) -> Dict:
    """Evaluate the ultra-accurate model with comprehensive metrics."""
    print("\n📊 Evaluating Ultra-Accurate Model...")
    
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_ultra_model.pth')
    
    # Load test data
    _, _, test_loader = get_ultra_data_loaders()
    
    # Initialize inference
    inference = UltraAccurateInference(model_path, device, use_tta=True)
    
    all_predictions = []
    all_targets = []
    all_scores = []
    
    for batch in test_loader:
        images, masks = batch
        images = images.to(device)
        
        batch_results = []
        for i in range(images.size(0)):
            # Save temporary image for inference
            temp_path = "temp_image.jpg"
            image_np = images[i].cpu().numpy().transpose(1, 2, 0)
            image_np = ((image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            
            # Run inference
            result = inference.predict_single_with_mapping(temp_path)
            
            prediction = cv2.resize(result['prediction'], (config.PRIMARY_SIZE, config.PRIMARY_SIZE))
            target = masks[i].numpy()
            
            all_predictions.append(prediction)
            all_targets.append(target)
            
            # Calculate confidence score
            confidence_score = np.mean(result['confidence_map'])
            all_scores.append(confidence_score)
            
            batch_results.append(result)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Calculate ultra metrics
    evaluation_results = calculate_ultra_metrics(
        np.array(all_predictions), 
        np.array(all_targets), 
        np.array(all_scores)
    )
    
    print("✅ Evaluation completed!")
    return evaluation_results

def create_demo_maps(device: torch.device, model_path: str = None):
    """Create demonstration slum maps from test images."""
    print("\n🗺️ Creating Ultra-Accurate Slum Maps...")
    
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_ultra_model.pth')
    
    # Initialize inference
    inference = UltraAccurateInference(model_path, device, use_tta=True)
    
    # Get test images
    test_dir = os.path.join(config.DATA_ROOT, 'test', 'images')
    test_images = list(Path(test_dir).glob('*.tif'))[:10]  # Process first 10 for demo
    
    demo_results = []
    
    for i, image_path in enumerate(test_images):
        print(f"📍 Processing {image_path.name}...")
        
        # Run ultra-accurate inference with mapping
        result = inference.predict_single_with_mapping(str(image_path))
        
        # Save visualizations
        save_path_base = os.path.join(config.RESULTS_DIR, 'maps', f'demo_{i+1}')
        
        # Save original image
        cv2.imwrite(f'{save_path_base}_original.jpg', 
                   cv2.cvtColor(result['original_image'], cv2.COLOR_RGB2BGR))
        
        # Save slum map with markers
        cv2.imwrite(f'{save_path_base}_slum_map.jpg', 
                   cv2.cvtColor(result['slum_map'], cv2.COLOR_RGB2BGR))
        
        # Save clustered map
        cv2.imwrite(f'{save_path_base}_clustered_map.jpg', 
                   cv2.cvtColor(result['clustered_map'], cv2.COLOR_RGB2BGR))
        
        # Save prediction overlay
        prediction_overlay = create_prediction_overlay(result['original_image'], result['prediction'])
        cv2.imwrite(f'{save_path_base}_prediction.jpg', 
                   cv2.cvtColor(prediction_overlay, cv2.COLOR_RGB2BGR))
        
        # Save comprehensive visualization
        comprehensive_viz = create_comprehensive_visualization(result)
        cv2.imwrite(f'{save_path_base}_comprehensive.jpg', 
                   cv2.cvtColor(comprehensive_viz, cv2.COLOR_RGB2BGR))
        
        demo_results.append({
            'image_name': image_path.name,
            'num_slum_regions': len(result['slum_regions']),
            'total_slum_area': sum(r.get('area', 0) for r in result['slum_regions']),
            'avg_confidence': np.mean([r.get('avg_confidence', 0) for r in result['slum_regions']]) if result['slum_regions'] else 0
        })
    
    # Save demo statistics
    with open(os.path.join(config.RESULTS_DIR, 'demo_statistics.json'), 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print("✅ Demo maps created successfully!")
    return demo_results

def create_prediction_overlay(original_image: np.ndarray, prediction: np.ndarray, 
                            alpha: float = 0.6) -> np.ndarray:
    """Create prediction overlay on original image."""
    # Normalize prediction to 0-255
    pred_normalized = (prediction * 255).astype(np.uint8)
    
    # Create colored mask (red for slums)
    colored_mask = np.zeros_like(original_image)
    colored_mask[:, :, 0] = pred_normalized  # Red channel
    
    # Blend with original image
    overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay

def create_comprehensive_visualization(result: Dict) -> np.ndarray:
    """Create comprehensive 2x2 visualization grid."""
    original = result['original_image']
    slum_map = result['slum_map']
    clustered_map = result['clustered_map']
    prediction_overlay = create_prediction_overlay(original, result['prediction'])
    
    # Resize all to same size
    size = (512, 512)
    original_resized = cv2.resize(original, size)
    slum_map_resized = cv2.resize(slum_map, size)
    clustered_map_resized = cv2.resize(clustered_map, size)
    prediction_resized = cv2.resize(prediction_overlay, size)
    
    # Add titles
    def add_title(image, title):
        # Add white border at top for title
        bordered = np.ones((size[1] + 30, size[0], 3), dtype=np.uint8) * 255
        bordered[30:, :] = image
        
        # Add title text
        cv2.putText(bordered, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return bordered
    
    original_titled = add_title(original_resized, "Original Image")
    prediction_titled = add_title(prediction_resized, "Prediction Overlay")
    slum_titled = add_title(slum_map_resized, "Slum Markers")
    clustered_titled = add_title(clustered_map_resized, "Clustered Analysis")
    
    # Create 2x2 grid
    top_row = np.hstack([original_titled, prediction_titled])
    bottom_row = np.hstack([slum_titled, clustered_titled])
    comprehensive = np.vstack([top_row, bottom_row])
    
    return comprehensive

def run_inference_on_custom_image(device: torch.device, image_path: str, model_path: str = None):
    """Run ultra-accurate inference on a custom image."""
    print(f"\n🔍 Running Ultra-Accurate Inference on {image_path}...")
    
    if model_path is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_ultra_model.pth')
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Initialize inference
    inference = UltraAccurateInference(model_path, device, use_tta=True)
    
    # Run inference
    result = inference.predict_single_with_mapping(image_path)
    
    # Create output directory
    output_dir = os.path.join(config.RESULTS_DIR, 'custom_inference')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save all visualizations
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_original.jpg'),
               cv2.cvtColor(result['original_image'], cv2.COLOR_RGB2BGR))
    
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_slum_map.jpg'),
               cv2.cvtColor(result['slum_map'], cv2.COLOR_RGB2BGR))
    
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_clustered.jpg'),
               cv2.cvtColor(result['clustered_map'], cv2.COLOR_RGB2BGR))
    
    comprehensive = create_comprehensive_visualization(result)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_comprehensive.jpg'),
               cv2.cvtColor(comprehensive, cv2.COLOR_RGB2BGR))
    
    # Print statistics
    num_regions = len(result['slum_regions'])
    total_area = sum(r.get('area', 0) for r in result['slum_regions'])
    avg_confidence = np.mean([r.get('avg_confidence', 0) for r in result['slum_regions']]) if result['slum_regions'] else 0
    
    print(f"📊 Analysis Results:")
    print(f"   • Slum regions detected: {num_regions}")
    print(f"   • Total slum area: {total_area:.0f} pixels")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    print(f"   • Results saved to: {output_dir}")
    
    print("✅ Custom inference completed!")

def print_ultra_banner():
    """Print ultra-accurate slum detection banner."""
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    🏘️  ULTRA-ACCURATE SLUM DETECTION  🏘️                     ║
║                                                                               ║
║                        Advanced AI-Powered Mapping System                    ║
║                     With Geographic Clustering & Visualization               ║
║                                                                               ║
║  Features:                                                                    ║
║  • Optimized resolution processing ({config.PRIMARY_SIZE}x{config.PRIMARY_SIZE})                        ║
║  • Memory-efficient for 4GB GPU                                              ║
║  • Fast training ({config.EPOCHS} epochs, batch size {config.BATCH_SIZE})                               ║
║  • Advanced loss functions for high accuracy                                 ║
║  • Geographic clustering analysis                                            ║
║  • Reference-quality slum mapping                                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Main ultra-accurate slum detection pipeline."""
    parser = argparse.ArgumentParser(description='Ultra-Accurate Slum Detection Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference', 'demo'], 
                       default='demo', help='Pipeline mode')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--image_path', type=str, help='Path to image for inference')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Print banner
    print_ultra_banner()
    
    # Setup environment
    device = setup_ultra_environment()
    
    # Setup logging
    logger = setup_ultra_logging()
    
    try:
        if args.mode == 'train':
            print("\n🚀 Starting Ultra-Accurate Training Pipeline...")
            
            # Update epochs if specified
            if args.epochs != config.NUM_EPOCHS:
                config.NUM_EPOCHS = args.epochs
                print(f"📊 Training for {args.epochs} epochs")
            
            results = train_ultra_model(device)
            
            # Save training results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(config.RESULTS_DIR, f'training_results_{timestamp}.json')
            
            # Handle case where results might be None
            if results is None:
                results = {"error": "Training completed but no results returned"}
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Training completed. Results saved to {results_file}")
            
        elif args.mode == 'evaluate':
            print("\n📊 Starting Ultra-Accurate Evaluation...")
            evaluation_results = evaluate_ultra_model(device, args.model_path)
            
            # Print evaluation summary
            print("\n📈 Evaluation Results:")
            for metric, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    print(f"   • {metric}: {value:.4f}")
            
        elif args.mode == 'inference':
            if not args.image_path:
                print("❌ Please provide --image_path for inference mode")
                return
            
            run_inference_on_custom_image(device, args.image_path, args.model_path)
            
        elif args.mode == 'demo':
            print("\n🗺️ Starting Ultra-Accurate Demo Pipeline...")
            demo_results = create_demo_maps(device, args.model_path)
            
            # Print demo summary
            print("\n📊 Demo Results Summary:")
            total_regions = sum(r['num_slum_regions'] for r in demo_results)
            avg_confidence = np.mean([r['avg_confidence'] for r in demo_results if r['avg_confidence'] > 0])
            
            print(f"   • Images processed: {len(demo_results)}")
            print(f"   • Total slum regions detected: {total_regions}")
            print(f"   • Average confidence: {avg_confidence:.3f}")
            print(f"   • Maps saved to: {os.path.join(config.RESULTS_DIR, 'maps')}")
    
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise
    
    print("\n✅ Ultra-Accurate Slum Detection Pipeline completed successfully!")

if __name__ == "__main__":
    main()
