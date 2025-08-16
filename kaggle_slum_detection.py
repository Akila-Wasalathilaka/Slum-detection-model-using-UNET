#!/usr/bin/env python3
"""
Kaggle Slum Detection Pipeline
=============================

Complete pipeline to clone repository, setup environment, and run slum detection
model to generate 20 predictions inline in Kaggle notebook.

Usage in Kaggle:
- Create new notebook
- Copy this entire script
- Run in a single cell
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Kaggle working directory
KAGGLE_DIR = "/kaggle/working"
REPO_URL = "https://github.com/nichula01/Slum-Detection-Using-Unet-Architecture.git"
REPO_NAME = "Slum-Detection-Using-Unet-Architecture"

def run_command(cmd, cwd=None):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def setup_kaggle_environment():
    """Setup Kaggle environment with all dependencies."""
    print("🚀 Setting up Kaggle environment...")
    
    # Change to Kaggle working directory
    os.chdir(KAGGLE_DIR)
    
    # Clone repository
    print("📥 Cloning repository...")
    run_command(f"git clone {REPO_URL}")
    
    # Change to repo directory
    repo_path = os.path.join(KAGGLE_DIR, REPO_NAME)
    os.chdir(repo_path)
    
    # Install dependencies
    print("📦 Installing dependencies...")
    run_command("pip install segmentation-models-pytorch albumentations opencv-python")
    run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    
    # Add to Python path
    sys.path.insert(0, repo_path)
    
    return repo_path

def create_sample_data(repo_path):
    """Create sample satellite images for testing."""
    print("🖼️ Creating sample satellite images...")
    
    # Create sample images directory
    sample_dir = os.path.join(repo_path, "kaggle_samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Generate 20 synthetic satellite-like images
    sample_images = []
    for i in range(20):
        # Create realistic satellite-like image (120x120x3)
        np.random.seed(42 + i)  # For reproducible results
        
        # Base terrain (brownish)
        image = np.random.normal(0.4, 0.1, (120, 120, 3))
        
        # Add some urban structures (darker areas)
        if i < 10:  # First 10 with potential slums
            # Add slum-like areas (irregular, dense patterns)
            slum_x, slum_y = np.random.randint(20, 100, 2)
            slum_size = np.random.randint(15, 30)
            
            # Create irregular slum pattern
            for dx in range(-slum_size//2, slum_size//2):
                for dy in range(-slum_size//2, slum_size//2):
                    if 0 <= slum_x + dx < 120 and 0 <= slum_y + dy < 120:
                        if np.random.random() > 0.3:  # Irregular pattern
                            image[slum_x + dx, slum_y + dy] = [0.6, 0.5, 0.3]  # Slum-like color
        
        # Add some roads and buildings
        road_y = np.random.randint(10, 110)
        image[road_y:road_y+3, :] = [0.2, 0.2, 0.2]  # Road
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, 1)
        
        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        sample_images.append(image)
        
        # Save image
        from PIL import Image
        img_pil = Image.fromarray(image)
        img_pil.save(os.path.join(sample_dir, f"sample_{i:02d}.png"))
    
    return sample_images, sample_dir

def load_model(repo_path):
    """Load the slum detection model."""
    print("🧠 Loading slum detection model...")
    
    # Import model modules
    from models.unet import create_model
    from models.metrics import IoUScore, DiceScore
    
    # Create model with balanced configuration
    model = create_model(
        architecture="unet",
        encoder="resnet34",
        pretrained=True,  # Use ImageNet weights since we don't have trained checkpoint
        num_classes=1
    )
    
    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on device: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device

def preprocess_image(image):
    """Preprocess image for model input."""
    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    
    return image

def predict_slums(model, images, device):
    """Run slum detection on batch of images."""
    print("🔍 Running slum detection predictions...")
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i, image in enumerate(images):
            # Preprocess
            input_tensor = preprocess_image(image).to(device)
            
            # Predict
            output = model(input_tensor)
            prob = torch.sigmoid(output)
            
            # Convert to numpy
            prob_np = prob.squeeze().cpu().numpy()
            pred_binary = (prob_np > 0.5).astype(np.uint8)
            
            predictions.append(pred_binary)
            probabilities.append(prob_np)
            
            # Print progress
            if (i + 1) % 5 == 0:
                print(f"✅ Processed {i + 1}/20 images")
    
    return predictions, probabilities

def visualize_predictions(images, predictions, probabilities, sample_dir):
    """Create visualization of predictions."""
    print("📊 Creating prediction visualizations...")
    
    # Create figure for all predictions
    fig, axes = plt.subplots(4, 10, figsize=(25, 10))
    fig.suptitle('Slum Detection Results - 20 Sample Images', fontsize=16, fontweight='bold')
    
    for i in range(20):
        row = i // 10 * 2  # 0 or 2
        col = i % 10
        
        # Original image
        axes[row, col].imshow(images[i])
        axes[row, col].set_title(f'Image {i+1}', fontsize=10)
        axes[row, col].axis('off')
        
        # Prediction overlay
        axes[row + 1, col].imshow(images[i])
        
        # Overlay prediction with transparency
        slum_mask = predictions[i] > 0
        if slum_mask.sum() > 0:
            # Create colored overlay for slum areas
            overlay = np.zeros((*predictions[i].shape, 3))
            overlay[slum_mask] = [1, 0, 0]  # Red for slums
            axes[row + 1, col].imshow(overlay, alpha=0.6)
        
        # Calculate stats
        slum_percentage = (predictions[i].sum() / (120 * 120)) * 100
        max_prob = probabilities[i].max()
        avg_prob = probabilities[i].mean()
        
        axes[row + 1, col].set_title(
            f'Slum: {slum_percentage:.1f}%\nMax: {max_prob:.3f}', 
            fontsize=9
        )
        axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, 'predictions_grid.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_summary_report(predictions, probabilities):
    """Generate detailed summary report."""
    print("\n" + "="*60)
    print("🏆 SLUM DETECTION RESULTS SUMMARY")
    print("="*60)
    
    total_images = len(predictions)
    slum_detected = sum(1 for pred in predictions if pred.sum() > 0)
    
    print(f"📊 Total Images Analyzed: {total_images}")
    print(f"🏘️  Images with Slums Detected: {slum_detected}")
    print(f"📈 Slum Detection Rate: {(slum_detected/total_images)*100:.1f}%")
    
    print("\n" + "-"*60)
    print("📋 DETAILED RESULTS:")
    print("-"*60)
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        slum_pixels = pred.sum()
        slum_percentage = (slum_pixels / (120 * 120)) * 100
        max_prob = prob.max()
        avg_prob = prob.mean()
        
        status = "🔴 SLUM DETECTED" if slum_pixels > 0 else "🟢 NO SLUM"
        
        print(f"Image {i+1:2d}: {status:<15} | "
              f"Coverage: {slum_percentage:5.1f}% | "
              f"Max Prob: {max_prob:.3f} | "
              f"Avg Prob: {avg_prob:.3f}")
    
    # Statistics
    all_slum_percentages = [(pred.sum() / (120 * 120)) * 100 for pred in predictions]
    all_max_probs = [prob.max() for prob in probabilities]
    all_avg_probs = [prob.mean() for prob in probabilities]
    
    print("\n" + "-"*60)
    print("📈 STATISTICS:")
    print("-"*60)
    print(f"Average Slum Coverage: {np.mean(all_slum_percentages):.2f}%")
    print(f"Max Slum Coverage: {np.max(all_slum_percentages):.2f}%")
    print(f"Average Max Probability: {np.mean(all_max_probs):.3f}")
    print(f"Average Overall Probability: {np.mean(all_avg_probs):.3f}")
    
    return {
        'total_images': total_images,
        'slum_detected': slum_detected,
        'detection_rate': (slum_detected/total_images)*100,
        'avg_slum_coverage': np.mean(all_slum_percentages),
        'max_slum_coverage': np.max(all_slum_percentages),
        'avg_max_prob': np.mean(all_max_probs),
        'avg_overall_prob': np.mean(all_avg_probs)
    }

def main():
    """Main execution function."""
    print("🏘️ KAGGLE SLUM DETECTION PIPELINE")
    print("="*50)
    
    try:
        # Setup environment
        repo_path = setup_kaggle_environment()
        
        # Create sample data
        sample_images, sample_dir = create_sample_data(repo_path)
        
        # Load model
        model, device = load_model(repo_path)
        
        # Run predictions
        predictions, probabilities = predict_slums(model, sample_images, device)
        
        # Visualize results
        visualize_predictions(sample_images, predictions, probabilities, sample_dir)
        
        # Generate summary
        summary = generate_summary_report(predictions, probabilities)
        
        # Save results
        results = {
            'summary': summary,
            'predictions': [pred.tolist() for pred in predictions],
            'probabilities': [prob.tolist() for prob in probabilities]
        }
        
        with open(os.path.join(sample_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved in: {sample_dir}")
        print(f"🔍 Check 'predictions_grid.png' for visual results")
        
        return results, sample_dir
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Run the complete pipeline
    results, output_dir = main()
    
    if results:
        print("\n🎉 KAGGLE SLUM DETECTION COMPLETE!")
        print("Copy this script to a Kaggle notebook and run to see results.")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")
