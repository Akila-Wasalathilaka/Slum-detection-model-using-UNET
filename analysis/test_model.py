import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from unet_slum_detection import SlumDataset, get_transforms, create_model, calculate_metrics, device

def load_trained_model(model_path, config):
    """Load a trained model"""
    
    model = create_model(
        architecture=config['architecture'],
        encoder=config['encoder'],
        pretrained=False
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def test_model(model_path, test_data_dir='data/test', save_dir='test_results'):
    """Test the model on test set"""
    
    print(f"Testing model: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(
        architecture=config['architecture'],
        encoder=config['encoder'],
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Best validation IoU: {checkpoint['val_iou']:.4f}")
    
    # Create test dataset
    test_dataset = SlumDataset(
        image_dir=os.path.join(test_data_dir, 'images'),
        mask_dir=os.path.join(test_data_dir, 'masks'),
        transform=get_transforms('val'),
        target_size=config['target_size']
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Test on all samples
    all_preds = []
    all_targets = []
    sample_images = []
    sample_preds = []
    sample_targets = []
    
    print("Running inference...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, mask = test_dataset[i]
            
            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_batch)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            all_preds.append(pred)
            all_targets.append(mask.cpu().numpy())
            
            # Save some samples for visualization
            if i < 20:  # First 20 samples
                sample_images.append(image.cpu())
                sample_preds.append(pred)
                sample_targets.append(mask.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} samples")
    
    # Calculate overall metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = calculate_metrics(all_preds, all_targets)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Dice: {metrics['dice']:.4f}")
    
    # Create visualizations
    visualize_results(sample_images, sample_targets, sample_preds, save_dir)
    
    # Save detailed results
    save_detailed_results(all_preds, all_targets, metrics, save_dir)
    
    return metrics

def visualize_results(images, targets, predictions, save_dir, num_samples=16):
    """Create visualization of test results"""
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(images))):
        # Convert image for visualization
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        target = targets[i]
        pred = predictions[i]
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}: Input')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(target, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = img.copy()
        overlay[pred == 1] = [1, 0, 0]  # Red for predicted slums
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Prediction Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_results_visualization.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Test visualization saved to {save_dir}")

def save_detailed_results(predictions, targets, metrics, save_dir):
    """Save detailed test results"""
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    import json
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Confusion matrix
    cm = confusion_matrix(target_flat, pred_flat)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-Slum', 'Slum']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics to JSON
    results = {
        'test_metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'class_names': classes
    }
    
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to {save_dir}")

def compare_all_models(experiment_dir='experiments', test_data_dir='data/test'):
    """Compare all trained models on test set"""
    
    print("Comparing all trained models...")
    
    model_files = [f for f in os.listdir(experiment_dir) if f.startswith('best_model_') and f.endswith('.pth')]
    
    if not model_files:
        print("No trained models found!")
        return
    
    results = {}
    
    for model_file in model_files:
        loss_name = model_file.replace('best_model_', '').replace('.pth', '')
        model_path = os.path.join(experiment_dir, model_file)
        
        print(f"\nTesting {loss_name} model...")
        
        # Create save directory for this model
        save_dir = os.path.join('test_results', loss_name)
        
        # Test model
        metrics = test_model(model_path, test_data_dir, save_dir)
        results[loss_name] = metrics
    
    # Print comparison
    print(f"\n{'='*60}")
    print("TEST SET COMPARISON")
    print(f"{'='*60}")
    
    print(f"{'Model':<12} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'IoU':<6} | {'Dice':<6}")
    print("-" * 70)
    
    for loss_name, metrics in results.items():
        print(f"{loss_name:<12} | "
              f"{metrics['accuracy']:<8.4f} | "
              f"{metrics['precision']:<9.4f} | "
              f"{metrics['recall']:<6.4f} | "
              f"{metrics['f1']:<6.4f} | "
              f"{metrics['iou']:<6.4f} | "
              f"{metrics['dice']:<6.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['iou'])
    print(f"\n🏆 Best model on test set: {best_model[0]} (IoU: {best_model[1]['iou']:.4f})")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained slum detection models')
    parser.add_argument('--model', type=str, help='Path to specific model to test')
    parser.add_argument('--compare-all', action='store_true', help='Compare all trained models')
    parser.add_argument('--test-dir', type=str, default='data/test', help='Test data directory')
    
    args = parser.parse_args()
    
    if args.compare_all:
        compare_all_models(test_data_dir=args.test_dir)
    elif args.model:
        test_model(args.model, test_data_dir=args.test_dir)
    else:
        print("Please specify --model or --compare-all")
        print("Example: python test_model.py --compare-all")
        print("Example: python test_model.py --model experiments/best_model_focal.pth")
