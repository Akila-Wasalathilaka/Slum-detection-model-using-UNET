import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr
import cv2
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models import create_model
from config import get_model_config
from utils.transforms import get_val_transforms
from config.data_config import get_data_config

class GlobalSlumDetector:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimized config for global detection
        self.model_config = get_model_config("accurate")
        self.data_config = get_data_config("heavy_augmentation")
        
        # Create model
        self.model = create_model(
            architecture="unet++",
            encoder="efficientnet-b3",
            pretrained=True,
            num_classes=1
        )
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_val_transforms(self.data_config)
    
    def preprocess_image(self, image):
        """Enhanced preprocessing for global compatibility."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Resize to model input size
        image = image.resize((120, 120), Image.LANCZOS)
        image_array = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image_array)
        return transformed['image'].unsqueeze(0).to(self.device)
    
    def detect_slums(self, image, threshold=0.3, return_confidence=True):
        """Detect slums with confidence mapping."""
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Binary mask
        slum_mask = prob_map > threshold
        
        # Calculate statistics
        total_pixels = slum_mask.size
        slum_pixels = slum_mask.sum()
        slum_percentage = (slum_pixels / total_pixels) * 100
        
        # Confidence stats
        confidence_stats = {
            'mean_confidence': float(prob_map.mean()),
            'max_confidence': float(prob_map.max()),
            'slum_coverage': float(slum_percentage),
            'detection_quality': 'High' if prob_map.max() > 0.8 else 'Medium' if prob_map.max() > 0.5 else 'Low'
        }
        
        if return_confidence:
            return prob_map, slum_mask, confidence_stats
        return slum_mask, confidence_stats
    
    def create_visualization(self, original_image, prob_map, slum_mask, stats):
        """Create comprehensive visualization."""
        # Resize original to match prediction
        if isinstance(original_image, np.ndarray):
            original = cv2.resize(original_image, (120, 120))
        else:
            original = np.array(original_image.resize((120, 120)))
        
        # Create overlay
        overlay = original.copy()
        overlay[slum_mask] = [255, 0, 0]  # Red for slums
        blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
        
        # Create heatmap
        heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        return original, heatmap, blended

# Global detector instance
detector = GlobalSlumDetector()

def gradio_interface(image, threshold):
    """Gradio interface function."""
    try:
        # Detect slums
        prob_map, slum_mask, stats = detector.detect_slums(image, threshold)
        
        # Create visualizations
        original, heatmap, overlay = detector.create_visualization(image, prob_map, slum_mask, stats)
        
        # Results text
        results_text = f"""
🏘️ **Global Slum Detection Results**

📊 **Coverage Analysis:**
- Slum Coverage: {stats['slum_coverage']:.1f}%
- Detection Quality: {stats['detection_quality']}

🎯 **Confidence Metrics:**
- Mean Confidence: {stats['mean_confidence']:.3f}
- Peak Confidence: {stats['max_confidence']:.3f}

🌍 **Global Compatibility:** Optimized for worldwide detection
        """
        
        return original, heatmap, overlay, results_text
        
    except Exception as e:
        error_msg = f"❌ Detection failed: {str(e)}"
        blank = np.zeros((120, 120, 3), dtype=np.uint8)
        return blank, blank, blank, error_msg

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(title="🌍 Global Slum Detection System", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏘️ Global Slum Detection System
        **AI-powered detection of informal settlements worldwide using satellite imagery**
        
        Upload any satellite image to detect slum areas with high accuracy!
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="📡 Upload Satellite Image")
                threshold = gr.Slider(0.1, 0.9, value=0.3, label="🎯 Detection Threshold")
                detect_btn = gr.Button("🔍 Detect Slums", variant="primary")
            
            with gr.Column():
                results_text = gr.Markdown("Upload an image to start detection...")
        
        with gr.Row():
            original_out = gr.Image(label="📷 Original Image")
            heatmap_out = gr.Image(label="🔥 Confidence Heatmap") 
            overlay_out = gr.Image(label="🎯 Detection Overlay")
        
        # Examples
        gr.Examples(
            examples=[
                ["data/test/images/sample1.jpg", 0.3],
                ["data/test/images/sample2.jpg", 0.5],
            ],
            inputs=[input_image, threshold],
            label="📋 Try These Examples"
        )
        
        detect_btn.click(
            gradio_interface,
            inputs=[input_image, threshold],
            outputs=[original_out, heatmap_out, overlay_out, results_text]
        )
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)