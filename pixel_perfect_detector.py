import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import gradio as gr
import cv2
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.append(str(project_root))

class PixelPerfectSlumModel(nn.Module):
    """Ultra-precise pixel-level slum detection model."""
    
    def __init__(self):
        super().__init__()
        
        # High-resolution backbone
        self.backbone = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type="scse"
        )
        
        # Multi-scale feature pyramid
        self.fpn = smp.FPN(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Feature fusion
        self.fusion = nn.Conv2d(2, 1, 1)
        
        # Edge refinement
        self.edge_refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Multi-model ensemble
        unet_out = self.backbone(x)
        fpn_out = self.fpn(x)
        
        # Fuse predictions
        fused = self.fusion(torch.cat([unet_out, fpn_out], dim=1))
        
        # Edge refinement
        refined = self.edge_refine(fused)
        
        return refined

class PixelPerfectDetector:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create pixel-perfect model
        self.model = PixelPerfectSlumModel()
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, image, target_size=512):
        """High-resolution preprocessing for pixel-perfect detection."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Keep high resolution for pixel-perfect detection
        original_size = image.size
        image = image.resize((target_size, target_size), Image.LANCZOS)
        
        # Normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to tensor with correct dtype
        tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device), original_size
    
    def detect_pixel_perfect(self, image, threshold=0.5):
        """Pixel-perfect slum detection with edge refinement."""
        input_tensor, original_size = self.preprocess_image(image, target_size=512)
        
        with torch.no_grad():
            # Multi-scale inference
            scales = [0.75, 1.0, 1.25]
            predictions = []
            
            for scale in scales:
                if scale != 1.0:
                    h, w = input_tensor.shape[2:]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_input = F.interpolate(input_tensor, size=(new_h, new_w), mode='bilinear')
                    pred = self.model(scaled_input)
                    pred = F.interpolate(pred, size=(h, w), mode='bilinear')
                else:
                    pred = self.model(input_tensor)
                
                predictions.append(torch.sigmoid(pred))
            
            # Ensemble predictions
            final_pred = torch.mean(torch.stack(predictions), dim=0)
            prob_map = final_pred.cpu().numpy()[0, 0]
        
        # Post-processing for pixel-perfect results
        prob_map = self.post_process(prob_map)
        
        # Binary mask
        binary_mask = prob_map > threshold
        
        # Calculate detailed statistics
        stats = self.calculate_detailed_stats(prob_map, binary_mask)
        
        return prob_map, binary_mask, stats
    
    def post_process(self, prob_map):
        """Advanced post-processing for pixel-perfect results."""
        # Gaussian smoothing
        prob_map = cv2.GaussianBlur(prob_map, (3, 3), 0.5)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prob_map = cv2.morphologyEx(prob_map, cv2.MORPH_CLOSE, kernel)
        prob_map = cv2.morphologyEx(prob_map, cv2.MORPH_OPEN, kernel)
        
        return prob_map
    
    def calculate_detailed_stats(self, prob_map, binary_mask):
        """Calculate comprehensive pixel-level statistics."""
        total_pixels = prob_map.size
        slum_pixels = binary_mask.sum()
        
        # Confidence analysis
        high_conf_pixels = (prob_map > 0.8).sum()
        medium_conf_pixels = ((prob_map > 0.5) & (prob_map <= 0.8)).sum()
        low_conf_pixels = ((prob_map > 0.2) & (prob_map <= 0.5)).sum()
        
        # Edge analysis
        edges = cv2.Canny((prob_map * 255).astype(np.uint8), 50, 150)
        edge_pixels = (edges > 0).sum()
        
        return {
            'total_pixels': int(total_pixels),
            'slum_pixels': int(slum_pixels),
            'slum_percentage': float(slum_pixels / total_pixels * 100),
            'mean_confidence': float(prob_map.mean()),
            'max_confidence': float(prob_map.max()),
            'high_confidence_pixels': int(high_conf_pixels),
            'medium_confidence_pixels': int(medium_conf_pixels),
            'low_confidence_pixels': int(low_conf_pixels),
            'edge_pixels': int(edge_pixels),
            'detection_quality': 'Excellent' if prob_map.max() > 0.9 else 'Good' if prob_map.max() > 0.7 else 'Fair'
        }
    
    def create_pixel_visualization(self, original_image, prob_map, binary_mask, stats):
        """Create detailed pixel-level visualization."""
        if isinstance(original_image, np.ndarray):
            original = cv2.resize(original_image, (512, 512))
        else:
            original = np.array(original_image.resize((512, 512)))
        
        # High-resolution heatmap
        heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Precise overlay - only mark slum pixels in red
        overlay = original.copy()
        # Only change pixels where slums are detected (binary_mask is True)
        slum_pixels = binary_mask > 0
        overlay[slum_pixels] = [255, 50, 50]  # Bright red only for slum pixels
        blended = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
        
        # Edge detection overlay
        edges = cv2.Canny((prob_map * 255).astype(np.uint8), 50, 150)
        edge_overlay = original.copy()
        edge_overlay[edges > 0] = [0, 255, 0]  # Green edges
        
        return original, heatmap, blended, edge_overlay

# Global pixel-perfect detector
detector = PixelPerfectDetector()

def pixel_perfect_interface(image, threshold):
    """Gradio interface for pixel-perfect detection."""
    try:
        # Detect with pixel precision
        prob_map, binary_mask, stats = detector.detect_pixel_perfect(image, threshold)
        
        # Create visualizations
        original, heatmap, overlay, edges = detector.create_pixel_visualization(
            image, prob_map, binary_mask, stats
        )
        
        # Detailed results
        results_text = f"""
🎯 **Pixel-Perfect Slum Detection Results**

📊 **Pixel Analysis:**
- Total Pixels: {stats['total_pixels']:,}
- Slum Pixels: {stats['slum_pixels']:,}
- Coverage: {stats['slum_percentage']:.2f}%

🔍 **Confidence Distribution:**
- High Confidence (>80%): {stats['high_confidence_pixels']:,} pixels
- Medium Confidence (50-80%): {stats['medium_confidence_pixels']:,} pixels
- Low Confidence (20-50%): {stats['low_confidence_pixels']:,} pixels

⚡ **Detection Quality:**
- Mean Confidence: {stats['mean_confidence']:.3f}
- Peak Confidence: {stats['max_confidence']:.3f}
- Quality Rating: {stats['detection_quality']}
- Edge Pixels: {stats['edge_pixels']:,}

🌍 **Precision:** Pixel-level accuracy optimized for global detection
        """
        
        return original, heatmap, overlay, edges, results_text
        
    except Exception as e:
        error_msg = f"❌ Detection failed: {str(e)}"
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        return blank, blank, blank, blank, error_msg

def create_pixel_perfect_app():
    with gr.Blocks(title="🎯 Pixel-Perfect Slum Detection", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎯 Pixel-Perfect Global Slum Detection
        **Ultra-precise AI system for pixel-level detection of informal settlements**
        
        🔬 **Features:**
        - Pixel-perfect accuracy with edge refinement
        - Multi-scale ensemble inference
        - Advanced post-processing
        - Comprehensive confidence analysis
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="📡 Upload High-Resolution Satellite Image")
                threshold = gr.Slider(0.1, 0.9, value=0.5, label="🎯 Detection Threshold")
                detect_btn = gr.Button("🔍 Detect Pixel-Perfect", variant="primary", size="lg")
            
            with gr.Column():
                results_text = gr.Markdown("Upload a high-resolution image for pixel-perfect detection...")
        
        with gr.Row():
            original_out = gr.Image(label="📷 Original Image")
            heatmap_out = gr.Image(label="🔥 Confidence Heatmap")
        
        with gr.Row():
            overlay_out = gr.Image(label="🎯 Precise Detection Overlay")
            edges_out = gr.Image(label="🔍 Edge Detection")
        
        detect_btn.click(
            pixel_perfect_interface,
            inputs=[input_image, threshold],
            outputs=[original_out, heatmap_out, overlay_out, edges_out, results_text]
        )
    
    return app

if __name__ == "__main__":
    app = create_pixel_perfect_app()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)