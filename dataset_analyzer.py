"""
Comprehensive Dataset Analysis for Slum Detection
================================================
Analyze all classes and patterns in the dataset to build 100% accurate model
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import json

class SlumDatasetAnalyzer:
    def __init__(self, data_root="data"):
        self.data_root = Path(data_root)
        self.analysis_results = {}
        
    def analyze_complete_dataset(self):
        """Complete analysis of dataset classes and patterns."""
        print("🔍 COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 50)
        
        # 1. Analyze mask classes
        self.analyze_mask_classes()
        
        # 2. Analyze slum patterns
        self.analyze_slum_patterns()
        
        # 3. Analyze image characteristics
        self.analyze_image_characteristics()
        
        # 4. Generate comprehensive report
        self.generate_analysis_report()
        
        return self.analysis_results
    
    def analyze_mask_classes(self):
        """Analyze all unique classes in mask files."""
        print("\n📊 Analyzing mask classes...")
        
        all_classes = set()
        class_counts = Counter()
        slum_patterns = []
        
        # Check all mask directories
        mask_dirs = [
            self.data_root / "train" / "masks",
            self.data_root / "val" / "masks", 
            self.data_root / "test" / "masks"
        ]
        
        for mask_dir in mask_dirs:
            if not mask_dir.exists():
                continue
                
            print(f"  Analyzing: {mask_dir}")
            
            for mask_file in mask_dir.glob("*.png"):
                mask = cv2.imread(str(mask_file))
                if mask is None:
                    continue
                
                # Get unique colors/classes
                unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
                
                for color in unique_colors:
                    color_tuple = tuple(color)
                    all_classes.add(color_tuple)
                    class_counts[color_tuple] += 1
                    
                    # Check if this is a slum color
                    if self.is_slum_color(color_tuple):
                        slum_patterns.append({
                            'file': mask_file.name,
                            'color': color_tuple,
                            'pixels': np.sum(np.all(mask == color, axis=2))
                        })
        
        self.analysis_results['classes'] = {
            'all_classes': list(all_classes),
            'class_counts': dict(class_counts),
            'slum_patterns': slum_patterns,
            'total_classes': len(all_classes)
        }
        
        print(f"  Found {len(all_classes)} unique classes")
        print(f"  Identified {len(slum_patterns)} slum patterns")
        
    def is_slum_color(self, color):
        """Identify if a color represents slum areas."""
        # Common slum colors in datasets
        slum_colors = [
            (250, 235, 185),  # Light beige
            (255, 255, 0),    # Yellow
            (255, 165, 0),    # Orange
            (139, 69, 19),    # Brown
            (255, 192, 203),  # Pink
            (255, 255, 255),  # White (sometimes)
        ]
        
        # Check if color is close to known slum colors
        for slum_color in slum_colors:
            if np.linalg.norm(np.array(color) - np.array(slum_color)) < 30:
                return True
        return False
    
    def analyze_slum_patterns(self):
        """Analyze spatial patterns of slum areas."""
        print("\n🏘️ Analyzing slum spatial patterns...")
        
        slum_characteristics = {
            'density_patterns': [],
            'size_distributions': [],
            'shape_characteristics': [],
            'texture_features': []
        }
        
        # Analyze train masks for slum patterns
        train_masks = self.data_root / "train" / "masks"
        train_images = self.data_root / "train" / "images"
        
        if train_masks.exists() and train_images.exists():
            mask_files = list(train_masks.glob("*.png"))[:50]  # Sample for analysis
            
            for mask_file in mask_files:
                image_file = train_images / mask_file.name.replace('.png', '.jpg')
                if not image_file.exists():
                    image_file = train_images / mask_file.name
                
                if image_file.exists():
                    self.analyze_single_slum_pattern(image_file, mask_file, slum_characteristics)
        
        self.analysis_results['slum_patterns'] = slum_characteristics
        print(f"  Analyzed {len(slum_characteristics['density_patterns'])} slum patterns")
    
    def analyze_single_slum_pattern(self, image_file, mask_file, characteristics):
        """Analyze slum patterns in a single image-mask pair."""
        image = cv2.imread(str(image_file))
        mask = cv2.imread(str(mask_file))
        
        if image is None or mask is None:
            return
        
        # Convert to grayscale for analysis
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Find slum areas (non-zero pixels)
        slum_areas = gray_mask > 50  # Threshold for slum detection
        
        if np.sum(slum_areas) > 100:  # Minimum slum area
            # Density analysis
            density = np.sum(slum_areas) / slum_areas.size
            characteristics['density_patterns'].append(density)
            
            # Size analysis
            contours, _ = cv2.findContours(slum_areas.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sizes = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
            characteristics['size_distributions'].extend(sizes)
            
            # Shape analysis
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    # Calculate shape characteristics
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
                        characteristics['shape_characteristics'].append(circularity)
            
            # Texture analysis on slum areas
            slum_region = image[slum_areas]
            if len(slum_region) > 0:
                texture_variance = np.var(slum_region)
                characteristics['texture_features'].append(texture_variance)
    
    def analyze_image_characteristics(self):
        """Analyze image characteristics for better preprocessing."""
        print("\n📷 Analyzing image characteristics...")
        
        image_stats = {
            'resolutions': [],
            'color_distributions': [],
            'brightness_levels': [],
            'contrast_levels': []
        }
        
        # Analyze train images
        train_images = self.data_root / "train" / "images"
        if train_images.exists():
            image_files = list(train_images.glob("*.jpg"))[:30]  # Sample
            
            for image_file in image_files:
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # Resolution
                h, w = image.shape[:2]
                image_stats['resolutions'].append((h, w))
                
                # Color distribution
                mean_colors = np.mean(image, axis=(0, 1))
                image_stats['color_distributions'].append(mean_colors.tolist())
                
                # Brightness
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                image_stats['brightness_levels'].append(brightness)
                
                # Contrast
                contrast = np.std(gray)
                image_stats['contrast_levels'].append(contrast)
        
        self.analysis_results['image_characteristics'] = image_stats
        print(f"  Analyzed {len(image_stats['resolutions'])} images")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        print("\n📋 Generating analysis report...")
        
        report = {
            'dataset_summary': {
                'total_classes': self.analysis_results.get('classes', {}).get('total_classes', 0),
                'slum_patterns_found': len(self.analysis_results.get('slum_patterns', {}).get('density_patterns', [])),
                'images_analyzed': len(self.analysis_results.get('image_characteristics', {}).get('resolutions', []))
            },
            'slum_characteristics': self.get_slum_summary(),
            'recommended_preprocessing': self.get_preprocessing_recommendations(),
            'model_recommendations': self.get_model_recommendations()
        }
        
        # Save detailed analysis
        with open('dataset_analysis_complete.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save summary report
        with open('slum_detection_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.print_analysis_summary(report)
        
        return report
    
    def get_slum_summary(self):
        """Get summary of slum characteristics."""
        slum_patterns = self.analysis_results.get('slum_patterns', {})
        
        if not slum_patterns.get('density_patterns'):
            return "No slum patterns detected"
        
        return {
            'average_density': np.mean(slum_patterns['density_patterns']),
            'density_range': [np.min(slum_patterns['density_patterns']), np.max(slum_patterns['density_patterns'])],
            'average_size': np.mean(slum_patterns['size_distributions']) if slum_patterns['size_distributions'] else 0,
            'shape_complexity': np.mean(slum_patterns['shape_characteristics']) if slum_patterns['shape_characteristics'] else 0
        }
    
    def get_preprocessing_recommendations(self):
        """Get preprocessing recommendations based on analysis."""
        image_chars = self.analysis_results.get('image_characteristics', {})
        
        recommendations = []
        
        if image_chars.get('brightness_levels'):
            avg_brightness = np.mean(image_chars['brightness_levels'])
            if avg_brightness < 100:
                recommendations.append("Apply brightness enhancement")
            elif avg_brightness > 200:
                recommendations.append("Apply brightness reduction")
        
        if image_chars.get('contrast_levels'):
            avg_contrast = np.mean(image_chars['contrast_levels'])
            if avg_contrast < 30:
                recommendations.append("Apply contrast enhancement")
        
        recommendations.extend([
            "Use heavy data augmentation",
            "Apply edge enhancement for slum boundaries",
            "Use multi-scale training"
        ])
        
        return recommendations
    
    def get_model_recommendations(self):
        """Get model architecture recommendations."""
        return [
            "Use ensemble of UNet, UNet++, and DeepLabV3+",
            "Implement attention mechanisms for slum features",
            "Use focal loss for class imbalance",
            "Apply test-time augmentation",
            "Use multi-scale inference"
        ]
    
    def print_analysis_summary(self, report):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("🎯 SLUM DETECTION DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        summary = report['dataset_summary']
        print(f"📊 Total Classes Found: {summary['total_classes']}")
        print(f"🏘️ Slum Patterns Detected: {summary['slum_patterns_found']}")
        print(f"📷 Images Analyzed: {summary['images_analyzed']}")
        
        print(f"\n🏗️ Model Recommendations:")
        for rec in report['model_recommendations']:
            print(f"  • {rec}")
        
        print(f"\n⚙️ Preprocessing Recommendations:")
        for rec in report['recommended_preprocessing']:
            print(f"  • {rec}")
        
        print("\n✅ Analysis complete! Ready for model training.")

if __name__ == "__main__":
    analyzer = SlumDatasetAnalyzer()
    results = analyzer.analyze_complete_dataset()