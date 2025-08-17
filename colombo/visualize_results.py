#!/usr/bin/env python3
"""
Colombo Results Visualization Script
===================================

Creates comprehensive visualizations and analysis of slum detection results
for Colombo satellite imagery.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ColomboResultsVisualizer:
    """Create comprehensive visualizations of Colombo slum detection results."""
    
    def __init__(self, output_dir="colombo/visualizations"):
        """Initialize the visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Colombo Results Visualizer initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def load_results(self, results_file):
        """Load detection results from JSON file."""
        print(f"📂 Loading results from: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        predictions = results.get('predictions', [])
        analysis = results.get('analysis', {})
        metadata = results.get('metadata', {})
        
        print(f"   ✅ Loaded {len(predictions)} predictions")
        return predictions, analysis, metadata
    
    def create_spatial_heatmap(self, predictions, metadata):
        """Create a spatial heatmap of slum probabilities."""
        print("🗺️ Creating spatial heatmap...")
        
        if not metadata.get('tiles'):
            print("   ⚠️ No spatial metadata available, skipping heatmap")
            return None
        
        # Extract spatial information
        tiles_metadata = metadata['tiles']
        
        # Create coordinate arrays
        coords = []
        probabilities = []
        
        for pred in predictions:
            tile_name = Path(pred['tile_file']).stem
            
            # Find matching metadata
            for tile_meta in tiles_metadata:
                if f"colombo_tile_{tile_meta['tile_id']:04d}" in tile_name:
                    coords.append([tile_meta['x_start'], tile_meta['y_start']])
                    probabilities.append(pred['probability'])
                    break
        
        if not coords:
            print("   ⚠️ Could not match predictions with spatial metadata")
            return None
        
        coords = np.array(coords)
        probabilities = np.array(probabilities)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 12))
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], 
            c=probabilities, 
            cmap='RdYlBu_r',
            s=50,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Customize plot
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        ax.set_title('Spatial Distribution of Slum Probabilities - Colombo', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Slum Probability', fontsize=12, fontweight='bold')
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save
        heatmap_path = self.output_dir / "spatial_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Spatial heatmap saved: {heatmap_path}")
        return str(heatmap_path)
    
    def create_probability_analysis(self, predictions):
        """Create comprehensive probability analysis charts."""
        print("📊 Creating probability analysis...")
        
        probabilities = [p['probability'] for p in predictions]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Colombo Slum Detection - Probability Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Histogram
        axes[0, 0].hist(probabilities, bins=30, alpha=0.7, color='skyblue', 
                       edgecolor='black', density=True)
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                          label='Decision Threshold')
        axes[0, 0].set_xlabel('Slum Probability')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Probability Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot
        confidence_groups = {
            'High (>0.7)': [p for p in probabilities if p > 0.7],
            'Medium (0.3-0.7)': [p for p in probabilities if 0.3 <= p <= 0.7],
            'Low (<0.3)': [p for p in probabilities if p < 0.3]
        }
        
        box_data = [group for group in confidence_groups.values() if group]
        box_labels = [label for label, group in confidence_groups.items() if group]
        
        axes[0, 1].boxplot(box_data, labels=box_labels)
        axes[0, 1].set_ylabel('Slum Probability')
        axes[0, 1].set_title('Probability by Confidence Level')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_probs = np.sort(probabilities)
        cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        
        axes[1, 0].plot(sorted_probs, cumulative, linewidth=2, color='purple')
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                          label='Decision Threshold')
        axes[1, 0].set_xlabel('Slum Probability')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Decision statistics
        threshold_stats = []
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        for thresh in thresholds:
            slum_count = sum(1 for p in probabilities if p >= thresh)
            slum_percentage = (slum_count / len(probabilities)) * 100
            threshold_stats.append(slum_percentage)
        
        axes[1, 1].plot(thresholds, threshold_stats, 'o-', linewidth=2, 
                       markersize=6, color='orange')
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                          label='Default Threshold')
        axes[1, 1].set_xlabel('Decision Threshold')
        axes[1, 1].set_ylabel('Slum Area Percentage (%)')
        axes[1, 1].set_title('Slum Area vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_path = self.output_dir / "probability_analysis.png"
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Probability analysis saved: {analysis_path}")
        return str(analysis_path)
    
    def create_confidence_analysis(self, predictions):
        """Create confidence-based analysis."""
        print("🎯 Creating confidence analysis...")
        
        # Group by confidence
        confidence_groups = {
            'High': [p for p in predictions if p['confidence'] == 'high'],
            'Medium': [p for p in predictions if p['confidence'] == 'medium'],
            'Low': [p for p in predictions if p['confidence'] == 'low']
        }
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Confidence Analysis - Colombo Slum Detection', 
                     fontsize=16, fontweight='bold')
        
        # 1. Confidence distribution
        conf_counts = [len(group) for group in confidence_groups.values()]
        conf_labels = list(confidence_groups.keys())
        colors = ['green', 'orange', 'red']
        
        axes[0].pie(conf_counts, labels=conf_labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=(0.05, 0.05, 0.05))
        axes[0].set_title('Confidence Distribution')
        
        # 2. Slum detection by confidence
        slum_by_conf = {}
        for conf, group in confidence_groups.items():
            slum_count = sum(1 for p in group if p['prediction'] == 'slum')
            total_count = len(group)
            slum_percentage = (slum_count / total_count * 100) if total_count > 0 else 0
            slum_by_conf[conf] = slum_percentage
        
        bars = axes[1].bar(slum_by_conf.keys(), slum_by_conf.values(), 
                          color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Slum Detection Rate (%)')
        axes[1].set_title('Slum Detection Rate by Confidence')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, slum_by_conf.values()):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Probability ranges by confidence
        conf_probs = {conf: [p['probability'] for p in group] 
                     for conf, group in confidence_groups.items() if group}
        
        box_data = [probs for probs in conf_probs.values()]
        box_labels = list(conf_probs.keys())
        
        bp = axes[2].boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[2].set_ylabel('Slum Probability')
        axes[2].set_title('Probability Distribution by Confidence')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        confidence_path = self.output_dir / "confidence_analysis.png"
        plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Confidence analysis saved: {confidence_path}")
        return str(confidence_path)
    
    def create_detailed_summary(self, predictions, analysis):
        """Create a detailed summary visualization."""
        print("📋 Creating detailed summary...")
        
        # Create summary figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Colombo Slum Detection - Comprehensive Summary', 
                     fontsize=18, fontweight='bold')
        
        # 1. Overall statistics
        stats_data = [
            analysis['slum_tiles'],
            analysis['non_slum_tiles']
        ]
        labels = ['Slum Areas', 'Non-Slum Areas']
        colors = ['red', 'green']
        
        wedges, texts, autotexts = axes[0, 0].pie(stats_data, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Classification')
        
        # 2. Probability statistics
        prob_stats = [
            f"Total Tiles: {analysis['total_tiles']}",
            f"Slum Areas: {analysis['slum_tiles']} ({analysis['slum_percentage']:.1f}%)",
            f"Avg Probability: {analysis['avg_probability']:.3f}",
            f"Max Probability: {analysis['max_probability']:.3f}",
            f"Min Probability: {analysis['min_probability']:.3f}"
        ]
        
        axes[0, 1].text(0.1, 0.9, '\n'.join(prob_stats), transform=axes[0, 1].transAxes,
                       fontsize=12, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[0, 1].set_title('Statistics Summary')
        axes[0, 1].axis('off')
        
        # 3. Confidence distribution
        conf_dist = analysis['confidence_distribution']
        conf_counts = [conf_dist['high'], conf_dist['medium'], conf_dist['low']]
        conf_labels = ['High', 'Medium', 'Low']
        conf_colors = ['green', 'orange', 'red']
        
        bars = axes[0, 2].bar(conf_labels, conf_counts, color=conf_colors, 
                             alpha=0.7, edgecolor='black')
        axes[0, 2].set_ylabel('Number of Tiles')
        axes[0, 2].set_title('Confidence Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Probability histogram
        probabilities = [p['probability'] for p in predictions]
        axes[1, 0].hist(probabilities, bins=25, alpha=0.7, color='skyblue', 
                       edgecolor='black')
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                          label='Threshold')
        axes[1, 0].set_xlabel('Slum Probability')
        axes[1, 0].set_ylabel('Number of Tiles')
        axes[1, 0].set_title('Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Top slum areas
        high_prob_tiles = sorted([p for p in predictions if p['probability'] >= 0.7], 
                                key=lambda x: x['probability'], reverse=True)[:10]
        
        if high_prob_tiles:
            tile_names = [Path(p['tile_file']).stem[-4:] for p in high_prob_tiles]
            tile_probs = [p['probability'] for p in high_prob_tiles]
            
            bars = axes[1, 1].barh(range(len(tile_names)), tile_probs, 
                                  color='red', alpha=0.7)
            axes[1, 1].set_yticks(range(len(tile_names)))
            axes[1, 1].set_yticklabels(tile_names)
            axes[1, 1].set_xlabel('Probability')
            axes[1, 1].set_title('Top 10 Potential Slum Areas')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No high-probability\nslum areas found', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 1].set_title('Top Potential Slum Areas')
            axes[1, 1].axis('off')
        
        # 6. Detection timeline
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timeline_text = [
            f"Detection completed: {current_time}",
            f"Model threshold: {analysis['threshold_used']}",
            f"Processing summary:",
            f"• {analysis['total_tiles']} tiles analyzed",
            f"• {analysis['slum_tiles']} slum areas detected",
            f"• {analysis['slum_percentage']:.1f}% slum coverage"
        ]
        
        axes[1, 2].text(0.1, 0.9, '\n'.join(timeline_text), 
                       transform=axes[1, 2].transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 2].set_title('Detection Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        summary_path = self.output_dir / "comprehensive_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comprehensive summary saved: {summary_path}")
        return str(summary_path)
    
    def create_all_visualizations(self, results_file):
        """Create all visualizations for the results."""
        print(f"\n🎨 CREATING COLOMBO VISUALIZATIONS")
        print("=" * 50)
        
        # Load results
        predictions, analysis, metadata = self.load_results(results_file)
        
        created_files = []
        
        # Create spatial heatmap
        heatmap_path = self.create_spatial_heatmap(predictions, metadata)
        if heatmap_path:
            created_files.append(heatmap_path)
        
        # Create probability analysis
        prob_path = self.create_probability_analysis(predictions)
        created_files.append(prob_path)
        
        # Create confidence analysis
        conf_path = self.create_confidence_analysis(predictions)
        created_files.append(conf_path)
        
        # Create detailed summary
        summary_path = self.create_detailed_summary(predictions, analysis)
        created_files.append(summary_path)
        
        print(f"\n✅ VISUALIZATIONS COMPLETE!")
        print("=" * 50)
        print(f"📊 Created {len(created_files)} visualizations:")
        for file_path in created_files:
            print(f"   📈 {Path(file_path).name}")
        print(f"📁 All files saved to: {self.output_dir}")
        
        return created_files


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Create visualizations for Colombo slum detection results')
    parser.add_argument('--results', required=True, help='Path to detection results JSON file')
    parser.add_argument('--output', default='colombo/visualizations', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ColomboResultsVisualizer(output_dir=args.output)
    
    # Create all visualizations
    try:
        created_files = visualizer.create_all_visualizations(args.results)
        print(f"\n🎉 Successfully created {len(created_files)} visualizations!")
        
    except Exception as e:
        print(f"\n❌ Visualization creation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
