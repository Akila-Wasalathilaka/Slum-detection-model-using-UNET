#!/usr/bin/env python3
"""
Colombo Analysis Pipeline
========================

Complete end-to-end pipeline for processing Colombo satellite images 
and detecting slum areas using the trained UNet model.
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from process_colombo_image import ColomboImageProcessor
from detect_slums_fixed import ColomboSlumDetector
from visualize_results import ColomboResultsVisualizer


def run_complete_pipeline(
    image_path,
    checkpoint_path,
    output_dir="colombo",
    tile_size=120,
    overlap=0,
    threshold=0.5
):
    """
    Run the complete Colombo slum detection pipeline.
    
    Args:
        image_path: Path to input satellite image
        checkpoint_path: Path to trained model checkpoint
        output_dir: Base output directory
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles
        threshold: Detection threshold
    
    Returns:
        Dictionary with all results
    """
    print(f"\n🚀 COLOMBO SLUM DETECTION PIPELINE")
    print("=" * 60)
    print(f"Input image: {image_path}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Detection threshold: {threshold}")
    print()
    
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'pipeline_start': datetime.now().isoformat(),
        'input_image': str(image_path),
        'checkpoint': str(checkpoint_path),
        'parameters': {
            'tile_size': tile_size,
            'overlap': overlap,
            'threshold': threshold
        }
    }
    
    try:
        # Step 1: Process satellite image
        print("🔄 STEP 1: Processing satellite image...")
        processor = ColomboImageProcessor(
            output_dir=output_dir,
            tile_size=(tile_size, tile_size),
            overlap=overlap
        )
        
        processing_results = processor.process_image(image_path, "colombo_tile")
        results['processing'] = processing_results
        print("   ✅ Image processing completed")
        
        # Step 2: Detect slums
        print("\n🔄 STEP 2: Detecting slums...")
        detector = ColomboSlumDetector(
            checkpoint_path=checkpoint_path,
            output_dir=str(base_dir / "predictions")
        )
        
        tiles_dir = base_dir / "tiles"
        metadata_file = base_dir / "metadata" / "colombo_tile_metadata.json"
        
        detection_results = detector.detect_slums_fixed(
            tiles_dir=str(tiles_dir),
            threshold=threshold
        )
        results['detection'] = detection_results
        print("   ✅ Slum detection completed")
        
        # Step 3: Create visualizations
        print("\n🔄 STEP 3: Creating visualizations...")
        visualizer = ColomboResultsVisualizer(
            output_dir=str(base_dir / "visualizations")
        )
        
        results_file = detection_results['results_file']
        visualization_files = visualizer.create_all_visualizations(results_file)
        results['visualizations'] = visualization_files
        print("   ✅ Visualizations completed")
        
        # Step 4: Generate final report
        print("\n🔄 STEP 4: Generating final report...")
        report_data = generate_final_report(results, detection_results)
        
        report_file = base_dir / "colombo_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        results['report_file'] = str(report_file)
        results['pipeline_end'] = datetime.now().isoformat()
        
        print("   ✅ Final report generated")
        
        # Print summary
        print_pipeline_summary(detection_results['analysis'], base_dir)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        results['error'] = str(e)
        results['pipeline_end'] = datetime.now().isoformat()
        return results


def generate_final_report(pipeline_results, detection_results):
    """Generate a comprehensive final report."""
    analysis = detection_results['analysis']
    
    report = {
        'executive_summary': {
            'total_area_analyzed': f"{analysis['total_tiles']} tiles (120x120px each)",
            'slum_areas_detected': f"{analysis['slum_tiles']} tiles ({analysis['slum_percentage']:.1f}%)",
            'detection_confidence': f"Average: {analysis['avg_probability']:.3f}",
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'detailed_findings': {
            'spatial_coverage': analysis,
            'confidence_breakdown': analysis['confidence_distribution'],
            'probability_statistics': {
                'mean': analysis['avg_probability'],
                'range': [analysis['min_probability'], analysis['max_probability']],
                'threshold_used': analysis['threshold_used']
            }
        },
        'methodology': {
            'model_type': 'UNet with ResNet34 encoder',
            'input_preprocessing': '120x120 pixel tiles, RGB normalization',
            'detection_threshold': analysis['threshold_used'],
            'confidence_levels': 'High (>0.8), Medium (0.2-0.8), Low (<0.2)'
        },
        'file_locations': {
            'processed_tiles': pipeline_results.get('processing', {}).get('tile_files', []),
            'detection_results': detection_results.get('results_file'),
            'visualizations': pipeline_results.get('visualizations', [])
        },
        'recommendations': generate_recommendations(analysis)
    }
    
    return report


def generate_recommendations(analysis):
    """Generate recommendations based on analysis results."""
    recommendations = []
    
    slum_percentage = analysis['slum_percentage']
    avg_probability = analysis['avg_probability']
    
    if slum_percentage > 20:
        recommendations.append("High slum concentration detected. Immediate urban planning intervention recommended.")
    elif slum_percentage > 10:
        recommendations.append("Moderate slum presence. Consider targeted development programs.")
    elif slum_percentage > 5:
        recommendations.append("Low to moderate slum presence. Monitor for potential growth.")
    else:
        recommendations.append("Low slum presence detected. Continue monitoring for early detection.")
    
    if avg_probability < 0.3:
        recommendations.append("Low average detection confidence. Consider high-resolution imagery for better accuracy.")
    elif avg_probability > 0.7:
        recommendations.append("High confidence in detections. Results are reliable for planning purposes.")
    
    high_conf = analysis['confidence_distribution']['high']
    total_tiles = analysis['total_tiles']
    
    if high_conf / total_tiles > 0.5:
        recommendations.append("High confidence in most detections. Results suitable for decision-making.")
    else:
        recommendations.append("Mixed confidence levels. Consider field verification for important decisions.")
    
    return recommendations


def print_pipeline_summary(analysis, output_dir):
    """Print a formatted summary of the pipeline results."""
    print(f"\n🏆 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 ANALYSIS SUMMARY:")
    print(f"   🏘️ Total area analyzed: {analysis['total_tiles']} tiles")
    print(f"   🚨 Slum areas detected: {analysis['slum_tiles']} tiles ({analysis['slum_percentage']:.1f}%)")
    print(f"   🎯 Average confidence: {analysis['avg_probability']:.3f}")
    print(f"   📈 Confidence distribution:")
    print(f"      High: {analysis['confidence_distribution']['high']} tiles")
    print(f"      Medium: {analysis['confidence_distribution']['medium']} tiles")
    print(f"      Low: {analysis['confidence_distribution']['low']} tiles")
    print()
    print(f"📁 OUTPUTS SAVED TO: {output_dir}")
    print(f"   📂 tiles/ - Processed image tiles")
    print(f"   📂 predictions/ - Detection results and analysis")
    print(f"   📂 visualizations/ - Charts and visualizations")
    print(f"   📂 metadata/ - Processing metadata")
    print(f"   📄 colombo_analysis_report.json - Final comprehensive report")
    print("=" * 60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Complete Colombo slum detection pipeline')
    parser.add_argument('--image', required=True, help='Path to input satellite image')
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', default='colombo', help='Output directory')
    parser.add_argument('--tile-size', type=int, default=120, help='Tile size for processing')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between tiles')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.image).exists():
        print(f"❌ Input image not found: {args.image}")
        sys.exit(1)
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Model checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run pipeline
    try:
        results = run_complete_pipeline(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            threshold=args.threshold
        )
        
        if 'error' not in results:
            print(f"\n🎉 Colombo slum detection pipeline completed successfully!")
            
            # Show quick stats
            detection_results = results.get('detection', {})
            if detection_results:
                analysis = detection_results.get('analysis', {})
                print(f"🏘️ Found {analysis.get('slum_tiles', 0)} potential slum areas")
                print(f"📊 Coverage: {analysis.get('slum_percentage', 0):.1f}% of analyzed area")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
