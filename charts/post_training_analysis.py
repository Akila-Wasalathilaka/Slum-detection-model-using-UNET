"""
Post-Training Analysis Pipeline
==============================

Automatically runs comprehensive analysis after model training completes.
This script should be called from training scripts or run separately.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from charts.model_analysis import ModelAnalyzer
from charts.quick_analysis import quick_model_analysis
from config import get_model_config, get_data_config


def run_post_training_analysis(
    checkpoint_path,
    analysis_type="quick",
    output_dir="charts",
    model_config_name="balanced",
    data_config_name="standard"
):
    """
    Run post-training analysis pipeline.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        analysis_type: "quick" or "comprehensive" 
        output_dir: Directory to save analysis outputs
        model_config_name: Model configuration preset name
        data_config_name: Data configuration preset name
    
    Returns:
        Dictionary with analysis results
    """
    
    print("🎯 POST-TRAINING ANALYSIS PIPELINE")
    print("=" * 40)
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📊 Analysis Type: {analysis_type}")
    print(f"💾 Output Directory: {output_dir}")
    print(f"⚙️  Model Config: {model_config_name}")
    print(f"📋 Data Config: {data_config_name}")
    
    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = Path(output_dir) / f"analysis_{timestamp}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Analysis will be saved to: {analysis_dir}")
    
    try:
        if analysis_type.lower() == "quick":
            print("\n🚀 Running QUICK analysis...")
            results = quick_model_analysis(
                checkpoint_path=checkpoint_path,
                output_dir=str(analysis_dir),
                show_plots=False
            )
            
            # Create summary
            summary = {
                'analysis_type': 'quick',
                'timestamp': timestamp,
                'checkpoint': str(checkpoint_path),
                'results': results
            }
            
        elif analysis_type.lower() == "comprehensive":
            print("\n🚀 Running COMPREHENSIVE analysis...")
            
            # Load configurations
            model_config = get_model_config(model_config_name)
            data_config = get_data_config(data_config_name)
            
            # Run comprehensive analysis
            analyzer = ModelAnalyzer(str(analysis_dir))
            results = analyzer.run_complete_analysis(
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                data_config=data_config,
                device='auto'
            )
            
            summary = {
                'analysis_type': 'comprehensive',
                'timestamp': timestamp,
                'checkpoint': str(checkpoint_path),
                'model_config': model_config_name,
                'data_config': data_config_name,
                'results': results
            }
            
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Save summary
        with open(analysis_dir / "analysis_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create a simple report
        create_simple_report(summary, analysis_dir)
        
        print(f"\n✅ Analysis complete! Results saved to: {analysis_dir}")
        return summary
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        # Save error info
        error_info = {
            'error': str(e),
            'timestamp': timestamp,
            'checkpoint': str(checkpoint_path),
            'analysis_type': analysis_type
        }
        with open(analysis_dir / "analysis_error.json", 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2)
        raise


def create_simple_report(summary, output_dir):
    """Create a simple text report of the analysis."""
    
    timestamp = summary['timestamp']
    analysis_type = summary['analysis_type']
    
    report = f"""
SLUM DETECTION MODEL - ANALYSIS REPORT
=====================================

Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Analysis Type: {analysis_type.upper()}
Checkpoint: {summary['checkpoint']}

"""
    
    if analysis_type == 'quick':
        results = summary['results']
        report += f"""
PERFORMANCE METRICS
==================
AUC-ROC: {results['AUC_ROC']:.4f}
Optimal Threshold: {results['Optimal_Threshold']:.3f}
Accuracy: {results['Accuracy']:.4f}
Precision: {results['Precision']:.4f}
Recall: {results['Recall']:.4f}
F1-Score: {results['F1_Score']:.4f}
Specificity: {results['Specificity']:.4f}

CONFUSION MATRIX
===============
True Positives: {results['True_Positives']}
True Negatives: {results['True_Negatives']}
False Positives: {results['False_Positives']}
False Negatives: {results['False_Negatives']}

CHARTS GENERATED
===============
✅ ROC Curve
✅ Confusion Matrix
✅ Performance Metrics
✅ Precision-Recall Curve
"""
    
    elif analysis_type == 'comprehensive':
        results = summary['results']
        report += f"""
MODEL CONFIGURATION
==================
Architecture: {results['model_info']['architecture']}
Encoder: {results['model_info']['encoder']}
Training Epoch: {results['model_info']['epoch']}

DATASET INFORMATION
==================
Test Samples: {results['test_dataset']['num_samples']}
Positive Class Ratio: {results['test_dataset']['positive_ratio']:.3f}

PERFORMANCE METRICS
==================
AUC-ROC: {results['roc_analysis']['auc']:.4f}
Average Precision: {results['pr_analysis']['average_precision']:.4f}
Optimal Threshold: {results['threshold_analysis']['best_threshold']:.3f}
Best F1-Score: {results['threshold_analysis']['best_f1_score']:.4f}

CHARTS GENERATED
===============
✅ Confusion Matrices (multiple thresholds)
✅ ROC Curve with optimal threshold
✅ Precision-Recall Curve
✅ Threshold Analysis
✅ Performance Summary
✅ Classification Report
✅ Prediction Samples
"""
    
    report += f"""

ANALYSIS LOCATION
================
All charts and detailed results are available in:
{output_dir}

Analysis completed successfully!
"""
    
    with open(output_dir / "ANALYSIS_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(report)


def auto_find_latest_checkpoint(experiments_dir="experiments"):
    """Automatically find the latest checkpoint in experiments directory."""
    experiments_path = Path(experiments_dir)
    
    if not experiments_path.exists():
        return None
    
    # Look for checkpoint files
    checkpoint_files = []
    for exp_dir in experiments_path.iterdir():
        if exp_dir.is_dir():
            checkpoints_dir = exp_dir / "checkpoints"
            if checkpoints_dir.exists():
                for ckpt_file in checkpoints_dir.glob("*.pth"):
                    checkpoint_files.append(ckpt_file)
    
    if not checkpoint_files:
        return None
    
    # Return the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return str(latest_checkpoint)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Post-Training Analysis Pipeline')
    
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint (auto-detected if not provided)')
    parser.add_argument('--analysis-type', choices=['quick', 'comprehensive'], 
                       default='quick', help='Type of analysis to run')
    parser.add_argument('--output-dir', default='charts',
                       help='Output directory for analysis results')
    parser.add_argument('--model-config', default='balanced',
                       help='Model configuration preset')
    parser.add_argument('--data-config', default='standard',
                       help='Data configuration preset')
    parser.add_argument('--auto-find', action='store_true',
                       help='Automatically find latest checkpoint')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.auto_find or args.checkpoint is None:
        print("🔍 Auto-detecting latest checkpoint...")
        checkpoint_path = auto_find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoint found automatically. Please specify --checkpoint")
            return
        print(f"✅ Found checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
    
    # Run analysis
    try:
        results = run_post_training_analysis(
            checkpoint_path=checkpoint_path,
            analysis_type=args.analysis_type,
            output_dir=args.output_dir,
            model_config_name=args.model_config,
            data_config_name=args.data_config
        )
        
        print("\n🎉 SUCCESS: Post-training analysis completed!")
        
        # Print key metrics if quick analysis
        if args.analysis_type == 'quick' and 'results' in results:
            metrics = results['results']
            print(f"\n📊 KEY METRICS:")
            print(f"   AUC-ROC: {metrics['AUC_ROC']:.4f}")
            print(f"   F1-Score: {metrics['F1_Score']:.4f}")
            print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        
    except Exception as e:
        print(f"\n💥 FAILED: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
