"""
Example: Run Comprehensive Model Analysis
=========================================

Example script showing how to run detailed analysis on a trained model.
This demonstrates all the analysis capabilities available.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from charts.post_training_analysis import run_post_training_analysis, auto_find_latest_checkpoint


def run_example_analysis():
    """Run example comprehensive analysis."""
    
    print("🔍 EXAMPLE: Comprehensive Model Analysis")
    print("=" * 45)
    
    # Try to find latest checkpoint automatically
    print("🔎 Looking for latest checkpoint...")
    checkpoint_path = auto_find_latest_checkpoint()
    
    if checkpoint_path is None:
        print("❌ No trained model found!")
        print("   Please train a model first using:")
        print("   python scripts/train.py")
        return
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Run quick analysis first
    print("\n🚀 Running QUICK analysis...")
    try:
        quick_results = run_post_training_analysis(
            checkpoint_path=checkpoint_path,
            analysis_type="quick",
            output_dir="charts/example_quick"
        )
        
        print("\n📊 Quick Analysis Results:")
        metrics = quick_results['results']
        print(f"   AUC-ROC: {metrics['AUC_ROC']:.4f}")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   F1-Score: {metrics['F1_Score']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall: {metrics['Recall']:.4f}")
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
        return
    
    # Run comprehensive analysis
    print("\n🚀 Running COMPREHENSIVE analysis...")
    try:
        comprehensive_results = run_post_training_analysis(
            checkpoint_path=checkpoint_path,
            analysis_type="comprehensive",
            output_dir="charts/example_comprehensive"
        )
        
        print("\n📊 Comprehensive Analysis Complete!")
        print("   Generated charts:")
        print("   ✅ Confusion matrices (multiple thresholds)")
        print("   ✅ ROC curve with optimal threshold")
        print("   ✅ Precision-Recall curve")
        print("   ✅ Threshold analysis")
        print("   ✅ Performance summary")
        print("   ✅ Classification report")
        print("   ✅ Prediction samples")
        
        # Extract key results
        results = comprehensive_results['results']
        roc_auc = results['roc_analysis']['auc']
        best_threshold = results['threshold_analysis']['best_threshold']
        best_f1 = results['threshold_analysis']['best_f1_score']
        avg_precision = results['pr_analysis']['average_precision']
        
        print(f"\n🎯 Key Performance Metrics:")
        print(f"   AUC-ROC: {roc_auc:.4f}")
        print(f"   Average Precision: {avg_precision:.4f}")
        print(f"   Optimal Threshold: {best_threshold:.3f}")
        print(f"   Best F1-Score: {best_f1:.4f}")
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        print("   Quick analysis was successful, so you have basic charts available")
        return
    
    print("\n✅ EXAMPLE COMPLETE!")
    print("📁 Check the following directories for results:")
    print("   charts/example_quick/ - Quick analysis charts")
    print("   charts/example_comprehensive/ - Comprehensive analysis charts")
    
    print("\n💡 To run analysis on your own models:")
    print("   python charts/post_training_analysis.py --checkpoint your_checkpoint.pth")
    print("   python charts/post_training_analysis.py --auto-find --analysis-type comprehensive")


if __name__ == "__main__":
    run_example_analysis()
