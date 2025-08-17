"""
Charts and Model Analysis Module
===============================

Comprehensive tools for analyzing trained slum detection models.

This module provides:
- Quick analysis for immediate post-training evaluation
- Comprehensive analysis with detailed visualizations
- Automated post-training analysis pipeline
- Performance metrics and threshold analysis
"""

from .quick_analysis import quick_model_analysis
from .model_analysis import ModelAnalyzer
from .post_training_analysis import run_post_training_analysis, auto_find_latest_checkpoint

__all__ = [
    'quick_model_analysis',
    'ModelAnalyzer', 
    'run_post_training_analysis',
    'auto_find_latest_checkpoint'
]
