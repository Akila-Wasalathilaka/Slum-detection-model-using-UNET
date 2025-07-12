# Charts & Model Analysis

This folder contains comprehensive tools for analyzing trained slum detection models, generating visualizations, and evaluating performance.

## 📊 Available Analysis Tools

### 1. Quick Analysis (`quick_analysis.py`)
Fast post-training evaluation with essential charts:
- **ROC Curve** with AUC score and optimal threshold
- **Confusion Matrix** at optimal threshold
- **Performance Metrics** bar chart (Accuracy, Precision, Recall, F1, Specificity)
- **Precision-Recall Curve**

```bash
python charts/quick_analysis.py --checkpoint path/to/model.pth --output charts/my_analysis
```

### 2. Comprehensive Analysis (`model_analysis.py`)
Detailed analysis with all possible visualizations:
- Multiple confusion matrices at different thresholds
- ROC curve with optimal point marking
- Precision-Recall curve with average precision
- Threshold analysis (metrics vs threshold plots)
- Performance radar chart and summary
- Classification report visualization
- Sample predictions with ground truth comparison

```bash
python charts/model_analysis.py --checkpoint path/to/model.pth --output_dir charts/detailed
```

### 3. Post-Training Pipeline (`post_training_analysis.py`)
Automated analysis pipeline that runs after training:
- Auto-detects latest checkpoint
- Runs either quick or comprehensive analysis
- Creates timestamped output directories
- Generates summary reports

```bash
# Auto-find latest checkpoint and run quick analysis
python charts/post_training_analysis.py --auto-find

# Run comprehensive analysis on specific checkpoint
python charts/post_training_analysis.py --checkpoint model.pth --analysis-type comprehensive

# Specify custom configurations
python charts/post_training_analysis.py --checkpoint model.pth --model-config balanced --data-config standard
```

### 4. Example Analysis (`example_analysis.py`)
Demonstrates how to use the analysis tools:

```bash
python charts/example_analysis.py
```

## 🎯 Integration with Training

The training script (`scripts/train.py`) automatically runs quick analysis after training completes. You can disable this by modifying the training script.

## 📈 Generated Charts

### Quick Analysis Output:
```
charts/
├── quick_roc_curve.png           # ROC curve with AUC
├── quick_confusion_matrix.png    # Confusion matrix
├── quick_performance_metrics.png # Performance bar chart
├── quick_precision_recall.png    # PR curve
└── quick_analysis_metrics.json   # Metrics in JSON format
```

### Comprehensive Analysis Output:
```
charts/
├── confusion_matrices/
│   ├── confusion_matrix_t0.30.png
│   ├── confusion_matrix_t0.50.png
│   └── confusion_matrix_t{optimal}.png
├── roc_curves/
│   └── roc_curve.png
├── precision_recall/
│   └── precision_recall_curve.png
├── threshold_analysis/
│   ├── threshold_analysis.png
│   ├── combined_metrics.png
│   └── threshold_data.csv
├── performance_metrics/
│   ├── performance_summary.png
│   ├── classification_report.png
│   └── classification_report.csv
├── predictions/
│   └── prediction_samples.png
├── complete_analysis_results.json
└── ANALYSIS_SUMMARY_REPORT.txt
```

## 🔧 Configuration

The analysis tools use the same configuration system as the training scripts:

- **Model Config**: Specifies architecture, encoder, etc.
- **Data Config**: Defines data paths, preprocessing, etc.

Available presets:
- Model configs: `balanced`, `high_precision`, `high_recall`
- Data configs: `standard`, `augmented`, `minimal`

## 📊 Key Metrics Explained

### ROC Curve & AUC
- **AUC-ROC**: Area Under the ROC Curve (0.5 = random, 1.0 = perfect)
- **Optimal Threshold**: Point where True Positive Rate - False Positive Rate is maximized

### Precision-Recall
- **Average Precision**: Area under the PR curve
- **Optimal F1 Threshold**: Threshold that maximizes F1-score

### Confusion Matrix Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - How many predicted slums are actually slums
- **Recall (Sensitivity)**: TP / (TP + FN) - How many actual slums are detected
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Specificity**: TN / (TN + FP) - How many non-slums are correctly identified

## 🎨 Customization

### Custom Analysis
You can create custom analysis scripts by importing the `ModelAnalyzer` class:

```python
from charts.model_analysis import ModelAnalyzer
from config import get_model_config, get_data_config

analyzer = ModelAnalyzer("my_charts")
model_config = get_model_config("balanced")
data_config = get_data_config("standard")

results = analyzer.run_complete_analysis(
    checkpoint_path="model.pth",
    model_config=model_config,
    data_config=data_config
)
```

### Custom Thresholds
Modify the threshold analysis ranges in `model_analysis.py`:

```python
# In threshold_analysis method
thresholds = np.arange(0.1, 1.0, 0.05)  # Modify this range
```

### Custom Visualizations
Add new visualization methods to the `ModelAnalyzer` class:

```python
def create_custom_chart(self, data):
    plt.figure(figsize=(10, 6))
    # Your custom visualization code
    plt.savefig(self.charts_dir / "custom_chart.png")
```

## 🚀 Quick Start Examples

### 1. Analyze Latest Trained Model
```bash
# Find and analyze the most recent checkpoint
python charts/post_training_analysis.py --auto-find
```

### 2. Compare Different Thresholds
```bash
# Run comprehensive analysis to see threshold comparison
python charts/post_training_analysis.py --checkpoint model.pth --analysis-type comprehensive
```

### 3. Generate Specific Charts Only
```python
from charts.quick_analysis import quick_model_analysis

# Generate only essential charts
metrics = quick_model_analysis(
    checkpoint_path="model.pth",
    output_dir="my_analysis",
    show_plots=False
)
```

## 📋 Dependencies

All required dependencies are included in the main `requirements.txt`:
- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualizations
- `scikit-learn` - Metrics calculation
- `pandas` - Data manipulation for reports

## 🎯 Best Practices

1. **Always run analysis after training** to validate model performance
2. **Use comprehensive analysis** for final model evaluation
3. **Check threshold analysis** to find optimal operating point
4. **Save analysis results** with timestamps for comparison
5. **Review prediction samples** to understand model behavior

## 🐛 Troubleshooting

### Common Issues:

1. **Memory Error**: Reduce batch size in analysis scripts
2. **CUDA Error**: Set device to 'cpu' in analysis functions
3. **File Not Found**: Check checkpoint path and ensure model was trained
4. **Import Error**: Ensure project root is in Python path

### Debug Mode:
Add debug prints to understand what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- [ROC Analysis](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [Precision-Recall](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
- [Confusion Matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)

---

For more information about the overall project, see the main [README.md](../README.md).
