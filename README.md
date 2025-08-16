# 🏘️ Slum Detection with UNet

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-99.67%25-brightgreen)

> Semantic segmentation model for detecting slums in 120×120 satellite images using UNet + ResNet34. Achieves 99.67% AUC-ROC with advanced loss functions and strong class imbalance handling.

---


---

## 📈 Results Overview

**ROC Curve**  
<img src="images/roc_curve.png" width="500"/>

**Confusion Matrix**  
<img src="images/confusion_matrix.png" width="500"/>

**Performance Summary**  
<img src="images/performance_summary.png" width="500"/>

**Threshold Analysis**  
<img src="images/threshold_analysis.png" width="500"/>

---

## 🖼️ Sample Predictions

<img src="images/prediction_samples.png" width="800"/>

---

## 🔧 Quick Start

```bash
git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
cd Slum-detection-model-using-UNET
pip install -r requirements.txt
python scripts/train.py --model balanced --training development
