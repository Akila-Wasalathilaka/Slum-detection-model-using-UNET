# KAGGLE TRAINING PIPELINE
# Copy this into Kaggle notebook

# Clone repo and run training
!git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
%cd Slum-detection-model-using-UNET/SlumSeg

# Update dataset path in the pipeline file before running
!sed -i "s|/kaggle/input/slum-detection-dataset|/kaggle/input/YOUR-DATASET-NAME|g" kaggle_train_only.py
!python kaggle_train_only.py