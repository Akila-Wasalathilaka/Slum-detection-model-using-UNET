# KAGGLE TRAINING PIPELINE
# Copy this into Kaggle notebook

# Clone repo and run training
!git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
%cd Slum-detection-model-using-UNET/SlumSeg

# Dataset path already set to /kaggle/input/data
!python kaggle_train_only.py