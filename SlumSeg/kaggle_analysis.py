# KAGGLE ANALYSIS PIPELINE  
# Copy this into Kaggle notebook

# Clone repo and run analysis
!git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
%cd Slum-detection-model-using-UNET/SlumSeg

# Update dataset and model paths in the pipeline file before running
!sed -i "s|/kaggle/input/slum-detection-dataset|/kaggle/input/YOUR-DATASET-NAME|g" kaggle_analysis_pipeline.py
!sed -i "s|/kaggle/input/trained-model|/kaggle/input/YOUR-MODEL-DATASET|g" kaggle_analysis_pipeline.py
!python kaggle_analysis_pipeline.py