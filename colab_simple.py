# Clone and setup
!git clone https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git
%cd Slum-detection-model-using-UNET
!pip install -r requirements.txt

# Train
!python scripts/train_global.py --data_root data --batch_size 8 --epochs 50 --lr 1e-4