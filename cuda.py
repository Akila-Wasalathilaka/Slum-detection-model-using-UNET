import torch

print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? -> {is_cuda_available}")

if is_cuda_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    gpu_id = torch.cuda.current_device()
    print(f"Current GPU ID: {gpu_id}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(gpu_id)}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
else:
    print("CUDA is not available. PyTorch is running in CPU-only mode.")
    print("This may be due to an incorrect PyTorch installation or an NVIDIA driver issue.")