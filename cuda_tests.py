import torch
import torchvision
from ultralytics import YOLO

print(torch.__version__)          # Check PyTorch version
print(torch.cuda.is_available())  # Output True if GPU support is available, False otherwise
print(torch.version.cuda)         # Show CUDA version used by PyTorch (None if CPU-only version)
print(torchvision.__version__)    # Check torchvision version

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="./cfg/coco8.yaml", epochs=1, imgsz=400, workers=0, cache=False)


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("hanlone/Capstone_msp24076")