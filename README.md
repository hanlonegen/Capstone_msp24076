# Installation Guide

## 1. Recommended Installation Methods (choose one)

### Pip Installation (Recommended)

Suitable for most users, supporting stable or development versions:

**Stable version (PyPI)**:

```bash
pip install ultralytics
```

**Development version (latest GitHub code)**:

```bash
pip install git+https://github.com/ultralytics/ultralytics.git@main
```

### Conda Installation

  Ideal for users working with Conda environments:

**Basic installation**:

```bash
conda install -c conda-forge ultralytics
```

**CUDA environments (install with PyTorch components)**:

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

### Git Clone (For Development)

  Suitable for modifying source code or contributing to development:

```bash
# Clone the repository
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
# Install in editable mode (no need to reinstall after code modifications)
pip install -e .
```

### Docker Installation (Isolated Environment)

  Provides multiple images (GPU/CPU/ARM64, etc.) for consistent environments:

```bash
# Pull the latest image
t=ultralytics/ultralytics:latest
sudo docker pull $t
# Run the container (with GPU support)
sudo docker run -it --ipc=host --gpus all $t # All GPUs
sudo docker run -it --ipc=host --gpus '"device=2,3"' $t # Specific GPUs
# Mount local directory to the container
sudo docker run -it --ipc=host --gpus all -v /local/path:/container/path $t
```

## 2. Dependency Notes

Ultralytics depends on PyTorch. It is recommended to install PyTorch first according to your system and CUDA requirements. Refer to the [official PyTorch guide](https://pytorch.org/get-started/locally/).

You should also download dataset and some pretrained model weight files.

You can input the following commands in you terminal.

```bash
git lfs install

git clone https://huggingface.co/datasets/hanlone/Capstone_msp24076
```

Or access  [hanlone/Capstone_msp24076 · Datasets at Hugging Face](https://huggingface.co/datasets/hanlone/Capstone_msp24076) to download dataset and pretrained model weight files.

<img src="file:///D:/文件/图片/markdown/2025-07-18-14-27-16-image.png" title="" alt="" data-align="center">

Replace the empty folders with the same names in the project root directory with the downloaded dataset.

## 3. Basic Usage (After Installation)

Use `cuda_test.py` to verify your installation. You should see the following:

```bash
2.6.0+cu118
True
11.8
0.21.0+cu118
```

as output of:

```python
print(torch.__version__)          # Check PyTorch version
print(torch.cuda.is_available())  # Output True if GPU support is available, False otherwise
print(torch.version.cuda)         # Show CUDA version used by PyTorch (None if CPU-only version)
print(torchvision.__version__)    # Check torchvision version
```

then a YOLO11n model is trained on COCO8 dataset. You should see the similar Result:

```bash
Ultralytics 8.3.107 🚀 Python-3.10.16 torch-2.6.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00, 28.55it/s]
                   all          4         17      0.598      0.683      0.624      0.459
                person          3         10      0.327        0.6      0.469      0.232
                   dog          1          1      0.617          1      0.995      0.796
                 horse          1          2      0.307          1      0.663      0.354
              elephant          1          2      0.797        0.5       0.62      0.374
              umbrella          1          1      0.541          1      0.995      0.995
          potted plant          1          1          1          0          0          0
Speed: 0.1ms preprocess, 2.2ms inference, 0.0ms loss, 0.8ms postprocess per image
Results saved to runs\detect\train

Process finished with exit code 0
```

You can use `synthetic.py`  and original real images under folder `scratches_notprocessed` to generate metal scratch images.

You can use `test_different_model.py` to compare the performance of  `YOLOv5`,`YOLOv8`, `YOLOv10`,  and `YOLO11`.

You can use `scratch_pose.py` to train `YOLOv11-pose` on our original `Scratch-Pose` dataset.

You can use `synthetic_train.py` to train `YOLOv11` on our original synthetic dataset.
