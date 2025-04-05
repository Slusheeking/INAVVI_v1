# Installing PyTorch with CUDA Support for GH200 GPU

This document outlines the process of installing PyTorch with CUDA 12.8 support for use with GH200 GPUs.

## Prerequisites

- CUDA 12.8 drivers installed
- Python 3.10 or later
- pip package manager
- A compatible NVIDIA GPU (GH200)
- Virtual environment (recommended)

## Installation Process

### Option 1: Using the Installation Script

We provide a convenience script `install_pytorch_cuda.sh` that handles the installation process automatically:

```bash
./install_pytorch_cuda.sh
```

The script performs the following steps:
1. Creates necessary symlinks for CUDA libraries
2. Removes any existing PyTorch installations
3. Installs PyTorch with CUDA 12.8 support
4. Verifies the installation by running a simple CUDA test

### Option 2: Manual Installation

If you prefer to install manually, you can use the following pip command:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

This command:
- Installs pre-release (nightly) versions of PyTorch packages
- Uses the CUDA 12.8 specific wheels
- Installs the core PyTorch library along with vision and audio extensions

## Verification

To verify your PyTorch installation is working correctly with CUDA, you can run:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

# Simple CUDA test
if torch.cuda.is_available():
    a = torch.cuda.FloatTensor([[1,2],[3,4]])
    b = torch.cuda.FloatTensor([[2,1],[1,2]])
    print("Testing CUDA with PyTorch...")
    print(f"Result: {torch.matmul(a, b)}")
    print("Test successful!")
else:
    print("CUDA is not available!")
```

## Expected Output

When PyTorch is installed correctly with CUDA support, you should see output similar to:

```
PyTorch version: 2.8.0.dev20250402+cu128
CUDA available: True
CUDA version: 12.8
Device count: 1
Testing CUDA with PyTorch...
Result: [[1. 3.]
 [3. 7.]]
Test successful!
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: Ensure CUDA drivers are installed and the GPU is recognized by the system.
2. **Version mismatch**: Make sure the CUDA version in the installation command matches your installed CUDA drivers.
3. **Dependencies missing**: If you encounter dependency errors, install the required packages:
   ```bash
   pip install filelock networkx sympy numpy pillow
   ```

### Updating PyTorch

To update to a newer nightly build, run the installation command again:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Additional Resources

- [PyTorch Official Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
