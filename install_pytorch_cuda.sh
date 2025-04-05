#!/bin/bash
# Script to install PyTorch with CUDA support for GH200 GPU

set -e  # Exit on error

echo "===== Installing PyTorch with CUDA 12.8 support for GH200 GPU ====="

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Create symlinks if needed
if [ ! -d "/usr/local/cuda/lib64" ]; then
    echo "Creating /usr/local/cuda/lib64 directory..."
    sudo mkdir -p /usr/local/cuda/lib64
fi

# Create symlinks from /usr/lib/aarch64-linux-gnu to /usr/local/cuda/lib64
echo "Creating symlinks for CUDA libraries..."
for lib in /usr/lib/aarch64-linux-gnu/libcudart.so*; do
    lib_name=$(basename $lib)
    if [ ! -e "/usr/local/cuda/lib64/$lib_name" ]; then
        echo "Creating symlink for $lib_name..."
        sudo ln -sf $lib /usr/local/cuda/lib64/$lib_name
    fi
done

# Uninstall current PyTorch
echo "Removing current PyTorch installation..."
pip uninstall -y torch torchvision torchaudio

# Install PyTorch nightly build with CUDA 12.8 support
echo "Installing PyTorch with CUDA 12.8 support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if hasattr(torch.version, 'cuda') else 'None'); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Test with a simple CUDA operation
echo "Testing PyTorch with CUDA..."
python -c "
import torch
if torch.cuda.is_available():
    print('Testing CUDA with PyTorch...')
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device='cuda')
    c = torch.matmul(a, b)
    print('Result:', c.cpu().numpy())
    print('Test successful!')
else:
    print('CUDA not available for PyTorch')
"

echo "===== Installation complete ====="