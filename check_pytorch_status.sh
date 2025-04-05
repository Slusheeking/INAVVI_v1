#!/bin/bash
# Quick status check script for PyTorch installation

set -e  # Exit on any error

echo "===== PyTorch Installation Status Check ====="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if we're in a virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Running in virtual environment: $VIRTUAL_ENV"
else
    echo "Not running in a virtual environment"
fi

# Check if PyTorch is installed
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch is installed"
    
    # Get PyTorch version
    PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorch version: $PYTORCH_VERSION"
    
    # Check CUDA availability
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        echo "CUDA is available"
        
        # Get CUDA version
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        echo "CUDA version: $CUDA_VERSION"
        
        # Get device count
        DEVICE_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
        echo "CUDA device count: $DEVICE_COUNT"
        
        # Get device name(s)
        echo "CUDA device(s):"
        for (( i=0; i<$DEVICE_COUNT; i++ )); do
            DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name($i))")
            echo "  Device $i: $DEVICE_NAME"
        done
        
        # Check if we have the GH200 GPU
        if python3 -c "import torch; print(any('GH200' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))" | grep -q "True"; then
            echo "✅ GH200 GPU detected!"
        else
            echo "❌ GH200 GPU not detected."
        fi
    else
        echo "❌ CUDA is not available with this PyTorch installation"
    fi
    
    # Check for torchvision
    if python3 -c "import torchvision" 2>/dev/null; then
        TORCHVISION_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)")
        echo "torchvision version: $TORCHVISION_VERSION"
    else
        echo "torchvision is not installed"
    fi
    
    # Check for torchaudio
    if python3 -c "import torchaudio" 2>/dev/null; then
        TORCHAUDIO_VERSION=$(python3 -c "import torchaudio; print(torchaudio.__version__)")
        echo "torchaudio version: $TORCHAUDIO_VERSION"
    else
        echo "torchaudio is not installed"
    fi
    
    echo -e "\nFor a more detailed test with performance metrics, run:"
    echo "python3 utils/verify_pytorch_cuda.py"
else
    echo "❌ PyTorch is not installed"
    echo "To install PyTorch with CUDA support, run:"
    echo "./install_pytorch_cuda.sh"
    echo "or"
    echo "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
fi

echo "========================================="
