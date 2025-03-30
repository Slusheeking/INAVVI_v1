#!/bin/bash
set -e

echo "Building XGBoost with GPU support for NVIDIA GH200"

# Create a temporary directory for building
BUILD_DIR=$(mktemp -d)
cd $BUILD_DIR

echo "Cloning XGBoost repository..."
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost

# Checkout the specific version (3.0.0)
git checkout v3.0.0

# Create build directory
mkdir -p build
cd build

echo "Configuring build with CMake..."
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES="90" -DCMAKE_INSTALL_PREFIX=/usr/local

echo "Building XGBoost..."
make -j$(nproc)

echo "Installing XGBoost libraries..."
sudo make install

cd ..

echo "Installing Python package..."
cd python-package
pip uninstall -y xgboost  # Remove existing installation
pip install -e .  # Install in development mode

echo "XGBoost with GPU support has been successfully built and installed."
echo "You can verify the installation with:"
echo "python3 -c \"import xgboost; print(xgboost.__version__); print('GPU support:', xgboost.build_info()['USE_CUDA'])\""

# Clean up
cd /
rm -rf $BUILD_DIR

echo "Build completed successfully."