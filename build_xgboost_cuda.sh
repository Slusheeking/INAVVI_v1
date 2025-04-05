#!/bin/bash
set -e

# 1. Install required system packages
sudo apt update
sudo apt install -y git cmake build-essential \
  libssl-dev libtbb-dev python3-dev python3-pip

# 2. Clone the XGBoost repo (latest version)
git clone --recursive https://github.com/dmlc/xgboost.git
cd xgboost

# 3. OPTIONAL: Check out the latest release (e.g., v3.0.0)
# git checkout tags/v3.0.0 -b v3.0.0

# 4. Create build directory
mkdir build
cd build

# 5. Configure build with CUDA support
cmake .. -DUSE_CUDA=ON

# 6. Compile using all available CPU cores
make -j$(nproc)

# 7. Install Python package from source
cd ../python-package
python3 setup.py install

echo "XGBoost with CUDA support installed successfully!"
