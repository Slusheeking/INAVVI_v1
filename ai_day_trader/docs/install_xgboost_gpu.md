# Installing XGBoost with GPU Support

This guide provides comprehensive instructions for installing XGBoost with GPU support on our ARM64-based system running CUDA 12.8.

## 1. System Requirements

Before installing XGBoost with GPU support, ensure your system meets the following requirements:

- **Operating System**: Ubuntu (ARM64 architecture)
- **CUDA**: Version 12.8 or compatible
- **NVIDIA Drivers**: Compatible with CUDA 12.8
- **Python**: 3.8 or higher (preferably in a virtual environment)
- **Development Tools**: Git, CMake, build-essential, and related libraries

## 2. Environment Preparation

### 2.1 Create and Activate a Python Virtual Environment

```bash
# Create a virtual environment if not already present
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2.2 Set Up Environment Variables

Ensure CUDA libraries are properly configured:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

> **Note**: Our system uses ARM64 architecture, so the libraries are in `/usr/lib/aarch64-linux-gnu` rather than the typical x86_64 locations.

### 2.3 Install Required System Packages

```bash
sudo apt update
sudo apt install -y git cmake build-essential libssl-dev libtbb-dev python3-dev python3-pip
```

## 3. Installing XGBoost with GPU Support

There are two approaches to installing XGBoost with GPU support:

1. Using a pre-built wheel (faster but may not be optimized for our system)
2. Building from source (slower but optimized for our specific hardware)

We recommend building from source for optimal performance on our ARM64 system.

### 3.1 Building XGBoost from Source

#### 3.1.1 Clone the XGBoost Repository

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
```

#### 3.1.2 Create and Configure the Build Directory

```bash
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
```

> **Note**: The `-DUSE_CUDA=ON` flag is critical for enabling GPU support.

#### 3.1.3 Compile XGBoost

```bash
make -j$(nproc)
```

This command utilizes all available CPU cores to speed up compilation. The build process may take several minutes.

#### 3.1.4 Install Python Bindings

Navigate to the python-package directory and install:

```bash
cd ../python-package
pip install .
```

### 3.2 Alternative: Using Pre-built Wheel

If building from source is too time-consuming, you can try installing a pre-built wheel:

```bash
pip install xgboost --extra-index-url https://pypi.anaconda.org/rapidsai/simple
```

However, this approach may not provide optimal performance for our specific ARM64 hardware.

## 4. Verifying the Installation

Create a simple test script to verify that XGBoost is correctly installed with GPU support:

```python
import xgboost as xgb

# Get XGBoost version
xgb_version = xgb.__version__
print(f"XGBoost version: {xgb_version}")

# Check for CUDA support
print("Checking CUDA support via parameters test")
test_params = {'device': 'cuda'}
test_dmatrix = xgb.DMatrix([[0, 0]], label=[0])
test_model = xgb.train(test_params, test_dmatrix, num_boost_round=1)
print("CUDA support confirmed - test model trained successfully")

# Train a simple model with GPU
dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
params = {
    'tree_method': 'hist',  # Use histogram-based algorithm
    'device': 'cuda',       # Use CUDA for GPU acceleration
    'objective': 'binary:logistic'
}
model = xgb.train(params, dtrain, num_boost_round=10)
print("Model trained successfully with GPU!")
```

Save this script as `test_xgb_gpu.py` and run it:

```bash
python test_xgb_gpu.py
```

If successful, you should see output similar to:

```
XGBoost version: 3.1.0-dev
Checking CUDA support via parameters test
CUDA support confirmed - test model trained successfully
Model trained successfully with GPU!
```

## 5. Important Notes and Best Practices

### 5.1 XGBoost GPU Parameters

When using XGBoost with GPU support, note the following parameter conventions:

- Modern approach (XGBoost 2.0+):
  ```python
  params = {
      'tree_method': 'hist',  # Use histogram-based algorithm
      'device': 'cuda',       # Use CUDA for GPU acceleration
  }
  ```

- Legacy approach (may see in older code):
  ```python
  params = {
      'tree_method': 'gpu_hist',  # Deprecated but still works
  }
  ```

### 5.2 Memory Management

GPU memory is limited. For large datasets, consider:

- Using `gpu_id` parameter to specify which GPU to use if multiple are available
- Adjusting the `max_bin` parameter (lower values use less memory)
- Monitoring GPU memory usage during training

### 5.3 Version Compatibility

Ensure compatibility between:
- CUDA version
- XGBoost version
- Python version
- GPU driver version

## 6. Troubleshooting

### 6.1 Common Issues

1. **CUDA not found during build**:
   - Ensure CUDA_HOME is correctly set
   - Verify LD_LIBRARY_PATH includes CUDA libraries

2. **Build failures**:
   - Check CMake and compiler versions
   - Ensure all dependencies are installed

3. **Runtime errors**:
   - Verify CUDA drivers are properly installed
   - Check for CUDA runtime library version mismatches

### 6.2 Fix Script

For quick fixes to common XGBoost installation issues, we've provided a script:

```bash
./fix_xgboost.sh
```

This script can handle:
- Removing existing XGBoost installations
- Attempting installation via pre-built wheel
- Falling back to source compilation if needed
- Verifying the installation

## 7. Performance Considerations

For our specific ARM64 system with CUDA 12.8, we've observed:

- Histogram-based tree method (`tree_method='hist'`, `device='cuda'`) provides the best performance
- Using mixed precision can further accelerate training
- Batch sizes should be tuned according to available GPU memory

---

For more information, refer to:
- [XGBoost Official Documentation](https://xgboost.readthedocs.io/)
- [XGBoost GPU Support Documentation](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
