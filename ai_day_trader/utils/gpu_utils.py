#!/usr/bin/env python3
"""
GPU Utilities Module

This module provides standardized GPU detection, initialization, and utility functions
for high-performance computing in the trading system.
"""

import logging
import os
from typing import Any, Dict, Callable # Removed unused: sys, traceback, Optional, Tuple, Union
import numpy as np # Moved numpy import here

CUDA_HOME = "/usr/lib/aarch64-linux-gnu"
if 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = CUDA_HOME
cuda_lib_dir = "/usr/lib/aarch64-linux-gnu"
ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
if cuda_lib_dir not in ld_library_path:
    os.environ["LD_LIBRARY_PATH"] = (ld_library_path + os.pathsep if ld_library_path else "") + cuda_lib_dir

# Configure logging
logger = logging.getLogger("gpu_utils")

# Import GPU acceleration libraries with fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    # Check if CUDA is available for CuPy
    CUPY_CUDA_AVAILABLE = cp.cuda.is_available()
    logger.info(f"CuPy available (version {cp.__version__})")
    if CUPY_CUDA_AVAILABLE:
        logger.info(f"CuPy CUDA available (version {cp.cuda.runtime.runtimeGetVersion()})")
except ImportError:
    CUPY_AVAILABLE = False
    CUPY_CUDA_AVAILABLE = False
    cp = None
    logger.warning("CuPy not available, GPU array processing will be limited")

# Import ONNX with fallback
try:
    import onnx
    ONNX_VERSION = onnx.__version__
    ONNX_AVAILABLE = True
    logger.info(f"ONNX available (version {ONNX_VERSION})")
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    logger.warning("ONNX not available, model optimization will be limited")

# Import PyTorch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch available (version {torch.__version__})")

    # Force PyTorch to re-evaluate CUDA availability after environment variables are set
    if hasattr(torch.cuda, '_initialized'):
        torch.cuda._initialized = False

    # Check for CUDA support
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        # We will set the specific device later in GPUManager
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        logger.info(f"PyTorch CUDA available. Devices: {device_count}, Name: {device_name}")
    else:
        # Try to diagnose why CUDA is not available
        logger.warning("PyTorch CUDA not available, using CPU")
        if os.path.exists('/usr/lib/aarch64-linux-gnu/libcudart.so.12'):
            logger.info("CUDA libraries found but PyTorch can't detect them. This may be a PyTorch build issue.")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None # Ensure torch is None if import fails
    logger.warning("PyTorch not available, tensor operations will be CPU-only")

# Import TensorRT with fallback
try:
    import tensorrt as trt
    TENSORRT_VERSION = trt.__version__
    TENSORRT_AVAILABLE = True
    logger.info(f"TensorRT available (version {TENSORRT_VERSION})")
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    logger.warning("TensorRT not available, GPU model optimization disabled")

# Environment variables for GPU configuration
USE_GPU = os.environ.get("TRADING_USE_GPU", "true").lower() == "true" # Use TRADING_ prefix
USE_GH200 = os.environ.get("TRADING_USE_GH200", "true").lower() == "true" # Use TRADING_ prefix
GPU_MEMORY_LIMIT_MB = int(os.environ.get("TRADING_GPU_MEMORY_LIMIT_MB", "16000")) # Use TRADING_ prefix
TENSORRT_PRECISION = os.environ.get("TRADING_TENSORRT_PRECISION", "FP16") # Use TRADING_ prefix
USE_MIXED_PRECISION = os.environ.get("TRADING_USE_MIXED_PRECISION", "true").lower() == "true" # Use TRADING_ prefix
USE_AMP = os.environ.get("TRADING_USE_AMP", "true").lower() == "true" # Use TRADING_ prefix

# Check for CUDA libraries in standard locations
cuda_lib_locations = [
    '/usr/local/cuda/lib64',
    '/usr/lib/aarch64-linux-gnu',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/lib/cuda/lib64'
]

# Find CUDA home directory
cuda_home_locations = [
    '/usr/local/cuda',
    '/usr/lib/cuda',
    '/opt/cuda'
]

# Set CUDA_HOME if not set
if 'CUDA_HOME' not in os.environ:
    for cuda_home in cuda_home_locations:
        if os.path.exists(cuda_home):
            os.environ['CUDA_HOME'] = cuda_home
            logger.info(f"CUDA_HOME automatically set to {os.environ['CUDA_HOME']}")
            break

# Find CUDA libraries
cuda_lib_path = None
for lib_path in cuda_lib_locations:
    if os.path.exists(lib_path) and (
        os.path.exists(os.path.join(lib_path, 'libcudart.so')) or
        os.path.exists(os.path.join(lib_path, 'libcudart.so.12'))
    ):
        cuda_lib_path = lib_path
        logger.info(f"Found CUDA libraries at {cuda_lib_path}")
        break

# Set LD_LIBRARY_PATH if CUDA libraries found
if cuda_lib_path:
    if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = cuda_lib_path
    elif cuda_lib_path not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    logger.info(f"LD_LIBRARY_PATH set to include CUDA libs: {os.environ.get('LD_LIBRARY_PATH')}")
else:
    logger.warning("Could not find CUDA libraries in standard locations")

class GPUManager:
    """GPU Manager for unified device handling"""

    def __init__(
        self,
        use_gpu: bool = USE_GPU,
        use_gh200: bool = USE_GH200,
        memory_limit_mb: int = GPU_MEMORY_LIMIT_MB,
        tensorrt_precision: str = TENSORRT_PRECISION,
        mixed_precision: bool = USE_MIXED_PRECISION,
        use_amp: bool = USE_AMP
    ) -> None:
        """Initialize GPU manager with configuration options"""
        self.use_gpu = use_gpu and (CUDA_AVAILABLE or (CUPY_AVAILABLE and cp.cuda.is_available()))
        self.use_gh200 = use_gh200 if self.use_gpu else False
        self.memory_limit_mb = memory_limit_mb
        self.tensorrt_precision = tensorrt_precision
        self.mixed_precision = mixed_precision
        self.use_amp = use_amp

        # Initialize device properties
        self.device_id = -1
        self.device_name = "CPU"
        self.total_memory = 0
        self.mempool = None
        self.pytorch_device = torch.device("cpu") if TORCH_AVAILABLE else None
        self.is_gh200 = False # Added flag

        # Configuration reference
        self.gpu_config = {
            "memory_limit_mb": self.memory_limit_mb,
            "tensorrt_precision": self.tensorrt_precision,
            "mixed_precision": self.mixed_precision,
            "use_amp": self.use_amp,
        }

        # Initialize GPU if available
        if self.use_gpu:
            self._initialize_gpu()
        else:
            logger.info("GPU acceleration disabled or unavailable")

    def _initialize_gpu(self) -> None:
        """Initialize GPU device and set properties"""
        try:
            # Force PyTorch to re-evaluate CUDA availability
            if TORCH_AVAILABLE and hasattr(torch.cuda, '_initialized'):
                torch.cuda._initialized = False

            # Try initialization with PyTorch
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count == 0:
                    logger.warning("PyTorch reports CUDA available but no devices found")
                    self.use_gpu = False
                    self.device_name = "CPU"
                    self.pytorch_device = torch.device("cpu")
                    return

                self.device_id = 0
                torch.cuda.set_device(self.device_id)
                self.pytorch_device = torch.device(f"cuda:{self.device_id}")

                try:
                    self.device_name = torch.cuda.get_device_name(self.device_id)
                    self.total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
                    logger.info(f"Using GPU: {self.device_name} with {self.total_memory / (1024**3):.2f} GB memory")
                    # Check specifically for GH200 in name
                    if "GH200" in self.device_name:
                         self.is_gh200 = True
                         logger.info("GH200 GPU detected.")

                except Exception as e:
                    logger.warning(f"Could not get device properties: {e}")
                    self.device_name = "Unknown CUDA Device"

                # Test PyTorch GPU
                try:
                    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.pytorch_device)
                    b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=self.pytorch_device)
                    c = torch.matmul(a, b)
                    result = c.cpu().numpy()
                    logger.info(f"PyTorch GPU test successful: {result}")
                    del a, b, c, result
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"PyTorch GPU test failed: {e}")
                    self.use_gpu = False
                    self.device_name = "CPU"
                    self.pytorch_device = torch.device("cpu")
            else:
                self.use_gpu = False
                logger.warning("PyTorch CUDA not available")

        except Exception as e:
            logger.error(f"GPU initialization error: {e}", exc_info=True)
            self.use_gpu = False
            self.device_name = "CPU"
            self.pytorch_device = torch.device("cpu") if TORCH_AVAILABLE else None

    def to_gpu(self, data: Any) -> Any:
        """Convert data to GPU format"""
        if not self.use_gpu: return data
        try:
            if isinstance(data, np.ndarray) and TORCH_AVAILABLE:
                return torch.from_numpy(data).to(self.pytorch_device)
            elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                return data.to(self.pytorch_device)
            return data
        except Exception as e:
            logger.error(f"Error converting to GPU: {e}")
            return data

    def from_gpu(self, data: Any) -> Any:
        """Convert data from GPU to CPU"""
        try:
            if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            elif CUPY_AVAILABLE and isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
            return data
        except Exception as e:
            logger.error(f"Error converting from GPU: {e}")
            return data

    def process_array(self, array: np.ndarray, func: Callable) -> np.ndarray:
        """Process a numpy array with a function, using GPU if available."""
        if not self.use_gpu or not CUPY_AVAILABLE or not cp.cuda.is_available():
            return func(array)
        try:
            gpu_array = cp.asarray(array)
            result_gpu = func(gpu_array)
            return cp.asnumpy(result_gpu)
        except Exception as e:
            logger.error(f"Error processing array on GPU with CuPy: {e}", exc_info=True)
            logger.warning("Falling back to CPU for array processing.")
            return func(array)

    def clear_memory(self) -> None:
        """Clear GPU memory cache"""
        try:
            if TORCH_AVAILABLE and CUDA_AVAILABLE and self.use_gpu:
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")


# Create singleton instance
gpu_manager = GPUManager()


# Utility functions for easy access
def is_gpu_available() -> bool:
    """Check if GPU acceleration is available and enabled"""
    return gpu_manager.use_gpu

def to_gpu(data: Any) -> Any:
    """Convert data to GPU format"""
    return gpu_manager.to_gpu(data)

def from_gpu(data: Any) -> Any:
    """Convert data from GPU format to CPU"""
    return gpu_manager.from_gpu(data)

def process_array(array: np.ndarray, func: Callable) -> np.ndarray:
    """Process a numpy array with a function, using GPU if available"""
    return gpu_manager.process_array(array, func)

def clear_gpu_memory() -> None:
    """Clear GPU memory cache"""
    return gpu_manager.clear_memory()

def get_device_info() -> Dict[str, Any]:
    """Get information about the current GPU device"""
    return {
        "device_id": gpu_manager.device_id,
        "device_name": gpu_manager.device_name,
        "total_memory": gpu_manager.total_memory,
        "use_gpu": gpu_manager.use_gpu,
        "is_gh200": gpu_manager.is_gh200, # Added flag
        "frameworks": {
            "cupy": CUPY_AVAILABLE,
            "torch": TORCH_AVAILABLE and CUDA_AVAILABLE,
            "onnx": ONNX_AVAILABLE,
            "tensorrt": TENSORRT_AVAILABLE
        }
    }

def run_diagnostics() -> Dict[str, Any]:
    """
    Run comprehensive GPU diagnostics and return detailed information.
    Used for system health checks and debugging.

    Returns:
        Dictionary with GPU diagnostic information
    """
    diagnostics = get_device_info() # Start with basic info
    diagnostics["configuration"] = gpu_manager.gpu_config

    # Add memory usage if available
    if TORCH_AVAILABLE and CUDA_AVAILABLE and gpu_manager.use_gpu:
        try:
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            diagnostics["memory_usage"] = {
                "allocated_gb": round(mem_allocated, 2),
                "reserved_gb": round(mem_reserved, 2),
                "percent_used": round(mem_allocated / gpu_manager.total_memory * 100, 2) if gpu_manager.total_memory > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")

    return diagnostics
