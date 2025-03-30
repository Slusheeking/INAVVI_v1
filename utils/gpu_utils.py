#!/usr/bin/env python3
"""
GPU Utilities Module

This module provides standardized GPU detection, initialization, and utility functions
for high-performance computing in the trading system. It supports:

1. Unified device detection across PyTorch, CuPy, ONNX, and TensorRT
2. Graceful fallbacks when GPU acceleration is unavailable
3. Memory management optimizations for different GPU architectures
4. Common tensor conversion utilities
5. GH200 Grace Hopper-specific optimizations

All components in the trading system should use this module for GPU-related operations
to ensure consistent behavior and optimal performance.
"""

import logging
import os
import sys
import traceback
from typing import Any, Dict, Optional, Tuple, Union, Callable

import numpy as np

# Configure logging
logger = logging.getLogger("gpu_utils")

# Import GPU acceleration libraries with fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info(f"CuPy available (version {cp.__version__})")
except ImportError:
    CUPY_AVAILABLE = False
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

    # Check for CUDA support
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        # We will set the specific device later in GPUManager
        logger.info(f"PyTorch CUDA available.")
    else:
        logger.warning("PyTorch CUDA not available, using CPU")
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
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
USE_GH200 = os.environ.get("USE_GH200", "true").lower() == "true"
GPU_MEMORY_LIMIT_MB = int(os.environ.get("GPU_MEMORY_LIMIT_MB", "16000")) # Default 16GB
TENSORRT_PRECISION = os.environ.get("TENSORRT_PRECISION", "FP16")
USE_MIXED_PRECISION = os.environ.get("USE_MIXED_PRECISION", "true").lower() == "true"
USE_AMP = os.environ.get("USE_AMP", "true").lower() == "true"


class GPUManager:
    """
    Unified GPU management class for the trading system

    This class provides a centralized interface for GPU operations including:
    - Device detection and selection (including GH200)
    - Memory management (including unified memory for GH200)
    - Tensor conversions
    - Model optimization hooks

    It supports multiple GPU frameworks (PyTorch, CuPy, TensorRT) with consistent
    interfaces and graceful fallbacks to CPU when needed.
    """

    def __init__(
        self,
        use_gpu: bool = USE_GPU,
        use_gh200: bool = USE_GH200,
        memory_limit_mb: int = GPU_MEMORY_LIMIT_MB,
        tensorrt_precision: str = TENSORRT_PRECISION,
        mixed_precision: bool = USE_MIXED_PRECISION,
        use_amp: bool = USE_AMP
    ) -> None:
        """
        Initialize GPU manager with configuration options

        Args:
            use_gpu: Whether to use GPU acceleration if available
            use_gh200: Whether to use GH200-specific optimizations if available
            memory_limit_mb: Memory limit in MB for GPU operations (applied to PyTorch)
            tensorrt_precision: Precision mode for TensorRT ("FP32", "FP16", or "INT8")
            mixed_precision: Whether to use mixed precision for PyTorch
            use_amp: Whether to use automatic mixed precision for PyTorch
        """
        # Determine initial GPU usability based on config and basic library availability
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
        self.autocast = None
        self.grad_scaler = None
        self.pytorch_device = torch.device("cpu") if TORCH_AVAILABLE else None

        # Store config directly on self for easier access
        self.gpu_config = {
            "memory_limit_mb": self.memory_limit_mb,
            "tensorrt_precision": self.tensorrt_precision,
            "mixed_precision": self.mixed_precision,
            "use_amp": self.use_amp,
        }

        # Initialize GPU if available and requested
        if self.use_gpu:
            self._initialize_gpu() # This will update self.use_gpu if init fails
        else:
            logger.info("GPU acceleration disabled by config or unavailable libraries")

    def _initialize_gpu(self) -> None:
        """
        Initialize GPU device and memory management, incorporating detailed checks
        and framework initializations (CuPy, PyTorch). Includes GH200 detection.
        Sets self.use_gpu to False if initialization fails.
        """
        try:
            cupy_initialized = False
            pytorch_initialized = False

            # --- Initialize CuPy (Primary device detection) ---
            if CUPY_AVAILABLE:
                if cp.cuda.is_available():
                    logger.info(f"CUDA is available through CuPy version {cp.__version__}")
                    device_count = cp.cuda.runtime.getDeviceCount()
                    gh200_found = False

                    if device_count > 0:
                        # Look for GH200 device
                        if self.use_gh200:
                            for i in range(device_count):
                                try:
                                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                                    dev_name_bytes = device_props.get("name") # Use get for safety
                                    if dev_name_bytes and isinstance(dev_name_bytes, bytes):
                                        device_name = dev_name_bytes.decode()
                                        if "GH200" in device_name:
                                            cp.cuda.Device(i).use()
                                            gh200_found = True
                                            self.mempool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
                                            cp.cuda.set_allocator(self.mempool.malloc)
                                            free, total = cp.cuda.runtime.memGetInfo()
                                            self.device_id = i
                                            self.device_name = device_name
                                            self.total_memory = total
                                            logger.info(f"Using GH200 device {i}: {device_name} with {free/(1024**3):.2f}GB free / {total/(1024**3):.2f}GB total memory (Unified Memory Pool)")
                                            cupy_initialized = True
                                            break
                                except Exception as dev_err:
                                    logger.warning(f"Could not query device {i} via CuPy: {dev_err}")

                        # Use first available GPU if GH200 not found or not requested
                        if not gh200_found:
                            try:
                                self.device_id = cp.cuda.Device().id # Get default device ID
                                device_props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                                dev_name_bytes = device_props.get("name")
                                self.device_name = dev_name_bytes.decode() if dev_name_bytes and isinstance(dev_name_bytes, bytes) else f"CUDA Device {self.device_id}"
                                self.mempool = cp.cuda.MemoryPool() # Standard memory pool
                                cp.cuda.set_allocator(self.mempool.malloc)
                                free, total = cp.cuda.runtime.memGetInfo()
                                self.total_memory = total
                                logger.info(f"Using default GPU device {self.device_id}: {self.device_name} (GH200 not found/requested)")
                                cupy_initialized = True
                            except Exception as dev_err:
                                 logger.error(f"Failed to initialize default CuPy device: {dev_err}")
                                 # Don't disable GPU yet, PyTorch might still work
                    else:
                         logger.warning("No CUDA devices found by CuPy.")
                else:
                    logger.warning("CUDA not available through CuPy.")
            else:
                 logger.info("CuPy not available.")


            # --- Initialize PyTorch ---
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        device_count_pt = torch.cuda.device_count()
                        if device_count_pt == 0:
                             logger.warning("No CUDA devices found by PyTorch.")
                             # If CuPy also failed, disable GPU
                             if not cupy_initialized: self.use_gpu = False
                             return # Exit init if PyTorch finds no devices

                        # Ensure PyTorch uses the same device selected by CuPy (if any)
                        target_device_id = self.device_id if self.device_id >= 0 and self.device_id < device_count_pt else 0

                        if target_device_id != self.device_id:
                             logger.warning(f"Device ID mismatch or invalid. CuPy={self.device_id}, PyTorch using {target_device_id}.")
                             self.device_id = target_device_id # Align to PyTorch's choice

                        torch.cuda.set_device(self.device_id)
                        self.pytorch_device = torch.device(f"cuda:{self.device_id}")
                        pt_device_name = torch.cuda.get_device_name(self.device_id)
                        logger.info(f"PyTorch using CUDA device {self.device_id}: {pt_device_name}")

                        # Update device name and memory if CuPy didn't set them
                        if self.device_name == "CPU": self.device_name = pt_device_name
                        if self.total_memory == 0: self.total_memory = torch.cuda.get_device_properties(self.device_id).total_memory

                        # Set memory fraction if configured
                        if self.gpu_config["memory_limit_mb"] > 0:
                            pytorch_total_mem = torch.cuda.get_device_properties(self.device_id).total_memory
                            fraction = min(1.0, (self.gpu_config["memory_limit_mb"] * 1024 * 1024) / pytorch_total_mem)
                            try:
                                torch.cuda.set_per_process_memory_fraction(fraction, self.device_id)
                                logger.info(f"Set PyTorch memory fraction to {fraction:.2f} on device {self.device_id}")
                            except RuntimeError as mem_err:
                                logger.warning(f"Could not set PyTorch memory fraction: {mem_err}")


                        # Enable mixed precision if configured
                        if self.gpu_config["mixed_precision"]:
                            try:
                                from torch.cuda.amp import autocast, GradScaler
                                self.autocast = autocast
                                self.grad_scaler = GradScaler() if self.gpu_config["use_amp"] else None
                                logger.info("Enabled mixed precision (float16) for PyTorch")
                            except Exception as amp_err:
                                logger.warning(f"Could not enable mixed precision: {amp_err}")

                        # Test PyTorch GPU
                        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.pytorch_device)
                        b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=self.pytorch_device)
                        c = torch.matmul(a, b)
                        logger.info(f"PyTorch GPU test successful: {c.cpu().numpy()}")
                        pytorch_initialized = True
                    else:
                        logger.warning("No GPU detected by PyTorch.")
                        # If CuPy also failed, disable GPU
                        if not cupy_initialized: self.use_gpu = False

                except Exception as pt_err:
                    logger.exception(f"Error initializing PyTorch GPU support: {pt_err}")
                    # If CuPy also failed, disable GPU
                    if not cupy_initialized: self.use_gpu = False
            else:
                 logger.info("PyTorch not available or CUDA disabled.")


            # --- Final Logging and Status Update ---
            self.use_gpu = self.use_gpu and (cupy_initialized or pytorch_initialized) # Final check

            if self.use_gpu:
                logger.info("GPU acceleration successfully initialized:")
                logger.info(f"- Device: {self.device_name} (ID: {self.device_id})")
                logger.info(f"- Total Memory: {self.total_memory / (1024**3):.2f} GB")
                if self.use_gh200 and gh200_found: # Check if GH200 was actually found and used
                    logger.info("- GH200 optimizations: Enabled (Unified Memory)")
                if cupy_initialized:
                    logger.info(f"- CuPy: Available (v{cp.__version__})")
                if pytorch_initialized:
                    logger.info(f"- PyTorch: Available (v{torch.__version__}, CUDA {torch.version.cuda})")
                    if ONNX_AVAILABLE: logger.info(f"- ONNX: Available (v{ONNX_VERSION})")
                    if TENSORRT_AVAILABLE: logger.info(f"- TensorRT: Available (v{TENSORRT_VERSION})")
            else:
                logger.warning("GPU initialization failed or disabled, falling back to CPU.")
                self.device_name = "CPU" # Ensure name reflects status
                self.device_id = -1

        except Exception as e:
            logger.error(f"Critical error during GPU initialization: {e}")
            logger.error(traceback.format_exc())
            self.use_gpu = False
            self.use_gh200 = False
            self.device_name = "CPU"
            self.device_id = -1

    def to_gpu(self, data: Any) -> Any:
        """
        Convert data to GPU format based on available frameworks

        Args:
            data: Data to convert (numpy array, torch tensor, or compatible type)

        Returns:
            Data in GPU format (cupy array or torch tensor) or original data if GPU disabled/failed.
        """
        if not self.use_gpu:
            return data

        try:
            # Handle numpy arrays
            if isinstance(data, np.ndarray):
                if CUPY_AVAILABLE and cp.cuda.is_available():
                    return cp.asarray(data) # CuPy preferred for arrays if available
                elif TORCH_AVAILABLE and CUDA_AVAILABLE:
                    return torch.from_numpy(data).to(self.pytorch_device)
                return data

            # Handle torch tensors (ensure correct device)
            elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                return data.to(self.pytorch_device)

            # Handle lists
            elif isinstance(data, list):
                if CUPY_AVAILABLE and cp.cuda.is_available():
                    return cp.asarray(data) # CuPy preferred
                elif TORCH_AVAILABLE and CUDA_AVAILABLE:
                    return torch.tensor(data, device=self.pytorch_device)
                return data

            # Return as is for other types
            return data

        except Exception as e:
            logger.error(f"Error converting data to GPU: {e}")
            logger.error(traceback.format_exc())
            return data # Return original data on error

    def from_gpu(self, data: Any) -> Any:
        """
        Convert data from GPU format to CPU (numpy array)

        Args:
            data: Data to convert (cupy array, torch tensor, or compatible type)

        Returns:
            Data as numpy array on CPU or original data if not a recognized GPU type.
        """
        try:
            # Handle cupy arrays
            if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
                return cp.asnumpy(data)

            # Handle torch tensors
            elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                return data.cpu().numpy()

            # Return as is for other types
            return data

        except Exception as e:
            logger.error(f"Error converting data from GPU: {e}")
            logger.error(traceback.format_exc())
            return data # Return original data on error

    def process_array(self, array: np.ndarray, func: Callable) -> np.ndarray:
        """
        Process a numpy array with a function, using GPU (CuPy preferred) if available.

        Args:
            array: Input numpy array
            func: Function to apply to the array (should accept CuPy/NumPy array)

        Returns:
            Processed numpy array
        """
        if not self.use_gpu or not CUPY_AVAILABLE or not cp.cuda.is_available():
            # Fallback to CPU if GPU disabled or CuPy not usable
            return func(array)

        try:
            # Convert to GPU (CuPy)
            gpu_array = cp.asarray(array)

            # Apply function
            result_gpu = func(gpu_array)

            # Convert back to CPU
            return cp.asnumpy(result_gpu)

        except Exception as e:
            logger.error(f"Error processing array on GPU with CuPy: {e}")
            logger.error(traceback.format_exc())
            # Fall back to CPU
            logger.warning("Falling back to CPU for array processing.")
            return func(array)

    def optimize_model(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        output_path: Optional[str] = None,
        precision: Optional[str] = None
    ) -> Optional[str]:
        """
        Optimize a model for GPU inference using TensorRT (requires ONNX model input).

        Args:
            model_path: Path to the ONNX model file.
            input_shape: Input shape for the model (e.g., (batch_size, channels, height, width)).
            output_path: Path to save the optimized TensorRT engine (default: model_path + ".engine").
            precision: Precision mode ("FP32", "FP16", or "INT8"). Defaults to manager setting.

        Returns:
            Path to the optimized model file, or None if optimization failed or prerequisites missing.
        """
        if not self.use_gpu or not TENSORRT_AVAILABLE or not ONNX_AVAILABLE:
            logger.warning("TensorRT or ONNX not available, model optimization skipped")
            return None

        try:
            # Set default output path if not provided
            if output_path is None:
                output_path = model_path + ".engine"

            # Set default precision if not provided
            if precision is None:
                precision = self.tensorrt_precision

            # Load ONNX model
            onnx_model = onnx.load(model_path)

            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING) # Use WARNING level

            # Create builder and network
            # EXPLICIT_BATCH flag allows dynamic batch sizes
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with trt.Builder(trt_logger) as builder, \
                 builder.create_network(network_flags) as network, \
                 trt.OnnxParser(network, trt_logger) as parser:

                # Parse ONNX model
                if not parser.parse(onnx_model.SerializeToString()):
                    error_msgs = ""
                    for i in range(parser.num_errors):
                        error_msgs += f"{parser.get_error(i)}\n"
                    raise RuntimeError(f"Failed to parse ONNX model: {error_msgs}")

                # Create builder config
                config = builder.create_builder_config()

                # Set max workspace size (adjust as needed)
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

                # Set precision
                if precision == "FP16" and builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("Using FP16 precision for TensorRT")
                elif precision == "INT8" and builder.platform_has_fast_int8:
                    # INT8 requires calibration - more complex setup needed here
                    # config.set_flag(trt.BuilderFlag.INT8)
                    # config.int8_calibrator = ... # Set calibrator instance
                    logger.warning("INT8 precision requested but calibration not implemented. Using FP32.")
                    precision = "FP32" # Fallback
                else:
                    precision = "FP32" # Ensure precision reflects actual setting
                    logger.info("Using FP32 precision for TensorRT")

                # Build serialized engine
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    raise RuntimeError("Failed to build TensorRT engine.")

                # Save engine
                with open(output_path, "wb") as f:
                    f.write(serialized_engine)

                logger.info(f"TensorRT engine ({precision}) saved to {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Error optimizing model with TensorRT: {e}")
            logger.error(traceback.format_exc())
            return None

    def clear_memory(self) -> None:
        """Clear GPU memory cache for PyTorch and CuPy."""
        try:
            if TORCH_AVAILABLE and CUDA_AVAILABLE and self.use_gpu:
                torch.cuda.empty_cache()
                logger.info("PyTorch CUDA memory cache cleared")

            if CUPY_AVAILABLE and self.mempool and self.use_gpu:
                self.mempool.free_all_blocks()
                logger.info("CuPy memory pool cleared")

        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")
            logger.error(traceback.format_exc())


# Create a singleton instance for global use
# This ensures GPU initialization happens only once.
gpu_manager = GPUManager()


# Utility functions for easy access to common operations
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
    gpu_manager.clear_memory()


def get_device_info() -> Dict[str, Any]:
    """Get information about the current GPU device"""
    return {
        "device_id": gpu_manager.device_id,
        "device_name": gpu_manager.device_name,
        "total_memory": gpu_manager.total_memory,
        "use_gpu": gpu_manager.use_gpu,
        "use_gh200": gpu_manager.use_gh200 and "GH200" in gpu_manager.device_name, # Check name
        "frameworks": {
            "cupy": CUPY_AVAILABLE and cp.cuda.is_available(),
            "torch": TORCH_AVAILABLE and CUDA_AVAILABLE,
            "onnx": ONNX_AVAILABLE,
            "tensorrt": TENSORRT_AVAILABLE
        }
    }