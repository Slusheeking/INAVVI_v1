#!/usr/bin/env python3
"""
Data Pipeline Module

This module provides a unified data pipeline for the trading system:
1. Data loading from various sources (Polygon.io, Unusual Whales)
2. Data preprocessing and feature engineering
3. Technical indicator calculation with GPU acceleration
4. Target generation for ML models
5. Efficient caching with Redis integration
6. Batched processing for optimal performance
7. Real-time and historical data handling
8. Market data analysis and ticker selection

The pipeline is optimized for NVIDIA GH200 Grace Hopper Superchips.
"""

import asyncio
import datetime
import json
import logging
import os
import pickle
import time
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from retrying import retry
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import MinMaxScaler

from config import config

# Load environment variables from .env file
load_dotenv()

# Machine learning imports
try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# GPU acceleration imports
try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

# PyTorch imports
try:
    import torch
    import torch.onnx
    from torch.utils.data import DataLoader, TensorDataset
    
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    
    # Check for TensorRT support via ONNX
    try:
        import tensorrt as trt
        import onnx
        from onnx_tf.backend import prepare
        
        TENSORRT_AVAILABLE = True
    except ImportError:
        TENSORRT_AVAILABLE = False
        trt = None
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    TENSORRT_AVAILABLE = False
    trt = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(os.environ.get("LOGS_DIR", "./logs"),
                         "data_pipeline.log"),
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("data_pipeline")

# Import Prometheus client if available
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client available for metrics collection")

    # Define Prometheus metrics
    DATA_PROCESSING_TIME = Histogram(
        "data_pipeline_processing_time_seconds",
        "Time spent processing data",
        ["operation", "use_gpu"],
    )

    CACHE_HITS = Counter(
        "data_pipeline_cache_hits_total", "Number of cache hits", [
            "data_type"],
    )

    CACHE_MISSES = Counter(
        "data_pipeline_cache_misses_total", "Number of cache misses", [
            "data_type"],
    )

    API_REQUEST_COUNT = Counter(
        "data_pipeline_api_requests_total",
        "Number of API requests",
        ["api", "endpoint"],
    )

    API_ERROR_COUNT = Counter(
        "data_pipeline_api_errors_total",
        "Number of API errors",
        ["api", "endpoint", "error_type"],
    )

    GPU_MEMORY_USAGE = Gauge(
        "data_pipeline_gpu_memory_usage_bytes", "GPU memory usage in bytes", [
            "device"],
    )

    DATA_ROWS_PROCESSED = Counter(
        "data_pipeline_rows_processed_total",
        "Number of data rows processed",
        ["operation"],
    )

    FEATURE_COUNT = Gauge(
        "data_pipeline_feature_count",
        "Number of features used in models",
        ["model_type"],
    )

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "Prometheus client not available. Metrics collection will be limited.",
    )


class DataPipeline:
    """
    Unified data pipeline for the trading system

    This class combines functionality from:
    - Data loading from APIs
    - Data preprocessing and caching
    - Feature engineering and technical indicators
    - ML data preparation
    - Market data analysis and filtering

    It provides both synchronous and asynchronous interfaces and is optimized
    for GPU acceleration when available.
    """

    def __init__(
        self,
        polygon_client=None,
        polygon_ws=None,
        unusual_whales_client=None,
        redis_client=None,
        config=None,
        use_gpu=True,
        use_gh200=True,
        test_mode=False,
    ) -> None:
        """
        Initialize the data pipeline

        Args:
            polygon_client: Polygon API client
            polygon_ws: Polygon WebSocket client
            unusual_whales_client: Unusual Whales API client
            redis_client: Redis client for caching
            config: Configuration parameters (dict or loaded from config.py)
            use_gpu: Whether to use GPU acceleration
            use_gh200: Whether to use GH200-specific optimizations
            test_mode: Whether to use mock data for testing
        """
        self.polygon = polygon_client
        self.polygon_ws = polygon_ws
        self.unusual_whales = unusual_whales_client
        self.redis = redis_client

        # Test mode for using synthetic data
        self.test_mode = test_mode

        # GPU settings
        self.use_gpu = use_gpu if CUDA_AVAILABLE else False
        self.use_gh200 = use_gh200 if self.use_gpu else False

        # Default configuration
        self.default_config = {
            "cache_dir": os.environ.get("DATA_CACHE_DIR", "./data/cache"),
            "cache_expiry": 86400,  # 1 day in seconds
            "rate_limit": {
                "polygon": 5,  # requests per second
                "unusual_whales": 2,  # requests per second
            },
            "retry_settings": {
                "stop_max_attempt_number": 3,
                "wait_exponential_multiplier": 1000,
                "wait_exponential_max": 10000,
            },
            "data_dir": os.environ.get("DATA_DIR", "./data"),
            "monitoring_dir": os.environ.get("MONITORING_DIR", "./monitoring"),
            "min_samples": 1000,
            "lookback_days": 30,
            "monitoring": {"enabled": True, "drift_threshold": 0.05},
            "feature_selection": {
                "enabled": True,
                "method": "importance",
                "threshold": 0.01,
                "n_features": 20,
            },
            "time_series_cv": {"n_splits": 5, "embargo_size": 10},
            "watchlist": {
                "refresh_interval": 900,  # 15 minutes
                "max_size": 100,
                "min_price": 5.0,
                "min_volume": 500000,
            },
        }

        # Update with provided config
        self.config = self.default_config.copy()
        if config:
            self._update_config_recursive(self.config, config)

        # Ensure directories exist
        self._ensure_directories()

        # Initialize GPU memory for data processing if available
        if self.use_gpu:
            self._initialize_gpu()

        # Create event loop for async calls
        try:
            # Try to get the current event loop
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists in this thread, create a new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        logger.info("Data Pipeline initialized")

        # Send notification to frontend
        if self.redis:
            gpu_status = "enabled" if self.use_gpu else "disabled"
            gh200_status = "enabled" if self.use_gh200 else "disabled"
            self._send_frontend_notification(
                message="Data Pipeline initialized successfully",
                level="info",
                category="system_startup",
                details={
                    "gpu_acceleration": gpu_status,
                    "gh200_optimizations": gh200_status,
                    "test_mode": self.test_mode,
                    "cache_dir": self.config["cache_dir"],
                    "timestamp": time.time()
                }
            )

    def _update_config_recursive(
        self, target: dict[str, Any], source: dict[str, Any],
    ) -> None:
        """
        Recursively update a nested dictionary with values from another dictionary

        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    # If both values are dictionaries, update recursively
                    self._update_config_recursive(target[key], value)
                else:
                    # Otherwise, update the value directly
                    target[key] = value
            else:
                # If the key doesn't exist in the target, add it
                target[key] = value

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist with proper permissions"""
        for directory in [
            self.config["data_dir"],
            self.config["monitoring_dir"],
            self.config["cache_dir"],
        ]:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.exception(
                    f"Error creating directory {directory}: {e!s}")

    def _initialize_gpu(self) -> None:
        """Initialize GPU memory for data processing with TensorFlow, TensorRT, and CuPy integration"""
        try:
            # Initialize GPU configuration
            self.gpu_config = {
                "memory_limit_mb": int(
                    os.environ.get("TF_CUDA_HOST_MEM_LIMIT_IN_MB", "16000"),
                ),
                "tensorrt_precision": os.environ.get("TENSORRT_PRECISION_MODE", "FP16"),
                "mixed_precision": os.environ.get("TF_MIXED_PRECISION", "true").lower()
                == "true",
                "memory_growth": os.environ.get(
                    "TF_FORCE_GPU_ALLOW_GROWTH", "true",
                ).lower()
                == "true",
                "xla_optimization": os.environ.get("TF_XLA_FLAGS", "").find(
                    "auto_jit=2",
                )
                != -1,
            }

            # Initialize CuPy if available
            if CUDA_AVAILABLE:
                # Check if CUDA is available through CuPy
                if cp.cuda.is_available():
                    logger.info(
                        f"CUDA is available through CuPy version {cp.__version__}",
                    )

                    # Find and use GH200 if available
                    device_count = cp.cuda.runtime.getDeviceCount()
                    gh200_found = False

                    if device_count > 0:
                        # Look for GH200 device
                        for i in range(device_count):
                            device_props = cp.cuda.runtime.getDeviceProperties(
                                i)
                            device_name = device_props["name"].decode()

                            if "GH200" in device_name and self.use_gh200:
                                # Use the GH200 device
                                cp.cuda.Device(i).use()
                                gh200_found = True

                                # GH200-specific optimizations
                                # Use unified memory for better performance on GH200
                                self.mempool = cp.cuda.MemoryPool(
                                    cp.cuda.malloc_managed,
                                )
                                cp.cuda.set_allocator(self.mempool.malloc)

                                # Get memory info
                                free, total = cp.cuda.runtime.memGetInfo()
                                logger.info(
                                    f"Using GH200 device with {free/(1024**3):.2f}GB free / {total/(1024**3):.2f}GB total memory",
                                )

                                # Store device ID for TensorFlow
                                self.gpu_device_id = i
                                break

                    if not gh200_found and device_count > 0:
                        # Use the first available GPU if GH200 not found
                        device_id = cp.cuda.Device().id
                        device_props = cp.cuda.runtime.getDeviceProperties(
                            device_id)
                        device_name = device_props["name"].decode()

                        # Standard memory pool for non-GH200 GPUs
                        self.mempool = cp.cuda.MemoryPool()
                        cp.cuda.set_allocator(self.mempool.malloc)

                        # Store device ID for TensorFlow
                        self.gpu_device_id = device_id

                        logger.info(
                            f"Using GPU device: {device_name} (GH200 not found)",
                        )

                    # Create GPU utility functions for CuPy
                    self.gpu_utils = {
                        "to_gpu": lambda data: (
                            cp.asarray(data)
                            if isinstance(data, np.ndarray | list)
                            else data
                        ),
                        "from_gpu": lambda data: (
                            cp.asnumpy(data) if isinstance(
                                data, cp.ndarray) else data
                        ),
                        "process_dataframe": self._process_dataframe_with_gpu,
                    }
                    logger.info(
                        "CuPy GPU utilities initialized for data processing")
                else:
                    logger.warning("CUDA is not available through CuPy")
                    self.use_gpu = False
            else:
                logger.warning(
                    "CuPy not available, CuPy GPU processing disabled")
                self.use_gpu = False

            # Initialize PyTorch if available
            if TORCH_AVAILABLE and self.use_gpu:
                try:
                    # Configure PyTorch for GPU
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        logger.info(f"PyTorch detected CUDA device: {torch.cuda.get_device_name(0)}")
                        
                        # Set memory fraction if configured
                        if self.gpu_config["memory_limit_mb"] > 0:
                            total_memory = torch.cuda.get_device_properties(0).total_memory
                            fraction = min(1.0, (self.gpu_config["memory_limit_mb"] * 1024 * 1024) / total_memory)
                            torch.cuda.set_per_process_memory_fraction(fraction)
                            logger.info(f"Set PyTorch memory fraction to {fraction:.2f}")

                        # Enable mixed precision if configured
                        if self.gpu_config["mixed_precision"]:
                            try:
                                from torch.cuda.amp import autocast
                                self.autocast = autocast
                                logger.info("Enabled mixed precision (float16) for PyTorch")
                            except Exception as e:
                                logger.warning(f"Could not enable mixed precision: {e}")

                        # Test PyTorch GPU
                        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
                        b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
                        c = torch.matmul(a, b)
                        logger.info(f"PyTorch GPU test successful: {c.cpu().numpy()}")

                        # Add PyTorch utility functions
                        self.torch_utils = {
                            "to_tensor": lambda data: torch.tensor(data, dtype=torch.float32, device=device),
                            "from_tensor": lambda tensor: tensor.cpu().numpy(),
                            "process_with_torch": self._process_with_pytorch,
                        }
                        logger.info("PyTorch GPU utilities initialized")

                        # Check TensorRT availability for ONNX optimization
                        if TENSORRT_AVAILABLE:
                            logger.info(f"TensorRT is available for model optimization via ONNX (precision: {self.gpu_config['tensorrt_precision']})")
                    else:
                        logger.warning("No GPU detected by PyTorch")
                        self.torch_utils = None
                except Exception as e:
                    logger.exception(f"Error initializing PyTorch: {e}")
                    self.torch_utils = None
            else:
                logger.warning(
                    "PyTorch not available or GPU disabled, PyTorch GPU processing disabled",
                )
                self.torch_utils = None

            # Log final GPU status
            if self.use_gpu:
                logger.info("GPU acceleration successfully initialized with:")
                if CUDA_AVAILABLE and cp.cuda.is_available():
                    logger.info("- CuPy for array operations")
                if TF_AVAILABLE and hasattr(self, "tf_utils") and self.tf_utils:
                    logger.info("- TensorFlow for tensor operations")
                    if TENSORRT_AVAILABLE:
                        logger.info("- TensorRT for model optimization")

                # Send notification to frontend
                self._send_frontend_notification(
                    message="GPU acceleration successfully initialized",
                    level="success",
                    category="system_status",
                    details={
                        "gpu_enabled": True,
                        "cupy_available": CUDA_AVAILABLE and cp.cuda.is_available(),
                        "tensorflow_available": TF_AVAILABLE and hasattr(self, "tf_utils") and self.tf_utils is not None,
                        "tensorrt_available": TENSORRT_AVAILABLE,
                        "device_name": cp.cuda.Device().name.decode() if CUDA_AVAILABLE and cp.cuda.is_available() else "Unknown"
                    }
                )

        except Exception as e:
            logger.exception(f"Error initializing GPU: {e}")
            self.use_gpu = False
            self.tf_utils = None

            # Send notification to frontend about GPU initialization failure
            self._send_frontend_notification(
                message="GPU initialization failed, falling back to CPU mode",
                level="warning",
                category="system_status",
                details={
                    "gpu_enabled": False,
                    "error": str(e)
                }
            )

    def _process_dataframe_with_gpu(self, df, batch_size=1000):
        """
        Process DataFrame with GPU acceleration using CuPy

        Args:
            df: Input DataFrame
            batch_size: Batch size for processing

        Returns:
            Processed DataFrame
        """
        if not self.use_gpu or df is None or df.empty:
            return df

        try:
            # Start timing for Prometheus metrics
            start_time = time.time()

            # Select numeric columns for GPU processing
            numeric_cols = df.select_dtypes(
                include=["number"]).columns.tolist()
            if not numeric_cols:
                return df

            # Create a copy to avoid modifying the original
            result = df.copy()

            # Process numeric columns with GPU
            if self.use_gh200:
                # GH200-specific optimizations: process in batches for better memory management
                # Pre-allocate GPU memory for all columns to avoid fragmentation
                gpu_arrays = {}
                for col in numeric_cols:
                    if col in ["close", "open", "high", "low", "volume"]:
                        # Use unified memory for better performance on GH200
                        gpu_arrays[col] = cp.asarray(
                            df[col].values, dtype=cp.float32)

                # Calculate technical indicators using GPU acceleration

                # Simple Moving Averages - use cumsum for faster calculation
                if "close" in gpu_arrays:
                    close_prices = gpu_arrays["close"]

                    # SMA 5
                    window_5 = 5
                    if len(close_prices) > window_5:
                        padded_prices = cp.pad(
                            close_prices, (window_5 - 1, 0), "constant",
                        )
                        cumsum = cp.cumsum(padded_prices)
                        sma_5 = (cumsum[window_5:] -
                                 cumsum[:-window_5]) / window_5
                        # Ensure array length matches by padding with NaN values
                        full_sma_5 = cp.full(len(close_prices), cp.nan)
                        full_sma_5[window_5 - 1:] = sma_5
                        result["sma5"] = cp.asnumpy(full_sma_5)

                    # SMA 10
                    window_10 = 10
                    if len(close_prices) > window_10:
                        padded_prices = cp.pad(
                            close_prices, (window_10 - 1, 0), "constant",
                        )
                        cumsum = cp.cumsum(padded_prices)
                        sma_10 = (cumsum[window_10:] -
                                  cumsum[:-window_10]) / window_10
                        # Ensure array length matches by padding with NaN values
                        full_sma_10 = cp.full(len(close_prices), cp.nan)
                        full_sma_10[window_10 - 1:] = sma_10
                        result["sma10"] = cp.asnumpy(full_sma_10)

                    # SMA 20
                    window_20 = 20
                    if len(close_prices) > window_20:
                        padded_prices = cp.pad(
                            close_prices, (window_20 - 1, 0), "constant",
                        )
                        cumsum = cp.cumsum(padded_prices)
                        sma_20 = (cumsum[window_20:] -
                                  cumsum[:-window_20]) / window_20
                        # Ensure array length matches by padding with NaN values
                        full_sma_20 = cp.full(len(close_prices), cp.nan)
                        full_sma_20[window_20 - 1:] = sma_20
                        result["sma20"] = cp.asnumpy(full_sma_20)

                    # Exponential Moving Averages - optimized implementation
                    # EMA 5
                    if len(close_prices) > 5:
                        alpha = 2.0 / (5 + 1)
                        ema_5 = cp.zeros_like(close_prices)
                        # Initialize with first value
                        ema_5[0] = close_prices[0]
                        for i in range(1, len(close_prices)):
                            ema_5[i] = (
                                alpha * close_prices[i] +
                                (1 - alpha) * ema_5[i - 1]
                            )
                        result["ema5"] = cp.asnumpy(ema_5)

                    # EMA 10
                    if len(close_prices) > 10:
                        alpha = 2.0 / (10 + 1)
                        ema_10 = cp.zeros_like(close_prices)
                        # Initialize with first value
                        ema_10[0] = close_prices[0]
                        for i in range(1, len(close_prices)):
                            ema_10[i] = (
                                alpha * close_prices[i] +
                                (1 - alpha) * ema_10[i - 1]
                            )
                        result["ema10"] = cp.asnumpy(ema_10)

                    # EMA 20
                    if len(close_prices) > 20:
                        alpha = 2.0 / (20 + 1)
                        ema_20 = cp.zeros_like(close_prices)
                        # Initialize with first value
                        ema_20[0] = close_prices[0]
                        for i in range(1, len(close_prices)):
                            ema_20[i] = (
                                alpha * close_prices[i] +
                                (1 - alpha) * ema_20[i - 1]
                            )
                        result["ema20"] = cp.asnumpy(ema_20)

                # Volume weighted average price if both close and volume are available
                if "close" in gpu_arrays and "volume" in gpu_arrays:
                    close_prices = gpu_arrays["close"]
                    volumes = gpu_arrays["volume"]

                    # Use parallel reduction for better performance
                    price_volume = close_prices * volumes
                    total_price_volume = cp.sum(price_volume)
                    total_volume = cp.sum(volumes)
                    vwap = total_price_volume / total_volume if total_volume > 0 else 0
                    result["vwap"] = float(cp.asnumpy(vwap))

                # Calculate RSI with optimized implementation
                if "close" in gpu_arrays and len(close_prices) > 14:
                    # Calculate price changes
                    delta = cp.diff(close_prices)

                    # Separate gains and losses
                    gains = cp.maximum(delta, 0)
                    losses = cp.abs(cp.minimum(delta, 0))

                    # Initialize arrays for average gains and losses
                    avg_gains = cp.zeros_like(delta)
                    avg_losses = cp.zeros_like(delta)

                    # Calculate first averages (simple average for first 14 periods)
                    avg_gains[13] = cp.mean(gains[:14])
                    avg_losses[13] = cp.mean(losses[:14])

                    # Calculate subsequent averages using EMA formula
                    for i in range(14, len(delta)):
                        avg_gains[i] = (avg_gains[i - 1] * 13 + gains[i]) / 14
                        avg_losses[i] = (avg_losses[i - 1] *
                                         13 + losses[i]) / 14

                    # Calculate RS and RSI
                    rs = cp.zeros_like(avg_gains)
                    rsi = cp.zeros_like(avg_gains)

                    # Avoid division by zero
                    nonzero_mask = avg_losses > 0
                    rs[nonzero_mask] = (
                        avg_gains[nonzero_mask] / avg_losses[nonzero_mask]
                    )
                    # When avg_loss is zero, RS is effectively infinite
                    rs[~nonzero_mask] = 100.0

                    # Calculate RSI
                    rsi = 100 - (100 / (1 + rs))

                    # Create full-length array with padding
                    full_rsi = cp.full(len(close_prices), cp.nan)
                    full_rsi[14:] = rsi[13:]  # Offset by 14 periods

                    # Store in result
                    result["rsi"] = cp.asnumpy(full_rsi)

                # Bollinger Bands with optimized implementation
                if "close" in gpu_arrays and len(close_prices) > 20:
                    # Middle band is SMA 20
                    if "sma20" in result.columns:
                        bb_middle = cp.asarray(
                            result["sma20"].values, dtype=cp.float32)
                    else:
                        # Calculate SMA 20 if not already done
                        bb_middle = cp.zeros_like(close_prices)
                        for i in range(20, len(close_prices) + 1):
                            bb_middle[i - 1] = cp.mean(close_prices[i - 20: i])
                        result["bb_middle"] = cp.asnumpy(bb_middle)

                    # Calculate rolling standard deviation
                    bb_std = cp.zeros_like(close_prices)
                    for i in range(20, len(close_prices) + 1):
                        bb_std[i - 1] = cp.std(close_prices[i - 20: i])

                    # Calculate upper and lower bands
                    bb_upper = bb_middle + 2 * bb_std
                    bb_lower = bb_middle - 2 * bb_std

                    # Calculate bandwidth
                    bb_width = cp.zeros_like(close_prices)
                    nonzero_mask = bb_middle != 0
                    bb_width[nonzero_mask] = (
                        bb_upper[nonzero_mask] - bb_lower[nonzero_mask]
                    ) / bb_middle[nonzero_mask]

                    # Store results
                    result["bb_upper"] = cp.asnumpy(bb_upper)
                    result["bb_lower"] = cp.asnumpy(bb_lower)
                    result["bb_width"] = cp.asnumpy(bb_width)

                # MACD calculation
                if "close" in gpu_arrays and len(close_prices) > 26:
                    # Calculate EMA 12 and EMA 26
                    ema12 = cp.zeros_like(close_prices)
                    ema26 = cp.zeros_like(close_prices)

                    # Initialize with first value
                    ema12[0] = close_prices[0]
                    ema26[0] = close_prices[0]

                    # Calculate EMAs
                    alpha12 = 2.0 / (12 + 1)
                    alpha26 = 2.0 / (26 + 1)

                    for i in range(1, len(close_prices)):
                        ema12[i] = (
                            alpha12 * close_prices[i] +
                            (1 - alpha12) * ema12[i - 1]
                        )
                        ema26[i] = (
                            alpha26 * close_prices[i] +
                            (1 - alpha26) * ema26[i - 1]
                        )

                    # Calculate MACD line
                    macd = ema12 - ema26

                    # Calculate signal line (9-day EMA of MACD)
                    signal = cp.zeros_like(macd)
                    signal[0] = macd[0]
                    alpha_signal = 2.0 / (9 + 1)

                    for i in range(1, len(macd)):
                        signal[i] = (
                            alpha_signal * macd[i] +
                            (1 - alpha_signal) * signal[i - 1]
                        )

                    # Calculate histogram
                    histogram = macd - signal

                    # Store results
                    result["macd"] = cp.asnumpy(macd)
                    result["macd_signal"] = cp.asnumpy(signal)
                    result["macd_hist"] = cp.asnumpy(histogram)
            else:
                # Standard GPU processing for non-GH200 GPUs
                for col in numeric_cols:
                    # Transfer to GPU
                    gpu_array = cp.asarray(df[col].values)

                    # Example processing: calculate moving averages
                    if len(gpu_array) > 5:
                        result[f"{col}_sma5"] = cp.asnumpy(
                            cp.convolve(gpu_array, cp.ones(
                                5) / 5, mode="valid"),
                        )

            # Free GPU memory explicitly
            cp.get_default_memory_pool().free_all_blocks()

            # Record GPU memory usage in Prometheus
            if PROMETHEUS_AVAILABLE:
                try:
                    device_id = cp.cuda.Device().id
                    device_props = cp.cuda.runtime.getDeviceProperties(
                        device_id)
                    device_name = device_props["name"].decode()
                    mem_info = cp.cuda.runtime.memGetInfo()
                    mem_used = mem_info[1] - mem_info[0]  # total - free = used
                    GPU_MEMORY_USAGE.labels(device=device_name).set(mem_used)
                except Exception as mem_e:
                    logger.warning(
                        f"Error recording GPU memory usage: {mem_e}")

            # Record processing time in Prometheus
            if PROMETHEUS_AVAILABLE:
                processing_time = time.time() - start_time
                DATA_PROCESSING_TIME.labels(
                    operation="gpu_dataframe_processing", use_gpu="true",
                ).observe(processing_time)

                # Record number of rows processed
                if hasattr(result, "shape") and len(result.shape) > 0:
                    DATA_ROWS_PROCESSED.labels(operation="gpu_processing").inc(
                        result.shape[0],
                    )

            # Return processed DataFrame
            # Send notification about successful GPU processing
            if self.redis and hasattr(result, 'shape') and len(result.shape) > 0 and result.shape[0] > 0:
                self._send_frontend_notification(
                    message=f"GPU-accelerated data processing completed successfully",
                    level="info",
                    category="gpu_processing",
                    details={
                        "rows_processed": result.shape[0],
                        "columns_processed": len(result.columns),
                        "processing_time": time.time() - start_time,
                        "gh200_optimizations": self.use_gh200
                    }
                )

            return result

        except Exception as e:
            logger.exception(f"Error in GPU processing: {e}")
            # Record error in Prometheus
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    api="gpu",
                    endpoint="dataframe_processing",
                    error_type=type(e).__name__,
                ).inc()

            # Send error notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Error in GPU data processing: {str(e)}",
                    level="error",
                    category="gpu_processing",
                    details={
                        "error_type": type(e).__name__,
                        "dataframe_size": len(df) if df is not None and hasattr(df, '__len__') else 0
                    }
                )

            return df

    def _process_with_tensorflow(self, df, batch_size=1000):
        """
        Process DataFrame with TensorFlow GPU acceleration

        Args:
            df: Input DataFrame
            batch_size: Batch size for processing

        Returns:
            Processed DataFrame
        """
        if (
            not TF_AVAILABLE
            or not self.use_gpu
            or df is None
            or df.empty
            or not hasattr(self, "tf_utils")
        ):
            return df

        try:
            # Select numeric columns for processing
            numeric_cols = df.select_dtypes(
                include=["number"]).columns.tolist()
            if not numeric_cols:
                return df

            # Create a copy to avoid modifying the original
            result = df.copy()

            # Process with TensorFlow on GPU
            with tf.device("/GPU:0"):
                # Convert key columns to tensors
                tensors = {}
                for col in numeric_cols:
                    if col in ["close", "open", "high", "low", "volume"]:
                        tensors[col] = tf.convert_to_tensor(
                            df[col].values, dtype=tf.float32,
                        )

                # Process in batches if data is large
                if len(df) > batch_size and "close" in tensors:
                    # Calculate momentum features
                    close_tensor = tensors["close"]

                    # Calculate momentum (percent change)
                    mom1 = tf.concat(
                        [
                            tf.zeros(1),
                            (close_tensor[1:] - close_tensor[:-1]
                             ) / close_tensor[:-1],
                        ],
                        axis=0,
                    )
                    result["tf_mom1"] = mom1.numpy()

                    # Calculate 5-day momentum
                    if len(close_tensor) >= 5:
                        mom5 = tf.concat(
                            [
                                tf.zeros(5),
                                (close_tensor[5:] - close_tensor[:-5])
                                / close_tensor[:-5],
                            ],
                            axis=0,
                        )
                        result["tf_mom5"] = mom5.numpy()

                    # Calculate 10-day momentum
                    if len(close_tensor) >= 10:
                        mom10 = tf.concat(
                            [
                                tf.zeros(10),
                                (close_tensor[10:] - close_tensor[:-10])
                                / close_tensor[:-10],
                            ],
                            axis=0,
                        )
                        result["tf_mom10"] = mom10.numpy()

                    # Calculate volatility (standard deviation)
                    if len(close_tensor) >= 10:
                        volatility = []
                        for i in range(len(close_tensor)):
                            if i < 10:
                                volatility.append(0.0)
                            else:
                                window = close_tensor[i - 10: i]
                                std = tf.math.reduce_std(window)
                                mean = tf.math.reduce_mean(window)
                                vol = std / mean if mean != 0 else 0
                                volatility.append(vol.numpy())
                        result["tf_volatility"] = volatility

                # If we have OHLC data, calculate candlestick patterns
                if all(col in tensors for col in ["open", "high", "low", "close"]):
                    open_tensor = tensors["open"]
                    high_tensor = tensors["high"]
                    low_tensor = tensors["low"]
                    close_tensor = tensors["close"]

                    # Calculate candlestick body size
                    body_size = tf.abs(close_tensor - open_tensor)
                    result["body_size"] = body_size.numpy()

                    # Calculate upper shadow
                    upper_shadow = tf.where(
                        close_tensor > open_tensor,
                        high_tensor - close_tensor,
                        high_tensor - open_tensor,
                    )
                    result["upper_shadow"] = upper_shadow.numpy()

                    # Calculate lower shadow
                    lower_shadow = tf.where(
                        close_tensor > open_tensor,
                        open_tensor - low_tensor,
                        close_tensor - low_tensor,
                    )
                    result["lower_shadow"] = lower_shadow.numpy()

                    # Identify potential doji patterns (small body relative to shadows)
                    total_range = high_tensor - low_tensor
                    doji_condition = body_size < (0.1 * total_range)
                    result["is_doji"] = doji_condition.numpy().astype(int)

                    # Identify potential hammer patterns (small body, long lower shadow)
                    hammer_condition = tf.logical_and(
                        body_size < (
                            0.3 * total_range), lower_shadow > (2 * body_size),
                    )
                    result["is_hammer"] = hammer_condition.numpy().astype(int)

            return result

        except Exception as e:
            logger.exception(f"Error in TensorFlow processing: {e}")
            return df

    # ===== DATA LOADING METHODS =====

    @retry(
        stop_max_attempt_number=3,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def load_price_data(self, tickers, start_date, end_date, timeframe="1m"):
        """
        Load historical price data for specified tickers

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe ('1m', '5m', '1h', '1d')

        Returns:
            Dictionary of ticker -> DataFrame with OHLCV data
        """
        try:
            logger.info(
                f"Loading price data for {len(tickers)} tickers from {start_date} to {end_date}",
            )

            # If in test mode, return mock data
            if self.test_mode:
                logger.info("Using mock data for testing")
                return self._generate_mock_price_data(
                    tickers, start_date, end_date, timeframe,
                )

            # Format dates
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Determine multiplier and timespan for Polygon API
            if timeframe == "1m":
                multiplier = 1
                timespan = "minute"
            elif timeframe == "5m":
                multiplier = 5
                timespan = "minute"
            elif timeframe == "1h":
                multiplier = 1
                timespan = "hour"
            elif timeframe == "1d":
                multiplier = 1
                timespan = "day"
            else:
                msg = f"Unsupported timeframe: {timeframe}"
                raise ValueError(msg)

            # Load data for each ticker
            results = {}

            for ticker in tickers:
                try:
                    # Check cache first
                    cache_key = f"{ticker}_{timeframe}_{start_str}_{end_str}"
                    cache_path = os.path.join(
                        self.config["cache_dir"], f"{cache_key}.pkl",
                    )

                    # Try to load from cache
                    cached_data = self._load_from_cache(cache_path)
                    if cached_data is not None:
                        # Convert timestamp to datetime if needed
                        if (
                            "timestamp" in cached_data.columns
                            and not pd.api.types.is_datetime64_dtype(
                                cached_data["timestamp"],
                            )
                        ):
                            cached_data["timestamp"] = pd.to_datetime(
                                cached_data["timestamp"],
                            )

                        results[ticker] = cached_data
                        logger.debug(
                            f"Loaded {ticker} data from cache: {len(cached_data)} rows",
                        )
                    else:
                        # Fetch from API
                        async_result = self._run_async(
                            self.polygon.get_aggregates(
                                ticker=ticker,
                                multiplier=multiplier,
                                timespan=timespan,
                                from_date=start_str,
                                to_date=end_str,
                                limit=50000,
                            ),
                        )

                        # Process API response
                        df = self._process_polygon_response(
                            async_result, ticker)
                        if df is None or df.empty:
                            logger.warning(f"No data returned for {ticker}")
                            continue

                        # Save to cache
                        self._save_to_cache(df, cache_path)

                        # Process with GPU if available
                        if self.use_gpu:
                            try:
                                processed_df = self._process_dataframe_with_gpu(
                                    df)
                                results[ticker] = processed_df
                                logger.debug(
                                    f"Processed {ticker} data with GPU")
                            except Exception as e:
                                logger.warning(
                                    f"GPU processing failed for {ticker}: {e}",
                                )
                                results[ticker] = df
                        else:
                            results[ticker] = df

                        logger.debug(
                            f"Fetched {ticker} data from API: {len(df)} rows")

                    # Rate limiting
                    time.sleep(1.0 / self.config["rate_limit"]["polygon"])

                except Exception as e:
                    logger.exception(
                        f"Error loading price data for {ticker}: {e!s}")

            logger.info(f"Loaded price data for {len(results)} tickers")

            # Send notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Successfully loaded price data for {len(results)} tickers",
                    level="success",
                    category="data_loading",
                    details={
                        "tickers_loaded": list(results.keys()),
                        "timeframe": timeframe,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "total_rows": sum(len(df) for df in results.values() if isinstance(df, pd.DataFrame))
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error loading price data: {e!s}", exc_info=True)

            # Send error notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Error loading price data: {str(e)}",
                    level="error",
                    category="data_loading",
                    details={
                        "tickers_requested": tickers[:10] + (["..."] if len(tickers) > 10 else []),
                        "timeframe": timeframe,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "error_type": type(e).__name__
                    }
                )

            return {}

    @retry(
        stop_max_attempt_number=3,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def load_options_data(self, tickers, start_date, end_date):
        """
        Load options data for specified tickers

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary of ticker -> options data
        """
        try:
            logger.info(
                f"Loading options data for {len(tickers)} tickers from {start_date} to {end_date}",
            )

            # If in test mode, return mock data
            if self.test_mode:
                logger.info("Using mock options data for testing")
                return self._generate_mock_options_data(tickers, start_date, end_date)

            # Format dates
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Load data for each ticker
            results = {}

            for ticker in tickers:
                try:
                    # Check cache first
                    cache_key = f"{ticker}_options_{start_str}_{end_str}"
                    cache_path = os.path.join(
                        self.config["cache_dir"], f"{cache_key}.pkl",
                    )

                    # Try to load from cache
                    cached_data = self._load_from_cache(cache_path)
                    if cached_data is not None:
                        # Convert timestamp to datetime if needed
                        if (
                            "timestamp" in cached_data.columns
                            and not pd.api.types.is_datetime64_dtype(
                                cached_data["timestamp"],
                            )
                        ):
                            cached_data["timestamp"] = pd.to_datetime(
                                cached_data["timestamp"],
                            )

                        results[ticker] = cached_data
                        logger.debug(
                            f"Loaded {ticker} options data from cache: {len(cached_data)} rows",
                        )
                    else:
                        # Fetch from Unusual Whales API
                        async_result = self._run_async(
                            self.unusual_whales.get_flow_alerts(
                                ticker=ticker, limit=1000,
                            ),
                        )

                        # Process API response
                        df = self._process_unusual_whales_response(
                            async_result, ticker)
                        if df is None or df.empty:
                            logger.warning(f"No options data for {ticker}")
                            continue

                        # Save to cache
                        self._save_to_cache(df, cache_path)

                        results[ticker] = df
                        logger.debug(
                            f"Fetched {ticker} options data: {len(df)} rows")

                    # Rate limiting
                    time.sleep(
                        1.0 / self.config["rate_limit"]["unusual_whales"])

                except Exception as e:
                    logger.exception(
                        f"Error loading options data for {ticker}: {e!s}")

            logger.info(f"Loaded options data for {len(results)} tickers")

            # Send notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Successfully loaded options data for {len(results)} tickers",
                    level="success",
                    category="data_loading",
                    details={
                        "tickers_loaded": list(results.keys()),
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "total_rows": sum(len(df) for df in results.values() if isinstance(df, pd.DataFrame))
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error loading options data: {e!s}", exc_info=True)

            # Send error notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Error loading options data: {str(e)}",
                    level="error",
                    category="data_loading",
                    details={
                        "tickers_requested": tickers[:10] + (["..."] if len(tickers) > 10 else []),
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "error_type": type(e).__name__
                    }
                )

            return {}

    @retry(
        stop_max_attempt_number=3,
        wait_exponential_multiplier=1000,
        wait_exponential_max=10000,
    )
    def load_market_data(self, start_date, end_date, symbols=None):
        """
        Load market data for specified symbols

        Args:
            start_date: Start date
            end_date: End date
            symbols: List of market symbols to load

        Returns:
            DataFrame with market data
        """
        try:
            # Use default market symbols if none provided
            if symbols is None:
                # Use market index tickers from configuration
                market_symbols = [
                    ticker
                    for ticker in config["stock_selection"]["universe"][
                        "default_tickers"
                    ]
                    if ticker in ["SPY", "QQQ", "IWM", "DIA"]
                ]
                # Fallback to SPY and QQQ if no market indices in default tickers
                symbols = market_symbols if market_symbols else ["SPY", "QQQ"]
                logger.info(f"Using default market symbols: {symbols}")

            logger.info(
                f"Loading market data for {symbols} from {start_date} to {end_date}",
            )

            # If in test mode, return mock data
            if self.test_mode:
                logger.info("Using mock market data for testing")
                return self._generate_mock_market_data(start_date, end_date, symbols)

            # Format dates
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Check cache first
            cache_key = f"market_data_{','.join(symbols)}_{start_str}_{end_str}"
            cache_path = os.path.join(
                self.config["cache_dir"], f"{cache_key}.pkl")

            # Try to load from cache
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                # Convert timestamp to datetime if needed
                if (
                    "timestamp" in cached_data.columns
                    and not pd.api.types.is_datetime64_dtype(cached_data["timestamp"])
                ):
                    cached_data["timestamp"] = pd.to_datetime(
                        cached_data["timestamp"])

                logger.debug(
                    f"Loaded market data from cache: {len(cached_data)} rows")
                return cached_data
            # Load data for each symbol
            dfs = []

            for symbol in symbols:
                try:
                    # Fetch from Polygon API
                    async_result = self._run_async(
                        self.polygon.get_aggregates(
                            ticker=symbol,
                            multiplier=1,
                            timespan="minute",
                            from_date=start_str,
                            to_date=end_str,
                            limit=50000,
                        ),
                    )

                    # Process API response with symbol-specific column prefixes
                    df = self._process_polygon_response(
                        async_result, symbol, prefix=symbol.lower(),
                    )
                    if df is None or df.empty:
                        logger.warning(f"No data returned for {symbol}")
                        continue

                    # Add to list
                    dfs.append(df)
                    logger.debug(
                        f"Fetched {symbol} market data: {len(df)} rows")

                    # Rate limiting
                    time.sleep(1.0 / self.config["rate_limit"]["polygon"])

                except Exception as e:
                    logger.exception(
                        f"Error loading market data for {symbol}: {e!s}",
                    )

            # Merge DataFrames
            if not dfs:
                logger.warning("No market data fetched")
                return pd.DataFrame()

            # Ensure all dataframes have timestamp column
            for i in range(len(dfs)):
                if (
                    "timestamp" not in dfs[i].columns
                    and dfs[i].index.name == "timestamp"
                ):
                    dfs[i] = dfs[i].reset_index()

            # Merge on timestamp (nearest match)
            result = dfs[0].copy()
            for df in dfs[1:]:
                result = pd.merge_asof(
                    result,
                    df,
                    on="timestamp",
                    direction="nearest",
                    suffixes=(
                        "",
                        (
                            f'_{df.iloc[0].get("ticker", "market")}'
                            if "ticker" in df.columns
                            else "_market"
                        ),
                    ),
                )

            # Add market metrics
            for symbol in symbols:
                symbol_lower = symbol.lower()

                # Daily change
                if f"{symbol_lower}_close" in result.columns:
                    result[f"{symbol_lower}_change"] = result[
                        f"{symbol_lower}_close"
                    ].pct_change()

                    # Calculate day-over-day change
                    result[f"{symbol_lower}_daily_change"] = result[
                        f"{symbol_lower}_close"
                    ].pct_change(
                        periods=390,
                    )  # ~1 trading day in minutes

            # Save to cache
            self._save_to_cache(result, cache_path)

            logger.info(f"Loaded market data: {len(result)} rows")

            # Send notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Successfully loaded market data with {len(result)} rows",
                    level="success",
                    category="data_loading",
                    details={
                        "symbols": symbols,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "row_count": len(result),
                        "column_count": len(result.columns) if not result.empty else 0
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error loading market data: {e!s}", exc_info=True)

            # Send error notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Error loading market data: {str(e)}",
                    level="error",
                    category="data_loading",
                    details={
                        "symbols": symbols,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "error_type": type(e).__name__
                    }
                )

            return pd.DataFrame()

    # ===== DATA PROCESSING METHODS =====

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for price data

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Group by ticker if multiple tickers
            if "ticker" in df.columns:
                grouped = df.groupby("ticker")
                result_dfs = []

                for ticker, group_df in grouped:
                    result_df = self._calculate_indicators_for_group(group_df)
                    result_df["ticker"] = ticker
                    result_dfs.append(result_df)

                return pd.concat(result_dfs)
            return self._calculate_indicators_for_group(df)

        except Exception as e:
            logger.error(
                f"Error calculating technical indicators: {e!s}", exc_info=True,
            )
            return df

    def _calculate_indicators_for_group(self, df):
        """
        Calculate indicators for a single ticker dataframe

        Args:
            df: DataFrame for a single ticker

        Returns:
            DataFrame with technical indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Simple Moving Averages
        result["sma5"] = result["close"].rolling(window=5).mean()
        result["sma10"] = result["close"].rolling(window=10).mean()
        result["sma20"] = result["close"].rolling(window=20).mean()

        # Exponential Moving Averages
        result["ema5"] = result["close"].ewm(span=5, adjust=False).mean()
        result["ema10"] = result["close"].ewm(span=10, adjust=False).mean()
        result["ema20"] = result["close"].ewm(span=20, adjust=False).mean()

        # MACD
        result["ema12"] = result["close"].ewm(span=12, adjust=False).mean()
        result["ema26"] = result["close"].ewm(span=26, adjust=False).mean()
        result["macd"] = result["ema12"] - result["ema26"]
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]

        # Relative strength to price
        result["price_rel_sma5"] = result["close"] / result["sma5"] - 1
        result["price_rel_sma10"] = result["close"] / result["sma10"] - 1
        result["price_rel_sma20"] = result["close"] / result["sma20"] - 1

        # Momentum
        result["mom1"] = result["close"].pct_change(1)
        result["mom5"] = result["close"].pct_change(5)
        result["mom10"] = result["close"].pct_change(10)

        # Volatility
        result["volatility"] = (
            result["close"].rolling(window=10).std()
            / result["close"].rolling(window=10).mean()
        )

        # Volume-based indicators
        if "volume" in result.columns:
            result["volume_sma5"] = result["volume"].rolling(window=5).mean()
            result["volume_ratio"] = result["volume"] / result["volume_sma5"]

            # Money Flow Index (simplified)
            result["money_flow"] = result["close"] * result["volume"]
            result["money_flow_sma"] = result["money_flow"].rolling(
                window=14).mean()

        # RSI (14)
        delta = result["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Handle division by zero
        rs = gain / loss.replace(0, 1e-9)
        result["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        result["bb_middle"] = result["close"].rolling(window=20).mean()
        result["bb_std"] = result["close"].rolling(window=20).std()
        result["bb_upper"] = result["bb_middle"] + 2 * result["bb_std"]
        result["bb_lower"] = result["bb_middle"] - 2 * result["bb_std"]
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result[
            "bb_middle"
        ]

        # Handle any remaining infinity values
        for col in result.select_dtypes(include=[np.number]).columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            result[col] = result[col].ffill().bfill().fillna(0)

        return result

    def generate_targets(self, df):
        """
        Generate target variables for supervised learning

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added target variables
        """
        try:
            # Make a copy
            result = df.copy()

            # Signal detection target (1 if price increases by 1% within next 10 bars)
            future_returns = (
                result.groupby("ticker")["close"].pct_change(10).shift(-10)
                if "ticker" in result.columns
                else result["close"].pct_change(10).shift(-10)
            )

            # Use a lower threshold for more balanced classes
            result["signal_target"] = (future_returns > 0.001).astype(int)

            # Check if we have both classes represented
            if result["signal_target"].nunique() < 2:
                # If not, force some samples to be positive to ensure balanced classes
                logger.warning(
                    "Only one class detected in targets, creating synthetic samples",
                )
                # Set the top 20% of future returns as positive signals
                positive_count = max(int(len(result) * 0.2), 100)
                top_indices = future_returns.nlargest(positive_count).index
                result.loc[top_indices, "signal_target"] = 1
                logger.info(
                    f"Created {positive_count} synthetic positive samples")

            # Price prediction targets
            if "ticker" in result.columns:
                result["future_return_5min"] = (
                    result.groupby("ticker")["close"].pct_change(5).shift(-5)
                )
                result["future_return_10min"] = (
                    result.groupby("ticker")["close"].pct_change(10).shift(-10)
                )
                result["future_return_30min"] = (
                    result.groupby("ticker")["close"].pct_change(30).shift(-30)
                )
            else:
                result["future_return_5min"] = result["close"].pct_change(
                    5).shift(-5)
                result["future_return_10min"] = (
                    result["close"].pct_change(10).shift(-10)
                )
                result["future_return_30min"] = (
                    result["close"].pct_change(30).shift(-30)
                )

            # Direction target (1 for up, 0 for down)
            result["future_direction"] = (
                result["future_return_10min"] > 0).astype(int)

            # Risk assessment target (ATR as % of price)
            high_low = result["high"] - result["low"]
            high_close = abs(result["high"] - result["close"].shift())
            low_close = abs(result["low"] - result["close"].shift())
            tr = pd.concat([high_low, high_close, low_close],
                           axis=1).max(axis=1)
            result["atr14"] = tr.rolling(14).mean()
            result["atr_pct"] = result["atr14"] / result["close"]

            # Handle any NaN or Inf values in targets
            for col in [
                "signal_target",
                "future_return_5min",
                "future_return_10min",
                "future_return_30min",
                "future_direction",
                "atr_pct",
            ]:
                if col in result.columns:
                    result[col] = result[col].replace(
                        [np.inf, -np.inf], np.nan)
                    result[col] = result[col].ffill().bfill().fillna(0)

            # Exit strategy target (optimal exit time within next 30 bars)
            for _ticker, group in (
                result.groupby("ticker")
                if "ticker" in result.columns
                else [("", result)]
            ):
                future_prices = [
                    group["close"].shift(-i) for i in range(1, 31)]
                future_prices_df = pd.concat(future_prices, axis=1)
                max_price = future_prices_df.max(axis=1)
                optimal_exit = max_price / group["close"] - 1
                result.loc[group.index, "optimal_exit"] = optimal_exit

            # Handle NaN in optimal_exit
            if "optimal_exit" in result.columns:
                result["optimal_exit"] = result["optimal_exit"].replace(
                    [np.inf, -np.inf], np.nan,
                )
                result["optimal_exit"] = (
                    result["optimal_exit"].ffill().bfill().fillna(0)
                )

            return result

        except Exception as e:
            logger.error(f"Error generating targets: {e!s}", exc_info=True)
            return df

    def prepare_training_data(self, price_data, options_data=None, market_data=None):
        """
        Prepare combined data for model training

        Args:
            price_data: Dictionary of ticker -> price DataFrame
            options_data: Dictionary of ticker -> options DataFrame
            market_data: DataFrame with market data

        Returns:
            DataFrame with combined data ready for training
        """
        try:
            # Check if we have valid price data
            if not price_data or (
                isinstance(price_data, dict) and len(price_data) == 0
            ):
                logger.error("No price data available for training")
                return pd.DataFrame()

            # Create master dataframe from price data
            dfs = []
            if isinstance(price_data, dict):
                for ticker, df in price_data.items():
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["ticker"] = ticker
                        dfs.append(df)
            elif isinstance(price_data, pd.DataFrame) and not price_data.empty:
                dfs.append(price_data)

            # Check if we have any valid dataframes to concatenate
            if not dfs:
                logger.error("No valid price dataframes to concatenate")
                return pd.DataFrame()

            # Concatenate the dataframes
            combined_price = pd.concat(dfs, ignore_index=True)

            # Ensure timestamp index
            if "timestamp" in combined_price.columns:
                combined_price = combined_price.set_index("timestamp")

            # Calculate technical indicators
            combined_price = self.calculate_technical_indicators(
                combined_price)

            # Merge options data if available
            if options_data is not None:
                if isinstance(options_data, dict):
                    # Convert dict of dataframes to a single dataframe
                    options_dfs = []
                    for ticker, data in options_data.items():
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            ticker_df = data.copy()
                            if "ticker" not in ticker_df.columns:
                                ticker_df["ticker"] = ticker
                            options_dfs.append(ticker_df)

                    if options_dfs:
                        options_df = pd.concat(options_dfs, ignore_index=True)
                    else:
                        logger.warning(
                            "No valid options dataframes to concatenate")
                        options_df = pd.DataFrame()
                else:
                    options_df = options_data

                if not options_df.empty:
                    # Ensure datetime index for merging
                    if "timestamp" in options_df.columns:
                        options_df = options_df.set_index("timestamp")

                # Merge on timestamp and ticker
                if not options_df.empty:
                    combined_price_reset = combined_price.reset_index().copy()
                    combined_price_reset["timestamp"] = pd.to_datetime(
                        combined_price_reset["timestamp"],
                    ).astype("datetime64[ns]")
                    options_df_reset = options_df.reset_index().copy()
                    options_df_reset["timestamp"] = pd.to_datetime(
                        options_df_reset["timestamp"],
                    ).astype("datetime64[ns]")

                    combined_data = pd.merge_asof(
                        combined_price_reset,
                        options_df_reset,
                        on="timestamp",
                        by="ticker",
                        direction="backward",
                        tolerance=pd.Timedelta("1h"),
                        suffixes=("", "_options"),
                    )
                else:
                    combined_data = combined_price.reset_index()
            else:
                combined_data = combined_price.reset_index()

            # Merge market data
            if market_data is not None and not market_data.empty:
                # Ensure datetime index
                market_data_reset = (
                    market_data.reset_index().copy()
                    if market_data.index.name == "timestamp"
                    else market_data.copy()
                )

                # Ensure timestamp dtypes match
                combined_data["timestamp"] = pd.to_datetime(
                    combined_data["timestamp"],
                ).astype("datetime64[ns]")
                market_data_reset["timestamp"] = pd.to_datetime(
                    market_data_reset["timestamp"],
                ).astype("datetime64[ns]")

                # Merge on timestamp
                combined_data = pd.merge_asof(
                    combined_data,
                    market_data_reset,
                    on="timestamp",
                    direction="backward",
                    suffixes=("", "_market"),
                )

            # Generate target variables
            combined_data = self.generate_targets(combined_data)

            # Drop rows with missing values in critical columns
            critical_columns = ["close", "high", "low", "volume", "timestamp"]
            combined_data = combined_data.dropna(subset=critical_columns)

            # Clean up extreme values
            numeric_cols = combined_data.select_dtypes(
                include=[np.number]).columns
            for col in numeric_cols:
                if col in combined_data.columns:
                    q1, q3 = combined_data[col].quantile([0.01, 0.99])
                    iqr = q3 - q1
                    combined_data.loc[:, col] = combined_data[col].clip(
                        q1 - 3 * iqr, q3 + 3 * iqr,
                    )

            # Send notification to frontend
            if self.redis and not combined_data.empty:
                tickers = combined_data['ticker'].unique().tolist(
                ) if 'ticker' in combined_data.columns else []
                self._send_frontend_notification(
                    message=f"Training data prepared successfully with {len(combined_data)} samples",
                    level="success",
                    category="data_processing",
                    details={
                        "sample_count": len(combined_data),
                        "feature_count": len(combined_data.columns),
                        "tickers": tickers[:10] + (["..."] if len(tickers) > 10 else []),
                        "has_price_data": price_data is not None and (isinstance(price_data, dict) and len(price_data) > 0 or isinstance(price_data, pd.DataFrame) and not price_data.empty),
                        "has_options_data": options_data is not None,
                        "has_market_data": market_data is not None and not (isinstance(market_data, pd.DataFrame) and market_data.empty)
                    }
                )

            return combined_data

        except Exception as e:
            logger.error(
                f"Error preparing training data: {e!s}", exc_info=True)

            # Send error notification to frontend
            if self.redis:
                self._send_frontend_notification(
                    message=f"Error preparing training data: {str(e)}",
                    level="error",
                    category="data_processing",
                    details={
                        "error_type": type(e).__name__,
                        "has_price_data": price_data is not None and (isinstance(price_data, dict) and len(price_data) > 0 or isinstance(price_data, pd.DataFrame) and not price_data.empty),
                        "has_options_data": options_data is not None,
                        "has_market_data": market_data is not None and not (isinstance(market_data, pd.DataFrame) and market_data.empty)
                    }
                )

            return pd.DataFrame()

    # ===== ML PREPARATION METHODS =====

    def prepare_signal_detection_data(self, data):
        """
        Prepare data for signal detection model

        Args:
            data: Combined DataFrame with features

        Returns:
            X: Features DataFrame
            y: Target Series
        """
        try:
            # Select features
            feature_columns = [
                # Price-based features
                "close",
                "open",
                "high",
                "low",
                "volume",
                # Technical indicators
                "sma5",
                "sma10",
                "sma20",
                "ema5",
                "ema10",
                "ema20",
                "macd",
                "macd_signal",
                "macd_hist",
                "price_rel_sma5",
                "price_rel_sma10",
                "price_rel_sma20",
                "mom1",
                "mom5",
                "mom10",
                "volatility",
                "volume_ratio",
                "rsi",
                "bb_width",
                # Market features (if available)
                "spy_close",
                "vix_close",
                "spy_change",
                "vix_change",
                # Options features (if available)
                "put_call_ratio",
                "implied_volatility",
                "option_volume",
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 5:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return pd.DataFrame(), pd.Series()

            # Select data
            X = data[available_columns].copy()
            y = data["signal_target"].copy()

            # Drop rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            # Handle any remaining infinity values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())

            logger.info(
                f"Prepared signal detection data with {len(X)} samples and {len(available_columns)} features",
            )

            return X, y

        except Exception as e:
            logger.error(
                f"Error preparing signal detection data: {e!s}", exc_info=True,
            )
            return pd.DataFrame(), pd.Series()

    def prepare_price_prediction_data(self, data):
        """
        Prepare data for price prediction model (LSTM)

        Args:
            data: Combined DataFrame with features

        Returns:
            X: 3D numpy array of sequences
            y: 2D numpy array of targets
        """
        try:
            # Select features
            feature_columns = [
                # Price-based features
                "close",
                "high",
                "low",
                "volume",
                # Technical indicators
                "price_rel_sma5",
                "price_rel_sma10",
                "price_rel_sma20",
                "macd",
                "rsi",
                "volatility",
                # Market features (if available)
                "spy_close",
                "vix_close",
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 4:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return np.array([]), np.array([])

            # Target columns
            target_columns = [
                "future_return_5min",
                "future_return_10min",
                "future_return_30min",
            ]
            available_targets = [
                col for col in target_columns if col in data.columns]

            if len(available_targets) == 0:
                logger.warning("No target variables available")
                return np.array([]), np.array([])

            # Group by ticker to create sequences
            sequences = []
            targets = []

            for _ticker, group in data.groupby("ticker"):
                # Sort by timestamp
                group = group.sort_index()

                # Select features and targets
                X = group[available_columns].values
                y = group[available_targets].values

                # Create sequences (lookback of 20 intervals)
                for i in range(20, len(X)):
                    sequences.append(X[i - 20: i])
                    targets.append(y[i])

            # Convert to numpy arrays
            X_array = np.array(sequences)
            y_array = np.array(targets)

            # More robust handling of NaN or infinite values
            if (
                np.isnan(X_array).any()
                or np.isinf(X_array).any()
                or np.isnan(y_array).any()
                or np.isinf(y_array).any()
            ):
                logger.warning(
                    "NaN or infinite values detected. Performing robust cleaning...",
                )

                # First, identify rows with NaN or inf in either X or y
                X_has_invalid = np.any(
                    np.isnan(X_array) | np.isinf(X_array), axis=(1, 2),
                )
                y_has_invalid = np.any(
                    np.isnan(y_array) | np.isinf(y_array), axis=1)
                valid_indices = ~(X_has_invalid | y_has_invalid)

                # If we have enough valid data, filter out invalid rows
                if np.sum(valid_indices) > 100:
                    logger.info(
                        f"Filtering out {np.sum(~valid_indices)} invalid rows, keeping {np.sum(valid_indices)}",
                    )
                    X_array = X_array[valid_indices]
                    y_array = y_array[valid_indices]
                else:
                    # If not enough valid data, replace NaN and inf with zeros/means
                    logger.warning(
                        "Not enough valid rows, replacing NaN values instead of filtering",
                    )
                    X_array = np.nan_to_num(
                        X_array, nan=0.0, posinf=0.0, neginf=0.0)
                    y_array = np.nan_to_num(
                        y_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            scaler = MinMaxScaler()
            n_samples, n_timesteps, n_features = X_array.shape
            X_reshaped = X_array.reshape(n_samples * n_timesteps, n_features)
            X_scaled = scaler.fit_transform(X_reshaped)
            X_array = X_scaled.reshape(n_samples, n_timesteps, n_features)

            # Store scaler in memory
            self.price_prediction_scaler = scaler

            logger.info(
                f"Prepared price prediction data with {len(sequences)} sequences",
            )

            return X_array, y_array

        except Exception as e:
            logger.error(
                f"Error preparing price prediction data: {e!s}", exc_info=True,
            )
            return np.array([]), np.array([])

    # ===== MARKET DATA HELPER METHODS =====

    async def get_all_active_tickers(self):
        """
        Get all active tickers from Polygon API dynamically

        Returns:
            List of ticker objects
        """
        try:
            if self.test_mode:
                logger.info(
                    "Using default tickers from configuration for testing")
                return [
                    {"ticker": t}
                    for t in config["stock_selection"]["universe"]["default_tickers"]
                ]

            # Use the v3 reference/tickers endpoint
            endpoint = "v3/reference/tickers"
            params = {"market": "stocks", "active": "true", "limit": 1000}

            response = await self.polygon._make_request(endpoint, params)

            # Process the response based on the actual API structure
            if isinstance(response, dict) and "results" in response:
                return response["results"]
            logger.warning(
                f"Unexpected response format from Polygon API: {type(response)}",
            )
            return []
        except Exception as e:
            logger.exception(f"Error getting active tickers: {e!s}")
            return []

    async def get_previous_day_data(self, ticker):
        """
        Get previous day's trading data

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with previous day data or None
        """
        try:
            # Check Redis cache first
            if self.redis:
                cached_data = self.redis.hgetall(
                    f"stock:{ticker}:previous_day")
                if cached_data:
                    return {
                        "close": float(cached_data.get(b"close", 0)),
                        "volume": int(cached_data.get(b"volume", 0)),
                        "open": float(cached_data.get(b"open", 0)),
                        "high": float(cached_data.get(b"high", 0)),
                        "low": float(cached_data.get(b"low", 0)),
                    }

            # Get data from Polygon API
            response = await self.polygon.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_date=(
                    datetime.datetime.now() - datetime.timedelta(days=5)
                ).strftime("%Y-%m-%d"),
                to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                limit=5,
            )

            # Process the response
            if isinstance(response, pd.DataFrame) and not response.empty:
                # Extract the most recent day's data
                latest_data = response.iloc[0]
                result = {
                    "close": float(latest_data.get("close", 0)),
                    "volume": int(latest_data.get("volume", 0)),
                    "open": float(latest_data.get("open", 0)),
                    "high": float(latest_data.get("high", 0)),
                    "low": float(latest_data.get("low", 0)),
                }

                # Cache in Redis
                if self.redis:
                    self.redis.hmset(
                        f"stock:{ticker}:previous_day",
                        {
                            "close": result["close"],
                            "volume": result["volume"],
                            "open": result["open"],
                            "high": result["high"],
                            "low": result["low"],
                        },
                    )
                    # Expire after 1 hour
                    self.redis.expire(f"stock:{ticker}:previous_day", 3600)

                return result
            logger.warning(f"No data returned for {ticker}")
            return None
        except Exception as e:
            logger.exception(
                f"Error getting previous day data for {ticker}: {e!s}")
            return None

    async def check_options_availability(self, ticker):
        """
        Check if options are available for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Boolean indicating if options are available
        """
        try:
            # Check Redis cache first
            if self.redis:
                cached_result = self.redis.get(
                    f"options_availability:{ticker}")
                if cached_result is not None:
                    return cached_result == b"1"

            # Use Unusual Whales client to check for options data
            options_data = await self.unusual_whales.get_flow_alerts(ticker, limit=1)

            # Determine if options are available
            has_options = len(options_data) > 0

            # Cache the result
            if self.redis:
                self.redis.setex(
                    f"options_availability:{ticker}", 3600, "1" if has_options else "0",
                )

            return has_options
        except Exception as e:
            logger.exception(
                f"Error checking options availability for {ticker}: {e!s}")
            # Return False if there's an error
            return False

    async def get_pre_market_movers(self):
        """
        Get pre-market movers

        Returns:
            Set of ticker symbols with significant pre-market movement
        """
        try:
            if not self.redis:
                return set()

            # Use Redis to get pre-market data from websocket client
            pre_market_movers = set()
            pre_market_keys = self.redis.keys("stock:*:pre_market")

            for key in pre_market_keys:
                ticker = key.decode("utf-8").split(":")[1]
                data = self.redis.hgetall(key)

                # Check for significant movement (e.g., > 2%)
                if b"percent_change" in data:
                    percent_change = float(data[b"percent_change"])
                    if abs(percent_change) > 2.0:
                        pre_market_movers.add(ticker)

            return pre_market_movers
        except Exception as e:
            logger.exception(f"Error getting pre-market movers: {e!s}")
            return set()

    async def get_unusual_options_activity(self):
        """
        Get stocks with unusual options activity

        Returns:
            Set of ticker symbols with unusual options activity
        """
        try:
            # Get data from Unusual Whales
            data = await self.unusual_whales.get_alerts(limit=100)

            if isinstance(data, list) and data:
                # Extract tickers
                return {item.get("ticker") for item in data if "ticker" in item}
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Handle DataFrame format if that's what the client returns
                return set(data["ticker"].unique())

            return set()

        except Exception as e:
            logger.exception(f"Error getting unusual options activity: {e!s}")
            return set()

    async def get_technical_setups(self):
        """
        Get stocks with technical setups

        Returns:
            Set of ticker symbols with technical patterns
        """
        try:
            if not self.redis:
                return set()

            # Use Redis data stored by API clients
            technical_setups = set()
            technical_keys = self.redis.keys("stock:*:technical")

            for key in technical_keys:
                ticker = key.decode("utf-8").split(":")[1]
                data = self.redis.hgetall(key)

                # Check for technical setups based on indicators
                # Golden cross (SMA 5 crossing above SMA 20)
                if b"sma_5" in data and b"sma_20" in data:
                    sma_5 = float(data[b"sma_5"])
                    sma_20 = float(data[b"sma_20"])
                    if sma_5 > sma_20:
                        technical_setups.add(ticker)

                # Check for RSI conditions
                if b"rsi" in data:
                    rsi = float(data[b"rsi"])
                    # Oversold condition (RSI < 30)
                    if rsi < 30 or rsi > 70:
                        technical_setups.add(ticker)

            return technical_setups
        except Exception as e:
            logger.exception(f"Error getting technical setups: {e!s}")
            return set()

    async def get_market_regime(self) -> str | None:
        """
        Get current market regime

        Returns:
            String indicating market regime: 'bullish', 'bearish', 'volatile', or 'normal'
        """
        try:
            if not self.redis:
                return "normal"

            # Get SPY and VIX data from Redis
            spy_data = self.redis.hgetall("stock:SPY:technical")
            vix_data = self.redis.hgetall("stock:VIX:technical")

            if spy_data and vix_data:
                # Get SPY trend (above or below 20-day SMA)
                spy_price = float(spy_data.get(b"last_price", 0))
                spy_sma20 = float(spy_data.get(b"sma_20", 0))

                # Get VIX level
                vix_price = float(vix_data.get(b"last_price", 0))

                # Classify regime
                if spy_price > spy_sma20 and vix_price < 20:
                    return "bullish"
                if spy_price < spy_sma20 and vix_price > 30:
                    return "bearish"
                if vix_price > 25:
                    return "volatile"

    async def get_current_price(self, ticker):
        """
        Get current price for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price (float) or None
        """
        try:
            # Check Redis for last price
            if self.redis:
                price_data = self.redis.hgetall(f"stock:{ticker}:last_price")
                if price_data and b"price" in price_data:
                    return float(price_data[b"price"])

            # If not in Redis, get from API
            response = await self.polygon.get_aggregates(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                limit=1,
            )

            if isinstance(response, pd.DataFrame) and not response.empty:
                return float(response["close"].iloc[0])

            return None

        except Exception as e:
            logger.exception(
                f"Error getting current price for {ticker}: {e!s}")
            return None

    async def get_options_flow(self, ticker):
        """
        Get options flow data for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            List or DataFrame of options flow data
        """
        try:
            # Use Unusual Whales client to get options flow data
            return await self.unusual_whales.get_flow_alerts(ticker, limit=50)
        except Exception as e:
            logger.exception(f"Error getting options flow for {ticker}: {e!s}")
            return []

    # ===== WEBSOCKET AND WATCHLIST METHODS =====

    def subscribe_to_watchlist_channels(self) -> None:
        """
        Subscribe to websocket channels for watchlist stocks
        """
        try:
            if not self.polygon_ws or not self.redis:
                return

            # Get current watchlist
            watchlist_data = self.redis.zrevrange("watchlist:active", 0, -1)
            watchlist = [
                item.decode("utf-8") if isinstance(item, bytes) else item
                for item in watchlist_data
            ]

            if not watchlist:
                return

            # Subscribe to trades, quotes, and minute aggregates
            self.polygon_ws.subscribe_to_quotes(watchlist)
            self.polygon_ws.subscribe_to_trades(watchlist)
            self.polygon_ws.subscribe_to_minute_aggs(watchlist)

            logger.info(
                f"Subscribed to websocket channels for {len(watchlist)} watchlist stocks",
            )
        except Exception as e:
            logger.exception(f"Error subscribing to websocket channels: {e!s}")

    def should_update_watchlist(self):
        """
        Check if watchlist should be updated

        Returns:
            Boolean indicating if update is needed
        """
        if not self.redis:
            return True

        # Get last update time
        last_update = self.redis.get("watchlist:active:last_update")

        if not last_update:
            return True

        # Convert to datetime
        last_update_time = datetime.datetime.fromisoformat(
            last_update.decode("utf-8"))
        now = datetime.datetime.now()

        # Update if more than refresh_interval seconds since last update
        elapsed_seconds = (now - last_update_time).total_seconds()
        return elapsed_seconds >= self.config["watchlist"]["refresh_interval"]

    # ===== UTILITY METHODS =====

    def _generate_mock_price_data(self, tickers, start_date, end_date, timeframe="1d"):
        """
        Generate mock price data for testing purposes

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe ('1m', '5m', '1h', '1d')

        Returns:
            Dictionary of ticker -> DataFrame with mock price data
        """
        results = {}

        # Calculate number of days between start and end date
        days = (end_date - start_date).days + 1
        if days <= 0:
            days = 1

        for ticker in tickers:
            # Create date range
            dates = [start_date + timedelta(days=i) for i in range(days)]

            # Generate random price data
            base_price = float(os.getenv("DEFAULT_BASE_PRICE", "50.0"))

            data = []
            for _i, date in enumerate(dates):
                # Generate slightly random prices
                close_price = base_price + (np.random.random() * 2 - 1)
                open_price = close_price - (np.random.random() * 1)
                high_price = max(close_price, open_price) + \
                    (np.random.random() * 1)
                low_price = min(close_price, open_price) - \
                    (np.random.random() * 1)
                volume = int(1000000 * np.random.random())

                data.append(
                    {
                        "timestamp": date,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                    },
                )

            results[ticker] = pd.DataFrame(data)
        return results

    def _generate_mock_options_data(self, tickers, start_date, end_date):
        """
        Generate mock options data for testing purposes

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary of ticker -> DataFrame with mock options data
        """
        results = {}

        for ticker in tickers:
            # Generate random options data
            data = []
            for i in range(5):  # Generate 5 mock options
                expiration = start_date + timedelta(days=30 * (i + 1))
                strike = 100.0 + (i * 10)

                data.append(
                    {
                        "ticker": ticker,
                        "timestamp": datetime.datetime.now(),
                        "side": "call" if i % 2 == 0 else "put",
                        "strike": strike,
                        "expiration": expiration,
                        "volume": int(1000 * np.random.random()),
                        "open_interest": int(5000 * np.random.random()),
                        "implied_volatility": 0.2 + (np.random.random() * 0.3),
                        "delta": 0.5 + (np.random.random() * 0.4 - 0.2),
                        "gamma": 0.05 * np.random.random(),
                        "theta": -0.05 * np.random.random(),
                        "vega": 0.1 * np.random.random(),
                        "premium": 2.0 + (np.random.random() * 3),
                    },
                )

            results[ticker] = pd.DataFrame(data)
        return results

    def _generate_mock_market_data(self, start_date, end_date, symbols=None):
        """
        Generate mock market data for testing purposes

        Args:
            start_date: Start date
            end_date: End date
            symbols: List of market symbols

        Returns:
            DataFrame with mock market data
        """
        # Calculate number of days between start and end date
        days = (end_date - start_date).days + 1
        if days <= 0:
            days = 1

        # Create date range
        dates = [start_date + timedelta(days=i) for i in range(days)]

        # Use default market symbols if none provided
        if symbols is None:
            # Use market index tickers from configuration
            market_symbols = [
                ticker
                for ticker in config["stock_selection"]["universe"]["default_tickers"]
                if ticker in ["SPY", "QQQ", "IWM", "DIA"]
            ]
            # Fallback to SPY and QQQ if no market indices in default tickers
            symbols = market_symbols if market_symbols else ["SPY", "QQQ"]
            logger.info(f"Using default market symbols: {symbols}")

        # Generate random market data
        data = []
        for date in dates:
            row = {"timestamp": date}

            for symbol in symbols:
                symbol_lower = symbol.lower()
                base_price = 400.0 if symbol == "SPY" else 20.0

                # Generate random prices
                close_price = base_price + (np.random.random() * 10 - 5)
                row[f"{symbol_lower}_close"] = close_price
                row[f"{symbol_lower}_open"] = close_price - \
                    (np.random.random() * 2 - 1)
                row[f"{symbol_lower}_high"] = close_price + \
                    (np.random.random() * 2)
                row[f"{symbol_lower}_low"] = close_price - \
                    (np.random.random() * 2)
                row[f"{symbol_lower}_volume"] = int(
                    10000000 * np.random.random())

            data.append(row)
        return pd.DataFrame(data)

    def _process_polygon_response(self, response, ticker, prefix=None):
        """
        Process response from Polygon API

        Args:
            response: API response
            ticker: Ticker symbol
            prefix: Optional prefix for column names

        Returns:
            Processed DataFrame
        """
        try:
            # Handle empty response
            if not response:
                return None

            # Handle DataFrame response
            if isinstance(response, pd.DataFrame):
                df = response.copy()
                df["ticker"] = ticker

                # Ensure timestamp column exists
                if "t" in df.columns and "timestamp" not in df.columns:
                    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")

                # Rename columns with optional prefix
                col_prefix = f"{prefix}_" if prefix else ""
                column_mapping = {
                    "o": f"{col_prefix}open",
                    "h": f"{col_prefix}high",
                    "l": f"{col_prefix}low",
                    "c": f"{col_prefix}close",
                    "v": f"{col_prefix}volume",
                    "vw": f"{col_prefix}vwap",
                }
                return df.rename(
                    columns={k: v for k, v in column_mapping.items()
                             if k in df.columns},
                )

            # Handle dictionary response
            if isinstance(response, dict) and "results" in response:
                df = pd.DataFrame(response["results"])
                df["ticker"] = ticker

                # Ensure timestamp column exists
                if "t" in df.columns and "timestamp" not in df.columns:
                    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")

                # Rename columns with optional prefix
                col_prefix = f"{prefix}_" if prefix else ""
                column_mapping = {
                    "o": f"{col_prefix}open",
                    "h": f"{col_prefix}high",
                    "l": f"{col_prefix}low",
                    "c": f"{col_prefix}close",
                    "v": f"{col_prefix}volume",
                    "vw": f"{col_prefix}vwap",
                }
                return df.rename(
                    columns={k: v for k, v in column_mapping.items()
                             if k in df.columns},
                )

            # Handle list response
            if isinstance(response, list) and response:
                df = pd.DataFrame(response)
                df["ticker"] = ticker

                # Ensure timestamp column exists
                if "t" in df.columns and "timestamp" not in df.columns:
                    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")

                # Rename columns with optional prefix
                col_prefix = f"{prefix}_" if prefix else ""
                column_mapping = {
                    "o": f"{col_prefix}open",
                    "h": f"{col_prefix}high",
                    "l": f"{col_prefix}low",
                    "c": f"{col_prefix}close",
                    "v": f"{col_prefix}volume",
                    "vw": f"{col_prefix}vwap",
                }
                return df.rename(
                    columns={k: v for k, v in column_mapping.items()
                             if k in df.columns},
                )

            logger.warning(
                f"Unexpected response format from Polygon API: {type(response)}",
            )
            return None

        except Exception as e:
            logger.exception(f"Error processing Polygon response: {e}")
            return None

    def _process_unusual_whales_response(self, response, ticker):
        """
        Process response from Unusual Whales API

        Args:
            response: API response
            ticker: Ticker symbol

        Returns:
            Processed DataFrame
        """
        try:
            # Handle empty response
            if not response:
                return pd.DataFrame()

            # Handle DataFrame response
            if isinstance(response, pd.DataFrame):
                return response

            # Handle dictionary response
            if isinstance(response, dict) and "data" in response:
                df = pd.DataFrame(response["data"])
                if "ticker" not in df.columns:
                    df["ticker"] = ticker
                return df

            # Handle list response
            if isinstance(response, list):
                df = pd.DataFrame(response)
                if "ticker" not in df.columns:
                    df["ticker"] = ticker
                return df

            logger.warning(
                f"Unexpected response format from Unusual Whales API: {type(response)}",
            )
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"Error processing Unusual Whales response: {e}")
            return pd.DataFrame()

    def _load_from_cache(self, cache_path):
        """
        Load data from cache file

        Args:
            cache_path: Path to cache file

        Returns:
            Cached data or None if not found/expired
        """
        try:
            if os.path.exists(cache_path):
                # Check if cache is expired
                if (
                    time.time() - os.path.getmtime(cache_path)
                    < self.config["cache_expiry"]
                ):
                    # Load data
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
            return None
        except Exception as e:
            logger.exception(f"Error loading from cache: {e}")
            return None

    def _save_to_cache(self, data, cache_path) -> bool | None:
        """
        Save data to cache file

        Args:
            data: Data to cache
            cache_path: Path to cache file

        Returns:
            Boolean indicating success
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # Save data
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            # Record cache write in Prometheus
            if PROMETHEUS_AVAILABLE:
                data_type = (
                    os.path.basename(cache_path).split("_")[0]
                    if os.path.basename(cache_path).find("_") > 0
                    else "unknown"
                )
                # Track the number of rows processed
                if hasattr(data, "shape") and len(data.shape) > 0:
                    DATA_ROWS_PROCESSED.labels(operation=f"cache_{data_type}").inc(
                        data.shape[0],
                    )

            logger.debug(f"Saved data to cache: {cache_path}")
            return True
        except Exception as e:
            logger.exception(f"Error saving to cache: {e}")
            # Record error in Prometheus
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    api="cache", endpoint="save", error_type=type(e).__name__,
                ).inc()
            return False

    def _send_frontend_notification(self, message, level="info", category="data_pipeline", details=None):
        """Send notification to frontend via Redis

        Args:
            message (str): Notification message
            level (str): Notification level (info, warning, error, success)
            category (str): Notification category for filtering
            details (dict): Additional details for the notification
        """
        if not self.redis:
            logger.debug(
                f"Redis not available, skipping notification: {message}")
            return

        try:
            # Create notification object
            notification = {
                "type": category,
                "message": message,
                "level": level,
                "timestamp": time.time(),
                "details": details or {}
            }

            # Add to general notifications list
            self.redis.lpush("frontend:notifications",
                             json.dumps(notification))
            self.redis.ltrim("frontend:notifications", 0, 99)  # Keep last 100

            # Add to category-specific list
            category_key = f"frontend:{category}"
            self.redis.lpush(category_key, json.dumps(notification))
            self.redis.ltrim(category_key, 0, 49)  # Keep last 50 per category

            # Log based on level
            if level == "error":
                logger.error(f"Frontend notification: {message}")
            elif level == "warning":
                logger.warning(f"Frontend notification: {message}")
            else:
                logger.info(f"Frontend notification: {message}")

            # Update system status if this is a system-level notification
            if category in ["system_status", "data_system"]:
                try:
                    system_status = json.loads(self.redis.get(
                        "frontend:system:status") or "{}")
                    system_status["last_update"] = time.time()
                    system_status["last_message"] = message
                    system_status["status"] = level
                    self.redis.set("frontend:system:status",
                                   json.dumps(system_status))
                except Exception as e:
                    logger.error(f"Error updating system status: {e}")

        except Exception as e:
            logger.error(f"Error sending frontend notification: {e}")

    def _run_async(self, coroutine):
        """
        Run a coroutine in the event loop

        Args:
            coroutine: Coroutine to run

        Returns:
            Result of the coroutine
        """
        try:
            # Start timing for Prometheus metrics
            start_time = time.time()

            # Use run_until_complete if the loop is not running
            if not self.loop.is_running():
                result = self.loop.run_until_complete(coroutine)
            else:
                # If loop is already running, use a future
                future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
                # 60 second timeout
                result = future.result(timeout=60)

            # Record execution time in Prometheus
            if PROMETHEUS_AVAILABLE:
                execution_time = time.time() - start_time
                DATA_PROCESSING_TIME.labels(
                    operation="async_execution", use_gpu=str(self.use_gpu),
                ).observe(execution_time)

                # Record API request if this is an API call
                if (
                    hasattr(coroutine, "__qualname__")
                    and "get_" in coroutine.__qualname__
                ):
                    api_name = (
                        coroutine.__qualname__.split(".")[0]
                        if "." in coroutine.__qualname__
                        else "unknown"
                    )
                    endpoint = (
                        coroutine.__qualname__.split(".")[-1]
                        if "." in coroutine.__qualname__
                        else coroutine.__qualname__
                    )
                    API_REQUEST_COUNT.labels(
                        api=api_name, endpoint=endpoint).inc()

            return result
        except Exception as e:
            logger.exception(f"Error running async coroutine: {e}")

            # Record error in Prometheus
            if PROMETHEUS_AVAILABLE:
                if hasattr(coroutine, "__qualname__"):
                    api_name = (
                        coroutine.__qualname__.split(".")[0]
                        if "." in coroutine.__qualname__
                        else "unknown"
                    )
                    endpoint = (
                        coroutine.__qualname__.split(".")[-1]
                        if "." in coroutine.__qualname__
                        else coroutine.__qualname__
                    )
                    API_ERROR_COUNT.labels(
                        api=api_name, endpoint=endpoint, error_type=type(
                            e).__name__,
                    ).inc()
                else:
                    API_ERROR_COUNT.labels(
                        api="unknown",
                        endpoint="async_execution",
                        error_type=type(e).__name__,
                    ).inc()

            return None

    def same_day(self, timestamp_str):
        """
        Check if timestamp is from the same day (ET)

        Args:
            timestamp_str: Timestamp string

        Returns:
            Boolean indicating if timestamp is from today
        """
        try:
            # Parse timestamp
            timestamp = datetime.datetime.fromisoformat(timestamp_str)

            # Get current date (ET)
            now = datetime.datetime.now(pytz.timezone("US/Eastern"))

            # Check if same day
            return timestamp.date() == now.date()
        except Exception as e:
            logger.exception(f"Error checking same day: {e}")
            return False

    # ===== ML FEATURE SELECTION METHODS =====

    def select_features(self, features, target, task_type="classification"):
        """
        Select important features based on feature selection method

        Args:
            features: DataFrame of features
            target: Series of target values
            task_type: 'classification' or 'regression'

        Returns:
            DataFrame with selected features
        """
        try:
            logger.info(f"Performing feature selection for {task_type} task")

            if not self.config["feature_selection"]["enabled"]:
                return features

            if len(features) == 0 or len(target) == 0:
                logger.warning(
                    "Empty features or target, skipping feature selection")
                return features

            method = self.config["feature_selection"]["method"]

            # Create appropriate estimator based on task type
            if task_type == "classification":
                if method == "importance" and XGBOOST_AVAILABLE:
                    estimator = XGBClassifier(
                        n_estimators=100, learning_rate=0.05)
                elif method == "rfe":
                    estimator = RandomForestClassifier(n_estimators=50)
                else:  # mutual_info
                    return self._select_using_mutual_info(features, target, task_type)
            elif method == "importance" and XGBOOST_AVAILABLE:
                estimator = XGBRegressor(n_estimators=100, learning_rate=0.05)
            elif method == "rfe":
                estimator = RandomForestRegressor(n_estimators=50)
            else:  # mutual_info
                return self._select_using_mutual_info(features, target, task_type)

            # Apply selection method
            if method == "importance":
                threshold = self.config["feature_selection"]["threshold"]
                selector = SelectFromModel(estimator, threshold=threshold)
                selector.fit(features, target)
                selected_features_mask = selector.get_support()
            elif method == "rfe":
                n_features = min(
                    self.config["feature_selection"]["n_features"], features.shape[1],
                )
                selector = RFE(estimator, n_features_to_select=n_features)
                selector.fit(features, target)
                selected_features_mask = selector.get_support()

            # Get selected feature names
            selected_features = features.columns[selected_features_mask].tolist(
            )

            if not selected_features:
                logger.warning("No features selected, using all features")
                return features

            logger.info(
                f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}...",
            )
            return features[selected_features]

        except Exception as e:
            logger.error(f"Error in feature selection: {e!s}", exc_info=True)
            return features

    def _select_using_mutual_info(self, features, target, task_type):
        """
        Helper method for mutual information based feature selection

        Args:
            features: DataFrame of features
            target: Series of target values
            task_type: 'classification' or 'regression'

        Returns:
            DataFrame with selected features
        """
        try:
            # Calculate mutual information
            if task_type == "classification":
                mi_scores = mutual_info_classif(features, target)
            else:
                mi_scores = mutual_info_regression(features, target)

            # Create a ranking of features
            mi_ranking = pd.Series(mi_scores, index=features.columns)
            mi_ranking = mi_ranking.sort_values(ascending=False)

            # Select top features
            n_features = min(
                self.config["feature_selection"]["n_features"], features.shape[1],
            )
            selected_features = mi_ranking.index[:n_features].tolist()

            logger.info(
                f"Selected {len(selected_features)} features using mutual info: {', '.join(selected_features[:5])}...",
            )
            return features[selected_features]

        except Exception as e:
            logger.error(
                f"Error in mutual info feature selection: {e!s}", exc_info=True,
            )
            return features

    def create_time_series_splits(self, features, target):
        """
        Create time series cross-validation splits

        Args:
            features: DataFrame of features
            target: Series of target values

        Returns:
            List of (train_idx, test_idx) tuples
        """
        try:
            n_splits = self.config["time_series_cv"]["n_splits"]
            embargo_size = self.config["time_series_cv"]["embargo_size"]

            # Total data size
            n_samples = len(features)

            # Calculate split sizes
            test_size = int(n_samples / (n_splits + 1))

            splits = []
            for i in range(n_splits):
                # Calculate indices
                test_start = (i + 1) * test_size
                test_end = test_start + test_size

                # Apply embargo - gap between train and test
                if embargo_size > 0:
                    train_end = max(0, test_start - embargo_size)
                else:
                    train_end = test_start

                # Create index arrays
                train_idx = list(range(train_end))
                test_idx = list(range(test_start, min(test_end, n_samples)))

                splits.append((train_idx, test_idx))

            return splits

        except Exception as e:
            logger.error(
                f"Error creating time series splits: {e!s}", exc_info=True)
            # Return a simple 80/20 split as fallback
            n_samples = len(features)
            split_idx = int(n_samples * 0.8)
            return [(list(range(split_idx)), list(range(split_idx, n_samples)))]
