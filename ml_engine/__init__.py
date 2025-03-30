#!/usr/bin/env python3
"""
ML Engine Package

This package provides a unified machine learning engine for the trading system:
1. GPU acceleration and optimization
2. Technical indicator calculation
3. Model training for various prediction tasks
4. Utility functions for feature selection, time series CV, and diagnostics
5. Reporting and monitoring functionality
6. XGBoost integration with PyTorch support

The ML Engine is designed for high-performance trading systems with robust
error handling and monitoring capabilities.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import key components for easier access
from ml_engine.base import MLModelTrainer
from ml_engine.data_processor import MLDataProcessor
from utils.metrics_registry import MODEL_TRAINING_TIME, PREDICTION_LATENCY
from ml_engine.indicators import (
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_adx,
    calculate_obv,
    calculate_rsi,
    calculate_atr,
)
from ml_engine.utils import (
    select_features,
    create_time_series_splits,
    optimize_hyperparameters,
)

# Import XGBoost components
try:
    from ml_engine.xgboost_model import (
        XGBoostModel,
        create_pretrained_model,
        load_pretrained_model,
    )
    from ml_engine.xgboost_integration import XGBoostIntegration
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import GPU utilities if available
try:
    from utils.gpu_utils import is_gpu_available, gpu_manager, clear_gpu_memory, run_diagnostics
except ImportError:
    # Define fallback functions if GPU utilities are not available
    def is_gpu_available():
        return False
        
    def clear_gpu_memory():
        pass
        
    def run_diagnostics():
        return {"gpu_available": False}
    
    gpu_manager = None

# Configure logging
import logging
logger = logging.getLogger("ml_engine")
logger.setLevel(logging.INFO)

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error

# Version
__version__ = "1.0.0"