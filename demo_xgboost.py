#!/usr/bin/env python3
"""
XGBoost Demo Script

This script demonstrates how to use the XGBoost model with the data pipeline:
1. Creating a pretrained XGBoost model
2. Loading the model
3. Making predictions with sample data

Run this script to verify that XGBoost is properly installed and integrated.
"""

import os
import numpy as np
import pandas as pd
import datetime
import logging
import packaging
from xgboost_model import create_pretrained_model, load_pretrained_model
from xgboost_integration import XGBoostIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("demo_xgboost")

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    logger.info(f"PyTorch version {torch.__version__} is available (CUDA: {CUDA_AVAILABLE})")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.warning("PyTorch is not available. Install with 'pip install torch'")

def run_simple_demo():
    """Run a simple demo with synthetic data"""
    logger.info("Running simple XGBoost demo with synthetic data")
    
    # Determine if PyTorch integration should be used
    use_pytorch = TORCH_AVAILABLE
    
    # Create a pretrained model
    model_path = 'models/xgboost/demo_model.json'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    logger.info(f"Creating pretrained model at {model_path}")
    create_pretrained_model(model_path, use_pytorch=use_pytorch)
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = load_pretrained_model(model_path, use_pytorch=use_pytorch)
    
    # Create sample data
    np.random.seed(42)
    sample_data = np.random.rand(10, 10)
    feature_names = [f'feature_{i}' for i in range(10)]
    sample_df = pd.DataFrame(sample_data, columns=feature_names)
    
    # Make predictions
    logger.info("Making predictions with sample data")
    predictions = model.predict(sample_df)
    
    logger.info(f"Sample predictions: {predictions}")
    logger.info("Simple demo completed successfully")
    
    return predictions

def run_integration_demo():
    """Run a demo with the XGBoost integration"""
    logger.info("Running XGBoost integration demo")
    
    try:
        # Determine if PyTorch integration should be used
        use_pytorch = TORCH_AVAILABLE
        use_gpu = CUDA_AVAILABLE
        
        # Initialize the integration with test mode and PyTorch support
        integration = XGBoostIntegration(
            create_if_missing=True,
            use_gpu=use_gpu,
            use_pytorch=use_pytorch
        )
        
        # Log PyTorch integration status
        if hasattr(integration, 'use_pytorch_integration') and integration.use_pytorch_integration:
            logger.info("Using XGBoost with PyTorch integration")
        else:
            logger.info("Using standard XGBoost without PyTorch integration")
        
        # Define date range and tickers for demo
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)  # 30 days of data
        tickers = ['AAPL', 'MSFT']  # Just a couple of tickers for the demo
        
        logger.info(f"Creating and training model with test data for {tickers}")
        
        # The actual API calls will be mocked in test mode
        integration.train_model(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            verbose=True
        )
        
        logger.info("Integration demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in integration demo: {e}")

def check_xgboost_installation():
    """Check if XGBoost is properly installed and check for PyTorch integration"""
    try:
        import xgboost as xgb
        
        # Check XGBoost version
        xgb_version = xgb.__version__
        logger.info(f"XGBoost version {xgb_version} is installed")
        
        # Check if XGBoost 3.0.0+ is available for PyTorch integration
        xgboost_3_available = packaging.version.parse(xgb_version) >= packaging.version.parse("3.0.0")
        if xgboost_3_available:
            logger.info("XGBoost 3.0.0+ detected - PyTorch integration is supported")
        else:
            logger.warning(f"XGBoost version {xgb_version} detected. Version 3.0.0 or later recommended for PyTorch integration")
        
        # Check if PyTorch integration is available
        use_pytorch_integration = TORCH_AVAILABLE and xgboost_3_available
        if use_pytorch_integration:
            logger.info("PyTorch integration with XGBoost is available")
            if CUDA_AVAILABLE:
                logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
                logger.info("GPU acceleration will be used for XGBoost with PyTorch integration")
            else:
                logger.info("CUDA is not available - CPU will be used for XGBoost with PyTorch integration")
        else:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch is not available - standard XGBoost will be used")
            elif not xgboost_3_available:
                logger.warning("XGBoost 3.0.0+ is required for PyTorch integration - standard XGBoost will be used")
        
        return True
    except ImportError:
        logger.error("XGBoost is not installed. Please install with 'pip install xgboost>=3.0.0'")
        return False

if __name__ == "__main__":
    # Check XGBoost installation
    if check_xgboost_installation():
        # Run the demos
        run_simple_demo()
        run_integration_demo()
    else:
        logger.error("XGBoost installation check failed. Please install XGBoost first.")