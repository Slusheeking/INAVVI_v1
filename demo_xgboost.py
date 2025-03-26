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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("demo_xgboost")

# Import the XGBoost model and integration
from xgboost_model import XGBoostModel, create_pretrained_model, load_pretrained_model
from xgboost_integration import XGBoostIntegration

def run_simple_demo():
    """Run a simple demo with synthetic data"""
    logger.info("Running simple XGBoost demo with synthetic data")
    
    # Create a pretrained model
    model_path = 'models/xgboost/demo_model.json'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    logger.info(f"Creating pretrained model at {model_path}")
    create_pretrained_model(model_path)
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = load_pretrained_model(model_path)
    
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
        # Initialize the integration with test mode
        integration = XGBoostIntegration(create_if_missing=True)
        
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
    """Check if XGBoost is properly installed"""
    try:
        import xgboost as xgb
        logger.info(f"XGBoost version {xgb.__version__} is installed")
        return True
    except ImportError:
        logger.error("XGBoost is not installed. Please install with 'pip install xgboost'")
        return False

if __name__ == "__main__":
    # Check XGBoost installation
    if check_xgboost_installation():
        # Run the demos
        run_simple_demo()
        run_integration_demo()
    else:
        logger.error("XGBoost installation check failed. Please install XGBoost first.")