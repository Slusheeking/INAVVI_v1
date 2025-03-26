#!/usr/bin/env python3
"""
XGBoost Model Module

This module provides functionality for loading and using pretrained XGBoost models:
1. Loading pretrained models from disk
2. Making predictions with loaded models
3. Evaluating model performance
4. Integration with the data pipeline

The module is designed to work with the existing data pipeline and supports
GPU acceleration when available.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("xgboost_model")

# Check if XGBoost is available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info(f"XGBoost version {xgb.__version__} is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost is not available. Install with 'pip install xgboost'")


class XGBoostModel:
    """
    XGBoost model wrapper for loading and using pretrained models
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "classifier",
        use_gpu: bool = True,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the XGBoost model wrapper"""
        self.model_path = model_path
        self.model_type = model_type
        self.use_gpu = use_gpu and XGBOOST_AVAILABLE
        self.feature_names = feature_names
        self.model = None
        
        # Check if XGBoost is available
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost is not available. Install with 'pip install xgboost'")
            return
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No model path provided or model file not found")
            
            # Create a default model
            if model_type == "classifier":
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    tree_method='gpu_hist' if use_gpu else 'hist',
                    eval_metric='logloss'
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    tree_method='gpu_hist' if use_gpu else 'hist',
                    eval_metric='rmse'
                )
            
            logger.info(f"Created default {model_type} model")
    
    def load_model(self, model_path: str) -> bool:
        """Load a pretrained XGBoost model from disk"""
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost is not available. Cannot load model.")
                return False
            
            # Check if the file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the model
            if model_path.endswith('.json'):
                # Load from JSON format
                if self.model_type == "classifier":
                    self.model = xgb.XGBClassifier()
                else:
                    self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
            elif model_path.endswith('.pkl'):
                # Load from pickle format
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # Try to load as binary format
                if self.model_type == "classifier":
                    self.model = xgb.XGBClassifier()
                else:
                    self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
            
            logger.info(f"Successfully loaded model from {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: str, format: str = 'json') -> bool:
        """Save the XGBoost model to disk"""
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot save model.")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            if format == 'json':
                self.model.save_model(model_path)
            elif format == 'pkl':
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
            elif format == 'binary':
                self.model.save_model(model_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Successfully saved model to {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with the loaded model"""
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot make predictions.")
                return np.array([])
            
            # Make predictions
            predictions = self.model.predict(features)
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def train(self, features: pd.DataFrame, target: pd.Series, eval_set=None, **kwargs) -> bool:
        """Train the XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot train model.")
                return False
            
            # Store feature names
            self.feature_names = features.columns.tolist()
            
            # Train the model
            self.model.fit(features, target, eval_set=eval_set, **kwargs)
            
            logger.info("Successfully trained model")
            return True
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False


def create_pretrained_model(save_path: str = 'models/xgboost/pretrained_model.json'):
    """Create and save a pretrained XGBoost model for demonstration purposes"""
    try:
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost is not available. Cannot create pretrained model.")
            return False
        
        # Create a simple dataset
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Create and train a model
        model = XGBoostModel(model_type='classifier')
        model.train(pd.DataFrame(X, columns=feature_names), pd.Series(y))
        
        # Save the model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_model(save_path)
        
        logger.info(f"Created and saved pretrained model to {save_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating pretrained model: {e}")
        return False


def load_pretrained_model(model_path: str = 'models/xgboost/pretrained_model.json'):
    """Load a pretrained XGBoost model"""
    model = XGBoostModel(model_path=model_path)
    return model


if __name__ == "__main__":
    # Create a pretrained model
    create_pretrained_model()
    
    # Load the pretrained model
    model = load_pretrained_model()
    
    # Make a prediction
    if model.model is not None:
        X = np.random.rand(5, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        predictions = model.predict(pd.DataFrame(X, columns=feature_names))
        print(f"Predictions: {predictions}")
