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
import json
import pickle
import time
import numpy as np
from packaging import version
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
from utils.metrics_registry import PREDICTION_LATENCY

# Import sklearn components for enhanced functionality
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Configure logging
from utils.logging_config import get_logger
logger = get_logger("ml_engine.xgboost_model")

# Load environment variables
load_dotenv()

# Get XGBoost configuration from environment variables
XGBOOST_USE_GPU = os.environ.get("XGBOOST_USE_GPU", "true").lower() == "true"
XGBOOST_USE_PYTORCH = os.environ.get(
    "XGBOOST_USE_PYTORCH", "true").lower() == "true"
# Default to 'hist' method which works on both CPU and GPU
XGBOOST_TREE_METHOD = os.environ.get("XGBOOST_TREE_METHOD", "hist")
XGBOOST_DEVICE = os.environ.get(
    "XGBOOST_DEVICE", "cuda:0" if XGBOOST_USE_GPU else "cpu")
XGBOOST_MAX_BINS = int(os.environ.get("XGBOOST_MAX_BINS", "256"))
XGBOOST_GPU_PREDICTOR = os.environ.get(
    "XGBOOST_GPU_PREDICTOR", "true").lower() == "true"

# Model versioning and caching
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
MODEL_CACHE = {}  # Simple in-memory cache for loaded models

# Check if XGBoost is available
try:
    import xgboost as xgb  # type: ignore

    # Check if XGBoost has GPU support
    XGBOOST_GPU_SUPPORT = False
    try:
        # Create a small test model with GPU settings
        test_model = xgb.XGBClassifier(tree_method='gpu_hist')
        test_model.fit(np.array([[1, 2], [3, 4]]),
                       np.array([0, 1]), verbose=False)
        XGBOOST_GPU_SUPPORT = True
        logger.info("XGBoost GPU support is available")
    except Exception as e:
        if "XGBoost version not compiled with GPU support" in str(e):
            logger.warning(
                "XGBoost is not compiled with GPU support. Using CPU instead.")
        else:
            logger.warning(
                f"Error testing GPU support: {e}. Using CPU instead.")
        XGBOOST_GPU_SUPPORT = False

    XGBOOST_AVAILABLE = True
    logger.info(f"XGBoost version {xgb.__version__} is available")

    # Check if we're using XGBoost 3.0.0 or later
    XGBOOST_VERSION = version.parse(xgb.__version__)
    XGBOOST_3_AVAILABLE = XGBOOST_VERSION >= version.parse("3.0.0")
    if XGBOOST_3_AVAILABLE:
        logger.info("Using XGBoost 3.0.0 or later with PyTorch integration")
    else:
        logger.warning(
            f"XGBoost version {xgb.__version__} detected. Version 3.0.0 or later recommended for PyTorch integration")
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_3_AVAILABLE = False
    XGBOOST_GPU_SUPPORT = False
    logger.warning(
        "XGBoost is not available. Install with 'pip install xgboost>=3.0.0'")

# Check if PyTorch is available
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch version {torch.__version__} is available")

    # Check if CUDA is available
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA is not available for PyTorch")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.warning(
        "PyTorch is not available. Install with 'pip install torch'")


class XGBoostModel:
    """
    XGBoost model wrapper for loading and using pretrained models
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "classifier",
        use_gpu: Optional[bool] = None,
        feature_names: Optional[List[str]] = None,
        model_version: Optional[str] = None,
        cache_model: bool = True,
        use_pytorch: Optional[bool] = None,
    ) -> None:
        """Initialize the XGBoost model wrapper"""
        self.model_path = model_path
        self.model_type = model_type

        # Use environment variables if parameters are not provided
        self.use_gpu = use_gpu if use_gpu is not None else XGBOOST_USE_GPU
        self.use_gpu = self.use_gpu and XGBOOST_AVAILABLE
        # Only use GPU if XGBoost was compiled with GPU support
        self.use_gpu = self.use_gpu and XGBOOST_GPU_SUPPORT

        self.use_pytorch = use_pytorch if use_pytorch is not None else XGBOOST_USE_PYTORCH
        self.use_pytorch = self.use_pytorch and TORCH_AVAILABLE and XGBOOST_3_AVAILABLE

        self.model_version = model_version if model_version is not None else MODEL_VERSION
        self.cache_model = cache_model
        self.feature_names = feature_names
        self.model = None
        self.device = torch.device(
            "cuda" if CUDA_AVAILABLE and self.use_gpu else "cpu") if TORCH_AVAILABLE else None

        # Log configuration
        tree_method = XGBOOST_TREE_METHOD if self.use_gpu else 'hist'
        device = XGBOOST_DEVICE if self.use_gpu else 'cpu'
        logger.info(
            f"XGBoostModel initialized with: use_gpu={self.use_gpu}, use_pytorch={self.use_pytorch}")
        logger.info(
            f"Using tree_method={tree_method}, device={device}, max_bins={XGBOOST_MAX_BINS}")

        # Check if XGBoost is available
        if not XGBOOST_AVAILABLE:
            logger.warning(
                "XGBoost is not available. Install with 'pip install xgboost>=3.0.0'")
            return

        # Check if model is in cache
        cache_key = f"{model_path}_{model_type}_{self.use_gpu}_{self.use_pytorch}_{self.model_version}"
        if self.cache_model and model_path and cache_key in MODEL_CACHE:
            logger.info(f"Loading model from cache: {cache_key}")
            self.model = MODEL_CACHE[cache_key]
            self.feature_names = MODEL_CACHE.get(f"{cache_key}_features")
            logger.info(
                f"Model loaded from cache with version {self.model_version}")
            return

        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No model path provided or model file not found")

            # Create a default model
            if model_type == "classifier":
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Use PyTorch integration for XGBoost 3.0.0+
                    self.model = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method='hist',  # Default to hist method which works on CPU
                        eval_metric='logloss',
                        device='cpu',  # Default to CPU
                        enable_categorical=True,
                        objective='binary:logistic',
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='auto'  # Default to auto
                    )

                    # Update parameters if GPU is available and supported
                    if self.use_gpu:
                        try:
                            self.model.set_params(
                                tree_method=XGBOOST_TREE_METHOD,
                                device='cuda' if CUDA_AVAILABLE else 'cpu',
                                predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR else 'auto'
                            )
                            logger.info(
                                "Created XGBoost classifier with PyTorch integration and GPU support")
                        except Exception as e:
                            logger.warning(
                                f"Failed to set GPU parameters: {e}. Using CPU instead.")
                            self.use_gpu = False
                    else:
                        logger.info(
                            "Created XGBoost classifier with PyTorch integration (CPU only)")
                else:
                    # Use standard XGBoost
                    self.model = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                        eval_metric='logloss',
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
                    logger.info("Created standard XGBoost classifier")
            else:
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Use PyTorch integration for XGBoost 3.0.0+
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                        eval_metric='rmse',
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        enable_categorical=True,
                        objective='reg:squarederror',
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
                    logger.info(
                        "Created XGBoost regressor with PyTorch integration")
                else:
                    # Use standard XGBoost
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                        eval_metric='rmse',
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
                    logger.info("Created standard XGBoost regressor")

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
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBClassifier(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True,
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                    else:
                        self.model = xgb.XGBClassifier(
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                else:
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBRegressor(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True,
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                    else:
                        self.model = xgb.XGBRegressor(
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                self.model.load_model(model_path)
            elif model_path.endswith('.pkl'):
                # Load from pickle format
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)

                # If using PyTorch with XGBoost 3.0.0+, update the device
                if self.use_pytorch and XGBOOST_3_AVAILABLE and hasattr(self.model, 'set_params'):
                    self.model.set_params(
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
            else:
                # Try to load as binary format
                if self.model_type == "classifier":
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBClassifier(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True,
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                    else:
                        self.model = xgb.XGBClassifier(
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                else:
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBRegressor(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True,
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                    else:
                        self.model = xgb.XGBRegressor(
                            tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',

                            max_bin=XGBOOST_MAX_BINS,
                            predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                        )
                self.model.load_model(model_path)

            logger.info(f"Successfully loaded model from {model_path}")

            # Log PyTorch integration status
            if self.use_pytorch and XGBOOST_3_AVAILABLE:
                device = getattr(self.model, 'device', 'unknown')
                logger.info(
                    f"Model loaded with PyTorch integration (device: {device})")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def save_model(self, model_path: str, format: str = 'json') -> bool:
        """Save the XGBoost model to disk"""
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error(
                    "XGBoost is not available or no model loaded. Cannot save model.")
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save the model
            if format == 'json':
                # For PyTorch integration with XGBoost 3.0.0+, we need to handle device
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Get current device
                    current_device = getattr(self.model, 'device', None)

                    # Temporarily set to CPU for saving if needed
                    if current_device == 'cuda':
                        logger.info(
                            "Temporarily setting device to CPU for saving model")
                        self.model.set_params(device='cpu')

                    # Save the model
                    self.model.save_model(model_path)

                    # Restore device setting
                    if current_device == 'cuda':
                        self.model.set_params(device='cuda')
                else:
                    # Standard save for non-PyTorch models
                    self.model.save_model(model_path)
            elif format == 'pkl':
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
            elif format == 'binary':
                # For PyTorch integration with XGBoost 3.0.0+, we need to handle device
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Get current device
                    current_device = getattr(self.model, 'device', None)

                    # Temporarily set to CPU for saving if needed
                    if current_device == 'cuda':
                        logger.info(
                            "Temporarily setting device to CPU for saving model")
                        self.model.set_params(device='cpu')

                    # Save the model
                    self.model.save_model(model_path)

                    # Restore device setting
                    if current_device == 'cuda':
                        self.model.set_params(device='cuda')
                else:
                    # Standard save for non-PyTorch models
                    self.model.save_model(model_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

            logger.info(f"Successfully saved model to {model_path}")

            # Log PyTorch integration status
            if self.use_pytorch and XGBOOST_3_AVAILABLE:
                device = getattr(self.model, 'device', 'unknown')
                logger.info(
                    f"Model saved with PyTorch integration (device: {device})")

            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with the loaded model"""
        try:
            start_time = time.time()
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot make predictions.")
                return np.array([])
            
            # Handle feature mismatch by selecting only the features used during training
            if isinstance(features, pd.DataFrame):
                # Get the feature names used during training
                model_features = None
                if hasattr(self.model, 'feature_names_in_'):
                    # scikit-learn style model
                    model_features = self.model.feature_names_in_
                elif hasattr(self.model, 'feature_names'):
                    # XGBoost style model
                    model_features = self.model.feature_names
                elif self.feature_names:
                    # Use stored feature names
                    model_features = self.feature_names
                
                # Select only the features used during training if feature names are available
                if model_features is not None:
                    # Check if all required features are available
                    missing_features = [f for f in model_features if f not in features.columns]
                    extra_features = [f for f in features.columns if f not in model_features]
                    
                    if missing_features:
                        logger.warning(f"Missing features: {missing_features}. Predictions may be inaccurate.")
                        # Fill missing features with zeros
                        for f in missing_features:
                            features[f] = 0
                    
                    if extra_features:
                        logger.warning(f"Extra features detected: {extra_features}. These will be ignored.")
                    
                    # Select only the features used during training
                    features_to_use = features[model_features]
                    logger.info(f"Using {len(model_features)} features for prediction")
                else:
                    # No feature names available, try to use as is
                    logger.warning("No feature names available. Attempting to use features as provided.")
                    features_to_use = features
            else:
                # Not a DataFrame, use as is
                features_to_use = features
            
            # For PyTorch integration with XGBoost 3.0.0+
            if self.use_pytorch and XGBOOST_3_AVAILABLE:
                # Convert input to PyTorch tensor if needed
                if TORCH_AVAILABLE and isinstance(features_to_use, np.ndarray):
                    # Log the conversion
                    logger.debug("Converting numpy array to PyTorch tensor for prediction")
                    
                    # Make predictions directly with the model
                    predictions = self.model.predict(features_to_use)
                    
                    # Log device information
                    device = getattr(self.model, 'device', 'unknown')
                    logger.debug(f"Made predictions with PyTorch integration (device: {device})")
                else:
                    # Make predictions with DataFrame or other format
                    predictions = self.model.predict(features_to_use)
            else:
                # Standard prediction for non-PyTorch models
                predictions = self.model.predict(features_to_use)
            
            # Record prediction latency
            PREDICTION_LATENCY.labels(model_name=self.model_version).observe(time.time() - start_time)
            
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
            
            # Extract parameters that need special handling
            early_stopping_rounds = kwargs.pop('early_stopping_rounds', None)
            verbose = kwargs.pop('verbose', False)
            eval_set = kwargs.pop('eval_set', None)
            
            # For XGBoost 3.0.0+, we need to handle parameters differently
            if XGBOOST_3_AVAILABLE:
                # Create a clean set of parameters for fit()
                fit_params = {
                    'verbose': verbose
                }
                
                # Add eval_set if provided
                if eval_set:
                    fit_params['eval_set'] = eval_set
                
                # In XGBoost 3.0.0+, early stopping is configured through callbacks
                # or directly in the model constructor, not in fit()
                if early_stopping_rounds is not None:
                    # Only use early stopping if eval_set is provided
                    if eval_set:
                        logger.info(f"Configuring early stopping with {early_stopping_rounds} rounds")
                        # We need to set early_stopping in the model parameters
                        # This is done by recreating the model with the early_stopping parameter
                        if self.model_type == "classifier":
                            params = self.model.get_params()
                            params.update({
                                'early_stopping_rounds': early_stopping_rounds,
                                'callbacks': [xgb.callback.EarlyStopping(rounds=early_stopping_rounds)]
                            })
                            # Create a new model with the updated parameters
                            if self.use_pytorch:
                                self.model = xgb.XGBClassifier(**params)
                            else:
                                self.model = xgb.XGBClassifier(**params)
                        else:
                            params = self.model.get_params()
                            params.update({
                                'early_stopping_rounds': early_stopping_rounds,
                                'callbacks': [xgb.callback.EarlyStopping(rounds=early_stopping_rounds)]
                            })
                            # Create a new model with the updated parameters
                            if self.use_pytorch:
                                self.model = xgb.XGBRegressor(**params)
                            else:
                                self.model = xgb.XGBRegressor(**params)
                    else:
                        logger.warning("Early stopping requested but no eval_set provided. Disabling early stopping.")
                
                # For PyTorch integration with XGBoost 3.0.0+
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Log PyTorch integration
                    device = getattr(self.model, 'device', 'cpu')
                    logger.info(f"Training model with PyTorch integration (device: {device})")
                    
                    try:
                        # Check if we need to enable mixed precision training
                        if TORCH_AVAILABLE and CUDA_AVAILABLE and self.use_gpu and kwargs.pop('enable_amp', False):
                            logger.info("Enabling automatic mixed precision (AMP) for training")
                            
                            # Train with mixed precision
                            with torch.cuda.amp.autocast():
                                self.model.fit(features, target, **fit_params)
                        else:
                            # Standard training with PyTorch backend
                            self.model.fit(features, target, **fit_params)
                    except Exception as e:
                        if "XGBoost version not compiled with GPU support" in str(e):
                            logger.warning("XGBoost is not compiled with GPU support. Falling back to CPU.")
                            # Update model parameters to use CPU
                            self.model.set_params(
                                tree_method='hist',
                                device='cpu',
                                predictor='auto'
                            )
                            self.use_gpu = False
                            # Retry with CPU
                            self.model.fit(features, target, **fit_params)
                        elif "Must have at least 1 validation dataset for early stopping" in str(e):
                            logger.warning("No validation dataset provided for early stopping. Disabling early stopping.")
                            # Remove early stopping parameters
                            params = self.model.get_params()
                            if 'early_stopping_rounds' in params:
                                params.pop('early_stopping_rounds')
                            if 'callbacks' in params:
                                params.pop('callbacks')
                            
                            # Recreate model without early stopping
                            if self.model_type == "classifier":
                                self.model = xgb.XGBClassifier(**params)
                            else:
                                self.model = xgb.XGBRegressor(**params)
                            
                            # Retry without early stopping
                            self.model.fit(features, target, **fit_params)
                        else:
                            # Re-raise other exceptions
                            logger.error(f"Error training model: {e}")
                            raise
                else:
                    # Standard training
                    self.model.fit(features, target, **fit_params)
            else:
                # For older XGBoost versions, we can pass all parameters to fit()
                fit_params = kwargs.copy()
                
                # Add eval_set if provided
                if eval_set:
                    fit_params['eval_set'] = eval_set
                
                # Add verbose parameter
                fit_params['verbose'] = verbose
                
                # Add early_stopping_rounds if provided
                if early_stopping_rounds:
                    fit_params['early_stopping_rounds'] = early_stopping_rounds
                    logger.info(f"Configured early stopping with {early_stopping_rounds} rounds")
                
                # Standard training
                self.model.fit(features, target, **fit_params)
            
            logger.info("Successfully trained model")
            return True
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def tune_hyperparameters(self, features: pd.DataFrame, target: pd.Series,
                            param_grid: Dict = None, cv: int = 5,
                            method: str = 'grid', n_iter: int = 10,
                            time_series: bool = False) -> Dict:
        """
        Tune hyperparameters using GridSearchCV or RandomizedSearchCV
        
        Args:
            features: DataFrame of features
            target: Series of target values
            param_grid: Dictionary of hyperparameters to search
            cv: Number of cross-validation folds
            method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            n_iter: Number of iterations for RandomizedSearchCV
            time_series: Whether to use TimeSeriesSplit for cross-validation
            
        Returns:
            Dictionary of best parameters
        """
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost is not available. Cannot tune hyperparameters.")
                return {}
            
            # Default parameter grid if not provided
            if param_grid is None:
                if self.model_type == "classifier":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2]
                    }
                else:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2]
                    }
            
            # Create cross-validation strategy
            if time_series:
                cv_strategy = TimeSeriesSplit(n_splits=cv)
            else:
                cv_strategy = cv
            
            # Create base model
            if self.model_type == "classifier":
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    base_model = xgb.XGBClassifier(
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                        eval_metric='logloss',
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        enable_categorical=True,
                        objective='binary:logistic',
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
                    logger.info("Using XGBoost classifier with PyTorch integration for hyperparameter tuning")
                else:
                    base_model = xgb.XGBClassifier(
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                        use_label_encoder=False,
                        eval_metric='logloss',
                        
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
                    logger.info("Using standard XGBoost classifier for hyperparameter tuning")
            else:
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    base_model = xgb.XGBRegressor(
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                        eval_metric='rmse',
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        enable_categorical=True,
                        objective='reg:squarederror',
                        
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
                    logger.info("Using XGBoost regressor with PyTorch integration for hyperparameter tuning")
                else:
                    base_model = xgb.XGBRegressor(
                        tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                        eval_metric='rmse',
                        
                        max_bin=XGBOOST_MAX_BINS,
                        predictor='gpu_predictor' if XGBOOST_GPU_PREDICTOR and self.use_gpu else 'auto'
                    )
                    logger.info("Using standard XGBoost regressor for hyperparameter tuning")
            
            # Create search object
            if method == 'grid':
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring='accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error',
                    verbose=1,
                    n_jobs=-1
                )
            else:  # random search
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_strategy,
                    scoring='accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error',
                    verbose=1,
                    n_jobs=-1
                )
            
            # Fit the search
            search.fit(features, target)
            
            # Update model with best parameters
            self.model = search.best_estimator_
            
            logger.info(f"Best parameters found: {search.best_params_}")
            logger.info(f"Best score: {search.best_score_}")
            
            return search.best_params_
            
        except Exception as e:
            logger.error(f"Error tuning hyperparameters: {e}")
            return {}
    
    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Evaluate the model performance
        
        Args:
            features: DataFrame of features
            target: Series of target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot evaluate model.")
                return {}
            
            # Make predictions
            predictions = self.predict(features)
            
            # Calculate metrics based on model type
            metrics = {}
            if self.model_type == "classifier":
                # Classification metrics
                metrics['accuracy'] = accuracy_score(target, predictions)
                metrics['precision'] = precision_score(target, predictions, average='weighted')
                metrics['recall'] = recall_score(target, predictions, average='weighted')
                metrics['f1'] = f1_score(target, predictions, average='weighted')
                
                # ROC AUC if binary classification
                if len(np.unique(target)) == 2:
                    try:
                        # Get probability predictions
                        proba_predictions = self.model.predict_proba(features)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(target, proba_predictions)
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC: {e}")
                        metrics['roc_auc'] = None
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}


def create_pretrained_model(
    model_type: str = "classifier",
    use_gpu: bool = None,
    use_pytorch: bool = None,
    params: Dict = None,
    feature_names: List[str] = None,
) -> XGBoostModel:
    """
    Create a new XGBoost model with predefined parameters
    
    Args:
        model_type: Type of model ('classifier' or 'regressor')
        use_gpu: Whether to use GPU acceleration
        use_pytorch: Whether to use PyTorch integration
        params: Dictionary of model parameters
        feature_names: List of feature names
        
    Returns:
        XGBoostModel instance
    """
    try:
        # Create model instance
        model = XGBoostModel(
            model_type=model_type,
            use_gpu=use_gpu,
            use_pytorch=use_pytorch,
            feature_names=feature_names
        )
        
        # Set custom parameters if provided
        if params and model.model is not None:
            model.model.set_params(**params)
            logger.info(f"Created pretrained model with custom parameters: {params}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error creating pretrained model: {e}")
        # Return a default model instance
        return XGBoostModel(model_type=model_type)


    def cross_validate(self, features: pd.DataFrame, target: pd.Series, 
                      cv: int = 5, time_series: bool = True, 
                      scoring: str = None, **kwargs) -> Dict[str, float]:
        """
        Perform cross-validation on the model
        
        Args:
            features: DataFrame of features
            target: Series of target values
            cv: Number of cross-validation folds
            time_series: Whether to use TimeSeriesSplit for cross-validation
            scoring: Scoring metric to use (e.g., 'accuracy', 'neg_mean_squared_error')
            **kwargs: Additional arguments to pass to the model's train method
            
        Returns:
            Dictionary of cross-validation results
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot perform cross-validation.")
                return {}
            
            # Import required components
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
            
            # Set default scoring based on model type
            if scoring is None:
                scoring = 'accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error'
            
            # Create cross-validation strategy
            if time_series:
                cv_strategy = TimeSeriesSplit(n_splits=cv)
                logger.info(f"Using TimeSeriesSplit with {cv} splits for cross-validation")
            else:
                cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
                logger.info(f"Using KFold with {cv} splits for cross-validation")
            
            # Initialize results
            results = {}
            fold_scores = []
            
            # Perform manual cross-validation to have more control
            for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(features)):
                logger.info(f"Training fold {fold+1}/{cv}")
                
                # Split data
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                
                # Train model on this fold
                fold_success = self.train(X_train, y_train, **kwargs)
                
                if not fold_success:
                    logger.warning(f"Training failed for fold {fold+1}")
                    continue
                
                # Evaluate on test set
                y_pred = self.predict(X_test)
                
                # Calculate metrics based on model type
                if self.model_type == 'classifier':
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    fold_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred_binary),
                        'precision': precision_score(y_test, y_pred_binary, average='weighted'),
                        'recall': recall_score(y_test, y_pred_binary, average='weighted'),
                        'f1': f1_score(y_test, y_pred_binary, average='weighted')
                    }
                    
                    # Add ROC AUC if binary classification
                    if len(np.unique(y_test)) == 2:
                        fold_metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
                else:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    fold_metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                
                # Log fold metrics
                logger.info(f"Fold {fold+1} metrics: {fold_metrics}")
                fold_scores.append(fold_metrics)
            
            # Calculate average metrics across folds
            if fold_scores:
                for metric in fold_scores[0].keys():
                    results[metric] = np.mean([fold[metric] for fold in fold_scores])
                    results[f"{metric}_std"] = np.std([fold[metric] for fold in fold_scores])
                
                logger.info(f"Cross-validation results: {results}")
            else:
                logger.warning("No valid fold results to aggregate")
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing cross-validation: {e}")
            return {}


    def create_pipeline(self, numeric_features=None, categorical_features=None, 
                       text_features=None, handle_missing=True) -> object:
        """
        Create a scikit-learn pipeline for preprocessing and model training
        
        Args:
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            text_features: List of text feature column names
            handle_missing: Whether to handle missing values
            
        Returns:
            sklearn.pipeline.Pipeline object
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot create pipeline.")
                return None
            
            # Import required components
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            
            transformers = []
            
            # Add numeric features transformer
            if numeric_features:
                numeric_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median') if handle_missing else 'passthrough'),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('numeric', numeric_pipeline, numeric_features))
                logger.info(f"Added numeric transformer for {len(numeric_features)} features")
            
            # Add categorical features transformer
            if categorical_features:
                categorical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent') if handle_missing else 'passthrough'),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])
                transformers.append(('categorical', categorical_pipeline, categorical_features))
                logger.info(f"Added categorical transformer for {len(categorical_features)} features")
            
            # Add text features transformer if needed
            if text_features:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    text_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='constant', fill_value='') if handle_missing else 'passthrough'),
                        ('tfidf', TfidfVectorizer(max_features=1000))
                    ])
                    transformers.append(('text', text_pipeline, text_features))
                    logger.info(f"Added text transformer for {len(text_features)} features")
                except ImportError:
                    logger.warning("TfidfVectorizer not available. Text features will not be processed.")
            
            # Create the full pipeline
            preprocessor = ColumnTransformer(transformers, remainder='drop')
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', self.model)
            ])
            
            logger.info("Created preprocessing pipeline successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            return None


    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                    threshold: float = 0.1, method: str = 'ks') -> Dict[str, float]:
        """
        Detect feature drift between reference and current data
        
        Args:
            reference_data: Reference data (e.g., training data)
            current_data: Current data to check for drift
            threshold: Threshold for drift detection
            method: Method for drift detection ('ks' for Kolmogorov-Smirnov, 'js' for Jensen-Shannon)
            
        Returns:
            Dictionary of feature drift scores
        """
        try:
            # Ensure both datasets have the same columns
            common_columns = set(reference_data.columns).intersection(set(current_data.columns))
            if not common_columns:
                logger.error("No common columns between reference and current data")
                return {}
            
            reference_data = reference_data[list(common_columns)]
            current_data = current_data[list(common_columns)]
            
            drift_scores = {}
            
            # Calculate drift for each feature
            for col in common_columns:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(reference_data[col]) or not pd.api.types.is_numeric_dtype(current_data[col]):
                    logger.warning(f"Skipping non-numeric column: {col}")
                    continue
                
                # Handle missing values
                ref_values = reference_data[col].dropna().values
                cur_values = current_data[col].dropna().values
                
                if len(ref_values) == 0 or len(cur_values) == 0:
                    logger.warning(f"No valid values for column: {col}")
                    continue
                
                # Calculate drift score based on method
                if method == 'ks':
                    # Kolmogorov-Smirnov test
                    from scipy.stats import ks_2samp
                    statistic, p_value = ks_2samp(ref_values, cur_values)
                    drift_scores[col] = statistic
                elif method == 'js':
                    # Jensen-Shannon divergence
                    from scipy.spatial.distance import jensenshannon
                    from numpy import histogram
                    
                    # Create histograms with the same bins
                    bins = 20
                    ref_hist, bin_edges = histogram(ref_values, bins=bins, density=True)
                    cur_hist, _ = histogram(cur_values, bins=bin_edges, density=True)
                    
                    # Calculate Jensen-Shannon divergence
                    js_div = jensenshannon(ref_hist, cur_hist)
                    drift_scores[col] = js_div
                
                # Log significant drift
                if drift_scores[col] > threshold:
                    logger.warning(f"Significant drift detected for feature {col}: {drift_scores[col]:.4f}")
            
            return drift_scores
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {}


    def explain_predictions(self, features: pd.DataFrame, 
                           method: str = 'shap', 
                           n_samples: int = 100) -> Dict[str, Any]:
        """
        Generate explanations for model predictions
        
        Args:
            features: Features to explain predictions for
            method: Explanation method ('shap' or 'permutation')
            n_samples: Number of samples to use for explanation
            
        Returns:
            Dictionary of explanation results
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot explain predictions.")
                return {}
            
            # Limit number of samples for performance
            if len(features) > n_samples:
                logger.info(f"Limiting explanation to {n_samples} samples")
                features = features.sample(n_samples, random_state=42)
            
            # Generate explanations based on method
            if method == 'shap':
                try:
                    import shap
                    
                    # Create explainer based on model type
                    if hasattr(self.model, 'get_booster'):
                        # For XGBoost models
                        explainer = shap.TreeExplainer(self.model)
                    else:
                        # Fallback to KernelExplainer
                        explainer = shap.KernelExplainer(self.model.predict, features)
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(features)
                    
                    # Return explanation results
                    return {
                        'method': 'shap',
                        'shap_values': shap_values,
                        'features': features,
                        'feature_names': list(features.columns)
                    }
                except ImportError:
                    logger.warning("SHAP package not available. Falling back to permutation importance.")
                    method = 'permutation'
            
            if method == 'permutation':
                from sklearn.inspection import permutation_importance
                
                # Calculate permutation importance
                result = permutation_importance(self.model, features, self.predict(features), 
                                              n_repeats=10, random_state=42)
                
                # Return explanation results
                return {
                    'method': 'permutation',
                    'importances_mean': result.importances_mean,
                    'importances_std': result.importances_std,
                    'feature_names': list(features.columns)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error explaining predictions: {e}")
            return {}


    def version_model(self, version: str = None, metadata: Dict = None) -> str:
        """
        Version the model with metadata for tracking
        
        Args:
            version: Version string (e.g., '1.0.0')
            metadata: Dictionary of metadata to store with the model
            
        Returns:
            Version string
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot version model.")
                return ""
            
            # Generate version if not provided
            if version is None:
                from datetime import datetime
                version = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Create metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add version to metadata
            metadata['version'] = version
            metadata['timestamp'] = time.time()
            
            # Add model type and parameters
            if hasattr(self.model, 'get_params'):
                metadata['model_params'] = self.model.get_params()
            
            # Add feature names if available
            if self.feature_names:
                metadata['feature_names'] = self.feature_names
            
            # Store metadata in model
            self.model_version = version
            self.model_metadata = metadata
            
            logger.info(f"Model versioned as {version}")
            return version
            
        except Exception as e:
            logger.error(f"Error versioning model: {e}")
            return ""


    def export_for_deployment(self, export_path: str, format: str = 'onnx') -> bool:
        """
        Export the model for deployment in various formats
        
        Args:
            export_path: Path to export the model to
            format: Export format ('onnx', 'pmml', 'json', 'binary')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot export model.")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Export based on format
            if format == 'onnx':
                try:
                    # Check if onnx and onnxmltools are available
                    import onnx
                    import onnxmltools
                    
                    # Convert model to ONNX format
                    if hasattr(self.model, 'get_booster'):
                        # For XGBoost models
                        from onnxmltools.convert.xgboost import convert_xgboost
                        
                        # Get initial type for ONNX conversion
                        from onnxmltools.convert.common.data_types import FloatTensorType
                        initial_type = [('float_input', FloatTensorType([None, len(self.feature_names or [])]))]
                        
                        # Convert model to ONNX
                        onnx_model = convert_xgboost(self.model, initial_types=initial_type)
                        
                        # Save model
                        onnx.save_model(onnx_model, export_path)
                        logger.info(f"Model exported to ONNX format at {export_path}")
                        return True
                    else:
                        logger.error("Model is not an XGBoost model. Cannot export to ONNX.")
                        return False
                except ImportError:
                    logger.error("ONNX or onnxmltools not available. Cannot export to ONNX format.")
                    return False
            elif format == 'pmml':
                try:
                    # Check if nyoka is available
                    import nyoka
                    
                    # Export to PMML
                    nyoka.xgboost_to_pmml(self.model, self.feature_names or [], export_path)
                    logger.info(f"Model exported to PMML format at {export_path}")
                    return True
                except ImportError:
                    logger.error("Nyoka not available. Cannot export to PMML format.")
                    return False
            elif format == 'json':
                # Export to JSON format
                self.save_model(export_path, format='json')
                logger.info(f"Model exported to JSON format at {export_path}")
                return True
            elif format == 'binary':
                # Export to binary format
                self.save_model(export_path, format='binary')
                logger.info(f"Model exported to binary format at {export_path}")
                return True
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return False


    def monitor_performance(self, features: pd.DataFrame, target: pd.Series, 
                           metrics_path: str = None) -> Dict[str, float]:
        """
        Monitor model performance on new data and log metrics
        
        Args:
            features: Features to evaluate
            target: Target values
            metrics_path: Path to save metrics to
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot monitor performance.")
                return {}
            
            # Make predictions
            predictions = self.predict(features)
            
            # Calculate metrics based on model type
            metrics = {}
            if self.model_type == 'classifier':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                from sklearn.metrics import confusion_matrix, classification_report
                
                # Convert predictions to binary for classification metrics
                y_pred = (predictions > 0.5).astype(int)
                
                # Calculate basic metrics
                metrics['accuracy'] = float(accuracy_score(target, y_pred))
                metrics['precision'] = float(precision_score(target, y_pred, average='weighted'))
                metrics['recall'] = float(recall_score(target, y_pred, average='weighted'))
                metrics['f1'] = float(f1_score(target, y_pred, average='weighted'))
                
                # Add ROC AUC if binary classification
                if len(np.unique(target)) == 2:
                    metrics['roc_auc'] = float(roc_auc_score(target, predictions))
                
                # Add confusion matrix
                cm = confusion_matrix(target, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                
                # Add classification report
                report = classification_report(target, y_pred, output_dict=True)
                metrics['classification_report'] = report
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Calculate regression metrics
                metrics['mse'] = float(mean_squared_error(target, predictions))
                metrics['rmse'] = float(np.sqrt(mean_squared_error(target, predictions)))
                metrics['mae'] = float(mean_absolute_error(target, predictions))
                metrics['r2'] = float(r2_score(target, predictions))
            
            # Add timestamp
            metrics['timestamp'] = time.time()
            metrics['datetime'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add feature drift if we have previous data
            if hasattr(self, 'previous_features') and self.previous_features is not None:
                drift_scores = self.detect_drift(self.previous_features, features)
                metrics['feature_drift'] = drift_scores
            
            # Store current features for future drift detection
            self.previous_features = features.copy()
            
            # Log metrics
            logger.info(f"Model performance metrics: {metrics}")
            
            # Save metrics to file if path provided
            if metrics_path:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                    
                    # Save metrics to JSON file
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    logger.info(f"Metrics saved to {metrics_path}")
                except Exception as e:
                    logger.error(f"Error saving metrics to file: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            return {}


    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the model to a dictionary
        
        Returns:
            Dictionary representation of the model
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot serialize model.")
                return {}
            
            # Create model metadata
            model_dict = {
                'model_type': self.model_type,
                'use_gpu': self.use_gpu,
                'use_pytorch': self.use_pytorch,
                'feature_names': self.feature_names,
                'model_version': getattr(self, 'model_version', None),
                'model_metadata': getattr(self, 'model_metadata', {}),
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add model parameters if available
            if hasattr(self.model, 'get_params'):
                model_dict['model_params'] = self.model.get_params()
            
            # Add feature importance if available
            feature_importance = self.get_feature_importance()
            if feature_importance:
                model_dict['feature_importance'] = feature_importance
            
            # Save model to temporary file
            temp_path = f"temp_model_{int(time.time())}.json"
            self.save_model(temp_path, format='json')
            
            # Read model data
            with open(temp_path, 'r') as f:
                model_data = f.read()
            
            # Add model data to dictionary
            model_dict['model_data'] = model_data
            
            # Remove temporary file
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
            return model_dict
            
        except Exception as e:
            logger.error(f"Error serializing model: {e}")
            return {}
    
    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> 'XGBoostModel':
        """
        Create a model from a dictionary representation
        
        Args:
            model_dict: Dictionary representation of the model
            
        Returns:
            XGBoostModel instance
        """
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost is not available. Cannot deserialize model.")
                return None
            
            # Extract model data
            model_data = model_dict.get('model_data')
            if not model_data:
                logger.error("No model data found in dictionary")
                return None
            
            # Save model data to temporary file
            temp_path = f"temp_model_{int(time.time())}.json"
            with open(temp_path, 'w') as f:
                f.write(model_data)
            
            # Create model instance
            model = cls(
                model_path=temp_path,
                model_type=model_dict.get('model_type', 'classifier'),
                use_gpu=model_dict.get('use_gpu', None),
                use_pytorch=model_dict.get('use_pytorch', None),
                feature_names=model_dict.get('feature_names', None)
            )
            
            # Set additional attributes
            model.model_version = model_dict.get('model_version')
            model.model_metadata = model_dict.get('model_metadata', {})
            
            # Remove temporary file
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
            return model
            
        except Exception as e:
            logger.error(f"Error deserializing model: {e}")
            return None


    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model
        
        Returns:
            Dict of feature name to importance value
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            logger.error("XGBoost is not available or no model loaded. Cannot get feature importance.")
            return {}
        
        try:
            # For XGBoost models
            if hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                importance_type = "gain"
                
                # Get feature importance
                importance = booster.get_score(importance_type=importance_type)
                
                # If feature names are available, map them to the importance values
                if self.feature_names:
                    # Create a mapping of feature index to feature name
                    feature_map = {f"f{i}": name for i, name in enumerate(self.feature_names)}
                    
                    # Map the feature indices to feature names
                    importance = {feature_map.get(k, k): v for k, v in importance.items()}
                
                return importance
            
            # For scikit-learn models
            elif hasattr(self.model, 'feature_importances_'):
                return {str(i): float(v) for i, v in enumerate(self.model.feature_importances_)}
            
            return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

def load_pretrained_model(
    model_path: str,
    model_type: str = "classifier",
    use_gpu: bool = None,
    use_pytorch: bool = None,
) -> XGBoostModel:
    """
    Load a pretrained XGBoost model from a file
    
    Args:
        model_path: Path to the model file
        model_type: Type of model ('classifier' or 'regressor')
        use_gpu: Whether to use GPU acceleration
        use_pytorch: Whether to use PyTorch integration
        
    Returns:
        XGBoostModel instance with loaded model
    """
    try:
        # Create model instance
        model = XGBoostModel(
            model_path=model_path,
            model_type=model_type,
            use_gpu=use_gpu,
            use_pytorch=use_pytorch
        )
        
        # Check if model was loaded successfully
        if model.model is None:
            logger.error(f"Failed to load model from {model_path}")
            return XGBoostModel(model_type=model_type)
        
        logger.info(f"Successfully loaded pretrained model from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        # Return a default model instance
        return XGBoostModel(model_type=model_type)
