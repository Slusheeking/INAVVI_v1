#!/usr/bin/env python3
"""
ML Model Trainer Base Module

This module provides the MLModelTrainer class, which is the main entry point for the ML engine:
1. Initializes GPU acceleration and optimization
2. Manages model training for different prediction tasks
3. Handles data loading and preprocessing
4. Provides prediction functionality
5. Integrates with monitoring and metrics collection

The MLModelTrainer uses the specialized trainers in the trainers package and
leverages the utility functions for feature selection, drift detection, etc.
"""

import json
import logging
import os
from datetime import datetime # Import datetime
import time
# from dotenv import load_dotenv # Config handles this
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.logging_config import get_logger
from utils.config import Config, get_config # Import Config system
from utils.exceptions import ConfigurationError, ModelError, DataError, RedisError # Import custom exceptions
from utils.gpu_utils import is_gpu_available, gpu_manager, clear_gpu_memory
from utils.metrics_registry import MODEL_TRAINING_TIME, PREDICTION_LATENCY, register_metrics_server_if_needed

from ml_engine.data_processor import MLDataProcessor
from ml_engine.trainers.signal_detection import SignalDetectionTrainer
from ml_engine.trainers.price_prediction import PricePredictionTrainer
from ml_engine.trainers.risk_assessment import RiskAssessmentTrainer
from ml_engine.trainers.exit_strategy import ExitStrategyTrainer
from ml_engine.trainers.market_regime import MarketRegimeTrainer
from ml_engine.utils import optimize_hyperparameters

# load_dotenv() # Handled by Config

# XGBoost settings will be loaded from Config object later

# Configure logging
logger = get_logger("ml_engine.base")

# Import Prometheus client if available
try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics will not be exposed.")

# Import XGBoost with error handling
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. Some functionality will be limited.")

# Import XGBoostModel with error handling
try:
    from ml_engine.xgboost_model import XGBoostModel
    XGBOOST_MODEL_AVAILABLE = True
except ImportError:
    XGBOOST_MODEL_AVAILABLE = False
    logger.warning("XGBoostModel class not available. Using direct XGBoost integration.")
    XGBoostModel = None # Define as None if not available


# Forward declaration for type hints if DataPipeline is used directly
if TYPE_CHECKING:
    from data_pipeline.main import DataPipeline
    from utils.redis_helpers import RedisClient

class MLModelTrainer:
    """
    ML Model Trainer for trading system
    Builds and trains models using live market data
    """

    def __init__(self, config: Config, redis_client: 'RedisClient', data_loader: 'DataPipeline') -> None:
        """
        Initialize ML Model Trainer

        Args:
            config: Centralized configuration object
            redis_client: Shared Redis client instance
            data_loader: Shared Data loader instance
        """
        self.config = config
        self.redis = redis_client # Use injected shared client
        self.data_loader = data_loader # Use injected shared loader
        self.logger = get_logger(__name__) # Use configured logger
        # Initialize GPU acceleration
        # Initialize GPU acceleration based on config
        self.use_gpu = self.config.get_bool("USE_GPU", True)
        self.gpu_available = False
        self.device_name = "CPU"
        self.device_memory = 0
        if self.use_gpu:
            self.gpu_available = is_gpu_available()
            if self.gpu_available:
                self.device_name = gpu_manager.device_name
                self.device_memory = gpu_manager.total_memory
                self.logger.info(f"GPU acceleration enabled: {self.device_name}")
            else:
                self.logger.warning("Config 'USE_GPU' is true, but no compatible GPU detected. Falling back to CPU.")
                self.use_gpu = False # Force disable if not available
        else:
             self.logger.info("GPU acceleration disabled by configuration ('USE_GPU': false).")

        # Load ML-specific configurations from the central Config object
        # These keys should be defined in DEFAULT_CONFIG in utils/config.py
        self.models_dir = self.config.get_path("MODEL_DIR", "./models")
        self.monitoring_dir = self.config.get_path("MONITORING_DIR", "./monitoring")
        self.data_dir = self.config.get_path("DATA_DIR", "./data")
        self.min_samples = self.config.get_int("ML_MIN_SAMPLES", 1000)
        self.lookback_days = self.config.get_int("ML_LOOKBACK_DAYS", 30)
        self.feature_selection_config = self.config.get_dict("ML_FEATURE_SELECTION", {
            "enabled": True, "method": "importance", "threshold": 0.01, "n_features": 20
        })
        self.time_series_cv_config = self.config.get_dict("ML_TIME_SERIES_CV", {
            "enabled": True, "n_splits": 5, "embargo_size": 10
        })
        self.monitoring_config = self.config.get_dict("ML_MONITORING", {
            "enabled": True, "drift_threshold": 0.05
        })
        self.test_size = self.config.get_float("ML_TEST_SIZE", 0.2)
        self.random_state = self.config.get_int("ML_RANDOM_STATE", 42)
        self.model_configs = self.config.get_dict("ML_MODEL_CONFIGS", {}) # Load model specifics

        # Load XGBoost specific settings from config
        self.xgboost_use_gpu = self.config.get_bool("XGBOOST_USE_GPU", True) and self.gpu_available
        self.xgboost_use_pytorch = self.config.get_bool("XGBOOST_USE_PYTORCH", True)
        self.xgboost_tree_method = self.config.get("XGBOOST_TREE_METHOD", "gpu_hist" if self.xgboost_use_gpu else "hist")
        self.xgboost_gpu_id = self.config.get_int("XGBOOST_GPU_ID", 0)

        # Load Redis keys from config
        self.redis_notify_key = self.config.get("REDIS_KEY_NOTIFICATIONS", "frontend:notifications")
        self.redis_notify_limit = self.config.get_int("REDIS_LIMIT_NOTIFICATIONS", 100)
        self.redis_category_limit = self.config.get_int("REDIS_LIMIT_CATEGORY", 50)
        self.redis_status_key = self.config.get("REDIS_KEY_SYSTEM_STATUS", "frontend:system:status")
        self.redis_model_info_key = self.config.get("REDIS_KEY_MODEL_INFO", "models:info")
        self.redis_predictions_prefix = self.config.get("REDIS_PREFIX_PREDICTIONS", "predictions:")

        # Ensure model directory exists
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             raise ConfigurationError(f"Failed to create models directory {self.models_dir}: {e}") from e

        # Initialize data processor
        # Initialize data processor (assuming it accepts Config)
        self.data_processor = MLDataProcessor(
            config=self.config, data_loader=self.data_loader, redis_client=self.redis
        )

        # Initialize tracking variables
        self.model_training_times = {}
        self.training_start_time = None

        # Initialize model trainers
        self._init_model_trainers()

        # Start Prometheus metrics server if enabled in config
        register_metrics_server_if_needed(self.config)

        self.logger.info("ML Model Trainer initialized")

    def _send_frontend_notification(self, message, level="info", category="ml_engine", details=None):
        """
        Send notification to frontend via Redis

        Args:
            message (str): Notification message
            level (str): Notification level (info, warning, error, success)
            category (str): Notification category for filtering
            details (dict): Additional details for the notification
        """
        # No need to check self.redis, assume it's initialized if we got here
        # if not self.redis:
        #     self.logger.debug(f"Redis not available, skipping notification: {message}")
        #     return

        try:
            # Create notification object
            notification = {
                "type": category,
                "message": message,
                "level": level,
                "timestamp": time.time(),
                "details": details or {}
            }

            # Add to general notifications list using configured key and limit
            self.redis.lpush(self.redis_notify_key, json.dumps(notification))
            self.redis.ltrim(self.redis_notify_key, 0, self.redis_notify_limit - 1)

            # Add to category-specific list using configured limit
            category_key = f"frontend:{category}" # Keep prefix for now, maybe configure later
            self.redis.lpush(category_key, json.dumps(notification))
            self.redis.ltrim(category_key, 0, self.redis_category_limit - 1)

            # Log based on level
            if level == "error":
                logger.error(f"Frontend notification: {message}")
            elif level == "warning":
                logger.warning(f"Frontend notification: {message}")
            else:
                logger.info(f"Frontend notification: {message}")

            # Update system status if this is a system-level notification
            if category in ["system_status", "ml_system"]:
                try:
                    system_status = json.loads(self.redis.get(self.redis_status_key) or "{}")
                    system_status["last_update"] = time.time()
                    system_status["last_message"] = message
                    system_status["status"] = level
                    self.redis.set(self.redis_status_key, json.dumps(system_status))
                except (json.JSONDecodeError, RedisError) as e: # Catch specific errors
                    self.logger.error(f"Error updating system status in Redis: {e}")
                except Exception as e: # Fallback
                     self.logger.error(f"Unexpected error updating system status: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error sending frontend notification: {e}", exc_info=True)

    def _init_model_trainers(self) -> None:
        """Initialize model trainers"""
        try:
            self.trainers = {
                "signal_detection": SignalDetectionTrainer(
                    config=self.config, # Pass central config
                    redis_client=self.redis,
                ),
                "price_prediction": PricePredictionTrainer(
                    config=self.config, # Pass central config
                    redis_client=self.redis,
                    use_gpu=self.use_gpu, # Pass determined GPU status
                ),
                "risk_assessment": RiskAssessmentTrainer(
                    config=self.config, # Pass central config
                    redis_client=self.redis,
                ),
                "exit_strategy": ExitStrategyTrainer(
                    config=self.config, # Pass central config
                    redis_client=self.redis,
                ),
                "market_regime": MarketRegimeTrainer(
                    config=self.config, # Pass central config
                    redis_client=self.redis,
                ),
            }
            # Filter out trainers whose model configs might be missing
            self.trainers = {name: trainer for name, trainer in self.trainers.items() if name in self.model_configs}
            if not self.trainers:
                 self.logger.warning("No valid model configurations found in ML_MODEL_CONFIGS. No trainers initialized.")
        except ImportError as e:
            logger.warning(f"Could not initialize all model trainers: {e!s}")
            # Continue with available trainers

    def train_all_models(self) -> None:
        """Train all trading models"""
        logger.info("Starting training for all models")

        # Send notification to frontend
        self._send_frontend_notification(
            message="Starting ML model training for all models",
            level="info",
            category="ml_training",
            details={
                "models": list(self.trainers.keys()),
                "gpu_enabled": self.use_gpu,
                "start_time": time.time()
            }
        )

        # Start tracking total training time
        self.training_start_time = time.time()

        # Run hyperparameter optimization if enabled
        # Check config for optimization flag
        if self.config.get_bool("ML_OPTIMIZE_HYPERPARAMS", False):
            self.logger.info("Running hyperparameter optimization (as configured)")
            # Assuming optimize_hyperparameters uses config internally or needs it passed
            try:
                 # Load historical data (might be redundant if _train_all_models also loads)
                 historical_data = self.data_processor.load_historical_data()
                 if historical_data is not None and not historical_data.empty:
                      # Example: Optimize signal detection model
                      # This function needs access to the central config now
                      optimize_hyperparameters(
                          historical_data,
                          "signal_detection",
                          self.config, # Pass central config
                          self.data_processor,
                      )
                      # TODO: Add optimization calls for other relevant models
                 else:
                      self.logger.error("Cannot run hyperparameter optimization: Failed to load historical data.")

            except ImportError:
                 self.logger.warning("Hyperparameter optimization library (e.g., Optuna) not available. Skipping optimization.")
            except Exception as e:
                 self.logger.error(f"Error during hyperparameter optimization: {e}", exc_info=True)

        # Continue with regular training
        self._train_all_models()

    def _train_all_models(self) -> bool | None:
        """Internal method to train all models with current hyperparameters"""
        try:
            # Load historical data
            logger.info("Loading historical data")
            historical_data = self.data_processor.load_historical_data()

            if historical_data is None or (
                isinstance(historical_data,
                           pd.DataFrame) and historical_data.empty
            ):
                logger.error("Failed to load sufficient historical data")
                self._send_frontend_notification("Failed to load historical data for training.", level="error", category="ml_training")
                return False

            # Store reference data for drift detection
            self.data_processor.store_reference_data(historical_data)

            # Train each model
            model_results = {}

            # Signal detection model
            if "signal_detection" in self.trainers:
                start_time = time.time()
                features, target = self.data_processor.prepare_signal_detection_data(
                    historical_data,
                )
                success = self.trainers["signal_detection"].train(
                    features, target, self.data_processor,
                )
                training_time = time.time() - start_time
                self.model_training_times["signal_detection"] = training_time
                model_results["signal_detection"] = {
                    "success": success,
                    "time": training_time,
                }

                # Record metrics if available
                if PROMETHEUS_AVAILABLE:
                    MODEL_TRAINING_TIME.labels(
                        model_name="signal_detection", 
                        model_type="classification"
                    ).observe(training_time)

            # Price prediction model
            if "price_prediction" in self.trainers:
                start_time = time.time()
                sequences, targets = self.data_processor.prepare_price_prediction_data(
                    historical_data,
                )
                success = self.trainers["price_prediction"].train(
                    sequences, targets)
                training_time = time.time() - start_time
                self.model_training_times["price_prediction"] = training_time
                model_results["price_prediction"] = {
                    "success": success,
                    "time": training_time,
                }

                # Record metrics if available
                if PROMETHEUS_AVAILABLE:
                    MODEL_TRAINING_TIME.labels(
                        model_name="price_prediction", 
                        model_type="regression"
                    ).observe(training_time)

            # Risk assessment model
            if "risk_assessment" in self.trainers:
                start_time = time.time()
                features, targets = self.data_processor.prepare_signal_detection_data(
                    historical_data,
                )  # Use same features but different target
                if "atr_pct" in historical_data.columns:
                    targets = historical_data["atr_pct"]
                    success = self.trainers["risk_assessment"].train(
                        features, targets, self.data_processor,
                    )
                    training_time = time.time() - start_time
                    self.model_training_times["risk_assessment"] = training_time
                    model_results["risk_assessment"] = {
                        "success": success,
                        "time": training_time,
                    }

                    # Record metrics if available
                    if PROMETHEUS_AVAILABLE:
                        MODEL_TRAINING_TIME.labels(
                            model_name="risk_assessment", 
                            model_type="regression"
                        ).observe(training_time)
                else:
                    logger.warning(
                        "No risk target variable (atr_pct) available")
                    model_results["risk_assessment"] = {
                        "success": False,
                        "error": "No target variable",
                    }

            # Exit strategy model
            if "exit_strategy" in self.trainers:
                start_time = time.time()
                features, _ = self.data_processor.prepare_signal_detection_data(
                    historical_data,
                )  # Use same features but different target
                if "optimal_exit" in historical_data.columns:
                    targets = historical_data["optimal_exit"]
                    success = self.trainers["exit_strategy"].train(
                        features, targets, self.data_processor,
                    )
                    training_time = time.time() - start_time
                    self.model_training_times["exit_strategy"] = training_time
                    model_results["exit_strategy"] = {
                        "success": success,
                        "time": training_time,
                    }

                    # Record metrics if available
                    if PROMETHEUS_AVAILABLE:
                        MODEL_TRAINING_TIME.labels(
                            model_name="exit_strategy", 
                            model_type="regression"
                        ).observe(training_time)
                else:
                    logger.warning(
                        "No exit strategy target variable (optimal_exit) available",
                    )
                    model_results["exit_strategy"] = {
                        "success": False,
                        "error": "No target variable",
                    }

            # Market regime model
            if "market_regime" in self.trainers:
                start_time = time.time()
                # Extract market features
                market_features = [
                    col
                    for col in historical_data.columns
                    if "spy_" in col or "vix_" in col
                ]
                if market_features:
                    market_data = historical_data[market_features].dropna()
                    success = self.trainers["market_regime"].train(market_data)
                    training_time = time.time() - start_time
                    self.model_training_times["market_regime"] = training_time
                    model_results["market_regime"] = {
                        "success": success,
                        "time": training_time,
                    }

                    # Record metrics if available
                    if PROMETHEUS_AVAILABLE:
                        MODEL_TRAINING_TIME.labels(
                            model_name="market_regime", 
                            model_type="clustering"
                        ).observe(training_time)
                else:
                    logger.warning(
                        "No market features available for regime detection")
                    model_results["market_regime"] = {
                        "success": False,
                        "error": "No market features",
                    }

            # Update Redis with model info
            self.update_model_info()

            # Calculate total training time
            total_training_time = time.time() - self.training_start_time

            # Clear GPU memory if available
            if self.use_gpu and self.gpu_available:
                clear_gpu_memory()
                logger.info("Cleared GPU memory after training")

            # Send notification to frontend about successful training
            self._send_frontend_notification(
                message=f"All models trained successfully in {total_training_time:.2f} seconds",
                level="success",
                category="ml_training",
                details={
                    "total_time": total_training_time,
                    "model_results": model_results,
                    "gpu_used": self.use_gpu if hasattr(self, "use_gpu") else False,
                    "training_times": self.model_training_times,
                    "model_configs_used": {k: self.model_configs.get(k, {}).get('params', {}) for k in model_results}
                }
            )

            logger.info(
                f"All models trained successfully in {total_training_time:.2f} seconds",
            )
            return True

        except DataError as e:
             self.logger.error(f"Data error during model training: {e}", exc_info=True)
             self._send_frontend_notification(f"Data error during training: {e}", level="error", category="ml_training")
             return False
        except ModelError as e:
             self.logger.error(f"Model training error: {e}", exc_info=True)
             self._send_frontend_notification(f"Model training error: {e}", level="error", category="ml_training")
             return False
        except Exception as e:
            self.logger.error(f"Unexpected error training models: {e}", exc_info=True)
            self._send_frontend_notification(f"Unexpected error during training: {e}", level="error", category="ml_training")
            return False

    def update_model_info(self) -> None:
        """Update Redis with model information"""
        try:
            # Collect model info
            models_info = {}

            for model_name, model_config in self.model_configs.items():
                model_type = model_config.get('type', 'unknown')
                # Determine expected file extension based on type (could be more robust)
                if model_type == 'xgboost':
                     # Prefer .json if available, fallback to .xgb
                     model_paths = [
                          self.models_dir / f"{model_name}_model.json",
                          self.models_dir / f"{model_name}_model.xgb"
                     ]
                     model_path = next((path for path in model_paths if path.exists()), None)
                elif model_type in ['random_forest', 'kmeans', 'scaler']: # Added scaler
                     model_path = self.models_dir / f"{model_name}_model.pkl"
                elif model_type == 'lstm': # Assuming Keras/TF format
                     model_path = self.models_dir / f"{model_name}_model.keras" # Or SavedModel directory
                else:
                     model_path = None # Unknown type

                # Check scaler path separately if needed (e.g., for signal detection)
                scaler_path = self.models_dir / f"{model_name}_scaler.pkl"

                model_info_entry = {"type": model_type}
                if model_path and model_path.exists():
                    file_stats = model_path.stat()
                    model_info_entry.update({
                        "path": str(model_path), # Store as string for JSON
                        "size_bytes": file_stats.st_size,
                        "last_modified": int(file_stats.st_mtime),
                        "last_modified_str": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
                else:
                     model_info_entry["status"] = "Missing"

                # Add scaler info if it exists
                if scaler_path.exists():
                     scaler_stats = scaler_path.stat()
                     model_info_entry["scaler"] = {
                          "path": str(scaler_path),
                          "size_bytes": scaler_stats.st_size,
                          "last_modified": int(scaler_stats.st_mtime),
                     }

                models_info[model_name] = model_info_entry

            # Update Redis
            if self.redis:
                self.redis.set(self.redis_model_info_key, json.dumps(models_info))

            logger.info(f"Updated model info for {len(models_info)} models")

        except Exception as e:
            logger.error(f"Error updating model info: {e}", exc_info=True)

    def predict_signals(self, market_data):
        """
        Predict trading signals using trained models

        Args:
            market_data: DataFrame with latest market data

        Returns:
            Dictionary of ticker -> prediction results
        """
        try:
            start_time = time.time()
            logger.info(f"Making predictions with {len(market_data)} samples")

            # Prepare data
            features, _ = self.data_processor.prepare_signal_detection_data(
                market_data)

            if features.empty:
                logger.warning("No valid features for prediction")
                return {}

            # Check if we should use XGBoostModel or direct XGBoost
            if XGBOOST_MODEL_AVAILABLE:
                # Use XGBoostModel for prediction
                logger.info("Using XGBoostModel for signal prediction")
                
                # Check for model files - try both .json and .xgb formats
                # Use configured models_dir (Path object)
                model_paths = [
                    self.models_dir / "signal_detection_model.json",
                    self.models_dir / "signal_detection_model.xgb"
                ]
                signal_model_path = next((path for path in model_paths if path.exists()), None)
                signal_scaler_path = self.models_dir / "signal_detection_scaler.pkl"

                if not signal_model_path or not signal_scaler_path.exists():
                    msg = "Signal detection model or scaler file not found."
                    logger.error(msg)
                    self._send_frontend_notification(msg, level="error", category="ml_prediction")
                    raise ModelError(msg) # Raise specific error

                # Load scaler
                import joblib
                signal_scaler = joblib.load(signal_scaler_path)

                # Scale features
                features_scaled = signal_scaler.transform(features)
                
                # Convert scaled features to DataFrame for XGBoostModel
                features_df = pd.DataFrame(features_scaled, columns=features.columns)
                
                # Create XGBoostModel instance
                model = XGBoostModel(
                    model_path=str(signal_model_path), # Pass path as string
                    model_type="classifier",
                    use_gpu=self.xgboost_use_gpu, # Use loaded config
                    use_pytorch=self.xgboost_use_pytorch, # Use loaded config
                    tree_method=self.xgboost_tree_method, # Use loaded config
                    gpu_id=self.xgboost_gpu_id # Use loaded config
                )
                
                # Make predictions
                signal_scores = model.predict(features_df)
                
            else:
                # Use direct XGBoost for prediction
                logger.info("Using direct XGBoost for signal prediction")
                
                # Check for model files
                # Use configured models_dir (Path object)
                signal_model_path = self.models_dir / "signal_detection_model.xgb"
                signal_scaler_path = self.models_dir / "signal_detection_scaler.pkl"

                if not signal_model_path.exists() or not signal_scaler_path.exists():
                    msg = "Signal detection model (.xgb) or scaler file not found."
                    logger.error(msg)
                    self._send_frontend_notification(msg, level="error", category="ml_prediction")
                    raise ModelError(msg) # Raise specific error

                # Load model and scaler
                import joblib
                signal_model = xgb.Booster()
                signal_model.load_model(signal_model_path)
                signal_scaler = joblib.load(signal_scaler_path)

                # Scale features
                features_scaled = signal_scaler.transform(features)

                # Make predictions
                dmatrix = xgb.DMatrix(features_scaled)
                signal_scores = signal_model.predict(dmatrix)

            # Organize predictions by ticker
            predictions = {}

            # Add ticker info
            if "ticker" in market_data.columns:
                for i, ticker in enumerate(market_data["ticker"]):
                    if i < len(signal_scores):
                        if ticker not in predictions:
                            predictions[ticker] = {
                                "signal_score": float(signal_scores[i]),
                                "signal": 1 if signal_scores[i] > 0.5 else 0,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }

            # Record prediction latency in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                prediction_time = time.time() - start_time
                PREDICTION_LATENCY.labels(model_name="signal_detection").observe(
                    prediction_time,
                )
                logger.info(
                    f"Prediction latency: {prediction_time:.4f} seconds")

            logger.info(
                f"Generated predictions for {len(predictions)} tickers")

            # Update predictions in Redis if available
            if self.redis:
                for ticker, pred in predictions.items():
                    # Use configured prefix
                    self.redis.hset(f"{self.redis_predictions_prefix}{ticker}", "signal", json.dumps(pred))

                # Send notification to frontend about new predictions
                try:
                    # Count positive signals
                    positive_signals = sum(
                        1 for p in predictions.values() if p.get("signal") == 1)

                    # Create notification for frontend
                    notification = {
                        "type": "ml_predictions",
                        "message": f"Generated predictions for {len(predictions)} tickers ({positive_signals} buy signals)",
                        "level": "info",
                        "timestamp": time.time(),
                        "details": {
                            "total_predictions": len(predictions),
                            "positive_signals": positive_signals,
                            "prediction_time": time.time() - start_time,
                            "tickers_with_signals": [ticker for ticker, pred in predictions.items() if pred.get("signal") == 1]
                        }
                    }

                    # Push to notifications list
                    # Use configured keys and limits
                    self.redis.lpush(self.redis_notify_key, json.dumps(notification))
                    self.redis.ltrim(self.redis_notify_key, 0, self.redis_notify_limit - 1)

                    # Also store in ml_predictions category
                    category_key = "frontend:ml_predictions" # Keep prefix for now
                    self.redis.lpush(category_key, json.dumps(notification))
                    self.redis.ltrim(category_key, 0, self.redis_category_limit - 1)

                    # Update system status
                    # Use configured status key
                    system_status = json.loads(self.redis.get(self.redis_status_key) or "{}")
                    system_status["last_prediction"] = time.time()
                    system_status["prediction_count"] = system_status.get("prediction_count", 0) + 1
                    system_status["last_positive_signals"] = positive_signals
                    self.redis.set(self.redis_status_key, json.dumps(system_status))

                    logger.info(
                        f"Prediction notification sent to frontend: {positive_signals} buy signals")
                except Exception as e:
                    logger.error(f"Error sending prediction notification: {e}", exc_info=True)

            return predictions

        except ModelError as e: # Catch model loading/prediction errors
             logger.error(f"Model error during prediction: {e}", exc_info=True)
             self._send_frontend_notification(f"Model error during prediction: {e}", level="error", category="ml_prediction")
             return {}
        except DataError as e: # Catch data preparation errors
             logger.error(f"Data error during prediction: {e}", exc_info=True)
             self._send_frontend_notification(f"Data error during prediction: {e}", level="error", category="ml_prediction")
             return {}
        except Exception as e: # General fallback
            logger.error(f"Unexpected error making predictions: {e}", exc_info=True)
            self._send_frontend_notification(f"Unexpected prediction error: {e}", level="error", category="ml_prediction")
            # Record error in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                from utils.metrics_registry import DRIFT_DETECTION
                DRIFT_DETECTION.labels(
                    model_name="signal_detection", result="error",
                ).inc()
            return {}