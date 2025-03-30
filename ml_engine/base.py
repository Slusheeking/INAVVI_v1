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
import time
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.logging_config import get_logger
from utils.gpu_utils import is_gpu_available, gpu_manager, clear_gpu_memory
from utils.metrics_registry import MODEL_TRAINING_TIME, PREDICTION_LATENCY

from ml_engine.data_processor import MLDataProcessor
from ml_engine.trainers.signal_detection import SignalDetectionTrainer
from ml_engine.trainers.price_prediction import PricePredictionTrainer
from ml_engine.trainers.risk_assessment import RiskAssessmentTrainer
from ml_engine.trainers.exit_strategy import ExitStrategyTrainer
from ml_engine.trainers.market_regime import MarketRegimeTrainer
from ml_engine.utils import optimize_hyperparameters

# Load environment variables
load_dotenv()

# Get XGBoost configuration from environment variables
XGBOOST_USE_GPU = os.environ.get("XGBOOST_USE_GPU", "true").lower() == "true"
XGBOOST_USE_PYTORCH = os.environ.get("XGBOOST_USE_PYTORCH", "true").lower() == "true"
XGBOOST_TREE_METHOD = os.environ.get("XGBOOST_TREE_METHOD", "gpu_hist")

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
    logger.warning("XGBoostModel not available. Using direct XGBoost integration.")


class MLModelTrainer:
    """
    ML Model Trainer for trading system
    Builds and trains models using live market data
    """

    def __init__(self, redis_client, data_loader) -> None:
        """
        Initialize ML Model Trainer
        
        Args:
            redis_client: Redis client for caching and notifications
            data_loader: Data loader instance
        """
        self.redis = redis_client
        self.data_loader = data_loader

        # Initialize GPU acceleration
        self.use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
        if self.use_gpu:
            # Use the GPU manager from utils.gpu_utils
            self.gpu_available = is_gpu_available()
            if self.gpu_available:
                logger.info(f"GPU acceleration enabled: {gpu_manager.device_name}")
                self.device_name = gpu_manager.device_name
                self.device_memory = gpu_manager.total_memory
            else:
                logger.warning("GPU acceleration requested but no GPU is available")
                self.use_gpu = False

        # Configuration
        self.config = {
            "models_dir": os.environ.get("MODELS_DIR", "./models"),
            "monitoring_dir": os.environ.get("MONITORING_DIR", "./monitoring"),
            "data_dir": os.environ.get("DATA_DIR", "./data"),
            "min_samples": 1000,
            "lookback_days": 30,
            "feature_selection": {
                "enabled": True,
                "method": "importance",  # 'importance', 'rfe', 'mutual_info'
                "threshold": 0.01,  # For importance-based selection
                "n_features": 20,  # For RFE
            },
            "time_series_cv": {
                "enabled": True,
                "n_splits": 5,
                "embargo_size": 10,  # Number of samples to exclude between train and test
            },
            "monitoring": {"enabled": True, "drift_threshold": 0.05},
            "test_size": 0.2,
            "random_state": 42,
            "model_configs": {
                "signal_detection": {
                    "type": "xgboost",
                    "params": {
                        "max_depth": 6,
                        "learning_rate": 0.03,
                        "subsample": 0.8,
                        "n_estimators": 200,
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                    },
                },
                "price_prediction": {
                    "type": "lstm",
                    "params": {
                        "units": [64, 32],
                        "dropout": 0.3,
                        "epochs": 50,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                    },
                },
                "risk_assessment": {
                    "type": "random_forest",
                    "params": {
                        "n_estimators": 100,
                        "max_depth": 6,
                        "max_features": "sqrt",
                        "min_samples_leaf": 30,
                    },
                },
                "exit_strategy": {
                    "type": "xgboost",
                    "params": {
                        "max_depth": 5,
                        "learning_rate": 0.02,
                        "subsample": 0.8,
                        "n_estimators": 150,
                        "objective": "reg:squarederror",
                    },
                },
                "market_regime": {
                    "type": "kmeans",
                    "params": {"n_clusters": 4, "random_state": 42},
                },
            },
        }

        # Initialize data processor
        self.data_processor = MLDataProcessor(
            data_loader=self.data_loader, redis_client=self.redis, config=self.config,
        )

        # Initialize tracking variables
        self.model_training_times = {}
        self.training_start_time = None

        # Initialize model trainers
        self._init_model_trainers()

        logger.info("ML Model Trainer initialized")

    def _send_frontend_notification(self, message, level="info", category="ml_engine", details=None):
        """
        Send notification to frontend via Redis

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
            if category in ["system_status", "ml_system"]:
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

    def _init_model_trainers(self) -> None:
        """Initialize model trainers"""
        try:
            self.trainers = {
                "signal_detection": SignalDetectionTrainer(
                    config=self.config,
                    redis_client=self.redis,
                ),
                "price_prediction": PricePredictionTrainer(
                    config=self.config,
                    redis_client=self.redis,
                    use_gpu=self.use_gpu if hasattr(self, "use_gpu") else False,
                ),
                "risk_assessment": RiskAssessmentTrainer(
                    config=self.config,
                    redis_client=self.redis,
                ),
                "exit_strategy": ExitStrategyTrainer(
                    config=self.config,
                    redis_client=self.redis,
                ),
                "market_regime": MarketRegimeTrainer(
                    config=self.config,
                    redis_client=self.redis,
                ),
            }
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
                "gpu_enabled": self.use_gpu if hasattr(self, "use_gpu") else False,
                "start_time": time.time()
            }
        )

        # Start tracking total training time
        self.training_start_time = time.time()

        # Run hyperparameter optimization if enabled
        if os.environ.get("OPTIMIZE_HYPERPARAMS", "false").lower() == "true":
            logger.info("Running hyperparameter optimization")
            if PROMETHEUS_AVAILABLE:
                # Load historical data
                historical_data = self.data_processor.load_historical_data()

                if historical_data is not None and not historical_data.empty:
                    # Optimize signal detection model
                    optimize_hyperparameters(
                        historical_data,
                        "signal_detection",
                        self.config,
                        self.data_processor,
                    )
            else:
                logger.warning(
                    "Optuna not available. Skipping hyperparameter optimization.",
                )

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
                    "training_times": self.model_training_times
                }
            )

            logger.info(
                f"All models trained successfully in {total_training_time:.2f} seconds",
            )
            return True

        except Exception as e:
            logger.error(f"Error training models: {e!s}", exc_info=True)
            return False

    def update_model_info(self) -> None:
        """Update Redis with model information"""
        try:
            # Collect model info
            models_info = {}

            for model_name, config in self.config["model_configs"].items():
                # Check for both .xgb and .json formats for XGBoost models
                if config['type'] == 'xgboost':
                    model_paths = [
                        os.path.join(self.config["models_dir"], f"{model_name}_model.xgb"),
                        os.path.join(self.config["models_dir"], f"{model_name}_model.json")
                    ]
                    # Use the first path that exists
                    model_path = next((path for path in model_paths if os.path.exists(path)), None)
                else:
                    model_path = os.path.join(
                        self.config["models_dir"],
                        f"{model_name}_model.{'pkl' if config['type'] in ['random_forest', 'kmeans'] else 'keras'}",
                    )

                if model_path and os.path.exists(model_path):
                    file_stats = os.stat(model_path)

                    models_info[model_name] = {
                        "type": config["type"],
                        "path": model_path,
                        "size_bytes": file_stats.st_size,
                        "last_modified": int(file_stats.st_mtime),
                        "last_modified_str": time.strftime(
                            "%Y-%m-%d %H:%M:%S", 
                            time.localtime(file_stats.st_mtime)
                        ),
                    }

            # Update Redis
            if self.redis:
                self.redis.set("models:info", json.dumps(models_info))

            logger.info(f"Updated model info for {len(models_info)} models")

        except Exception as e:
            logger.error(f"Error updating model info: {e!s}", exc_info=True)

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
                model_paths = [
                    os.path.join(self.config["models_dir"], "signal_detection_model.json"),
                    os.path.join(self.config["models_dir"], "signal_detection_model.xgb")
                ]
                signal_model_path = next((path for path in model_paths if os.path.exists(path)), None)
                signal_scaler_path = os.path.join(
                    self.config["models_dir"], "signal_detection_scaler.pkl",
                )

                if not signal_model_path or not os.path.exists(signal_scaler_path):
                    logger.error("Signal detection model or scaler not found")
                    return {}

                # Load scaler
                import joblib
                signal_scaler = joblib.load(signal_scaler_path)

                # Scale features
                features_scaled = signal_scaler.transform(features)
                
                # Convert scaled features to DataFrame for XGBoostModel
                features_df = pd.DataFrame(features_scaled, columns=features.columns)
                
                # Create XGBoostModel instance
                model = XGBoostModel(
                    model_path=signal_model_path,
                    model_type="classifier",
                    use_gpu=self.use_gpu if hasattr(self, "use_gpu") else False,
                    use_pytorch=XGBOOST_USE_PYTORCH,
                    tree_method=XGBOOST_TREE_METHOD if self.use_gpu else 'hist',
                    gpu_id=int(os.environ.get("XGBOOST_GPU_ID", "0"))
                )
                
                # Make predictions
                signal_scores = model.predict(features_df)
                
            else:
                # Use direct XGBoost for prediction
                logger.info("Using direct XGBoost for signal prediction")
                
                # Check for model files
                signal_model_path = os.path.join(
                    self.config["models_dir"], "signal_detection_model.xgb",
                )
                signal_scaler_path = os.path.join(
                    self.config["models_dir"], "signal_detection_scaler.pkl",
                )

                if not os.path.exists(signal_model_path) or not os.path.exists(
                    signal_scaler_path,
                ):
                    logger.error("Signal detection model or scaler not found")
                    return {}

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
                    self.redis.hset(
                        f"predictions:{ticker}", "signal", json.dumps(pred))

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
                    self.redis.lpush("frontend:notifications",
                                     json.dumps(notification))
                    self.redis.ltrim("frontend:notifications", 0, 99)

                    # Also store in ml_predictions category
                    self.redis.lpush("frontend:ml_predictions",
                                     json.dumps(notification))
                    self.redis.ltrim("frontend:ml_predictions", 0, 49)

                    # Update system status
                    system_status = json.loads(self.redis.get(
                        "frontend:system:status") or "{}")
                    system_status["last_prediction"] = time.time()
                    system_status["prediction_count"] = system_status.get(
                        "prediction_count", 0) + 1
                    system_status["last_positive_signals"] = positive_signals
                    self.redis.set("frontend:system:status",
                                   json.dumps(system_status))

                    logger.info(
                        f"Prediction notification sent to frontend: {positive_signals} buy signals")
                except Exception as e:
                    logger.error(f"Error sending prediction notification: {e}")

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e!s}", exc_info=True)
            # Record error in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                from utils.metrics_registry import DRIFT_DETECTION
                DRIFT_DETECTION.labels(
                    model_name="signal_detection", result="error",
                ).inc()
            return {}