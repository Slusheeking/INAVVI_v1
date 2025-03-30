#!/usr/bin/env python3
"""
Base Trainer Module

This module provides the BaseTrainer class that all specialized trainers inherit from:
1. Common initialization and configuration
2. Shared utility methods
3. Standardized training interface
4. Metrics recording and model saving

The BaseTrainer ensures consistent behavior across all model trainers
and reduces code duplication.
"""

import json
import logging
import os
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.trainers.base")

# Import Prometheus client if available
try:
    import prometheus_client as prom
    from utils.metrics_registry import MODEL_EVALUATION_METRICS, FEATURE_IMPORTANCE, FEATURE_DRIFT, MODEL_VERSION_METRICS
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics will not be exposed.")


class BaseTrainer:
    """Base class for all model trainers"""

    def __init__(self, config, redis_client=None, model_type=None) -> None:
        """
        Initialize base trainer
        
        Args:
            config: Configuration dictionary
            redis_client: Redis client for caching and notifications
            model_type: Type of model being trained
        """
        self.config = config
        self.redis = redis_client
        self.model_type = model_type or "unknown"
        self.model_path = None
        self.metrics_path = None
        self.scaler_path = None
        self.feature_history_path = None
        self.model_version = config.get("model_version", "1.0.0")
        self.feature_drift_threshold = config.get("feature_drift_threshold", 0.1)
        self.enable_ab_testing = config.get("enable_ab_testing", False)
        self.ab_testing_ratio = config.get("ab_testing_ratio", 0.1)  # 10% of traffic to new model
        
        # Set up paths
        if "models_dir" in self.config:
            self.model_path = os.path.join(
                self.config["models_dir"], f"{self.model_type}_model"
            )
            self.metrics_path = os.path.join(
                self.config["models_dir"], f"{self.model_type}_metrics.json"
            )
            self.scaler_path = os.path.join(
                self.config["models_dir"], f"{self.model_type}_scaler.pkl"
            )
            self.feature_history_path = os.path.join(
                self.config["models_dir"], f"{self.model_type}_feature_history.json"
            )
        
        # Ensure models directory exists
        if "models_dir" in self.config and not os.path.exists(self.config["models_dir"]):
            try:
                os.makedirs(self.config["models_dir"], exist_ok=True)
                logger.info(f"Created models directory: {self.config['models_dir']}")
            except Exception as e:
                logger.error(f"Error creating models directory: {e}")

        # Initialize feature history
        self.feature_history = self.load_feature_history()

    def train(self, features, target, data_processor=None) -> bool | None:
        """
        Train model (to be implemented by subclasses)
        
        Args:
            features: Feature data
            target: Target data
            data_processor: Data processor instance
            
        Returns:
            True if training successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement train method")

    def track_feature_drift(self, feature_importance: Dict[str, float]) -> Dict[str, float]:
        """
        Track feature importance over time to detect drift
        
        Args:
            feature_importance: Dictionary of feature importance values
            
        Returns:
            Dictionary of feature drift values
        """
        timestamp = datetime.datetime.now().isoformat()
        drift_values = {}
        
        # Calculate drift for each feature by comparing with historical values
        for feature, importance in feature_importance.items():
            # Record current importance with timestamp
            if feature not in self.feature_history:
                self.feature_history[feature] = []
            
            self.feature_history[feature].append({"timestamp": timestamp, "importance": importance})
            
            # Calculate drift if we have historical data
            if len(self.feature_history[feature]) > 1:
                previous_importance = self.feature_history[feature][-2]["importance"]
                drift = abs(importance - previous_importance)
                drift_values[feature] = drift
                
                # Record drift in Prometheus if available
                if PROMETHEUS_AVAILABLE:
                    FEATURE_DRIFT.labels(
                        model_name=self.model_type,
                        feature_name=feature,
                        timestamp=timestamp
                    ).set(drift)
                
                # Log significant drift
                if drift > self.feature_drift_threshold:
                    logger.warning(
                        f"Significant feature drift detected for {feature}: {drift:.4f}"
                    )
            
        # Save updated feature history
        self.save_feature_history()
        
        return drift_values

    def load_feature_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load feature history from disk
        
        Returns:
            Dictionary of feature history
        """
        if not self.feature_history_path or not os.path.exists(self.feature_history_path):
            return {}
            
        try:
            with open(self.feature_history_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading feature history: {e}")
            return {}

    def save_feature_history(self) -> bool:
        """
        Save feature history to disk
        
        Returns:
            True if saving successful, False otherwise
        """
        if not self.feature_history_path:
            logger.error("Feature history path not set")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.feature_history_path), exist_ok=True)
            
            with open(self.feature_history_path, "w") as f:
                json.dump(self.feature_history, f)
                
            logger.info(f"Feature history saved to {self.feature_history_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving feature history: {e}")
            return False

    def register_model_version(self, metrics: Dict[str, float]) -> None:
        """
        Register model version metrics for A/B testing
        
        Args:
            metrics: Dictionary of model metrics
        """
        if not PROMETHEUS_AVAILABLE:
            return
            
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                MODEL_VERSION_METRICS.labels(
                    model_name=self.model_type,
                    version=self.model_version,
                    metric_name=metric_name
                ).set(value)
                
        logger.info(f"Registered metrics for model version {self.model_version}")

    def save_model(self, model, file_extension="pkl") -> bool:
        """
        Save model to disk
        
        Args:
            model: Model to save
            file_extension: File extension for model file
            
        Returns:
            True if saving successful, False otherwise
        """
        try:
            if self.model_path is None:
                logger.error("Model path not set")
                return False
                
            model_path = f"{self.model_path}.{file_extension}"
            
            # Add version to model path if A/B testing is enabled
            if self.enable_ab_testing:
                model_path = f"{self.model_path}_v{self.model_version}.{file_extension}"
            
            # Different saving methods based on model type
            if file_extension == "xgb":
                model.save_model(model_path)
            elif file_extension == "keras":
                model.save(model_path)
            elif file_extension == "json":
                # Check if model is XGBoostModel
                if hasattr(model, 'save_model'):
                    model.save_model(model_path, format='json')
                else:
                    # Fallback to regular JSON serialization
                    with open(model_path, 'w') as f:
                        json.dump(model, f)
            else:
                import joblib
                joblib.dump(model, model_path)
                
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return False

    def save_metrics(self, metrics) -> bool:
        """
        Save metrics to disk and update Redis
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            True if saving successful, False otherwise
        """
        try:
            if self.metrics_path is None:
                logger.error("Metrics path not set")
                return False
                
            # Save to disk
            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f)
                
            logger.info(f"Metrics saved to {self.metrics_path}")
            
            # Update Redis if available
            if self.redis:
                self.redis.hset(
                    "models:metrics", self.model_type, json.dumps(metrics)
                )
                logger.info(f"Metrics updated in Redis for {self.model_type}")
                
            # Record metrics in Prometheus if available
            if PROMETHEUS_AVAILABLE:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        MODEL_EVALUATION_METRICS.labels(
                            model_name=self.model_type, metric=metric_name
                        ).set(value)
                        
                # Record feature importance if available
                if "feature_importance" in metrics and isinstance(metrics["feature_importance"], dict):
                    for feature, importance in metrics["feature_importance"].items():
                        FEATURE_IMPORTANCE.labels(
                            model_name=self.model_type, feature=feature
                        ).set(importance)
                        
                    # Track feature drift
                    drift_values = self.track_feature_drift(metrics["feature_importance"])
                    metrics["feature_drift"] = drift_values
                    
                    # Register model version metrics for A/B testing
                    self.register_model_version(metrics)
                        
                logger.info(f"Metrics recorded in Prometheus for {self.model_type}")
                
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}", exc_info=True)
            return False

    def save_scaler(self, scaler) -> bool:
        """
        Save scaler to disk
        
        Args:
            scaler: Scaler to save
            
        Returns:
            True if saving successful, False otherwise
        """
        try:
            if self.scaler_path is None:
                logger.error("Scaler path not set")
                return False
                
            import joblib
            joblib.dump(scaler, self.scaler_path)
            logger.info(f"Scaler saved to {self.scaler_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving scaler: {e}", exc_info=True)
            return False

    def load_model(self, file_extension="pkl"):
        """
        Load model from disk

        For A/B testing, this will load the appropriate model version
        based on the ab_testing_ratio configuration.
        
        Args:
            file_extension: File extension for model file
            
        Returns:
            Loaded model or None if loading failed
        """
        try:
            if self.model_path is None:
                logger.error("Model path not set")
                return None
                
            model_path = f"{self.model_path}.{file_extension}"
            
            # Handle A/B testing
            if self.enable_ab_testing:
                # Check if we have multiple model versions
                import glob
                model_versions = glob.glob(f"{self.model_path}_v*.{file_extension}")
                
                if model_versions:
                    # Sort by version number
                    model_versions.sort()
                    # Use the latest version for the specified percentage of requests
                    if np.random.random() < self.ab_testing_ratio:
                        model_path = model_versions[-1]  # Latest version
                        logger.info(f"A/B testing: Using model version {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
                
            # Different loading methods based on model type
            if file_extension == "xgb":
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(model_path)
            elif file_extension == "keras":
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
            elif file_extension == "json":
                # Try to load as XGBoostModel first
                try:
                    from ml_engine.xgboost_model import XGBoostModel
                    model = XGBoostModel(model_path=model_path)
                except (ImportError, Exception) as e:
                    logger.warning(f"Could not load as XGBoostModel: {e}")
                    # Fallback to regular JSON deserialization
                    with open(model_path, 'r') as f:
                        model = json.load(f)
            else:
                import joblib
                model = joblib.load(model_path)
                
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return None

    def load_scaler(self):
        """
        Load scaler from disk
        
        Returns:
            Loaded scaler or None if loading failed
        """
        try:
            if self.scaler_path is None:
                logger.error("Scaler path not set")
                return None
                
            if not os.path.exists(self.scaler_path):
                logger.error(f"Scaler file not found: {self.scaler_path}")
                return None
                
            import joblib
            scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"Error loading scaler: {e}", exc_info=True)
            return None

    def get_model_config(self):
        """
        Get model configuration
        
        Returns:
            Model configuration dictionary
        """
        if self.model_type in self.config.get("model_configs", {}):
            return self.config["model_configs"][self.model_type]
        return {}

    def send_notification(self, message, level="info", details=None):
        """
        Send notification to frontend via Redis
        
        Args:
            message: Notification message
            level: Notification level (info, warning, error, success)
            details: Additional details
        """
        if not self.redis:
            return
            
        try:
            # Create notification object
            notification = {
                "type": "model_training",
                "model": self.model_type,
                "message": message,
                "level": level,
                "timestamp": time.time(),
                "details": details or {}
            }
            
            # Add to notifications list
            self.redis.lpush("frontend:notifications", json.dumps(notification))
            self.redis.ltrim("frontend:notifications", 0, 99)
            
            # Add to model_training category
            self.redis.lpush("frontend:model_training", json.dumps(notification))
            self.redis.ltrim("frontend:model_training", 0, 49)
            
            logger.info(f"Notification sent: {message}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")