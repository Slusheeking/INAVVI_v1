#!/usr/bin/env python3
"""
Signal Detection Trainer Module

This module provides the SignalDetectionTrainer class for training signal detection models:
1. Uses XGBoost for binary classification
2. Applies feature selection and scaling
3. Handles time series cross-validation
4. Evaluates model performance with classification metrics
5. Saves model, scaler, and metrics

The signal detection model predicts trading signals (buy/sell) based on market data.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_engine.trainers.base import BaseTrainer
from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.trainers.signal_detection")

# Import XGBoost with error handling
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. Signal detection trainer will not work.")

# Import XGBoostModel with error handling
try:
    from ml_engine.xgboost_model import XGBoostModel
    XGBOOST_MODEL_AVAILABLE = True
except ImportError:
    XGBOOST_MODEL_AVAILABLE = False
    logger.warning("XGBoostModel not available. Using direct XGBoost integration.")


class SignalDetectionTrainer(BaseTrainer):
    """Trainer for signal detection model using XGBoost"""

    def __init__(self, config, redis_client=None) -> None:
        """
        Initialize signal detection trainer
        
        Args:
            config: Configuration dictionary
            redis_client: Redis client for caching and notifications
        """
        super().__init__(config, redis_client, "signal_detection")
        
        # Check if XGBoost is available
        if not XGB_AVAILABLE:
            logger.error(
                "XGBoost is not available. Cannot train signal detection model.",
            )
            msg = "XGBoost is required for signal detection model"
            raise ImportError(msg)

    def train(self, features, target, data_processor=None) -> bool | None:
        """
        Train signal detection model
        
        Args:
            features: Feature DataFrame
            target: Target Series
            data_processor: Data processor instance
            
        Returns:
            True if training successful, False otherwise
        """
        logger.info("Training signal detection model")

        try:
            if len(features) == 0 or len(target) == 0:
                logger.error("No valid data for signal detection model")
                return False

            # Apply feature selection if enabled
            if self.config["feature_selection"]["enabled"] and data_processor:
                features = data_processor.select_features(
                    features, target, "classification",
                )

            # Use time series cross-validation if enabled
            if self.config["time_series_cv"]["enabled"] and data_processor:
                # Create time series split
                splits = data_processor.create_time_series_splits(
                    features, target)

                # Use the last split for final evaluation
                train_idx, test_idx = splits[-1]
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            else:
                # Use traditional train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    features,
                    target,
                    test_size=self.config["test_size"],
                    random_state=self.config["random_state"],
                )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Save scaler
            self.save_scaler(scaler)

            # Get model config
            model_config = self.get_model_config()

            # Check for optimized parameters
            optimized_params_path = os.path.join(
                self.config["models_dir"], "signal_detection_optimized_params.json",
            )
            if os.path.exists(optimized_params_path):
                with open(optimized_params_path) as f:
                    model_config["params"].update(json.load(f))

            # Check if we should use XGBoostModel or direct XGBoost
            if XGBOOST_MODEL_AVAILABLE:
                # Train using XGBoostModel
                logger.info("Training signal detection model using XGBoostModel")
                
                # Convert scaled arrays back to DataFrames for XGBoostModel
                X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                
                # Create model with parameters
                model = XGBoostModel(
                    model_type="classifier",
                    use_gpu=True,
                    use_pytorch=True,
                    feature_names=list(X_train.columns)
                )
                
                # Set model parameters
                params = {
                    k: v
                    for k, v in model_config["params"].items()
                    if k != "n_estimators"
                }
                
                # Add number of estimators
                num_boost_round = model_config["params"].get("n_estimators", 200)
                
                # Train the model
                model.train(
                    X_train_df, 
                    y_train,
                    eval_set=[(X_train_df, y_train), (pd.DataFrame(X_test_scaled, columns=X_test.columns), y_test)],
                    early_stopping_rounds=20,
                    verbose=False,
                    **params
                )
                
                # Make predictions
                X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                y_pred = model.predict(X_test_df)
                
                # Save model in JSON format
                model_path = os.path.join(self.config["models_dir"], "signal_detection_model.json")
                model.save_model(model_path, format='json')
                
                # Also save in XGB format for backward compatibility
                xgb_model_path = os.path.join(self.config["models_dir"], "signal_detection_model.xgb")
                if hasattr(model.model, 'save_model'):
                    model.model.save_model(xgb_model_path)
                
            else:
                # Train using direct XGBoost
                logger.info("Training XGBoost signal detection model directly")
                dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
                dtest = xgb.DMatrix(X_test_scaled, label=y_test)

                eval_list = [(dtrain, "train"), (dtest, "test")]

                # Create a copy of params without n_estimators to avoid warning
                model = xgb.train(
                    params={
                        k: v
                        for k, v in model_config["params"].items()
                        if k != "n_estimators"
                    },
                    dtrain=dtrain,
                    evals=eval_list,
                    num_boost_round=model_config["params"].get(
                        "n_estimators", 200),
                    early_stopping_rounds=20,
                    verbose_eval=False,
                )
                
                # Make predictions
                y_pred = model.predict(dtest)
                
                # Save model
                self.save_model(model, "xgb")

            # Convert predictions to binary
            y_pred_binary = (y_pred > 0.5).astype(int)

            # Check if we have multiple classes in the test set
            unique_classes = np.unique(y_test)
            if len(unique_classes) < 2:
                logger.warning(
                    f"Only one class present in test set: {unique_classes}. Using simplified metrics.",
                )
                accuracy = accuracy_score(y_test, y_pred_binary)
                metrics = {"accuracy": float(accuracy)}
                logger.info(
                    f"Signal detection model metrics - Accuracy: {accuracy:.4f}",
                )
            else:
                accuracy = accuracy_score(y_test, y_pred_binary)
                precision = precision_score(y_test, y_pred_binary)
                recall = recall_score(y_test, y_pred_binary)
                f1 = f1_score(y_test, y_pred_binary)
                auc = roc_auc_score(y_test, y_pred)
                
                # Get feature importance
                if XGBOOST_MODEL_AVAILABLE and hasattr(model, 'model'):
                    # Get feature importance from XGBoostModel
                    try:
                        # Try to use get_feature_importance method first
                        if hasattr(model, 'get_feature_importance'):
                            feature_importance = model.get_feature_importance()
                        # Fall back to model's feature_importances_ attribute
                        elif hasattr(model.model, 'feature_importances_'):
                            feature_names = model.feature_names or [f"f{i}" for i in range(len(model.model.feature_importances_))]
                            feature_importance = {
                                str(k): float(v)
                                for k, v in zip(feature_names, model.model.feature_importances_)
                            }
                        else:
                            # Default empty dict if no feature importance available
                            feature_importance = {}
                            logger.warning("No feature importance available from model")
                    except Exception as e:
                        logger.warning(f"Error getting feature importance: {e}")
                        feature_importance = {}
                else:
                    # Get feature importance from direct XGBoost model
                    try:
                        # Try to use feature_importances_ attribute
                        if hasattr(model, 'feature_importances_'):
                            feature_names = getattr(model, 'feature_names_in_', [f"f{i}" for i in range(len(model.feature_importances_))])
                            feature_importance = {
                                str(k): float(v)
                                for k, v in zip(feature_names, model.feature_importances_)
                            }
                        else:
                            # Default empty dict if no feature importance available
                            feature_importance = {}
                            logger.warning("No feature importance available from model")
                    except Exception as e:
                        logger.warning(f"Error getting feature importance: {e}")
                        feature_importance = {}
                
                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "auc": float(auc),
                    "feature_importance": feature_importance,
                }
                logger.info(
                    f"Signal detection model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}",
                )

            # Save metrics
            self.save_metrics(metrics)

            # Send notification
            self.send_notification(
                message="Signal detection model trained successfully",
                level="success",
                details=metrics
            )

            logger.info("Signal detection model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training signal detection model: {e!s}", exc_info=True,
            )
            
            # Send notification
            self.send_notification(
                message=f"Error training signal detection model: {str(e)}",
                level="error",
                details={"error": str(e)}
            )
            
            return False