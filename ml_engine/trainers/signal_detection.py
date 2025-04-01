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
import time # Import time
from datetime import datetime # Import datetime
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
from utils.config import Config # Import Config
from utils.exceptions import ModelTrainingError, ConfigurationError, DataError # Import custom exceptions

# Configure logging
logger = get_logger("ml_engine.trainers.signal_detection")

# Import XGBoost with error handling
# XGBoost import check remains important
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    # Error logged in __init__ if needed

# Import XGBoostModel with error handling
# XGBoostModel import check remains important
try:
    from ml_engine.xgboost_model import XGBoostModel
    XGBOOST_MODEL_AVAILABLE = True
except ImportError:
    XGBOOST_MODEL_AVAILABLE = False
    XGBoostModel = None # Define as None if not available
    # Warning logged in __init__ if needed


# Import MLflow if available
try:
    import mlflow
    import mlflow.xgboost
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Warning will be logged if used without being available

class SignalDetectionTrainer(BaseTrainer):
    """Trainer for signal detection model using XGBoost"""

    def __init__(self, config: Config, redis_client=None) -> None: # Accept Config object
        """
        Initialize signal detection trainer

        Args:
            config: Centralized configuration object.
            redis_client: Optional shared Redis client instance.
        """
        # Pass model_name to BaseTrainer constructor
        super().__init__(config, redis_client, model_name="signal_detection")
        self.logger = get_logger(__name__) # Use configured logger

        # Check dependencies
        if not XGB_AVAILABLE:
            self.logger.error("XGBoost library not found. SignalDetectionTrainer cannot function.")
            raise ImportError("XGBoost is required for SignalDetectionTrainer.")
        if not XGBOOST_MODEL_AVAILABLE:
             self.logger.warning("XGBoostModel wrapper not found. Training will use direct xgboost.train.")

        # Load relevant config sections using get_* methods
        self.feature_selection_config = self.config.get_dict("ML_FEATURE_SELECTION", {})
        self.time_series_cv_config = self.config.get_dict("ML_TIME_SERIES_CV", {})
        self.test_size = self.config.get_float("ML_TEST_SIZE", 0.2)
        self.random_state = self.config.get_int("ML_RANDOM_STATE", 42)
        self.models_dir = self.config.get_path("MODEL_DIR", "./models") # Used for optimized params path

        # XGBoost specific settings from central config
        self.use_gpu = self.config.get_bool("XGBOOST_USE_GPU", True) and self.config.get_bool("USE_GPU", True) # Check both flags
        self.use_pytorch = self.config.get_bool("XGBOOST_USE_PYTORCH", True)
        self.tree_method = self.config.get("XGBOOST_TREE_METHOD", "auto")
        self.gpu_id = self.config.get_int("XGBOOST_GPU_ID", 0)

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
        self.logger.info(f"Starting training for {self.model_name} model...")
        run_success = False
        # --- MLflow Integration Start ---
        if not MLFLOW_AVAILABLE:
             self.logger.warning("MLflow not available. Skipping MLflow logging.")

        # Use context manager for the run
        with mlflow.start_run(run_name=f"{self.model_name}_train_{int(time.time())}") if MLFLOW_AVAILABLE else self._dummy_context():
            if MLFLOW_AVAILABLE:
                 mlflow.log_param("model_name", self.model_name)
                 mlflow.log_param("training_timestamp", datetime.now().isoformat())

        try:
            if features is None or target is None or features.empty or target.empty:
                self.logger.error("Input features or target data is missing or empty.")
                raise DataError("Cannot train signal detection model with empty data.")
            if len(features) != len(target):
                 self.logger.error(f"Feature count ({len(features)}) and target count ({len(target)}) mismatch.")
                 raise DataError("Feature and target length mismatch.")

            # Apply feature selection if enabled
            # Log initial feature count
            if MLFLOW_AVAILABLE: mlflow.log_param("initial_feature_count", len(features.columns))

            # Apply feature selection if enabled
            if self.feature_selection_config.get("enabled", False) and data_processor:
                self.logger.info("Applying feature selection...")
                features = data_processor.select_features(features, target, "classification")
                if features.empty:
                     self.logger.error("No features remaining after feature selection.")
                     raise DataError("Feature selection resulted in zero features.")
                self.logger.info(f"Features remaining after selection: {len(features.columns)}")
                if MLFLOW_AVAILABLE:
                     mlflow.log_param("feature_selection_enabled", True)
                     mlflow.log_param("feature_selection_method", self.feature_selection_config.get("method"))
                     mlflow.log_param("final_feature_count", len(features.columns))
                     # Log selected features list? Maybe as artifact
                     mlflow.log_dict({"selected_features": list(features.columns)}, "selected_features.json")
            elif MLFLOW_AVAILABLE:
                 mlflow.log_param("feature_selection_enabled", False)

            # Use time series cross-validation if enabled
            # Use time series cross-validation if enabled
            if self.time_series_cv_config.get("enabled", False) and data_processor:
                self.logger.info("Using Time Series CV split.")
                if MLFLOW_AVAILABLE: mlflow.log_param("cv_type", "time_series")
                splits = data_processor.create_time_series_splits(features, target)
                if not splits:
                     self.logger.error("Failed to create time series splits.")
                     raise DataError("Time series split creation failed.")
                # Use the last split for final evaluation and training for now
                # TODO: Implement proper CV training loop if needed
                train_idx, test_idx = splits[-1]
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                if MLFLOW_AVAILABLE:
                     mlflow.log_param("cv_n_splits", self.time_series_cv_config.get("n_splits"))
                     mlflow.log_param("cv_embargo_size", self.time_series_cv_config.get("embargo_size"))
            else:
                # Use traditional train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    features,
                    target,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Save scaler
            # Save scaler (using BaseTrainer method, assumed to handle path)
            scaler_path = self.save_scaler(scaler)
            if MLFLOW_AVAILABLE and scaler_path:
                 try:
                      # Log scaler as sklearn model or artifact
                      mlflow.sklearn.log_model(scaler, "scaler")
                      self.logger.info("Logged scaler to MLflow.")
                 except Exception as log_err:
                      self.logger.warning(f"Failed to log scaler to MLflow: {log_err}")

            # Get model config
            model_config = self.get_model_config()

            # Check for optimized parameters
            # Load optimized parameters if file exists
            optimized_params = {}
            optimized_params_path = self.models_dir / f"{self.model_name}_optimized_params.json"
            if optimized_params_path.exists():
                try:
                    with open(optimized_params_path, 'r') as f:
                        optimized_params = json.load(f)
                    self.logger.info(f"Loaded optimized parameters from {optimized_params_path}")
                    # Merge optimized params into model config params
                    model_config["params"].update(optimized_params)
                    if MLFLOW_AVAILABLE:
                         mlflow.log_param("optimized_params_loaded", True)
                         mlflow.log_params({f"opt_{k}": v for k,v in optimized_params.items()}) # Log optimized params separately
                except (json.JSONDecodeError, OSError) as e:
                    self.logger.warning(f"Failed to load optimized parameters from {optimized_params_path}: {e}")
                    if MLFLOW_AVAILABLE: mlflow.log_param("optimized_params_loaded", False)
            elif MLFLOW_AVAILABLE:
                 mlflow.log_param("optimized_params_loaded", False)

            # Log final effective parameters to MLflow
            if MLFLOW_AVAILABLE:
                 mlflow.log_params(model_config.get("params", {}))

            # Check if we should use XGBoostModel or direct XGBoost
            # Decide training approach (Wrapper preferred if available)
            # Use loaded config for GPU/PyTorch settings
            if XGBOOST_MODEL_AVAILABLE and XGBoostModel is not None:
                # Train using XGBoostModel
                logger.info("Training signal detection model using XGBoostModel")
                
                # Convert scaled arrays back to DataFrames for XGBoostModel
                X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                
                # Create model with parameters
                model = XGBoostModel(
                    model_type="classifier", # Hardcoded for signal detection
                    use_gpu=self.use_gpu,
                    use_pytorch=self.use_pytorch,
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
                # Save model using BaseTrainer (handles path) and log to MLflow
                # Prefer JSON format for saving via BaseTrainer if possible
                saved_model_path = self.save_model(model, model_format='json') # Request JSON format

                if MLFLOW_AVAILABLE and model.model: # Check if underlying model exists
                     try:
                          # Log the XGBoost model itself to MLflow
                          mlflow.xgboost.log_model(
                              xgb_model=model.model,
                              artifact_path=self.model_name, # Use model name as artifact path
                              input_example=X_train_df.iloc[:5], # Add input example
                              # signature=... # TODO: Define model signature
                          )
                          self.logger.info(f"Logged {self.model_name} model to MLflow.")
                     except Exception as log_err:
                          self.logger.warning(f"Failed to log {self.model_name} model to MLflow: {log_err}")
                
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
                # Save model using BaseTrainer (handles path) and log to MLflow
                saved_model_path = self.save_model(model, model_format='xgb') # Request XGB format

                if MLFLOW_AVAILABLE:
                     try:
                          mlflow.xgboost.log_model(
                              xgb_model=model,
                              artifact_path=self.model_name,
                              input_example=X_train_scaled[:5], # Use numpy array example
                              # signature=... # TODO: Define model signature
                          )
                          self.logger.info(f"Logged {self.model_name} model to MLflow.")
                     except Exception as log_err:
                          self.logger.warning(f"Failed to log {self.model_name} model to MLflow: {log_err}")

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
            # Save metrics using BaseTrainer (handles path) and log to MLflow
            self.save_metrics(metrics)
            if MLFLOW_AVAILABLE:
                 # Log main metrics to MLflow tracking UI
                 mlflow_metrics = {k: v for k, v in metrics.items() if k != "feature_importance"}
                 mlflow.log_metrics(mlflow_metrics)
                 # Log feature importance separately as dict/artifact
                 if "feature_importance" in metrics:
                      mlflow.log_dict(metrics["feature_importance"], "feature_importance.json")
                 self.logger.info("Logged metrics to MLflow.")

            # Send notification
            self.send_notification(
                message="Signal detection model trained successfully",
                level="success",
                details=metrics
            )

            self.logger.info(f"{self.model_name} model trained successfully.")
            run_success = True
            return True

        except (DataError, ConfigurationError) as e: # Catch specific known errors
             self.logger.error(f"Error during training setup for {self.model_name}: {e}", exc_info=True)
             self.send_notification(f"Training failed for {self.model_name}: {e}", level="error", details={"error": str(e)})
             if MLFLOW_AVAILABLE: mlflow.set_tag("status", "failed") # Mark run as failed
             return False
        except ImportError as e: # Catch missing dependencies like XGBoost
             self.logger.error(f"Import error during training for {self.model_name}: {e}", exc_info=True)
             self.send_notification(f"Training failed for {self.model_name} due to missing dependency: {e}", level="error", details={"error": str(e)})
             if MLFLOW_AVAILABLE: mlflow.set_tag("status", "failed")
             return False
        except Exception as e: # Catch unexpected training errors
            self.logger.error(f"Unexpected error training {self.model_name} model: {e}", exc_info=True)
            self.send_notification(f"Unexpected training error for {self.model_name}: {e}", level="error", details={"error": str(e)})
            if MLFLOW_AVAILABLE: mlflow.set_tag("status", "failed")
            # Optionally wrap in ModelTrainingError before returning?
            return False
        finally:
             # Ensure MLflow run ends even if errors occur
             if MLFLOW_AVAILABLE:
                  if run_success:
                       mlflow.set_tag("status", "completed")
                  # mlflow.end_run() # Context manager handles ending the run
                  self.logger.info("MLflow run ended.")