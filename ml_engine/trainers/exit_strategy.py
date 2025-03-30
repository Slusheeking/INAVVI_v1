#!/usr/bin/env python3
"""
Exit Strategy Trainer Module

This module provides the ExitStrategyTrainer class for training exit strategy models:
1. Uses XGBoost for regression
2. Predicts optimal exit points for trades
3. Handles feature selection and scaling
4. Evaluates model performance with regression metrics
5. Saves model, scaler, and metrics

The exit strategy model predicts optimal exit points to maximize profits or minimize losses.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_engine.trainers.base import BaseTrainer
from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.trainers.exit_strategy")

# Import XGBoost with error handling
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. Exit strategy trainer will not work.")

# Import XGBoostModel with error handling
try:
    from ml_engine.xgboost_model import XGBoostModel
    XGBOOST_MODEL_AVAILABLE = True
except ImportError:
    XGBOOST_MODEL_AVAILABLE = False
    logger.warning("XGBoostModel not available. Using direct XGBoost integration.")


class ExitStrategyTrainer(BaseTrainer):
    """Trainer for exit strategy model using XGBoost"""

    def __init__(self, config, redis_client=None) -> None:
        """
        Initialize exit strategy trainer
        
        Args:
            config: Configuration dictionary
            redis_client: Redis client for caching and notifications
        """
        super().__init__(config, redis_client, "exit_strategy")
        
        # Check if XGBoost is available
        if not XGB_AVAILABLE:
            logger.error(
                "XGBoost is not available. Cannot train exit strategy model.")
            msg = "XGBoost is required for exit strategy model"
            raise ImportError(msg)

    def train(self, features, targets, data_processor=None) -> bool | None:
        """
        Train exit strategy model
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            data_processor: Data processor instance
            
        Returns:
            True if training successful, False otherwise
        """
        logger.info("Training exit strategy model")

        try:
            if len(features) == 0 or len(targets) == 0:
                logger.error("No valid data for exit strategy model")
                return False

            # Apply feature selection if enabled
            if self.config["feature_selection"]["enabled"] and data_processor:
                features = data_processor.select_features(
                    features, targets, "regression",
                )

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                targets,
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

            # Check if we should use XGBoostModel or direct XGBoost
            if XGBOOST_MODEL_AVAILABLE:
                # Train using XGBoostModel
                logger.info("Training exit strategy model using XGBoostModel")
                
                # Convert scaled arrays back to DataFrames for XGBoostModel
                X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                
                # Create model with parameters
                model = XGBoostModel(
                    model_type="regressor",
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
                model_path = os.path.join(self.config["models_dir"], "exit_strategy_model.json")
                model.save_model(model_path, format='json')
                
                # Also save in XGB format for backward compatibility
                xgb_model_path = os.path.join(self.config["models_dir"], "exit_strategy_model.xgb")
                if hasattr(model.model, 'save_model'):
                    model.model.save_model(xgb_model_path)
                    
                # Get feature importance
                if hasattr(model.model, 'get_score'):
                    feature_importance_dict = model.model.get_score(importance_type="gain")
                else:
                    # Fallback if get_score is not available
                    feature_importance_dict = {}
                    if hasattr(model.model, 'feature_importances_'):
                        for i, importance in enumerate(model.model.feature_importances_):
                            feature_importance_dict[f'f{i}'] = importance
                
            else:
                # Train using direct XGBoost
                logger.info("Training XGBoost exit strategy model directly")
                dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
                dtest = xgb.DMatrix(X_test_scaled, label=y_test)

                eval_list = [(dtrain, "train"), (dtest, "test")]

                # Create a copy of params without n_estimators to avoid warning
                xgb_params = {
                    k: v for k, v in model_config["params"].items() if k != "n_estimators"
                }

                model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=model_config["params"].get(
                        "n_estimators", 200),
                    evals=eval_list,
                    early_stopping_rounds=20,
                    verbose_eval=False,
                )
                
                # Make predictions
                y_pred = model.predict(dtest)
                
                # Save model
                self.save_model(model, "xgb")
                
                # Get feature importance
                feature_importance_dict = model.get_score(importance_type="gain")

            # Calculate metrics
            mse = np.mean((y_pred - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_pred - y_test))
            mean_actual = np.mean(y_test)
            mean_pred = np.mean(y_pred)

            logger.info(
                f"Exit strategy model metrics - RMSE: {rmse:.6f}, MAE: {mae:.6f}, Mean Target: {mean_actual:.6f}, Mean Prediction: {mean_pred:.6f}",
            )

            # Extract feature importance
            feature_importance = {}
            for k, v in feature_importance_dict.items():
                # Map feature index to feature name if possible
                try:
                    feature_idx = int(k.replace('f', ''))
                    if feature_idx < len(features.columns):
                        feature_name = features.columns[feature_idx]
                        feature_importance[feature_name] = float(v)
                    else:
                        feature_importance[k] = float(v)
                except:
                    feature_importance[k] = float(v)
            
            # Sort feature importance
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda item: item[1], 
                reverse=True
            )[:20])  # Keep top 20 features

            # Save metrics
            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mean_actual": float(mean_actual),
                "mean_pred": float(mean_pred),
                "feature_importance": sorted_importance,
            }

            self.save_metrics(metrics)

            # Send notification
            self.send_notification(
                message="Exit strategy model trained successfully",
                level="success",
                details={
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "top_features": list(sorted_importance.keys())[:5]
                }
            )

            logger.info("Exit strategy model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training exit strategy model: {e!s}", exc_info=True)
            
            # Send notification
            self.send_notification(
                message=f"Error training exit strategy model: {str(e)}",
                level="error",
                details={"error": str(e)}
            )
            
            return False