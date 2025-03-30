#!/usr/bin/env python3
"""
Risk Assessment Trainer Module

This module provides the RiskAssessmentTrainer class for training risk assessment models:
1. Uses Random Forest for regression
2. Predicts volatility and risk metrics
3. Handles feature selection and scaling
4. Evaluates model performance with regression metrics
5. Saves model, scaler, and metrics

The risk assessment model predicts volatility and risk metrics for trading positions.
"""

import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_engine.trainers.base import BaseTrainer
from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.trainers.risk_assessment")


class RiskAssessmentTrainer(BaseTrainer):
    """Trainer for risk assessment model using Random Forest"""

    def __init__(self, config, redis_client=None) -> None:
        """
        Initialize risk assessment trainer
        
        Args:
            config: Configuration dictionary
            redis_client: Redis client for caching and notifications
        """
        super().__init__(config, redis_client, "risk_assessment")

    def train(self, features, targets, data_processor=None) -> bool | None:
        """
        Train risk assessment model
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            data_processor: Data processor instance
            
        Returns:
            True if training successful, False otherwise
        """
        logger.info("Training risk assessment model")

        try:
            if len(features) == 0 or len(targets) == 0:
                logger.error("No valid data for risk assessment model")
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

            # Create model
            model = RandomForestRegressor(
                n_estimators=model_config["params"]["n_estimators"],
                max_depth=min(model_config["params"]["max_depth"], 4),
                min_samples_leaf=max(
                    model_config["params"].get("min_samples_leaf", 50), 50),
                max_features="sqrt",
                random_state=self.config["random_state"],
                n_jobs=-1,  # Use all available cores
            )

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            mse = np.mean((y_pred - y_test) ** 2)
            rmse = np.sqrt(mse)
            r2 = model.score(X_test_scaled, y_test)
            mae = np.mean(np.abs(y_pred - y_test))

            logger.info(
                f"Risk assessment model metrics - MSE: {mse:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}, MAE: {mae:.6f}"
            )

            # Save model
            self.save_model(model, "pkl")

            # Extract feature importance
            feature_importance = {}
            for i, feature in enumerate(features.columns):
                feature_importance[feature] = float(model.feature_importances_[i])
            
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
                "r2": float(r2),
                "mae": float(mae),
                "feature_importance": sorted_importance,
            }

            self.save_metrics(metrics)

            # Send notification
            self.send_notification(
                message="Risk assessment model trained successfully",
                level="success",
                details={
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "top_features": list(sorted_importance.keys())[:5]
                }
            )

            logger.info("Risk assessment model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training risk assessment model: {e!s}", exc_info=True,
            )
            
            # Send notification
            self.send_notification(
                message=f"Error training risk assessment model: {str(e)}",
                level="error",
                details={"error": str(e)}
            )
            
            return False