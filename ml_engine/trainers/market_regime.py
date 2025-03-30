#!/usr/bin/env python3
"""
Market Regime Trainer Module

This module provides the MarketRegimeTrainer class for training market regime models:
1. Uses KMeans clustering to identify market regimes
2. Analyzes market data to detect different states
3. Handles feature scaling and preprocessing
4. Evaluates model performance with clustering metrics
5. Saves model, scaler, and metrics

The market regime model identifies different market states (e.g., trending, mean-reverting,
high-volatility) to adapt trading strategies to current conditions.
"""

import pickle
import logging
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ml_engine.trainers.base import BaseTrainer
from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.trainers.market_regime")


class MarketRegimeTrainer(BaseTrainer):
    """Trainer for market regime model using KMeans clustering"""

    def __init__(self, config, redis_client=None) -> None:
        """
        Initialize market regime trainer
        
        Args:
            config: Configuration dictionary
            redis_client: Redis client for caching and notifications
        """
        super().__init__(config, redis_client, "market_regime")

    def train(self, features, data_processor=None) -> bool | None:
        """
        Train market regime classifier model
        
        Args:
            features: Feature DataFrame
            data_processor: Data processor instance (optional)
            
        Returns:
            True if training successful, False otherwise
        """
        logger.info("Training market regime model")

        try:
            if len(features) == 0:
                logger.error("No valid data for market regime model")
                return False

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Save scaler
            self.save_scaler(scaler)

            # Get model config
            model_config = self.get_model_config()

            # Create model
            model = KMeans(
                n_clusters=model_config["params"]["n_clusters"],
                random_state=model_config["params"]["random_state"],
                n_init=10,  # Run clustering multiple times for better results
                max_iter=300,  # Increase max iterations for better convergence
            )

            # Train model
            model.fit(features_scaled)

            # Calculate metrics
            inertia = model.inertia_
            cluster_counts = np.bincount(model.labels_)
            
            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(features_scaled, model.labels_)
                logger.info(f"Silhouette score: {silhouette:.4f}")
            except:
                silhouette = None
                logger.warning("Could not calculate silhouette score")

            logger.info(
                f"Market regime model metrics - Inertia: {inertia:.2f}, Cluster counts: {cluster_counts}",
            )

            # Save model
            self.save_model(model, "pkl")

            # Calculate cluster centers in original feature space
            cluster_centers_scaled = model.cluster_centers_
            cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
            
            # Create cluster profiles
            cluster_profiles = {}
            for i in range(len(cluster_centers_original)):
                # Create profile with top features
                profile = {}
                for j, feature in enumerate(features.columns):
                    profile[feature] = float(cluster_centers_original[i, j])
                
                # Sort by absolute deviation from mean
                feature_means = np.mean(cluster_centers_original, axis=0)
                deviations = np.abs(cluster_centers_original[i] - feature_means)
                top_indices = np.argsort(deviations)[-5:]  # Top 5 distinctive features
                
                distinctive_features = {}
                for idx in top_indices:
                    feature = features.columns[idx]
                    value = float(cluster_centers_original[i, idx])
                    mean_value = float(feature_means[idx])
                    distinctive_features[feature] = {
                        "value": value,
                        "mean": mean_value,
                        "deviation": float(deviations[idx]),
                        "deviation_pct": float((value - mean_value) / (mean_value + 1e-8) * 100)
                    }
                
                cluster_profiles[f"cluster_{i}"] = {
                    "size": int(cluster_counts[i]),
                    "size_pct": float(cluster_counts[i] / sum(cluster_counts) * 100),
                    "distinctive_features": distinctive_features
                }

            # Save metrics
            metrics = {
                "inertia": float(inertia),
                "cluster_counts": [int(count) for count in cluster_counts],
                "silhouette": float(silhouette) if silhouette is not None else None,
                "cluster_centers": [
                    [float(value) for value in center]
                    for center in model.cluster_centers_
                ],
                "cluster_profiles": cluster_profiles
            }

            self.save_metrics(metrics)

            # Send notification
            self.send_notification(
                message="Market regime model trained successfully",
                level="success",
                details={
                    "n_clusters": model_config["params"]["n_clusters"],
                    "cluster_counts": [int(count) for count in cluster_counts],
                    "silhouette": float(silhouette) if silhouette is not None else None
                }
            )

            logger.info("Market regime model trained successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error training market regime model: {e!s}", exc_info=True)

            # If training failed, create a simple fallback model
            try:
                logger.info(
                    "Creating fallback market regime model with default parameters",
                )

                # Create a simple KMeans model with default parameters
                model = KMeans(n_clusters=4, random_state=42)

                # Fit on a small dummy dataset to initialize the model
                dummy_data = np.random.rand(10, 5)  # 10 samples, 5 features
                model.fit(dummy_data)

                # Save the model
                self.save_model(model, "pkl")
                
                logger.info(
                    "Created and saved fallback market regime model"
                )

                # Update metrics with minimal model info
                metrics = {
                    "fallback": True,
                    "error": str(e),
                    "cluster_counts": [2, 3, 3, 2],  # Dummy counts
                }
                
                self.save_metrics(metrics)

                # Send notification
                self.send_notification(
                    message="Created fallback market regime model due to training error",
                    level="warning",
                    details={"error": str(e)}
                )
                
                return False
            except Exception as fallback_error:
                logger.exception(
                    f"Error creating fallback market regime model: {fallback_error!s}",
                )
                
                # Send notification
                self.send_notification(
                    message=f"Error training market regime model: {str(e)}",
                    level="error",
                    details={"error": str(e), "fallback_error": str(fallback_error)}
                )
                
                return False