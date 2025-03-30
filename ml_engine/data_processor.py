#!/usr/bin/env python3
"""
ML Data Processor Module

This module provides the MLDataProcessor class for data processing:
1. Loads and prepares historical data for model training
2. Handles feature engineering and selection
3. Creates time series cross-validation splits
4. Detects data drift for model monitoring
5. Prepares data for different model types

The MLDataProcessor is a critical component that ensures data quality and consistency
across the ML pipeline.
"""

import datetime
import json
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

from ml_engine.utils import select_features, create_time_series_splits
from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.data_processor")

# Import Prometheus client if available
try:
    import prometheus_client as prom
    from utils.metrics_registry import DRIFT_DETECTION
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics will not be exposed.")


class MLDataProcessor:
    """Data processor for ML model training"""

    def __init__(self, data_loader, redis_client=None, config=None) -> None:
        """
        Initialize data processor
        
        Args:
            data_loader: Data loader instance
            redis_client: Redis client for caching and notifications
            config: Configuration dictionary
        """
        self.data_loader = data_loader
        self.redis = redis_client
        self.config = config
        self.reference_data = None

    def load_historical_data(self):
        """
        Load historical data for model training
        
        Returns:
            DataFrame with historical data or None if loading failed
        """
        try:
            # Get lookback days from config
            lookback_days = self.config.get("lookback_days", 30)

            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)

            # Use data loader to get price data
            price_data = self.data_loader.load_price_data(
                tickers=self.data_loader.get_watchlist_tickers(),
                start_date=start_date,
                end_date=end_date,
                timeframe="1m",
            )

            # Get options data if available
            options_data = self.data_loader.load_options_data(
                tickers=self.data_loader.get_watchlist_tickers(),
                start_date=start_date,
                end_date=end_date,
            )

            # Get market data
            market_data = self.data_loader.load_market_data(
                start_date=start_date, end_date=end_date,
            )

            # Prepare combined dataset
            combined_data = self.data_loader.prepare_training_data(
                price_data=price_data,
                options_data=options_data,
                market_data=market_data,
            )

            logger.info(
                f"Loaded historical data: {len(combined_data)} samples")
            return combined_data

        except Exception as e:
            logger.error(
                f"Error loading historical data: {e!s}", exc_info=True)
            return None

    def store_reference_data(self, data) -> bool | None:
        """
        Store reference data for drift detection
        
        Args:
            data: DataFrame with reference data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a copy to avoid modifying the original
            self.reference_data = data.copy()

            # Save to disk for persistence
            if self.config and "monitoring_dir" in self.config:
                # Ensure directory exists
                os.makedirs(self.config["monitoring_dir"], exist_ok=True)
                
                ref_path = os.path.join(
                    self.config["monitoring_dir"], "reference_data.pkl",
                )
                with open(ref_path, "wb") as f:
                    pickle.dump(self.reference_data, f)

            logger.info(
                f"Stored reference data: {len(self.reference_data)} samples")
            return True
        except Exception as e:
            logger.error(f"Error storing reference data: {e!s}", exc_info=True)
            return False

    def prepare_signal_detection_data(self, data):
        """
        Prepare data for signal detection model
        
        Args:
            data: DataFrame with raw data
            
        Returns:
            Tuple of (features, target) DataFrames
        """
        try:
            # Select features
            feature_columns = [
                # Price-based features
                "close",
                "open",
                "high",
                "low",
                "volume",
                # Technical indicators
                "sma5",
                "sma10",
                "sma20",
                "ema5",
                "ema10",
                "ema20",
                "macd",
                "macd_signal",
                "macd_hist",
                "price_rel_sma5",
                "price_rel_sma10",
                "price_rel_sma20",
                "mom1",
                "mom5",
                "mom10",
                "volatility",
                "volume_ratio",
                "rsi",
                "bb_width",
                # Market features (if available)
                "spy_close",
                "vix_close",
                "spy_change",
                "vix_change",
                # Options features (if available)
                "put_call_ratio",
                "implied_volatility",
                "option_volume",
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 5:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return pd.DataFrame(), pd.Series()

            # Select data
            X = data[available_columns].copy()
            y = data["signal_target"].copy() if "signal_target" in data.columns else pd.Series()
            
            if y.empty:
                logger.warning("No target variable 'signal_target' found in data")
                return X, y

            # Drop rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            # Handle any remaining infinity values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())

            logger.info(
                f"Prepared signal detection data with {len(X)} samples and {len(available_columns)} features",
            )

            return X, y

        except Exception as e:
            logger.error(
                f"Error preparing signal detection data: {e!s}", exc_info=True,
            )
            return pd.DataFrame(), pd.Series()

    def prepare_price_prediction_data(self, data):
        """
        Prepare data for price prediction model (LSTM)
        
        Args:
            data: DataFrame with raw data
            
        Returns:
            Tuple of (sequences, targets) numpy arrays
        """
        try:
            # Select features
            feature_columns = [
                # Price-based features
                "close",
                "high",
                "low",
                "volume",
                # Technical indicators
                "price_rel_sma5",
                "price_rel_sma10",
                "price_rel_sma20",
                "macd",
                "rsi",
                "volatility",
                # Market features (if available)
                "spy_close",
                "vix_close",
            ]

            # Keep only available columns
            available_columns = [
                col for col in feature_columns if col in data.columns]

            if len(available_columns) < 4:
                logger.warning(
                    f"Too few features available: {len(available_columns)}")
                return np.array([]), np.array([])

            # Target columns
            target_columns = [
                "future_return_5min",
                "future_return_10min",
                "future_return_30min",
            ]
            available_targets = [
                col for col in target_columns if col in data.columns]

            if len(available_targets) == 0:
                logger.warning("No target variables available")
                return np.array([]), np.array([])

            # Group by ticker to create sequences
            sequences = []
            targets = []

            for ticker, group in data.groupby("ticker"):
                # Sort by timestamp
                group = group.sort_index()

                # Select features and targets
                X = group[available_columns].values
                y = group[available_targets].values

                # Create sequences (lookback of 20 intervals)
                for i in range(20, len(X)):
                    sequences.append(X[i - 20: i])
                    targets.append(y[i])

            # Convert to numpy arrays
            X_array = np.array(sequences)
            y_array = np.array(targets)

            # Handle NaN or infinite values
            if (
                np.isnan(X_array).any()
                or np.isinf(X_array).any()
                or np.isnan(y_array).any()
                or np.isinf(y_array).any()
            ):
                logger.warning(
                    "NaN or infinite values detected. Performing robust cleaning...",
                )

                # Identify rows with NaN or inf in either X or y
                X_has_invalid = np.any(
                    np.isnan(X_array) | np.isinf(X_array), axis=(1, 2),
                )
                y_has_invalid = np.any(
                    np.isnan(y_array) | np.isinf(y_array), axis=1)
                valid_indices = ~(X_has_invalid | y_has_invalid)

                # Filter out invalid rows if enough valid data
                if np.sum(valid_indices) > 100:
                    X_array = X_array[valid_indices]
                    y_array = y_array[valid_indices]
                else:
                    # Replace NaN and inf with zeros/means if not enough valid data
                    X_array = np.nan_to_num(
                        X_array, nan=0.0, posinf=0.0, neginf=0.0)
                    y_array = np.nan_to_num(
                        y_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            scaler = MinMaxScaler()
            n_samples, n_timesteps, n_features = X_array.shape
            X_reshaped = X_array.reshape(n_samples * n_timesteps, n_features)
            X_scaled = scaler.fit_transform(X_reshaped)
            X_array = X_scaled.reshape(n_samples, n_timesteps, n_features)

            # Store scaler in memory
            self.price_prediction_scaler = scaler

            logger.info(
                f"Prepared price prediction data with {len(sequences)} sequences",
            )

            return X_array, y_array

        except Exception as e:
            logger.error(
                f"Error preparing price prediction data: {e!s}", exc_info=True,
            )
            return np.array([]), np.array([])

    def create_time_series_splits(self, X, y):
        """
        Create time series cross-validation splits
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        return create_time_series_splits(
            X,
            y,
            self.config["time_series_cv"]["n_splits"],
            self.config["time_series_cv"]["embargo_size"],
        )

    def select_features(self, X, y, problem_type="classification"):
        """
        Select important features based on feature selection method
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: Type of problem ('classification' or 'regression')
            
        Returns:
            DataFrame with selected features
        """
        return select_features(
            X,
            y,
            problem_type,
            self.config["feature_selection"]["method"],
            self.config["feature_selection"]["threshold"],
            self.config["feature_selection"]["n_features"],
        )

    def detect_drift(self, current_data):
        """
        Detect drift in feature distributions
        
        Args:
            current_data: DataFrame with current data
            
        Returns:
            Tuple of (drift_detected, drift_features)
        """
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return False, {}

        # Record drift detection attempt in Prometheus if available
        if PROMETHEUS_AVAILABLE:
            try:
                # Start timing the drift detection
                start_time = time.time()

                # Perform drift detection
                drift_detected, drift_features = detect_feature_drift(
                    current_data,
                    self.reference_data,
                    self.config["monitoring"]["drift_threshold"],
                )

                # Record the result in Prometheus
                DRIFT_DETECTION.labels(
                    model_name="data_features",
                    result="detected" if drift_detected else "not_detected",
                ).inc()

                # Record detection time
                detection_time = time.time() - start_time
                logger.info(
                    f"Drift detection completed in {detection_time:.4f} seconds",
                )

                # Send notification to frontend if drift detected
                if drift_detected and hasattr(self, 'redis') and self.redis:
                    try:
                        # Create notification for frontend
                        notification = {
                            "type": "drift_detection",
                            "message": f"Data drift detected in {len(drift_features)} features",
                            "level": "warning",
                            "timestamp": time.time(),
                            "details": {
                                "drift_features": drift_features,
                                "detection_time": detection_time,
                                "threshold": self.config["monitoring"]["drift_threshold"],
                                "total_features": len(current_data.columns)
                            }
                        }

                        # Push to notifications list
                        self.redis.lpush("frontend:notifications",
                                         json.dumps(notification))
                        self.redis.ltrim("frontend:notifications", 0, 99)

                        # Also store in drift_detection category
                        self.redis.lpush(
                            "frontend:drift_detection", json.dumps(notification))
                        self.redis.ltrim("frontend:drift_detection", 0, 49)

                        logger.warning(
                            f"Drift detection notification sent to frontend: {len(drift_features)} features affected")
                    except Exception as e:
                        logger.error(
                            f"Error sending drift detection notification: {e}")

                return drift_detected, drift_features
            except Exception as e:
                logger.exception(
                    f"Error in drift detection with Prometheus: {e}")
                # Record error in Prometheus
                DRIFT_DETECTION.labels(
                    model_name="data_features", result="error").inc()

                # Fall back to regular detection
                return detect_feature_drift(
                    current_data,
                    self.reference_data,
                    self.config["monitoring"]["drift_threshold"],
                )
        else:
            # Regular drift detection without Prometheus
            return detect_feature_drift(
                current_data,
                self.reference_data,
                self.config["monitoring"]["drift_threshold"],
            )


def detect_feature_drift(current_data, reference_data, threshold=0.05):
    """
    Detect drift in feature distributions using KS test
    
    Args:
        current_data: DataFrame with current data
        reference_data: DataFrame with reference data
        threshold: p-value threshold for drift detection
        
    Returns:
        Tuple of (drift_detected, drift_features)
    """
    try:
        # Select numeric features only
        numeric_features = reference_data.select_dtypes(
            include=[np.number]).columns

        drift_detected = False
        drift_features = {}

        for feature in numeric_features:
            if feature in current_data.columns:
                # Get clean samples from both datasets
                ref_values = reference_data[feature].dropna().values
                cur_values = current_data[feature].dropna().values

                if len(ref_values) > 10 and len(cur_values) > 10:
                    # Perform KS test
                    ks_statistic, p_value = ks_2samp(ref_values, cur_values)

                    if p_value < threshold:
                        drift_detected = True
                        drift_features[feature] = {
                            "ks_statistic": float(ks_statistic),
                            "p_value": float(p_value),
                        }

        return drift_detected, drift_features

    except Exception as e:
        logger.exception(f"Error detecting feature drift: {e!s}")
        return False, {}