#!/usr/bin/env python3
"""
Data Processing Module

Contains functionality for:
- Data preprocessing and cleaning
- Feature engineering
- Technical indicator calculation
- Data normalization and scaling
- GPU-accelerated processing (where applicable)
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Optional, Union, List # Added List

from utils.logging_config import get_logger
from utils.metrics_registry import DATA_PROCESSING_TIME, DATA_ROWS_PROCESSED # Removed GPU_MEMORY_USAGE for now
# Import GPU utilities
from utils.gpu_utils import is_gpu_available, process_array, cp, CUPY_AVAILABLE

logger = get_logger("data_pipeline.processing")

def clean_market_data(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Clean raw market data

    Args:
        df: Raw market data DataFrame
        strict: If True, removes invalid rows. If False, replaces invalid values.

    Returns:
        Cleaned DataFrame with standardized columns and types
    """
    try:
        if df is None or df.empty:
             logger.warning("clean_market_data received empty or None DataFrame.")
             return pd.DataFrame() # Return empty DataFrame

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            # Try different formats, prioritize unit='ms' if likely unix timestamp
            try:
                 if df['timestamp'].dtype in ['int64', 'float64'] and df['timestamp'].iloc[0] > 1e11: # Heuristic for ms timestamp
                      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                 else:
                      df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            except Exception as ts_err:
                 logger.warning(f"Timestamp conversion failed: {ts_err}. Attempting infer_datetime_format.")
                 df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True, infer_datetime_format=True)

        elif 'time' in df.columns:
            # Similar conversion logic for 'time' column
            try:
                 if df['time'].dtype in ['int64', 'float64'] and df['time'].iloc[0] > 1e11:
                      df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True)
                 else:
                      df['timestamp'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
            except Exception as ts_err:
                 logger.warning(f"Time conversion failed: {ts_err}. Attempting infer_datetime_format.")
                 df['timestamp'] = pd.to_datetime(df['time'], errors='coerce', utc=True, infer_datetime_format=True)
            df = df.drop('time', axis=1, errors='ignore')
        else:
             logger.warning("No 'timestamp' or 'time' column found for cleaning.")
             # Cannot proceed reliably without a timestamp
             return pd.DataFrame()


        # Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Replace infinity with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # Validate values are within reasonable bounds only if data exists
                if not df[col].isna().all():
                    if col in ['open', 'high', 'low', 'close']:
                        upper_limit = min(df[col].quantile(0.999, interpolation='nearest'), 1e7) # Increased limit slightly
                        lower_limit = max(df[col].quantile(0.001, interpolation='nearest'), 0.001) # Added lower limit
                        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
                    elif col == 'volume':
                        upper_limit = min(df[col].quantile(0.999, interpolation='nearest'), 1e12)
                        df[col] = df[col].clip(lower=0, upper=upper_limit)

        # Handle invalid values based on strict mode
        critical_cols = ['timestamp', 'close']
        # Drop rows where critical columns are NaT/NaN
        df = df.dropna(subset=critical_cols)

        if not strict:
             # If not strict, fill remaining NaNs in numeric cols (e.g., open, high, low)
             # Use forward fill first, then backfill
             for col in numeric_cols:
                  if col in df.columns and col not in critical_cols:
                       df[col] = df[col].ffill().bfill()
                       # Fill with 0 if still NaN after ffill/bfill (e.g., all NaNs)
                       df[col] = df[col].fillna(0)

        # Final check for NaNs in numeric columns after potential filling
        for col in numeric_cols:
            if col in df.columns and df[col].isna().any():
                logger.warning(f"Column {col} still contains {df[col].isna().sum()} NaN values after cleaning (strict={strict}). Filling with 0.")
                df[col] = df[col].fillna(0) # Final fallback fill

        # Sort by timestamp if timestamp column exists
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df
    except Exception as e:
        logger.exception(f"Error cleaning market data: {e}") # Use exception for full traceback
        # Return empty dataframe on critical error during cleaning
        return pd.DataFrame()

def calculate_technical_indicators(
    df: pd.DataFrame,
    window_sizes: list[int] = [5, 10, 20, 50, 200]
    # Removed gpu_utils parameter
) -> pd.DataFrame:
    """Calculate technical indicators for market data using Pandas.

    Note: Rolling window calculations are currently CPU-bound using Pandas.
          GPU acceleration for these requires more specialized implementation.

    Args:
        df: Cleaned market data DataFrame (must include 'close', 'high', 'low', 'volume').
        window_sizes: List of window sizes for moving averages.

    Returns:
        DataFrame with added technical indicators.
    """
    if df is None or df.empty or 'close' not in df.columns:
        logger.warning("Cannot calculate indicators: DataFrame is empty or missing 'close' column.")
        return df # Return original or empty df

    required_cols = ['close', 'high', 'low', 'volume']
    if not all(col in df.columns for col in required_cols):
         logger.warning(f"Missing required columns for indicators: {required_cols}. Skipping some indicators.")
         # Allow partial calculation if possible

    try:
        start_time = time.time()
        n_rows = len(df)
        df_out = df.copy() # Work on a copy

        # Calculate simple moving averages (CPU - Pandas)
        for window in window_sizes:
            df_out[f'sma_{window}'] = df_out['close'].rolling(window, min_periods=window).mean()

        # Calculate RSI (14-day) (CPU - Pandas)
        delta = df_out['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, 1e-6) # Replace 0 loss with small number
        df_out['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df_out['rsi'] = df_out['rsi'].fillna(50) # Fill initial NaNs with neutral 50

        # Calculate Bollinger Bands (20-day) (CPU - Pandas)
        rolling_mean_20 = df_out['close'].rolling(20, min_periods=20).mean()
        rolling_std_20 = df_out['close'].rolling(20, min_periods=20).std()
        df_out['bollinger_upper'] = rolling_mean_20 + (rolling_std_20 * 2)
        df_out['bollinger_lower'] = rolling_mean_20 - (rolling_std_20 * 2)

        # Calculate MACD (CPU - Pandas)
        ema12 = df_out['close'].ewm(span=12, adjust=False).mean()
        ema26 = df_out['close'].ewm(span=26, adjust=False).mean()
        df_out['macd'] = ema12 - ema26
        df_out['macd_signal'] = df_out['macd'].ewm(span=9, adjust=False).mean()

        # Calculate daily returns (CPU - Pandas)
        df_out['daily_return'] = df_out['close'].pct_change()

        # Calculate volatility (CPU - Pandas)
        df_out['volatility_30d'] = df_out['daily_return'].rolling(30, min_periods=30).std() * np.sqrt(252) # Assuming daily data for annualization

        # Fill initial NaNs resulting from rolling windows
        df_out = df_out.fillna(method='bfill').fillna(0) # Backfill first, then fill remaining with 0

        # Record metrics
        processing_time = time.time() - start_time
        DATA_PROCESSING_TIME.labels(operation="calculate_technical_indicators").observe(processing_time)
        DATA_ROWS_PROCESSED.labels(operation="calculate_technical_indicators").inc(n_rows)

        # GPU Memory metric removed as it's harder to track accurately here

        return df_out
    except Exception as e:
        logger.exception("Error calculating technical indicators") # Use exception for full traceback
        return df # Return original DataFrame on error

def normalize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str = 'standard'
    # Removed gpu_utils parameter
) -> pd.DataFrame:
    """Normalize feature columns using CPU or GPU (CuPy) if available.

    Args:
        df: DataFrame containing features.
        feature_columns: List of columns to normalize.
        method: Normalization method ('standard', 'minmax', 'robust').

    Returns:
        DataFrame with normalized features.
    """
    if df is None or df.empty:
        logger.warning("normalize_features received empty or None DataFrame.")
        return df

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Columns missing for normalization: {missing_cols}. Skipping them.")
        feature_columns = [col for col in feature_columns if col in df.columns]
        if not feature_columns:
            logger.error("No valid feature columns left for normalization.")
            return df

    try:
        features_np = df[feature_columns].to_numpy(dtype=np.float32) # Work with numpy array

        if method == 'standard':
            mean = process_array(features_np, lambda x: np.mean(x, axis=0))
            std = process_array(features_np, lambda x: np.std(x, axis=0))
            std[std == 0] = 1e-6 # Avoid division by zero
            normalized_features = process_array(features_np, lambda x: (x - mean) / std)
        elif method == 'minmax':
            min_val = process_array(features_np, lambda x: np.min(x, axis=0))
            max_val = process_array(features_np, lambda x: np.max(x, axis=0))
            range_val = max_val - min_val
            range_val[range_val == 0] = 1e-6 # Avoid division by zero
            normalized_features = process_array(features_np, lambda x: (x - min_val) / range_val)
        elif method == 'robust':
            # CuPy doesn't have direct quantile/median like numpy for process_array lambda
            # Fallback to CPU (Pandas) for robust scaling for simplicity
            logger.debug("Robust scaling currently uses CPU (Pandas).")
            features_pd = df[feature_columns].copy()
            median = features_pd.median()
            q1 = features_pd.quantile(0.25)
            q3 = features_pd.quantile(0.75)
            iqr = q3 - q1
            iqr[iqr == 0] = 1e-6 # Avoid division by zero
            normalized_features = ((features_pd - median) / iqr).to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Update DataFrame with normalized features
        df_out = df.copy()
        df_out[feature_columns] = normalized_features
        return df_out
    except Exception as e:
        logger.exception(f"Error normalizing features (method: {method})")
        return df # Return original DataFrame on error

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to DataFrame (CPU-bound).

    Args:
        df: DataFrame with timestamp column.

    Returns:
        DataFrame with added time features.
    """
    if df is None or df.empty: return df
    try:
        if 'timestamp' not in df.columns or pd.api.types.is_datetime64_any_dtype(df['timestamp']) is False:
             logger.warning("DataFrame must have a datetime 'timestamp' column for add_time_features.")
             return df

        df_out = df.copy()
        ts = df_out['timestamp'] # Use .dt accessor

        # Extract time components
        df_out['hour'] = ts.dt.hour
        df_out['day_of_week'] = ts.dt.dayofweek
        df_out['day_of_month'] = ts.dt.day
        # Use dt.isocalendar().week for pandas >= 1.1.0
        if hasattr(ts.dt, 'isocalendar'):
             df_out['week_of_year'] = ts.dt.isocalendar().week.astype(int)
        else: # Fallback for older pandas
             df_out['week_of_year'] = ts.dt.weekofyear
        df_out['month'] = ts.dt.month
        df_out['quarter'] = ts.dt.quarter

        # Market session features (assuming US/Eastern timezone context)
        # For accuracy, timestamp should be localized or assumed to be ET
        # This calculation might be inaccurate if timestamp is UTC without conversion
        # Consider localizing timestamp before this step if needed
        df_out['is_premarket'] = (df_out['hour'] < 9).astype(int) # Simplified check
        df_out['is_market_hours'] = ((df_out['hour'] >= 9) & (df_out['hour'] < 16)).astype(int) # Simplified check
        df_out['is_after_hours'] = (df_out['hour'] >= 16).astype(int) # Simplified check

        # Time since market open (Needs timezone awareness for accuracy)
        # Assuming timestamp is already ET or calculation is relative within the day
        # market_open_time = ts.dt.normalize() + pd.Timedelta(hours=9, minutes=30)
        # df_out['minutes_since_open'] = (ts - market_open_time).dt.total_seconds() / 60
        # Simplified version: minutes past midnight
        df_out['minutes_past_midnight'] = ts.dt.hour * 60 + ts.dt.minute


        return df_out
    except Exception as e:
        logger.exception("Error adding time features")
        return df # Return original DataFrame on error

def detect_anomalies(
    df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 3.0
    # Removed gpu_utils parameter
) -> pd.DataFrame:
    """Detect anomalies using modified Z-score (CPU or GPU via process_array).

    Args:
        df: DataFrame containing features.
        feature_columns: Columns to analyze for anomalies.
        threshold: Z-score threshold for anomaly detection.

    Returns:
        DataFrame with added anomaly flags.
    """
    if df is None or df.empty: return df

    df_out = df.copy()
    try:
        for col in feature_columns:
            if col not in df_out.columns:
                logger.warning(f"Column {col} not found for anomaly detection.")
                continue

            data_col = df_out[col].to_numpy(dtype=np.float32)

            # Calculate median and MAD using process_array for potential GPU use
            median = process_array(data_col, lambda x: np.median(x) if isinstance(x, np.ndarray) else cp.median(x))
            abs_diff_from_median = process_array(data_col, lambda x: np.abs(x - median) if isinstance(x, np.ndarray) else cp.abs(x - median))
            mad = process_array(abs_diff_from_median, lambda x: np.median(x) if isinstance(x, np.ndarray) else cp.median(x))

            # Avoid division by zero in MAD
            if mad == 0:
                logger.warning(f"MAD is zero for column {col}. Skipping anomaly detection.")
                df_out[f'{col}_anomaly'] = 0 # No anomalies if no deviation
                continue

            # Calculate modified Z-score (on CPU or GPU via process_array)
            modified_z = process_array(data_col, lambda x: 0.6745 * (x - median) / mad)

            # Flag anomalies
            anomaly_flags = process_array(modified_z, lambda x: (np.abs(x) > threshold).astype(int))
            df_out[f'{col}_anomaly'] = anomaly_flags

        return df_out
    except Exception as e:
        logger.exception(f"Error detecting anomalies")
        return df # Return original DataFrame on error
