"""
XGBoost Integration Utilities for Prediction.

Provides functions to load and predict with XGBoost models.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Any, Union

from ai_day_trader.utils.logging_config import get_logger # Use new utils path

# Configure logging
logger = get_logger("ai_day_trader.ml.xgboost_integration")

# Import from the relative xgboost module within ai_day_trader.ml
from .xgboost import (
    load_model as load_xgboost_model, # Rename to avoid conflict if needed elsewhere
    XGBOOST_AVAILABLE
)

def predict_with_xgboost(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions with an XGBoost model, returning predictions and probabilities.

    Args:
        model: Trained XGBoost model instance.
        data: Data to make predictions on (DataFrame or Numpy array).
        feature_columns: List of feature column names (if data is DataFrame).

    Returns:
        Tuple of (predictions, probabilities). Probabilities are None for regressors.
    """
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost library not available for prediction.")
        raise ImportError("XGBoost library not found.")
    if model is None:
        logger.error("No model provided for prediction.")
        raise ValueError("Model cannot be None for prediction.")

    try:
        # Select features if feature_columns is provided and data is DataFrame
        if isinstance(data, pd.DataFrame) and feature_columns:
            missing_features = [f for f in feature_columns if f not in data.columns]
            if missing_features:
                # Log warning but proceed, assuming missing features handled upstream or by model
                logger.warning(f"Missing features during prediction: {missing_features}. Using available columns.")
                features = data[[col for col in feature_columns if col in data.columns]]
            else:
                features = data[feature_columns]
        else:
            features = data # Assume data is already prepared (e.g., numpy array)

        if hasattr(features, 'empty') and features.empty:
             logger.warning("Input data for prediction is empty.")
             return np.array([]), np.array([]) if hasattr(model, 'predict_proba') else None


        # Make predictions
        predictions = model.predict(features)

        # Get probabilities for classifiers
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                # Get probabilities for the positive class (assuming binary classification)
                # Adjust if multi-class
                proba_result = model.predict_proba(features)
                if proba_result.ndim == 2 and proba_result.shape[1] >= 2:
                     probabilities = proba_result[:, 1] # Probability of class 1
                elif proba_result.ndim == 1: # Handle cases where predict_proba might return 1D array
                     probabilities = proba_result
                else:
                     logger.warning(f"Unexpected probability shape: {proba_result.shape}")
                     # Fallback: use prediction as probability? Or return None? Returning None for now.
                     probabilities = None

            except Exception as proba_e:
                 logger.warning(f"Could not get prediction probabilities: {proba_e}")


        # logger.debug(f"Made predictions for {len(features)} samples")
        return predictions, probabilities

    except Exception as e:
        logger.error(f"Error making XGBoost predictions: {e}", exc_info=True)
        # Return empty arrays or raise? Raising might halt processes unexpectedly.
        # Returning empty arrays allows caller to handle.
        return np.array([]), None

# load_xgboost_model is imported and renamed from .xgboost
