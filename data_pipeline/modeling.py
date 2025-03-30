#!/usr/bin/env python3
"""
Data Modeling Module

Contains functionality for:
- Feature selection
- Model training and evaluation
- Hyperparameter tuning
- Time series cross-validation
- Model interpretation
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Union

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from utils.logging_config import get_logger
from utils.metrics_registry import MODEL_OPTIMIZATION_COUNT

logger = get_logger("data_pipeline.modeling")

def select_features_rfe(
    features: pd.DataFrame,
    target: pd.Series,
    estimator: Any,
    n_features: int,
    step: int = 1
) -> pd.DataFrame:
    """Select features using Recursive Feature Elimination
    
    Args:
        features: DataFrame of features
        target: Series of target values
        estimator: Model to use for feature ranking
        n_features: Number of features to select
        step: Number of features to remove at each iteration
        
    Returns:
        DataFrame with selected features
    """
    try:
        selector = RFE(estimator, n_features_to_select=n_features, step=step)
        selector.fit(features, target)
        
        selected_features = features.columns[selector.support_].tolist()
        logger.info(
            f"Selected {len(selected_features)} features using RFE: {', '.join(selected_features[:5])}..."
        )
        return features[selected_features]
    except Exception as e:
        logger.error(f"Error in RFE feature selection: {e!s}", exc_info=True)
        return features

def select_features_model_based(
    features: pd.DataFrame,
    target: pd.Series,
    estimator: Any,
    threshold: Optional[str] = None
) -> pd.DataFrame:
    """Select features based on model importance
    
    Args:
        features: DataFrame of features
        target: Series of target values
        estimator: Model to use for feature importance
        threshold: Threshold for feature selection
        
    Returns:
        DataFrame with selected features
    """
    try:
        selector = SelectFromModel(estimator, threshold=threshold)
        selector.fit(features, target)
        
        selected_features = features.columns[selector.get_support()].tolist()
        logger.info(
            f"Selected {len(selected_features)} features using model-based selection: {', '.join(selected_features[:5])}..."
        )
        return features[selected_features]
    except Exception as e:
        logger.error(f"Error in model-based feature selection: {e!s}", exc_info=True)
        return features

def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "classification",
    model_class: str = "random_forest",
    params: Optional[dict] = None
) -> Any:
    """Train a machine learning model
    
    Args:
        features: DataFrame of features
        target: Series of target values
        model_type: Type of model ('classification' or 'regression')
        model_class: Class of model ('random_forest', 'xgboost')
        params: Model hyperparameters
        
    Returns:
        Trained model
    """
    try:
        if params is None:
            params = {}
            
        if model_class == "random_forest":
            if model_type == "classification":
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)
        elif model_class == "xgboost":
            if model_type == "classification":
                model = XGBClassifier(**params)
            else:
                model = XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
            
        model.fit(features, target)
        # Increment the optimization count with required labels
        MODEL_OPTIMIZATION_COUNT.labels(
            model_type=model_type,
            framework=model_class
        ).inc()
        return model
    except Exception as e:
        logger.error(f"Error training model: {e!s}", exc_info=True)
        raise

def evaluate_model(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "classification"
) -> dict:
    """Evaluate a trained model
    
    Args:
        model: Trained model
        features: DataFrame of features
        target: Series of target values
        model_type: Type of model ('classification' or 'regression')
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        preds = model.predict(features)
        
        if model_type == "classification":
            return {
                "accuracy": accuracy_score(target, preds),
                "precision": precision_score(target, preds, average='weighted'),
                "recall": recall_score(target, preds, average='weighted'),
                "f1": f1_score(target, preds, average='weighted')
            }
        else:
            return {
                "mse": mean_squared_error(target, preds),
                "mae": mean_absolute_error(target, preds),
                "r2": r2_score(target, preds)
            }
    except Exception as e:
        logger.error(f"Error evaluating model: {e!s}", exc_info=True)
        raise

def create_time_series_splits(
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    embargo_size: int = 0
) -> list:
    """Create time series cross-validation splits
    
    Args:
        features: DataFrame of features
        target: Series of target values
        n_splits: Number of splits
        embargo_size: Size of embargo period between train and test
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    try:
        # Total data size
        n_samples = len(features)

        # Calculate split sizes
        test_size = int(n_samples / (n_splits + 1))

        splits = []
        for i in range(n_splits):
            # Calculate indices
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            # Apply embargo - gap between train and test
            if embargo_size > 0:
                train_end = max(0, test_start - embargo_size)
            else:
                train_end = test_start

            # Create index arrays
            train_idx = list(range(train_end))
            test_idx = list(range(test_start, min(test_end, n_samples)))

            splits.append((train_idx, test_idx))

        return splits
    except Exception as e:
        logger.error(f"Error creating time series splits: {e!s}", exc_info=True)
        # Return a simple 80/20 split as fallback
        n_samples = len(features)
        split_idx = int(n_samples * 0.8)
        return [(list(range(split_idx)), list(range(split_idx, n_samples)))]
