#!/usr/bin/env python3
"""
ML Engine Utilities Module

This module provides utility functions for the ML engine:
1. Hyperparameter optimization
2. Feature selection
3. Model explainability
4. Drift detection
5. Confidence calculation
6. Batch processing optimization

These utilities are used by the various ML model trainers.
"""

import logging
import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.utils")

# Try to import Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
    logger.info("Optuna is available for hyperparameter optimization")
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Hyperparameter optimization will be disabled.")

# Try to import SHAP for model explainability
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP is available for model explainability")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Model explainability will be limited.")

# Try to import alibi-detect for drift detection
try:
    import alibi_detect
    from alibi_detect.cd import KSDrift
    ALIBI_AVAILABLE = True
    logger.info("Alibi-detect is available for drift detection")
except ImportError:
    ALIBI_AVAILABLE = False
    logger.warning("Alibi-detect not available. Drift detection will be limited.")

def create_time_series_splits(data: pd.DataFrame, n_splits: int = 5, 
                             test_size: float = 0.2, gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits
    
    Args:
        data: DataFrame with time series data
        n_splits: Number of splits
        test_size: Size of test set as a fraction of the data
        gap: Number of samples to exclude between train and test sets
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    try:
        # Ensure n_splits is an integer
        if not isinstance(n_splits, int):
            logger.warning(f"n_splits is not an integer: {type(n_splits)}. Converting to int.")
            n_splits = 5  # Default to 5 splits if conversion fails
            
        from sklearn.model_selection import TimeSeriesSplit
        
        # Create TimeSeriesSplit object
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=int(len(data) * test_size))
        
        # Generate splits
        splits = []
        for train_idx, test_idx in tscv.split(data):
            splits.append((train_idx, test_idx))
        
        logger.info(f"Created {n_splits} time series splits with test_size={test_size} and gap={gap}")
        
        return splits
        
    except Exception as e:
        logger.error(f"Error creating time series splits: {e}")
        
        # Fallback to manual splitting
        logger.info("Falling back to manual time series splitting")
        
        # Ensure n_splits is an integer
        if not isinstance(n_splits, int):
            n_splits = 5  # Default to 5 splits
        
        # Ensure test_size is reasonable
        if test_size > 0.5:
            test_size = 0.2  # Default to 20% if test_size is too large
            
        splits = []
        total_samples = len(data)
        test_samples = int(total_samples * test_size)
        
        # Ensure we have at least one split
        if test_samples >= total_samples:
            test_samples = int(total_samples * 0.2)  # Default to 20%
            logger.warning(f"Test size too large, using {test_samples} samples instead")
        
        # Create at least one split
        if len(splits) == 0:
            train_size = int(total_samples * 0.8)
            train_indices = np.arange(0, train_size)
            test_indices = np.arange(train_size, total_samples)
            splits.append((train_indices, test_indices))
            logger.info(f"Created fallback split with {len(train_indices)} train samples and {len(test_indices)} test samples")
        
        for i in range(n_splits):
            # Calculate split points
            test_end = total_samples - i * test_samples
            test_start = test_end - test_samples
            train_end = test_start - gap
            
            # Ensure valid indices
            if train_end <= 0:
                break
            
            # Create indices
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits

def select_features(features: pd.DataFrame, target: pd.Series, 
                   problem_type: str = "classification",
                   method: str = "importance", 
                   threshold: float = 0.01,
                   max_features: int = 50) -> pd.DataFrame:
    """
    Select features based on importance or correlation
    
    Args:
        features: Feature DataFrame
        target: Target Series
        problem_type: Type of problem ('classification' or 'regression')
        method: Feature selection method ('importance', 'correlation', 'mutual_info')
        threshold: Importance threshold for feature selection
        max_features: Maximum number of features to select
        
    Returns:
        DataFrame with selected features
    """
    logger.info(f"Performing feature selection using {method} method")
    
    try:
        if method == "importance":
            # Use a simple model to get feature importance
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Determine if classification or regression
            if problem_type == "classification" or len(np.unique(target)) < 10 or target.dtype == bool:
                # Classification
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            else:
                # Regression
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                
            # Fit model
            model.fit(features, target)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create DataFrame with feature names and importance
            importance_df = pd.DataFrame({
                'feature': features.columns,
                'importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Select features above threshold
            selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
            
            # Limit to max_features
            selected_features = selected_features[:max_features]
            
            logger.info(f"Selected {len(selected_features)} features using importance threshold")
            
            return features[selected_features]
            
        elif method == "correlation":
            # Calculate correlation with target
            correlation = features.apply(lambda x: x.corr(target) if x.dtype.kind in 'bifc' else 0)
            
            # Get absolute correlation
            abs_correlation = correlation.abs()
            
            # Sort by correlation
            sorted_correlation = abs_correlation.sort_values(ascending=False)
            
            # Select features above threshold
            selected_features = sorted_correlation[sorted_correlation > threshold].index.tolist()
            
            # Limit to max_features
            selected_features = selected_features[:max_features]
            
            logger.info(f"Selected {len(selected_features)} features using correlation threshold")
            
            return features[selected_features]
            
        elif method == "mutual_info":
            # Use mutual information for feature selection
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # Determine if classification or regression
            if problem_type == "classification" or len(np.unique(target)) < 10 or target.dtype == bool:
                # Classification
                mi = mutual_info_classif(features, target, random_state=42)
            else:
                # Regression
                mi = mutual_info_regression(features, target, random_state=42)
                
            # Create DataFrame with feature names and mutual information
            mi_df = pd.DataFrame({
                'feature': features.columns,
                'mutual_info': mi
            })
            
            # Sort by mutual information
            mi_df = mi_df.sort_values('mutual_info', ascending=False)
            
            # Select features above threshold
            selected_features = mi_df[mi_df['mutual_info'] > threshold]['feature'].tolist()
            
            # Limit to max_features
            selected_features = selected_features[:max_features]
            
            logger.info(f"Selected {len(selected_features)} features using mutual information threshold")
            
            return features[selected_features]
            
        else:
            logger.warning(f"Unknown feature selection method: {method}. Using all features.")
            return features
            
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        return features


def detect_drift(reference_data: np.ndarray, current_data: np.ndarray, 
                feature_names: List[str] = None, 
                p_val_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect drift between reference and current data
    
    Args:
        reference_data: Reference data (training data)
        current_data: Current data to check for drift
        feature_names: List of feature names
        p_val_threshold: P-value threshold for drift detection
        
    Returns:
        Dictionary with drift detection results
    """
    if not ALIBI_AVAILABLE:
        logger.warning("Alibi-detect not available. Using basic drift detection.")
        return basic_drift_detection(reference_data, current_data, feature_names, p_val_threshold)
    
    try:
        logger.info("Detecting drift using KS-drift detector")
        
        # Initialize drift detector
        drift_detector = KSDrift(
            reference_data,
            p_val=p_val_threshold,
            alternative='two-sided'
        )
        
        # Run drift detection
        drift_result = drift_detector.predict(current_data)
        
        # Extract results
        is_drift = drift_result['data']['is_drift']
        p_values = drift_result['data']['p_val']
        
        # Create feature-level drift results
        feature_drift = {}
        if feature_names is not None:
            for i, feature in enumerate(feature_names):
                feature_drift[feature] = {
                    'p_value': float(p_values[i]),
                    'drift_detected': bool(p_values[i] < p_val_threshold)
                }
        
        # Create result dictionary
        result = {
            'drift_detected': bool(is_drift),
            'p_values': p_values.tolist() if isinstance(p_values, np.ndarray) else p_values,
            'feature_drift': feature_drift,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log drift detection results
        if is_drift:
            logger.warning(f"Drift detected in {sum(p_values < p_val_threshold)} features")
        else:
            logger.info("No drift detected")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in drift detection: {e}")
        return {
            'drift_detected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def basic_drift_detection(reference_data: np.ndarray, current_data: np.ndarray, 
                         feature_names: List[str] = None, 
                         p_val_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Basic drift detection using statistical tests
    
    Args:
        reference_data: Reference data (training data)
        current_data: Current data to check for drift
        feature_names: List of feature names
        p_val_threshold: P-value threshold for drift detection
        
    Returns:
        Dictionary with drift detection results
    """
    try:
        from scipy import stats
        
        # Initialize results
        p_values = []
        is_drift_feature = []
        
        # Check each feature for drift
        for i in range(reference_data.shape[1]):
            # Get feature data
            ref_feature = reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(ref_feature, curr_feature)
            
            # Store results
            p_values.append(float(p_value))
            is_drift_feature.append(bool(p_value < p_val_threshold))
        
        # Overall drift detection
        is_drift = any(is_drift_feature)
        
        # Create feature-level drift results
        feature_drift = {}
        if feature_names is not None:
            for i, feature in enumerate(feature_names):
                feature_drift[feature] = {
                    'p_value': p_values[i],
                    'drift_detected': is_drift_feature[i]
                }
        
        # Create result dictionary
        result = {
            'drift_detected': is_drift,
            'p_values': p_values,
            'feature_drift': feature_drift,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log drift detection results
        if is_drift:
            logger.warning(f"Drift detected in {sum(is_drift_feature)} features")
        else:
            logger.info("No drift detected")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in basic drift detection: {e}")
        return {
            'drift_detected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def calculate_confidence(predictions: np.ndarray, model_type: str = "classifier", 
                        probabilities: np.ndarray = None, 
                        actual: np.ndarray = None) -> np.ndarray:
    """
    Calculate confidence scores for predictions
    
    Args:
        predictions: Model predictions
        model_type: Type of model ('classifier' or 'regressor')
        probabilities: Probability predictions for classifiers
        actual: Actual values (optional)
        
    Returns:
        Array of confidence scores
    """
    try:
        if model_type == "classifier":
            # For classifiers, use probability estimates if available
            if probabilities is not None:
                # Use max probability as confidence
                confidence = np.max(probabilities, axis=1)
            else:
                # Fallback to distance from decision boundary (0.5)
                confidence = 1 - np.abs(0.5 - predictions)
        else:
            # For regressors, use prediction error if actual values are available
            if actual is not None:
                # Calculate absolute error
                abs_error = np.abs(predictions - actual)
                
                # Normalize errors to [0, 1] range (inverted, so lower error = higher confidence)
                max_error = np.max(abs_error) + 1e-8  # Avoid division by zero
                confidence = 1 - (abs_error / max_error)
            else:
                # Without actual values, use a simple heuristic based on prediction magnitude
                # Assumption: predictions closer to zero are less confident
                confidence = 1 / (1 + np.exp(-np.abs(predictions) * 2))
        
        return confidence
        
    except Exception as e:
        logger.error(f"Error calculating confidence scores: {e}")
        return np.ones_like(predictions) * 0.5  # Default confidence


def optimize_batch_size(data_size: int, feature_size: int, 
                       memory_limit_mb: int = 4096, 
                       use_gpu: bool = False) -> int:
    """
    Optimize batch size for large data processing
    
    Args:
        data_size: Number of data points
        feature_size: Number of features
        memory_limit_mb: Memory limit in MB
        use_gpu: Whether GPU is being used
        
    Returns:
        Optimal batch size
    """
    try:
        # Estimate memory per sample (in bytes)
        # Assuming float32 (4 bytes per value)
        bytes_per_sample = feature_size * 4
        
        # Add overhead for PyTorch tensors (approximately 2x)
        bytes_per_sample_with_overhead = bytes_per_sample * 2
        
        # Convert memory limit to bytes
        memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Calculate maximum batch size based on memory
        max_batch_size = memory_limit_bytes // bytes_per_sample_with_overhead
        
        # Adjust for GPU memory if using GPU
        if use_gpu:
            # GPU memory is more constrained, use a more conservative estimate
            max_batch_size = max_batch_size // 2
        
        # Ensure batch size is at least 1
        max_batch_size = max(1, max_batch_size)
        
        # Limit batch size to data size
        max_batch_size = min(max_batch_size, data_size)
        
        # Round to power of 2 for better performance
        batch_size = 2 ** int(np.log2(max_batch_size))
        
        logger.info(f"Optimized batch size: {batch_size} (data size: {data_size}, feature size: {feature_size})")
        
        return batch_size
        
    except Exception as e:
        logger.error(f"Error optimizing batch size: {e}")
        # Default to a conservative batch size
        return 32


def generate_shap_values(model: Any, data: np.ndarray, 
                        background_data: np.ndarray = None, 
                        feature_names: List[str] = None,
                        model_type: str = "tree",
                        max_samples: int = 100) -> Dict[str, Any]:
    """
    Generate SHAP values for model explainability
    
    Args:
        model: Trained model
        data: Data to explain
        background_data: Background data for explainer
        feature_names: List of feature names
        model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
        max_samples: Maximum number of samples to explain
        
    Returns:
        Dictionary with SHAP values and summary
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP is not available. Cannot generate SHAP values.")
        return {"error": "SHAP is not available"}
    
    try:
        logger.info(f"Generating SHAP values for model explainability (model type: {model_type})")
        
        # Limit number of samples to explain
        if data.shape[0] > max_samples:
            logger.info(f"Limiting SHAP analysis to {max_samples} samples")
            data = data[:max_samples]
        
        # Create background data if not provided
        if background_data is None and data.shape[0] > 10:
            background_data = data[:min(10, data.shape[0])]
        
        # Create explainer based on model type
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, background_data)
        elif model_type == "deep":
            explainer = shap.DeepExplainer(model, background_data)
        elif model_type == "kernel":
            explainer = shap.KernelExplainer(model.predict, background_data)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using KernelExplainer.")
            explainer = shap.KernelExplainer(model.predict, background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data)
        
        # Handle different return types
        if isinstance(shap_values, list):
            # For multi-class models
            shap_values_list = [sv.tolist() if isinstance(sv, np.ndarray) else sv for sv in shap_values]
            shap_result = {
                "shap_values": shap_values_list,
                "multi_class": True,
                "num_classes": len(shap_values_list)
            }
        else:
            # For binary classification and regression
            shap_result = {
                "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                "multi_class": False
            }
        
        # Add feature names if provided
        if feature_names is not None:
            shap_result["feature_names"] = feature_names
        
        # Add expected value if available
        if hasattr(explainer, "expected_value"):
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value.tolist()
            elif isinstance(expected_value, list):
                expected_value = [float(ev) if isinstance(ev, np.number) else ev for ev in expected_value]
            else:
                expected_value = float(expected_value) if isinstance(expected_value, np.number) else expected_value
            
            shap_result["expected_value"] = expected_value
        
        # Calculate feature importance
        if isinstance(shap_values, list):
            # For multi-class, average across classes
            importance = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dictionary
        feature_importance = {}
        if feature_names is not None:
            for i, feature in enumerate(feature_names):
                feature_importance[feature] = float(importance[i])
        else:
            for i in range(len(importance)):
                feature_importance[f"feature_{i}"] = float(importance[i])
        
        # Add feature importance to result
        shap_result["feature_importance"] = feature_importance
        
        logger.info(f"Generated SHAP values for {data.shape[0]} samples")
        
        return shap_result
        
    except Exception as e:
        logger.error(f"Error generating SHAP values: {e}")
        return {"error": str(e)}


def optimize_hyperparameters(model_fn: Callable, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           param_space: Dict[str, Any],
                           n_trials: int = 20,
                           timeout: int = 600,
                           metric: str = "accuracy") -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna
    
    Args:
        model_fn: Function that creates and returns a model given parameters
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        param_space: Parameter space definition
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        metric: Metric to optimize ('accuracy', 'f1', 'auc', 'rmse', 'mae')
        
    Returns:
        Dictionary with best parameters and optimization results
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna is not available. Cannot optimize hyperparameters.")
        return {"error": "Optuna is not available"}
    
    try:
        logger.info(f"Optimizing hyperparameters using Optuna (metric: {metric}, trials: {n_trials})")
        
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_config["values"])
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"], param_config.get("step", 1))
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=param_config.get("log", False))
                else:
                    logger.warning(f"Unknown parameter type: {param_config['type']} for {param_name}")
            
            # Create and train model
            model = model_fn(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            if metric == "accuracy":
                from sklearn.metrics import accuracy_score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            elif metric == "f1":
                from sklearn.metrics import f1_score
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average="weighted")
            elif metric == "auc":
                from sklearn.metrics import roc_auc_score
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred)
                else:
                    y_pred = model.predict(X_val)
                    score = roc_auc_score(y_val, y_pred)
            elif metric == "rmse":
                from sklearn.metrics import mean_squared_error
                y_pred = model.predict(X_val)
                score = -np.sqrt(mean_squared_error(y_val, y_pred))  # Negative for minimization
            elif metric == "mae":
                from sklearn.metrics import mean_absolute_error
                y_pred = model.predict(X_val)
                score = -mean_absolute_error(y_val, y_pred)  # Negative for minimization
            else:
                logger.warning(f"Unknown metric: {metric}. Using accuracy.")
                from sklearn.metrics import accuracy_score
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            
            return score
        
        # Create study
        study = optuna.create_study(direction="maximize")
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Create result dictionary
        result = {
            "best_params": best_params,
            "best_value": float(best_value),
            "n_trials": n_trials,
            "metric": metric,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Hyperparameter optimization completed. Best {metric}: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error optimizing hyperparameters: {e}")
        return {"error": str(e)}