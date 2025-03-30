#!/usr/bin/env python3
"""
Model Explainability Module

This module provides tools for model explainability:
1. SHAP (SHapley Additive exPlanations) for feature importance
2. Partial Dependence Plots (PDP) for feature relationships
3. Feature importance visualization
4. Prediction confidence scoring
5. Model interpretation utilities

These tools help understand model decisions and build trust in the system.
"""

import os
import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime

from utils.logging_config import get_logger

# Configure logging
logger = get_logger("ml_engine.explainability")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP is available for model explainability")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with 'pip install shap' for model explainability")

# Try to import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib is available for visualization")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install with 'pip install matplotlib' for visualization")


class ModelExplainer:
    """
    Model explainability class for generating explanations and visualizations
    
    This class provides methods for:
    1. Generating SHAP explanations
    2. Calculating feature importance
    3. Visualizing model predictions
    4. Calculating prediction confidence
    5. Generating feature drift reports
    """
    
    def __init__(self, model: Any, model_type: str = "tree", feature_names: Optional[List[str]] = None):
        """
        Initialize ModelExplainer
        
        Args:
            model: Trained model to explain
            model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
            feature_names: List of feature names
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        
        # Check if SHAP is available
        if not SHAP_AVAILABLE:
            logger.warning("SHAP is not available. Model explainability will be limited.")
        
        # Initialize explainer if SHAP is available
        if SHAP_AVAILABLE and model is not None:
            try:
                logger.info(f"Initializing {model_type} explainer")
                if model_type == "tree":
                    self.explainer = shap.TreeExplainer(model)
                # Other explainer types will be initialized when needed with data
            except Exception as e:
                logger.error(f"Error initializing explainer: {e}")
    
    def explain(self, data: Union[pd.DataFrame, np.ndarray], 
               background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
               max_samples: int = 100,
               output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate model explanations
        
        Args:
            data: Data to explain
            background_data: Background data for explainer
            max_samples: Maximum number of samples to explain
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with explanation results
        """
        return explain_model(
            self.model, 
            data, 
            feature_names=self.feature_names,
            model_type=self.model_type,
            background_data=background_data,
            max_samples=max_samples,
            output_dir=output_dir
        )
    
    def calculate_confidence(self, predictions: np.ndarray, 
                           probabilities: Optional[np.ndarray] = None,
                           actual: Optional[np.ndarray] = None,
                           shap_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate confidence scores for predictions
        
        Args:
            predictions: Model predictions
            probabilities: Probability predictions for classifiers
            actual: Actual values (optional)
            shap_values: SHAP values (optional)
            
        Returns:
            Array of confidence scores
        """
        model_type = "classifier" if self.model_type in ["tree", "linear"] else "regressor"
        return calculate_prediction_confidence(
            predictions,
            model_type=model_type,
            probabilities=probabilities,
            actual=actual,
            shap_values=shap_values
        )
    
    def detect_drift(self, reference_data: Union[pd.DataFrame, np.ndarray],
                    current_data: Union[pd.DataFrame, np.ndarray],
                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate feature drift report comparing reference and current data
        
        Args:
            reference_data: Reference data (training data)
            current_data: Current data to check for drift
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with drift report
        """
        return generate_feature_drift_report(
            reference_data,
            current_data,
            feature_names=self.feature_names,
            model=self.model,
            model_type=self.model_type,
            output_dir=output_dir
        )


def explain_model(model: Any, data: Union[pd.DataFrame, np.ndarray], 
                 feature_names: Optional[List[str]] = None,
                 model_type: str = "tree",
                 background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                 max_samples: int = 100,
                 output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate model explanations using SHAP
    
    Args:
        model: Trained model
        data: Data to explain
        feature_names: List of feature names
        model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
        background_data: Background data for explainer
        max_samples: Maximum number of samples to explain
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with explanation results
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP is not available. Cannot explain model.")
        return {"error": "SHAP is not available"}
    
    try:
        logger.info(f"Generating model explanations using SHAP (model type: {model_type})")
        
        # Convert data to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            if feature_names is None:
                feature_names = data.columns.tolist()
            data_values = data.values
        else:
            data_values = data
        
        # Limit number of samples to explain
        if data_values.shape[0] > max_samples:
            logger.info(f"Limiting SHAP analysis to {max_samples} samples")
            data_values = data_values[:max_samples]
        
        # Create background data if not provided
        if background_data is None and data_values.shape[0] > 10:
            background_values = data_values[:min(10, data_values.shape[0])]
        elif background_data is not None:
            if isinstance(background_data, pd.DataFrame):
                background_values = background_data.values
            else:
                background_values = background_data
        else:
            background_values = None
        
        # Create explainer based on model type
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, background_values)
        elif model_type == "deep":
            explainer = shap.DeepExplainer(model, background_values)
        elif model_type == "kernel":
            # For black-box models
            predict_fn = getattr(model, "predict_proba", None)
            if predict_fn is None:
                predict_fn = model.predict
            explainer = shap.KernelExplainer(predict_fn, background_values)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using KernelExplainer.")
            predict_fn = getattr(model, "predict_proba", None)
            if predict_fn is None:
                predict_fn = model.predict
            explainer = shap.KernelExplainer(predict_fn, background_values)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data_values)
        
        # Create result dictionary
        result = {
            "model_type": model_type,
            "num_samples": data_values.shape[0],
            "num_features": data_values.shape[1],
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle different return types
        if isinstance(shap_values, list):
            # For multi-class models
            result["multi_class"] = True
            result["num_classes"] = len(shap_values)
            
            # Calculate feature importance for each class
            feature_importance = {}
            for i, sv in enumerate(shap_values):
                class_importance = np.abs(sv).mean(axis=0)
                class_name = f"class_{i}"
                
                if feature_names is not None:
                    class_importance_dict = {
                        feature: float(importance)
                        for feature, importance in zip(feature_names, class_importance)
                    }
                else:
                    class_importance_dict = {
                        f"feature_{j}": float(importance)
                        for j, importance in enumerate(class_importance)
                    }
                
                feature_importance[class_name] = class_importance_dict
            
            # Calculate overall feature importance
            overall_importance = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
        else:
            # For binary classification and regression
            result["multi_class"] = False
            
            # Calculate feature importance
            overall_importance = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary for single output
            feature_importance = {"overall": {}}
            if feature_names is not None:
                feature_importance["overall"] = {
                    feature: float(importance)
                    for feature, importance in zip(feature_names, overall_importance)
                }
            else:
                feature_importance["overall"] = {
                    f"feature_{i}": float(importance)
                    for i, importance in enumerate(overall_importance)
                }
        
        # Add feature importance to result
        result["feature_importance"] = feature_importance
        
        # Add expected value if available
        if hasattr(explainer, "expected_value"):
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value.tolist()
            elif isinstance(expected_value, list):
                expected_value = [float(ev) if isinstance(ev, np.number) else ev for ev in expected_value]
            else:
                expected_value = float(expected_value) if isinstance(expected_value, np.number) else expected_value
            
            result["expected_value"] = expected_value
        
        # Generate visualizations if output directory is provided
        if output_dir is not None and MATPLOTLIB_AVAILABLE:
            os.makedirs(output_dir, exist_ok=True)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            if result["multi_class"]:
                # For multi-class, plot first class as example
                shap.summary_plot(
                    shap_values[0],
                    data_values,
                    feature_names=feature_names,
                    show=False
                )
            else:
                shap.summary_plot(
                    shap_values,
                    data_values,
                    feature_names=feature_names,
                    show=False
                )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            if result["multi_class"]:
                # For multi-class, plot first class as example
                shap.summary_plot(
                    shap_values[0],
                    data_values,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False
                )
            else:
                shap.summary_plot(
                    shap_values,
                    data_values,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False
                )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_importance.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            # Save result as JSON
            with open(os.path.join(output_dir, "shap_results.json"), "w") as f:
                # Convert numpy arrays to lists
                json_result = result.copy()
                json.dump(json_result, f, indent=2)
            
            result["visualizations"] = {
                "summary_plot": os.path.join(output_dir, "shap_summary.png"),
                "importance_plot": os.path.join(output_dir, "shap_importance.png"),
                "results_json": os.path.join(output_dir, "shap_results.json")
            }
        
        logger.info(f"Generated SHAP explanations for {data_values.shape[0]} samples")
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating model explanations: {e}")
        return {"error": str(e)}


def calculate_prediction_confidence(predictions: np.ndarray, model_type: str = "classifier", 
                                  probabilities: Optional[np.ndarray] = None, 
                                  actual: Optional[np.ndarray] = None,
                                  shap_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate confidence scores for predictions
    
    Args:
        predictions: Model predictions
        model_type: Type of model ('classifier' or 'regressor')
        probabilities: Probability predictions for classifiers
        actual: Actual values (optional)
        shap_values: SHAP values (optional)
        
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
                
            # Adjust confidence based on SHAP values if available
            if shap_values is not None:
                # Higher absolute SHAP values indicate stronger influence on prediction
                # Use mean absolute SHAP value as a measure of confidence
                if isinstance(shap_values, list):
                    # For multi-class models, use the class with highest prediction
                    class_indices = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else np.ones(len(predictions), dtype=int)
                    shap_magnitude = np.array([np.abs(shap_values[class_indices[i]][i]).mean() for i in range(len(predictions))])
                else:
                    shap_magnitude = np.abs(shap_values).mean(axis=1)
                
                # Normalize SHAP magnitude to [0, 0.5] range
                max_shap = np.max(shap_magnitude) + 1e-8  # Avoid division by zero
                shap_confidence = 0.5 * (shap_magnitude / max_shap)
                
                # Combine probability-based confidence with SHAP-based confidence
                confidence = 0.5 * confidence + shap_confidence
        else:
            # For regressors, use prediction error if actual values are available
            if actual is not None:
                # Calculate absolute error
                abs_error = np.abs(predictions - actual)
                
                # Normalize errors to [0, 1] range (inverted, so lower error = higher confidence)
                max_error = np.max(abs_error) + 1e-8  # Avoid division by zero
                confidence = 1 - (abs_error / max_error)
            elif shap_values is not None:
                # Use SHAP values to estimate uncertainty
                # Higher absolute SHAP values indicate stronger influence on prediction
                shap_magnitude = np.abs(shap_values).mean(axis=1)
                
                # Normalize SHAP magnitude to [0, 1] range
                max_shap = np.max(shap_magnitude) + 1e-8  # Avoid division by zero
                confidence = shap_magnitude / max_shap
            else:
                # Without actual values or SHAP values, use a simple heuristic based on prediction magnitude
                # Assumption: predictions closer to zero are less confident
                confidence = 1 / (1 + np.exp(-np.abs(predictions) * 2))
        
        return confidence
        
    except Exception as e:
        logger.error(f"Error calculating confidence scores: {e}")
        return np.ones_like(predictions) * 0.5  # Default confidence


def generate_feature_drift_report(reference_data: Union[pd.DataFrame, np.ndarray],
                                current_data: Union[pd.DataFrame, np.ndarray],
                                feature_names: Optional[List[str]] = None,
                                model: Optional[Any] = None,
                                model_type: str = "tree",
                                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate feature drift report comparing reference and current data
    
    Args:
        reference_data: Reference data (training data)
        current_data: Current data to check for drift
        feature_names: List of feature names
        model: Trained model (optional)
        model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with drift report
    """
    try:
        logger.info("Generating feature drift report")
        
        # Convert data to DataFrames if numpy arrays
        if isinstance(reference_data, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(reference_data.shape[1])]
            reference_df = pd.DataFrame(reference_data, columns=feature_names)
        else:
            reference_df = reference_data
            if feature_names is None:
                feature_names = reference_df.columns.tolist()
        
        if isinstance(current_data, np.ndarray):
            current_df = pd.DataFrame(current_data, columns=feature_names)
        else:
            current_df = current_data
        
        # Calculate basic statistics
        ref_stats = reference_df.describe().transpose()
        curr_stats = current_df.describe().transpose()
        
        # Calculate drift metrics
        drift_metrics = {}
        for feature in feature_names:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
                
            ref_values = reference_df[feature].values
            curr_values = current_df[feature].values
            
            # Calculate statistical tests
            from scipy import stats
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
            
            # Jensen-Shannon divergence
            try:
                # Create histograms
                ref_hist, bin_edges = np.histogram(ref_values, bins=20, density=True)
                curr_hist, _ = np.histogram(curr_values, bins=bin_edges, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                ref_hist = ref_hist + epsilon
                curr_hist = curr_hist + epsilon
                
                # Normalize
                ref_hist = ref_hist / np.sum(ref_hist)
                curr_hist = curr_hist / np.sum(curr_hist)
                
                # Calculate JS divergence
                m = 0.5 * (ref_hist + curr_hist)
                js_div = 0.5 * (stats.entropy(ref_hist, m) + stats.entropy(curr_hist, m))
            except Exception as e:
                logger.warning(f"Error calculating JS divergence for {feature}: {e}")
                js_div = None
            
            # Calculate percent change in mean and std
            mean_change_pct = ((curr_stats.loc[feature, "mean"] - ref_stats.loc[feature, "mean"]) / 
                              (abs(ref_stats.loc[feature, "mean"]) + 1e-10)) * 100
            std_change_pct = ((curr_stats.loc[feature, "std"] - ref_stats.loc[feature, "std"]) / 
                             (abs(ref_stats.loc[feature, "std"]) + 1e-10)) * 100
            
            drift_metrics[feature] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "js_divergence": float(js_div) if js_div is not None else None,
                "mean_change_pct": float(mean_change_pct),
                "std_change_pct": float(std_change_pct),
                "drift_detected": float(ks_pvalue) < 0.05  # Using KS test p-value < 0.05 as drift indicator
            }
        
        # Calculate feature importance if model is provided
        feature_importance = {}
        if model is not None and SHAP_AVAILABLE:
            try:
                # Generate SHAP values for reference data
                ref_shap_result = explain_model(
                    model,
                    reference_df,
                    feature_names=feature_names,
                    model_type=model_type,
                    max_samples=min(100, len(reference_df))
                )
                
                # Generate SHAP values for current data
                curr_shap_result = explain_model(
                    model,
                    current_df,
                    feature_names=feature_names,
                    model_type=model_type,
                    max_samples=min(100, len(current_df))
                )
                
                # Extract feature importance
                if "feature_importance" in ref_shap_result and "feature_importance" in curr_shap_result:
                    ref_importance = ref_shap_result["feature_importance"]["overall"] if "overall" in ref_shap_result["feature_importance"] else ref_shap_result["feature_importance"]
                    curr_importance = curr_shap_result["feature_importance"]["overall"] if "overall" in curr_shap_result["feature_importance"] else curr_shap_result["feature_importance"]
                    
                    # Calculate importance change
                    for feature in feature_names:
                        if feature in ref_importance and feature in curr_importance:
                            ref_imp = ref_importance[feature]
                            curr_imp = curr_importance[feature]
                            
                            # Calculate percent change
                            imp_change_pct = ((curr_imp - ref_imp) / (abs(ref_imp) + 1e-10)) * 100
                            
                            feature_importance[feature] = {
                                "reference_importance": float(ref_imp),
                                "current_importance": float(curr_imp),
                                "importance_change_pct": float(imp_change_pct),
                                "importance_drift": abs(imp_change_pct) > 20  # Flag if importance changed by more than 20%
                            }
            except Exception as e:
                logger.error(f"Error calculating feature importance drift: {e}")
        
        # Create result dictionary
        result = {
            "reference_data_size": len(reference_df),
            "current_data_size": len(current_df),
            "num_features": len(feature_names),
            "drift_metrics": drift_metrics,
            "feature_importance_drift": feature_importance,
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate overall drift score
        drift_detected_features = sum(1 for feature, metrics in drift_metrics.items() if metrics["drift_detected"])
        result["drift_score"] = drift_detected_features / len(drift_metrics) if drift_metrics else 0
        result["drift_detected"] = result["drift_score"] > 0.1  # Flag if more than 10% of features have drift
        
        # Generate visualizations if output directory is provided
        if output_dir is not None and MATPLOTLIB_AVAILABLE:
            os.makedirs(output_dir, exist_ok=True)
            
            # Distribution comparison plots for top drifting features
            top_drift_features = sorted(
                drift_metrics.keys(),
                key=lambda f: drift_metrics[f]["ks_statistic"],
                reverse=True
            )[:min(5, len(drift_metrics))]
            
            for feature in top_drift_features:
                plt.figure(figsize=(10, 6))
                plt.hist(reference_df[feature], bins=30, alpha=0.5, label="Reference")
                plt.hist(current_df[feature], bins=30, alpha=0.5, label="Current")
                plt.title(f"Distribution Comparison: {feature}")
                plt.xlabel(feature)
                plt.ylabel("Frequency")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"drift_{feature}.png"), dpi=300, bbox_inches="tight")
                plt.close()
            
            # Drift score heatmap
            plt.figure(figsize=(12, 8))
            drift_scores = [drift_metrics[f]["ks_statistic"] for f in feature_names if f in drift_metrics]
            feature_labels = [f for f in feature_names if f in drift_metrics]
            
            # Sort by drift score
            sorted_indices = np.argsort(drift_scores)[::-1]
            sorted_scores = [drift_scores[i] for i in sorted_indices]
            sorted_labels = [feature_labels[i] for i in sorted_indices]
            
            # Plot horizontal bar chart
            plt.barh(range(len(sorted_scores)), sorted_scores, align='center')
            plt.yticks(range(len(sorted_scores)), sorted_labels)
            plt.xlabel('Drift Score (KS Statistic)')
            plt.title('Feature Drift Scores')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "drift_scores.png"), dpi=300, bbox_inches="tight")
            plt.close()
            
            # Save result as JSON
            with open(os.path.join(output_dir, "drift_report.json"), "w") as f:
                json.dump(result, f, indent=2)
            
            result["visualizations"] = {
                "drift_scores": os.path.join(output_dir, "drift_scores.png"),
                "feature_distributions": [os.path.join(output_dir, f"drift_{feature}.png") for feature in top_drift_features],
                "report_json": os.path.join(output_dir, "drift_report.json")
            }
        
        logger.info(f"Generated feature drift report. Drift detected: {result['drift_detected']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating feature drift report: {e}")
        return {"error": str(e)}