"""Statistical drift detection for model monitoring."""

import numpy as np
from scipy import stats
from typing import Dict, Tuple
import logging

class DriftDetector:
    """
    Detects statistical drift in model inputs and outputs.
    
    Methods:
    - KS test for feature distribution drift
    - Population stability index (PSI)
    - Prediction drift monitoring
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.reference_distributions = {}

    def set_reference(self, feature_name: str, values: np.ndarray):
        """
        Set reference distribution for a feature.
        
        Args:
            feature_name: Name of feature to monitor
            values: Reference values (training data distribution)
        """
        self.reference_distributions[feature_name] = {
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values)
        }

    def detect_feature_drift(self, feature_name: str, new_values: np.ndarray) -> Tuple[bool, float]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            feature_name: Feature to check
            new_values: New sample values
            
        Returns:
            Tuple of (drift_detected, p_value)
        """
        if feature_name not in self.reference_distributions:
            raise ValueError(f"No reference distribution for {feature_name}")
            
        ref = self.reference_distributions[feature_name]['values']
        stat, p_value = stats.ks_2samp(ref, new_values)
        return p_value < self.alpha, p_value

    def calculate_psi(self, feature_name: str, new_values: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            feature_name: Feature to analyze
            new_values: New sample values
            bins: Number of bins for histogram
            
        Returns:
            PSI score (values > 0.25 indicate significant drift)
        """
        ref = self.reference_distributions[feature_name]['values']
        
        # Create bins based on reference distribution
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(ref, percentiles)
        
        # Handle single value edge case
        if len(set(bin_edges)) < len(bin_edges):
            return 0.0
            
        # Calculate histograms
        ref_hist, _ = np.histogram(ref, bins=bin_edges)
        new_hist, _ = np.histogram(new_values, bins=bin_edges)
        
        # Convert to percentages
        ref_perc = ref_hist / len(ref)
        new_perc = new_hist / len(new_values)
        
        # Calculate PSI
        psi = np.sum((new_perc - ref_perc) * np.log(new_perc / ref_perc))
        return psi

    def detect_prediction_drift(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Detect drift in model predictions vs actuals.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of drift metrics
        """
        metrics = {}
        
        # Classification metrics drift
        if len(np.unique(y_true)) <= 10:  # Assume classification
            metrics['accuracy_drift'] = np.abs(np.mean(y_true == y_pred) - 0.5)
        else:  # Regression
            residuals = y_true - y_pred
            metrics['residual_mean'] = np.mean(residuals)
            metrics['residual_std'] = np.std(residuals)
            
        return metrics
