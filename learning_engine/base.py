"""Base learning engine class with core training functionality."""

import logging
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score
)

class LearningEngine:
    """
    Base learning engine that provides core training and evaluation functionality.

    Subclasses should implement:
    - _train_model() - Model-specific training logic
    - _evaluate_model() - Model-specific evaluation
    - model_type - String identifier for model type
    - model_name - Name of the model
    """

    def __init__(self):
        """Initialize the learning engine."""
        self.model = None
        self.model_type = "base"
        self.model_name = "base_model"
        self.hyperparams = {}

    def train_and_evaluate(self, data) -> Tuple[Any, Dict[str, float]]:
        """
        Train and evaluate a model.

        Args:
            data: Training data with features and labels

        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        logging.info(f"Training {self.model_name} ({self.model_type})")

        # Train model
        self.model = self._train_model(data)

        # Evaluate model
        metrics = self._evaluate_model(data)

        return self.model, metrics

    def _train_model(self, data) -> Any:
        """
        Train the model (to be implemented by subclasses).

        Args:
            data: Training data

        Returns:
            Trained model
        """
        raise NotImplementedError("Subclasses must implement _train_model")

    def _evaluate_model(self, data) -> Dict[str, float]:
        """
        Evaluate the model (to be implemented by subclasses).

        Args:
            data: Evaluation data

        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement _evaluate_model")

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the current hyperparameters.

        Returns:
            Dictionary of hyperparameters
        """
        return self.hyperparams.copy()

    def set_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Set hyperparameters for the model.

        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.hyperparams.update(hyperparams)

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calculate common evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
