"""Model registry for versioning and managing trained models."""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

class ModelRegistry:
    """
    Model registry that manages:
    - Model versioning
    - Model storage
    - Metadata tracking
    - Redis integration (optional)
    """

    def __init__(self, models_dir: str, redis_client=None):
        """
        Initialize the model registry.

        Args:
            models_dir: Directory to store model files
            redis_client: Optional Redis client
        """
        self.models_dir = models_dir
        self.redis = redis_client
        self.models_metadata = {}

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Load existing metadata
        self._load_metadata()

    def register_model(
        self,
        model_name: str,
        model: Any,
        model_type: str,
        metrics: Dict[str, float],
        feature_names: list,
        hyperparams: Dict[str, Any]
    ) -> int:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            model: Trained model object
            model_type: Type of model (e.g. 'xgboost', 'pytorch')
            metrics: Dictionary of evaluation metrics
            feature_names: List of feature names
            hyperparams: Dictionary of hyperparameters

        Returns:
            Version number assigned
        """
        # Initialize model metadata if new
        if model_name not in self.models_metadata:
            self.models_metadata[model_name] = {
                "type": model_type,
                "versions": {},
                "latest_version": 0,
                "created": datetime.now().isoformat()
            }

        # Get next version number
        version = self.models_metadata[model_name]["latest_version"] + 1

        # Save model file
        model_filename = f"{model_name}_v{version}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Store metadata
        self.models_metadata[model_name]["versions"][str(version)] = {
            "path": model_path,
            "metrics": metrics,
            "feature_names": feature_names,
            "hyperparams": hyperparams,
            "registered": datetime.now().isoformat(),
            "deployed": False
        }

        # Update latest version
        self.models_metadata[model_name]["latest_version"] = version

        # Save metadata
        self._save_metadata()

        # Update Redis if available
        if self.redis:
            try:
                self.redis.hset(
                    f"model:metadata:{model_name}:{version}",
                    mapping={
                        "metrics": json.dumps(metrics),
                        "feature_names": json.dumps(feature_names),
                        "hyperparams": json.dumps(hyperparams),
                        "registered": datetime.now().isoformat(),
                        "deployed": "False"
                    }
                )
            except Exception as e:
                logging.exception(f"Error updating Redis model metadata: {e}")

        logging.info(f"Registered {model_name} v{version}")
        return version

    def get_model(self, model_name: str, version: int) -> Any:
        """
        Get a registered model.

        Args:
            model_name: Name of the model
            version: Version number

        Returns:
            Loaded model object
        """
        if model_name not in self.models_metadata:
            raise ValueError(f"Model {model_name} not found")

        if str(version) not in self.models_metadata[model_name]["versions"]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        model_path = self.models_metadata[model_name]["versions"][str(version)]["path"]
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def get_model_metadata(self, model_name: str, version: int) -> Dict[str, Any]:
        """
        Get metadata for a model version.

        Args:
            model_name: Name of the model
            version: Version number

        Returns:
            Dictionary of model metadata
        """
        if model_name not in self.models_metadata:
            raise ValueError(f"Model {model_name} not found")

        if str(version) not in self.models_metadata[model_name]["versions"]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        return self.models_metadata[model_name]["versions"][str(version)]

    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_path = os.path.join(self.models_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.models_metadata = json.load(f)
            except Exception as e:
                logging.exception(f"Error loading model metadata: {e}")
                self.models_metadata = {}

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_path = os.path.join(self.models_dir, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.models_metadata, f, indent=2)
        except Exception as e:
            logging.exception(f"Error saving model metadata: {e}")

    def mark_deployed(self, model_name: str, version: int):
        """
        Mark a model version as deployed.

        Args:
            model_name: Name of the model
            version: Version number
        """
        if model_name not in self.models_metadata:
            raise ValueError(f"Model {model_name} not found")

        if str(version) not in self.models_metadata[model_name]["versions"]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        self.models_metadata[model_name]["versions"][str(version)]["deployed"] = True
        self._save_metadata()

        # Update Redis if available
        if self.redis:
            try:
                self.redis.hset(
                    f"model:metadata:{model_name}:{version}",
                    "deployed",
                    "True"
                )
            except Exception as e:
                logging.exception(f"Error updating Redis deployment status: {e}")
