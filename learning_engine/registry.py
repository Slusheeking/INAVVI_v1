"""Model registry for versioning and managing trained models."""

import json
import logging
import os
import pickle
import mlflow # Added MLflow import
import mlflow.xgboost # Import specific flavors as needed
import mlflow.sklearn # Example for sklearn models
# Add other flavors like mlflow.pytorch if needed
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
        mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        # --- MLflow Integration Start ---
        mlflow_model_path = f"{model_name}_v{version}" # Path within the run's artifacts
        model_uri = None
        try:
            # Log model artifact to MLflow artifact store
            # This needs to happen within an active MLflow run context
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path=mlflow_model_path,
                    # input_example=..., # Optional: Provide sample input
                    # signature=..., # Optional: Define model signature
                )
            elif model_type == 'sklearn': # Example
                 mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=mlflow_model_path
                 )
            # Add other model types (e.g., pytorch) here
            else:
                # Fallback: Log as a generic artifact (less useful)
                # Or raise an error if type is unsupported
                 logging.warning(f"MLflow flavor for model type '{model_type}' not explicitly handled. Logging as generic artifact.")
                 # You might need a temporary save mechanism here if the model isn't easily pickled by MLflow directly
                 # temp_path = os.path.join(self.models_dir, f"{model_name}_v{version}_temp.pkl")
                 # with open(temp_path, 'wb') as f: pickle.dump(model, f)
                 # mlflow.log_artifact(temp_path, artifact_path=mlflow_model_path)
                 # os.remove(temp_path) # Clean up temp file
                 # For now, we skip logging unsupported types directly
                 raise NotImplementedError(f"MLflow logging not implemented for model type: {model_type}")


            # Construct the model URI from the run
            if mlflow_run_id:
                 logged_model_uri = f"runs:/{mlflow_run_id}/{mlflow_model_path}"

                 # Register the model in MLflow Model Registry
                 registered_model_info = mlflow.register_model(
                     model_uri=logged_model_uri,
                     name=model_name
                 )
                 model_uri = registered_model_info.source # Store the 'models:/...' URI
                 logging.info(f"Registered model in MLflow: {model_uri} (version {registered_model_info.version})")
            else:
                 logging.warning("No active MLflow run found. Model artifact logged but not registered.")
                 model_uri = f"local_artifact:{mlflow_model_path}" # Placeholder URI

        except Exception as mlflow_e:
            logging.exception(f"Error logging or registering model with MLflow: {mlflow_e}")
            # Decide how to handle failure: raise error, return failure, etc.
            # For now, we'll continue and store metadata locally/redis without MLflow URI

        # --- MLflow Integration End ---

        # Remove direct model saving with pickle
        # model_filename = f"{model_name}_v{version}.pkl"
        # model_path = os.path.join(self.models_dir, model_filename)
        # with open(model_path, 'wb') as f:
        #     pickle.dump(model, f)

        # Store metadata
        # Store metadata (including MLflow info if available)
        self.models_metadata[model_name]["versions"][str(version)] = {
            # "path": model_path, # Replaced by MLflow URI
            "mlflow_uri": model_uri, # Store the MLflow URI (runs:/... or models:/...)
            "mlflow_run_id": mlflow_run_id, # Store the run ID for traceability
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
                        "mlflow_uri": model_uri or "",
                        "mlflow_run_id": mlflow_run_id or "",
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

        # Get MLflow URI from metadata
        metadata = self.models_metadata[model_name]["versions"][str(version)]
        model_uri = metadata.get("mlflow_uri")

        if not model_uri:
            # Fallback or error if MLflow URI is missing (e.g., older model)
            # For now, try the old path if it exists (assuming 'path' key might still exist for old models)
            old_path = metadata.get("path")
            if old_path and os.path.exists(old_path):
                 logging.warning(f"MLflow URI not found for {model_name} v{version}. Loading from old path: {old_path}")
                 with open(old_path, 'rb') as f:
                     return pickle.load(f)
            else:
                 raise ValueError(f"Cannot load model {model_name} v{version}: MLflow URI or valid path missing.")

        # --- MLflow Integration Start ---
        try:
            logging.info(f"Loading model {model_name} v{version} from MLflow URI: {model_uri}")
            # Determine flavor based on model type stored in metadata
            model_type = self.models_metadata[model_name].get("type", "unknown")

            if model_type == 'xgboost':
                loaded_model = mlflow.xgboost.load_model(model_uri)
            elif model_type == 'sklearn':
                loaded_model = mlflow.sklearn.load_model(model_uri)
            # Add other flavors as needed (e.g., pytorch)
            # elif model_type == 'pytorch':
            #     loaded_model = mlflow.pytorch.load_model(model_uri)
            else:
                 # Attempt generic pyfunc loading if flavor unknown or not specific
                 try:
                     loaded_model = mlflow.pyfunc.load_model(model_uri)
                     logging.warning(f"Loaded model {model_name} v{version} using generic pyfunc flavor.")
                 except Exception as pyfunc_e:
                     raise ValueError(f"Unsupported model type '{model_type}' or failed generic load for URI {model_uri}: {pyfunc_e}")

            return loaded_model
        except Exception as mlflow_e:
            logging.exception(f"Error loading model from MLflow URI {model_uri}: {mlflow_e}")
            raise # Re-raise the exception after logging
        # --- MLflow Integration End ---

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
