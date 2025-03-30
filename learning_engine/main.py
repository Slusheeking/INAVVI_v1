"""Main learning engine module that orchestrates model training and deployment."""

import logging
import time
from typing import Optional

from .base import LearningEngine
from .registry import ModelRegistry
from .deployer import ModelDeployer
from .tracker import ModelPerformanceTracker

class LearningEngineMain:
    """
    Main learning engine orchestrator that manages:
    - Model training
    - Versioning
    - Deployment
    - Performance tracking
    """

    def __init__(self, models_dir: str, redis_client=None):
        """
        Initialize the learning engine.

        Args:
            models_dir: Directory to store model files
            redis_client: Optional Redis client
        """
        self.registry = ModelRegistry(models_dir, redis_client)
        self.deployer = ModelDeployer(self.registry, redis_client)
        self.tracker = ModelPerformanceTracker(redis_client)
        self.engine = LearningEngine()

        # Configuration
        self.performance_thresholds = {
            "accuracy": 0.85,
            "f1_score": 0.80,
            "precision": 0.75,
            "recall": 0.75
        }

    def run_training_cycle(self, data):
        """
        Run a complete training cycle:
        1. Train new model
        2. Evaluate performance
        3. Register version
        4. Check for deployment
        """
        logging.info("Starting training cycle")

        # Train and evaluate model
        model, metrics = self.engine.train_and_evaluate(data)

        # Register new version
        model_name = self.engine.model_name
        version = self.registry.register_model(
            model_name=model_name,
            model=model,
            model_type=self.engine.model_type,
            metrics=metrics,
            feature_names=data.feature_names,
            hyperparams=self.engine.get_hyperparameters()
        )

        # Track initial metrics
        self.tracker.track_metrics(model_name, version, metrics)

        # Check if we should deploy this version
        self._check_deployment(model_name, version, metrics)

        logging.info(f"Completed training cycle for {model_name} v{version}")

    def _check_deployment(self, model_name: str, version: int, metrics: dict):
        """
        Determine if a new model version should be deployed.

        Args:
            model_name: Name of the model
            version: Version number
            metrics: Performance metrics
        """
        current_version = self.deployer.get_deployed_version(model_name)

        # If no current deployment, deploy this version
        if current_version is None:
            self.deployer.deploy_model(model_name, version)
            return

        # Get current version metrics
        current_metrics = self.tracker.get_latest_metrics(model_name, current_version)
        if current_metrics is None:
            self.deployer.deploy_model(model_name, version)
            return

        # Compare metrics
        improvement = False
        for metric, threshold in self.performance_thresholds.items():
            if metrics.get(metric, 0) > (current_metrics.get(metric, 0) + 0.05):
                improvement = True
                break

        # Deploy if significant improvement
        if improvement:
            logging.info(f"Deploying improved version {version} of {model_name}")
            self.deployer.deploy_model(model_name, version)
        else:
            logging.info(f"Version {version} doesn't meet improvement threshold")

    def monitor_performance(self):
        """
        Continuously monitor deployed model performance.
        """
        while True:
            try:
                # Get all deployed models
                for model_name, deployments in self.deployer.deployments.items():
                    for env, deployment in deployments.items():
                        version = deployment["version"]

                        # Check for performance degradation
                        for metric, threshold in self.performance_thresholds.items():
                            degraded = self.tracker.check_performance_degradation(
                                model_name,
                                version,
                                metric,
                                threshold
                            )

                            if degraded:
                                logging.warning(
                                    f"Performance degradation detected for {model_name} v{version} "
                                    f"in {env} environment"
                                )

                # Sleep between checks
                time.sleep(3600)  # Check hourly

            except Exception as e:
                logging.exception(f"Error in performance monitoring: {e}")
                time.sleep(60)  # Retry after short delay

    def run(self, data, monitor: bool = True):
        """
        Main entry point to run the learning engine.

        Args:
            data: Training data
            monitor: Whether to start performance monitoring
        """
        # Start performance monitoring in background if requested
        if monitor:
            import threading
            monitor_thread = threading.Thread(
                target=self.monitor_performance,
                daemon=True
            )
            monitor_thread.start()

        # Run initial training
        self.run_training_cycle(data)

        # Schedule periodic retraining
        while True:
            time.sleep(24 * 3600)  # Retrain daily
            self.run_training_cycle(data)
