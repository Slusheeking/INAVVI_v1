"""Model deployment manager for promoting and managing model versions."""

import json
import logging
import os
import time
from typing import Optional

class ModelDeployer:
    """
    Handles model deployment and version management.

    Features:
    - Model version promotion/demotion
    - A/B testing support
    - Deployment status tracking
    - Redis integration
    """

    def __init__(self, model_registry, redis_client=None):
        """
        Initialize the model deployer.

        Args:
            model_registry: ModelRegistry instance
            redis_client: Optional Redis client
        """
        self.registry = model_registry
        self.redis = redis_client
        self.deployments = {}

    def deploy_model(self, model_name: str, version: int, environment: str = "production"):
        """
        Deploy a specific model version to an environment.

        Args:
            model_name: Name of the model
            version: Version number to deploy
            environment: Target environment ('production', 'staging', etc.)
        """
        # Validate model exists
        if model_name not in self.registry.models_metadata:
            raise ValueError(f"Model {model_name} not found in registry")

        # Validate version exists
        if str(version) not in self.registry.models_metadata[model_name]["versions"]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        # Update deployment tracking
        if model_name not in self.deployments:
            self.deployments[model_name] = {}

        self.deployments[model_name][environment] = {
            "version": version,
            "timestamp": time.time(),
            "status": "active"
        }

        # Update Redis if available
        if self.redis:
            try:
                self.redis.hset(
                    f"model:deployment:{model_name}:{environment}",
                    mapping={
                        "version": str(version),
                        "timestamp": str(time.time()),
                        "status": "active"
                    }
                )
            except Exception as e:
                logging.exception(f"Error updating Redis deployment status: {e}")

        logging.info(f"Deployed {model_name} v{version} to {environment}")

    def get_deployed_version(self, model_name: str, environment: str = "production") -> Optional[int]:
        """
        Get the currently deployed version for a model in an environment.

        Args:
            model_name: Name of the model
            environment: Target environment

        Returns:
            Version number if deployed, None otherwise
        """
        # Check Redis first if available
        if self.redis:
            try:
                version = self.redis.hget(
                    f"model:deployment:{model_name}:{environment}",
                    "version"
                )
                if version:
                    return int(version)
            except Exception as e:
                logging.exception(f"Error checking Redis for deployed version: {e}")

        # Fall back to local tracking
        if model_name in self.deployments and environment in self.deployments[model_name]:
            return self.deployments[model_name][environment]["version"]

        return None

    def create_ab_test(self, model_name: str, version_a: int, version_b: int, split: float = 0.5):
        """
        Set up an A/B test between two model versions.

        Args:
            model_name: Name of the model
            version_a: First version to test
            version_b: Second version to test
            split: Traffic split ratio (0-1)
        """
        # Validate versions exist
        if str(version_a) not in self.registry.models_metadata[model_name]["versions"]:
            raise ValueError(f"Version {version_a} not found for model {model_name}")
        if str(version_b) not in self.registry.models_metadata[model_name]["versions"]:
            raise ValueError(f"Version {version_b} not found for model {model_name}")

        # Store A/B test configuration
        ab_test_key = f"{model_name}_ab_test"
        self.deployments[ab_test_key] = {
            "version_a": version_a,
            "version_b": version_b,
            "split": split,
            "start_time": time.time()
        }

        # Update Redis if available
        if self.redis:
            try:
                self.redis.hset(
                    f"model:ab_test:{model_name}",
                    mapping={
                        "version_a": str(version_a),
                        "version_b": str(version_b),
                        "split": str(split),
                        "start_time": str(time.time())
                    }
                )
            except Exception as e:
                logging.exception(f"Error updating Redis A/B test status: {e}")

        logging.info(f"Created A/B test for {model_name}: v{version_a} vs v{version_b}")

    def end_ab_test(self, model_name: str, winning_version: Optional[int] = None):
        """
        End an A/B test and optionally promote the winning version.

        Args:
            model_name: Name of the model
            winning_version: Version to promote (None to keep current)
        """
        ab_test_key = f"{model_name}_ab_test"
        if ab_test_key not in self.deployments:
            raise ValueError(f"No active A/B test for {model_name}")

        # Promote winning version if specified
        if winning_version is not None:
            self.deploy_model(model_name, winning_version)

        # Clean up A/B test
        del self.deployments[ab_test_key]

        # Clean up Redis if available
        if self.redis:
            try:
                self.redis.delete(f"model:ab_test:{model_name}")
            except Exception as e:
                logging.exception(f"Error cleaning up Redis A/B test: {e}")

        logging.info(f"Ended A/B test for {model_name}")
