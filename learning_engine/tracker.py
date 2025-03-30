"""Model performance tracking and monitoring."""

import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from .drift_detector import DriftDetector

class ModelPerformanceTracker:
    """
    Tracks and analyzes model performance over time.

    Features:
    - Performance metric storage
    - Model comparison
    - Performance degradation alerts
    - Historical performance analysis
    - Redis integration
    """

    def __init__(self, redis_client=None, alert_thresholds: Dict[str, float] = None):
        """
        Initialize the performance tracker.

        Args:
            redis_client: Optional Redis client
            alert_thresholds: Dictionary of metric thresholds for alerts
        """
        self.redis = redis_client
        self.metrics = {}
        self.drift_detector = DriftDetector()
        self.alert_thresholds = alert_thresholds or {
            'accuracy': 0.1,  # 10% drop
            'f1_score': 0.15,  # 15% drop
            'psi': 0.25  # Significant drift
        }
        self.alert_handlers = []

    def add_alert_handler(self, handler):
        """Add an alert handler callback function."""
        self.alert_handlers.append(handler)

    def _trigger_alert(self, model_name: str, version: int, message: str):
        """Trigger alerts to all registered handlers."""
        for handler in self.alert_handlers:
            try:
                handler(model_name, version, message)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")

    def track_metrics(
        self,
        model_name: str,
        version: int,
        metrics: Dict[str, Any],
        features: Optional[Dict[str, np.ndarray]] = None,
        timestamp: Optional[float] = None
    ):
        """
        Track performance metrics for a model version.

        Args:
            model_name: Name of the model
            version: Version number
            metrics: Dictionary of metrics
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        # Store metrics locally
        key = f"{model_name}_v{version}"
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append({
            "timestamp": timestamp,
            "metrics": metrics
        })

        # Check for performance alerts
        self._check_for_alerts(model_name, version, metrics)

        # Check for feature drift if features provided
        if features:
            self._check_feature_drift(model_name, version, features)

        # Store in Redis if available
        if self.redis:
            try:
                # Store latest metrics
                self.redis.hset(
                    f"model:metrics:{model_name}:{version}:latest",
                    mapping=metrics
                )

                # Add to historical metrics
                self.redis.zadd(
                    f"model:metrics:{model_name}:{version}:history",
                    {json.dumps(metrics): timestamp}
                )
            except Exception as e:
                logging.exception(f"Error storing metrics in Redis: {e}")

        logging.info(f"Tracked metrics for {model_name} v{version}")

    def get_latest_metrics(
        self,
        model_name: str,
        version: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest performance metrics for a model version.

        Args:
            model_name: Name of the model
            version: Version number

        Returns:
            Dictionary of metrics if available, None otherwise
        """
        # Try Redis first if available
        if self.redis:
            try:
                metrics = self.redis.hgetall(
                    f"model:metrics:{model_name}:{version}:latest"
                )
                if metrics:
                    return {k: float(v) for k, v in metrics.items()}
            except Exception as e:
                logging.exception(f"Error getting metrics from Redis: {e}")

        # Fall back to local storage
        key = f"{model_name}_v{version}"
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]["metrics"]

        return None

    def get_historical_metrics(
        self,
        model_name: str,
        version: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> list:
        """
        Get historical performance metrics for a model version.

        Args:
            model_name: Name of the model
            version: Version number
            start_time: Optional start timestamp
            end_time: Optional end timestamp

        Returns:
            List of metric dictionaries with timestamps
        """
        results = []

        # Try Redis first if available
        if self.redis:
            try:
                # Get range from Redis
                redis_key = f"model:metrics:{model_name}:{version}:history"
                if start_time is None and end_time is None:
                    items = self.redis.zrange(redis_key, 0, -1, withscores=True)
                else:
                    items = self.redis.zrangebyscore(
                        redis_key,
                        start_time or "-inf",
                        end_time or "+inf",
                        withscores=True
                    )

                # Parse results
                for item, timestamp in items:
                    metrics = json.loads(item)
                    results.append({
                        "timestamp": timestamp,
                        "metrics": metrics
                    })
            except Exception as e:
                logging.exception(f"Error getting historical metrics from Redis: {e}")

        # Fall back to local storage
        key = f"{model_name}_v{version}"
        if key in self.metrics:
            for entry in self.metrics[key]:
                if ((start_time is None or entry["timestamp"] >= start_time) and
                    (end_time is None or entry["timestamp"] <= end_time)):
                    results.append(entry)

        return sorted(results, key=lambda x: x["timestamp"])

    def _check_feature_drift(self, model_name: str, version: int, features: Dict[str, np.ndarray]):
        """Check for feature drift and trigger alerts."""
        for feature_name, values in features.items():
            try:
                psi = self.drift_detector.calculate_psi(feature_name, values)
                if psi > self.alert_thresholds['psi']:
                    msg = (f"Feature drift detected for {model_name} v{version} - "
                          f"{feature_name} PSI: {psi:.3f}")
                    self._trigger_alert(model_name, version, msg)
            except Exception as e:
                logging.error(f"Failed to check feature drift: {e}")

    def _check_for_alerts(self, model_name: str, version: int, metrics: Dict[str, float]):
        """Check metrics against alert thresholds."""
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics:
                if metrics[metric] < threshold:
                    msg = (f"Performance alert for {model_name} v{version} - "
                          f"{metric} {metrics[metric]:.3f} < threshold {threshold}")
                    self._trigger_alert(model_name, version, msg)

    def check_performance_degradation(
        self,
        model_name: str,
        version: int,
        metric_name: str,
        threshold: Optional[float] = None,
        window_hours: int = 24
    ) -> bool:
        """
        Check if a metric has degraded beyond a threshold.

        Args:
            model_name: Name of the model
            version: Version number
            metric_name: Name of metric to check
            threshold: Degradation threshold
            window_hours: Time window to analyze (hours)

        Returns:
            True if degradation detected, False otherwise
        """
        now = time.time()
        start_time = now - (window_hours * 3600)

        # Get historical metrics
        history = self.get_historical_metrics(
            model_name, version, start_time, now
        )

        if len(history) < 2:
            return False

        # Calculate average over window
        values = [entry["metrics"].get(metric_name, 0) for entry in history]
        avg_value = sum(values) / len(values)

        # Compare to threshold
        if avg_value < threshold:
            logging.warning(
                f"Performance degradation detected for {model_name} v{version}: "
                f"{metric_name} average {avg_value:.3f} < threshold {threshold}"
            )
            return True

        return False
