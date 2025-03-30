#!/usr/bin/env python3
"""
Metrics Registry Module

This module provides a standardized metrics collection system for the trading system:

1. Unified Prometheus metric naming conventions
2. Centralized metric registry to prevent duplication
3. Standard label sets for consistent categorization
4. Common histogram buckets for latency and size measurements
5. Utility functions for metric recording and monitoring

All components in the trading system should use this module for metrics collection
to ensure consistent naming, labeling, and monitoring.
"""

import logging
import os
import time
import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger("metrics_registry")

# Import Prometheus client with fallback
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    from prometheus_client import CollectorRegistry, push_to_gateway
    
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client available for metrics collection")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics collection will be disabled.")
    
    # Create dummy metric classes for graceful fallback
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        
        def labels(self, **kwargs):
            return self
        
        def inc(self, amount=1):
            pass
        
        def dec(self, amount=1):
            pass
        
        def set(self, value):
            pass
        
        def observe(self, value):
            pass
    
    Counter = DummyMetric
    Gauge = DummyMetric
    Histogram = DummyMetric
    Summary = DummyMetric
    CollectorRegistry = object
    
    def push_to_gateway(*args, **kwargs):
        pass


# Environment variables for metrics configuration
METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "true").lower() == "true"
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9090"))
METRICS_PUSH_GATEWAY = os.environ.get("METRICS_PUSH_GATEWAY", "")
METRICS_PUSH_INTERVAL = int(os.environ.get("METRICS_PUSH_INTERVAL", "15"))
METRICS_JOB_NAME = os.environ.get("METRICS_JOB_NAME", "trading_system")


# Standard metric name prefixes
class MetricPrefix(Enum):
    API = "api"
    CACHE = "cache"
    DATA = "data"
    GPU = "gpu"
    MODEL = "model"
    SYSTEM = "system"
    TRADING = "trading"
    WEBSOCKET = "websocket"


# Standard label names for consistent categorization
class MetricLabel(Enum):
    CLIENT = "client"
    ENDPOINT = "endpoint"
    METHOD = "method"
    STATUS = "status"
    ERROR_TYPE = "error_type"
    OPERATION = "operation"
    CACHE_TYPE = "cache_type"
    DEVICE = "device"
    MODEL_TYPE = "model_type"
    FRAMEWORK = "framework"
    PRECISION = "precision"
    TICKER = "ticker"
    TIMEFRAME = "timeframe"
    STRATEGY = "strategy"
    COMPONENT = "component"


# Standard histogram buckets for different metric types
LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0
)

SIZE_BUCKETS = (
    1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024 * 1024
)

COUNT_BUCKETS = (
    1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000
)

MEMORY_BUCKETS = (
    1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024, 
    1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024,
    8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024
)


# Registry to store all metrics to prevent duplication
_metrics_registry: Dict[str, Any] = {}


def get_metric_name(prefix: Union[str, MetricPrefix], name: str) -> str:
    """
    Generate a standardized metric name with prefix
    
    Args:
        prefix: Metric prefix (e.g., "api", "cache", "data")
        name: Metric name
        
    Returns:
        Standardized metric name
    """
    if isinstance(prefix, MetricPrefix):
        prefix = prefix.value
    
    return f"{prefix}_{name}"


def register_counter(
    prefix: Union[str, MetricPrefix],
    name: str,
    description: str,
    labels: Optional[List[Union[str, MetricLabel]]] = None,
    registry: Optional[CollectorRegistry] = None
) -> Counter:
    """
    Register a Counter metric with standardized naming
    
    Args:
        prefix: Metric prefix (e.g., "api", "cache", "data")
        name: Metric name
        description: Metric description
        labels: List of label names
        registry: Optional custom registry
        
    Returns:
        Prometheus Counter metric
    """
    if not PROMETHEUS_AVAILABLE or not METRICS_ENABLED:
        return Counter()
    
    metric_name = get_metric_name(prefix, name)
    
    # Return existing metric if already registered
    if metric_name in _metrics_registry:
        return _metrics_registry[metric_name]
    
    # Convert MetricLabel enums to strings if needed
    if labels:
        labels = [label.value if isinstance(label, MetricLabel) else label for label in labels]
    
    # Create and register metric
    counter = Counter(
        metric_name,
        description,
        labels or [],
        registry=registry
    )
    
    _metrics_registry[metric_name] = counter
    return counter


def register_gauge(
    prefix: Union[str, MetricPrefix],
    name: str,
    description: str,
    labels: Optional[List[Union[str, MetricLabel]]] = None,
    registry: Optional[CollectorRegistry] = None
) -> Gauge:
    """
    Register a Gauge metric with standardized naming
    
    Args:
        prefix: Metric prefix (e.g., "api", "cache", "data")
        name: Metric name
        description: Metric description
        labels: List of label names
        registry: Optional custom registry
        
    Returns:
        Prometheus Gauge metric
    """
    if not PROMETHEUS_AVAILABLE or not METRICS_ENABLED:
        return Gauge()
    
    metric_name = get_metric_name(prefix, name)
    
    # Return existing metric if already registered
    if metric_name in _metrics_registry:
        return _metrics_registry[metric_name]
    
    # Convert MetricLabel enums to strings if needed
    if labels:
        labels = [label.value if isinstance(label, MetricLabel) else label for label in labels]
    
    # Create and register metric
    gauge = Gauge(
        metric_name,
        description,
        labels or [],
        registry=registry
    )
    
    _metrics_registry[metric_name] = gauge
    return gauge


def register_histogram(
    prefix: Union[str, MetricPrefix],
    name: str,
    description: str,
    labels: Optional[List[Union[str, MetricLabel]]] = None,
    buckets: Optional[Tuple[float, ...]] = None,
    registry: Optional[CollectorRegistry] = None
) -> Histogram:
    """
    Register a Histogram metric with standardized naming and buckets
    
    Args:
        prefix: Metric prefix (e.g., "api", "cache", "data")
        name: Metric name
        description: Metric description
        labels: List of label names
        buckets: Custom histogram buckets
        registry: Optional custom registry
        
    Returns:
        Prometheus Histogram metric
    """
    if not PROMETHEUS_AVAILABLE or not METRICS_ENABLED:
        return Histogram()
    
    metric_name = get_metric_name(prefix, name)
    
    # Return existing metric if already registered
    if metric_name in _metrics_registry:
        return _metrics_registry[metric_name]
    
    # Convert MetricLabel enums to strings if needed
    if labels:
        labels = [label.value if isinstance(label, MetricLabel) else label for label in labels]
    
    # Use appropriate default buckets based on metric name
    if buckets is None:
        if "latency" in name or "time" in name or "duration" in name:
            buckets = LATENCY_BUCKETS
        elif "size" in name or "bytes" in name:
            buckets = SIZE_BUCKETS
        elif "count" in name:
            buckets = COUNT_BUCKETS
        elif "memory" in name:
            buckets = MEMORY_BUCKETS
    
    # Create and register metric
    histogram = Histogram(
        metric_name,
        description,
        labels or [],
        buckets=buckets,
        registry=registry
    )
    
    _metrics_registry[metric_name] = histogram
    return histogram


def register_summary(
    prefix: Union[str, MetricPrefix],
    name: str,
    description: str,
    labels: Optional[List[Union[str, MetricLabel]]] = None,
    registry: Optional[CollectorRegistry] = None
) -> Summary:
    """
    Register a Summary metric with standardized naming
    
    Args:
        prefix: Metric prefix (e.g., "api", "cache", "data")
        name: Metric name
        description: Metric description
        labels: List of label names
        registry: Optional custom registry
        
    Returns:
        Prometheus Summary metric
    """
    if not PROMETHEUS_AVAILABLE or not METRICS_ENABLED:
        return Summary()
    
    metric_name = get_metric_name(prefix, name)
    
    # Return existing metric if already registered
    if metric_name in _metrics_registry:
        return _metrics_registry[metric_name]
    
    # Convert MetricLabel enums to strings if needed
    if labels:
        labels = [label.value if isinstance(label, MetricLabel) else label for label in labels]
    
    # Create and register metric
    summary = Summary(
        metric_name,
        description,
        labels or [],
        registry=registry
    )
    
    _metrics_registry[metric_name] = summary
    return summary


def time_function(
    metric: Optional[Histogram] = None,
    labels: Optional[Dict[str, str]] = None
) -> Callable:
    """
    Decorator to measure function execution time
    
    Args:
        metric: Histogram metric to record time
        labels: Labels to apply to the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if PROMETHEUS_AVAILABLE and METRICS_ENABLED and metric is not None:
                    if labels:
                        metric.labels(**labels).observe(duration)
                    else:
                        metric.observe(duration)
        return wrapper
    return decorator


def count_calls(
    metric: Optional[Counter] = None,
    labels: Optional[Dict[str, str]] = None,
    count_exceptions: bool = True
) -> Callable:
    """
    Decorator to count function calls
    
    Args:
        metric: Counter metric to increment
        labels: Labels to apply to the metric
        count_exceptions: Whether to count calls that raise exceptions
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if PROMETHEUS_AVAILABLE and METRICS_ENABLED and metric is not None:
                    if labels:
                        metric.labels(**labels).inc()
                    else:
                        metric.inc()
                return result
            except Exception as e:
                if count_exceptions and PROMETHEUS_AVAILABLE and METRICS_ENABLED and metric is not None:
                    if labels:
                        metric.labels(**labels).inc()
                    else:
                        metric.inc()
                raise e
        return wrapper
    return decorator


def track_in_progress(
    metric: Optional[Gauge] = None,
    labels: Optional[Dict[str, str]] = None
) -> Callable:
    """
    Decorator to track in-progress operations
    
    Args:
        metric: Gauge metric to track in-progress operations
        labels: Labels to apply to the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if PROMETHEUS_AVAILABLE and METRICS_ENABLED and metric is not None:
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
            try:
                return func(*args, **kwargs)
            finally:
                if PROMETHEUS_AVAILABLE and METRICS_ENABLED and metric is not None:
                    if labels:
                        metric.labels(**labels).dec()
                    else:
                        metric.dec()
        return wrapper
    return decorator


def push_metrics_to_gateway(
    registry: Optional[CollectorRegistry] = None,
    gateway: Optional[str] = None,
    job: Optional[str] = None,
    grouping_key: Optional[Dict[str, str]] = None
) -> bool:
    """
    Push metrics to Prometheus Pushgateway
    
    Args:
        registry: Registry to push
        gateway: Pushgateway URL
        job: Job name
        grouping_key: Grouping key for metrics
        
    Returns:
        bool: True if metrics were successfully pushed, False otherwise
    """
    if not PROMETHEUS_AVAILABLE or not METRICS_ENABLED:
        return False
    
    try:
        gateway = gateway or METRICS_PUSH_GATEWAY
        if not gateway:
            logger.warning("No Pushgateway URL specified")
            return False
        
        job = job or METRICS_JOB_NAME
        registry = registry or prometheus_client.REGISTRY
        
        push_to_gateway(gateway, job, registry, grouping_key)
        logger.debug(f"Pushed metrics to gateway: {gateway}")
        return True
    
    except Exception as e:
        logger.error(f"Error pushing metrics to gateway: {e}")
        return False


# Define standard metrics for common use cases

# API metrics
API_REQUEST_COUNT = register_counter(
    MetricPrefix.API,
    "request_count",
    "Number of API requests made",
    [MetricLabel.CLIENT, MetricLabel.ENDPOINT, MetricLabel.METHOD]
)

API_REQUEST_LATENCY = register_histogram(
    MetricPrefix.API,
    "request_latency_seconds",
    "API request latency in seconds",
    [MetricLabel.CLIENT, MetricLabel.ENDPOINT, MetricLabel.METHOD]
)

API_ERROR_COUNT = register_counter(
    MetricPrefix.API,
    "error_count",
    "Number of API errors",
    [MetricLabel.CLIENT, MetricLabel.ENDPOINT, MetricLabel.METHOD, MetricLabel.ERROR_TYPE]
)

API_RATE_LIMIT_REMAINING = register_gauge(
    MetricPrefix.API,
    "rate_limit_remaining",
    "Remaining API rate limit",
    [MetricLabel.CLIENT, MetricLabel.ENDPOINT]
)

# Cache metrics
CACHE_HIT_COUNT = register_counter(
    MetricPrefix.CACHE,
    "hit_count",
    "Number of cache hits",
    [MetricLabel.CLIENT, MetricLabel.CACHE_TYPE]
)

CACHE_MISS_COUNT = register_counter(
    MetricPrefix.CACHE,
    "miss_count",
    "Number of cache misses",
    [MetricLabel.CLIENT, MetricLabel.CACHE_TYPE]
)

CACHE_SIZE = register_gauge(
    MetricPrefix.CACHE,
    "size",
    "Number of items in cache",
    [MetricLabel.CLIENT, MetricLabel.CACHE_TYPE]
)

CACHE_OPERATION_LATENCY = register_histogram(
    MetricPrefix.CACHE,
    "operation_latency_seconds",
    "Cache operation latency in seconds",
    [MetricLabel.OPERATION]
)

# GPU metrics
GPU_MEMORY_USAGE = register_gauge(
    MetricPrefix.GPU,
    "memory_usage_bytes",
    "GPU memory usage in bytes",
    [MetricLabel.DEVICE]
)

GPU_PROCESSING_TIME = register_histogram(
    MetricPrefix.GPU,
    "processing_time_seconds",
    "GPU processing time in seconds",
    [MetricLabel.OPERATION]
)

# Data processing metrics
DATA_PROCESSING_TIME = register_histogram(
    MetricPrefix.DATA,
    "processing_time_seconds",
    "Time spent processing data",
    [MetricLabel.OPERATION]
)

DATA_ROWS_PROCESSED = register_counter(
    MetricPrefix.DATA,
    "rows_processed",
    "Number of data rows processed",
    [MetricLabel.OPERATION]
)

# Model metrics
MODEL_INFERENCE_TIME = register_histogram(
    MetricPrefix.MODEL,
    "inference_time_seconds",
    "Time spent on model inference",
    [MetricLabel.MODEL_TYPE, MetricLabel.FRAMEWORK, MetricLabel.PRECISION]
)

PREDICTION_LATENCY = register_histogram(
    MetricPrefix.MODEL,
    "prediction_latency_seconds",
    "Time spent on model prediction including data preparation",
    ["model_name"]
)

PREDICTION_THROUGHPUT = register_counter(
    MetricPrefix.MODEL,
    "prediction_throughput",
    "Number of predictions processed per batch",
    ["model_name", "batch_size"]
)

BATCH_PROCESSING_TIME = register_histogram(
    MetricPrefix.MODEL,
    "batch_processing_time",
    "Time taken to process a batch of predictions",
    ["model_name", "batch_size"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
)

PREDICTION_CONFIDENCE = register_histogram(
    MetricPrefix.MODEL,
    "prediction_confidence",
    "Confidence scores for model predictions",
    ["model_name", "prediction_type"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
MODEL_TRAINING_TIME = register_histogram(
    MetricPrefix.MODEL,
    "training_time_seconds",
    "Time spent on model training",
    [MetricLabel.MODEL_TYPE, MetricLabel.FRAMEWORK]
)

MODEL_ACCURACY = register_gauge(
    MetricPrefix.MODEL,
    "accuracy",
    "Model accuracy",
    [MetricLabel.MODEL_TYPE]
)

MODEL_OPTIMIZATION_COUNT = register_counter(
    MetricPrefix.MODEL,
    "optimization_count",
    "Number of model optimizations performed",
    [MetricLabel.MODEL_TYPE, MetricLabel.FRAMEWORK]
)

# Feature drift metrics
FEATURE_DRIFT = register_gauge(
    MetricPrefix.MODEL,
    "feature_drift",
    "Feature drift over time",
    ["model_name", "feature_name", "timestamp"]
)

# Cache metrics
CACHE_HIT_COUNT = register_counter(
    MetricPrefix.MODEL,
    "cache_hit_count",
    "Number of cache hits",
    ["client", "cache_type"]
)

CACHE_MISS_COUNT = register_counter(
    MetricPrefix.MODEL,
    "cache_miss_count",
    "Number of cache misses",
    ["client", "cache_type"]
)

CACHE_SIZE = register_gauge(
    MetricPrefix.MODEL,
    "cache_size",
    "Current size of the cache",
    ["client", "cache_type"]
)

# Model version metrics
MODEL_VERSION_METRICS = register_gauge(
    MetricPrefix.MODEL,
    "version_metrics",
    "Performance metrics for different model versions",
    ["model_name", "version", "metric_name"]
)
# WebSocket metrics
WEBSOCKET_MESSAGES = register_counter(
    MetricPrefix.WEBSOCKET,
    "messages",
    "Number of WebSocket messages",
    [MetricLabel.CLIENT, "message_type"]
)

WEBSOCKET_RECONNECTS = register_counter(
    MetricPrefix.WEBSOCKET,
    "reconnects",
    "Number of WebSocket reconnections",
    [MetricLabel.CLIENT, MetricLabel.ENDPOINT]
)

WEBSOCKET_ERRORS = register_counter(
    MetricPrefix.WEBSOCKET,
    "errors",
    "Number of WebSocket errors",
    [MetricLabel.CLIENT, MetricLabel.ENDPOINT, MetricLabel.ERROR_TYPE]
)

WEBSOCKET_LATENCY = register_histogram(
    MetricPrefix.WEBSOCKET,
    "latency_seconds",
    "WebSocket message latency in seconds",
    [MetricLabel.CLIENT, "message_type"]
)

# System metrics
SYSTEM_MEMORY_USAGE = register_gauge(
    MetricPrefix.SYSTEM,
    "memory_usage_bytes",
    "System memory usage in bytes",
    ["type"]
)

SYSTEM_CPU_USAGE = register_gauge(
    MetricPrefix.SYSTEM,
    "cpu_usage_percent",
    "System CPU usage percentage",
    ["cpu"]
)

SYSTEM_DISK_USAGE = register_gauge(
    MetricPrefix.SYSTEM,
    "disk_usage_bytes",
    "System disk usage in bytes",
    ["mount_point", "type"]
)

# Trading metrics
TRADING_ORDER_COUNT = register_counter(
    MetricPrefix.TRADING,
    "order_count",
    "Number of trading orders",
    [MetricLabel.TICKER, "order_type", "direction"]
)

TRADING_POSITION_SIZE = register_gauge(
    MetricPrefix.TRADING,
    "position_size",
    "Trading position size",
    [MetricLabel.TICKER, "direction"]
)

TRADING_PNL = register_gauge(
    MetricPrefix.TRADING,
    "pnl",
    "Trading profit and loss",
    [MetricLabel.TICKER, MetricLabel.STRATEGY, "timeframe"]
)

# Stock selection metrics
UNIVERSE_SIZE = register_gauge(
    MetricPrefix.TRADING,
    "universe_size",
    "Number of stocks in the universe",
    [MetricLabel.STRATEGY]
)

WATCHLIST_SIZE = register_gauge(
    MetricPrefix.TRADING,
    "watchlist_size",
    "Number of stocks in the watchlist",
    [MetricLabel.STRATEGY]
)

FOCUSED_LIST_SIZE = register_gauge(
    MetricPrefix.TRADING,
    "focused_list_size",
    "Number of stocks in the focused list",
    [MetricLabel.STRATEGY]
)

STOCK_SCORES = register_gauge(
    MetricPrefix.TRADING,
    "stock_scores",
    "Stock selection scores",
    [MetricLabel.TICKER, MetricLabel.STRATEGY]
)

# Trading execution metrics
TRADES_EXECUTED = register_counter(
    MetricPrefix.TRADING,
    "trades_executed",
    "Number of trades executed",
    [MetricLabel.TICKER, "order_type", "direction", MetricLabel.STRATEGY]
)

TRADE_PNL = register_gauge(
    MetricPrefix.TRADING,
    "trade_pnl",
    "Profit and loss per trade",
    [MetricLabel.TICKER, MetricLabel.STRATEGY]
)

TRADE_LATENCY = register_histogram(
    MetricPrefix.TRADING,
    "trade_latency_seconds",
    "Trade execution latency in seconds",
    [MetricLabel.TICKER, "order_type", MetricLabel.STRATEGY]
)

# Market data metrics
MARKET_DATA_LATENCY = register_histogram(
    MetricPrefix.DATA,
    "market_data_latency_seconds",
    "Market data retrieval latency in seconds",
    [MetricLabel.ENDPOINT, MetricLabel.OPERATION]
)

MARKET_DATA_ERRORS = register_counter(
    MetricPrefix.DATA,
    "market_data_errors",
    "Number of market data retrieval errors",
    [MetricLabel.ENDPOINT, MetricLabel.ERROR_TYPE]
)

# Drift detection metrics
DRIFT_DETECTION = register_counter(
    MetricPrefix.MODEL,
    "drift_detection",
    "Number of drift detection checks",
    ["model_name", "result"]
)
