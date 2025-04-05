#!/usr/bin/env python3
"""
Resource Manager Module

This module provides utilities for managing system resources:
1. Memory tracking and optimization
2. GPU resource management and cleanup
3. Resource usage monitoring
4. Circuit breaker pattern for resource protection

The module ensures efficient resource utilization and prevents memory leaks.
"""

import gc
import time
import logging
import threading
import weakref
from functools import wraps # Added missing import
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager # Added missing import

# Configure logging
logger = logging.getLogger("resource_manager")

# Try to import GPU-related libraries
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.info("PyTorch not available, GPU resource management will be limited")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.info("psutil not available, memory tracking will be limited")

# enhanced_metrics.py was removed, setting METRICS_AVAILABLE based on its previous state
METRICS_AVAILABLE = False
logger.info(
    "Enhanced metrics (track_memory_usage, record_resource_error) not available, resource tracking will be limited")


class ResourceTracker:
    """
    Tracks resource usage and provides alerts when thresholds are exceeded
    """

    def __init__(
        self,
        memory_threshold_percent: float = 80.0,
        gpu_memory_threshold_percent: float = 80.0,
        check_interval_seconds: float = 5.0,
        enable_circuit_breaker: bool = True
    ):
        """
        Initialize resource tracker

        Args:
            memory_threshold_percent: Memory usage threshold percentage
            gpu_memory_threshold_percent: GPU memory usage threshold percentage
            check_interval_seconds: Interval between resource checks
            enable_circuit_breaker: Whether to enable circuit breaker
        """
        self.memory_threshold = memory_threshold_percent
        self.gpu_memory_threshold = gpu_memory_threshold_percent
        self.check_interval = check_interval_seconds
        self.enable_circuit_breaker = enable_circuit_breaker

        # Resource tracking
        self.memory_usage_history = []
        self.gpu_memory_usage_history = []
        self.cpu_usage_history = []

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_time = None
        self.circuit_reset_timeout = 60.0  # seconds

        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None

        # Callbacks
        self.threshold_callbacks = []

    def start_monitoring(self) -> bool:
        """
        Start resource monitoring in a background thread

        Returns:
            True if monitoring started, False otherwise
        """
        if self.monitoring_active:
            logger.warning("Resource monitoring already active")
            return False

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")
        return True

    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None
        logger.info("Resource monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self.check_resources()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.check_interval * 2)  # Back off on error

    def check_resources(self) -> Dict[str, Any]:
        """
        Check current resource usage

        Returns:
            Dictionary with resource usage information
        """
        result = {
            "timestamp": datetime.now(),
            "memory": {},
            "gpu": {},
            "cpu": {}
        }

        # Check memory usage
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                result["memory"] = {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "percent": memory.percent
                }

                # Add to history (keep last 100 entries)
                self.memory_usage_history.append(result["memory"])
                if len(self.memory_usage_history) > 100:
                    self.memory_usage_history.pop(0)

                # Check threshold
                if memory.percent > self.memory_threshold:
                    self._handle_threshold_exceeded("memory", memory.percent)
            except Exception as e:
                logger.error(f"Error checking memory usage: {e}")

        # Check GPU memory usage
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                gpu_info = {}
                for i in range(torch.cuda.device_count()):
                    total_memory = torch.cuda.get_device_properties(
                        i).total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    reserved_memory = torch.cuda.memory_reserved(i)
                    free_memory = total_memory - reserved_memory

                    percent_allocated = (allocated_memory / total_memory) * 100 if total_memory > 0 else 0
                    percent_reserved = (reserved_memory / total_memory) * 100 if total_memory > 0 else 0


                    gpu_info[f"device_{i}"] = {
                        "total_bytes": total_memory,
                        "allocated_bytes": allocated_memory,
                        "reserved_bytes": reserved_memory,
                        "free_bytes": free_memory,
                        "percent_allocated": percent_allocated,
                        "percent_reserved": percent_reserved
                    }

                    # Check threshold
                    if percent_reserved > self.gpu_memory_threshold:
                        self._handle_threshold_exceeded(
                            "gpu", percent_reserved, device_id=i)

                result["gpu"] = gpu_info

                # Add to history (keep last 100 entries)
                self.gpu_memory_usage_history.append(result["gpu"])
                if len(self.gpu_memory_usage_history) > 100:
                    self.gpu_memory_usage_history.pop(0)
            except Exception as e:
                logger.error(f"Error checking GPU memory usage: {e}")

        # Check CPU usage
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                result["cpu"] = {
                    "percent": cpu_percent
                }

                # Add to history (keep last 100 entries)
                self.cpu_usage_history.append(result["cpu"])
                if len(self.cpu_usage_history) > 100:
                    self.cpu_usage_history.pop(0)
            except Exception as e:
                logger.error(f"Error checking CPU usage: {e}")

        return result

    def _handle_threshold_exceeded(
        self,
        resource_type: str,
        current_value: float,
        device_id: Optional[int] = None
    ) -> None:
        """
        Handle threshold exceeded event

        Args:
            resource_type: Type of resource (memory, gpu, cpu)
            current_value: Current usage value
            device_id: Device ID for GPU resources
        """
        device_str = f" (device {device_id})" if device_id is not None else ""
        logger.warning(
            f"{resource_type.upper()}{device_str} usage exceeded threshold: {current_value:.1f}%")

        # Record metric if available (placeholder, as enhanced_metrics removed)
        # if METRICS_AVAILABLE:
        #     record_resource_error("resource_tracker", resource_type)

        # Trigger callbacks
        for callback in self.threshold_callbacks:
            try:
                callback(resource_type, current_value, device_id)
            except Exception as e:
                logger.error(f"Error in threshold callback: {e}")

        # Check circuit breaker
        if self.enable_circuit_breaker and not self.circuit_open:
            self._open_circuit()

    def _open_circuit(self) -> None:
        """Open the circuit breaker"""
        self.circuit_open = True
        self.circuit_open_time = datetime.now()
        logger.warning(
            "Circuit breaker opened due to resource threshold exceeded")

        # Perform emergency cleanup
        self.cleanup_resources()

    def _check_circuit_reset(self) -> None:
        """Check if circuit breaker should be reset"""
        if not self.circuit_open:
            return

        # Check if timeout has elapsed
        if datetime.now() - self.circuit_open_time > timedelta(seconds=self.circuit_reset_timeout):
            self.circuit_open = False
            logger.info("Circuit breaker reset")

    def can_allocate_resources(self) -> bool:
        """
        Check if resources can be allocated

        Returns:
            True if resources can be allocated, False if circuit breaker is open
        """
        self._check_circuit_reset()
        return not self.circuit_open

    def register_threshold_callback(self, callback: Callable[[str, float, Optional[int]], None]) -> None:
        """
        Register a callback for threshold exceeded events

        Args:
            callback: Function to call when threshold is exceeded
        """
        self.threshold_callbacks.append(callback)

    def cleanup_resources(self) -> None:
        """Perform resource cleanup"""
        # Force garbage collection
        gc.collect()

        # Clear GPU memory if available
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared GPU memory cache")
            except Exception as e:
                logger.error(f"Error clearing GPU memory: {e}")


class ResourceManager:
    """
    Manages resources and ensures proper cleanup
    """

    def __init__(self):
        """Initialize resource manager"""
        self.active_resources = weakref.WeakSet()
        self.resource_tracker = ResourceTracker()
        self.resource_tracker.start_monitoring()

        # Register cleanup callback
        self.resource_tracker.register_threshold_callback(
            self._threshold_callback)

    def register_resource(self, resource: Any) -> None:
        """
        Register a resource for tracking

        Args:
            resource: Resource to track
        """
        self.active_resources.add(resource)

    def cleanup_resources(self) -> None:
        """Clean up all tracked resources"""
        # Clear active resources
        self.active_resources.clear()

        # Force garbage collection
        gc.collect()

        # Clear GPU memory if available
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared GPU memory")
            except Exception as e:
                logger.error(f"Error clearing GPU memory: {e}")

    def _threshold_callback(
        self,
        resource_type: str,
        current_value: float,
        device_id: Optional[int] = None
    ) -> None:
        """
        Callback for threshold exceeded events

        Args:
            resource_type: Type of resource
            current_value: Current usage value
            device_id: Device ID for GPU resources
        """
        logger.warning(
            f"Resource threshold exceeded: {resource_type} = {current_value:.1f}%")

        # Perform cleanup if memory or GPU threshold exceeded
        if resource_type in ["memory", "gpu"]:
            self.cleanup_resources()


class ResourceContext:
    """
    Context manager for resource tracking and cleanup
    """

    def __init__(self, component: str, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize resource context

        Args:
            component: Component name
            resource_manager: Resource manager instance
        """
        self.component = component
        self.resource_manager = resource_manager or _global_resource_manager
        self.start_time = None
        self.start_memory = None # Removed tracking via enhanced_metrics

    def __enter__(self):
        """Start resource context"""
        self.start_time = time.time()
        # Removed memory tracking call
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End resource context and cleanup"""
        elapsed_time = time.time() - self.start_time
        logger.info(f"Resource usage for {self.component}: Time: {elapsed_time:.2f}s") # Simplified log

        # Cleanup on exception
        if exc_type is not None:
            logger.warning(
                f"Exception in {self.component}: {exc_type.__name__}: {exc_val}")
            self.resource_manager.cleanup_resources()

        return False  # Don't suppress exceptions


def resource_managed(component: str):
    """
    Decorator for resource-managed functions

    Args:
        component: Component name

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func) # Use wraps to preserve function metadata
        def wrapper(*args, **kwargs):
            with ResourceContext(f"{component}.{func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global resource manager instance
_global_resource_manager = ResourceManager()

# Exported functions


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    return _global_resource_manager


def cleanup_all_resources() -> None:
    """Clean up all tracked resources"""
    _global_resource_manager.cleanup_resources()


def register_resource(resource: Any) -> None:
    """Register a resource for tracking"""
    _global_resource_manager.register_resource(resource)


def check_resources() -> Dict[str, Any]:
    """Check current resource usage"""
    return _global_resource_manager.resource_tracker.check_resources()


def can_allocate_resources() -> bool:
    """Check if resources can be allocated"""
    return _global_resource_manager.resource_tracker.can_allocate_resources()
