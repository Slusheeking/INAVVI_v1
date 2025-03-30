#!/usr/bin/env python3
"""
Exceptions Module

This module provides a standardized exception hierarchy for the trading system:

1. Base application exceptions
2. GPU-specific errors
3. Data validation errors
4. API/network errors
5. Trading-specific exceptions

All components in the trading system should use these exceptions for consistent
error handling, logging, and recovery.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Union

# Configure logging
logger = logging.getLogger("exceptions")


class TradingSystemError(Exception):
    """Base exception class for all trading system errors"""
    
    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """
        Initialize exception with enhanced error information
        
        Args:
            message: Error message
            code: Error code for categorization
            details: Additional error details
            cause: Original exception that caused this error
        """
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
        
        # Build error message
        error_msg = message
        if code:
            error_msg = f"[{code}] {error_msg}"
        
        super().__init__(error_msg)
        
        # Log exception with details
        log_msg = f"Exception {self.__class__.__name__}: {error_msg}"
        if details:
            log_msg += f" - Details: {details}"
        if cause:
            log_msg += f" - Caused by: {cause}"
            
        logger.error(log_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        result = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        
        if self.code:
            result["code"] = self.code
            
        if self.details:
            result["details"] = self.details
            
        if self.cause:
            result["cause"] = str(self.cause)
            
        return result


# Configuration Errors

class ConfigurationError(TradingSystemError):
    """Error in system configuration"""
    pass


class EnvironmentVariableError(ConfigurationError):
    """Error related to environment variables"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration value or structure"""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing"""
    pass


# GPU Errors

class GPUError(TradingSystemError):
    """Base class for GPU-related errors"""
    pass


class GPUNotAvailableError(GPUError):
    """GPU is not available when required"""
    pass


class GPUMemoryError(GPUError):
    """GPU memory allocation or management error"""
    pass


class GPUOperationError(GPUError):
    """Error during GPU operation execution"""
    pass


class CUDAError(GPUError):
    """CUDA-specific error"""
    pass


class TensorRTError(GPUError):
    """TensorRT-specific error"""
    pass


# Data Errors

class DataError(TradingSystemError):
    """Base class for data-related errors"""
    pass


class DataValidationError(DataError):
    """Data validation failed"""
    pass


class DataFormatError(DataError):
    """Data format is incorrect"""
    pass


class DataIntegrityError(DataError):
    """Data integrity check failed"""
    pass


class DataProcessingError(DataError):
    """Error during data processing"""
    pass


class DataSourceError(DataError):
    """Error with data source"""
    pass


class InsufficientDataError(DataError):
    """Not enough data available for operation"""
    pass


# API Errors

class APIError(TradingSystemError):
    """Base class for API-related errors"""
    pass


class APIConnectionError(APIError):
    """Error connecting to API"""
    pass


class APITimeoutError(APIError):
    """API request timed out"""
    pass


class APIResponseError(APIError):
    """Invalid or error response from API"""
    pass


class APIRateLimitError(APIError):
    """API rate limit exceeded"""
    pass


class APIAuthenticationError(APIError):
    """API authentication failed"""
    pass


class APIPermissionError(APIError):
    """API permission denied"""
    pass


# Network Errors

class NetworkError(TradingSystemError):
    """Base class for network-related errors"""
    pass


class ConnectionError(NetworkError):
    """Error establishing connection"""
    pass


class TimeoutError(NetworkError):
    """Connection or operation timed out"""
    pass


class WebSocketError(NetworkError):
    """WebSocket-specific error"""
    pass


# Cache Errors

class CacheError(TradingSystemError):
    """Base class for cache-related errors"""
    pass


class CacheConnectionError(CacheError):
    """Error connecting to cache"""
    pass


class RedisConnectionError(CacheConnectionError):
    """Error connecting to Redis"""
    pass


class CacheOperationError(CacheError):
    """Error during cache operation"""
    pass


class CacheSerializationError(CacheError):
    """Error serializing or deserializing cached data"""
    pass


# Model Errors

class ModelError(TradingSystemError):
    """Base class for model-related errors"""
    pass


class ModelLoadError(ModelError):
    """Error loading model"""
    pass


class ModelSaveError(ModelError):
    """Error saving model"""
    pass


class ModelInferenceError(ModelError):
    """Error during model inference"""
    pass


class ModelTrainingError(ModelError):
    """Error during model training"""
    pass


class ModelValidationError(ModelError):
    """Model validation failed"""
    pass


class ModelOptimizationError(ModelError):
    """Error during model optimization"""
    pass


# Trading Errors

class TradingError(TradingSystemError):
    """Base class for trading-related errors"""
    pass


class OrderError(TradingError):
    """Error related to order placement or management"""
    pass


class ExecutionError(TradingError):
    """Error during order execution"""
    pass


class PositionError(TradingError):
    """Error related to position management"""
    pass


class MarketDataError(TradingError):
    """Error with market data"""
    pass


class StrategyError(TradingError):
    """Error in trading strategy"""
    pass


class RiskLimitError(TradingError):
    """Risk limit exceeded"""
    pass


# System Errors

class SystemError(TradingSystemError):
    """Base class for system-related errors"""
    pass


class ResourceExhaustedError(SystemError):
    """System resource exhausted"""
    pass


class ConcurrencyError(SystemError):
    """Error related to concurrent operations"""
    pass


class DeadlockError(SystemError):
    """Deadlock detected"""
    pass


class ShutdownError(SystemError):
    """Error during system shutdown"""
    pass


# Utility Functions

def handle_exception(
    exc: Exception,
    log_level: int = logging.ERROR,
    reraise: bool = True,
    default_message: str = "An unexpected error occurred"
) -> Optional[TradingSystemError]:
    """
    Handle exception with consistent logging and wrapping
    
    Args:
        exc: Exception to handle
        log_level: Logging level to use
        reraise: Whether to reraise the exception
        default_message: Default message for generic exceptions
        
    Returns:
        Wrapped exception if not reraising, None otherwise
        
    Raises:
        TradingSystemError: Wrapped exception if reraise is True
    """
    # Get exception details
    exc_type = type(exc)
    exc_msg = str(exc)
    exc_traceback = traceback.format_exc()
    
    # Log exception
    logger.log(log_level, f"Exception {exc_type.__name__}: {exc_msg}")
    logger.log(log_level, f"Traceback: {exc_traceback}")
    
    # Wrap exception if not already a TradingSystemError
    if isinstance(exc, TradingSystemError):
        wrapped_exc = exc
    else:
        message = exc_msg or default_message
        wrapped_exc = TradingSystemError(message, cause=exc)
    
    # Reraise if requested
    if reraise:
        raise wrapped_exc
    
    return wrapped_exc


def is_retryable_exception(exc: Exception) -> bool:
    """
    Check if an exception is retryable
    
    Args:
        exc: Exception to check
        
    Returns:
        True if exception is retryable, False otherwise
    """
    # Network and API errors are generally retryable
    if isinstance(exc, (NetworkError, APITimeoutError, APIConnectionError)):
        return True
    
    # Rate limit errors are retryable with backoff
    if isinstance(exc, APIRateLimitError):
        return True
    
    # GPU memory errors might be retryable after cleanup
    if isinstance(exc, GPUMemoryError):
        return True
    
    # Some cache errors are retryable
    if isinstance(exc, CacheConnectionError):
        return True
    
    # Check for specific error messages in other exceptions
    error_msg = str(exc).lower()
    retryable_patterns = [
        "timeout", "connection reset", "connection refused",
        "temporarily unavailable", "retry", "try again",
        "too many requests", "service unavailable"
    ]
    
    return any(pattern in error_msg for pattern in retryable_patterns)


def format_exception(
    exc: Exception,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Format exception for API responses or logging
    
    Args:
        exc: Exception to format
        include_traceback: Whether to include traceback
        
    Returns:
        Formatted exception as dictionary
    """
    if isinstance(exc, TradingSystemError):
        result = exc.to_dict()
    else:
        result = {
            "error": exc.__class__.__name__,
            "message": str(exc)
        }
    
    if include_traceback:
        result["traceback"] = traceback.format_exc()
    
    return result
