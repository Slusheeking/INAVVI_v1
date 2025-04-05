"""ai_day_trader Utilities Package"""

# Import key components to make them available at the package level
from .config import Config, get_env_var, normalize_env_var_name # Removed get_config
from .exceptions import TradingSystemError, APIError, DataError, GPUError, ConfigurationError, DataValidationError, DataFormatError, DataIntegrityError, DataProcessingError, InsufficientDataError, RedisError, CacheError
from .gpu_utils import gpu_manager, is_gpu_available, to_gpu, from_gpu, process_array, clear_gpu_memory
from .logging_config import get_logger, configure_logging # Correct function name
from .metrics_registry import MetricPrefix, MetricLabel, register_counter, register_gauge, register_histogram
from .resource_manager import resource_managed, ResourceContext

__all__ = [
    # Config
    'Config', 'get_env_var', 'normalize_env_var_name', # Removed 'get_config'
    # Exceptions
    'TradingSystemError', 'APIError', 'DataError', 'GPUError', 'ConfigurationError',
    'DataValidationError', 'DataFormatError', 'DataIntegrityError', 'DataProcessingError',
    'InsufficientDataError', 'RedisError', 'CacheError',
    # GPU Utils
    'gpu_manager', 'is_gpu_available', 'to_gpu', 'from_gpu', 'process_array', 'clear_gpu_memory',
    # Logging
    'get_logger', 'configure_logging', # Correct function name in __all__
    # Metrics
    'MetricPrefix', 'MetricLabel', 'register_counter', 'register_gauge', 'register_histogram',
    # Resource Manager
    'resource_managed', 'ResourceContext',
]
