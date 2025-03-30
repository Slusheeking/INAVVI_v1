# Trading System Utilities

This directory contains standardized utility modules for the trading system. These modules provide common functionality used across the codebase, ensuring consistency and reducing duplication.

## Module Overview

### `__init__.py`

The package initialization file that exports commonly used functions and classes from the utility modules.

### `gpu_utils.py`

Handles GPU detection, initialization, and operations with proper fallbacks to CPU.

- **Key Components**:
  - `is_gpu_available()`: Checks if GPU is available
  - `gpu_manager`: Singleton for managing GPU resources
  - `to_gpu()`: Transfers data to GPU
  - `from_gpu()`: Transfers data from GPU to CPU
  - `process_array()`: Processes arrays on GPU with fallback
  - `clear_gpu_memory()`: Frees GPU memory

- **Configuration**:
  - GPU memory limits are set in `config.py`
  - Device selection can be controlled via environment variables

### `redis_helpers.py`

Provides standardized Redis client configuration and caching utilities.

- **Key Components**:
  - `RedisCache`: Class for caching with Redis
  - `get_redis_client()`: Factory function for Redis clients
  - `send_notification()`: Utility for Redis pub/sub notifications

- **Configuration**:
  - Redis connection parameters are in `config.py`
  - Connection pooling settings are in `config.py`
  - Default TTL values are in `config.py`

### `metrics_registry.py`

Defines standardized Prometheus metrics used throughout the application.

- **Key Components**:
  - API metrics (request counts, latencies, errors)
  - Cache metrics (hits, misses)
  - Data processing metrics
  - GPU usage metrics
  - Model metrics

- **Configuration**:
  - Metric naming conventions are defined in this file
  - Histogram buckets are configured here
  - Labels are standardized here

### `exceptions.py`

Defines a hierarchy of custom exceptions for the application.

- **Key Components**:
  - Base application exceptions
  - API-specific exceptions
  - Data processing exceptions
  - GPU-specific exceptions

- **Usage**:
  - Import specific exceptions from this module
  - Use them for error handling in your code
  - Extend the hierarchy for new exception types

### `config.py`

Centralizes configuration management with environment variable support.

- **Key Components**:
  - `config`: Dictionary with all configuration values
  - `get_config()`: Function to get configuration with defaults

- **Configuration Sources**:
  - Environment variables (primary)
  - Configuration files (secondary)
  - Default values (fallback)

- **Key Configuration Areas**:
  - API credentials and endpoints
  - Redis connection parameters
  - GPU settings
  - Logging levels
  - Performance tuning parameters

### `logging_config.py`

Standardizes logging formats and handlers.

- **Key Components**:
  - `get_logger()`: Factory function for loggers
  - Standardized log formats
  - Common handlers and filters

- **Configuration**:
  - Log levels are in `config.py`
  - Log file paths are in `config.py`
  - Format strings are defined in this module

### `async_utils.py`

Provides utilities for asynchronous operations.

- **Key Components**:
  - `async_retry()`: Retry decorator for async functions
  - `with_timeout()`: Timeout wrapper for async functions
  - `RateLimiter`: Class for API rate limiting
  - `TaskGroup`: Utility for managing groups of tasks
  - `AsyncLimiter`: Concurrency limiter

- **Configuration**:
  - Default retry parameters are in `config.py`
  - Timeout values are in `config.py`
  - Rate limiting parameters are in `config.py`

## Usage Guidelines

### Importing Utilities

```python
# Import specific utilities
from utils import get_logger, config

# Import from specific modules
from utils.gpu_utils import is_gpu_available, to_gpu, from_gpu
from utils.metrics_registry import API_REQUEST_COUNT
from utils.exceptions import APIError
```

### Configuration

1. **Environment Variables**: Set these in your `.env` file or system environment:
   ```
   # Redis configuration
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   
   # GPU configuration
   USE_GPU=true
   GPU_MEMORY_LIMIT=4096  # MB
   
   # API configuration
   POLYGON_API_KEY=your_api_key
   UNUSUAL_WHALES_API_KEY=your_api_key
   
   # Logging configuration
   LOG_LEVEL=INFO
   LOGS_DIR=./logs
   ```

2. **Configuration File**: You can also use a configuration file:
   ```python
   # config.json
   {
     "redis": {
       "host": "localhost",
       "port": 6379,
       "db": 0
     },
     "gpu": {
       "use_gpu": true,
       "memory_limit": 4096
     }
   }
   ```

3. **Default Values**: If neither environment variables nor configuration files are provided, the system will use default values defined in `config.py`.

### Error Handling

Use the custom exceptions defined in `exceptions.py` for consistent error handling:

```python
from utils.exceptions import APIError, APITimeoutError

try:
    # API call
    response = await client.get_data()
except APITimeoutError:
    # Handle timeout
    logger.warning("API timeout, retrying...")
except APIError as e:
    # Handle other API errors
    logger.error(f"API error: {e}")
```

### Metrics Collection

Use the standardized metrics from `metrics_registry.py`:

```python
from utils.metrics_registry import API_REQUEST_COUNT, API_REQUEST_LATENCY

# Increment counter
API_REQUEST_COUNT.labels(api="polygon", endpoint="stocks").inc()

# Record latency
with API_REQUEST_LATENCY.labels(api="polygon", endpoint="stocks").time():
    response = await client.get_stocks()
```

### GPU Operations

Use the GPU utilities for GPU-accelerated operations:

```python
from utils.gpu_utils import is_gpu_available, to_gpu, from_gpu

if is_gpu_available():
    # Transfer data to GPU
    gpu_data = to_gpu(data)
    
    # Process on GPU
    result = process_gpu_data(gpu_data)
    
    # Transfer back to CPU
    cpu_result = from_gpu(result)
else:
    # Fallback to CPU processing
    cpu_result = process_cpu_data(data)
```

## Extending the Utilities

### Adding New Metrics

To add new metrics, update `metrics_registry.py`:

```python
# Add a new counter
NEW_METRIC = Counter(
    "app_new_metric_total",
    "Description of the new metric",
    ["label1", "label2"]
)
```

### Adding New Exceptions

To add new exceptions, update `exceptions.py`:

```python
class NewError(BaseError):
    """New error type for specific scenarios."""
    pass
```

### Adding New Configuration

To add new configuration options, update `config.py`:

```python
# Add default value
DEFAULT_CONFIG = {
    # ...existing config...
    "new_feature": {
        "enabled": False,
        "timeout": 30
    }
}

# Add environment variable mapping
ENV_MAPPING = {
    # ...existing mapping...
    "NEW_FEATURE_ENABLED": ("new_feature.enabled", bool),
    "NEW_FEATURE_TIMEOUT": ("new_feature.timeout", int)
}