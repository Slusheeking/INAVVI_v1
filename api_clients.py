#!/usr/bin/env python3
"""
Unified API Clients Module

This module provides GPU-optimized clients for market data APIs:
1. Polygon.io REST API Client
2. Polygon.io WebSocket Client
3. Unusual Whales API Client

All clients share common infrastructure for caching, connection pooling,
GPU acceleration, and error handling while providing specialized functionality
for their respective APIs.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aiohttp
import numpy as np
import pandas as pd
import redis
import websockets
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed, WebSocketException

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_clients.log')
    ]
)
logger = logging.getLogger("api_clients")

# Load environment variables from .env file
load_dotenv()

# Third-party imports

# Import GPU acceleration libraries with fallback
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import PyTorch with fallback
try:
    import torch
    TORCH_AVAILABLE = True
    
    # Check for CUDA support
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = None

# Import TensorRT directly with fallback
try:
    import tensorrt as trt
    TENSORRT_VERSION = trt.__version__
    TENSORRT_AVAILABLE = True
    logger.info(f"Using TensorRT version {TENSORRT_VERSION}")
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    logger.warning("TensorRT not available, GPU model optimization disabled")

# Import Prometheus client for metrics
try:
    # Import only what we need
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure exception hook to log unhandled exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Call the default handler for KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# Configure asyncio exception handler
def handle_asyncio_exception(loop, context):
    msg = context.get("exception", context["message"])
    logger.error(f"Asyncio exception: {msg}")
    if "exception" in context:
        exc = context["exception"]
        logger.error(f"Exception details: {traceback.format_exception(type(exc), exc, exc.__traceback__)}")

asyncio.get_event_loop().set_exception_handler(handle_asyncio_exception)

# Environment variables with defaults from config.py

# API Keys
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
UNUSUAL_WHALES_API_KEY = os.environ.get("UNUSUAL_WHALES_API_KEY", "")

# Redis Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))  # Updated to match FastAPI demo
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "trading_system_2025")
REDIS_SSL = os.environ.get("REDIS_SSL", "false").lower() == "true"
REDIS_TIMEOUT = int(os.environ.get("REDIS_TIMEOUT", "5"))

# Connection Settings
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = float(os.environ.get("RETRY_BACKOFF_FACTOR", "0.5"))
CONNECTION_TIMEOUT = int(os.environ.get("CONNECTION_TIMEOUT", "15"))
MAX_POOL_SIZE = int(os.environ.get("MAX_POOL_SIZE", "30"))
RECONNECT_DELAY = float(os.environ.get("RECONNECT_DELAY", "2.0"))
MAX_RECONNECT_ATTEMPTS = int(os.environ.get("MAX_RECONNECT_ATTEMPTS", "10"))

# Cache Settings
POLYGON_CACHE_TTL = int(os.environ.get("POLYGON_CACHE_TTL", "3600"))
UNUSUAL_WHALES_CACHE_TTL = int(
    os.environ.get("UNUSUAL_WHALES_CACHE_TTL", "300"))

# Processing Settings
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", "1000"))

# Endpoint refresh intervals (in seconds)
POLYGON_REFRESH_INTERVALS = {
    "market_status": 60,  # Every minute
    "ticker_details": 86400,  # Daily
    "ticker_news": 1800,  # Every 30 minutes
    "aggregates": {
        "minute": 60,  # Every minute during market hours
        "hour": 3600,  # Every hour
        "day": 86400,  # Daily
    },
    "previous_close": 86400,  # Daily
    "last_quote": 5,  # Every 5 seconds during market hours
    "last_trade": 5,  # Every 5 seconds during market hours
}

UNUSUAL_WHALES_REFRESH_INTERVALS = {
    "unusual_activity": 30,  # Every 30 seconds
    "flow": 10,  # Every 10 seconds
    "latest_sweep": 5,  # Every 5 seconds
    "dark_pool": 60,  # Every 60 seconds
    "alerts": 60,  # Every 60 seconds
    "alert_configs": 3600,  # Every hour
    "insider": 3600,  # Every hour
}

# Initialize Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    # API client metrics
    API_REQUEST_COUNT = Counter(
        "api_client_request_count",
        "Number of API requests made",
        ["client", "endpoint", "method"],
    )
    API_REQUEST_LATENCY = Histogram(
        "api_client_request_latency_seconds",
        "API request latency in seconds",
        ["client", "endpoint", "method"],
    )
    API_ERROR_COUNT = Counter(
        "api_client_error_count",
        "Number of API errors",
        ["client", "endpoint", "method", "error_type"],
    )
    API_CACHE_HIT_COUNT = Counter(
        "api_client_cache_hit_count",
        "Number of cache hits",
        ["client", "cache_type"],
    )
    API_CACHE_MISS_COUNT = Counter(
        "api_client_cache_miss_count",
        "Number of cache misses",
        ["client", "cache_type"],
    )
    API_RATE_LIMIT_REMAINING = Gauge(
        "api_client_rate_limit_remaining",
        "Remaining API rate limit",
        ["client", "endpoint"],
    )
    API_WEBSOCKET_RECONNECTS = Counter(
        "api_client_websocket_reconnects",
        "Number of WebSocket reconnections",
        ["client", "endpoint"],
    )
    API_WEBSOCKET_MESSAGES = Counter(
        "api_client_websocket_messages",
        "Number of WebSocket messages received",
        ["client", "message_type"],
    )
    GPU_MEMORY_USAGE = Gauge(
        "api_client_gpu_memory_usage_bytes",
        "GPU memory usage in bytes",
        ["device"],
    )
    GPU_PROCESSING_TIME = Histogram(
        "api_client_gpu_processing_time_seconds",
        "GPU processing time in seconds",
        ["operation"],
    )


###################
# UTILITY CLASSES #
###################


class RedisCache:
    """Redis-based cache for API responses with fallback to in-memory cache
    
    This class provides a hybrid caching solution that combines Redis for distributed caching
    with an in-memory LRU cache for fast access to frequently used data. It includes:
    
    Features:
        - Automatic fallback to in-memory cache when Redis is unavailable
        - LRU-like memory cache with size limits and automatic cleanup
        - Support for both general data and DataFrame-specific caching
        - Built-in serialization using pickle with error handling
        - Prometheus metrics for monitoring cache performance
        - Comprehensive logging for debugging and monitoring
        
    Performance Optimizations:
        - In-memory cache for fastest access to hot data
        - Efficient key generation using MD5 hashing
        - Batch operations using Redis pipeline when available
        - Automatic cleanup of expired entries
        
    Example:
        >>> cache = RedisCache(prefix="api_cache")
        >>> cache.set(["user", 123], {"name": "John"})
        >>> data = cache.get(["user", 123])
        >>> df = pd.DataFrame({"col1": [1, 2, 3]})
        >>> cache.store_dataframe(["metrics", "daily"], df)
        >>> retrieved_df = cache.get_dataframe(["metrics", "daily"])
    """

    def __init__(
        self,
        prefix: str = "api_cache",
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        ttl: int = 3600,
        username: str = REDIS_USERNAME,
        password: str = REDIS_PASSWORD,
        *,  # Force remaining args to be keyword-only
        ssl: bool = REDIS_SSL,
        max_memory_items: int = 10000,
        connection_pool: Optional[redis.ConnectionPool] = None
    ) -> None:
        """Initialize Redis cache with prefix for namespace separation
        
        Args:
            prefix: Namespace prefix for cache keys
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Default time-to-live for cache entries in seconds
            username: Redis username
            password: Redis password
            ssl: Whether to use SSL for Redis connection
            max_memory_items: Maximum number of items in memory cache
            connection_pool: Optional existing Redis connection pool
        """
        self.prefix = prefix
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.enabled = True
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.size_limit = max_memory_items

        try:
            if connection_pool:
                self.redis_client = redis.Redis(
                    connection_pool=connection_pool,
                    decode_responses=False  # Keep binary data for efficient serialization
                )
                logger.info(f"Using existing Redis connection pool [prefix: {self.prefix}]")
            else:
                # Create a new connection pool
                pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    username=username,
                    password=password,
                    ssl=ssl,
                    ssl_cert_reqs=None,
                    socket_timeout=REDIS_TIMEOUT,
                    socket_connect_timeout=5,
                    decode_responses=False,  # Keep binary data for efficient serialization
                    max_connections=50  # Limit maximum connections
                )
                self.redis_client = redis.Redis(connection_pool=pool)
                logger.info(f"Created new Redis connection pool [prefix: {self.prefix}]")

            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port} [prefix: {self.prefix}]")

            # Monitor Redis memory usage
            if PROMETHEUS_AVAILABLE:
                try:
                    info = self.redis_client.info()
                    used_memory = info.get('used_memory', 0)
                    max_memory = info.get('maxmemory', 0)
                    logger.info(f"Redis memory usage: {used_memory}/{max_memory} bytes")
                except redis.RedisError as e:
                    logger.warning(f"Could not get Redis memory info: {e}")

        except (redis.RedisError, ConnectionError) as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            logger.error(f"Connection details: host={host}, port={port}, db={db}, ssl={ssl}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            logger.warning("Falling back to in-memory cache")
            self.enabled = False

    def _generate_key(self, key_parts) -> str:
        """Generate a consistent cache key from parts"""
        if isinstance(key_parts, str):
            key_parts = [key_parts]

        # Join all parts and create a hash
        key_str = ":".join([str(part) for part in key_parts])
        return f"{self.prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"

    def get(self, key_parts: Union[str, List[Any]]) -> Optional[Any]:
        """Get value from cache with fallback to Redis
        
        Args:
            key_parts: Key parts to generate cache key from
            
        Returns:
            Cached value if found and not expired, None otherwise
            
        Example:
            >>> cache = RedisCache(prefix="api_cache")
            >>> value = cache.get(["user", 123, "profile"])
            >>> if value is None:
            ...     value = fetch_from_db()
            ...     cache.set(["user", 123, "profile"], value)
        """
        try:
            start_time = time.time()
            key = self._generate_key(key_parts)
            logger.debug(f"Attempting to get cache key: {key}")

            # Try memory cache first for fastest access
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry["expiry"]:
                    self.hits += 1
                    # Update Prometheus metrics if available
                    if PROMETHEUS_AVAILABLE:
                        API_CACHE_HIT_COUNT.labels(
                            client=self.prefix,
                            cache_type="memory",
                        ).inc()
                    logger.debug(f"Memory cache hit for key: {key}")
                    return entry["value"]
                # Entry expired
                logger.debug(f"Memory cache entry expired for key: {key}")
                del self.memory_cache[key]

            # Try Redis if enabled
            if self.enabled:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        self.hits += 1
                        # Update Prometheus metrics if available
                        if PROMETHEUS_AVAILABLE:
                            API_CACHE_HIT_COUNT.labels(
                                client=self.prefix,
                                cache_type="redis",
                            ).inc()
                        try:
                            # Store in memory cache for faster subsequent access
                            value = pickle.loads(data)
                            self.memory_cache[key] = {
                                "value": value,
                                "expiry": time.time() + self.ttl,
                            }
                            logger.debug(f"Redis cache hit for key: {key}")
                            return value
                        except pickle.PickleError as e:
                            logger.error(f"Error deserializing Redis data for key {key}: {e}")
                            logger.error(f"Stack trace: {traceback.format_exc()}")
                            # Remove corrupted data from Redis
                            self.redis_client.delete(key)
                except redis.RedisError as e:
                    logger.error(f"Redis error retrieving key {key}: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                except Exception as e:
                    logger.error(f"Unexpected error retrieving from Redis: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")

            self.misses += 1
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                API_CACHE_MISS_COUNT.labels(
                    client=self.prefix, cache_type="all").inc()
                # Record cache operation latency
                latency = time.time() - start_time
                API_REQUEST_LATENCY.labels(
                    client=self.prefix,
                    endpoint="cache_get",
                    method="GET"
                ).observe(latency)
            
            logger.debug(f"Cache miss for key: {key}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in cache get: {e}")
            logger.error(f"Key parts: {key_parts}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def set(self, key_parts: Union[str, List[Any]], value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL
        
        Args:
            key_parts: Key parts to generate cache key from
            value: Value to store in cache
            ttl: Optional time-to-live in seconds (defaults to self.ttl)
            
        Returns:
            bool: True if value was successfully stored, False otherwise
            
        Example:
            >>> cache = RedisCache(prefix="api_cache")
            >>> success = cache.set(["user", 123, "profile"], {"name": "John"}, ttl=3600)
        """
        try:
            start_time = time.time()

            if value is None:
                logger.warning("Attempted to cache None value")
                return False

            key = self._generate_key(key_parts)
            ttl = ttl or self.ttl
            expiry = time.time() + ttl

            logger.debug(f"Setting cache key: {key} (TTL: {ttl}s)")

            # Implement LRU-like behavior by limiting memory cache size
            if len(self.memory_cache) >= self.size_limit:
                # Remove random 10% of entries when cache gets too large
                keys_to_remove = random.sample(
                    list(self.memory_cache.keys()),
                    int(len(self.memory_cache) * 0.1),
                )
                removed_count = len(keys_to_remove)
                for k in keys_to_remove:
                    self.memory_cache.pop(k, None)
                logger.debug(f"Removed {removed_count} entries from memory cache")

            # Store in memory cache
            self.memory_cache[key] = {"value": value, "expiry": expiry}
            logger.debug("Value stored in memory cache")

            # Store in Redis if enabled
            if self.enabled:
                try:
                    serialized = pickle.dumps(value)
                    success = self.redis_client.setex(key, ttl, serialized)
                    if success:
                        logger.debug("Value successfully stored in Redis")
                    else:
                        logger.warning(f"Redis setex returned False for key: {key}")
                    
                    # Record cache operation latency
                    if PROMETHEUS_AVAILABLE:
                        latency = time.time() - start_time
                        API_REQUEST_LATENCY.labels(
                            client=self.prefix,
                            endpoint="cache_set",
                            method="SET"
                        ).observe(latency)
                    
                    return success
                except pickle.PickleError as e:
                    logger.error(f"Error serializing value for Redis: {e}")
                    logger.error(f"Key: {key}, Value type: {type(value)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    return False
                except redis.RedisError as e:
                    logger.error(f"Redis error setting key {key}: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    return False
                except Exception as e:
                    logger.error(f"Unexpected error storing in Redis: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Unexpected error in cache set: {e}")
            logger.error(f"Key parts: {key_parts}")
            logger.error(f"Value type: {type(value)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
def store_dataframe(self, key_parts: Union[str, List[Any]], df: pd.DataFrame, ttl: Optional[int] = None) -> bool:
    """Store DataFrame in cache with optimized serialization
    
    Args:
        key_parts: Key parts to generate cache key from
        df: DataFrame to store in cache
        ttl: Optional time-to-live in seconds (defaults to self.ttl)
        
    Returns:
        bool: True if DataFrame was successfully stored, False otherwise
        
    Example:
        >>> cache = RedisCache(prefix="api_cache")
        >>> df = pd.DataFrame({"col1": [1, 2, 3]})
        >>> success = cache.store_dataframe(["data", "metrics"], df, ttl=3600)
    """
    try:
        start_time = time.time()

        if df is None or df.empty:
            logger.warning("Attempted to cache None or empty DataFrame")
            return False

        # Use pickle for serialization instead of PyArrow due to compatibility issues
        try:
            key = self._generate_key(key_parts)
            ttl = ttl or self.ttl
            logger.debug(f"Serializing DataFrame for key: {key} (TTL: {ttl}s)")
            
            try:
                serialized = pickle.dumps(df)
            except pickle.PickleError as e:
                logger.error(f"Error serializing DataFrame: {e}")
                logger.error(f"DataFrame info: {df.info()}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                return False

            if self.enabled:
                try:
                    success = self.redis_client.setex(key, ttl, serialized)
                    if success:
                        logger.debug("DataFrame successfully stored in Redis")
                    else:
                        logger.warning(f"Redis setex returned False for key: {key}")
                    
                    # Record cache operation latency
                    if PROMETHEUS_AVAILABLE:
                        latency = time.time() - start_time
                        API_REQUEST_LATENCY.labels(
                            client=self.prefix,
                            endpoint="cache_store_df",
                            method="SET"
                        ).observe(latency)
                    
                    return success
                except redis.RedisError as e:
                    logger.error(f"Redis error storing DataFrame: {e}")
                    logger.error(f"Key: {key}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")

            # Store in memory cache
            self.memory_cache[key] = {"value": df, "expiry": time.time() + ttl}
            logger.debug("DataFrame stored in memory cache")
            return True

        except Exception as e:
            logger.error(f"Error in DataFrame serialization/storage: {e}")
            logger.error(f"Key parts: {key_parts}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Fallback to regular set method
            return self.set(key_parts, df, ttl)

    except Exception as e:
        logger.error(f"Unexpected error in store_dataframe: {e}")
        logger.error(f"Key parts: {key_parts}")
        logger.error(f"DataFrame shape: {df.shape if df is not None else None}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

    def get_dataframe(self, key_parts: Union[str, List[Any]]) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from cache with optimized deserialization
        
        Args:
            key_parts: Key parts to generate cache key from
            
        Returns:
            Optional[pd.DataFrame]: Retrieved DataFrame if found and valid, None otherwise
            
        Example:
            >>> cache = RedisCache(prefix="api_cache")
            >>> df = cache.get_dataframe(["data", "metrics"])
            >>> if df is not None:
            ...     print(f"Retrieved DataFrame with shape: {df.shape}")
        """
        try:
            start_time = time.time()
            key = self._generate_key(key_parts)
            logger.debug(f"Attempting to get DataFrame for key: {key}")

            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry["expiry"]:
                    self.hits += 1
                    # Update Prometheus metrics if available
                    if PROMETHEUS_AVAILABLE:
                        API_CACHE_HIT_COUNT.labels(
                            client=self.prefix,
                            cache_type="memory",
                        ).inc()
                    logger.debug(f"Memory cache hit for DataFrame key: {key}")
                    return entry["value"]
                logger.debug(f"Memory cache entry expired for key: {key}")
                del self.memory_cache[key]

            # Try Redis if enabled
            if self.enabled:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        self.hits += 1
                        # Update Prometheus metrics if available
                        if PROMETHEUS_AVAILABLE:
                            API_CACHE_HIT_COUNT.labels(
                                client=self.prefix,
                                cache_type="redis",
                            ).inc()
                            
                        # Deserialize using pickle
                        try:
                            df = pickle.loads(data)
                            if not isinstance(df, pd.DataFrame):
                                logger.error(f"Cached value is not a DataFrame: {type(df)}")
                                return None

                            # Store in memory cache
                            self.memory_cache[key] = {
                                "value": df,
                                "expiry": time.time() + self.ttl,
                            }
                            
                            # Record cache operation latency
                            if PROMETHEUS_AVAILABLE:
                                latency = time.time() - start_time
                                API_REQUEST_LATENCY.labels(
                                    client=self.prefix,
                                    endpoint="cache_get_df",
                                    method="GET"
                                ).observe(latency)
                                
                            logger.debug(f"Redis cache hit for DataFrame key: {key}")
                            return df
                        except pickle.PickleError as e:
                            logger.error(f"Error deserializing DataFrame: {e}")
                            logger.error(f"Key: {key}")
                            logger.error(f"Stack trace: {traceback.format_exc()}")
                            # Remove corrupted data from Redis
                            self.redis_client.delete(key)
                        except Exception as e:
                            logger.error(f"Unexpected error deserializing DataFrame: {e}")
                            logger.error(f"Key: {key}")
                            logger.error(f"Stack trace: {traceback.format_exc()}")
                except redis.RedisError as e:
                    logger.error(f"Redis error retrieving DataFrame: {e}")
                    logger.error(f"Key: {key}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                except Exception as e:
                    logger.error(f"Unexpected error retrieving from Redis: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")

            self.misses += 1
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                API_CACHE_MISS_COUNT.labels(
                    client=self.prefix, cache_type="all").inc()
                # Record cache operation latency
                latency = time.time() - start_time
                API_REQUEST_LATENCY.labels(
                    client=self.prefix,
                    endpoint="cache_get_df",
                    method="GET"
                ).observe(latency)
            
            logger.debug(f"Cache miss for DataFrame key: {key}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in get_dataframe: {e}")
            logger.error(f"Key parts: {key_parts}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return None


class AsyncConnectionPool:
    """Asynchronous HTTP connection pool with retry logic"""

    def __init__(
        self,
        max_retries=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        max_pool_size=MAX_POOL_SIZE,
        timeout=CONNECTION_TIMEOUT,
    ) -> None:
        """Initialize connection pool"""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.session = None
        self.connector = None

        logger.info(
            f"Async connection pool initialized with max_pool_size={max_pool_size}, max_retries={max_retries}",
        )

    async def initialize(self) -> None:
        """Initialize the aiohttp session"""
        if self.session is None or self.session.closed:
            # Configure TCP connector with connection pooling
            self.connector = aiohttp.TCPConnector(
                limit=MAX_POOL_SIZE,
                ttl_dns_cache=300,  # Cache DNS results for 5 minutes
                use_dns_cache=True,
                ssl=False,  # API uses HTTPS but we don't need to verify
            )

            # Create session with default headers and timeout
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                headers={
                    "User-Agent": "GPUTradingClient/1.0",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )

    async def get(self, url, params=None, headers=None):
        """Make GET request with retry logic"""
        if self.session is None or self.session.closed:
            await self.initialize()

        last_error = None

        for retry in range(self.max_retries + 1):
            try:
                async with self.session.get(
                    url,
                    params=params,
                    headers=headers,
                ) as response:
                    # Check if response is successful
                    if response.status == 200:
                        return await response.json()

                    # Handle rate limiting
                    if response.status == 429:
                        wait_time = (2**retry) * self.backoff_factor + random.uniform(
                            0,
                            1,
                        )
                        logger.warning(
                            f"Rate limit hit, retrying in {wait_time:.2f}s (attempt {retry+1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    # Handle other errors
                    error_text = await response.text()
                    last_error = f"HTTP {response.status}: {error_text}"

                    # Don't retry on client errors except rate limits
                    if 400 <= response.status < 500 and response.status != 429:
                        return {"status": "ERROR", "error": last_error}

                    # Retry on server errors
                    wait_time = (2**retry) * self.backoff_factor + \
                        random.uniform(0, 1)
                    logger.warning(
                        f"Server error, retrying in {wait_time:.2f}s (attempt {retry+1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = f"Request failed: {type(e).__name__}: {e!s}"

                # Exponential backoff with jitter
                wait_time = (2**retry) * self.backoff_factor + \
                    random.uniform(0, 1)
                logger.warning(
                    f"Connection error, retrying in {wait_time:.2f}s (attempt {retry+1}/{self.max_retries}): {e}"
                )
                await asyncio.sleep(wait_time)

        # All retries failed
        return {"status": "ERROR", "error": last_error or "Max retries exceeded"}

    async def close(self) -> None:
        """Close the session and all connections"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Async connection pool closed")


class GPUAccelerator:
    """GPU acceleration utilities for data processing with PyTorch, TensorRT, and CuPy"""

    def __init__(self, use_gpu=USE_GPU) -> None:
        """Initialize GPU acceleration"""
        self.use_gpu = use_gpu
        self.gpu_initialized = False
        self.torch_initialized = False
        self.device_name = None
        self.device_id = 0
        self.memory_limit = int(os.environ.get(
            "PYTORCH_CUDA_MEM_LIMIT_MB", "16000"))
        self.use_mixed_precision = (
            os.environ.get("PYTORCH_MIXED_PRECISION", "true").lower() == "true"
        )
        self.tensorrt_precision = os.environ.get(
            "TENSORRT_PRECISION_MODE", "FP16")

        # Initialize GPU if available and requested
        if self.use_gpu:
            self._initialize_gpu()

    def _initialize_gpu(self) -> None:
        """Initialize GPU for data processing with PyTorch, TensorRT, and CuPy"""
        # Initialize CuPy if available
        if CUPY_AVAILABLE:
            try:
                # Get device count
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count > 0:
                    # Find GH200 device if available
                    for i in range(device_count):
                        device_props = cp.cuda.runtime.getDeviceProperties(i)
                        device_name = device_props["name"].decode()
                        if "GH200" in device_name:
                            cp.cuda.Device(i).use()
                            self.device_name = device_name
                            self.device_id = i
                            logger.info(
                                f"Using GH200 GPU at index {i}: {device_name}")
                            break

                    if not self.device_name:
                        # Use first available GPU if GH200 not found
                        device_id = cp.cuda.Device().id
                        device_props = cp.cuda.runtime.getDeviceProperties(
                            device_id)
                        self.device_name = device_props["name"].decode()
                        self.device_id = device_id
                        logger.info(
                            f"Using GPU at index {device_id}: {self.device_name}",
                        )

                    # Use unified memory for better performance
                    cp.cuda.set_allocator(
                        cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc,
                    )
                    logger.info("CuPy configured with unified memory")
                    self.gpu_initialized = True
                else:
                    logger.warning("No GPU devices found for CuPy")
                    self.gpu_initialized = False
            except Exception as e:
                logger.exception(f"Error initializing CuPy: {e}")
                self.gpu_initialized = False
        else:
            logger.warning(
                "CuPy not available, CuPy GPU acceleration disabled")
            self.gpu_initialized = False

        # Initialize PyTorch if available
        if TORCH_AVAILABLE and self.use_gpu:
            try:
                if CUDA_AVAILABLE:
                    # Get device count and properties
                    device_count = torch.cuda.device_count()
                    if device_count > 0:
                        logger.info(f"PyTorch detected {device_count} GPU(s)")
                        
                        # Find GH200 device if available
                        for i in range(device_count):
                            device_name = torch.cuda.get_device_name(i)
                            if "GH200" in device_name:
                                torch.cuda.set_device(i)
                                self.device_name = device_name
                                self.device_id = i
                                logger.info(f"Using GH200 GPU at index {i}: {device_name}")
                                break
                        
                        if not self.device_name:
                            # Use first available GPU if GH200 not found
                            self.device_id = 0
                            self.device_name = torch.cuda.get_device_name(0)
                            torch.cuda.set_device(0)
                            logger.info(f"Using GPU at index 0: {self.device_name}")
                        
                        # Set memory limit if specified
                        if self.memory_limit > 0:
                            try:
                                # PyTorch doesn't have a direct API for memory limits
                                # but we can use CUDA caching allocator settings
                                torch.cuda.set_per_process_memory_fraction(
                                    min(self.memory_limit * 1024 * 1024 / torch.cuda.get_device_properties(self.device_id).total_memory, 1.0)
                                )
                                logger.info(f"Set PyTorch memory limit to {self.memory_limit}MB")
                            except Exception as e:
                                logger.warning(f"Error setting PyTorch memory limit: {e}")
                        
                        # Enable mixed precision if configured
                        if self.use_mixed_precision:
                            try:
                                # PyTorch uses amp (automatic mixed precision)
                                logger.info("Enabled mixed precision (float16) for PyTorch")
                            except Exception as e:
                                logger.warning(f"Could not set mixed precision: {e}")
                        
                        # Test PyTorch GPU
                        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
                        b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=DEVICE)
                        c = torch.matmul(a, b)
                        logger.info(f"PyTorch GPU test successful: {c.cpu().numpy()}")
                        
                        self.torch_initialized = True
                        
                        # Check TensorRT availability
                        if TENSORRT_AVAILABLE:
                            logger.info(f"TensorRT is available for model optimization (precision: {self.tensorrt_precision})")
                    else:
                        logger.warning("No GPU detected by PyTorch")
                        self.torch_initialized = False
                else:
                    logger.warning("CUDA not available for PyTorch")
                    self.torch_initialized = False
            except Exception as e:
                logger.exception(f"Error initializing PyTorch: {e}")
                self.torch_initialized = False
        else:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, PyTorch GPU acceleration disabled")
            self.torch_initialized = False

        # Update use_gpu flag based on initialization results
        self.use_gpu = self.gpu_initialized or self.torch_initialized

        if self.use_gpu:
            logger.info("GPU acceleration successfully initialized with:")
            if self.gpu_initialized:
                logger.info("- CuPy for array operations")
            if self.torch_initialized:
                logger.info("- PyTorch for tensor operations")
                if TENSORRT_AVAILABLE:
                    logger.info("- TensorRT for model optimization")
        else:
            logger.warning(
                "GPU acceleration disabled - neither CuPy nor PyTorch initialized successfully"
            )

    def process_numpy_array(self, array, func):
        """
        Process numpy array with GPU acceleration if available

        Args:
            array: Numpy array to process
            func: Function to apply (should work with both numpy and cupy)

        Returns:
            Result as numpy array
        """
        if not self.use_gpu or not self.gpu_initialized or not CUPY_AVAILABLE:
            return func(array)

        try:
            # Convert to cupy array
            cp_array = cp.array(array)

            # Apply function
            cp_result = func(cp_array)

            # Convert back to numpy
            return cp.asnumpy(cp_result)
        except Exception as e:
            logger.exception(f"Error in GPU processing: {e}")
            return func(array)

    def process_with_pytorch(self, data, func):
        """
        Process data with PyTorch GPU acceleration

        Args:
            data: Input data (numpy array or pandas DataFrame)
            func: Function to apply (should work with PyTorch)

        Returns:
            Processed data
        """
        if not self.use_gpu or not self.torch_initialized or not TORCH_AVAILABLE:
            return func(data)

        try:
            # Convert input data to PyTorch tensor if needed
            if isinstance(data, np.ndarray):
                torch_data = torch.tensor(data, dtype=torch.float32, device=DEVICE)
            elif isinstance(data, pd.DataFrame):
                # For DataFrames, convert numeric columns to tensors
                numeric_cols = data.select_dtypes(include=["number"]).columns
                torch_data = {
                    col: torch.tensor(
                        data[col].values, dtype=torch.float32, device=DEVICE)
                    for col in numeric_cols
                }
            else:
                # Try direct conversion
                torch_data = torch.tensor(data, dtype=torch.float32, device=DEVICE)

            # Process with PyTorch on GPU
            with torch.cuda.device(self.device_id):
                result = func(torch_data)

            # Convert result back to numpy if it's a tensor
            if isinstance(result, torch.Tensor):
                return result.cpu().numpy()
            return result

        except Exception as e:
            logger.exception(f"Error in PyTorch processing: {e}")
            return func(data)

    def optimize_with_tensorrt(self, model_path, output_path=None, precision_mode=None):
        """
        Optimize a PyTorch model with TensorRT via ONNX

        Args:
            model_path: Path to saved PyTorch model (.pt or .pth file)
            output_path: Path to save optimized model (default: model_path + '_trt')
            precision_mode: Precision mode ('FP32', 'FP16', or 'INT8')

        Returns:
            Path to optimized model or None if optimization failed
        """
        if not self.use_gpu or not self.torch_initialized or not TENSORRT_AVAILABLE:
            logger.warning("TensorRT optimization not available")
            return None

        try:
            # Set default output path if not provided
            if output_path is None:
                output_path = model_path + "_trt"

            # Set default precision mode if not provided
            if precision_mode is None:
                precision_mode = self.tensorrt_precision

            logger.info(
                f"Optimizing PyTorch model with TensorRT via ONNX: {model_path} -> {output_path} (precision: {precision_mode})",
            )

            # Step 1: Load PyTorch model
            model = torch.load(model_path, map_location=DEVICE)
            model.eval()  # Set to evaluation mode

            # Step 2: Export to ONNX (intermediate step)
            onnx_path = model_path + ".onnx"
            
            # Note: The following is a placeholder. In a real implementation,
            # you would need to know the input shape and names for the model
            # These would typically be provided by the user or inferred from the model
            dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)  # Example input
            input_names = ["input"]
            output_names = ["output"]
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=13,
                do_constant_folding=True,
            )
            
            logger.info(f"Exported PyTorch model to ONNX: {onnx_path}")

            # Step 3: Use TensorRT to optimize the ONNX model
            import tensorrt as trt
            
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"TensorRT ONNX parser error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
            
            config = builder.create_builder_config()
            config.max_workspace_size = 8 * (1 << 30)  # 8GB
            
            # Set precision mode
            if precision_mode == "FP16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision_mode == "INT8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # Note: INT8 calibration would be needed here for proper INT8 support
            
            # Build and save engine
            serialized_engine = builder.build_serialized_network(network, config)
            with open(output_path, "wb") as f:
                f.write(serialized_engine)
            
            logger.info(f"Successfully optimized model with TensorRT: {output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"Error optimizing model with TensorRT: {e}")
            return None

    def clear_memory(self) -> None:
        """Clear GPU memory to prevent fragmentation"""
        # Clear CuPy memory if initialized
        if self.gpu_initialized and CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("CuPy GPU memory cleared")
            except Exception as e:
                logger.exception(f"Error clearing CuPy GPU memory: {e}")

        # Clear PyTorch memory if initialized
        if self.torch_initialized and TORCH_AVAILABLE:
            try:
                # Clear PyTorch memory
                torch.cuda.empty_cache()
                logger.info("PyTorch GPU memory cleared")
            except Exception as e:
                logger.exception(f"Error clearing PyTorch GPU memory: {e}")


##########################
# POLYGON REST API CLIENT #
##########################


class PolygonRESTClient:
    """GPU-Optimized client for the Polygon.io REST API"""

    def __init__(
        self,
        api_key=POLYGON_API_KEY,
        redis_client=None,
        use_gpu=USE_GPU,
        max_pool_size=MAX_POOL_SIZE,
        max_retries=MAX_RETRIES,
        cache_ttl=POLYGON_CACHE_TTL,
    ) -> None:
        """
        Initialize GPU-Optimized Polygon API client

        Args:
            api_key: API key for authentication
            redis_client: Optional Redis client for caching
            use_gpu: Whether to use GPU acceleration
            max_pool_size: Maximum connection pool size
            max_retries: Maximum number of retries for failed requests
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.use_gpu = use_gpu
        self.redis_client = redis_client

        # Verify API key is provided
        if not self.api_key:
            logger.warning(
                "No API key provided. Set the POLYGON_API_KEY environment variable.",
            )
        else:
            logger.info(
                f"Initialized Polygon API client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}",
            )

        # Initialize base URL
        self.base_url = "https://api.polygon.io"

        # Initialize cache
        self.cache = RedisCache(prefix="polygon", ttl=cache_ttl)

        # Initialize connection pool
        self.connection_pool = AsyncConnectionPool(
            max_pool_size=max_pool_size,
            max_retries=max_retries,
        )

        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Initialize GPU accelerator
        self.gpu_accelerator = GPUAccelerator(use_gpu=use_gpu)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Flag to track if client is running
        self.running = True

        # Scheduled tasks
        self.scheduled_tasks = {}
        self.event_loop = None

    def _handle_signal(self, signum, frame) -> None:
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
        asyncio.create_task(self.close())

    def _send_frontend_notification(self, message, level="info", category="api", details=None):
        """
        Send notification to frontend via Redis

        Args:
            message: Notification message
            level: Notification level (info, success, warning, error)
            category: Notification category
            details: Additional details as dictionary
        """
        if not self.redis_client:
            return

        try:
            notification = {
                "timestamp": time.time(),
                "message": message,
                "level": level,
                "category": category,
                "source": "polygon_rest_api",
                "details": details or {}
            }

            # Add to notifications list
            self.redis_client.lpush(
                "frontend:notifications", json.dumps(notification))

            # Trim list to prevent unbounded growth
            self.redis_client.ltrim("frontend:notifications", 0, 999)

            # Publish event for real-time updates
            self.redis_client.publish("frontend:events", json.dumps({
                "type": "notification",
                "data": notification
            }))

            logger.debug(f"Sent frontend notification: {message}")
        except Exception as e:
            logger.error(f"Error sending frontend notification: {e}")

    async def _make_request(self, endpoint, params=None):
        """
        Make API request with caching

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            API response as a dictionary
        """
        # Generate cache key
        # Skip cache for certain endpoints that change frequently
        skip_cache = False
        if "range" in endpoint and "minute" in endpoint:
            skip_cache = True

        cache_key = [endpoint]
        if params:
            for key, value in sorted(params.items()):
                cache_key.append(f"{key}={value}")

        # Check cache first
        if not skip_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_data

        # Prepare request
        url = f"{self.base_url}/{endpoint}"

        # Verify API key is set and valid format
        if not self.api_key or len(self.api_key) < 10:
            logger.error(f"Invalid API key format: {self.api_key[:4]}...")
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    client="polygon",
                    endpoint=endpoint,
                    method="GET",
                    error_type="invalid_api_key",
                ).inc()
            return {"status": "ERROR", "error": "Invalid API key format"}

        # Add API key to params for Polygon API
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        # Record request start time for latency measurement
        start_time = time.time()

        # Make request
        try:
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                API_REQUEST_COUNT.labels(
                    client="polygon",
                    endpoint=endpoint,
                    method="GET",
                ).inc()

            data = await self.connection_pool.get(url, params=params)

            # Calculate request latency
            latency = time.time() - start_time

            # Update latency metrics
            if PROMETHEUS_AVAILABLE:
                API_REQUEST_LATENCY.labels(
                    client="polygon",
                    endpoint=endpoint,
                    method="GET",
                ).observe(latency)

            # Cache successful responses
            if isinstance(data, dict):
                if data.get("status") == "OK" and not skip_cache:
                    self.cache.set(cache_key, data)
                elif "results" in data and not skip_cache:
                    # Some endpoints return results without a status field
                    self.cache.set(cache_key, data)
                elif data.get("status") == "ERROR":
                    # Update error metrics
                    if PROMETHEUS_AVAILABLE:
                        API_ERROR_COUNT.labels(
                            client="polygon",
                            endpoint=endpoint,
                            method="GET",
                            error_type="api_error",
                        ).inc()
                    logger.error(
                        f"API error for {endpoint}: {data.get('error', 'Unknown error')}"
                    )

            return data
        except Exception as e:
            # Calculate request latency even for failed requests
            latency = time.time() - start_time

            # Update error metrics
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    client="polygon",
                    endpoint=endpoint,
                    method="GET",
                    error_type=type(e).__name__,
                ).inc()
                API_REQUEST_LATENCY.labels(
                    client="polygon",
                    endpoint=endpoint,
                    method="GET",
                ).observe(latency)

            logger.exception(
                f"Unexpected error during API request: {type(e).__name__}: {e}",
            )
            return {
                "status": "ERROR",
                "error": f"Unexpected error: {type(e).__name__}: {e}",
            }

    async def get_market_status(self):
        """
        Get the current market status

        Returns:
            dict: Market status information
        """
        try:
            # Make request to market status endpoint
            endpoint = "v1/marketstatus/now"
            data = await self._make_request(endpoint)

            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Market status retrieved: {data.get('market')}")

                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.set("market:status", json.dumps(data))
                    self.redis_client.set(
                        "market:status:last_update",
                        datetime.now().isoformat(),
                    )

                    # Send notification to frontend
                    self._send_frontend_notification(
                        message=f"Market status updated: {data.get('market', 'unknown')}",
                        level="info",
                        category="market_data",
                        details={
                            "status": data.get("market", "unknown"),
                            "early_hours": data.get("early_hours", False),
                            "after_hours": data.get("after_hours", False),
                            "currencies": data.get("currencies", {}),
                            "server_time": data.get("serverTime")
                        }
                    )

                return data
            error_msg = (
                data.get("error", "Unknown error")
                if isinstance(data, dict)
                else "Unknown error"
            )
            logger.warning(f"Failed to get market status: {error_msg}")

            # Create a fallback market status
            fallback_status = {
                "status": "OK",
                "market": "open",
                "serverTime": datetime.now().isoformat(),
            }
            # Store fallback in Redis if available
            if self.redis_client:
                self.redis_client.set(
                    "market:status", json.dumps(fallback_status))
                self.redis_client.set(
                    "market:status:last_update",
                    datetime.now().isoformat(),
                )
            return None
        except Exception as e:
            logger.exception(f"Error getting market status: {e}")
            return None

    async def get_ticker_details(self, ticker):
        """
        Get details for a specific ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            dict: Ticker details
        """
        try:
            endpoint = f"v3/reference/tickers/{ticker}"
            data = await self._make_request(endpoint)

            if data and "status" in data and data["status"] == "OK":
                logger.info(f"Ticker details retrieved for {ticker}")

                # Store in Redis if available
                if self.redis_client:
                    result = data.get("results")
                    if result:
                        self.redis_client.hset(
                            f"stock:{ticker}:metadata",
                            mapping=result,
                        )
                        self.redis_client.set(
                            f"stock:{ticker}:metadata:last_update",
                            datetime.now().isoformat(),
                        )

                        # Send notification to frontend
                        self._send_frontend_notification(
                            message=f"Updated ticker details for {ticker}",
                            level="info",
                            category="market_data",
                            details={
                                "ticker": ticker,
                                "name": result.get("name", ""),
                                "market_cap": result.get("market_cap", 0),
                                "primary_exchange": result.get("primary_exchange", ""),
                                "type": result.get("type", ""),
                                "active": result.get("active", True)
                            }
                        )

                return data.get("results")
            logger.warning(
                f"Failed to get ticker details for {ticker}: {data.get('error', 'Unknown error')}"
            )
            return None
        except Exception as e:
            logger.exception(f"Error getting ticker details for {ticker}: {e}")
            return None

    async def get_aggregates(
        self,
        ticker,
        multiplier=1,
        timespan="minute",
        from_date=None,
        to_date=None,
        limit=5000,
        adjusted=True,
    ):
        """
        Get aggregated data for a ticker

        Args:
            ticker: Stock ticker symbol
            multiplier: The size of the timespan multiplier
            timespan: The size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: The start date (format: YYYY-MM-DD)
            to_date: The end date (format: YYYY-MM-DD)
            limit: Maximum number of results to return
            adjusted: Whether the results are adjusted for splits

        Returns:
            DataFrame: Aggregated data
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Validate ticker
        if not ticker:
            logger.warning("Empty ticker provided")
            return pd.DataFrame()

        ticker = ticker.upper()  # Ensure ticker is uppercase for consistency

        # Check cache for DataFrame
        cache_key = [
            f"aggregates:{ticker}:{multiplier}:{timespan}:{from_date}:{to_date}:{adjusted}"
        ]
        cached_df = self.cache.get_dataframe(cache_key)
        if cached_df is not None and not cached_df.empty:
            logger.debug(f"Cache hit for aggregates: {ticker}")
            return cached_df

        # Prepare endpoint and parameters
        endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"limit": min(limit, 50000),
                  "adjusted": str(adjusted).lower()}

        # Make request
        data = await self._make_request(endpoint, params)

        # Process response
        if data.get("status") == "OK" and "results" in data and data["results"]:
            # Convert to DataFrame
            df = pd.DataFrame(data["results"])

            # Rename columns
            column_map = {
                "v": "volume",
                "o": "open",
                "c": "close",
                "h": "high",
                "l": "low",
                "t": "timestamp",
                "n": "transactions",
            }
            df = df.rename(columns=column_map)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Set timestamp as index
            df = df.set_index("timestamp")

            # Sort by timestamp
            df = df.sort_index()

            logger.info(f"Retrieved {len(df)} aggregates for {ticker}")

            # Store in Redis if available
            if self.redis_client:
                # Store in Redis hash map
                redis_key = f"stock:{ticker}:candles:{timespan}"

                # Store the most recent candles in Redis
                recent_df = df.tail(100)  # Store last 100 candles

                # Convert to dictionary for Redis
                candles_dict = {}
                for idx, row in recent_df.iterrows():
                    timestamp = int(idx.timestamp())
                    candle_data = {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "timestamp": timestamp,
                    }
                    candles_dict[timestamp] = json.dumps(candle_data)

                # Store in Redis
                if candles_dict:
                    # Use pipeline for efficiency
                    pipeline = self.redis_client.pipeline()
                    pipeline.delete(redis_key)
                    pipeline.hmset(redis_key, candles_dict)
                    pipeline.set(f"{redis_key}:last_update",
                                 datetime.now().isoformat())
                    pipeline.execute()

                    # Send notification to frontend
                    self._send_frontend_notification(
                        message=f"Retrieved {len(df)} price candles for {ticker}",
                        level="success",
                        category="market_data",
                        details={
                            "ticker": ticker,
                            "timespan": timespan,
                            "multiplier": multiplier,
                            "from_date": from_date,
                            "to_date": to_date,
                            "candle_count": len(df),
                            "latest_price": float(df["close"].iloc[-1]) if not df.empty else None,
                            "price_change": float(df["close"].iloc[-1] - df["close"].iloc[0]) if len(df) > 1 else 0,
                            "volume": int(df["volume"].sum()) if not df.empty else 0
                        }
                    )

            # Cache the DataFrame
            self.cache.store_dataframe(cache_key, df)

            return df
        if data.get("status") == "OK" and (
            "results" not in data or not data["results"]
        ):
            # No results found, but API call was successful
            logger.warning(
                f"No data found for {ticker} in the specified date range")
            return pd.DataFrame()
        # API call failed
        error_msg = (
            data.get("error", "Unknown error")
            if isinstance(data, dict)
            else "Unknown error"
        )
        logger.warning(f"Failed to get aggregates for {ticker}: {error_msg}")
        return pd.DataFrame()

    async def get_aggregates_batch(
        self,
        tickers,
        multiplier=1,
        timespan="minute",
        from_date=None,
        to_date=None,
        limit=5000,
        adjusted=True,
    ):
        """
        Get aggregated data for multiple tickers in parallel

        Args:
            tickers: List of stock ticker symbols
            multiplier: The size of the timespan multiplier
            timespan: The size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: The start date (format: YYYY-MM-DD)
            to_date: The end date (format: YYYY-MM-DD)
            limit: Maximum number of results to return
            adjusted: Whether the results are adjusted for splits

        Returns:
            dict: Dictionary of DataFrames, keyed by ticker
        """
        # Set default date range if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)
                         ).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")

        # Create tasks for each ticker
        tasks = []
        for ticker in tickers:
            tasks.append(
                self.get_aggregates(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_date=from_date,
                    to_date=to_date,
                    limit=limit,
                    adjusted=adjusted,
                ),
            )

        # Execute tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        data = {}
        for ticker, result in zip(tickers, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {ticker}: {result}")
                continue

            if not result.empty:
                data[ticker] = result

        # Send notification to frontend about batch completion
        if self.redis_client:
            self._send_frontend_notification(
                message=f"Completed batch data retrieval for {len(tickers)} tickers",
                level="success",
                category="market_data",
                details={
                    "tickers_requested": tickers,
                    "tickers_retrieved": list(data.keys()),
                    "timespan": timespan,
                    "multiplier": multiplier,
                    "from_date": from_date,
                    "to_date": to_date,
                    "success_rate": f"{len(data)}/{len(tickers)} ({int(len(data)/max(1, len(tickers))*100)}%)"
                }
            )

        return data

    async def process_data_with_gpu(self, data):
        """
        Process data using GPU acceleration

        Args:
            data: Dictionary of DataFrames, keyed by ticker

        Returns:
            dict: Dictionary of processed DataFrames
        """
        if not data:
            return {}

        if not self.gpu_accelerator.use_gpu or not self.gpu_accelerator.gpu_initialized:
            logger.warning(
                "GPU processing requested but GPU is not available or initialized",
            )

            # Send notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message="Falling back to CPU processing - GPU not available",
                    level="warning",
                    category="gpu_processing",
                    details={
                        "reason": "GPU not available or initialized",
                        "tickers": list(data.keys()),
                        "fallback": "CPU processing"
                    }
                )

            return self._process_data_with_cpu(data)

        results = {}

        for ticker, df in data.items():
            if df.empty:
                continue

            try:
                # Record start time for performance tracking
                start_time = time.time()

                # Convert to CuPy arrays for GPU processing
                if CUPY_AVAILABLE:
                    close_prices = cp.array(
                        df["close"].values, dtype=cp.float32)
                    volumes = cp.array(df["volume"].values, dtype=cp.float32)

                    # Calculate moving averages
                    window_5 = 5
                    window_20 = 20

                    # Use optimized algorithms for better performance
                    # Simple moving averages - use cumsum for faster
                    # calculation
                    padded_prices = cp.pad(
                        close_prices, (window_5 - 1, 0), "constant")
                    cumsum = cp.cumsum(padded_prices)
                    sma_5 = (cumsum[window_5:] - cumsum[:-window_5]) / window_5

                    padded_prices = cp.pad(
                        close_prices, (window_20 - 1, 0), "constant")
                    cumsum = cp.cumsum(padded_prices)
                    sma_20 = (cumsum[window_20:] -
                              cumsum[:-window_20]) / window_20

                    # Volume weighted average price
                    # Use parallel reduction for better performance
                    price_volume = close_prices * volumes
                    total_price_volume = cp.sum(price_volume)
                    total_volume = cp.sum(volumes)
                    vwap = total_price_volume / total_volume if total_volume > 0 else 0

                    # Calculate RSI
                    delta = cp.diff(close_prices)
                    gain = cp.where(delta > 0, delta, 0)
                    loss = cp.where(delta < 0, -delta, 0)

                    # Use exponential moving average for RSI
                    avg_gain = cp.mean(gain[:14]) if len(gain) >= 14 else 0
                    avg_loss = cp.mean(loss[:14]) if len(loss) >= 14 else 0

                    if len(gain) >= 14:
                        for i in range(14, len(gain)):
                            avg_gain = (avg_gain * 13 + gain[i]) / 14
                            avg_loss = (avg_loss * 13 + loss[i]) / 14

                    rs = avg_gain / avg_loss if avg_loss > 0 else 0
                    rsi = 100 - (100 / (1 + rs))

                    # Convert back to numpy arrays
                    sma_5_np = cp.asnumpy(sma_5)
                    sma_20_np = cp.asnumpy(sma_20)
                    vwap_np = float(cp.asnumpy(vwap))
                    rsi_np = float(cp.asnumpy(rsi))
                else:
                    # Fallback to CPU
                    return self._process_data_with_cpu(data)

                # Create result DataFrame
                result_df = pd.DataFrame(
                    {
                        "ticker": ticker,
                        "last_price": df["close"].iloc[-1],
                        "sma_5": sma_5_np[-1] if len(sma_5_np) > 0 else None,
                        "sma_20": sma_20_np[-1] if len(sma_20_np) > 0 else None,
                        "vwap": vwap_np,
                        "rsi": rsi_np,
                        "volume": df["volume"].sum(),
                    },
                    index=[0],
                )

                results[ticker] = result_df

                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.hmset(
                        f"stock:{ticker}:technical",
                        {
                            "sma_5": float(sma_5_np[-1]) if len(sma_5_np) > 0 else 0,
                            "sma_20": float(sma_20_np[-1]) if len(sma_20_np) > 0 else 0,
                            "vwap": float(vwap_np),
                            "rsi": float(rsi_np),
                            "last_update": datetime.now().isoformat(),
                        },
                    )

                    # Send notification to frontend about successful GPU processing
                    self._send_frontend_notification(
                        message=f"GPU-accelerated analysis completed for {ticker}",
                        level="success",
                        category="gpu_processing",
                        details={
                            "ticker": ticker,
                            "sma_5": float(sma_5_np[-1]) if len(sma_5_np) > 0 else 0,
                            "sma_20": float(sma_20_np[-1]) if len(sma_20_np) > 0 else 0,
                            "vwap": float(vwap_np),
                            "rsi": float(rsi_np),
                            "processing_time": time.time() - start_time if 'start_time' in locals() else 0
                        }
                    )

            except Exception as e:
                logger.exception(
                    f"Error processing data for {ticker} on GPU: {e}")

                # Send error notification to frontend
                if self.redis_client:
                    self._send_frontend_notification(
                        message=f"Error processing {ticker} data on GPU: {str(e)}",
                        level="error",
                        category="gpu_processing",
                        details={
                            "ticker": ticker,
                            "error_type": type(e).__name__,
                            "fallback": "CPU processing may be used"
                        }
                    )

        return results

    def _process_data_with_cpu(self, data):
        """
        Process data using CPU (fallback when GPU is not available)

        Args:
            data: Dictionary of DataFrames, keyed by ticker

        Returns:
            dict: Dictionary of processed DataFrames
        """
        results = {}

        for ticker, df in data.items():
            if df.empty:
                continue

            try:
                # Calculate indicators using pandas
                sma_5 = df["close"].rolling(
                    5).mean().iloc[-1] if len(df) >= 5 else None
                sma_20 = (
                    df["close"].rolling(20).mean(
                    ).iloc[-1] if len(df) >= 20 else None
                )

                # Calculate VWAP
                vwap = (
                    (df["close"] * df["volume"]).sum() / df["volume"].sum()
                    if df["volume"].sum() > 0
                    else 0
                )

                # Calculate RSI
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                avg_gain = gain.rolling(14).mean(
                ).iloc[-1] if len(gain) >= 14 else 0
                avg_loss = loss.rolling(14).mean(
                ).iloc[-1] if len(loss) >= 14 else 0

                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))

                # Create result DataFrame
                result_df = pd.DataFrame(
                    {
                        "ticker": ticker,
                        "last_price": df["close"].iloc[-1],
                        "sma_5": sma_5,
                        "sma_20": sma_20,
                        "vwap": vwap,
                        "rsi": rsi,
                        "volume": df["volume"].sum(),
                    },
                    index=[0],
                )

                results[ticker] = result_df

                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.hmset(
                        f"stock:{ticker}:technical",
                        {
                            "sma_5": float(sma_5) if sma_5 is not None else 0,
                            "sma_20": float(sma_20) if sma_20 is not None else 0,
                            "vwap": float(vwap),
                            "rsi": float(rsi),
                            "last_update": datetime.now().isoformat(),
                        },
                    )

                    # Send notification to frontend about market data processing
                    # Only send periodically to avoid flooding
                    if random.random() < 0.1:  # 10% chance to send notification
                        self._send_frontend_notification(
                            message=f"Processed market data for {ticker}",
                            level="info",
                            category="market_data_stream",
                            details={
                                "ticker": ticker,
                                "sma_5": float(sma_5) if sma_5 is not None else 0,
                                "sma_20": float(sma_20) if sma_20 is not None else 0,
                                "vwap": float(vwap),
                                "rsi": float(rsi),
                                "timestamp": datetime.now().isoformat()
                            }
                        )

            except Exception as e:
                logger.exception(
                    f"Error processing data for {ticker} on CPU: {e}")

                # Send error notification to frontend
                if self.redis_client:
                    self._send_frontend_notification(
                        message=f"Error processing market data for {ticker}",
                        level="error",
                        category="market_data_stream",
                        details={
                            "ticker": ticker,
                            "error_type": type(e).__name__,
                            "error": str(e)
                        }
                    )

        return results

    async def close(self) -> None:
        """Close all connections and resources"""
        logger.info("Closing Polygon API client")
        self.running = False

        # Cancel all scheduled tasks
        for name, task in self.scheduled_tasks.items():
            if not task.done():
                logger.info(f"Cancelling scheduled task: {name}")
                task.cancel()

        # Close connection pool
        await self.connection_pool.close()

        # Clean up GPU resources
        self.gpu_accelerator.clear_memory()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)

        logger.info("Polygon API client closed")


##############################
# POLYGON WEBSOCKET CLIENT #
##############################


class PolygonWebSocketClient:
    """GPU-Optimized client for the Polygon.io WebSocket API"""

    def __init__(
        self,
        api_key=POLYGON_API_KEY,
        redis_client=None,
        use_gpu=USE_GPU,
        max_reconnect_attempts=MAX_RECONNECT_ATTEMPTS,
        reconnect_delay=RECONNECT_DELAY,
        buffer_size=BUFFER_SIZE,
    ) -> None:
        """
        Initialize GPU-Optimized Polygon WebSocket client

        Args:
            api_key: API key for authentication
            redis_client: Optional Redis client for data storage
            use_gpu: Whether to use GPU acceleration
            max_reconnect_attempts: Maximum number of reconnection attempts
            reconnect_delay: Initial delay between reconnection attempts (will use exponential backoff)
            buffer_size: Number of messages to buffer before batch processing
        """
        self.api_key = api_key
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.use_gpu = use_gpu
        self.buffer_size = buffer_size
        self.redis_client = redis_client

        # Verify API key is provided
        if not self.api_key:
            logger.warning(
                "No API key provided. Set the POLYGON_API_KEY environment variable.",
            )
            logger.error(
                "API key is required for Polygon.io WebSocket connection")
        else:
            logger.info(
                f"Initialized Polygon WebSocket client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}",
            )

        # WebSocket connection
        self.ws = None
        self.ws_url = "wss://socket.polygon.io/stocks"

        # Subscription tracking
        self.subscriptions = set()

        # Message handlers
        self.message_handlers = {}

        # Connection status
        self.connected = False
        self.reconnect_count = 0
        self.last_heartbeat = 0

        # Control flags
        self.running = False
        self.heartbeat_task = None
        self.event_loop = None
        self.main_task = None
        self.background_thread = None

        # Message buffers for batch processing
        self.trade_buffer = []
        self.quote_buffer = []
        self.agg_buffer = []
        self.buffer_lock = threading.Lock()

        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Initialize GPU accelerator
        self.gpu_accelerator = GPUAccelerator(use_gpu=use_gpu)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.stop()

    def _send_frontend_notification(self, message, level="info", category="websocket", details=None):
        """
        Send notification to frontend via Redis

        Args:
            message: Notification message
            level: Notification level (info, success, warning, error)
            category: Notification category
            details: Additional details as dictionary
        """
        if not self.redis_client:
            return

        try:
            notification = {
                "timestamp": time.time(),
                "message": message,
                "level": level,
                "category": category,
                "source": "polygon_websocket",
                "details": details or {}
            }

            # Add to notifications list
            self.redis_client.lpush(
                "frontend:notifications", json.dumps(notification))

            # Trim list to prevent unbounded growth
            self.redis_client.ltrim("frontend:notifications", 0, 999)

            # Publish event for real-time updates
            self.redis_client.publish("frontend:events", json.dumps({
                "type": "notification",
                "data": notification
            }))

            logger.debug(f"Sent frontend notification: {message}")
        except Exception as e:
            logger.error(f"Error sending frontend notification: {e}")

    async def _connect(self) -> bool | None:
        """Establish WebSocket connection"""
        if not self.api_key:
            logger.error("Cannot connect: No API key provided")
            return False

        try:
            logger.info(f"Connecting to {self.ws_url}...")
            # Connect with API key in the URL path
            connection_url = f"wss://socket.polygon.io/stocks?apiKey={self.api_key}"
            self.ws = await websockets.connect(connection_url)
            logger.info("WebSocket connection established")

            # Authenticate
            await self._authenticate()

            # Wait for auth response
            response = await self.ws.recv()
            response_data = json.loads(response)

            if (
                isinstance(response_data, list)
                and response_data
                and response_data[0].get("status") == "connected"
            ):
                logger.info("Authentication successful")
                self.connected = True
                self.reconnect_count = 0
                self.last_heartbeat = time.time()

                # Send notification to frontend
                if self.redis_client:
                    self._send_frontend_notification(
                        message="Connected to Polygon WebSocket API",
                        level="success",
                        category="websocket_connection",
                        details={
                            "connection_url": self.ws_url,
                            "subscription_count": len(self.subscriptions),
                            "reconnect_count": self.reconnect_count
                        }
                    )

                # Resubscribe to previous subscriptions
                if self.subscriptions:
                    await self._resubscribe()

                return True
            logger.error(f"Authentication failed: {response_data}")

            # Send notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message="Failed to authenticate with Polygon WebSocket API",
                    level="error",
                    category="websocket_connection",
                    details={
                        "error": str(response_data),
                        "connection_url": self.ws_url
                    }
                )

            return False

        except Exception as e:
            logger.exception(f"Connection error: {e}")
            self.connected = False

            # Send notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message=f"WebSocket connection error: {str(e)}",
                    level="error",
                    category="websocket_connection",
                    details={
                        "error_type": type(e).__name__,
                        "connection_url": self.ws_url,
                        "reconnect_count": self.reconnect_count
                    }
                )

            return False

    async def _authenticate(self) -> None:
        """Authenticate with the Polygon WebSocket API"""
        try:
            # Send authentication message
            auth_message = {"action": "auth", "params": self.api_key}
            await self.ws.send(json.dumps(auth_message))
            logger.info("Sent authentication request")
        except Exception as e:
            logger.exception(f"Authentication error: {e}")
            raise

    async def _resubscribe(self) -> None:
        """Resubscribe to all previous subscriptions"""
        if not self.subscriptions:
            return

        logger.info(f"Resubscribing to {len(self.subscriptions)} channels...")

        # Group subscriptions by cluster (stocks, options, forex, crypto)
        subscriptions_by_cluster = {}
        for sub in self.subscriptions:
            parts = sub.split(".")
            if len(parts) >= 2:
                cluster = parts[0]
                if cluster not in subscriptions_by_cluster:
                    subscriptions_by_cluster[cluster] = []
                subscriptions_by_cluster[cluster].append(sub)

        # Subscribe to each cluster
        for cluster, subs in subscriptions_by_cluster.items():
            subscribe_message = {
                "action": "subscribe", "params": ",".join(subs)}

            try:
                await self.ws.send(json.dumps(subscribe_message))
                logger.info(f"Resubscribed to {len(subs)} {cluster} channels")

                # Send notification to frontend
                if self.redis_client:
                    self._send_frontend_notification(
                        message=f"Resubscribed to {len(subs)} {cluster} channels",
                        level="info",
                        category="websocket_subscription",
                        details={
                            "cluster": cluster,
                            "channel_count": len(subs),
                            "channels": subs[:10] + (["..."] if len(subs) > 10 else [])
                        }
                    )

            except Exception as e:
                logger.exception(
                    f"Error resubscribing to {cluster} channels: {e}")

                # Send notification to frontend
                if self.redis_client:
                    self._send_frontend_notification(
                        message=f"Failed to resubscribe to {cluster} channels",
                        level="error",
                        category="websocket_subscription",
                        details={
                            "cluster": cluster,
                            "error_type": type(e).__name__,
                            "error": str(e)
                        }
                    )

    async def _heartbeat(self) -> None:
        """Send heartbeat messages and monitor connection health"""
        while self.running and self.connected:
            try:
                # Check if we've received a heartbeat recently
                if (
                    time.time() - self.last_heartbeat > 30
                ):  # No heartbeat for 30 seconds
                    logger.warning(
                        "No heartbeat received for 30 seconds, reconnecting...",
                    )
                    self.connected = False
                    await self._reconnect()
                    continue

                # Sleep for a while
                await asyncio.sleep(15)

            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in heartbeat task: {e}")
                await asyncio.sleep(5)

    async def _reconnect(self):
        """Attempt to reconnect to the WebSocket"""
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.error(
                f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached"
            )
            self.running = False
            return False

        self.reconnect_count += 1
        delay = self.reconnect_delay * (
            2 ** (self.reconnect_count - 1)
        )  # Exponential backoff

        logger.info(
            f"Attempting to reconnect (attempt {self.reconnect_count}/{self.max_reconnect_attempts}) in {delay:.2f} seconds...",
        )

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            API_WEBSOCKET_RECONNECTS.labels(
                client="polygon", endpoint="stocks").inc()

        try:
            # Close existing connection if any
            if self.ws:
                await self.ws.close()

            # Wait before reconnecting
            await asyncio.sleep(delay)

            # Attempt to reconnect
            return await self._connect()

        except Exception as e:
            logger.exception(f"Reconnection error: {e}")
            # Update error metrics
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    client="polygon",
                    endpoint="websocket",
                    method="reconnect",
                    error_type=type(e).__name__,
                ).inc()
            return False

    async def _listen(self) -> None:
        """Listen for WebSocket messages"""
        while self.running:
            try:
                if not self.connected:
                    success = await self._connect()
                    if not success:
                        await self._reconnect()
                        continue

                # Receive message
                message = await self.ws.recv()

                # Update heartbeat timestamp
                self.last_heartbeat = time.time()

                # Parse message
                try:
                    data = json.loads(message)

                    # Handle message based on type
                    if isinstance(data, list):
                        for item in data:
                            await self._process_message(item)
                    else:
                        await self._process_message(data)

                except json.JSONDecodeError:
                    logger.warning(
                        f"Received invalid JSON: {message[:100]}...")

            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                await self._reconnect()

            except asyncio.CancelledError:
                logger.info("Listen task cancelled")
                break

            except Exception as e:
                logger.exception(f"Error in listen task: {e}")
                self.connected = False
                await asyncio.sleep(1)

    async def _process_message(self, message) -> None:
        """Process a WebSocket message"""
        # Check for status messages
        if "status" in message:
            status = message.get("status")

            if status == "connected":
                logger.info("Connected to Polygon WebSocket")
                if self.redis_client:
                    self._send_frontend_notification(
                        message="Connected to Polygon WebSocket",
                        level="success",
                        category="websocket_status",
                        details={"status": "connected"}
                    )

            elif status == "auth_success":
                logger.info("Authentication successful")
                if self.redis_client:
                    self._send_frontend_notification(
                        message="Authentication successful with Polygon WebSocket",
                        level="success",
                        category="websocket_status",
                        details={"status": "authenticated"}
                    )

            elif status == "success":
                if message.get("message") == "authenticated":
                    logger.info("Authentication successful")
                    if self.redis_client:
                        self._send_frontend_notification(
                            message="Authentication successful with Polygon WebSocket",
                            level="success",
                            category="websocket_status",
                            details={"status": "authenticated"}
                        )
                else:
                    action = message.get("action")
                    if action == "subscribe":
                        params = message.get('params', '')
                        logger.info(f"Successfully subscribed to: {params}")
                        if self.redis_client:
                            self._send_frontend_notification(
                                message=f"Successfully subscribed to market data channels",
                                level="success",
                                category="websocket_subscription",
                                details={
                                    "channels": params,
                                    "action": "subscribe"
                                }
                            )
                    elif action == "unsubscribe":
                        params = message.get('params', '')
                        logger.info(
                            f"Successfully unsubscribed from: {params}")
                        if self.redis_client:
                            self._send_frontend_notification(
                                message=f"Successfully unsubscribed from market data channels",
                                level="info",
                                category="websocket_subscription",
                                details={
                                    "channels": params,
                                    "action": "unsubscribe"
                                }
                            )

            elif status == "error":
                error_msg = message.get('message', '')
                logger.error(f"Error from Polygon WebSocket: {error_msg}")
                if self.redis_client:
                    self._send_frontend_notification(
                        message=f"Error from Polygon WebSocket: {error_msg}",
                        level="error",
                        category="websocket_error",
                        details={
                            "error": error_msg,
                            "raw_message": message
                        }
                    )

            return

        # Handle data messages
        event_type = message.get("ev")
        if not event_type:
            logger.warning(f"Received message without event type: {message}")
            return

        # Buffer messages for batch processing
        with self.buffer_lock:
            if event_type == "T":  # Trade
                self.trade_buffer.append(message)
                if len(self.trade_buffer) >= self.buffer_size:
                    # Process trade buffer
                    trades_to_process = self.trade_buffer.copy()
                    self.trade_buffer = []
                    self.thread_pool.submit(
                        self._process_trade_batch,
                        trades_to_process,
                    )

            elif event_type == "Q":  # Quote
                self.quote_buffer.append(message)
                if len(self.quote_buffer) >= self.buffer_size:
                    # Process quote buffer
                    quotes_to_process = self.quote_buffer.copy()
                    self.quote_buffer = []
                    self.thread_pool.submit(
                        self._process_quote_batch,
                        quotes_to_process,
                    )

            elif event_type in ["AM", "A"]:  # Aggregates
                self.agg_buffer.append(message)
                if len(self.agg_buffer) >= self.buffer_size:
                    # Process aggregate buffer
                    aggs_to_process = self.agg_buffer.copy()
                    self.agg_buffer = []
                    self.thread_pool.submit(
                        self._process_agg_batch, aggs_to_process)

        # Call appropriate handler
        if event_type in self.message_handlers:
            for handler in self.message_handlers[event_type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.exception(
                        f"Error in message handler for {event_type}: {e}")

    def _process_trade_batch(self, trades) -> None:
        """Process a batch of trade messages with GPU acceleration"""
        if not trades:
            return

        try:
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                API_WEBSOCKET_MESSAGES.labels(
                    client="polygon",
                    message_type="trade",
                ).inc(len(trades))

            # Group trades by ticker
            trades_by_ticker = {}
            for trade in trades:
                ticker = trade.get("sym")
                if ticker not in trades_by_ticker:
                    trades_by_ticker[ticker] = []
                trades_by_ticker[ticker].append(trade)

            # Process each ticker's trades
            for ticker, ticker_trades in trades_by_ticker.items():
                # Extract trade data
                prices = [t.get("p", 0) for t in ticker_trades]
                sizes = [t.get("s", 0) for t in ticker_trades]
                # timestamps not used directly but extracted for completeness

                # Record GPU processing start time
                gpu_start_time = time.time()

                # Use GPU for calculations if available
                if (
                    self.gpu_accelerator.use_gpu
                    and self.gpu_accelerator.gpu_initialized
                ):
                    if CUPY_AVAILABLE:
                        # Convert to CuPy arrays
                        cp_prices = cp.array(prices, dtype=cp.float32)
                        cp_sizes = cp.array(sizes, dtype=cp.float32)

                        # Calculate VWAP
                        price_volume = cp_prices * cp_sizes
                        total_price_volume = cp.sum(price_volume)
                        total_volume = cp.sum(cp_sizes)
                        vwap = (
                            total_price_volume / total_volume if total_volume > 0 else 0
                        )

                        # Calculate statistics
                        avg_price = cp.mean(cp_prices)
                        max_price = cp.max(cp_prices)
                        min_price = cp.min(cp_prices)

                        # Convert back to numpy
                        vwap = float(cp.asnumpy(vwap))
                        avg_price = float(cp.asnumpy(avg_price))
                        max_price = float(cp.asnumpy(max_price))
                        min_price = float(cp.asnumpy(min_price))

                        # Update GPU memory usage metrics
                        if PROMETHEUS_AVAILABLE:
                            device_id = cp.cuda.Device().id
                            device_props = cp.cuda.runtime.getDeviceProperties(
                                device_id,
                            )
                            device_name = device_props["name"].decode()
                            mem_info = cp.cuda.runtime.memGetInfo()
                            # total - free = used
                            mem_used = mem_info[1] - mem_info[0]
                            GPU_MEMORY_USAGE.labels(
                                device=device_name).set(mem_used)
                    else:
                        # CPU calculations as fallback
                        price_volume = np.array(prices) * np.array(sizes)
                        total_price_volume = np.sum(price_volume)
                        total_volume = np.sum(sizes)
                        vwap = (
                            total_price_volume / total_volume if total_volume > 0 else 0
                        )

                        avg_price = np.mean(prices)
                        max_price = np.max(prices)
                        min_price = np.min(prices)
                else:
                    # CPU calculations
                    price_volume = np.array(prices) * np.array(sizes)
                    total_price_volume = np.sum(price_volume)
                    total_volume = np.sum(sizes)
                    vwap = total_price_volume / total_volume if total_volume > 0 else 0

                    avg_price = np.mean(prices)
                    max_price = np.max(prices)
                    min_price = np.min(prices)

                # Record GPU processing time
                gpu_processing_time = time.time() - gpu_start_time
                if PROMETHEUS_AVAILABLE:
                    GPU_PROCESSING_TIME.labels(operation="trade_processing").observe(
                        gpu_processing_time,
                    )

                # Get latest trade
                latest_trade = max(ticker_trades, key=lambda t: t.get("t", 0))
                latest_price = latest_trade.get("p", 0)
                latest_size = latest_trade.get("s", 0)
                latest_timestamp = latest_trade.get("t", 0)

                # Store in Redis if available
                if self.redis_client:
                    # Update last trade
                    self.redis_client.hmset(
                        f"stock:{ticker}:last_trade",
                        {
                            "price": latest_price,
                            "size": latest_size,
                            "timestamp": latest_timestamp,
                            "exchange": latest_trade.get("x", ""),
                        },
                    )

                    # Update last price
                    self.redis_client.hmset(
                        f"stock:{ticker}:last_price",
                        {"price": latest_price, "timestamp": latest_timestamp},
                    )

                    # Store trade statistics
                    self.redis_client.hmset(
                        f"stock:{ticker}:trade_stats",
                        {
                            "vwap": vwap,
                            "avg_price": avg_price,
                            "max_price": max_price,
                            "min_price": min_price,
                            "total_volume": int(total_volume),
                            "trade_count": len(ticker_trades),
                            "last_update": datetime.now().isoformat(),
                        },
                    )

                    # Publish update to subscribers
                    self.redis_client.publish(
                        f"price_update:{ticker}",
                        json.dumps(
                            {
                                "type": "trade",
                                "ticker": ticker,
                                "price": latest_price,
                                "size": latest_size,
                                "timestamp": latest_timestamp,
                            },
                        ),
                    )

        except Exception as e:
            logger.exception(f"Error processing trade batch: {e}")
            # Update error metrics
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    client="polygon",
                    endpoint="websocket",
                    method="trade_processing",
                    error_type=type(e).__name__,
                ).inc()

    def _process_quote_batch(self, quotes) -> None:
        """Process a batch of quote messages with GPU acceleration"""
        if not quotes:
            return

        try:
            # Group quotes by ticker
            quotes_by_ticker = {}
            for quote in quotes:
                ticker = quote.get("sym")
                if ticker not in quotes_by_ticker:
                    quotes_by_ticker[ticker] = []
                quotes_by_ticker[ticker].append(quote)

            # Process each ticker's quotes
            for ticker, ticker_quotes in quotes_by_ticker.items():
                # Extract quote data
                bid_prices = [q.get("bp", 0) for q in ticker_quotes]
                # bid_sizes extracted but used in Redis storage later
                [q.get("bs", 0) for q in ticker_quotes]
                ask_prices = [q.get("ap", 0) for q in ticker_quotes]
                # ask_sizes extracted but used in Redis storage later
                [q.get("as", 0) for q in ticker_quotes]
                # timestamps not used directly but extracted for completeness

                # Use GPU for calculations if available
                if (
                    self.gpu_accelerator.use_gpu
                    and self.gpu_accelerator.gpu_initialized
                ):
                    if CUPY_AVAILABLE:
                        # Convert to CuPy arrays
                        cp_bid_prices = cp.array(bid_prices, dtype=cp.float32)
                        cp_ask_prices = cp.array(ask_prices, dtype=cp.float32)

                        # Calculate mid prices
                        cp_mid_prices = (cp_bid_prices + cp_ask_prices) / 2

                        # Calculate spreads
                        cp_spreads = cp_ask_prices - cp_bid_prices

                        # Calculate statistics
                        avg_bid = cp.mean(cp_bid_prices)
                        avg_ask = cp.mean(cp_ask_prices)
                        avg_mid = cp.mean(cp_mid_prices)
                        avg_spread = cp.mean(cp_spreads)

                        # Convert back to numpy
                        avg_bid = float(cp.asnumpy(avg_bid))
                        avg_ask = float(cp.asnumpy(avg_ask))
                        avg_mid = float(cp.asnumpy(avg_mid))
                        avg_spread = float(cp.asnumpy(avg_spread))
                    else:
                        # Fallback to CPU calculations
                        mid_prices = [
                            (b + a) / 2
                            for b, a in zip(bid_prices, ask_prices, strict=False)
                        ]
                        spreads = [
                            a - b for a, b in zip(ask_prices, bid_prices, strict=False)
                        ]

                        avg_bid = np.mean(bid_prices)
                        avg_ask = np.mean(ask_prices)
                        avg_mid = np.mean(mid_prices)
                        avg_spread = np.mean(spreads)
                else:
                    # CPU calculations
                    mid_prices = [
                        (b + a) / 2
                        for b, a in zip(bid_prices, ask_prices, strict=False)
                    ]
                    spreads = [
                        a - b for a, b in zip(ask_prices, bid_prices, strict=False)
                    ]

                    avg_bid = np.mean(bid_prices)
                    avg_ask = np.mean(ask_prices)
                    avg_mid = np.mean(mid_prices)
                    avg_spread = np.mean(spreads)

                # Get latest quote
                latest_quote = max(ticker_quotes, key=lambda q: q.get("t", 0))
                latest_bid = latest_quote.get("bp", 0)
                latest_ask = latest_quote.get("ap", 0)
                latest_bid_size = latest_quote.get("bs", 0)
                latest_ask_size = latest_quote.get("as", 0)
                latest_timestamp = latest_quote.get("t", 0)
                latest_mid = (
                    (latest_bid + latest_ask) / 2
                    if latest_bid > 0 and latest_ask > 0
                    else 0
                )
                latest_spread = (
                    latest_ask - latest_bid if latest_bid > 0 and latest_ask > 0 else 0
                )

                # Store in Redis if available
                if self.redis_client:
                    # Update last quote
                    self.redis_client.hmset(
                        f"stock:{ticker}:last_quote",
                        {
                            "bid_price": latest_bid,
                            "bid_size": latest_bid_size,
                            "ask_price": latest_ask,
                            "ask_size": latest_ask_size,
                            "mid_price": latest_mid,
                            "spread": latest_spread,
                            "timestamp": latest_timestamp,
                        },
                    )

                    # Update last price
                    self.redis_client.hmset(
                        f"stock:{ticker}:last_price",
                        {
                            "bid": latest_bid,
                            "ask": latest_ask,
                            "mid": latest_mid,
                            "timestamp": latest_timestamp,
                        },
                    )

                    # Publish update to subscribers
                    self.redis_client.publish(
                        f"price_update:{ticker}",
                        json.dumps(
                            {
                                "type": "quote",
                                "ticker": ticker,
                                "bid": latest_bid,
                                "ask": latest_ask,
                                "mid": latest_mid,
                                "timestamp": latest_timestamp,
                            },
                        ),
                    )

        except Exception as e:
            logger.exception(f"Error processing quote batch: {e}")

    def _process_agg_batch(self, aggs) -> None:
        """Process a batch of aggregate messages with GPU acceleration"""
        if not aggs:
            return

        try:
            # Group aggregates by ticker
            aggs_by_ticker = {}
            for agg in aggs:
                ticker = agg.get("sym")
                if ticker not in aggs_by_ticker:
                    aggs_by_ticker[ticker] = []
                aggs_by_ticker[ticker].append(agg)

            # Process each ticker's aggregates
            for ticker, ticker_aggs in aggs_by_ticker.items():
                # We don't need to extract all aggregate data here since we're just
                # using the latest aggregate for Redis storage

                # Get latest aggregate
                latest_agg = max(ticker_aggs, key=lambda a: a.get("s", 0))
                latest_open = latest_agg.get("o", 0)
                latest_high = latest_agg.get("h", 0)
                latest_low = latest_agg.get("l", 0)
                latest_close = latest_agg.get("c", 0)
                latest_volume = latest_agg.get("v", 0)
                latest_timestamp = latest_agg.get("s", 0)

                # Store in Redis if available
                if self.redis_client:
                    # Store latest candle
                    self.redis_client.hmset(
                        f"stock:{ticker}:latest_candle",
                        {
                            "open": latest_open,
                            "high": latest_high,
                            "low": latest_low,
                            "close": latest_close,
                            "volume": latest_volume,
                            "timestamp": latest_timestamp,
                        },
                    )

                    # Store in candles hash
                    timespan = "minute" if latest_agg.get(
                        "ev") == "AM" else "second"
                    candle_key = f"stock:{ticker}:candles:{timespan}"

                    # Add each candle to the hash
                    for agg in ticker_aggs:
                        timestamp = agg.get("s", 0)
                        candle_data = {
                            "open": agg.get("o", 0),
                            "high": agg.get("h", 0),
                            "low": agg.get("l", 0),
                            "close": agg.get("c", 0),
                            "volume": agg.get("v", 0),
                            "timestamp": timestamp,
                        }
                        self.redis_client.hset(
                            candle_key,
                            timestamp,
                            json.dumps(candle_data),
                        )

                    # Publish update to subscribers
                    self.redis_client.publish(
                        f"candle_update:{ticker}",
                        json.dumps(
                            {
                                "type": "candle",
                                "ticker": ticker,
                                "timespan": timespan,
                                "open": latest_open,
                                "high": latest_high,
                                "low": latest_low,
                                "close": latest_close,
                                "volume": latest_volume,
                                "timestamp": latest_timestamp,
                            },
                        ),
                    )

        except Exception as e:
            logger.exception(f"Error processing aggregate batch: {e}")

    def start(self) -> None:
        """Start the WebSocket client in a background thread"""
        if self.running:
            logger.warning("WebSocket client is already running")
            return

        self.running = True

        def run_event_loop() -> None:
            """Run the event loop in a background thread"""
            try:
                # Create new event loop for this thread
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)

                # Create tasks
                self.main_task = self.event_loop.create_task(self._listen())
                self.heartbeat_task = self.event_loop.create_task(
                    self._heartbeat())

                # Run the event loop
                self.event_loop.run_until_complete(
                    asyncio.gather(self.main_task, self.heartbeat_task),
                )

            except Exception as e:
                logger.exception(f"Error in WebSocket thread: {e}")

            finally:
                # Clean up
                if self.event_loop and not self.event_loop.is_closed():
                    self.event_loop.close()
                logger.info("WebSocket thread stopped")

        # Start background thread
        self.background_thread = threading.Thread(target=run_event_loop)
        self.background_thread.daemon = True
        self.background_thread.start()

        logger.info("WebSocket client started")

    def is_connected(self):
        """
        Check if the WebSocket client is connected

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected

    def stop(self) -> None:
        """Stop the WebSocket client"""
        if not self.running:
            logger.warning("WebSocket client is not running, nothing to stop")
            return

        logger.info("Stopping WebSocket client - setting running flag to False")
        self.running = False

        # Cancel tasks
        if self.event_loop and not self.event_loop.is_closed():
            logger.info(
                "Creating task to cancel WebSocket tasks in event loop")
            try:
                # Create a future and wait for it to complete
                future = asyncio.run_coroutine_threadsafe(
                    self._cancel_tasks(),
                    self.event_loop,
                )
                # Wait for the future to complete with a timeout
                future.result(timeout=5)
                logger.info("WebSocket tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for WebSocket tasks to cancel")
            except Exception as e:
                logger.exception(f"Error cancelling WebSocket tasks: {e}")
            else:
                logger.info("WebSocket tasks cancelled successfully")
        else:
            logger.warning("Cannot cancel tasks: event loop is closed or None")

        # Wait for background thread to stop
        if self.background_thread and self.background_thread.is_alive():
            logger.info("Waiting for background thread to stop (timeout: 5s)")
            self.background_thread.join(timeout=5)
            if self.background_thread.is_alive():
                logger.warning("Background thread did not stop within timeout")
            else:
                logger.info("Background thread stopped successfully")
        else:
            logger.info("No background thread to stop")

        # Process remaining buffered messages
        logger.info("Processing remaining buffered messages")
        with self.buffer_lock:
            if self.trade_buffer:
                logger.info(
                    f"Processing {len(self.trade_buffer)} remaining trade messages",
                )
                self._process_trade_batch(self.trade_buffer)
                self.trade_buffer = []

            if self.quote_buffer:
                logger.info(
                    f"Processing {len(self.quote_buffer)} remaining quote messages",
                )
                self._process_quote_batch(self.quote_buffer)
                self.quote_buffer = []

            if self.agg_buffer:
                logger.info(
                    f"Processing {len(self.agg_buffer)} remaining aggregate messages",
                )
                self._process_agg_batch(self.agg_buffer)
                self.agg_buffer = []

        # Clean up GPU resources
        self.gpu_accelerator.clear_memory()

        # Shutdown thread pool
        logger.info("Shutting down thread pool")
        self.thread_pool.shutdown(wait=False)

        logger.info("WebSocket client stopped")

    async def _cancel_tasks(self) -> None:
        """Cancel all tasks"""
        logger.info("Starting to cancel tasks in WebSocket client")

        # Track tasks to wait for
        tasks_to_wait = []

        if self.main_task:
            logger.info("Cancelling main task")
            try:
                self.main_task.cancel()
                tasks_to_wait.append(self.main_task)
            except Exception as e:
                logger.exception(f"Error cancelling main task: {e}")

        if self.heartbeat_task:
            logger.info("Cancelling heartbeat task")
            try:
                self.heartbeat_task.cancel()
                tasks_to_wait.append(self.heartbeat_task)
            except Exception as e:
                logger.exception(f"Error cancelling heartbeat task: {e}")

        # Wait for tasks to be cancelled with a timeout
        if tasks_to_wait:
            try:
                logger.info(
                    f"Waiting for {len(tasks_to_wait)} tasks to be cancelled (with 2s timeout)"
                )
                # Wait for tasks to be cancelled with a timeout
                done, pending = await asyncio.wait(tasks_to_wait, timeout=2.0)
                if pending:
                    logger.warning(
                        f"{len(pending)} tasks still pending after timeout")
                else:
                    logger.info("All tasks cancelled successfully")
            except Exception as e:
                logger.exception(
                    f"Error waiting for tasks to be cancelled: {e}")

        if self.ws:
            logger.info("Closing WebSocket connection")
            try:
                # Close with a timeout
                await asyncio.wait_for(self.ws.close(), timeout=2.0)
                logger.info("WebSocket connection closed successfully")
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for WebSocket connection to close")
            except Exception as e:
                logger.exception(f"Error closing WebSocket connection: {e}")
        else:
            logger.info("No WebSocket connection to close")

        self.connected = False
        logger.info("All tasks canceled and connections closed")

    def subscribe(self, channels) -> None:
        """
        Subscribe to WebSocket channels

        Args:
            channels: Channel or list of channels to subscribe to
        """
        if isinstance(channels, str):
            channels = [channels]

        # Add to subscription set
        new_channels = []
        for channel in channels:
            if channel not in self.subscriptions:
                self.subscriptions.add(channel)
                new_channels.append(channel)

        if not new_channels:
            logger.info("No new channels to subscribe to")
            return

        # Send subscription message if connected
        if self.connected and self.event_loop and not self.event_loop.is_closed():
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(new_channels),
            }

            async def send_subscribe() -> None:
                try:
                    await self.ws.send(json.dumps(subscribe_message))
                    logger.info(
                        f"Subscribed to channels: {', '.join(new_channels)}")
                except Exception as e:
                    logger.exception(f"Error subscribing to channels: {e}")

            asyncio.run_coroutine_threadsafe(send_subscribe(), self.event_loop)
        else:
            logger.info(
                f"Added channels to subscription list (will subscribe when connected): {', '.join(new_channels)}",
            )

    def unsubscribe(self, channels) -> None:
        """
        Unsubscribe from WebSocket channels

        Args:
            channels: Channel or list of channels to unsubscribe from
        """
        if isinstance(channels, str):
            channels = [channels]

        # Remove from subscription set
        removed_channels = []
        for channel in channels:
            if channel in self.subscriptions:
                self.subscriptions.remove(channel)
                removed_channels.append(channel)

        if not removed_channels:
            logger.info("No channels to unsubscribe from")
            return

        # Send unsubscription message if connected
        if self.connected and self.event_loop and not self.event_loop.is_closed():
            unsubscribe_message = {
                "action": "unsubscribe",
                "params": ",".join(removed_channels),
            }

            async def send_unsubscribe() -> None:
                try:
                    await self.ws.send(json.dumps(unsubscribe_message))
                    logger.info(
                        f"Unsubscribed from channels: {', '.join(removed_channels)}",
                    )
                except Exception as e:
                    logger.exception(f"Error unsubscribing from channels: {e}")

            asyncio.run_coroutine_threadsafe(
                send_unsubscribe(), self.event_loop)
        else:
            logger.info(
                f"Removed channels from subscription list: {', '.join(removed_channels)}"
            )

    def add_message_handler(self, event_type, handler) -> None:
        """
        Add a message handler for a specific event type

        Args:
            event_type: Event type to handle (e.g., 'T' for trades, 'Q' for quotes)
            handler: Callback function to handle the message
        """
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []

        self.message_handlers[event_type].append(handler)
        logger.info(f"Added message handler for event type: {event_type}")

    def remove_message_handler(self, event_type, handler) -> None:
        """
        Remove a message handler for a specific event type

        Args:
            event_type: Event type to handle (e.g., 'T' for trades, 'Q' for quotes)
            handler: Callback function to remove
        """
        if event_type in self.message_handlers:
            if handler in self.message_handlers[event_type]:
                self.message_handlers[event_type].remove(handler)
                logger.info(
                    f"Removed message handler for event type: {event_type}")

            if not self.message_handlers[event_type]:
                del self.message_handlers[event_type]


# Helper functions for common subscriptions


def subscribe_to_trades(client, tickers) -> None:
    """
    Subscribe to trade events for specific tickers

    Args:
        client: PolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"T.{ticker}" for ticker in tickers]
    client.subscribe(channels)


def subscribe_to_quotes(client, tickers) -> None:
    """
    Subscribe to quote events for specific tickers

    Args:
        client: PolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"Q.{ticker}" for ticker in tickers]
    client.subscribe(channels)


def subscribe_to_minute_aggs(client, tickers) -> None:
    """
    Subscribe to minute aggregates for specific tickers

    Args:
        client: PolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"AM.{ticker}" for ticker in tickers]
    client.subscribe(channels)


def subscribe_to_second_aggs(client, tickers) -> None:
    """
    Subscribe to second aggregates for specific tickers

    Args:
        client: PolygonWebSocketClient instance
        tickers: Ticker or list of tickers to subscribe to
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    channels = [f"A.{ticker}" for ticker in tickers]
    client.subscribe(channels)


##############################
# UNUSUAL WHALES API CLIENT #
##############################


class UnusualWhalesClient:
    """GPU-Optimized client for the Unusual Whales API"""

    def __init__(
        self,
        api_key=UNUSUAL_WHALES_API_KEY,
        redis_client=None,
        use_gpu=USE_GPU,
        max_pool_size=MAX_POOL_SIZE,
        max_retries=MAX_RETRIES,
        cache_ttl=UNUSUAL_WHALES_CACHE_TTL,
    ) -> None:
        """
        Initialize GPU-Optimized Unusual Whales API client

        Args:
            api_key: API key for authentication
            redis_client: Optional Redis client for caching
            use_gpu: Whether to use GPU acceleration
            max_pool_size: Maximum connection pool size
            max_retries: Maximum number of retries for failed requests
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.use_gpu = use_gpu
        self.redis_client = redis_client

        # Verify API key is provided
        if not self.api_key:
            logger.warning(
                "No API key provided. Set the UNUSUAL_WHALES_API_KEY environment variable."
            )
        else:
            logger.info(
                f"Initialized Unusual Whales API client with API key: {self.api_key[:4]}****{self.api_key[-4:] if len(self.api_key) > 8 else ''}",
            )

        # Initialize base URL
        self.base_url = "https://api.unusualwhales.com/api"

        # Initialize cache
        self.cache = RedisCache(prefix="unusual_whales", ttl=cache_ttl)

        # Initialize connection pool
        self.connection_pool = AsyncConnectionPool(
            max_pool_size=max_pool_size,
            max_retries=max_retries,
        )

        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        # Initialize GPU accelerator
        self.gpu_accelerator = GPUAccelerator(use_gpu=use_gpu)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Flag to track if client is running
        self.running = True

        # Scheduled tasks
        self.scheduled_tasks = {}
        self.event_loop = None

        # Send initialization notification to frontend
        if self.redis_client:
            self._send_frontend_notification(
                message="Polygon REST API client initialized",
                level="info",
                category="system_startup",
                details={
                    "gpu_acceleration": "enabled" if self.use_gpu else "disabled",
                    "cache_ttl": self.cache_ttl,
                    "max_retries": max_retries,
                    "timestamp": time.time()
                }
            )

    def _handle_signal(self, signum, frame) -> None:
        """Handle signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False

        # Check if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.close())
        except RuntimeError:
            # No running event loop, schedule close for later
            logger.warning(
                "No running event loop available, resources will be closed later",
            )

    def _send_frontend_notification(self, message, level="info", category="unusual_whales", details=None):
        """
        Send notification to frontend via Redis

        Args:
            message: Notification message
            level: Notification level (info, success, warning, error)
            category: Notification category
            details: Additional details as dictionary
        """
        if not self.redis_client:
            return

        try:
            notification = {
                "timestamp": time.time(),
                "message": message,
                "level": level,
                "category": category,
                "source": "unusual_whales_api",
                "details": details or {}
            }

            # Add to notifications list
            self.redis_client.lpush(
                "frontend:notifications", json.dumps(notification))

            # Trim list to prevent unbounded growth
            self.redis_client.ltrim("frontend:notifications", 0, 999)

            # Publish event for real-time updates
            self.redis_client.publish("frontend:events", json.dumps({
                "type": "notification",
                "data": notification
            }))

            logger.debug(f"Sent frontend notification: {message}")
        except Exception as e:
            logger.error(f"Error sending frontend notification: {e}")

    async def _make_request(self, endpoint, params=None):
        """
        Make API request with caching

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            API response as a dictionary
        """
        # Generate cache key
        cache_key = [endpoint]
        if params:
            for key, value in sorted(params.items()):
                cache_key.append(f"{key}={value}")

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data

        # Prepare request
        url = f"{self.base_url}/{endpoint}"

        # Verify API key is set
        if not self.api_key:
            logger.error("API key not provided")
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    client="unusual_whales",
                    endpoint=endpoint,
                    method="GET",
                    error_type="missing_api_key",
                ).inc()
            return {"status": "ERROR", "error": "API key not provided"}

        # Set proper headers for Unusual Whales API
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Record request start time for latency measurement
        start_time = time.time()

        # Make request
        try:
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                API_REQUEST_COUNT.labels(
                    client="unusual_whales",
                    endpoint=endpoint,
                    method="GET",
                ).inc()

            data = await self.connection_pool.get(url, params=params, headers=headers)

            # Calculate request latency
            latency = time.time() - start_time

            # Update latency metrics
            if PROMETHEUS_AVAILABLE:
                API_REQUEST_LATENCY.labels(
                    client="unusual_whales",
                    endpoint=endpoint,
                    method="GET",
                ).observe(latency)

            # Cache successful responses
            if isinstance(data, dict) and "data" in data:
                self.cache.set(cache_key, data)
            elif isinstance(data, dict) and data.get("status") == "ERROR":
                # Update error metrics
                if PROMETHEUS_AVAILABLE:
                    API_ERROR_COUNT.labels(
                        client="unusual_whales",
                        endpoint=endpoint,
                        method="GET",
                        error_type="api_error",
                    ).inc()
                logger.error(
                    f"API error for {endpoint}: {data.get('error', 'Unknown error')}",
                )

            return data
        except Exception as e:
            # Calculate request latency even for failed requests
            latency = time.time() - start_time

            # Update error metrics
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    client="unusual_whales",
                    endpoint=endpoint,
                    method="GET",
                    error_type=type(e).__name__,
                ).inc()
                API_REQUEST_LATENCY.labels(
                    client="unusual_whales",
                    endpoint=endpoint,
                    method="GET",
                ).observe(latency)

            logger.exception(
                f"Unexpected error during API request: {type(e).__name__}: {e}",
            )
            return {
                "status": "ERROR",
                "error": f"Unexpected error: {type(e).__name__}: {e}",
            }

    async def check_api_health(self):
        """Check the health of the Unusual Whales API"""
        try:
            endpoint = "health"
            data = await self._make_request(endpoint)

            if data and "status" in data:
                logger.info(f"API health check result: {data['status']}")
                return data["status"] == "OK"
            if data and isinstance(data, dict):
                # Some APIs return a 200 status code with a different structure
                logger.info(f"API health check returned data: {data}")
                return True
            # If we get any response at all, consider the API healthy
            logger.warning(
                f"API health check returned unexpected format: {data}")
            return True
        except Exception as e:
            logger.exception(f"Error checking API health: {e}")
            return False

    async def get_flow_alerts(
        self,
        ticker,
        is_ask_side=True,
        is_bid_side=True,
        limit=100,
    ):
        """
        Get flow alerts for a specific ticker

        Args:
            ticker: Stock ticker symbol
            is_ask_side: Boolean flag whether a transaction is ask side
            is_bid_side: Boolean flag whether a transaction is bid side
            limit: How many items to return (max: 200)

        Returns:
            list: Flow alerts data
        """
        try:
            endpoint = f"stock/{ticker}/flow-alerts"
            params = {
                "is_ask_side": str(is_ask_side).lower(),
                "is_bid_side": str(is_bid_side).lower(),
                "limit": min(limit, 200),
            }

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                alert_count = len(data["data"])
                logger.info(
                    f"Flow alerts retrieved for {ticker}: {alert_count} items")

                if alert_count == 0:
                    logger.warning(f"No flow alerts found for {ticker}")

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete(f"unusual_whales:flow:{ticker}")

                    # Add each alert to the sorted set
                    for alert in data["data"]:
                        # Use created_at timestamp as score for sorting
                        created_at = alert.get("created_at", "")
                        if created_at:
                            try:
                                dt = datetime.fromisoformat(
                                    created_at.replace("Z", "+00:00"),
                                )
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd(
                            f"unusual_whales:flow:{ticker}",
                            {json.dumps(alert): score},
                        )

                    # Store last update timestamp
                    pipeline.set(
                        f"unusual_whales:flow:{ticker}:last_update",
                        datetime.now().isoformat(),
                    )

                    # Execute pipeline
                    pipeline.execute()

                # Process with GPU if available
                if self.use_gpu and self.gpu_accelerator.gpu_initialized:
                    self.thread_pool.submit(
                        self._process_flow_alerts_gpu,
                        ticker,
                        data["data"],
                    )

                # Send notification to frontend
                if self.redis_client:
                    self._send_frontend_notification(
                        message=f"Retrieved {alert_count} flow alerts for {ticker}",
                        level="success",
                        category="unusual_whales_data",
                        details={
                            "ticker": ticker,
                            "alert_count": alert_count,
                            "is_ask_side": is_ask_side,
                            "is_bid_side": is_bid_side,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

                return data["data"]
            error_msg = (
                data.get("error", "Unknown error")
                if isinstance(data, dict)
                else "Unknown error"
            )
            logger.warning(
                f"Failed to get flow alerts for {ticker}: {error_msg}")

            # Send error notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message=f"Failed to get flow alerts for {ticker}",
                    level="error",
                    category="unusual_whales_data",
                    details={
                        "ticker": ticker,
                        "error": error_msg,
                        "is_ask_side": is_ask_side,
                        "is_bid_side": is_bid_side
                    }
                )

            return []  # Return empty list on error
        except Exception as e:
            logger.exception(f"Error getting flow alerts for {ticker}: {e}")

            # Send error notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message=f"Error getting flow alerts for {ticker}",
                    level="error",
                    category="unusual_whales_data",
                    details={
                        "ticker": ticker,
                        "error_type": type(e).__name__,
                        "error": str(e)
                    }
                )

            return []

    def _process_flow_alerts_gpu(self, ticker, alerts) -> None:
        """Process flow alerts with GPU acceleration"""
        if not alerts:
            return

        try:
            # Extract relevant data
            premiums = [float(alert.get("total_premium", 0))
                        for alert in alerts]
            volumes = [int(alert.get("total_size", 0)) for alert in alerts]
            strikes = [float(alert.get("strike", 0)) for alert in alerts]

            # Use GPU for calculations
            if self.use_gpu and self.gpu_accelerator.gpu_initialized and CUPY_AVAILABLE:
                # Convert to CuPy arrays
                cp_premiums = cp.array(premiums, dtype=cp.float32)
                cp_volumes = cp.array(volumes, dtype=cp.float32)
                cp_strikes = cp.array(strikes, dtype=cp.float32)

                # Calculate statistics
                total_premium = cp.sum(cp_premiums)
                total_volume = cp.sum(cp_volumes)
                avg_premium = cp.mean(cp_premiums)
                avg_volume = cp.mean(cp_volumes)
                avg_strike = cp.mean(cp_strikes)

                # Convert back to numpy
                total_premium = float(cp.asnumpy(total_premium))
                total_volume = float(cp.asnumpy(total_volume))
                avg_premium = float(cp.asnumpy(avg_premium))
                avg_volume = float(cp.asnumpy(avg_volume))
                avg_strike = float(cp.asnumpy(avg_strike))
            else:
                # CPU calculations
                total_premium = sum(premiums)
                total_volume = sum(volumes)
                avg_premium = sum(premiums) / len(premiums) if premiums else 0
                avg_volume = sum(volumes) / len(volumes) if volumes else 0
                avg_strike = sum(strikes) / len(strikes) if strikes else 0

            # Count call vs put
            call_count = sum(
                1 for alert in alerts if alert.get("type") == "call")
            put_count = sum(
                1 for alert in alerts if alert.get("type") == "put")

            # Store statistics in Redis if available
            if self.redis_client:
                self.redis_client.hmset(
                    f"unusual_whales:flow_stats:{ticker}",
                    {
                        "total_premium": total_premium,
                        "total_volume": total_volume,
                        "avg_premium": avg_premium,
                        "avg_volume": avg_volume,
                        "avg_strike": avg_strike,
                        "call_count": call_count,
                        "put_count": put_count,
                        "alert_count": len(alerts),
                        "last_update": datetime.now().isoformat(),
                    },
                )

        except Exception as e:
            logger.exception(
                f"Error processing flow alerts for {ticker} on GPU: {e}")

    async def get_alerts(
        self,
        config_ids=None,
        intraday_only=True,
        limit=100,
        noti_types=None,
        page=0,
        ticker_symbols=None,
    ):
        """
        Get all alerts that have been triggered for the user

        Args:
            config_ids: A list of alert ids to filter by
            intraday_only: Boolean flag whether to return only intraday alerts
            limit: How many items to return (max: 200)
            noti_types: A list of notification types
            page: The page number to return
            ticker_symbols: A comma separated list of tickers

        Returns:
            list: Alerts data
        """
        try:
            endpoint = "alerts"
            params = {
                # Empty string instead of None
                "ticker_symbols": ticker_symbols if ticker_symbols is not None else "",
                "intraday_only": str(intraday_only).lower(),
                "limit": min(limit, 200),
                "page": max(page, 0),
            }

            if config_ids:
                if isinstance(config_ids, list):
                    params["config_ids[]"] = config_ids
                else:
                    params["config_ids[]"] = [config_ids]

            if noti_types:
                if isinstance(noti_types, list):
                    params["noti_types[]"] = noti_types
                else:
                    params["noti_types[]"] = [noti_types]

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                alert_count = len(data["data"])
                logger.info(f"Alerts retrieved: {alert_count} items")
                logger.info(
                    f"Alert types: { {alert.get('type', 'unknown') for alert in data['data']} }"
                )

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data if page is 0
                    if page == 0:
                        pipeline.delete("unusual_whales:alerts")

                    # Add each alert to the sorted set
                    for alert in data["data"]:
                        # Use tape_time timestamp as score for sorting
                        tape_time = alert.get("tape_time", "")
                        if tape_time:
                            try:
                                dt = datetime.fromisoformat(
                                    tape_time.replace("Z", "+00:00"),
                                )
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd(
                            "unusual_whales:alerts",
                            {json.dumps(alert): score},
                        )

                    # Store last update timestamp
                    pipeline.set(
                        "unusual_whales:alerts:last_update",
                        datetime.now().isoformat(),
                    )

                    # Execute pipeline
                    pipeline.execute()

                return data["data"]
            error_msg = (
                data.get("error", "Unknown error")
                if isinstance(data, dict)
                else "Unknown error"
            )
            logger.warning(f"Failed to get Unusual Whales alerts: {error_msg}")
            return []
        except Exception as e:
            logger.exception(f"Error getting alerts: {e}")
            return []

    async def get_alert_configurations(self):
        """
        Get all alert configurations of the user

        Returns:
            list: Alert configurations
        """
        try:
            endpoint = "alerts/configuration"

            data = await self._make_request(endpoint)

            if data and "data" in data:
                logger.info(
                    f"Alert configurations retrieved: {len(data['data'])} items",
                )

                # Store in Redis if available
                if self.redis_client:
                    # Store as a hash in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete("unusual_whales:alert_configs")

                    # Add each config to the hash
                    for config in data["data"]:
                        config_id = config.get("id", "")
                        if config_id:
                            pipeline.hset(
                                "unusual_whales:alert_configs",
                                config_id,
                                json.dumps(config),
                            )

                    # Store last update timestamp
                    pipeline.set(
                        "unusual_whales:alert_configs:last_update",
                        datetime.now().isoformat(),
                    )

                    # Execute pipeline
                    pipeline.execute()

                return data["data"]
            error_msg = (
                data.get("error", "Unknown error")
                if isinstance(data, dict)
                else "Unknown error"
            )
            logger.warning(f"Failed to get alert configurations: {error_msg}")
            return []
        except Exception as e:
            logger.exception(f"Error getting alert configurations: {e}")
            return []

    async def get_recent_dark_pool_trades(
        self,
        date=None,
        limit=100,
        max_premium=None,
        max_size=None,
        max_volume=None,
        min_premium=0,
        min_size=0,
        min_volume=0,
    ):
        """
        Get the latest darkpool trades

        Args:
            date: A trading date in the format of YYYY-MM-DD
            limit: How many items to return (max: 200)
            max_premium: The maximum premium requested trades should have
            max_size: The maximum size requested trades should have
            max_volume: The maximum consolidated volume requested trades should have
            min_premium: The minimum premium requested trades should have
            min_size: The minimum size requested trades should have
            min_volume: The minimum consolidated volume requested trades should have

        Returns:
            list: Dark pool trades data
        """
        try:
            endpoint = "darkpool/recent"
            params = {
                "limit": min(limit, 200),
                "min_premium": min_premium,
                "min_size": min_size,
                "min_volume": min_volume,
            }

            if date:
                params["date"] = date

            if max_premium is not None:
                params["max_premium"] = max_premium

            if max_size is not None:
                params["max_size"] = max_size

            if max_volume is not None:
                params["max_volume"] = max_volume

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                logger.info(
                    f"Recent dark pool trades retrieved: {len(data['data'])} items",
                )

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete("unusual_whales:darkpool:recent")

                    # Add each trade to the sorted set
                    for trade in data["data"]:
                        # Use executed_at timestamp as score for sorting
                        executed_at = trade.get("executed_at", "")
                        if executed_at:
                            try:
                                dt = datetime.fromisoformat(
                                    executed_at.replace("Z", "+00:00"),
                                )
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd(
                            "unusual_whales:darkpool:recent",
                            {json.dumps(trade): score},
                        )

                    # Store last update timestamp
                    pipeline.set(
                        "unusual_whales:darkpool:recent:last_update",
                        datetime.now().isoformat(),
                    )

                    # Execute pipeline
                    pipeline.execute()

                # Process with GPU if available
                if self.use_gpu and self.gpu_accelerator.gpu_initialized:
                    self.thread_pool.submit(
                        self._process_dark_pool_trades_gpu,
                        data["data"],
                    )

                # Send notification to frontend
                if self.redis_client:
                    trade_count = len(data["data"])
                    tickers = list(set(trade.get("ticker", "")
                                   for trade in data["data"] if trade.get("ticker")))
                    self._send_frontend_notification(
                        message=f"Retrieved {trade_count} recent dark pool trades",
                        level="success",
                        category="unusual_whales_data",
                        details={
                            "trade_count": trade_count,
                            "tickers": tickers[:10] + (["..."] if len(tickers) > 10 else []),
                            "min_premium": min_premium,
                            "min_size": min_size,
                            "min_volume": min_volume,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

                return data["data"]
            error_msg = (
                data.get("error", "Unknown error")
                if isinstance(data, dict)
                else "Unknown error"
            )
            logger.warning(
                f"Failed to get recent dark pool trades: {error_msg}")

            # Send error notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message="Failed to get recent dark pool trades",
                    level="error",
                    category="unusual_whales_data",
                    details={
                        "error": error_msg,
                        "min_premium": min_premium,
                        "min_size": min_size,
                        "min_volume": min_volume,
                        "date": date
                    }
                )

            return []
        except Exception as e:
            logger.exception(f"Error getting recent dark pool trades: {e}")

            # Send error notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message="Error getting recent dark pool trades",
                    level="error",
                    category="unusual_whales_data",
                    details={
                        "error_type": type(e).__name__,
                        "error": str(e)
                    }
                )

            return []

    def _process_dark_pool_trades_gpu(self, trades) -> None:
        """Process dark pool trades with GPU acceleration"""
        if not trades:
            return

        try:
            # Record GPU processing start time
            gpu_start_time = time.time()

            # Group trades by ticker
            trades_by_ticker = {}
            for trade in trades:
                ticker = trade.get("ticker")
                if ticker:
                    if ticker not in trades_by_ticker:
                        trades_by_ticker[ticker] = []
                    trades_by_ticker[ticker].append(trade)

            # Process each ticker's trades
            for ticker, ticker_trades in trades_by_ticker.items():
                # Extract relevant data
                premiums = [float(trade.get("premium", 0))
                            for trade in ticker_trades]
                sizes = [int(trade.get("size", 0)) for trade in ticker_trades]
                prices = [float(trade.get("price", 0))
                          for trade in ticker_trades]

                # Use GPU for calculations
                if (
                    self.use_gpu
                    and self.gpu_accelerator.gpu_initialized
                    and CUPY_AVAILABLE
                ):
                    # Convert to CuPy arrays
                    cp_premiums = cp.array(premiums, dtype=cp.float32)
                    cp_sizes = cp.array(sizes, dtype=cp.float32)
                    cp_prices = cp.array(prices, dtype=cp.float32)

                    # Calculate statistics
                    total_premium = cp.sum(cp_premiums)
                    total_size = cp.sum(cp_sizes)
                    avg_premium = cp.mean(cp_premiums)
                    avg_size = cp.mean(cp_sizes)
                    avg_price = cp.mean(cp_prices)

                    # Convert back to numpy
                    total_premium = float(cp.asnumpy(total_premium))
                    total_size = float(cp.asnumpy(total_size))
                    avg_premium = float(cp.asnumpy(avg_premium))
                    avg_size = float(cp.asnumpy(avg_size))
                    avg_price = float(cp.asnumpy(avg_price))

                    # Update GPU memory usage metrics
                    if PROMETHEUS_AVAILABLE:
                        device_id = cp.cuda.Device().id
                        device_props = cp.cuda.runtime.getDeviceProperties(
                            device_id)
                        device_name = device_props["name"].decode()
                        mem_info = cp.cuda.runtime.memGetInfo()
                        mem_used = mem_info[1] - \
                            mem_info[0]  # total - free = used
                        GPU_MEMORY_USAGE.labels(
                            device=device_name).set(mem_used)
                else:
                    # CPU calculations
                    total_premium = sum(premiums)
                    total_size = sum(sizes)
                    avg_premium = sum(premiums) / \
                        len(premiums) if premiums else 0
                    avg_size = sum(sizes) / len(sizes) if sizes else 0
                    avg_price = sum(prices) / len(prices) if prices else 0

                # Store statistics in Redis if available
                if self.redis_client:
                    self.redis_client.hmset(
                        f"unusual_whales:darkpool_stats:{ticker}",
                        {
                            "total_premium": total_premium,
                            "total_size": total_size,
                            "avg_premium": avg_premium,
                            "avg_size": avg_size,
                            "avg_price": avg_price,
                            "trade_count": len(ticker_trades),
                            "last_update": datetime.now().isoformat(),
                        },
                    )

            # Record GPU processing time
            gpu_processing_time = time.time() - gpu_start_time
            if PROMETHEUS_AVAILABLE:
                GPU_PROCESSING_TIME.labels(operation="darkpool_processing").observe(
                    gpu_processing_time,
                )

            # Send notification to frontend about successful GPU processing
            if self.redis_client and random.random() < 0.1:  # Only send 10% of the time to avoid flooding
                self._send_frontend_notification(
                    message="GPU-accelerated dark pool trade processing completed",
                    level="info",
                    category="gpu_processing",
                    details={
                        "ticker_count": len(trades_by_ticker),
                        "tickers": list(trades_by_ticker.keys())[:5] + (["..."] if len(trades_by_ticker) > 5 else []),
                        "processing_time": gpu_processing_time,
                        "trade_count": len(trades)
                    }
                )

        except Exception as e:
            logger.exception(f"Error processing dark pool trades on GPU: {e}")
            # Update error metrics
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(
                    client="unusual_whales",
                    endpoint="darkpool",
                    method="gpu_processing",
                    error_type=type(e).__name__,
                ).inc()

            # Send error notification to frontend
            if self.redis_client:
                self._send_frontend_notification(
                    message="Error processing dark pool trades on GPU",
                    level="error",
                    category="gpu_processing",
                    details={
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "trade_count": len(trades) if trades else 0
                    }
                )

    async def get_dark_pool_trades(
        self,
        ticker,
        date=None,
        limit=500,
        max_premium=None,
        max_size=None,
        max_volume=None,
        min_premium=0,
        min_size=0,
        min_volume=0,
        newer_than=None,
        older_than=None,
    ):
        """
        Get the darkpool trades for the given ticker on a given day

        Args:
            ticker: Stock ticker symbol
            date: A trading date in the format of YYYY-MM-DD
            limit: How many items to return (max: 500)
            max_premium: The maximum premium requested trades should have
            max_size: The maximum size requested trades should have
            max_volume: The maximum consolidated volume requested trades should have
            min_premium: The minimum premium requested trades should have
            min_size: The minimum size requested trades should have
            min_volume: The minimum consolidated volume requested trades should have
            newer_than: The unix time in milliseconds or seconds
            older_than: The unix time in milliseconds or seconds

        Returns:
            list: Dark pool trades data
        """
        try:
            endpoint = f"darkpool/{ticker}"
            params = {
                "limit": min(limit, 500),
                "min_premium": min_premium,
                "min_size": min_size,
                "min_volume": min_volume,
            }

            if date:
                params["date"] = date

            if max_premium is not None:
                params["max_premium"] = max_premium

            if max_size is not None:
                params["max_size"] = max_size

            if max_volume is not None:
                params["max_volume"] = max_volume

            if newer_than is not None:
                params["newer_than"] = newer_than

            if older_than is not None:
                params["older_than"] = older_than

            data = await self._make_request(endpoint, params)

            if data and "data" in data:
                logger.info(
                    f"Dark pool trades retrieved for {ticker}: {len(data['data'])} items"
                )

                # Store in Redis if available
                if self.redis_client:
                    # Store as a sorted set in Redis
                    pipeline = self.redis_client.pipeline()

                    # Clear existing data
                    pipeline.delete(f"unusual_whales:darkpool:{ticker}")

                    # Add each trade to the sorted set
                    for trade in data["data"]:
                        # Use executed_at timestamp as score for sorting
                        executed_at = trade.get("executed_at", "")
                        if executed_at:
                            try:
                                dt = datetime.fromisoformat(
                                    executed_at.replace("Z", "+00:00"),
                                )
                                score = dt.timestamp()
                            except ValueError:
                                score = time.time()
                        else:
                            score = time.time()

                        pipeline.zadd(
                            f"unusual_whales:darkpool:{ticker}",
                            {json.dumps(trade): score},
                        )

                    # Store last update timestamp
                    pipeline.set(
                        f"unusual_whales:darkpool:{ticker}:last_update",
                        datetime.now().isoformat(),
                    )

                    # Execute pipeline
                    pipeline.execute()

                # Process with GPU if available
                if self.use_gpu and self.gpu_accelerator.gpu_initialized:
                    self.thread_pool.submit(
                        self._process_dark_pool_trades_gpu,
                        data["data"],
                    )

                return data["data"]
            error_msg = (
                data.get("error", "Unknown error")
                if isinstance(data, dict)
                else "Unknown error"
            )
            logger.warning(
                f"Failed to get dark pool trades for {ticker}: {error_msg}",
            )
            return []
        except Exception as e:
            logger.exception(
                f"Error getting dark pool trades for {ticker}: {e}")
            return []

    async def start_scheduled_tasks(self, event_loop=None) -> None:
        """
        Start scheduled tasks for data collection

        Args:
            event_loop: Optional event loop to use
        """
        if event_loop:
            self.event_loop = event_loop
        else:
            self.event_loop = asyncio.get_event_loop()

        # Schedule unusual activity updates
        self.scheduled_tasks["unusual_activity"] = self.event_loop.create_task(
            self._schedule_task(
                self.get_alerts,
                UNUSUAL_WHALES_REFRESH_INTERVALS["unusual_activity"],
            ),
        )

        # Schedule dark pool updates
        self.scheduled_tasks["dark_pool"] = self.event_loop.create_task(
            self._schedule_task(
                self.get_recent_dark_pool_trades,
                UNUSUAL_WHALES_REFRESH_INTERVALS["dark_pool"],
            ),
        )

        logger.info("Scheduled tasks started")

    async def _schedule_task(self, task_func, interval, *args, **kwargs) -> None:
        """
        Schedule a task to run at regular intervals

        Args:
            task_func: Function to run
            interval: Interval in seconds
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
        """
        while self.running:
            try:
                await task_func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Error in scheduled task {task_func.__name__}: {e}")

            await asyncio.sleep(interval)

    async def schedule_ticker_updates(self, tickers, intervals=None) -> None:
        """
        Schedule updates for specific tickers

        Args:
            tickers: List of tickers to update
            intervals: Optional dictionary of intervals for different endpoints
        """
        if not intervals:
            intervals = {
                "flow": UNUSUAL_WHALES_REFRESH_INTERVALS["flow"],
                "dark_pool": UNUSUAL_WHALES_REFRESH_INTERVALS["dark_pool"],
            }

        # Schedule flow updates
        self.scheduled_tasks["flow"] = self.event_loop.create_task(
            self._schedule_flow_updates(tickers, intervals["flow"]),
        )

        # Schedule dark pool updates
        self.scheduled_tasks["dark_pool_tickers"] = self.event_loop.create_task(
            self._schedule_dark_pool_updates(tickers, intervals["dark_pool"]),
        )

        logger.info(f"Scheduled updates for {len(tickers)} tickers")

    async def _schedule_flow_updates(self, tickers, interval) -> None:
        """Schedule flow updates for specific tickers"""
        while self.running:
            for ticker in tickers:
                if not self.running:
                    break

                try:
                    await self.get_flow_alerts(ticker)
                except Exception as e:
                    logger.exception(
                        f"Error updating flow alerts for {ticker}: {e}")

                # Small delay between requests to avoid rate limits
                await asyncio.sleep(0.5)

            # Wait until next update cycle
            await asyncio.sleep(interval)

    async def _schedule_dark_pool_updates(self, tickers, interval) -> None:
        """Schedule dark pool updates for specific tickers"""
        while self.running:
            for ticker in tickers:
                if not self.running:
                    break

                try:
                    await self.get_dark_pool_trades(ticker)
                except Exception as e:
                    logger.exception(
                        f"Error updating dark pool trades for {ticker}: {e}",
                    )

                # Small delay between requests to avoid rate limits
                await asyncio.sleep(0.5)

            # Wait until next update cycle
            await asyncio.sleep(interval)

    async def close(self) -> None:
        """Close all connections and resources"""
        logger.info("Closing Unusual Whales API client")
        self.running = False

        # Cancel all scheduled tasks
        for name, task in self.scheduled_tasks.items():
            if not task.done():
                logger.info(f"Cancelling scheduled task: {name}")
                try:
                    task.cancel()
                except Exception as e:
                    logger.exception(f"Error cancelling task {name}: {e}")

        # Close connection pool
        try:
            await self.connection_pool.close()
        except Exception as e:
            logger.exception(f"Error closing connection pool: {e}")

        # Clean up GPU resources
        self.gpu_accelerator.clear_memory()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)

        logger.info("Unusual Whales API client closed")
