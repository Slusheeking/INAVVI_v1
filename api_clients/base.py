"""
Base Utilities for API Clients

This module provides shared utilities and configurations for API clients:
- Logging setup
- GPU configuration
- Exception handlers
- Metrics
- Redis caching
- Connection pooling
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
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable

import aiohttp
import numpy as np
import pandas as pd
import redis
import websockets
from dotenv import load_dotenv
from prometheus_client import Counter, Gauge, Histogram
from websockets.exceptions import ConnectionClosed, WebSocketException

# Load environment variables
load_dotenv()

# Configure logging with a detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api_clients.log")
    ]
)
logger = logging.getLogger("api_clients")

# Environment variables with defaults
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
UNUSUAL_WHALES_API_KEY = os.environ.get("UNUSUAL_WHALES_API_KEY", "")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "trading_system_2025")
REDIS_SSL = os.environ.get("REDIS_SSL", "false").lower() == "true"
REDIS_TIMEOUT = int(os.environ.get("REDIS_TIMEOUT", "5"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = float(os.environ.get("RETRY_BACKOFF_FACTOR", "0.5"))
CONNECTION_TIMEOUT = int(os.environ.get("CONNECTION_TIMEOUT", "15"))
MAX_POOL_SIZE = int(os.environ.get("MAX_POOL_SIZE", "30"))
RECONNECT_DELAY = float(os.environ.get("RECONNECT_DELAY", "2.0"))
MAX_RECONNECT_ATTEMPTS = int(os.environ.get("MAX_RECONNECT_ATTEMPTS", "10"))
POLYGON_CACHE_TTL = int(os.environ.get("POLYGON_CACHE_TTL", "3600"))
UNUSUAL_WHALES_CACHE_TTL = int(os.environ.get("UNUSUAL_WHALES_CACHE_TTL", "300"))
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", "1000"))

# Prometheus metrics (same as before but with additional metrics)
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

# Configure exception hook to log unhandled exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler"""
    if not issubclass(exc_type, KeyboardInterrupt):
        logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def handle_asyncio_exception(loop, context):
    """Asyncio exception handler"""
    msg = context.get("exception", context["message"])
    logger.error(f"Asyncio exception: {msg}")
    if "exception" in context:
        exc = context["exception"]
        logger.error(f"Exception details: {traceback.format_exception(type(exc), exc, exc.__traceback__)}")

asyncio.get_event_loop().set_exception_handler(handle_asyncio_exception)

# Import GPU acceleration libraries with fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import ONNX with fallback
try:
    import onnx
    ONNX_VERSION = onnx.__version__
    ONNX_AVAILABLE = True
    logger.info(f"Using ONNX version {ONNX_VERSION}")
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    logger.warning("ONNX not available, model optimization will be limited")

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

class RedisCache:
    """Redis-based cache for API responses with fallback to in-memory cache"""
    
    def __init__(
        self,
        prefix="api_cache",
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        ttl=3600,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        ssl=REDIS_SSL,
        max_memory_items=10000,
        connection_pool=None
    ) -> None:
        self.prefix = prefix
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self.enabled = True
        self.memory_cache = {}
        self.hits = 0
        self.misses = 0
        self.size_limit = max_memory_items

        try:
            if connection_pool:
                self.redis_client = redis.Redis(
                    connection_pool=connection_pool,
                    decode_responses=False
                )
            else:
                pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    username=username,
                    password=password,
                    socket_timeout=REDIS_TIMEOUT,
                    decode_responses=False
                )
                self.redis_client = redis.Redis(connection_pool=pool)
            self.redis_client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False

    def _generate_key(self, key_parts):
        """Generate a consistent cache key from parts"""
        if isinstance(key_parts, str):
            key_parts = [key_parts]
        key_str = ":".join([str(part) for part in key_parts])
        return f"{self.prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"

    def get(self, key_parts):
        """Get value from cache with fallback to Redis"""
        try:
            key = self._generate_key(key_parts)
            
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry["expiry"]:
                    self.hits += 1
                    API_CACHE_HIT_COUNT.labels(client=self.prefix, cache_type="memory").inc()
                    return entry["value"]
                del self.memory_cache[key]

            # Try Redis if enabled
            if self.enabled:
                data = self.redis_client.get(key)
                if data:
                    self.hits += 1
                    API_CACHE_HIT_COUNT.labels(client=self.prefix, cache_type="redis").inc()
                    value = pickle.loads(data)
                    self.memory_cache[key] = {"value": value, "expiry": time.time() + self.ttl}
                    return value

            self.misses += 1
            API_CACHE_MISS_COUNT.labels(client=self.prefix, cache_type="all").inc()
            return None
        except Exception as e:
            logger.error(f"Error in cache get: {e}")
            return None

    def set(self, key_parts, value, ttl=None):
        """Set value in cache with optional TTL"""
        try:
            key = self._generate_key(key_parts)
            ttl = ttl or self.ttl
            expiry = time.time() + ttl

            # Implement LRU-like behavior
            if len(self.memory_cache) >= self.size_limit:
                keys_to_remove = random.sample(list(self.memory_cache.keys()), int(len(self.memory_cache) * 0.1))
                for k in keys_to_remove:
                    self.memory_cache.pop(k, None)

            # Store in memory cache
            self.memory_cache[key] = {"value": value, "expiry": expiry}

            # Store in Redis if enabled
            if self.enabled:
                serialized = pickle.dumps(value)
                return self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error in cache set: {e}")
            return False

    def store_dataframe(self, key_parts, df, ttl=None):
        """Store DataFrame in cache with optimized serialization"""
        try:
            if df is None or df.empty:
                return False

            key = self._generate_key(key_parts)
            ttl = ttl or self.ttl
            serialized = pickle.dumps(df)

            if self.enabled:
                success = self.redis_client.setex(key, ttl, serialized)
                if success:
                    self.memory_cache[key] = {"value": df, "expiry": time.time() + ttl}
                return success
            return True
        except Exception as e:
            logger.error(f"Error storing DataFrame: {e}")
            return False

    def get_dataframe(self, key_parts):
        """Retrieve DataFrame from cache with optimized deserialization"""
        try:
            key = self._generate_key(key_parts)
            
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry["expiry"]:
                    self.hits += 1
                    API_CACHE_HIT_COUNT.labels(client=self.prefix, cache_type="memory").inc()
                    return entry["value"]
                del self.memory_cache[key]

            # Try Redis if enabled
            if self.enabled:
                data = self.redis_client.get(key)
                if data:
                    self.hits += 1
                    API_CACHE_HIT_COUNT.labels(client=self.prefix, cache_type="redis").inc()
                    df = pickle.loads(data)
                    self.memory_cache[key] = {"value": df, "expiry": time.time() + self.ttl}
                    return df

            self.misses += 1
            API_CACHE_MISS_COUNT.labels(client=self.prefix, cache_type="all").inc()
            return None
        except Exception as e:
            logger.error(f"Error getting DataFrame: {e}")
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
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.session = None
        self.connector = None

    async def initialize(self) -> None:
        """Initialize the aiohttp session"""
        if self.session is None or self.session.closed:
            self.connector = aiohttp.TCPConnector(
                limit=self.max_pool_size,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=False,
            )
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
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 429:
                        wait_time = (2**retry) * self.backoff_factor + random.uniform(0, 1)
                        await asyncio.sleep(wait_time)
                        continue
                    error_text = await response.text()
                    last_error = f"HTTP {response.status}: {error_text}"
                    if 400 <= response.status < 500 and response.status != 429:
                        return {"status": "ERROR", "error": last_error}
                    wait_time = (2**retry) * self.backoff_factor + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = f"Request failed: {type(e).__name__}: {e!s}"
                wait_time = (2**retry) * self.backoff_factor + random.uniform(0, 1)
                await asyncio.sleep(wait_time)

        return {"status": "ERROR", "error": last_error or "Max retries exceeded"}

    async def close(self) -> None:
        """Close the session and all connections"""
        if self.session and not self.session.closed:
            await self.session.close()

# GPUAccelerator class removed - use utils.gpu_utils.gpu_manager instead
