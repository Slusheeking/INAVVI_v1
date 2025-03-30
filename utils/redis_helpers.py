#!/usr/bin/env python3
"""
Redis Helpers Module

This module provides standardized Redis client configuration and utility functions
for the trading system. It supports:

1. Unified connection management with connection pooling
2. Consistent key naming conventions
3. Standardized serialization formats
4. Common Redis operations with error handling
5. Compression for large data objects
6. Monitoring and metrics collection

All components in the trading system should use this module for Redis operations
to ensure consistent behavior and optimal performance.
"""

import asyncio # Import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import time
import traceback
import zlib
from datetime import datetime # Added for timestamp handling
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import redis
import redis.asyncio as aioredis # Import async redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("redis_helpers")

# Import Prometheus client if available
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True

    # Define Prometheus metrics
    REDIS_OPERATION_COUNT = Counter(
        "redis_operation_count",
        "Number of Redis operations",
        ["operation", "status"]
    )

    REDIS_OPERATION_LATENCY = Histogram(
        "redis_operation_latency_seconds",
        "Redis operation latency in seconds",
        ["operation"]
    )

    REDIS_MEMORY_USAGE = Gauge(
        "redis_memory_usage_bytes",
        "Redis memory usage in bytes",
        ["type"]
    )

    REDIS_CACHE_SIZE = Gauge(
        "redis_cache_size",
        "Number of keys in Redis cache",
        ["prefix"]
    )

    REDIS_CACHE_HIT_COUNT = Counter(
        "redis_cache_hit_count",
        "Number of cache hits",
        ["cache_type"]
    )

    REDIS_CACHE_MISS_COUNT = Counter(
        "redis_cache_miss_count",
        "Number of cache misses",
        ["cache_type"]
    )

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics collection will be limited.")

# Redis configuration from environment variables
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_SSL = os.environ.get("REDIS_SSL", "false").lower() == "true"
REDIS_TIMEOUT = int(os.environ.get("REDIS_TIMEOUT", "5"))
REDIS_SOCKET_CONNECT_TIMEOUT = int(os.environ.get("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
REDIS_MAX_CONNECTIONS = int(os.environ.get("REDIS_MAX_CONNECTIONS", "50"))
REDIS_COMPRESSION_THRESHOLD = int(os.environ.get("REDIS_COMPRESSION_THRESHOLD", "1024"))
REDIS_COMPRESSION_LEVEL = int(os.environ.get("REDIS_COMPRESSION_LEVEL", "6"))


# Global async connection pool for reuse across modules
_async_connection_pool = None


async def get_async_connection_pool() -> aioredis.ConnectionPool:
    """
    Get or create an async Redis connection pool

    Returns:
        Async Redis connection pool for reuse
    """
    global _async_connection_pool

    if _async_connection_pool is None:
        try:
            _async_connection_pool = aioredis.ConnectionPool(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                username=REDIS_USERNAME,
                password=REDIS_PASSWORD,
                socket_timeout=REDIS_TIMEOUT,
                socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=True,  # Decode responses for easier handling
                max_connections=REDIS_MAX_CONNECTIONS
            )
            logger.info(f"Created Async Redis connection pool: {REDIS_HOST}:{REDIS_PORT} (DB: {REDIS_DB})")
        except Exception as e:
            logger.error(f"Failed to create Async Redis connection pool: {e}")
            logger.error(traceback.format_exc())
            raise

    return _async_connection_pool


async def get_async_redis_client() -> aioredis.Redis:
    """
    Get an async Redis client using the shared connection pool

    Returns:
        Async Redis client instance
    """
    try:
        pool = await get_async_connection_pool()
        # Ensure decode_responses=True is set at pool level or here if needed
        client = aioredis.Redis(connection_pool=pool)

        # Test connection
        await client.ping()

        # Update metrics if available (async version)
        if PROMETHEUS_AVAILABLE:
            try:
                info = await client.info()
                used_memory = info.get('used_memory', 0)
                max_memory = info.get('maxmemory', 0)
                REDIS_MEMORY_USAGE.labels(type="used").set(used_memory)
                if max_memory > 0:
                    REDIS_MEMORY_USAGE.labels(type="max").set(max_memory)
                logger.debug(f"Redis memory usage: {used_memory}/{max_memory} bytes")
            except redis.RedisError as e: # Use base redis error type
                logger.warning(f"Could not get Redis memory info: {e}")

        return client

    except Exception as e:
        logger.error(f"Failed to get Async Redis client: {e}")
        logger.error(traceback.format_exc())
        raise


def generate_key(prefix: str, key_parts: Union[str, List[Any]]) -> str:
    """
    Generate a consistent Redis key from parts

    Args:
        prefix: Namespace prefix for the key
        key_parts: Key parts to generate cache key from

    Returns:
        Formatted Redis key
    """
    if isinstance(key_parts, str):
        key_parts = [key_parts]

    # Join all parts and create a hash
    key_str = ":".join([str(part) for part in key_parts])
    # Use prefix directly without hashing for simpler tick keys if desired,
    # but hashing avoids potential key format issues. Sticking with hash for now.
    return f"{prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"


def serialize_value(value: Any, use_json: bool = False) -> bytes:
    """
    Serialize a value for Redis storage with optional compression

    Args:
        value: Value to serialize
        use_json: Whether to use JSON serialization instead of pickle

    Returns:
        Serialized value as bytes
    """
    try:
        if value is None:
            return b""

        # Serialize the value
        if use_json:
            # Convert pandas DataFrame to dict if needed
            if isinstance(value, pd.DataFrame):
                serialized = json.dumps(value.to_dict(orient="records")).encode()
            else:
                serialized = json.dumps(value).encode()
        else:
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

        # Compress if larger than threshold
        if len(serialized) > REDIS_COMPRESSION_THRESHOLD:
            compressed = zlib.compress(serialized, level=REDIS_COMPRESSION_LEVEL)
            # Only use compression if it actually reduces size
            if len(compressed) < len(serialized):
                return b"c" + compressed  # Prefix with 'c' to indicate compression

        return b"u" + serialized  # Prefix with 'u' to indicate uncompressed

    except Exception as e:
        logger.error(f"Error serializing value: {e}")
        logger.error(traceback.format_exc())
        raise


def deserialize_value(data: Union[bytes, str], use_json: bool = False) -> Any:
    """
    Deserialize a value from Redis storage

    Args:
        data: Serialized data from Redis (bytes or string if decode_responses=True)
        use_json: Whether to use JSON deserialization instead of pickle

    Returns:
        Deserialized value
    """
    try:
        if not data:
            return None

        # Handle bytes if decode_responses=False was used somewhere
        if isinstance(data, bytes):
            # Check compression flag
            compression_flag = data[0:1]
            payload = data[1:]

            # Decompress if needed
            if compression_flag == b"c":
                payload = zlib.decompress(payload)

            # Deserialize the value
            if use_json:
                return json.loads(payload.decode())
            else:
                return pickle.loads(payload)
        # Handle string if decode_responses=True
        elif isinstance(data, str):
             # Assume no compression if it's already a string
             if use_json:
                  return json.loads(data)
             else:
                  # Cannot unpickle directly from string, needs bytes
                  logger.warning("Attempted to deserialize non-JSON string with pickle.")
                  # Try encoding back to bytes, might fail if complex chars
                  try:
                       return pickle.loads(data.encode('utf-8', 'surrogatepass'))
                  except Exception as pe:
                       logger.error(f"Pickle deserialization from string failed: {pe}")
                       return None
        else:
             logger.error(f"Unexpected data type for deserialization: {type(data)}")
             return None


    except Exception as e:
        logger.error(f"Error deserializing value: {e}")
        logger.error(traceback.format_exc())
        return None


class RedisClient:
    """
    Async Redis client wrapper providing specific methods for the trading engine.
    """
    def __init__(self):
        self.client: Optional[aioredis.Redis] = None
        self._init_task: Optional[asyncio.Task] = None

    async def _initialize(self):
        """Initialize the underlying async Redis client."""
        if self.client is None:
            try:
                self.client = await get_async_redis_client()
                logger.info("Async RedisClient initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Async RedisClient: {e}")
                self.client = None # Ensure client is None on failure
                raise # Re-raise the exception

    async def ensure_initialized(self):
        """Ensure the client is initialized, handling concurrent initialization."""
        if self.client:
            # Optional: Add a periodic ping check here if needed
            # try:
            #     await self.client.ping()
            # except Exception:
            #     logger.warning("Redis connection lost, attempting reinitialization.")
            #     self.client = None # Force reinitialization
            #     self._init_task = None # Reset task
            # else:
            #     return # Connection is good
            return

        if self._init_task is None or self._init_task.done():
            # If task finished with exception, allow retry
            if self._init_task and self._init_task.exception():
                 logger.warning(f"Previous Redis init failed ({self._init_task.exception()}), retrying.")
            self._init_task = asyncio.create_task(self._initialize())

        try:
            await self._init_task
        except Exception:
             # Initialization failed, client remains None
             logger.error("Redis client initialization failed after await.")
             # Do not raise here, allow methods to check self.client


    async def update_last_tick_data(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        volume: Optional[int] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        event_type: str = 'T' # 'T' for Trade, 'Q' for Quote
    ) -> bool:
        """
        Updates the latest tick data for a symbol in Redis using a Hash.

        Args:
            symbol: The stock symbol (e.g., AAPL).
            price: The latest price (trade price or mid-price for quotes).
            timestamp: The timestamp of the event (use server-side timestamp).
            volume: The volume of the trade (for Trade events).
            bid: The best bid price (for Quote events).
            ask: The best ask price (for Quote events).
            event_type: 'T' for Trade, 'Q' for Quote.

        Returns:
            True if the update was successful, False otherwise.
        """
        await self.ensure_initialized()
        if not self.client:
            logger.error("Redis client not available for update_last_tick_data.")
            return False

        key = f"tick:{symbol}"
        # Prepare data, converting numbers to strings for HSET
        data_to_store = {
            "price": str(price),
            "timestamp": timestamp.isoformat(), # Store timestamp as ISO string
            "event_type": event_type,
            "last_update": datetime.utcnow().isoformat() # Add processing timestamp
        }
        if volume is not None:
            data_to_store["volume"] = str(volume)
        if bid is not None:
            data_to_store["bid"] = str(bid)
        if ask is not None:
            data_to_store["ask"] = str(ask)

        try:
            start_time = time.time()
            # Use HSET with mapping (values must be primitive types or strings/bytes)
            await self.client.hset(key, mapping=data_to_store)
            # Optionally set an expiry for tick data if it should be cleaned up
            # await self.client.expire(key, 3600 * 24) # e.g., expire after 24 hours

            if PROMETHEUS_AVAILABLE:
                REDIS_OPERATION_COUNT.labels(operation="hset", status="success").inc()
                REDIS_OPERATION_LATENCY.labels(operation="hset").observe(time.time() - start_time)

            # logger.debug(f"Stored tick data for {symbol}: {data_to_store}")
            return True
        except Exception as e:
            logger.error(f"Error updating tick data for {symbol} in Redis: {e}")
            if PROMETHEUS_AVAILABLE:
                REDIS_OPERATION_COUNT.labels(operation="hset", status="error").inc()
            return False

    async def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Retrieves the last known price for a symbol from Redis.
        DEPRECATED: Use get_latest_tick_data for more complete info.
        """
        await self.ensure_initialized()
        if not self.client:
            logger.error("Redis client not available for get_last_price.")
            return None

        key = f"tick:{symbol}"
        try:
            price_str = await self.client.hget(key, "price")
            if price_str:
                # price_str is already decoded if decode_responses=True at pool
                return float(price_str)
            return None
        except Exception as e:
            logger.error(f"Error getting last price for {symbol} from Redis: {e}")
            return None

    async def get_latest_tick_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest tick data hash for a symbol from Redis.

        Returns:
            Dictionary containing the latest tick data, or None if not found/error.
            Values are returned as strings (due to Redis storage).
            Timestamps are ISO format strings.
        """
        await self.ensure_initialized()
        if not self.client:
            logger.error("Redis client not available for get_latest_tick_data.")
            return None

        key = f"tick:{symbol}"
        try:
            start_time = time.time()
            # hgetall returns dict[str, str] if decode_responses=True
            tick_data = await self.client.hgetall(key)

            if not tick_data:
                # logger.debug(f"No tick data found for {symbol} in Redis.")
                return None

            # Convert numeric fields back from string if needed by caller
            # For now, return as strings as stored. Caller can convert.
            # Example conversion (do this in the calling code if needed):
            # if 'price' in tick_data: tick_data['price'] = float(tick_data['price'])
            # if 'volume' in tick_data: tick_data['volume'] = int(tick_data['volume'])
            # if 'bid' in tick_data: tick_data['bid'] = float(tick_data['bid'])
            # if 'ask' in tick_data: tick_data['ask'] = float(tick_data['ask'])
            # if 'timestamp' in tick_data: tick_data['timestamp'] = datetime.fromisoformat(tick_data['timestamp'])
            # if 'last_update' in tick_data: tick_data['last_update'] = datetime.fromisoformat(tick_data['last_update'])

            if PROMETHEUS_AVAILABLE:
                REDIS_OPERATION_COUNT.labels(operation="hgetall", status="success").inc()
                REDIS_OPERATION_LATENCY.labels(operation="hgetall").observe(time.time() - start_time)

            return tick_data

        except Exception as e:
            logger.error(f"Error getting latest tick data for {symbol} from Redis: {e}")
            if PROMETHEUS_AVAILABLE:
                REDIS_OPERATION_COUNT.labels(operation="hgetall", status="error").inc()
            return None

    # Removed placeholder get_intraday_data method

    async def close(self):
        """Closes the Redis connection pool."""
        global _async_connection_pool
        if self.client:
            # Closing the client directly might not be necessary if using pool correctly
            # but ensures resources are released if needed.
            try:
                 # Use disconnect() which should handle pool release
                 await self.client.disconnect()
                 logger.info("Closed specific Async Redis client instance.")
            except Exception as e:
                 logger.error(f"Error closing specific Async Redis client: {e}")
            self.client = None

        # Close the pool itself if this RedisClient instance created it
        # This logic is tricky; usually pool is managed globally or by app lifecycle
        # For simplicity, let's assume pool is managed elsewhere and just disconnect client.
        # if _async_connection_pool:
        #     try:
        #         await _async_connection_pool.disconnect()
        #         logger.info("Disconnected Async Redis connection pool.")
        #         _async_connection_pool = None
        #     except Exception as e:
        #         logger.error(f"Error disconnecting Async Redis connection pool: {e}")


class RedisCache:
    """
    Redis-based cache with memory fallback and standardized operations
    (Using Async Redis Client)

    This class provides a hybrid caching solution that combines Redis for distributed caching
    with an in-memory LRU cache for fast access to frequently used data. It includes:

    Features:
        - Automatic fallback to in-memory cache when Redis is unavailable
        - LRU-like memory cache with size limits and automatic cleanup
        - Support for both general data and DataFrame-specific caching
        - Built-in serialization with error handling
        - Prometheus metrics for monitoring cache performance
        - Comprehensive logging for debugging and monitoring

    Performance Optimizations:
        - In-memory cache for fastest access to hot data
        - Efficient key generation using MD5 hashing
        - Batch operations using Redis pipeline when available
        - Automatic cleanup of expired entries
    """

    def __init__(
        self,
        prefix: str = "cache",
        ttl: int = 3600,
        max_memory_items: int = 10000,
        use_json: bool = False,
        redis_client: Optional[aioredis.Redis] = None # Expect async client
    ) -> None:
        """
        Initialize Redis cache with prefix for namespace separation

        Args:
            prefix: Namespace prefix for cache keys
            ttl: Default time-to-live for cache entries in seconds
            max_memory_items: Maximum number of items in memory cache
            use_json: Whether to use JSON serialization instead of pickle
            redis_client: Optional existing async Redis client
        """
        self.prefix = prefix
        self.ttl = ttl
        self.max_memory_items = max_memory_items
        self.use_json = use_json
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.redis_client = redis_client # Store provided client
        self.redis_available = False # Will be set in ensure_initialized
        self._init_task: Optional[asyncio.Task] = None
        self._provided_client = redis_client is not None # Track if client was provided


    async def _initialize(self):
        """Initialize the underlying async Redis client if not provided."""
        if self.redis_client is None:
            try:
                self.redis_client = await get_async_redis_client()
                self.redis_available = True
                logger.info(f"Async Redis cache initialized: {self.prefix} (TTL: {self.ttl}s)")
            except Exception as e:
                logger.error(f"Async Redis cache initialization failed: {e}")
                logger.error(traceback.format_exc())
                self.redis_client = None
                self.redis_available = False
                logger.warning(f"Falling back to in-memory cache only: {self.prefix}")
        elif not self.redis_available: # Check if provided client is actually working
             try:
                  await self.redis_client.ping()
                  self.redis_available = True
                  logger.info(f"Using provided Async Redis client for cache: {self.prefix}")
             except Exception as e:
                  logger.error(f"Provided Async Redis client failed ping: {e}")
                  self.redis_available = False
                  logger.warning(f"Falling back to in-memory cache only despite provided client: {self.prefix}")


    async def ensure_initialized(self):
        """Ensure the client is initialized, handling concurrent initialization."""
        if self.redis_available: # Already initialized and checked
            return
        # Avoid re-initializing if already failed
        if self.redis_client is None and self._init_task and self._init_task.done() and self._init_task.exception():
             return

        if self._init_task is None or self._init_task.done():
            self._init_task = asyncio.create_task(self._initialize())
        try:
             await self._init_task
        except Exception:
             # Initialization failed, ensure redis_available is False
             self.redis_available = False


    async def get(self, key_parts: Union[str, List[Any]]) -> Optional[Any]:
        """
        Get value from cache with fallback to memory cache (async version)

        Args:
            key_parts: Key parts to generate cache key from

        Returns:
            Cached value if found and not expired, None otherwise
        """
        try:
            start_time = time.time()
            key = generate_key(self.prefix, key_parts)
            # logger.debug(f"Attempting to get cache key: {key}")

            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry["expiry"]:
                    self.hits += 1
                    if PROMETHEUS_AVAILABLE:
                        REDIS_CACHE_HIT_COUNT.labels(cache_type="memory").inc()
                    # logger.debug(f"Memory cache hit for key: {key}")
                    return entry["value"]
                # logger.debug(f"Memory cache entry expired for key: {key}")
                del self.memory_cache[key]

            # Try Redis if available
            await self.ensure_initialized() # Make sure client is ready
            if self.redis_available and self.redis_client:
                try:
                    # Data will be string if decode_responses=True at pool
                    data = await self.redis_client.get(key)
                    if data:
                        self.hits += 1
                        if PROMETHEUS_AVAILABLE:
                            REDIS_CACHE_HIT_COUNT.labels(cache_type="redis").inc()

                        # Deserialize (handle string data)
                        value = deserialize_value(data, self.use_json)
                        if value is not None:
                            self.memory_cache[key] = {
                                "value": value,
                                "expiry": time.time() + self.ttl,
                            }
                            # logger.debug(f"Redis cache hit for key: {key}")
                            return value
                except redis.RedisError as e:
                    logger.error(f"Redis error retrieving key {key}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error retrieving from Redis: {e}")

            self.misses += 1
            if PROMETHEUS_AVAILABLE:
                REDIS_CACHE_MISS_COUNT.labels(cache_type="all").inc()
                latency = time.time() - start_time
                REDIS_OPERATION_LATENCY.labels(operation="get").observe(latency)

            # logger.debug(f"Cache miss for key: {key}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in async cache get: {e}")
            logger.error(f"Key parts: {key_parts}")
            logger.error(traceback.format_exc())
            return None


    async def set(
        self,
        key_parts: Union[str, List[Any]],
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with optional TTL (async version)

        Args:
            key_parts: Key parts to generate cache key from
            value: Value to store in cache
            ttl: Optional time-to-live in seconds (defaults to self.ttl)

        Returns:
            bool: True if value was successfully stored, False otherwise
        """
        try:
            start_time = time.time()

            if value is None:
                logger.warning("Attempted to cache None value")
                return False

            key = generate_key(self.prefix, key_parts)
            ttl = ttl or self.ttl
            expiry = time.time() + ttl

            # logger.debug(f"Setting cache key: {key} (TTL: {ttl}s)")

            # LRU-like memory cache cleanup
            if len(self.memory_cache) >= self.max_memory_items:
                keys_to_remove = random.sample(
                    list(self.memory_cache.keys()),
                    int(len(self.memory_cache) * 0.1),
                )
                for k in keys_to_remove:
                    del self.memory_cache[k]
                # logger.debug(f"Memory cache cleanup: removed {len(keys_to_remove)} entries")

            self.memory_cache[key] = {
                "value": value,
                "expiry": expiry,
            }

            # Store in Redis if available
            await self.ensure_initialized()
            if self.redis_available and self.redis_client:
                try:
                    # Serialize (returns bytes)
                    serialized = serialize_value(value, self.use_json)
                    # Setex expects bytes for value
                    await self.redis_client.setex(key, ttl, serialized)

                    if PROMETHEUS_AVAILABLE:
                        REDIS_OPERATION_COUNT.labels(operation="set", status="success").inc()
                        REDIS_OPERATION_LATENCY.labels(operation="set").observe(time.time() - start_time)

                    # logger.debug(f"Value stored in Redis: {key}")
                    return True

                except redis.RedisError as e:
                    logger.error(f"Redis error storing key {key}: {e}")
                    if PROMETHEUS_AVAILABLE:
                        REDIS_OPERATION_COUNT.labels(operation="set", status="error").inc()
                except Exception as e:
                    logger.error(f"Unexpected error storing in Redis: {e}")
                    logger.error(traceback.format_exc())
                    if PROMETHEUS_AVAILABLE:
                        REDIS_OPERATION_COUNT.labels(operation="set", status="error").inc()

            return True # Return True if memory cache updated

        except Exception as e:
            logger.error(f"Unexpected error in async cache set: {e}")
            logger.error(f"Key parts: {key_parts}")
            logger.error(traceback.format_exc())
            return False


    async def delete(self, key_parts: Union[str, List[Any]]) -> bool:
        """
        Delete value from cache (async version)

        Args:
            key_parts: Key parts to generate cache key from

        Returns:
            bool: True if value was successfully deleted, False otherwise
        """
        try:
            key = generate_key(self.prefix, key_parts)
            # logger.debug(f"Deleting cache key: {key}")

            if key in self.memory_cache:
                del self.memory_cache[key]

            await self.ensure_initialized()
            if self.redis_available and self.redis_client:
                try:
                    await self.redis_client.delete(key)
                    # logger.debug(f"Deleted from Redis: {key}")
                except redis.RedisError as e:
                    logger.error(f"Redis error deleting key {key}: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Unexpected error in async cache delete: {e}")
            logger.error(f"Key parts: {key_parts}")
            logger.error(traceback.format_exc())
            return False


    async def store_dataframe(
        self,
        key_parts: Union[str, List[Any]],
        df: pd.DataFrame,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store DataFrame in cache with optimized serialization (async version)

        Args:
            key_parts: Key parts to generate cache key from
            df: DataFrame to store
            ttl: Optional time-to-live in seconds (defaults to self.ttl)

        Returns:
            bool: True if DataFrame was successfully stored, False otherwise
        """
        try:
            if df is None or df.empty:
                logger.warning("Attempted to cache empty DataFrame")
                return False
            # Use pickle for DataFrames by default
            return await self.set(key_parts, df, ttl)
        except Exception as e:
            logger.error(f"Error storing DataFrame: {e}")
            logger.error(traceback.format_exc())
            return False


    async def get_dataframe(self, key_parts: Union[str, List[Any]]) -> Optional[pd.DataFrame]:
        """
        Get DataFrame from cache (async version)

        Args:
            key_parts: Key parts to generate cache key from

        Returns:
            DataFrame if found and not expired, None otherwise
        """
        try:
            # Use pickle for DataFrames by default
            result = await self.get(key_parts)

            # Check if the result is already a DataFrame (from memory cache or pickle)
            if isinstance(result, pd.DataFrame):
                return result
            # Handle potential JSON storage if use_json=True was used during set
            elif isinstance(result, list) and self.use_json:
                 try: return pd.DataFrame(result)
                 except Exception: logger.error("Failed to convert list to DataFrame"); return None
            elif isinstance(result, dict) and self.use_json:
                 try: return pd.DataFrame.from_dict(result)
                 except Exception: logger.error("Failed to convert dict to DataFrame"); return None


            if result is not None:
                 logger.warning(f"Cache returned non-DataFrame type for get_dataframe: {type(result)}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving DataFrame: {e}")
            logger.error(traceback.format_exc())
            return None


    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching pattern (async version)

        Args:
            pattern: Optional pattern to match keys (e.g., "user:*")
                     If None, clears all keys with this cache's prefix

        Returns:
            Number of keys cleared
        """
        try:
            # Clear memory cache
            keys_to_remove_mem = []
            if pattern:
                # Adjust pattern matching for memory cache keys (hashed)
                # This is difficult with hashed keys. We might need to iterate all memory keys.
                # For simplicity, clear all memory cache if pattern is used, or implement prefix matching on hash.
                # Let's clear all memory cache for this prefix if pattern is provided.
                logger.warning("Pattern matching for memory cache clear is approximate due to hashing.")
                keys_to_remove_mem = [k for k in self.memory_cache.keys() if k.startswith(f"{self.prefix}:")]

            else:
                # Clear all keys starting with the prefix
                keys_to_remove_mem = [k for k in self.memory_cache.keys() if k.startswith(f"{self.prefix}:")]


            for k in keys_to_remove_mem:
                del self.memory_cache[k]
            memory_cleared = len(keys_to_remove_mem)
            # logger.debug(f"Cleared {memory_cleared} keys from memory cache")

            # Clear Redis cache if available
            redis_cleared = 0
            await self.ensure_initialized()
            if self.redis_available and self.redis_client:
                try:
                    # Use direct pattern matching for Redis SCAN
                    redis_search_pattern = f"{self.prefix}:{pattern or '*'}"

                    total_deleted = 0
                    # Use pipeline for batch deletion
                    async with self.redis_client.pipeline(transaction=False) as pipe:
                         keys_to_delete_redis = []
                         async for key in self.redis_client.scan_iter(match=redis_search_pattern, count=500):
                              keys_to_delete_redis.append(key)
                              if len(keys_to_delete_redis) >= 500: # Batch delete
                                   await pipe.delete(*keys_to_delete_redis)
                                   total_deleted += len(keys_to_delete_redis)
                                   keys_to_delete_redis = []
                         # Delete remaining keys
                         if keys_to_delete_redis:
                              await pipe.delete(*keys_to_delete_redis)
                              total_deleted += len(keys_to_delete_redis)
                         await pipe.execute() # Execute the pipeline

                    redis_cleared = total_deleted
                    # logger.debug(f"Cleared {redis_cleared} keys from Redis matching '{redis_search_pattern}'")

                except redis.RedisError as e:
                    logger.error(f"Redis error clearing keys: {e}")

            return memory_cleared + redis_cleared
        except Exception as e:
            logger.error(f"Unexpected error clearing async cache: {e}")
            logger.error(traceback.format_exc())
            return 0


    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics (async version)

        Returns:
            Dictionary with cache statistics
        """
        try:
            stats = {
                "prefix": self.prefix,
                "memory_cache_size": len(self.memory_cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                "redis_available": self.redis_available
            }

            await self.ensure_initialized()
            if self.redis_available and self.redis_client:
                try:
                    key_count = 0
                    async for _ in self.redis_client.scan_iter(match=f"{self.prefix}:*", count=500):
                         key_count += 1
                    stats["redis_key_count"] = key_count

                    info = await self.redis_client.info()
                    stats["redis_used_memory"] = info.get("used_memory", 0)
                    stats["redis_max_memory"] = info.get("maxmemory", 0)

                    if PROMETHEUS_AVAILABLE:
                        REDIS_CACHE_SIZE.labels(prefix=self.prefix).set(key_count)

                except redis.RedisError as e:
                    logger.error(f"Redis error getting stats: {e}")

            return stats
        except Exception as e:
            logger.error(f"Unexpected error getting async cache stats: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    async def close(self):
         """Closes the underlying Redis client if it was created by this instance."""
         if self._init_task and not self._init_task.done():
              try:
                   await self._init_task
              except Exception:
                   pass

         # Only close if the client wasn't provided externally
         if self.redis_client and not self._provided_client:
              try:
                   await self.redis_client.disconnect() # Use disconnect for pool clients
                   logger.info(f"Closed internal Async Redis client for cache {self.prefix}")
              except Exception as e:
                   logger.error(f"Error closing internal Async Redis client for cache {self.prefix}: {e}")
         self.redis_client = None
         self.redis_available = False


# --- Standalone Functions (Consider migrating relevant ones into RedisClient or removing if redundant) ---

async def async_send_notification(
    channel: str,
    message: Dict[str, Any],
    redis_client: Optional[aioredis.Redis] = None
) -> bool:
    """
    Send a notification message to a Redis pub/sub channel (async version)

    Args:
        channel: Redis channel to publish to
        message: Message to send (will be JSON serialized)
        redis_client: Optional async Redis client to use

    Returns:
        bool: True if message was successfully sent, False otherwise
    """
    temp_client_created = False
    client = redis_client
    if client is None:
         try:
              client = await get_async_redis_client()
              temp_client_created = True
         except Exception as e:
              logger.error(f"Failed to get Redis client for notification: {e}")
              return False

    try:
        start_time = time.time()

        if "timestamp" not in message:
            message["timestamp"] = time.time()

        serialized = json.dumps(message)
        result = await client.publish(channel, serialized)

        if PROMETHEUS_AVAILABLE:
            REDIS_OPERATION_COUNT.labels(operation="publish", status="success").inc()
            REDIS_OPERATION_LATENCY.labels(operation="publish").observe(time.time() - start_time)

        # logger.debug(f"Published message to channel {channel}: {result} receivers")
        return result > 0 # result is number of receivers

    except Exception as e:
        logger.error(f"Error publishing async message to channel {channel}: {e}")
        logger.error(traceback.format_exc())
        if PROMETHEUS_AVAILABLE:
            REDIS_OPERATION_COUNT.labels(operation="publish", status="error").inc()
        return False
    finally:
         # Close temporary client if created
         if temp_client_created and client:
              await client.disconnect()


async def async_batch_get(
    keys: List[str],
    redis_client: Optional[aioredis.Redis] = None,
    use_json: bool = False
) -> Dict[str, Any]:
    """
    Get multiple values from Redis in a single batch operation (async version)

    Args:
        keys: List of keys to retrieve
        redis_client: Optional async Redis client to use
        use_json: Whether to use JSON deserialization

    Returns:
        Dictionary mapping keys to values
    """
    temp_client_created = False
    client = redis_client
    if client is None:
         try:
              client = await get_async_redis_client()
              temp_client_created = True
         except Exception as e:
              logger.error(f"Failed to get Redis client for batch_get: {e}")
              return {}

    try:
        start_time = time.time()

        if not keys:
            return {}

        # Values will be strings if decode_responses=True
        values = await client.mget(keys)
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = deserialize_value(value, use_json)

        if PROMETHEUS_AVAILABLE:
            REDIS_OPERATION_COUNT.labels(operation="mget", status="success").inc()
            REDIS_OPERATION_LATENCY.labels(operation="mget").observe(time.time() - start_time)

        return result
    except Exception as e:
        logger.error(f"Error async batch getting keys: {e}")
        logger.error(traceback.format_exc())
        if PROMETHEUS_AVAILABLE:
            REDIS_OPERATION_COUNT.labels(operation="mget", status="error").inc()
        return {}
    finally:
         if temp_client_created and client:
              await client.disconnect()


async def async_batch_set(
    key_values: Dict[str, Any],
    ttl: int = 3600,
    redis_client: Optional[aioredis.Redis] = None,
    use_json: bool = False
) -> bool:
    """
    Set multiple values in Redis in a single batch operation (async version)

    Args:
        key_values: Dictionary mapping keys to values
        ttl: Time-to-live in seconds
        redis_client: Optional async Redis client to use
        use_json: Whether to use JSON serialization

    Returns:
        bool: True if values were successfully stored, False otherwise
    """
    temp_client_created = False
    client = redis_client
    if client is None:
         try:
              client = await get_async_redis_client()
              temp_client_created = True
         except Exception as e:
              logger.error(f"Failed to get Redis client for batch_set: {e}")
              return False

    try:
        start_time = time.time()

        if not key_values:
            return True

        # Serialize values (returns bytes)
        serialized = {}
        for key, value in key_values.items():
            serialized[key] = serialize_value(value, use_json)

        async with client.pipeline(transaction=True) as pipe:
            # mset requires bytes for values
            await pipe.mset(serialized)
            for key in serialized:
                await pipe.expire(key, ttl)
            await pipe.execute()

        if PROMETHEUS_AVAILABLE:
            REDIS_OPERATION_COUNT.labels(operation="mset", status="success").inc()
            REDIS_OPERATION_LATENCY.labels(operation="mset").observe(time.time() - start_time)

        return True
    except Exception as e:
        logger.error(f"Error async batch setting keys: {e}")
        logger.error(traceback.format_exc())
        if PROMETHEUS_AVAILABLE:
            REDIS_OPERATION_COUNT.labels(operation="mset", status="error").inc()
        return False
    finally:
         if temp_client_created and client:
              await client.disconnect()


async def async_get_lock(
    name: str,
    timeout: int = 10,
    blocking_timeout: int = 5,
    redis_client: Optional[aioredis.Redis] = None
) -> Optional[aioredis.lock.Lock]:
    """
    Get a distributed lock using Redis (async version)

    Args:
        name: Lock name
        timeout: Lock timeout in seconds
        blocking_timeout: Maximum time to wait for lock acquisition
        redis_client: Optional async Redis client to use

    Returns:
        Async Redis lock object if acquired, None otherwise
    """
    # Note: Lock needs the client to persist, so creating temporary client here is problematic.
    # Caller should manage the client lifecycle for locks.
    if redis_client is None:
         logger.error("Redis client must be provided for async_get_lock")
         # Alternatively, manage a global client, but passing is cleaner.
         # client = await get_async_redis_client() # Avoid this if possible
         return None
    client = redis_client

    try:
        lock = client.lock(f"lock:{name}", timeout=timeout)

        if await lock.acquire(blocking=True, blocking_timeout=blocking_timeout):
            # logger.debug(f"Acquired lock: {name}")
            return lock
        else:
            logger.warning(f"Failed to acquire lock: {name}")
            return None
    except Exception as e:
        logger.error(f"Error getting async lock {name}: {e}")
        logger.error(traceback.format_exc())
        return None


async def async_release_lock(lock: aioredis.lock.Lock) -> bool:
    """
    Release a distributed lock (async version)

    Args:
        lock: Async Redis lock object

    Returns:
        bool: True if lock was successfully released, False otherwise
    """
    try:
        await lock.release()
        # logger.debug(f"Released lock: {lock.name}")
        return True
    # Catch specific LockError if redis-py version supports it well
    except redis.exceptions.LockError as le:
         logger.warning(f"Error releasing async lock {lock.name}: {le} (Possibly already released)")
         return False # Or True depending on desired idempotency
    except Exception as e:
        logger.error(f"Error releasing async lock {lock.name}: {e}")
        logger.error(traceback.format_exc())
        return False


# Create default async cache instance for common use
# Note: Initialization needs to be awaited where it's first used or in an async context
default_async_cache = RedisCache(prefix="app_cache_async")

# --- Deprecated Sync Functions (Mark or Remove) ---
# Keep sync functions for now if legacy code depends on them, but prefer async

# Global connection pool for reuse across modules (Sync version)
_sync_connection_pool = None

def get_sync_connection_pool() -> redis.ConnectionPool:
    """ Get or create a sync Redis connection pool """
    global _sync_connection_pool
    if _sync_connection_pool is None:
        try:
            _sync_connection_pool = redis.ConnectionPool(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                username=REDIS_USERNAME, password=REDIS_PASSWORD,
                socket_timeout=REDIS_TIMEOUT, socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=True, # Sync client might expect decoded
                max_connections=REDIS_MAX_CONNECTIONS
            )
            logger.info(f"Created Sync Redis connection pool: {REDIS_HOST}:{REDIS_PORT} (DB: {REDIS_DB})")
        except Exception as e:
            logger.error(f"Failed to create Sync Redis connection pool: {e}")
            raise
    return _sync_connection_pool

def get_redis_client() -> redis.Redis:
     """ Get a sync Redis client using the shared sync connection pool """
     logger.warning("get_redis_client (sync) is deprecated. Use RedisClient or get_async_redis_client.")
     try:
          pool = get_sync_connection_pool()
          client = redis.Redis(connection_pool=pool)
          client.ping() # Test connection
          # Sync metrics update omitted for brevity, focus on async
          return client
     except Exception as e:
          logger.error(f"Failed to get Sync Redis client: {e}")
          raise

# Keep other sync functions like send_notification, batch_get, batch_set, get_lock, release_lock
# if they are still needed by non-async parts of the codebase. Mark as deprecated if possible.

def send_notification(*args, **kwargs): logger.warning("Sync send_notification is deprecated"); return False
def batch_get(*args, **kwargs): logger.warning("Sync batch_get is deprecated"); return {}
def batch_set(*args, **kwargs): logger.warning("Sync batch_set is deprecated"); return False
def get_lock(*args, **kwargs): logger.warning("Sync get_lock is deprecated"); return None
def release_lock(*args, **kwargs): logger.warning("Sync release_lock is deprecated"); return False

# Deprecated default cache
default_cache = None # Remove or mark sync default_cache as deprecated
logger.warning("Sync default_cache is deprecated. Use default_async_cache.")
