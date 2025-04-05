"""
Asynchronous Redis Cache Implementation for AI Day Trader
"""

import asyncio
import hashlib
import pickle
import random
import time
import logging
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as aioredis
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None # type: ignore
    PANDAS_AVAILABLE = False


# Import from new structure
from ai_day_trader.utils.config import Config # Corrected import path for base Config
from ai_day_trader.config import load_ai_trader_config # Import loader
# Use the redis client functions from the new location
from ai_day_trader.clients.redis_client import get_async_redis_client

logger = logging.getLogger("ai_day_trader.clients.redis_cache")

class AsyncRedisCache:
    """Asynchronous Redis-based cache with in-memory fallback"""

    def __init__(
        self,
        prefix: str = "ai_cache",
        config: Optional[Config] = None, # Use the imported base Config
        ttl: int = 3600,
        max_memory_items: int = 1000,
    ) -> None:
        """Initialize the cache"""
        self.prefix = prefix
        self.config = config or load_ai_trader_config() # Load if not provided
        self.ttl = ttl
        self.enabled = False
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.size_limit = max_memory_items
        self.redis_client: Optional[aioredis.Redis] = None
        self._init_lock = asyncio.Lock()
        self._init_task: Optional[asyncio.Task] = None
        logger.info(f"AsyncRedisCache instance created with prefix '{prefix}'. Call _ensure_initialized() before use.")

    async def _ping_redis(self) -> bool:
        """Internal helper to ping redis, returns True if successful."""
        if self.redis_client:
            try:
                return await self.redis_client.ping()
            except Exception:
                logger.warning("Redis ping failed.")
                self.redis_client = None
        return False

    async def _ensure_initialized(self) -> bool:
        """Ensure Redis client is initialized using the shared pool."""
        if await self._ping_redis():
            return True

        async with self._init_lock:
            if await self._ping_redis():
                 return True

            if self._init_task and not self._init_task.done():
                try:
                    await asyncio.wait_for(self._init_task, timeout=15.0)
                    return await self._ping_redis()
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for existing Redis initialization task.")
                    return False
                except Exception as e:
                    logger.error(f"Error waiting for existing Redis init task: {e}")
                    return False
                finally:
                     self._init_task = None

            async def _init():
                try:
                    self.redis_client = await get_async_redis_client()
                    if self.redis_client and await self.redis_client.ping():
                         self.enabled = True
                         logger.info("AsyncRedisCache initialized with shared Redis client.")
                    else:
                         logger.error("Failed to get valid Redis client or ping failed after init.")
                         self.redis_client = None
                         self.enabled = False
                except Exception as e:
                    logger.error(f"Failed to get/ping shared async Redis client: {e}", exc_info=True)
                    self.redis_client = None
                    self.enabled = False

            self._init_task = asyncio.create_task(_init())
            try:
                await asyncio.wait_for(self._init_task, timeout=15.0)
            except asyncio.TimeoutError:
                logger.error("Timeout during new Redis client initialization.")
                self.redis_client = None; self.enabled = False
            except Exception as e:
                logger.error(f"Error during new Redis client initialization: {e}", exc_info=True)
                self.redis_client = None; self.enabled = False
            finally:
                 self._init_task = None

            return self.enabled and self.redis_client is not None

    def _generate_key(self, key_parts: Union[str, List[Any]]) -> str:
        """Generate a consistent cache key from parts"""
        if isinstance(key_parts, str):
            key_str = key_parts
        else:
            try:
                key_str = ":".join([str(part) for part in key_parts])
            except Exception:
                 key_str = repr(key_parts)
        full_key_str = f"{self.prefix}:{key_str}"
        return f"{self.prefix}:{hashlib.md5(full_key_str.encode()).hexdigest()}"

    async def get(self, key_parts: Union[str, List[Any]]) -> Optional[Any]:
        """Get value from cache with fallback to Redis"""
        key = self._generate_key(key_parts)
        try:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry["expiry"]:
                    self.hits += 1
                    logger.debug(f"Memory cache hit for key {key_parts}")
                    return entry["value"]
                else:
                    logger.debug(f"Memory cache expired for key {key_parts}")
                    del self.memory_cache[key]

            if await self._ensure_initialized() and self.redis_client:
                data = await self.redis_client.get(key)
                if data:
                    self.hits += 1
                    logger.debug(f"Redis cache hit for key {key_parts}")
                    try:
                        value = pickle.loads(data)
                        self._store_in_memory(key, value, self.ttl)
                        return value
                    except pickle.UnpicklingError:
                         logger.error(f"Failed to unpickle data from Redis for key {key}")
                    except Exception as e:
                         logger.error(f"Error processing data from Redis for key {key}: {e}")
                else:
                     logger.debug(f"Redis cache miss for key {key_parts}")

            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Error in cache get for key {key_parts}: {e}", exc_info=True)
            return None

    def _store_in_memory(self, key: str, value: Any, ttl: int):
         """Stores an item in the in-memory cache with LRU-like eviction."""
         if len(self.memory_cache) >= self.size_limit:
             keys_to_remove = random.sample(
                 list(self.memory_cache.keys()),
                 max(1, int(len(self.memory_cache) * 0.1))
             )
             for k in keys_to_remove:
                 self.memory_cache.pop(k, None)
             logger.debug(f"Memory cache reached size limit ({self.size_limit}), evicted {len(keys_to_remove)} items.")
         self.memory_cache[key] = {"value": value, "expiry": time.time() + ttl}


    async def set(
        self, key_parts: Union[str, List[Any]], value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL"""
        key = self._generate_key(key_parts)
        ttl = ttl or self.ttl
        try:
            self._store_in_memory(key, value, ttl)

            if await self._ensure_initialized() and self.redis_client:
                serialized = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, serialized)
                logger.debug(f"Stored key {key_parts} in Redis with TTL {ttl}s")
            else:
                 logger.warning(f"Redis not initialized, key {key_parts} only stored in memory cache.")

            return True
        except Exception as e:
            logger.error(f"Error in cache set for key {key_parts}: {e}", exc_info=True)
            self.memory_cache.pop(key, None)
            return False

    async def store_dataframe(
        self, key_parts: Union[str, List[Any]], df: Any, ttl: Optional[int] = None
    ) -> bool:
        """Store DataFrame in cache using pickle."""
        if not PANDAS_AVAILABLE:
             logger.error("Pandas not available, cannot store DataFrame.")
             return False
        # Check type only if pandas is available and df is not None
        # Use pd directly here since PANDAS_AVAILABLE is True if we reach this
        if pd is not None and df is not None and not isinstance(df, pd.DataFrame):
             logger.error(f"Value provided to store_dataframe for key {key_parts} is not a DataFrame (type: {type(df)}).")
             return False
        if df is None:
             logger.warning(f"Attempted to cache None DataFrame for key {key_parts}")
             return False
        return await self.set(key_parts, df, ttl)

    async def get_dataframe(self, key_parts: Union[str, List[Any]]) -> Optional[Any]:
        """Retrieve DataFrame from cache."""
        if not PANDAS_AVAILABLE:
             logger.error("Pandas not available, cannot retrieve DataFrame.")
             return None
        value = await self.get(key_parts)
        # Check PANDAS_AVAILABLE and pd is not None before isinstance check
        if value is not None and PANDAS_AVAILABLE and pd is not None and isinstance(value, pd.DataFrame):
            return value
        elif value is not None:
             logger.warning(f"Cache key {key_parts} contained non-DataFrame data: {type(value)}")
        return None

    async def delete(self, key_parts: Union[str, List[Any]]) -> bool:
        """Delete item from cache"""
        key = self._generate_key(key_parts)
        try:
            self.memory_cache.pop(key, None)

            if await self._ensure_initialized() and self.redis_client:
                await self.redis_client.delete(key)
            logger.debug(f"Deleted key {key_parts} from cache.")
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key_parts} from cache: {e}", exc_info=True)
            return False

    async def ping(self) -> bool:
        """Check Redis connection"""
        try:
            if not await self._ensure_initialized() or not self.redis_client:
                return False
            return await self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis ping failed: {e}")
            return False

    async def close(self) -> None:
        """Closes the Redis client connection if managed internally (no longer the case)."""
        logger.info("AsyncRedisCache close called (shared client managed externally).")
        self.enabled = False
        if self._init_task and not self._init_task.done():
            try:
                self._init_task.cancel()
                await asyncio.wait_for(asyncio.shield(self._init_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError): pass
            except Exception as e: logger.warning(f"Error cancelling init task: {e}")
        self.redis_client = None
