"""
Redis Client for AI Day Trader.

Handles connection pooling and provides an async Redis client instance.
"""
import asyncio # Add missing import
import logging
import os
from typing import Optional, Any
# Removed top-level redis import, use aioredis directly
import redis.asyncio as aioredis # Use alias for async client
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError # Import specific exception

from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.utils.exceptions import ConfigurationError # Use new utils path

logger = logging.getLogger(__name__)

# Global variable to hold the connection pool
_redis_pool: Optional[ConnectionPool] = None

async def init_redis_pool(config: Config):
    """Initializes the Redis connection pool."""
    global _redis_pool
    if _redis_pool is not None:
        # Ensure the existing pool is healthy before returning
        try:
            async with aioredis.Redis(connection_pool=_redis_pool) as test_client:
                if await test_client.ping():
                    logger.warning("Redis pool already initialized and healthy.")
                    return
                else:
                    logger.warning("Existing Redis pool ping failed. Re-initializing.")
                    await close_redis_pool() # Close potentially broken pool
        except Exception as e:
            logger.warning(f"Error checking existing Redis pool health ({e}). Re-initializing.")
            await close_redis_pool() # Close potentially broken pool

    redis_host = config.get("REDIS_HOST", "localhost")
    redis_port = config.get_int("REDIS_PORT", 6379)
    redis_db = config.get_int("REDIS_DB", 0)
    redis_password = config.get("REDIS_PASSWORD")
    max_connections = config.get_int("REDIS_MAX_CONNECTIONS", 10)

    logger.info(f"Creating new Redis connection pool to {redis_host}:{redis_port} (DB: {redis_db})")
    try:
        # Create pool using aioredis
        _redis_pool = aioredis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            max_connections=max_connections,
            decode_responses=False # Keep as bytes for flexibility, decode where needed
        )
        # Test connection using a client from the new pool
        async with aioredis.Redis(connection_pool=_redis_pool) as temp_client:
            await temp_client.ping()
        # Note: No need to explicitly close temp_client here as 'async with' handles it.
        logger.info(f"Successfully connected to Redis and initialized pool.")
    # Catch specific aioredis connection error
    except RedisConnectionError as e: # Use imported exception
        logger.error(f"Failed to connect to Redis at {redis_host}:{redis_port}: {e}", exc_info=True)
        _redis_pool = None # Ensure pool is None on failure
        raise ConnectionError(f"Failed to connect to Redis: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during Redis pool initialization: {e}", exc_info=True)
        _redis_pool = None
        raise ConfigurationError(f"Redis configuration error: {e}") from e

async def get_async_redis_client() -> aioredis.Redis: # Use alias in type hint
    """Gets an async Redis client instance from the pool."""
    if _redis_pool is None:
        raise ConnectionError("Redis connection pool not initialized. Call init_redis_pool first.")
    # Create a new client instance connected to the pool
    return aioredis.Redis(connection_pool=_redis_pool) # Use alias

async def close_redis_pool():
    """Closes the Redis connection pool."""
    global _redis_pool
    pool_to_close = _redis_pool
    _redis_pool = None # Set global to None immediately
    if pool_to_close:
        logger.info("Closing Redis connection pool...")
        try:
            # Use the connection pool's disconnect method
            await pool_to_close.disconnect()
            logger.info("Redis connection pool closed.")
        except Exception as e:
            logger.error(f"Error disconnecting Redis pool: {e}", exc_info=True)
    else:
        logger.info("Redis connection pool already closed or not initialized.")
        # Return a completed awaitable to satisfy type checkers in calling code
        return await asyncio.sleep(0)

# Example helper function (can add more as needed)
async def set_redis_key(key: str, value: Any, expire: Optional[int] = None):
    """Sets a key in Redis, optionally with an expiry, using async with."""
    try:
        # Use async with to get a client and automatically handle release
        async with await get_async_redis_client() as client:
            await client.set(key, value, ex=expire)
    except RedisConnectionError as e:
         logger.error(f"Redis connection error setting key '{key}': {e}")
         # Optionally re-raise or handle specific connection issues
    except Exception as e:
        logger.error(f"Error setting Redis key '{key}': {e}", exc_info=True)


async def get_redis_key(key: str) -> Optional[bytes]:
    """Gets a key from Redis using async with."""
    try:
        # Use async with to get a client and automatically handle release
        async with await get_async_redis_client() as client:
            value = await client.get(key)
            return value
    except RedisConnectionError as e:
         logger.error(f"Redis connection error getting key '{key}': {e}")
         return None # Return None on connection error
    except Exception as e:
        logger.error(f"Error getting Redis key '{key}': {e}", exc_info=True)
        return None
