"""Tests for ai_day_trader.clients.redis_cache"""
import pytest
import pandas as pd
import asyncio
from ai_day_trader.clients.redis_cache import AsyncRedisCache
from ai_day_trader.config import load_ai_trader_config
# Import redis pool initializer
from ai_day_trader.clients.redis_client import init_redis_pool, close_redis_pool

# Requires a running Redis instance

# Remove custom event_loop fixture, use default function-scoped one

# Change scope to function to align with default event loop
@pytest.fixture(scope="function")
async def redis_cache():
    """Provides an initialized AsyncRedisCache instance."""
    config = load_ai_trader_config()
    # Initialize the global pool first
    await init_redis_pool(config)
    cache = AsyncRedisCache(config=config, prefix="test_cache")
    # Ensure connection is established within the cache instance
    initialized = await cache._ensure_initialized()
    if not initialized:
        await close_redis_pool() # Clean up pool if init failed
        pytest.skip("Redis connection failed, skipping cache tests.")
    yield cache
    await cache.close()
    await close_redis_pool() # Clean up global pool

@pytest.mark.asyncio
async def test_cache_set_get_delete(redis_cache: AsyncRedisCache):
    """Test basic set, get, and delete operations."""
    key = "my_test_key"
    value = {"a": 1, "b": "hello"}
    set_success = await redis_cache.set(key, value, ttl=5)
    assert set_success is True

    retrieved_value = await redis_cache.get(key)
    assert retrieved_value == value

    delete_success = await redis_cache.delete(key)
    assert delete_success is True

    retrieved_again = await redis_cache.get(key)
    assert retrieved_again is None

@pytest.mark.asyncio
async def test_cache_dataframe(redis_cache: AsyncRedisCache):
    """Test storing and retrieving a pandas DataFrame."""
    key = "my_dataframe_key"
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    store_success = await redis_cache.store_dataframe(key, df, ttl=5)
    assert store_success is True

    retrieved_df = await redis_cache.get_dataframe(key)
    assert retrieved_df is not None
    pd.testing.assert_frame_equal(df, retrieved_df)

    await redis_cache.delete(key)

# Add tests for TTL expiry, different data types, etc.
