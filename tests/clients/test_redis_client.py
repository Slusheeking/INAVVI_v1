"""Tests for ai_day_trader.clients.redis_client"""
import pytest
from ai_day_trader.clients.redis_client import init_redis_pool, get_async_redis_client, close_redis_pool
from ai_day_trader.config import load_ai_trader_config

# Requires a running Redis instance accessible

@pytest.mark.asyncio
async def test_redis_connection():
    """Test initializing pool and getting a client."""
    config = load_ai_trader_config()
    await init_redis_pool(config)
    client = None
    try:
        client = await get_async_redis_client()
        assert client is not None
        assert await client.ping() is True
    finally:
        # No need to close the client explicitly when using a pool
        # The pool manages the connections.
        # if client:
        #     await client.aclose() # Incorrect for pooled connections
        await close_redis_pool()

# Add tests for set/get helpers if needed
