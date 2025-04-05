"""Tests for ai_day_trader.clients.polygon_rest_client"""
import pytest
import pandas as pd
import asyncio # Added import
from datetime import datetime, timedelta, timezone
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient
from ai_day_trader.config import load_ai_trader_config
# Import redis pool initializer
from ai_day_trader.clients.redis_client import init_redis_pool, close_redis_pool

# Requires Polygon API key in .env

# Change scope to function to avoid loop closed errors during teardown
@pytest.fixture(scope="function")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Change scope to function to match event_loop fixture
@pytest.fixture(scope="function")
async def polygon_client():
    """Provides an initialized PolygonRESTClient instance."""
    config = load_ai_trader_config()
    # Initialize global redis pool needed by internal cache
    await init_redis_pool(config)
    client = PolygonRESTClient(config=config)
    await client.connection_pool.initialize() # Ensure pool is initialized
    # Explicitly initialize the internal cache after redis pool is ready
    if hasattr(client, 'cache') and client.cache and hasattr(client.cache, '_ensure_initialized'):
        await client.cache._ensure_initialized()
    yield client
    await client.close()
    await close_redis_pool() # Clean up redis pool

@pytest.mark.asyncio
async def test_get_aggregates_valid(polygon_client: PolygonRESTClient):
    """Test fetching valid aggregate data."""
    symbol = "AAPL"
    to_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
    df = await polygon_client.get_aggregates(symbol, 1, "day", from_date, to_date, limit=5)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'c' in df.columns # Polygon uses 'c' for close in aggregates

# Add more tests for other methods, edge cases, caching, etc.
