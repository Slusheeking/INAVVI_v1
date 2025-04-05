"""Tests for ai_day_trader.clients.async_connection_pool"""
import pytest
import asyncio # Added import
from ai_day_trader.clients.async_connection_pool import AsyncConnectionPool
from ai_day_trader.config import load_ai_trader_config

@pytest.mark.asyncio
async def test_pool_initialization():
    """Test initializing the connection pool."""
    config = load_ai_trader_config()
    pool = AsyncConnectionPool(config=config)
    initialized = await pool.initialize()
    assert initialized is True
    assert pool.session is not None
    await pool.close()

@pytest.mark.asyncio
async def test_pool_get_request():
    """Test making a simple GET request."""
    # This requires an external service or mocking
    config = load_ai_trader_config()
    pool = AsyncConnectionPool(config=config)
    await pool.initialize()
    try:
        # Example: fetch google.com (might fail due to SSL/headers if not configured)
        # A better test would use a local mock server
        response = await pool.get("https://httpbin.org/get") # Use httpbin for testing
        assert isinstance(response, dict)
        assert "url" in response
        assert response["url"] == "https://httpbin.org/get"
    finally:
        await pool.close()

# Add tests for retry logic, error handling, etc.
