"""Tests for ai_day_trader.clients.unusual_whales_client"""
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from ai_day_trader.clients.unusual_whales_client import UnusualWhalesClient
from ai_day_trader.config import load_ai_trader_config

# Requires Unusual Whales API key in .env

# Remove custom event_loop fixture, use default function-scoped one

# Ensure scope is function to align with default event loop
@pytest.fixture(scope="function")
async def uw_client():
    """Provides an initialized UnusualWhalesClient instance."""
    config = load_ai_trader_config()
    client = UnusualWhalesClient(config=config)
    await client.connection_pool.initialize() # Ensure pool is initialized
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_get_flow_alerts_no_ticker(uw_client: UnusualWhalesClient):
    """Test fetching general flow alerts."""
    if not uw_client.api_key: pytest.skip("UW API key not configured")
    alerts = await uw_client.get_flow_alerts(limit=5)
    assert isinstance(alerts, list)
    # Further assertions depend on API response

@pytest.mark.asyncio
async def test_get_unusual_options(uw_client: UnusualWhalesClient):
    """Test fetching unusual options activity."""
    if not uw_client.api_key: pytest.skip("UW API key not configured")
    options = await uw_client.get_unusual_options(limit=5)
    assert isinstance(options, list)
    # Further assertions depend on API response

# Add more tests for other methods, edge cases, etc.
