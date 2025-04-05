"""Tests for ai_day_trader.clients.polygon_ws_client"""
import pytest
import asyncio
from ai_day_trader.clients.polygon_ws_client import PolygonWebSocketClient
from ai_day_trader.config import load_ai_trader_config

# Requires Polygon API key in .env

# Remove custom event_loop fixture, use default function-scoped one

# Change scope to function to align with default event loop
@pytest.fixture(scope="function")
async def polygon_ws_client():
    """Provides an initialized PolygonWebSocketClient instance."""
    config = load_ai_trader_config()
    received_messages = []
    async def test_handler(msg):
        print(f"Handler received: {msg.get('ev', 'unknown')}")
        received_messages.append(msg)

    client = PolygonWebSocketClient(config=config, message_handler=test_handler)
    await client.connect()
    client.received_messages = received_messages # Attach for test access
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_ws_connection_and_auth(polygon_ws_client: PolygonWebSocketClient):
    """Test if the client connects and authenticates successfully."""
    assert polygon_ws_client.is_connected() is True

@pytest.mark.asyncio
async def test_ws_subscribe_unsubscribe(polygon_ws_client: PolygonWebSocketClient):
    """Test subscribing and unsubscribing."""
    symbol = "AAPL"
    channel = f"T.{symbol}"
    await polygon_ws_client.subscribe([channel])
    assert channel in polygon_ws_client.subscribed_channels
    await asyncio.sleep(1) # Allow potential message receipt
    await polygon_ws_client.unsubscribe([channel])
    assert channel not in polygon_ws_client.subscribed_channels

# Add tests for message handling, reconnection logic, etc.
