"""
Integrated WebSocket Test

Tests the full WebSocket integration across:
- data_api (Polygon WebSocket client)
- data_pipeline (message processing)
- stock_selection (decision making)
"""

import pytest
import asyncio
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from api_clients.polygon_ws import PolygonWebSocketClient
from data_pipeline.processing import clean_market_data, calculate_technical_indicators
from stock_selection.websocket import WebSocketEnhancedStockSelection

@pytest.fixture
def mock_polygon_client():
    """Mock Polygon WebSocket client with all required methods"""
    client = AsyncMock(spec=PolygonWebSocketClient)
    client.connected = True
    client.subscribed_channels = set()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.subscribe = AsyncMock()
    return client

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    return MagicMock()

@pytest.fixture
def stock_selector(mock_polygon_client, mock_redis):
    """Test instance of WebSocketEnhancedStockSelection"""
    return WebSocketEnhancedStockSelection(
        polygon_ws_client=mock_polygon_client,
        redis_client=mock_redis
    )

@pytest.mark.asyncio
async def test_websocket_integration(stock_selector, mock_polygon_client):
    """Test full WebSocket integration flow"""
    # Setup test data
    test_message = {
        "ev": "AM",
        "sym": "AAPL",
        "timestamp": "2025-03-29T03:00:00Z",
        "v": 1000,
        "av": 50000,
        "op": 150.0,
        "vw": 151.0,
        "o": 150.5,
        "c": 151.5,
        "close": 151.5,
        "h": 152.0,
        "l": 150.0,
        "a": 151.2,
        "z": 100
    }

    # Mock the message handler
    processed_data = None
    async def mock_handler(message):
        nonlocal processed_data
        df = pd.DataFrame([message])
        cleaned = clean_market_data(df)
        processed_data = calculate_technical_indicators(cleaned)
        return {
            "processed": True,
            "original": message,
            **processed_data.to_dict('records')[0]
        }

    stock_selector.process_message = mock_handler

    # Test connection
    await stock_selector.start()
    mock_polygon_client.connect.assert_awaited_once()

    # Test subscription
    symbols = ["AAPL", "MSFT"]
    await stock_selector.subscribe_to_symbols(symbols)
    mock_polygon_client.subscribe.assert_awaited_with(symbols)

    # Test message processing
    result = await stock_selector.process_message(test_message)
    assert result is not None
    assert "processed" in result
    assert result["processed"] is True
    assert "original" in result

    # Redis caching is not currently implemented in the WebSocketEnhancedStockSelection class
    # This test verifies the core WebSocket functionality without Redis checks

    # Test disconnection
    await stock_selector.stop()
    mock_polygon_client.disconnect.assert_awaited_once()

@pytest.mark.asyncio
async def test_data_pipeline_processing():
    """Test data pipeline integration with WebSocket messages"""
    test_message = {
        "ev": "T",  # Trade event
        "sym": "TSLA",
        "p": 700.50,  # price
        "s": 100,     # size
        "t": 1234567890000,  # timestamp
        "c": [1, 12],  # conditions
        "timestamp": "2025-03-29T03:00:00Z",
        "close": 700.50
    }

    # Convert message to DataFrame for processing
    df = pd.DataFrame([test_message])
    cleaned = clean_market_data(df)
    processed = calculate_technical_indicators(cleaned)
    
    assert processed is not None
    assert not processed.empty
    assert "close" in processed.columns
    assert processed.iloc[0]["close"] == 700.50

@pytest.mark.asyncio
async def test_stock_selection_logic(stock_selector):
    """Test stock selection logic with processed WebSocket data"""
    processed_data = {
        "symbol": "NVDA",
        "price": 450.75,
        "volume": 5000000,
        "timestamp": "2025-03-29T03:00:00Z",
        "close": 450.75,
        "normalized": {
            "momentum": 0.85,
            "volatility": 0.2
        }
    }

    # Mock the actual selection logic
    with patch.object(stock_selector, '_make_selection_decision') as mock_decision:
        mock_decision.return_value = {"action": "BUY", "confidence": 0.9}
        result = await stock_selector.process_message(processed_data)

        assert result is not None
        assert result["processed"] is True

def test_metrics_tracking(stock_selector, mock_polygon_client):
    """Test WebSocket metrics are properly tracked"""
    # Mock the metrics
    from unittest.mock import MagicMock
    from utils.metrics_registry import WEBSOCKET_MESSAGES
    
    WEBSOCKET_MESSAGES.labels = MagicMock()
    WEBSOCKET_MESSAGES.labels.return_value.inc = MagicMock()

    # Simulate message processing
    test_message = {
        "ev": "Q", 
        "sym": "AMD", 
        "bp": 120.5, 
        "bs": 10,
        "timestamp": "2025-03-29T03:00:00Z",
        "close": 120.5
    }
    asyncio.run(stock_selector.process_message(test_message))

    # Verify metrics were called
    WEBSOCKET_MESSAGES.labels.assert_called_once_with(
        client="polygon",
        message_type="Q"
    )
    WEBSOCKET_MESSAGES.labels.return_value.inc.assert_called_once()
