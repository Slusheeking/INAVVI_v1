"""Tests for ai_day_trader.trading.execution.live"""
import pytest
from unittest.mock import patch, AsyncMock
from ai_day_trader.trading.execution.live import LiveExecution
from ai_day_trader.config import load_ai_trader_config

# Requires mocking alpaca_trade_api

from ai_day_trader.utils.config import Config # Import Config

from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient # Import for mocking

@pytest.mark.asyncio
@patch('ai_day_trader.trading.execution.live.tradeapi', new=None) # Mock alpaca library import
@patch('ai_day_trader.trading.execution.live.ALPACA_AVAILABLE', new=False) # Mock availability flag
async def test_live_execution_init_no_library(mocker): # Add mocker fixture
    """Test initialization when alpaca library is not available."""
    config = Config()
    mock_polygon_client = mocker.AsyncMock(spec=PolygonRESTClient) # Mock Polygon client
    live_exec = LiveExecution(config=config, polygon_client=mock_polygon_client)
    await live_exec.initialize()
    assert live_exec.api is None
    assert live_exec.polygon_client is mock_polygon_client # Check polygon client assignment
    assert live_exec.expected_prices == {} # Check expected_prices init

@pytest.mark.asyncio
@patch('ai_day_trader.trading.execution.live.tradeapi') # Mock the library itself
async def test_live_execution_init_no_creds(mock_tradeapi, mocker): # Add mocker
    """Test initialization without credentials."""
    config = Config()
    mock_polygon_client = mocker.AsyncMock(spec=PolygonRESTClient)
    with patch.object(config, 'get_str', side_effect=lambda key, default=None: None if key in ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"] else default):
        live_exec = LiveExecution(config=config, polygon_client=mock_polygon_client)
        await live_exec.initialize() # Should log errors but not raise, api is None
        assert live_exec.api is None
        assert live_exec.api_key is None
        assert live_exec.secret_key is None
        assert live_exec.polygon_client is mock_polygon_client
        assert live_exec.expected_prices == {}

@pytest.mark.asyncio
@patch('ai_day_trader.trading.execution.live.tradeapi')
async def test_live_execution_init_success(mock_tradeapi, mocker): # Add mocker
    """Test successful initialization."""
    mock_polygon_client = mocker.AsyncMock(spec=PolygonRESTClient)
    # Mock the Alpaca REST client and its methods
    mock_alpaca_rest_instance = AsyncMock()
    mock_alpaca_account = AsyncMock()
    mock_alpaca_account.status = 'ACTIVE'
    mock_alpaca_rest_instance.get_account.return_value = mock_alpaca_account
    mock_tradeapi.REST.return_value = mock_alpaca_rest_instance

    config = Config()
    with patch.object(config, 'get_str', side_effect=lambda key, default=None: {
        "APCA_API_KEY_ID": "dummy_key",
        "APCA_API_SECRET_KEY": "dummy_secret",
        "APCA_API_BASE_URL": "https://paper-api.alpaca.markets"
    }.get(key, default)):
        live_exec = LiveExecution(config=config, polygon_client=mock_polygon_client)
        await live_exec.initialize()
        assert live_exec.api is mock_alpaca_rest_instance # Check correct instance
        assert live_exec.polygon_client is mock_polygon_client
        assert live_exec.expected_prices == {}
        mock_tradeapi.REST.assert_called_once_with(
            key_id="dummy_key",
            secret_key="dummy_secret",
            base_url="https://paper-api.alpaca.markets",
            api_version='v2'
        )
        # Use run_in_executor mock if needed, or check if get_account was awaited
        # For simplicity, we check if the mock instance's method was called
        mock_alpaca_rest_instance.get_account.assert_awaited_once()


# TODO: Add tests for execute_order, cancel_order, get_positions with mocked Alpaca API methods

# --- Fixture for LiveExecution instance ---
@pytest.fixture
def live_exec_instance(mocker):
    """Provides a LiveExecution instance with mocked dependencies."""
    mock_polygon_client = mocker.AsyncMock(spec=PolygonRESTClient)
    mock_alpaca_rest_instance = AsyncMock()
    mock_alpaca_account = AsyncMock()
    mock_alpaca_account.status = 'ACTIVE'
    mock_alpaca_rest_instance.get_account.return_value = mock_alpaca_account

    config = Config()
    # Patch config get_str to provide dummy creds
    with patch.object(config, 'get_str', side_effect=lambda key, default=None: {
        "APCA_API_KEY_ID": "dummy_key",
        "APCA_API_SECRET_KEY": "dummy_secret",
        "APCA_API_BASE_URL": "https://paper-api.alpaca.markets"
    }.get(key, default)):
        # Patch the tradeapi.REST call within the fixture setup
        with patch('ai_day_trader.trading.execution.live.tradeapi.REST', return_value=mock_alpaca_rest_instance):
            instance = LiveExecution(config=config, polygon_client=mock_polygon_client)
            # Manually assign the mocked api instance as initialize won't be called here
            instance.api = mock_alpaca_rest_instance
            instance.polygon_client = mock_polygon_client # Ensure polygon client is assigned
            instance.logger = mocker.MagicMock() # Mock logger to check calls
            yield instance # Use yield to allow cleanup if needed

# --- Tests for Slippage Logic ---

@pytest.mark.asyncio
async def test_execute_order_stores_expected_price(live_exec_instance, mocker):
    """Test that execute_order stores the expected price for slippage calc."""
    # Arrange
    symbol = "AAPL"
    side = "buy"
    order_type = "market"
    client_order_id = "test_client_id_123"
    expected_ask = 150.50
    expected_bid = 150.40

    live_exec_instance.polygon_client.get_last_quote = AsyncMock(return_value={'ask': expected_ask, 'bid': expected_bid})

    # Mock Alpaca order submission
    mock_alpaca_order = AsyncMock()
    mock_alpaca_order.id = "server_order_id_456"
    mock_alpaca_order.client_order_id = client_order_id
    mock_alpaca_order.symbol = symbol
    mock_alpaca_order.qty = "10"
    mock_alpaca_order.side = side
    mock_alpaca_order.type = order_type
    mock_alpaca_order.time_in_force = "day"
    mock_alpaca_order.limit_price = None
    mock_alpaca_order.stop_price = None
    mock_alpaca_order.status = "accepted"
    mock_alpaca_order.filled_avg_price = None
    mock_alpaca_order.filled_qty = "0"
    mock_alpaca_order.submitted_at = AsyncMock() # Mock datetime attribute if needed
    mock_alpaca_order.submitted_at.timestamp.return_value = 1678886400.0 # Example timestamp

    # Patch run_in_executor used by submit_order
    mocker.patch('asyncio.get_event_loop').return_value.run_in_executor = AsyncMock(return_value=mock_alpaca_order)

    order_dict = {
        'symbol': symbol, 'quantity': 10, 'side': side,
        'order_type': order_type, 'client_order_id': client_order_id
    }

    # Act
    await live_exec_instance._execute_order_impl(order_dict)

    # Assert
    live_exec_instance.polygon_client.get_last_quote.assert_awaited_once_with(symbol)
    assert client_order_id in live_exec_instance.expected_prices
    assert live_exec_instance.expected_prices[client_order_id] == expected_ask # Buy uses ask

@pytest.mark.asyncio
async def test_get_order_status_calculates_slippage(live_exec_instance, mocker):
    """Test that get_order_status calculates and logs slippage for filled orders."""
    # Arrange
    symbol = "MSFT"
    side = "sell"
    order_type = "market"
    client_order_id = "test_client_id_456"
    server_order_id = "server_order_id_789"
    expected_bid = 280.20
    filled_price = 280.15 # Negative slippage for sell

    # Pre-populate expected price
    live_exec_instance.expected_prices[client_order_id] = expected_bid

    # Mock Alpaca get_order response
    mock_alpaca_order = AsyncMock()
    mock_alpaca_order.id = server_order_id
    mock_alpaca_order.client_order_id = client_order_id
    mock_alpaca_order.symbol = symbol
    mock_alpaca_order.qty = "5"
    mock_alpaca_order.side = side
    mock_alpaca_order.type = order_type
    mock_alpaca_order.time_in_force = "day"
    mock_alpaca_order.limit_price = None
    mock_alpaca_order.stop_price = None
    mock_alpaca_order.status = "filled" # Order is filled
    mock_alpaca_order.filled_avg_price = str(filled_price) # Alpaca returns string
    mock_alpaca_order.filled_qty = "5"
    mock_alpaca_order.submitted_at = AsyncMock()
    mock_alpaca_order.submitted_at.timestamp.return_value = 1678886400.0

    # Patch run_in_executor used by get_order
    mocker.patch('asyncio.get_event_loop').return_value.run_in_executor = AsyncMock(return_value=mock_alpaca_order)

    # Act
    await live_exec_instance._get_order_status_impl(server_order_id)

    # Assert
    # Check if logger.info was called with slippage details
    live_exec_instance.logger.info.assert_any_call(
        mocker.string_matching(f"Slippage Report for {client_order_id}.*Expected={expected_bid:.4f}.*Filled={filled_price:.4f}.*Slippage=")
    )
    # Check that the expected price was removed after calculation
    assert client_order_id not in live_exec_instance.expected_prices

@pytest.mark.asyncio
async def test_get_order_status_cleans_up_expected_price_on_terminal_status(live_exec_instance, mocker):
    """Test that expected_price is cleaned up for terminal non-filled orders."""
    # Arrange
    client_order_id = "test_client_id_789"
    server_order_id = "server_order_id_101"
    expected_ask = 100.0
    live_exec_instance.expected_prices[client_order_id] = expected_ask

    mock_alpaca_order = AsyncMock()
    mock_alpaca_order.client_order_id = client_order_id
    mock_alpaca_order.status = "canceled" # Terminal state other than filled
    # ... other attributes don't matter much here

    mocker.patch('asyncio.get_event_loop').return_value.run_in_executor = AsyncMock(return_value=mock_alpaca_order)

    # Act
    await live_exec_instance._get_order_status_impl(server_order_id)

    # Assert
    assert client_order_id not in live_exec_instance.expected_prices
