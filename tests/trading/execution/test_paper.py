"""Tests for ai_day_trader.trading.execution.paper"""
import pytest
import asyncio
import time # Import time for timestamp checks
from unittest.mock import AsyncMock, patch, MagicMock
from ai_day_trader.trading.execution.paper import PaperExecution
# Import necessary classes for mocking/instantiation
from ai_day_trader.utils.config import Config
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient
# Import Redis client type for mocking
from redis.asyncio import Redis as AsyncRedis

# Mock response object for Polygon client (if needed for fallback)
class MockPolygonResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

# Change scope to function to align with default event loop
# Default mock NBBO data
DEFAULT_NBBO = {'bid_price': 99.9, 'ask_price': 100.1, 'timestamp': time.time() * 1000}
PNLTEST_NBBO = {'bid_price': 104.9, 'ask_price': 105.1, 'timestamp': time.time() * 1000}
LIMITTEST_NBBO = {'bid_price': 95.0, 'ask_price': 95.2, 'timestamp': time.time() * 1000}

@pytest.fixture(scope="function")
async def paper_exec(event_loop): # event_loop is implicitly available
    """Provides an initialized PaperExecution instance with mocked dependencies."""
    config = Config() # Use a basic Config instance for tests
    # Mock Polygon client (only needed if Redis fails in _get_current_nbbo)
    mock_polygon_client = AsyncMock(spec=PolygonRESTClient)
    # Mock Redis client
    mock_redis_client = AsyncMock(spec=AsyncRedis)

    # Mock the _get_current_nbbo method directly within PaperExecution instance later
    # This avoids needing complex Redis mocking here

    # Instantiate PaperExecution with Config, mocked Polygon, and mocked Redis
    instance = PaperExecution(config=config, polygon_client=mock_polygon_client, redis_client=mock_redis_client)

    # --- Mock the internal _get_current_nbbo method ---
    # We use patch.object to mock the method on the *instance* after it's created
    # Default mock implementation
    async def default_mock_nbbo(symbol):
        if symbol == "PNLTEST": return PNLTEST_NBBO
        if symbol == "LIMITTEST": return LIMITTEST_NBBO
        return DEFAULT_NBBO

    with patch.object(instance, '_get_current_nbbo', side_effect=default_mock_nbbo) as mock_nbbo_method:
        instance._get_current_nbbo = mock_nbbo_method # Ensure the instance uses the mock
        await instance.initialize()
        yield instance # Yield the instance with the mocked method
        await instance.close()

@pytest.mark.asyncio
async def test_paper_execute_market_order(paper_exec: PaperExecution):
    """Test executing a market order in paper mode."""
    order = {"symbol": "TEST", "side": "buy", "quantity": 10, "order_type": "market"}
    details = await paper_exec.execute_order(order)
    assert details.get("status") == "filled"
    assert details.get("symbol") == "TEST"
    assert details.get("fill_quantity") == 10
    # Market buy fills at ask (100.1) + slippage
    expected_slippage = max(100.1 * paper_exec.slippage_pct, 0.01)
    expected_fill_price = 100.1 + expected_slippage
    assert details.get("fill_price") == pytest.approx(expected_fill_price)
    # Check commission
    expected_commission = max(paper_exec.commission_min_per_order, 10 * paper_exec.commission_per_share)
    assert details.get("commission") == pytest.approx(expected_commission)

    positions = await paper_exec.get_positions()
    assert "TEST" in positions
    assert positions["TEST"].get("quantity") == 10 # Use .get()

@pytest.mark.asyncio
async def test_paper_execute_limit_order(paper_exec: PaperExecution):
    """Test executing a limit order in paper mode."""
    # Sell limit below bid (99.9) should fill
    order = {"symbol": "TEST", "side": "sell", "quantity": 5, "order_type": "limit", "limit_price": 99.8}
    details = await paper_exec.execute_order(order)
    assert details.get("status") == "filled"
    assert details.get("symbol") == "TEST"
    assert details.get("fill_quantity") == 5
    assert details.get("fill_price") == 99.8 # Fills at limit price
    # Check commission
    expected_commission = max(paper_exec.commission_min_per_order, 5 * paper_exec.commission_per_share)
    assert details.get("commission") == pytest.approx(expected_commission)


@pytest.mark.asyncio
async def test_paper_execute_limit_order_not_filled(paper_exec: PaperExecution):
    """Test executing a limit order that shouldn't fill immediately."""
    # Buy limit below ask (100.1) should not fill
    order_buy = {"symbol": "TESTBUY", "side": "buy", "quantity": 5, "order_type": "limit", "limit_price": 100.0}
    details_buy = await paper_exec.execute_order(order_buy)
    assert details_buy.get("status") == "accepted"
    assert details_buy.get("fill_quantity") == 0
    assert details_buy.get("fill_price") is None
    assert details_buy.get("commission") == 0.0 # No commission if not filled

    # Sell limit above bid (99.9) should not fill
    order_sell = {"symbol": "TESTSELL", "side": "sell", "quantity": 5, "order_type": "limit", "limit_price": 100.0}
    details_sell = await paper_exec.execute_order(order_sell)
    assert details_sell.get("status") == "accepted"
    assert details_sell.get("fill_quantity") == 0
    assert details_sell.get("fill_price") is None
    assert details_sell.get("commission") == 0.0 # No commission if not filled


@pytest.mark.asyncio
async def test_paper_get_positions_pnl(paper_exec: PaperExecution):
    """Test getting positions with simulated PnL using NBBO."""
    # Buy 10 shares, fills at ask (100.1) + slippage
    buy_order = {"symbol": "PNLTEST", "side": "buy", "quantity": 10, "order_type": "market"}
    entry_details = await paper_exec.execute_order(buy_order)
    entry_fill_price = entry_details.get("fill_price")
    entry_commission = entry_details.get("commission")
    assert entry_details.get("status") == "filled"

    # get_positions uses _get_current_nbbo which is mocked for PNLTEST
    # NBBO = {'bid_price': 104.9, 'ask_price': 105.1}
    positions = await paper_exec.get_positions()
    assert "PNLTEST" in positions
    pos_info = positions["PNLTEST"]

    # Position quantity and cost basis should reflect the fill
    assert float(pos_info.get("quantity")) == 10.0
    expected_cost_basis = (10 * entry_fill_price) + entry_commission
    assert float(pos_info.get("cost_basis")) == pytest.approx(expected_cost_basis)

    # PnL is marked to the bid for long positions
    mark_price = PNLTEST_NBBO['bid_price'] # 104.9
    assert float(pos_info.get("current_price")) == pytest.approx(mark_price)
    market_value = 10 * mark_price
    expected_pl = market_value - expected_cost_basis
    expected_plpc = (expected_pl / abs(expected_cost_basis)) * 100 if expected_cost_basis != 0 else 0.0

    assert float(pos_info.get("market_value")) == pytest.approx(market_value)
    assert float(pos_info.get("unrealized_pl")) == pytest.approx(expected_pl)
    assert float(pos_info.get("unrealized_plpc")) == pytest.approx(expected_plpc)

@pytest.mark.asyncio
async def test_paper_get_order_status_fills_limit(paper_exec: PaperExecution):
    """Test that get_order_status fills an open limit order if price moves."""
    # Place a limit buy order that initially doesn't fill (limit=99.8 < ask=100.1)
    limit_price = 99.8
    order = {"symbol": "LIMITTEST", "side": "buy", "quantity": 5, "order_type": "limit", "limit_price": limit_price}
    details_initial = await paper_exec.execute_order(order)
    order_id = details_initial.get("order_id")
    assert details_initial.get("status") == "accepted"

    # Update the mock for _get_current_nbbo to simulate price movement
    async def mock_nbbo_limit_fill(symbol):
         if symbol == "LIMITTEST":
              # Ask price (95.2) is now <= limit_price (99.8)
              return {'bid_price': 95.0, 'ask_price': 95.2, 'timestamp': time.time() * 1000}
         return DEFAULT_NBBO # Default for other symbols if any

    # Re-patch the instance's method
    with patch.object(paper_exec, '_get_current_nbbo', side_effect=mock_nbbo_limit_fill) as mock_nbbo_method:
        paper_exec._get_current_nbbo = mock_nbbo_method
        # Call get_order_status, which should trigger the fill check using the new mock
        details_final = await paper_exec.get_order_status(order_id)

    assert details_final is not None
    assert details_final.get("status") == "filled"
    assert details_final.get("fill_quantity") == 5
    assert details_final.get("fill_price") == limit_price # Fills at limit price
    # Check commission was calculated on fill
    expected_commission = max(paper_exec.commission_min_per_order, 5 * paper_exec.commission_per_share)
    assert details_final.get("commission") == pytest.approx(expected_commission)

    # Check position
    positions = await paper_exec.get_positions()
    assert "LIMITTEST" in positions
    assert positions["LIMITTEST"].get("quantity") == 5

# TODO: Add tests for cancellation
