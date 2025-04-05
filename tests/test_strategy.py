"""Tests for ai_day_trader.strategy"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from ai_day_trader.strategy import AIStrategyRunner
from ai_day_trader.config import load_ai_trader_config
# Import specific types for mocking if needed
from ai_day_trader.signal_generator import SignalGenerator
from ai_day_trader.risk_manager import RiskManager
from ai_day_trader.feature_calculator import FeatureCalculator
from ai_day_trader.ml.predictor import Predictor as MLEnginePredictor


@pytest.fixture
def mock_dependencies():
    """Provides mocked dependencies for AIStrategyRunner."""
    config = load_ai_trader_config()
    # Mock all dependencies passed to __init__
    # Use spec=True for stricter mocking where possible
    deps = {
        "config": config,
        "redis_client": AsyncMock(spec=True),
        "polygon_ws_client": AsyncMock(spec=True, is_connected=lambda: True),
        "polygon_rest_client": AsyncMock(spec=True),
        "trade_manager": AsyncMock(spec=True),
        "signal_generator": AsyncMock(spec=SignalGenerator),
        "risk_manager": AsyncMock(spec=RiskManager),
        "execution_system": AsyncMock(spec=True),
        "stock_selector": AsyncMock(spec=True),
        "ml_predictor": AsyncMock(spec=MLEnginePredictor),
        "feature_calculator": AsyncMock(spec=FeatureCalculator),
        "alpaca_client": AsyncMock(spec=True)
    }
    # Set default return values for commonly called methods
    deps["signal_generator"].generate_signals = AsyncMock(return_value=[])
    deps["risk_manager"].get_remaining_daily_limit = AsyncMock(
        return_value=5000.0)
    deps["execution_system"].get_positions = AsyncMock(return_value={})
    deps["stock_selector"].select_candidate_symbols = AsyncMock(return_value=[
                                                                "AAPL", "MSFT"])
    # Mock methods needed for NBBO/Price fetching
    deps["redis_client"].hgetall = AsyncMock(return_value={  # Mock Redis NBBO response
        b'bid_price': b'99.9', b'ask_price': b'100.1', b'timestamp': str(time.time() * 1000).encode()
    })
    deps["redis_client"].get = AsyncMock(
        return_value=None)  # Mock Redis tick miss

    return deps


@pytest.fixture
async def runner_instance(mock_dependencies):
    """Fixture to create an initialized AIStrategyRunner instance."""
    runner = AIStrategyRunner(**mock_dependencies)
    # Mock internal methods that might be complex or make external calls
    runner._get_latest_price = AsyncMock(
        return_value=100.0)  # Default mock price
    runner._get_latest_nbbo = AsyncMock(return_value={
                                        'bid_price': 99.9, 'ask_price': 100.1, 'timestamp': time.time() * 1000})
    runner._evaluate_and_execute_entry = AsyncMock()  # Mock entry logic
    runner._check_exit_conditions = AsyncMock()  # Mock exit logic
    runner.get_latest_features = AsyncMock(
        return_value={'feat1': 1.0})  # Mock feature calculation result
    return runner


@pytest.mark.asyncio
async def test_strategy_runner_init(mock_dependencies):
    """Test AIStrategyRunner initialization."""
    # Use the fixture that creates the instance
    runner = AIStrategyRunner(**mock_dependencies)
    assert runner is not None
    assert runner._running is False
    assert runner.redis_client is not None  # Check dependencies are set


@pytest.mark.asyncio
# Use runner_instance fixture
async def test_strategy_runner_start_stop(runner_instance: AIStrategyRunner):
    """Test starting and stopping the runner loop."""
    runner = runner_instance  # Use the already initialized runner
    await runner.start()
    assert runner._running is True
    assert runner._main_loop_task is not None
    # Give the loop a chance to run at least one iteration
    await asyncio.sleep(runner.loop_interval * 1.1 if runner.loop_interval else 0.1)
    await runner.stop()
    assert runner._running is False
    # Check if task is done or cancelled
    assert runner._main_loop_task is None or runner._main_loop_task.done()


@pytest.mark.asyncio
async def test_process_entries(runner_instance: AIStrategyRunner):
    """Test the _process_entries method."""
    runner = runner_instance
    symbols = ["AAPL", "MSFT"]
    # Mock positions: Already holding MSFT
    positions = {"MSFT": {"symbol": "MSFT", "qty": "10"}}
    portfolio_value = 1000.0
    remaining_limit = 5000.0
    # Mock features map
    features_map = {"AAPL": {"f1": 1}, "MSFT": {"f1": 2}}
    # Mock signal generator to return signals for both, but MSFT should be skipped
    mock_signals = [
        {"symbol": "AAPL", "side": "buy", "source": "ML"},
        {"symbol": "MSFT", "side": "buy", "source": "Rule"},
    ]
    runner.signal_generator.generate_signals = AsyncMock(
        return_value=mock_signals)

    await runner._process_entries(symbols, positions, portfolio_value, remaining_limit, features_map)

    # Verify signal_generator was called correctly
    runner.signal_generator.generate_signals.assert_awaited_once_with(
        symbols, features_map)

    # Verify _evaluate_and_execute_entry was called ONLY for AAPL (not MSFT because position exists)
    runner._evaluate_and_execute_entry.assert_awaited_once_with(
        "AAPL", "buy", portfolio_value, remaining_limit, "ML", features_map["AAPL"]
    )


# TODO: Add tests for _evaluate_and_execute_entry logic (NBBO vs price, risk calls)
# TODO: Add tests for _monitor_exits logic (calling _check_exit_conditions with features)
# TODO: Add tests for _check_exit_conditions (NBBO price usage, calling sub-checks)
# TODO: Add tests for _check_stop_loss (NBBO price usage)
# TODO: Add tests for _get_latest_price (NBBO vs tick fallback)
# TODO: Add tests for _get_latest_nbbo (Redis hit/miss/stale)
# TODO: Add tests for EOD handling etc.
