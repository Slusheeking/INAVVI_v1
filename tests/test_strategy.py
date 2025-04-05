"""Tests for ai_day_trader.strategy"""
import pytest
import asyncio
import time
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call
from ai_day_trader.strategy import AIStrategyRunner
from ai_day_trader.config import load_ai_trader_config
# Import specific types for mocking if needed
from ai_day_trader.signal_generator import SignalGenerator
from ai_day_trader.risk_manager import RiskManager
from ai_day_trader.feature_calculator import FeatureCalculator
from ai_day_trader.ml.predictor import Predictor as MLEnginePredictor
from collections import deque


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

    # Configure risk manager for ATR stop loss
    deps["risk_manager"].stop_loss_type = "atr"
    deps["risk_manager"].atr_feature_name = "atr_14"
    deps["risk_manager"].atr_stop_multiplier = 2.0

    # Set default return values for commonly called methods
    deps["signal_generator"].generate_signals = AsyncMock(return_value=[])
    deps["risk_manager"].get_remaining_daily_limit = AsyncMock(
        return_value=5000.0)
    deps["risk_manager"].calculate_position_size = AsyncMock(
        return_value=(10, 1000.0, 95.0))  # qty, value, stop_price
    deps["execution_system"].get_positions = AsyncMock(return_value={})
    deps["stock_selector"].select_candidate_symbols = AsyncMock(return_value=[
                                                                "AAPL", "MSFT"])

    # Mock methods needed for NBBO/Price fetching
    current_time_ms = time.time() * 1000
    deps["redis_client"].hgetall = AsyncMock(return_value={  # Mock Redis NBBO response
        b'bid_price': b'99.9',
        b'ask_price': b'100.1',
        b'timestamp': str(current_time_ms).encode(),
        b'bid_size': b'100',
        b'ask_size': b'100'
    })
    deps["redis_client"].get = AsyncMock(
        return_value=None)  # Mock Redis tick miss

    return deps


@pytest.fixture
async def runner_instance(mock_dependencies):
    """Fixture to create an initialized AIStrategyRunner instance."""
    runner = AIStrategyRunner(**mock_dependencies)

    # Set up price cache for peak detection
    runner._recent_prices_cache = {
        "AAPL": deque([98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 102.5, 101.5, 100.5], maxlen=60),
        "MSFT": deque([245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 249.5, 248.5, 247.5], maxlen=60)
    }

    # Mock internal methods that might be complex or make external calls
    runner._get_latest_price = AsyncMock(
        return_value=100.0)  # Default mock price

    current_time_ms = time.time() * 1000
    runner._get_latest_nbbo = AsyncMock(return_value={
        'bid_price': 99.9,
        'ask_price': 100.1,
        'bid_size': 100.0,
        'ask_size': 100.0,
        'timestamp': current_time_ms
    })

    runner._evaluate_and_execute_entry = AsyncMock()  # Mock entry logic
    runner._check_exit_conditions = AsyncMock()  # Mock exit logic

    # Mock feature calculation with more comprehensive features including ATR
    runner.get_latest_features = AsyncMock(return_value={
        'feat1': 1.0,
        'atr_14': 2.5,
        'rsi_14': 65.0,
        'macd': 0.25,
        'macd_signal': 0.15,
        'macd_hist': 0.10,
        'ema_9': 99.5,
        'sma_20': 98.0,
        'volume_change': 1.2
    })

    return runner
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

    # Create a more detailed features map with ATR value
    features_map = {
        "AAPL": {
            "atr_14": 2.5,
            "rsi_14": 65.0,
            "macd": 0.25,
            "ema_9": 99.5,
            "sma_20": 98.0
        },
        "MSFT": {
            "atr_14": 3.2,
            "rsi_14": 55.0,
            "macd": -0.10,
            "ema_9": 248.5,
            "sma_20": 245.0
        }
    }

    # Mock signal generator to return signals for both, but MSFT should be skipped
    mock_signals = [
        {"symbol": "AAPL", "side": "buy", "source": "ML"},
        {"symbol": "MSFT", "side": "buy", "source": "Rule"},
    ]
    runner.signal_generator.generate_signals = AsyncMock(
        return_value=mock_signals)

    await runner._process_entries(symbols, positions, portfolio_value, remaining_limit, features_map)

    # Verify signal_generator was called correctly with the enhanced features_map
    runner.signal_generator.generate_signals.assert_awaited_once_with(
        symbols, features_map)

    # Verify _evaluate_and_execute_entry was called ONLY for AAPL (not MSFT because position exists)
    runner._evaluate_and_execute_entry.assert_awaited_once_with(
        "AAPL", "buy", portfolio_value, remaining_limit, "ML", features_map["AAPL"]
    )


@pytest.mark.asyncio
async def test_monitor_exits_with_features(runner_instance: AIStrategyRunner):
    """Test the _monitor_exits method with feature map."""
    runner = runner_instance

    # Mock positions
    positions = {
        "AAPL": {"symbol": "AAPL", "qty": "10", "avg_entry_price": "100.0"},
        # Short position
        "MSFT": {"symbol": "MSFT", "qty": "-5", "avg_entry_price": "250.0"}
    }

    # Create feature map with ATR values
    features_map = {
        "AAPL": {
            "atr_14": 2.5,
            "rsi_14": 65.0,
            "macd": 0.25,
            "ema_9": 99.5,
            "sma_20": 98.0
        },
        "MSFT": {
            "atr_14": 3.2,
            "rsi_14": 55.0,
            "macd": -0.10,
            "ema_9": 248.5,
            "sma_20": 245.0
        }
    }

    # Reset and mock _check_exit_conditions
    runner._check_exit_conditions = AsyncMock(return_value=None)

    await runner._monitor_exits(positions, features_map)

    # Verify _check_exit_conditions was called for each position with correct features
    assert runner._check_exit_conditions.await_count == 2

    # Check the calls with assert_any_await to ignore order
    runner._check_exit_conditions.assert_any_await(
        "AAPL", positions["AAPL"], features_map["AAPL"]
    )
    runner._check_exit_conditions.assert_any_await(
        "MSFT", positions["MSFT"], features_map["MSFT"]
    )


@pytest.mark.asyncio
async def test_check_stop_loss_using_atr(runner_instance: AIStrategyRunner):
    """Test the _check_stop_loss method using ATR-based stop."""
    runner = runner_instance

    # Temporarily restore the original method for testing
    original_method = runner._check_stop_loss
    runner._check_stop_loss = original_method.__get__(runner)

    symbol = "AAPL"
    pos_info = {"symbol": "AAPL", "qty": "10", "avg_entry_price": "100.0"}
    pos_qty = 10.0
    current_price = 95.0  # Below the stop price

    # Feature map with ATR value
    features = {
        "atr_14": 2.5,  # ATR value
        "rsi_14": 30.0,  # Low RSI indicating oversold
        "macd": -0.5    # Negative MACD indicating downtrend
    }

    # Stop price should be: entry_price - (atr * multiplier) = 100 - (2.5 * 2.0) = 95.0
    # Since current_price = 95.0, it's at the stop level and should trigger

    result = await runner._check_stop_loss(symbol, pos_info, pos_qty, current_price, features)

    # Verify the stop loss was triggered using ATR
    assert result is not None
    assert "ATR" in result
    assert "95.0" in result  # The stop price should be mentioned

    # Reset method to avoid interference with other tests
    runner._check_stop_loss = AsyncMock(return_value=None)


@pytest.mark.asyncio
async def test_get_latest_nbbo(runner_instance: AIStrategyRunner):
    """Test the _get_latest_nbbo method with Redis data."""
    runner = runner_instance

    # Temporarily restore the original method for testing
    original_method = runner._get_latest_nbbo
    runner._get_latest_nbbo = original_method.__get__(runner)

    # Set up Redis mock to return NBBO data
    current_time_ms = time.time() * 1000
    runner.redis_client.hgetall = AsyncMock(return_value={
        b'bid_price': b'149.5',
        b'ask_price': b'150.0',
        b'bid_size': b'200',
        b'ask_size': b'150',
        b'timestamp': str(current_time_ms).encode()
    })

    symbol = "AAPL"
    result = await runner._get_latest_nbbo(symbol)

    # Verify NBBO data was correctly processed
    assert result is not None
    assert result['bid_price'] == 149.5
    assert result['ask_price'] == 150.0
    assert result['bid_size'] == 200.0
    assert result['ask_size'] == 150.0
    assert 'timestamp' in result

    # Verify Redis was called with the correct key
    runner.redis_client.hgetall.assert_awaited_once_with(f"nbbo:{symbol}")

    # Reset the method to avoid interference with other tests
    runner._get_latest_nbbo = AsyncMock()

# These are just initial implementations of some of the TODO tests
# Additional tests can be implemented for:
# - _check_peak_exit with the deque-based price cache
# - _check_ml_exit with features
# - EOD handling logic
# - _get_latest_price with NBBO vs tick fallback
# TODO: Add tests for EOD handling etc.
