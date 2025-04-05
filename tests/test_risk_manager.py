"""Tests for ai_day_trader.risk_manager"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from ai_day_trader.risk_manager import RiskManager
from ai_day_trader.config import load_ai_trader_config
from ai_day_trader.utils.config import Config # Import Config from utils
from ai_day_trader.feature_calculator import FeatureCalculator # Import FeatureCalculator

# Mock FeatureCalculator for tests
@pytest.fixture
def mock_feature_calculator():
    mock_calc = MagicMock(spec=FeatureCalculator)
    # Set default attribute expected in RiskManager init
    mock_calc.atr_period = 14
    return mock_calc

@pytest.mark.asyncio
async def test_risk_manager_init(mock_feature_calculator): # Add fixture
    """Test RiskManager initialization."""
    config = load_ai_trader_config()
    mock_redis = AsyncMock()
    # Pass the mocked feature calculator
    rm = RiskManager(config=config, feature_calculator=mock_feature_calculator, redis_client=mock_redis)
    assert rm is not None
    assert rm.portfolio_size > 0

@pytest.mark.asyncio
async def test_position_sizing_percentage(mock_feature_calculator): # Add fixture
    """Test position size calculation using percentage stop loss."""
    # Ensure config uses percentage stop
    config_dict = load_ai_trader_config().as_dict()
    config_dict['STOP_LOSS_TYPE'] = 'percentage'
    config_dict['STOP_LOSS_PCT'] = 0.02
    config = Config(**config_dict)

    mock_redis = AsyncMock()
    # Mock redis get to return 0 daily limit used initially
    mock_redis.get = AsyncMock(return_value=None)
    rm = RiskManager(config=config, feature_calculator=mock_feature_calculator, redis_client=mock_redis)

    # Assume portfolio size 100k, max daily risk 5k (from .env), max trade risk 0.5% (500)
    # Stop loss 2%
    portfolio_value = 100000.0
    remaining_daily = 5000.0 # Assume full daily limit available
    entry_price = 150.0
    stop_loss_pct = rm.stop_loss_pct # Use value from rm instance
    risk_per_share = entry_price * stop_loss_pct # 150 * 0.02 = 3.0
    max_risk_from_trade_pct = portfolio_value * rm.max_trade_risk_pct # 100k * 0.005 = 500
    max_risk_per_trade = min(remaining_daily, max_risk_from_trade_pct) # min(5000, 500) = 500
    expected_qty = int(max_risk_per_trade / risk_per_share) # 500 / 3 = 166
    expected_stop_price = entry_price * (1 - stop_loss_pct) # 150 * 0.98 = 147.0

    # Pass None for features as percentage stop doesn't need them
    qty, value, stop_price_out = await rm.calculate_position_size(
        "AAPL", "buy", entry_price, portfolio_value, remaining_daily, latest_features=None
    )

    assert qty == expected_qty
    assert value == pytest.approx(expected_qty * entry_price)
    assert stop_price_out == pytest.approx(expected_stop_price)

@pytest.mark.asyncio
async def test_position_sizing_atr(mock_feature_calculator): # Add fixture
    """Test position size calculation using ATR stop loss."""
    # Ensure config uses ATR stop
    config_dict = load_ai_trader_config().as_dict()
    config_dict['STOP_LOSS_TYPE'] = 'atr'
    config_dict['ATR_STOP_MULTIPLIER'] = 2.0
    config = Config(**config_dict)

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None) # No daily limit used
    rm = RiskManager(config=config, feature_calculator=mock_feature_calculator, redis_client=mock_redis)

    portfolio_value = 100000.0
    remaining_daily = 5000.0
    entry_price = 150.0
    atr_value = 2.5 # Mock ATR value
    atr_multiplier = rm.atr_stop_multiplier # 2.0
    risk_per_share = atr_value * atr_multiplier # 2.5 * 2.0 = 5.0
    max_risk_from_trade_pct = portfolio_value * rm.max_trade_risk_pct # 500
    max_risk_per_trade = min(remaining_daily, max_risk_from_trade_pct) # 500
    expected_qty = int(max_risk_per_trade / risk_per_share) # 500 / 5.0 = 100
    expected_stop_price = entry_price - risk_per_share # 150.0 - 5.0 = 145.0

    # Mock features dictionary containing the ATR value
    mock_features = {f"atr_{rm.feature_calculator.atr_period}": atr_value}

    qty, value, stop_price_out = await rm.calculate_position_size(
        "NVDA", "buy", entry_price, portfolio_value, remaining_daily, latest_features=mock_features
    )

    assert qty == expected_qty
    assert value == pytest.approx(expected_qty * entry_price)
    assert stop_price_out == pytest.approx(expected_stop_price)


@pytest.mark.asyncio
async def test_position_sizing_hits_daily_limit(mock_feature_calculator): # Add fixture
    """Test position sizing when constrained by remaining daily limit (using percentage stop)."""
    config_dict = load_ai_trader_config().as_dict()
    config_dict['STOP_LOSS_TYPE'] = 'percentage'
    config_dict['STOP_LOSS_PCT'] = 0.02
    config = Config(**config_dict)

    mock_redis = AsyncMock()
    # Mock redis get to return 4800 daily limit used
    mock_redis.get = AsyncMock(return_value=b"4800.0")
    rm = RiskManager(config=config, feature_calculator=mock_feature_calculator, redis_client=mock_redis)

    portfolio_value = 100000.0
    remaining_daily = await rm.get_remaining_daily_limit() # Should be 5000 - 4800 = 200
    entry_price = 150.0
    stop_loss_pct = rm.stop_loss_pct # 0.02
    risk_per_share = entry_price * stop_loss_pct # 3.0
    max_risk_from_trade_pct = portfolio_value * rm.max_trade_risk_pct # 500
    max_risk_per_trade = min(remaining_daily, max_risk_from_trade_pct) # min(200, 500) = 200
    expected_qty = int(max_risk_per_trade / risk_per_share) # 200 / 3 = 66
    expected_stop_price = entry_price * (1 - stop_loss_pct) # 147.0

    qty, value, stop_price_out = await rm.calculate_position_size(
        "MSFT", "buy", entry_price, portfolio_value, remaining_daily, latest_features=None
    )

    assert qty == expected_qty
    assert value == pytest.approx(expected_qty * entry_price)
    assert stop_price_out == pytest.approx(expected_stop_price)

# TODO: Add tests for daily limit updates (incrbyfloat), check_entry_risk (if kept)
