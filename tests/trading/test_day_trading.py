"""Tests for ai_day_trader.trading.day_trading"""
import pytest
import logging
from unittest.mock import AsyncMock
import pandas as pd
from ai_day_trader.trading.day_trading import DayTradingStrategies
from ai_day_trader.config import load_ai_trader_config

# TODO: Mock clients (PolygonREST, Redis) and provide mock data

@pytest.mark.asyncio
async def test_day_trading_init():
    """Test DayTradingStrategies initialization."""
    config = load_ai_trader_config()
    mock_rest = AsyncMock()
    mock_redis = AsyncMock()
    logger = logging.getLogger("test_dt_logger")
    # Pass None for execution system as it's not used in __init__ anymore
    strategies = DayTradingStrategies(config, mock_rest, mock_redis, logger)
    assert strategies is not None

# TODO: Add tests for ORB, VWAP strategies with mock data/responses
