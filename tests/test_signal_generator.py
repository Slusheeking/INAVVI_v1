"""Tests for ai_day_trader.signal_generator"""
import pytest
from unittest.mock import AsyncMock
from ai_day_trader.signal_generator import SignalGenerator
from ai_day_trader.config import load_ai_trader_config

# TODO: Mock dependencies (DayTradingStrategies, MLEnginePredictor, FeatureCalculator, Redis)


@pytest.mark.asyncio
async def test_signal_generator_init():
    """Test SignalGenerator initialization."""
    config = load_ai_trader_config()
    mock_strategies = AsyncMock()
    mock_redis = AsyncMock()
    mock_predictor = AsyncMock()
    mock_calculator = AsyncMock()
    sg = SignalGenerator(config, mock_strategies, mock_redis,
                         mock_predictor, mock_calculator)
    assert sg is not None


@pytest.mark.asyncio
async def test_generate_signals_no_sources():
    """Test signal generation when no sources are available."""
    config = load_ai_trader_config()
    sg = SignalGenerator(config, None, None, None, None)  # No sources
    # Pass an empty features map
    signals = await sg.generate_signals(["AAPL", "MSFT"], {})
    assert signals == []


@pytest.mark.asyncio
async def test_generate_signals_with_mocks():
    """Test signal generation with mocked sources."""
    config = load_ai_trader_config()
    mock_strategies = AsyncMock()
    # Mock individual strategy methods called by generate_signals
    # ORB returns buy for AAPL, None for MSFT
    mock_strategies.run_opening_range_breakout = AsyncMock(side_effect=lambda symbol: {
                                                           "symbol": symbol, "side": "buy", "source": "ORB"} if symbol == "AAPL" else None)
    # VWAP returns sell for MSFT, None for AAPL
    mock_strategies.run_vwap_reversion = AsyncMock(side_effect=lambda symbol: {
                                                   "symbol": symbol, "side": "sell", "source": "VWAP"} if symbol == "MSFT" else None)
    # Add mocks for other strategies if they exist and are called

    mock_predictor = AsyncMock()
    # Mock ML: predict sell for AAPL (conf=0.7), buy for MSFT (conf=0.8), buy for GOOG (conf=0.5 - below threshold)

    async def mock_predict_side_effect(symbol, features):
        assert isinstance(features, dict)  # Ensure features are passed
        if symbol == "AAPL":
            return {"prediction": -1, "probability": 0.7}
        if symbol == "MSFT":
            return {"prediction": 1, "probability": 0.8}
        if symbol == "GOOG":
            return {"prediction": 1, "probability": 0.5}
        return None
    mock_predictor.predict_entry = AsyncMock(
        side_effect=mock_predict_side_effect)
    mock_calculator = AsyncMock()  # Needed for init
    mock_redis = AsyncMock()

    sg = SignalGenerator(config, mock_strategies, mock_redis,
                         mock_predictor, mock_calculator)

    # Mock features map (needs entries for symbols passed to ML)
    mock_features_map = {
        "AAPL": {"feat1": 1},
        "MSFT": {"feat1": 1},
        "GOOG": {"feat1": 1}  # Also need features for GOOG
    }

    signals = await sg.generate_signals(["AAPL", "MSFT", "GOOG"], mock_features_map)

    # Expected signals after prioritization:
    # 1. ML Sell AAPL (conf 0.7 >= 0.6, overrides ORB Buy)
    # 2. ML Buy MSFT (conf 0.8 >= 0.6, overrides VWAP Sell)
    # 3. GOOG: ML Buy conf 0.5 < 0.6, no rule signal -> No signal
    assert len(signals) == 2

    # Check signal details
    aapl_signal = next((s for s in signals if s['symbol'] == 'AAPL'), None)
    msft_signal = next((s for s in signals if s['symbol'] == 'MSFT'), None)
    goog_signal = next((s for s in signals if s['symbol'] == 'GOOG'), None)

    assert aapl_signal is not None
    assert aapl_signal['source'] == 'MLModel'
    assert aapl_signal['side'] == 'sell'
    assert aapl_signal['confidence'] == 0.7

    assert msft_signal is not None
    assert msft_signal['source'] == 'MLModel'
    assert msft_signal['side'] == 'buy'
    assert msft_signal['confidence'] == 0.8

    assert goog_signal is None  # Low confidence ML signal was discarded

    # Verify predict_entry was called with features
    assert mock_predictor.predict_entry.call_count == 3
    mock_predictor.predict_entry.assert_any_call(
        "AAPL", mock_features_map["AAPL"])
    mock_predictor.predict_entry.assert_any_call(
        "MSFT", mock_features_map["MSFT"])
    mock_predictor.predict_entry.assert_any_call(
        "GOOG", mock_features_map["GOOG"])


# TODO: Add tests for cases where only rules generate signals, error handling
