"""Tests for ai_day_trader.ml.predictor"""
import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path
from datetime import datetime, timezone

# Assume necessary imports from the main project are available
# Adjust imports based on actual project structure if needed
from ai_day_trader.ml.predictor import Predictor
from ai_day_trader.utils.config import Config
from ai_day_trader.trading.execution.base import PositionInfo

# --- Fixtures ---


@pytest.fixture
def mock_config():
    """Fixture for a mock Config object."""
    config = MagicMock(spec=Config)
    config.get.side_effect = lambda key, default=None: {
        "MODEL_DIR": "./mock_models",
        "PREDICTOR_LOOKBACK_MINUTES": 60,
        "PREDICTOR_LOOKBACK_BUFFER": 10,
        "PREDICTOR_CACHE_TTL_SECONDS": 30,
    }.get(key, default)
    config.get_str.side_effect = lambda key, default=None: str({
        "MODEL_DIR": "./mock_models",
    }.get(key, default))
    config.as_dict.return_value = {}  # For FeatureCalculator if needed
    return config


@pytest.fixture
def mock_polygon_rest_client():
    """Fixture for a mock PolygonRESTClient."""
    client = MagicMock()
    client.get_aggregates = AsyncMock()
    return client


@pytest.fixture
def mock_feature_calculator():
    """Fixture for a mock FeatureCalculator."""
    calculator = MagicMock()
    calculator.calculate_features = AsyncMock()
    # Mock the new method required by Predictor
    calculator.get_required_lookback = MagicMock(
        return_value=50)  # Example lookback
    return calculator


@pytest.fixture
def mock_redis_client():
    """Fixture for a mock RedisClient (async)."""
    client = MagicMock()
    client.get = AsyncMock(return_value=None)  # Default to cache miss
    client.set = AsyncMock()
    return client


@pytest.fixture
def mock_scaler():
    """Fixture for a mock scaler."""
    scaler = MagicMock()
    scaler.transform = MagicMock(return_value=np.array(
        [[0.5, 0.6]]))  # Example scaled data
    return scaler


@pytest.fixture
def mock_model():
    """Fixture for a mock XGBoost model."""
    model = MagicMock()
    # Mock whatever interface load_xgboost_model returns
    return model


@pytest.fixture
@patch("ai_day_trader.ml.predictor.load_xgboost_model")
@patch("ai_day_trader.ml.predictor.joblib.load")
@patch("builtins.open")
@patch("ai_day_trader.ml.predictor.json.load")
@patch("ai_day_trader.ml.predictor.Path.exists")
def predictor_instance(mock_path_exists, mock_json_load, mock_open, mock_joblib_load, mock_load_xgb,
                       mock_config, mock_polygon_rest_client, mock_feature_calculator, mock_redis_client,
                       mock_model, mock_scaler):
    """Fixture to create a Predictor instance with mocked dependencies and loaded models."""
    mock_path_exists.return_value = True  # Assume all model files exist
    mock_json_load.side_effect = [['feat1', 'feat2'], [
        'feat1', 'feat2', 'holding_time_minutes']]  # Entry features, Exit features
    mock_joblib_load.return_value = mock_scaler
    mock_load_xgb.return_value = mock_model

    predictor = Predictor(
        config=mock_config,
        polygon_rest_client=mock_polygon_rest_client,
        feature_calculator=mock_feature_calculator,
        redis_client=mock_redis_client
    )
    # Manually set models/scalers as load_models is async and patching _load_model_artifacts is complex with async init
    predictor.entry_model = mock_model
    predictor.entry_scaler = mock_scaler
    predictor.entry_features = ['feat1', 'feat2']
    predictor.exit_model = mock_model
    predictor.exit_scaler = mock_scaler
    predictor.exit_features = ['feat1', 'feat2',
                               'holding_time_minutes']  # Example exit features

    # Reset mocks that might be called during init indirectly if load_models was called
    mock_load_xgb.reset_mock()
    mock_joblib_load.reset_mock()
    mock_json_load.reset_mock()

    return predictor

# --- Test Cases ---


@pytest.mark.asyncio
async def test_predictor_init(mock_config, mock_polygon_rest_client, mock_feature_calculator, mock_redis_client):
    """Test predictor initialization sets config values correctly."""
    predictor = Predictor(mock_config, mock_polygon_rest_client,
                          mock_feature_calculator, mock_redis_client)
    assert predictor.config == mock_config
    assert predictor.rest_client == mock_polygon_rest_client
    assert predictor.feature_calculator == mock_feature_calculator
    assert predictor.redis_client == mock_redis_client
    assert predictor.model_dir == Path("./mock_models")
    assert predictor.lookback_minutes == 60
    assert predictor.lookback_limit_buffer == 10
    assert predictor.cache_ttl_seconds == 30
    assert predictor.entry_model is None  # Before loading
    assert predictor.exit_model is None  # Before loading


@pytest.mark.asyncio
@patch("ai_day_trader.ml.predictor.load_xgboost_model")
@patch("ai_day_trader.ml.predictor.joblib.load")
@patch("builtins.open")
@patch("ai_day_trader.ml.predictor.json.load")
@patch("ai_day_trader.ml.predictor.Path.exists")
async def test_load_models_success(mock_path_exists, mock_json_load, mock_open, mock_joblib_load, mock_load_xgb,
                                   mock_config, mock_polygon_rest_client, mock_feature_calculator, mock_redis_client,
                                   mock_model, mock_scaler):
    """Test successful loading of models."""
    mock_path_exists.return_value = True
    mock_json_load.side_effect = [['f1', 'f2'],
                                  ['f3', 'f4']]  # Entry, Exit features
    mock_joblib_load.return_value = mock_scaler
    mock_load_xgb.return_value = mock_model

    predictor = Predictor(mock_config, mock_polygon_rest_client,
                          mock_feature_calculator, mock_redis_client)
    await predictor.load_models()

    assert predictor.entry_model == mock_model
    assert predictor.entry_scaler == mock_scaler
    assert predictor.entry_features == ['f1', 'f2']
    assert predictor.exit_model == mock_model
    assert predictor.exit_scaler == mock_scaler
    assert predictor.exit_features == ['f3', 'f4']
    assert mock_load_xgb.call_count == 2
    assert mock_joblib_load.call_count == 2
    assert mock_json_load.call_count == 2


@pytest.mark.asyncio
@patch("ai_day_trader.ml.predictor.predict_with_xgboost")
async def test_predict_entry_fetch_features(mock_predict_xgb, predictor_instance, mock_polygon_rest_client, mock_feature_calculator, mock_redis_client, mock_scaler):
    """Test predict_entry when features are NOT pre-calculated (fetch path)."""
    symbol = "AAPL"
    mock_redis_client.get.return_value = None  # Cache miss for features
    mock_predict_xgb.return_value = (
        np.array([1]), np.array([0.85]))  # prediction, probability

    # Mock feature preparation steps (as called by _get_and_prepare_features)
    hist_data = pd.DataFrame({'o': [100], 'h': [101], 'l': [99], 'c': [
                             100.5], 'v': [1000]}, index=[pd.Timestamp.utcnow()])
    # Mock the return value of calculate_features (should be a DataFrame)
    features_calculated_df = pd.DataFrame(
        {'feat1': [10], 'feat2': [20]}, index=hist_data.index)
    # Mock the return value of scaler.transform (should be a numpy array or similar)
    features_scaled_np = np.array([[0.5, 0.6]])
    # The DataFrame created *after* scaling
    features_scaled_df = pd.DataFrame(
        features_scaled_np, index=hist_data.index, columns=['feat1', 'feat2'])

    mock_polygon_rest_client.get_aggregates.return_value = hist_data
    # Return DataFrame
    mock_feature_calculator.calculate_features.return_value = features_calculated_df
    mock_scaler.transform.return_value = features_scaled_np  # Return numpy array

    # Expected cache data (latest row, index as ms timestamp)
    # Create from post-scaling DataFrame
    expected_cache_df = features_scaled_df.iloc[[-1]].copy()
    expected_cache_df.index = expected_cache_df.index.astype(
        int) // 10**6  # Convert index for JSON
    expected_cache_json = expected_cache_df.to_json(orient='split')

    # Call predict_entry WITHOUT providing features, forcing fetch
    result = await predictor_instance.predict_entry(symbol)

    assert result == {"prediction": 1, "probability": 0.85}
    # Verify mocks for the fetch path
    mock_redis_client.get.assert_awaited_once_with(
        f"predictor:features:{symbol}:entry")
    mock_feature_calculator.get_required_lookback.assert_called_once_with(
        predictor_instance.entry_features)
    expected_fetch_lookback = 60  # max(50, 60)
    expected_limit = expected_fetch_lookback + 10  # 70
    mock_polygon_rest_client.get_aggregates.assert_awaited_once_with(
        symbol=symbol, multiplier=1, timespan="minute", limit=expected_limit)
    mock_feature_calculator.calculate_features.assert_awaited_once_with(
        symbol, hist_data)
    # Check transform was called with the selected features DataFrame
    pd.testing.assert_frame_equal(
        mock_scaler.transform.call_args[0][0], features_calculated_df[predictor_instance.entry_features])
    mock_redis_client.set.assert_awaited_once_with(
        f"predictor:features:{symbol}:entry", expected_cache_json, ex=30)
    mock_predict_xgb.assert_called_once()
    # Check the DataFrame passed to predict_with_xgboost (should be the single latest row after scaling)
    pd.testing.assert_frame_equal(
        mock_predict_xgb.call_args[0][1], features_scaled_df.iloc[[-1]])


@pytest.mark.asyncio
@patch("ai_day_trader.ml.predictor.predict_with_xgboost")
async def test_predict_entry_precalculated_features(mock_predict_xgb, predictor_instance, mock_polygon_rest_client, mock_feature_calculator, mock_redis_client, mock_scaler):
    """Test predict_entry when features ARE pre-calculated."""
    symbol = "TSLA"
    # Mock the pre-calculated features (unscaled)
    precalculated_features = {'feat1': 12.0, 'feat2': 22.0, 'other_feat': 5}
    # Mock the scaler output for these features
    features_scaled_np = np.array([[0.55, 0.65]])
    features_scaled_df = pd.DataFrame(features_scaled_np, columns=[
                                      'feat1', 'feat2'])  # Scaler only uses entry_features

    mock_scaler.transform.return_value = features_scaled_np
    mock_predict_xgb.return_value = (
        np.array([1]), np.array([0.7]))  # prediction, probability

    # Call predict_entry WITH features
    result = await predictor_instance.predict_entry(symbol, features=precalculated_features)

    assert result == {"prediction": 1, "probability": 0.7}
    # Verify mocks
    # Feature fetching/calculation should NOT be called
    mock_redis_client.get.assert_not_called()
    mock_feature_calculator.get_required_lookback.assert_not_called()
    mock_polygon_rest_client.get_aggregates.assert_not_called()
    mock_feature_calculator.calculate_features.assert_not_called()
    mock_redis_client.set.assert_not_called()
    # Scaler transform should be called with the relevant precalculated features
    mock_scaler.transform.assert_called_once()
    call_args, _ = mock_scaler.transform.call_args
    input_df_to_scaler = call_args[0]
    assert isinstance(input_df_to_scaler, pd.DataFrame)
    # ['feat1', 'feat2']
    assert list(input_df_to_scaler.columns) == predictor_instance.entry_features
    assert input_df_to_scaler['feat1'].iloc[0] == 12.0
    assert input_df_to_scaler['feat2'].iloc[0] == 22.0
    # Prediction should be called with the scaled features
    mock_predict_xgb.assert_called_once()
    pd.testing.assert_frame_equal(
        mock_predict_xgb.call_args[0][1], features_scaled_df)


@pytest.mark.asyncio
@patch("ai_day_trader.ml.predictor.predict_with_xgboost")
@patch("pandas.read_json")  # Patch read_json used for deserialization
async def test_predict_entry_cache_hit(mock_pd_read_json, mock_predict_xgb, predictor_instance, mock_redis_client, mock_polygon_rest_client, mock_feature_calculator, mock_scaler):
    """Test predict_entry cache hit (legacy path, features not provided)."""
    symbol = "MSFT"
    # Prepare cached data (DataFrame matching scaled features)
    cached_features_df = pd.DataFrame(
        {'feat1': [0.7], 'feat2': [0.8]}, index=[pd.Timestamp.utcnow()])
    # Simulate the structure stored in cache (split orient JSON, index as ms)
    cache_store_df = cached_features_df.copy()
    cache_store_df.index = cache_store_df.index.astype(int) // 10**6
    cached_data_json = cache_store_df.to_json(orient='split')

    mock_redis_client.get.return_value = cached_data_json
    # Mock pandas.read_json to return the DataFrame when called by predictor
    # Assume it handles index conversion correctly
    mock_pd_read_json.return_value = cached_features_df

    mock_predict_xgb.return_value = (
        np.array([0]), np.array([0.15]))  # prediction, probability

    # Call predict_entry WITHOUT providing features, forcing cache check
    result = await predictor_instance.predict_entry(symbol)

    assert result == {"prediction": 0, "probability": 0.15}
    # Verify mocks for cache hit path
    mock_redis_client.get.assert_awaited_once_with(
        f"predictor:features:{symbol}:entry")
    mock_pd_read_json.assert_called_once_with(
        json.loads(cached_data_json), orient='split')
    # Ensure feature calculation steps were NOT called
    mock_feature_calculator.get_required_lookback.assert_not_called()
    mock_polygon_rest_client.get_aggregates.assert_not_called()
    mock_feature_calculator.calculate_features.assert_not_called()
    mock_scaler.transform.assert_not_called()
    mock_redis_client.set.assert_not_called()  # Should not set if cache hit
    # Ensure prediction was called with the cached DataFrame
    mock_predict_xgb.assert_called_once()
    pd.testing.assert_frame_equal(
        mock_predict_xgb.call_args[0][1], cached_features_df)


@pytest.mark.asyncio
@patch("ai_day_trader.ml.predictor.predict_with_xgboost")
async def test_predict_exit_fetch_features(mock_predict_xgb, predictor_instance, mock_polygon_rest_client, mock_feature_calculator, mock_redis_client, mock_scaler):
    """Test predict_exit when features are NOT pre-calculated (fetch path)."""
    symbol = "GOOG"
    entry_time_dt = datetime.now(timezone.utc) - pd.Timedelta(hours=1)
    # Mock position info needs attributes accessed in predict_exit
    position_info = MagicMock()
    position_info.entry_time = entry_time_dt.isoformat()  # Store as string initially
    position_info.avg_entry_price = "2800.0"
    position_info.current_price = "2812.0"  # Assume this is available

    mock_redis_client.get.return_value = None  # Cache miss
    mock_predict_xgb.return_value = (np.array([1]), np.array(
        [0.92]))  # prediction (exit), probability

    # Mock feature preparation steps (as called by _get_and_prepare_features)
    latest_timestamp = datetime.now(timezone.utc)
    hist_data = pd.DataFrame({'o': [2810], 'h': [2815], 'l': [2805], 'c': [
                             2812], 'v': [500]}, index=[pd.Timestamp(latest_timestamp)])
    # Mock the return value of calculate_features (should be a DataFrame)
    # Note: Exit features include 'holding_time_minutes', but calculate_features doesn't add it
    features_calculated_df = pd.DataFrame(
        {'feat1': [15], 'feat2': [25], 'close': [2812]}, index=hist_data.index)
    # Mock the return value of scaler.transform (should be a numpy array or similar)
    # Scaler only sees base features ['feat1', 'feat2']
    features_scaled_np = np.array([[0.7, 0.75]])
    # The DataFrame created *after* scaling base features
    features_scaled_df = pd.DataFrame(
        features_scaled_np, index=hist_data.index, columns=['feat1', 'feat2'])

    mock_polygon_rest_client.get_aggregates.return_value = hist_data
    # Return DataFrame
    mock_feature_calculator.calculate_features.return_value = features_calculated_df
    mock_scaler.transform.return_value = features_scaled_np  # Return numpy array

    # Call predict_exit WITHOUT providing base_features
    result = await predictor_instance.predict_exit(symbol, position_info)

    assert result == {"prediction": 1, "probability": 0.92}
    # Verify mocks for the fetch path
    mock_redis_client.get.assert_awaited_once_with(
        f"predictor:features:{symbol}:exit")
    mock_feature_calculator.get_required_lookback.assert_called_once_with(
        predictor_instance.exit_features)
    mock_polygon_rest_client.get_aggregates.assert_awaited_once()
    mock_feature_calculator.calculate_features.assert_awaited_once()
    # Check scaler was called with the base features required by the exit model
    base_exit_features = [f for f in predictor_instance.exit_features if f not in [
        'holding_time_minutes', 'unrealized_pnl_pct']]
    pd.testing.assert_frame_equal(
        mock_scaler.transform.call_args[0][0], features_calculated_df[base_exit_features])
    # Cache set called by _get_and_prepare
    mock_redis_client.set.assert_awaited_once()

    # Verify prediction call
    mock_predict_xgb.assert_called_once()
    call_args, _ = mock_predict_xgb.call_args
    predictor_input_df = call_args[1]
    # Check that the DataFrame passed to predict has the correct columns and values
    assert 'feat1' in predictor_input_df.columns
    assert 'feat2' in predictor_input_df.columns
    assert 'holding_time_minutes' in predictor_input_df.columns
    assert predictor_input_df['feat1'].iloc[0] == pytest.approx(
        0.7)  # From scaled data
    assert predictor_input_df['feat2'].iloc[0] == pytest.approx(
        0.75)  # From scaled data
    # Check calculated holding time (approx 60 mins)
    expected_holding_minutes = (
        pd.Timestamp.utcnow() - entry_time_dt).total_seconds() / 60
    assert predictor_input_df['holding_time_minutes'].iloc[0] == pytest.approx(
        expected_holding_minutes, abs=1)  # Allow slight timing difference
    # Check column order matches exit_features
    assert list(predictor_input_df.columns) == predictor_instance.exit_features


@pytest.mark.asyncio
@patch("ai_day_trader.ml.predictor.predict_with_xgboost")
async def test_predict_exit_precalculated_features(mock_predict_xgb, predictor_instance, mock_polygon_rest_client, mock_feature_calculator, mock_redis_client, mock_scaler):
    """Test predict_exit when base features ARE pre-calculated."""
    symbol = "AMZN"
    entry_time_dt = datetime.now(timezone.utc) - pd.Timedelta(minutes=30)
    # Mock position info needs attributes accessed in predict_exit
    position_info = MagicMock()
    position_info.entry_time = entry_time_dt.isoformat()
    position_info.avg_entry_price = "3000.0"
    position_info.current_price = "3015.0"  # Assume this is available

    # Mock the pre-calculated base features (unscaled)
    # Include 'close' if needed for PnL calc
    precalculated_base_features = {
        'feat1': 18.0, 'feat2': 28.0, 'close': 3015.0}
    # Mock the scaler output for these base features
    features_scaled_np = np.array([[0.8, 0.85]])
    features_scaled_df = pd.DataFrame(
        features_scaled_np, columns=['feat1', 'feat2'])

    mock_scaler.transform.return_value = features_scaled_np
    # prediction (hold), probability
    mock_predict_xgb.return_value = (np.array([0]), np.array([0.2]))

    # Call predict_exit WITH base_features
    result = await predictor_instance.predict_exit(symbol, position_info, base_features=precalculated_base_features)

    assert result == {"prediction": 0, "probability": 0.2}
    # Verify mocks
    # Feature fetching/calculation should NOT be called
    mock_redis_client.get.assert_not_called()
    mock_feature_calculator.get_required_lookback.assert_not_called()
    mock_polygon_rest_client.get_aggregates.assert_not_called()
    mock_feature_calculator.calculate_features.assert_not_called()
    mock_redis_client.set.assert_not_called()
    # Scaler transform should be called with the relevant precalculated base features
    mock_scaler.transform.assert_called_once()
    call_args, _ = mock_scaler.transform.call_args
    input_df_to_scaler = call_args[0]
    assert isinstance(input_df_to_scaler, pd.DataFrame)
    base_exit_features = [f for f in predictor_instance.exit_features if f not in [
        'holding_time_minutes', 'unrealized_pnl_pct']]
    assert list(input_df_to_scaler.columns) == base_exit_features
    assert input_df_to_scaler['feat1'].iloc[0] == 18.0
    assert input_df_to_scaler['feat2'].iloc[0] == 28.0

    # Prediction should be called with the final feature vector
    mock_predict_xgb.assert_called_once()
    call_args, _ = mock_predict_xgb.call_args
    predictor_input_df = call_args[1]
    assert 'feat1' in predictor_input_df.columns
    assert 'feat2' in predictor_input_df.columns
    assert 'holding_time_minutes' in predictor_input_df.columns
    assert predictor_input_df['feat1'].iloc[0] == pytest.approx(0.8)  # Scaled
    assert predictor_input_df['feat2'].iloc[0] == pytest.approx(0.85)  # Scaled
    expected_holding_minutes = (
        pd.Timestamp.utcnow() - entry_time_dt).total_seconds() / 60
    assert predictor_input_df['holding_time_minutes'].iloc[0] == pytest.approx(
        expected_holding_minutes, abs=1)
    assert list(predictor_input_df.columns) == predictor_instance.exit_features
