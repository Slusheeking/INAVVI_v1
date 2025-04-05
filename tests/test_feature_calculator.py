"""Tests for ai_day_trader.feature_calculator (GPU version)"""
import pytest
import pandas as pd
import numpy as np

# Try importing cudf, skip tests if unavailable
try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

from ai_day_trader.feature_calculator import FeatureCalculator

# Skip all tests in this module if cudf is not installed
pytestmark = pytest.mark.skipif(
    not CUDF_AVAILABLE, reason="cudf not installed, skipping GPU tests")


@pytest.fixture
def sample_config():
    """Provides a sample config dictionary."""
    return {}  # Use defaults for now


@pytest.fixture
def feature_calculator_instance(sample_config):
    """Provides an instance of the FeatureCalculator."""
    return FeatureCalculator(config=sample_config)


@pytest.fixture
def sample_ohlcv_df():
    """Creates a sample pandas DataFrame for testing."""
    periods = 100
    dates = pd.date_range(start='2023-01-01',
                          periods=periods, freq='min', tz='UTC')
    data = {
        'o': np.random.rand(periods) * 10 + 150,
        'h': np.random.rand(periods) * 5 + 155,
        'l': np.random.rand(periods) * 5 + 145,
        'c': np.random.rand(periods) * 10 + 150,
        # Ensure volume is float
        'v': np.random.randint(1000, 10000, size=periods).astype(float)
    }
    # Ensure high is >= open/close and low is <= open/close
    data['h'] = np.maximum(data['h'], data['o'])
    data['h'] = np.maximum(data['h'], data['c'])
    data['l'] = np.minimum(data['l'], data['o'])
    data['l'] = np.minimum(data['l'], data['c'])
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.mark.asyncio
async def test_feature_calculator_init(feature_calculator_instance):
    """Test FeatureCalculator initialization."""
    fc = feature_calculator_instance
    assert fc is not None
    assert fc.sma_short_period > 0
    assert hasattr(fc, '_calculate_sma')  # Check if methods exist


@pytest.mark.asyncio
async def test_calculate_features_gpu(feature_calculator_instance, sample_ohlcv_df):
    """Test feature calculation using cuDF on sample data."""
    fc = feature_calculator_instance
    df = sample_ohlcv_df

    features = await fc.calculate_features("TEST_GPU", df)

    # Check if the result is a dictionary and not None
    assert features is not None, "calculate_features returned None"
    assert isinstance(features, dict), "Result is not a dictionary"
    assert len(features) > 0, "No features were calculated"

    # Check for specific feature keys (adjust names based on actual calculation)
    assert f'sma_{fc.sma_short_period}' in features
    assert f'rsi_{fc.rsi_period}' in features
    assert f'atr_{fc.atr_period}' in features
    assert 'macd' in features
    assert 'bb_upper' in features
    assert 'price' in features
    assert 'volume' in features

    # Check that values are standard Python floats and finite (not NaN or Inf)
    # The implementation now converts results to float and handles NaN/Inf
    for key, value in features.items():
        assert isinstance(
            value, float), f"Feature '{key}' is not a float: {value} ({type(value)})"
        assert np.isfinite(
            value), f"Feature '{key}' is not finite (NaN/Inf): {value}"


@pytest.mark.asyncio
async def test_calculate_features_insufficient_data(feature_calculator_instance):
    """Test feature calculation with insufficient data."""
    fc = feature_calculator_instance
    # Create DataFrame with fewer rows than the longest lookback (e.g., sma_long_period)
    periods = fc.sma_long_period - 5
    dates = pd.date_range(start='2023-01-01',
                          periods=periods, freq='min', tz='UTC')
    data = {
        'o': np.random.rand(periods) * 10 + 150,
        'h': np.random.rand(periods) * 5 + 155,
        'l': np.random.rand(periods) * 5 + 145,
        'c': np.random.rand(periods) * 10 + 150,
        'v': np.random.randint(1000, 10000, size=periods).astype(float)
    }
    df = pd.DataFrame(data, index=dates)

    features = await fc.calculate_features("TEST_SHORT", df)

    # Depending on implementation, this might return None or a dict with NaNs/zeros
    # The current implementation returns a dict with NaNs replaced by 0.0
    assert features is not None
    assert isinstance(features, dict)
    # Check a feature that requires longer lookback - it should likely be 0.0 due to NaN replacement
    assert f'sma_{fc.sma_long_period}' in features
    assert features[f'sma_{fc.sma_long_period}'] == 0.0

# TODO: Add more specific tests for individual indicator calculations using known inputs/outputs
# TODO: Add tests for edge cases like zero prices or volumes if applicable
