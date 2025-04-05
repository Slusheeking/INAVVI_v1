"""Tests for ai_day_trader.stock_selector"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from ai_day_trader.stock_selector import (
    StockSelector,
    DEFAULT_CANDIDATE_CACHE_KEY, # Import default for testing cache
    DEFAULT_CANDIDATE_CACHE_TTL_SECONDS,
    DEFAULT_MIN_SHARE_PRICE,
    DEFAULT_MIN_AVG_DOLLAR_VOLUME
)
from ai_day_trader.config import load_ai_trader_config
# Redis functions are patched, no direct import needed here for them

# Sample snapshot data for mocking
MOCK_SNAPSHOT_DATA = [
    { # Should pass
        "ticker": "AAPL",
        "prevDay": {"c": 150.0, "v": 50000000, "o": 149, "h": 151, "l": 148},
        "day": {"c": 151.0, "v": 10000000},
    },
    { # Should fail (price too low)
        "ticker": "LOWP",
        "prevDay": {"c": 4.0, "v": 10000000, "o": 3.9, "h": 4.1, "l": 3.8},
        "day": {"c": 4.1, "v": 100000},
    },
    { # Should fail (dollar volume too low)
        "ticker": "LOWV",
        "prevDay": {"c": 20.0, "v": 10000, "o": 19, "h": 21, "l": 18}, # 20*10k = 200k < 5M
        "day": {"c": 20.5, "v": 1000},
    },
    { # Should pass
        "ticker": "MSFT",
        "prevDay": {"c": 300.0, "v": 30000000, "o": 299, "h": 301, "l": 298},
        "day": {"c": 301.0, "v": 5000000},
    },
    { # Should fail (price too high based on default risk calc)
      # Max risk = 100k * 0.005 = 500. Stop = 0.02. Buffer = 1.2
      # Max price = (500 / 0.02) * 1.2 = 25000 * 1.2 = 30000
        "ticker": "HIGH",
        "prevDay": {"c": 35000.0, "v": 1000000, "o": 34900, "h": 35100, "l": 34800},
        "day": {"c": 35000.0, "v": 10000},
    },
     { # Ticker not in initial fetch list (should be ignored)
        "ticker": "XYZ",
        "prevDay": {"c": 50.0, "v": 60000000, "o": 49, "h": 51, "l": 48},
        "day": {"c": 50.0, "v": 100000},
    },
]

MOCK_TICKER_LIST = ["AAPL", "MSFT", "LOWP", "LOWV", "HIGH"] # Tickers from _fetch_all_active_tickers

@pytest.fixture
def mock_polygon_client():
    """Provides a mocked PolygonRESTClient with snapshot mock."""
    client = AsyncMock()

    # Mock the generic get method used for fetching the initial ticker list
    mock_ticker_response = AsyncMock()
    mock_ticker_response.status_code = 200
    mock_ticker_response.json = MagicMock(return_value={
        "results": [{"ticker": t} for t in MOCK_TICKER_LIST],
        "next_url": None # Simulate single page for simplicity
    })
    client.get = AsyncMock(return_value=mock_ticker_response)
    client.base_url = "https://api.polygon.io" # Needed for url replace logic

    # Mock the new snapshot method
    client.get_all_tickers_snapshot = AsyncMock(return_value=MOCK_SNAPSHOT_DATA)

    return client

# This fixture patches the redis client functions used by the selector
# We pass the mocks created by patch into the test function using their names
@pytest.fixture
def patch_redis_functions():
    # Patch the functions within the stock_selector module where they are imported/used
    with patch('ai_day_trader.stock_selector.get_redis_key', new_callable=AsyncMock) as mock_get, \
         patch('ai_day_trader.stock_selector.set_redis_key', new_callable=AsyncMock) as mock_set, \
         patch('ai_day_trader.stock_selector.get_async_redis_client', new_callable=AsyncMock) as mock_get_client:
        # Configure the mock client to be returned by get_async_redis_client
        mock_redis_instance = AsyncMock()
        mock_get_client.return_value = mock_redis_instance
        # Yield a dictionary of mocks to be used in the test
        yield {'get_redis_key': mock_get, 'set_redis_key': mock_set, 'get_async_redis_client': mock_get_client}


@pytest.mark.asyncio
async def test_stock_selector_init(mock_polygon_client):
    """Test StockSelector initialization and config loading."""
    config = load_ai_trader_config()
    # Optionally override config for test if needed, e.g., by patching load_ai_trader_config
    # or modifying the loaded config object before passing it in.
    selector = StockSelector(config, mock_polygon_client)
    assert selector is not None
    assert selector.polygon_client == mock_polygon_client
    # Check if config values were loaded (using defaults here)
    assert selector.min_share_price == DEFAULT_MIN_SHARE_PRICE
    assert selector.min_avg_dollar_volume == DEFAULT_MIN_AVG_DOLLAR_VOLUME
    assert selector.candidate_cache_key == DEFAULT_CANDIDATE_CACHE_KEY
    assert selector.candidate_cache_ttl == DEFAULT_CANDIDATE_CACHE_TTL_SECONDS

@pytest.mark.asyncio
async def test_select_candidate_symbols_no_cache(mock_polygon_client, patch_redis_functions):
    """Test selecting symbols when cache is empty, using snapshot API."""
    # Arrange
    config = load_ai_trader_config()
    selector = StockSelector(config, mock_polygon_client)
    mock_get_key = patch_redis_functions['get_redis_key']
    mock_set_key = patch_redis_functions['set_redis_key']
    mock_get_key.return_value = None # Ensure cache miss

    # Act
    candidates = await selector.select_candidate_symbols(force_refresh=False) # Test cache miss path

    # Assert
    assert isinstance(candidates, list)
    # Based on MOCK_SNAPSHOT_DATA and default filters, only AAPL and MSFT should pass
    assert len(candidates) == 2
    assert "AAPL" in candidates
    assert "MSFT" in candidates
    assert "LOWP" not in candidates
    assert "LOWV" not in candidates
    assert "HIGH" not in candidates
    assert "XYZ" not in candidates # Should be ignored as not in initial MOCK_TICKER_LIST

    # Check mocks
    mock_get_key.assert_called_once_with(DEFAULT_CANDIDATE_CACHE_KEY)
    mock_polygon_client.get.assert_called_once() # Called by _fetch_all_active_tickers
    mock_polygon_client.get_all_tickers_snapshot.assert_called_once() # Called by _filter_tickers_by_criteria
    # Check content being cached - ensure order doesn't matter for assertion
    mock_set_key.assert_called_once()
    args, kwargs = mock_set_key.call_args
    assert args[0] == DEFAULT_CANDIDATE_CACHE_KEY
    # Load the JSON string passed to set_redis_key and compare as sets
    cached_list = json.loads(args[1])
    assert set(cached_list) == {"AAPL", "MSFT"}
    assert kwargs['expire'] == DEFAULT_CANDIDATE_CACHE_TTL_SECONDS


@pytest.mark.asyncio
async def test_select_candidate_symbols_force_refresh(mock_polygon_client, patch_redis_functions):
    """Test selecting symbols with force_refresh=True ignores cache."""
    # Arrange
    config = load_ai_trader_config()
    selector = StockSelector(config, mock_polygon_client)
    mock_get_key = patch_redis_functions['get_redis_key']
    mock_set_key = patch_redis_functions['set_redis_key']
    # Set up a dummy cache value that should be ignored
    mock_get_key.return_value = json.dumps(["IGNOREME"]).encode('utf-8')

    # Act
    candidates = await selector.select_candidate_symbols(force_refresh=True)

    # Assert
    # Results should be the same as the no_cache test
    assert len(candidates) == 2
    assert "AAPL" in candidates
    assert "MSFT" in candidates

    # Check mocks
    mock_get_key.assert_not_called() # force_refresh=True should skip cache check
    mock_polygon_client.get.assert_called_once() # Fetch tickers called
    mock_polygon_client.get_all_tickers_snapshot.assert_called_once() # Snapshot called
    mock_set_key.assert_called_once() # Should cache the fresh result

@pytest.mark.asyncio
async def test_select_candidate_symbols_with_cache(mock_polygon_client, patch_redis_functions):
    """Test selecting symbols when cache is available."""
    # Arrange
    config = load_ai_trader_config()
    selector = StockSelector(config, mock_polygon_client)
    mock_get_key = patch_redis_functions['get_redis_key']
    mock_set_key = patch_redis_functions['set_redis_key']

    # Configure the patched get_redis_key mock to return cached data
    cached_symbols = ["TSLA", "GOOG"]
    mock_get_key.return_value = json.dumps(cached_symbols).encode('utf-8')

    # Act
    candidates = await selector.select_candidate_symbols(force_refresh=False)

    # Assert
    assert candidates == cached_symbols # Should return exactly what was cached
    mock_get_key.assert_called_once_with(DEFAULT_CANDIDATE_CACHE_KEY) # Check default key
    # API calls should NOT be made
    mock_polygon_client.get.assert_not_called()
    mock_polygon_client.get_all_tickers_snapshot.assert_not_called()
    mock_set_key.assert_not_called() # Cache should not be written again

@pytest.mark.asyncio
async def test_filter_tickers_by_criteria_logic(mock_polygon_client):
    """Test the filtering logic of _filter_tickers_by_criteria directly."""
    # Arrange
    config = load_ai_trader_config()
    # Can override specific config values for this test if needed
    # config.STOCK_SELECTOR_MIN_PRICE = 10.0
    selector = StockSelector(config, mock_polygon_client)

    # Act
    # Call the private method directly for focused testing
    # We pass the list of tickers that _fetch_all_active_tickers would have found
    # The method will then call the mocked get_all_tickers_snapshot
    filtered_candidates = await selector._filter_tickers_by_criteria(MOCK_TICKER_LIST)

    # Assert
    # Based on MOCK_SNAPSHOT_DATA and default filters
    assert len(filtered_candidates) == 2
    assert "AAPL" in filtered_candidates
    assert "MSFT" in filtered_candidates
    assert "LOWP" not in filtered_candidates
    assert "LOWV" not in filtered_candidates
    assert "HIGH" not in filtered_candidates
    assert "XYZ" not in filtered_candidates # Ignored because not in MOCK_TICKER_LIST input

    # Verify the snapshot API was called by the method
    mock_polygon_client.get_all_tickers_snapshot.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_all_active_tickers_pagination(mock_polygon_client):
    """Test that _fetch_all_active_tickers handles pagination correctly."""
    # Arrange
    config = load_ai_trader_config()
    selector = StockSelector(config, mock_polygon_client)

    # Reset the primary mock client's get method for this specific test
    mock_polygon_client.get = AsyncMock()

    # Override the mock 'get' specifically for this test to simulate pagination
    page1_response = AsyncMock()
    page1_response.status_code = 200
    page1_response.json = MagicMock(return_value={
        "results": [{"ticker": "TICKA"}, {"ticker": "TICKB"}],
        "next_url": "https://api.polygon.io/v3/reference/tickers?cursor=abc" # Simulate next page
    })

    page2_response = AsyncMock()
    page2_response.status_code = 200
    page2_response.json = MagicMock(return_value={
        "results": [{"ticker": "TICKC"}],
        "next_url": None # Last page
    })

    # Make the mock return page1 first, then page2
    mock_polygon_client.get.side_effect = [page1_response, page2_response]

    # Act
    tickers = await selector._fetch_all_active_tickers()

    # Assert
    assert len(tickers) == 3
    assert tickers == ["TICKA", "TICKB", "TICKC"]
    assert mock_polygon_client.get.call_count == 2 # Should have called 'get' twice

# TODO: Add tests for API errors during snapshot fetch, Redis errors
