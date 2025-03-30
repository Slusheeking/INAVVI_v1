#!/usr/bin/env python3
"""
Market Data Helpers Module

Contains utility functions for acquiring and preprocessing market data.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import redis # For type hint

# Import specific client types for clarity
from api_clients.polygon_rest import PolygonRESTClient
from api_clients.polygon_ws import PolygonWebSocketClient
from api_clients.unusual_whales import UnusualWhalesClient
# Use consistent async utilities
from utils.async_utils import async_retry, timeout_context
# Use specific exception types
from utils.exceptions import APIError, APITimeoutError, DataProcessingError
from utils.metrics_registry import (
    MARKET_DATA_LATENCY,
    MARKET_DATA_ERRORS
)

logger = logging.getLogger("market_data")

# Define more specific return types if the structure is known
# Example: Structure of historical data from Polygon
HistoricalData = Dict[str, Any] # Replace Any with specific types if known (e.g., List[Dict[str, float]])
# Example: Structure of a real-time quote from Polygon
QuoteData = Dict[str, Any] # Replace Any with specific types
# Example: Structure of unusual options activity data
OptionsActivityData = Dict[str, Any] # Replace Any with specific types

# Define a type for processed data if its structure is consistent
ProcessedData = Dict[str, Any] # Replace Any with specific types

@async_retry(max_retries=3, retry_delay=1.5)
async def get_historical_data(
    polygon_client: PolygonRESTClient,
    ticker: str,
    timespan: str = "day", # Example: Add timespan parameter
    multiplier: int = 1,   # Example: Add multiplier
    start_date: str = None, # Example: Add date range
    end_date: str = None,   # Example: Add date range
    limit: int = 5000,      # Example: Add limit
    redis_client: Optional[redis.Redis] = None # Keep redis_client optional for potential caching
) -> HistoricalData:
    """
    Retrieves historical aggregate bar data for a ticker using the Polygon REST client.

    Applies retry logic and timeout context.

    Args:
        polygon_client: Initialized PolygonRESTClient instance.
        ticker: The stock ticker symbol.
        timespan: The size of the time window (e.g., 'minute', 'hour', 'day').
        multiplier: The multiplier for the timespan (e.g., 1, 5).
        start_date: Start date for the data range (YYYY-MM-DD).
        end_date: End date for the data range (YYYY-MM-DD).
        limit: Maximum number of base aggregates queried.
        redis_client: Optional Redis client for caching (not implemented in this example).

    Returns:
        A dictionary containing the historical aggregate data from Polygon API.

    Raises:
        APITimeoutError: If the API call times out after retries.
        APIError: If any other API-related error occurs.
        ValueError: If required parameters for the API call are missing.
    """
    if not polygon_client:
        raise ValueError("PolygonRESTClient instance is required.")

    logger.debug(f"Fetching historical data for {ticker} ({timespan} bars)...")
    start_time = asyncio.get_event_loop().time()
    metric_labels = {"client": "polygon", "endpoint": "aggregates", "ticker": ticker}

    try:
        # Use timeout context for the API call within the retry attempt
        async with timeout_context(20.0): # Increased timeout for potentially large data requests
            # Assuming the client method handles parameter validation internally
            data = await polygon_client.get_aggregates(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_date=start_date,
                to_date=end_date,
                limit=limit
            )
            # Consider adding basic validation of the returned data structure here
            if data is None or not isinstance(data, dict):
                 logger.warning(f"Received unexpected data format for {ticker}: {type(data)}")
                 # Depending on strictness, could raise APIError here
                 return {} # Return empty dict on unexpected format

            latency = asyncio.get_event_loop().time() - start_time
            MARKET_DATA_LATENCY.labels(**metric_labels).observe(latency)
            logger.debug(f"Fetched historical data for {ticker} in {latency:.4f}s.")
            return data

    except asyncio.TimeoutError as e:
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="timeout").inc()
        logger.error(f"Timeout fetching historical data for {ticker} after retries.")
        raise APITimeoutError(f"Timeout fetching historical data for {ticker}") from e
    except APIError as e: # Catch specific API errors from the client
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="api_error").inc()
        logger.error(f"API error fetching historical data for {ticker}: {e}")
        raise # Re-raise the specific APIError
    except Exception as e:
        # Catch unexpected errors during the process
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="unknown").inc()
        logger.exception(f"Unexpected error fetching historical data for {ticker}: {e}")
        raise APIError(f"Unexpected error fetching historical data for {ticker}: {e}") from e


async def get_realtime_quotes(
    polygon_client: PolygonRESTClient,
    tickers: List[str],
    redis_client: Optional[redis.Redis] = None # Keep for potential caching
) -> Dict[str, QuoteData]:
    """
    Retrieves the last quote for multiple tickers using the Polygon REST client.

    Args:
        polygon_client: Initialized PolygonRESTClient instance.
        tickers: A list of stock ticker symbols.
        redis_client: Optional Redis client for caching.

    Returns:
        A dictionary mapping ticker symbols to their last quote data.

    Raises:
        APITimeoutError: If the API call times out.
        APIError: If any other API-related error occurs.
        ValueError: If polygon_client is not provided or tickers list is empty.
    """
    if not polygon_client:
        raise ValueError("PolygonRESTClient instance is required.")
    if not tickers:
        logger.warning("get_realtime_quotes called with empty tickers list.")
        return {}

    logger.debug(f"Fetching real-time quotes for {len(tickers)} tickers...")
    start_time = asyncio.get_event_loop().time()
    # Note: Batching requests might be more efficient if the API supports it
    # This implementation fetches quotes sequentially.

    results: Dict[str, QuoteData] = {}
    tasks = []

    # Create tasks for fetching each quote concurrently
    for ticker in tickers:
        metric_labels = {"client": "polygon", "endpoint": "last_quote", "ticker": ticker}
        async def fetch_single_quote(t: str, labels: Dict):
            try:
                async with timeout_context(5.0): # Timeout per quote request
                    quote = await polygon_client.get_last_quote(t)
                    if quote and isinstance(quote, dict):
                         return t, quote
                    else:
                         logger.warning(f"Received unexpected quote format for {t}: {type(quote)}")
                         MARKET_DATA_ERRORS.labels(**labels, error_type="invalid_format").inc()
                         return t, None # Indicate failure for this ticker
            except asyncio.TimeoutError:
                MARKET_DATA_ERRORS.labels(**labels, error_type="timeout").inc()
                logger.error(f"Timeout fetching quote for {t}.")
                return t, None # Indicate failure
            except APIError as e:
                MARKET_DATA_ERRORS.labels(**labels, error_type="api_error").inc()
                logger.error(f"API error fetching quote for {t}: {e}")
                return t, None # Indicate failure
            except Exception as e:
                MARKET_DATA_ERRORS.labels(**labels, error_type="unknown").inc()
                logger.exception(f"Unexpected error fetching quote for {t}: {e}")
                return t, None # Indicate failure

        tasks.append(fetch_single_quote(ticker, metric_labels))

    # Gather results from concurrent fetches
    quote_results = await asyncio.gather(*tasks)

    # Process results, filtering out failures
    for ticker, quote_data in quote_results:
        if quote_data is not None:
            results[ticker] = quote_data

    total_latency = asyncio.get_event_loop().time() - start_time
    # Latency metric here reflects total time for all concurrent requests
    MARKET_DATA_LATENCY.labels(client="polygon", endpoint="last_quote_batch", ticker="batch").observe(total_latency)
    logger.debug(f"Fetched {len(results)} quotes out of {len(tickers)} requested in {total_latency:.4f}s.")

    # Check if all requests failed, potentially raise an error
    if not results and tickers:
         logger.error("Failed to fetch any real-time quotes.")
         # Decide whether to raise an error or return empty dict based on requirements
         # raise APIError("Failed to fetch any real-time quotes for the provided tickers.")

    return results


async def preprocess_market_data(
    raw_data: Dict[str, Any], # Input type depends on the source
    use_gpu: bool = False # Keep GPU flag if preprocessing can be accelerated
) -> Tuple[ProcessedData, Optional[Dict[str, Any]]]:
    """
    Cleans, normalizes, and potentially transforms raw market data.

    Placeholder: Performs basic filtering of None values.
    In a real system, this could involve handling missing data, scaling features,
    calculating indicators, converting data types, etc.

    Args:
        raw_data: Raw market data dictionary (structure depends on source).
        use_gpu: Flag indicating if GPU acceleration should be attempted (if applicable).

    Returns:
        A tuple containing:
        - ProcessedData: The cleaned and transformed data.
        - Optional[Dict[str, Any]]: Optional dictionary containing metadata about the processing.

    Raises:
        DataProcessingError: If a critical error occurs during preprocessing.
    """
    logger.debug(f"Preprocessing market data containing {len(raw_data)} items...")
    start_time = asyncio.get_event_loop().time()
    processed_data: ProcessedData = {}
    metadata: Optional[Dict[str, Any]] = {"processing_start_time": start_time}

    try:
        # --- Placeholder Preprocessing Logic ---
        # Example: Simple filtering of None values
        # This could be done concurrently for large datasets
        processed_data = {k: v for k, v in raw_data.items() if v is not None}
        items_removed = len(raw_data) - len(processed_data)
        if items_removed > 0:
            logger.debug(f"Removed {items_removed} items with None values during preprocessing.")
            metadata["items_removed"] = items_removed

        # Example: Add more steps like type conversion, calculation, etc.
        # await asyncio.sleep(0.01) # Simulate work
        # --- End Placeholder Preprocessing Logic ---

        processing_time = asyncio.get_event_loop().time() - start_time
        metadata["processing_duration_seconds"] = round(processing_time, 4)
        logger.debug(f"Market data preprocessing complete in {processing_time:.4f}s.")

        return processed_data, metadata

    except Exception as e:
        logger.exception(f"Error during market data preprocessing: {e}")
        MARKET_DATA_ERRORS.labels(client="internal", endpoint="preprocess", error_type="processing_error").inc()
        raise DataProcessingError(f"Failed to preprocess market data: {e}") from e


@async_retry(max_retries=3, retry_delay=2.0)
async def get_unusual_options_activity(
    unusual_whales_client: UnusualWhalesClient,
    min_volume: int = 100,
    limit: int = 1000 # Example: Add limit parameter
) -> List[OptionsActivityData]:
    """
    Retrieves unusual options activity data using the UnusualWhales client.

    Applies retry logic and timeout context.

    Args:
        unusual_whales_client: Initialized UnusualWhalesClient instance.
        min_volume: Minimum volume threshold for filtering activity.
        limit: Maximum number of activity records to retrieve.

    Returns:
        A list of dictionaries, each representing an unusual options activity event.

    Raises:
        APITimeoutError: If the API call times out after retries.
        APIError: If any other API-related error occurs.
        ValueError: If unusual_whales_client is not provided.
    """
    if not unusual_whales_client:
        raise ValueError("UnusualWhalesClient instance is required.")

    logger.debug(f"Fetching unusual options activity (min_volume={min_volume}, limit={limit})...")
    start_time = asyncio.get_event_loop().time()
    metric_labels = {"client": "unusual_whales", "endpoint": "unusual_options"}

    try:
        # Use timeout context for the API call within the retry attempt
        async with timeout_context(30.0): # Allow more time for potentially complex queries
            # Assuming the client method exists and takes these parameters
            data = await unusual_whales_client.get_unusual_options(
                min_volume=min_volume,
                limit=limit
            )
            # Basic validation
            if data is None or not isinstance(data, list):
                 logger.warning(f"Received unexpected options data format: {type(data)}")
                 return [] # Return empty list on unexpected format

            latency = asyncio.get_event_loop().time() - start_time
            MARKET_DATA_LATENCY.labels(**metric_labels).observe(latency)
            logger.debug(f"Fetched {len(data)} unusual options activity records in {latency:.4f}s.")
            return data

    except asyncio.TimeoutError as e:
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="timeout").inc()
        logger.error("Timeout fetching unusual options activity after retries.")
        raise APITimeoutError("Timeout fetching unusual options activity") from e
    except APIError as e: # Catch specific API errors from the client
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="api_error").inc()
        logger.error(f"API error fetching unusual options activity: {e}")
        raise # Re-raise the specific APIError
    except Exception as e:
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="unknown").inc()
        logger.exception(f"Unexpected error fetching unusual options activity: {e}")
        raise APIError(f"Unexpected error fetching unusual options activity: {e}") from e


async def setup_market_data_websocket(
    polygon_ws_client: PolygonWebSocketClient,
    symbols: List[str]
) -> None:
    """
    Connects the Polygon WebSocket client and subscribes to market data for symbols.

    Note: Connection and subscription logic might be better handled within the
          WebSocketEnhancedStockSelection class or the client itself depending on design.
          This function assumes direct control is needed here.

    Args:
        polygon_ws_client: Initialized PolygonWebSocketClient instance.
        symbols: List of ticker symbols to subscribe to.

    Raises:
        APIError: If connection or subscription fails.
        ValueError: If polygon_ws_client is not provided.
    """
    if not polygon_ws_client:
        raise ValueError("PolygonWebSocketClient instance is required.")
    if not symbols:
        logger.warning("setup_market_data_websocket called with empty symbols list.")
        return

    metric_labels = {"client": "polygon", "endpoint": "websocket_setup"}
    logger.info(f"Setting up WebSocket connection for {len(symbols)} symbols...")

    try:
        # Connect (assuming connect is idempotent or handles reconnects)
        if not polygon_ws_client.is_connected(): # Check if already connected
             logger.debug("Connecting WebSocket client...")
             await polygon_ws_client.connect()
             logger.info("WebSocket client connected.")
        else:
             logger.debug("WebSocket client already connected.")

        # Subscribe
        logger.debug(f"Subscribing to symbols: {symbols}")
        await polygon_ws_client.subscribe(symbols) # Assuming subscribe handles batching/errors

        logger.info(f"WebSocket setup complete for {len(symbols)} symbols.")

    except APIError as e: # Catch specific errors from connect/subscribe
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="api_error").inc()
        logger.error(f"API error during WebSocket setup: {e}")
        raise # Re-raise
    except Exception as e:
        # Catch unexpected errors during setup
        MARKET_DATA_ERRORS.labels(**metric_labels, error_type="unknown").inc()
        logger.exception(f"Unexpected error during WebSocket setup: {e}")
        # Wrap in APIError for consistency?
        raise APIError(f"Unexpected error setting up market data WebSocket: {e}") from e
