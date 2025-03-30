"""
Polygon.io REST API Client
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aiohttp
from aiohttp import ClientSession, ClientTimeout

# Import relevant base components and config
from .base import (
    logger, # Use the logger from base
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY,
    API_ERROR_COUNT,
    API_CACHE_HIT_COUNT,
    API_CACHE_MISS_COUNT,
    API_RATE_LIMIT_REMAINING,
    RedisCache,
    AsyncConnectionPool,
    # GPUAccelerator removed
    POLYGON_API_KEY,
    POLYGON_CACHE_TTL,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    CONNECTION_TIMEOUT,
    MAX_POOL_SIZE,
    # GPU_PROCESSING_TIME removed
)
# Import centralized GPU utilities if needed for future direct use (unlikely here)
# from utils.gpu_utils import is_gpu_available, process_array, clear_gpu_memory
from utils.metrics_registry import PROMETHEUS_AVAILABLE

class PolygonRESTClient:
    """Polygon.io REST API client with caching and connection pooling."""

    def __init__(
        self,
        api_key: str = POLYGON_API_KEY,
        cache: Optional[RedisCache] = None,
        # use_gpu parameter removed
        max_retries: int = MAX_RETRIES,
        backoff_factor: float = RETRY_BACKOFF_FACTOR,
        timeout: int = CONNECTION_TIMEOUT,
        max_pool_size: int = MAX_POOL_SIZE
    ) -> None:
        """Initialize the Polygon REST client."""
        if not api_key:
             logger.error("Polygon API Key is required for PolygonRESTClient.")
             raise ValueError("Polygon API Key not provided.")
        self.api_key = api_key
        self.base_url = "https://api.polygon.io" # Base URL updated
        self.cache = cache or RedisCache(prefix="polygon", ttl=POLYGON_CACHE_TTL)
        # self.gpu_accelerator removed
        self.connection_pool = AsyncConnectionPool(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_pool_size=max_pool_size,
            timeout=timeout
        )
        self.running = True # Added running flag
        self.scheduled_tasks: Dict[str, asyncio.Task] = {} # Added task tracking

    async def get_aggregates(
        self,
        symbol: str, # Renamed from ticker for consistency
        multiplier: int,
        timespan: str,
        from_date: str, # Renamed for consistency
        to_date: str,   # Renamed for consistency
        limit: int = 50000,
        adjusted: bool = True,
        sort: str = "asc",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]: # Return DataFrame directly
        """Get aggregate bars for a stock ticker."""
        # Use more descriptive key parts
        cache_key = ["aggregates", symbol, multiplier, timespan, from_date, to_date, str(adjusted), sort, limit]
        metric_labels = {"client": "polygon", "endpoint": "get_aggregates", "method": "GET"}

        # Try cache first
        if use_cache:
            cached_df = self.cache.get_dataframe(cache_key) # Use get_dataframe
            if cached_df is not None:
                API_CACHE_HIT_COUNT.labels(client="polygon", cache_type="aggregates").inc()
                return cached_df

        API_CACHE_MISS_COUNT.labels(client="polygon", cache_type="aggregates").inc()

        # Construct endpoint and params
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "apiKey": self.api_key,
            "limit": limit,
            "adjusted": str(adjusted).lower(),
            "sort": sort
        }

        try:
            start_time = time.time()
            data = await self._make_request(endpoint, params)

            if data and "results" in data and data["results"]: # Check if results list is not empty
                results_df = pd.DataFrame(data["results"])
                # Basic cleaning (timestamp conversion) can happen here or later
                if 't' in results_df.columns:
                     results_df['timestamp'] = pd.to_datetime(results_df['t'], unit='ms', utc=True)
                     results_df = results_df.drop(columns=['t'])

                # GPU processing block removed

                # Store DataFrame in cache
                if use_cache:
                    self.cache.store_dataframe(cache_key, results_df) # Use store_dataframe

                # Update metrics
                if PROMETHEUS_AVAILABLE:
                    API_REQUEST_LATENCY.labels(**metric_labels).observe(time.time() - start_time)
                    API_REQUEST_COUNT.labels(**metric_labels).inc()

                return results_df # Return DataFrame
            elif data and "results" in data and not data["results"]:
                 logger.info(f"No aggregate results found for {symbol} with given parameters.")
                 # Cache empty result
                 if use_cache: self.cache.store_dataframe(cache_key, pd.DataFrame(), ttl=300) # Cache empty for 5 mins
                 return pd.DataFrame() # Return empty DataFrame
            else:
                 logger.error(f"Invalid response structure for aggregates {symbol}: {data}")
                 if PROMETHEUS_AVAILABLE: API_ERROR_COUNT.labels(**metric_labels, error_type="InvalidResponse").inc()
                 return None # Return None on invalid structure

        except Exception as e:
            logger.exception(f"Error getting aggregates for {symbol}: {e}")
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(**metric_labels, error_type=type(e).__name__).inc()
            return None

    # _process_aggregates_gpu method removed

    async def get_ticker_details(
        self,
        ticker: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get details for a specific ticker with caching"""
        cache_key = ["ticker_details", ticker]
        metric_labels = {"client": "polygon", "endpoint": "get_ticker_details", "method": "GET"}

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                API_CACHE_HIT_COUNT.labels(client="polygon", cache_type="ticker_details").inc()
                return cached

        API_CACHE_MISS_COUNT.labels(client="polygon", cache_type="ticker_details").inc()

        endpoint = f"/v3/reference/tickers/{ticker}" # Use v3 endpoint
        params = {"apiKey": self.api_key}

        try:
            start_time = time.time()
            data = await self._make_request(endpoint, params)

            # v3 response structure is different, often has a 'results' key
            if data and "results" in data:
                 details = data["results"]
                 if use_cache:
                      self.cache.set(cache_key, details) # Cache the results dict
                 if PROMETHEUS_AVAILABLE:
                      API_REQUEST_LATENCY.labels(**metric_labels).observe(time.time() - start_time)
                      API_REQUEST_COUNT.labels(**metric_labels).inc()
                 return details # Return the inner results dict
            else:
                 logger.warning(f"No 'results' found in ticker details response for {ticker}: {data}")
                 if PROMETHEUS_AVAILABLE: API_ERROR_COUNT.labels(**metric_labels, error_type="InvalidResponse").inc()
                 return None

        except Exception as e:
            logger.exception(f"Error getting details for {ticker}: {e}")
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(**metric_labels, error_type=type(e).__name__).inc()
            return None

    async def list_tickers(
        self,
        market: str = 'stocks',
        active: bool = True,
        limit: int = 1000, # Default limit per page for Polygon
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """List tickers with pagination handling."""
        # Note: Polygon's free tier might limit access or results significantly.
        cache_key = ["list_tickers", market, str(active), limit] # Limit matters for pagination
        metric_labels = {"client": "polygon", "endpoint": "list_tickers", "method": "GET"}

        if use_cache:
            cached_df = self.cache.get_dataframe(cache_key)
            if cached_df is not None:
                API_CACHE_HIT_COUNT.labels(client="polygon", cache_type="list_tickers").inc()
                return cached_df

        API_CACHE_MISS_COUNT.labels(client="polygon", cache_type="list_tickers").inc()

        endpoint = "/v3/reference/tickers"
        params = {
            "apiKey": self.api_key,
            "market": market,
            "active": str(active).lower(),
            "limit": limit, # Max 1000 per page
        }
        all_tickers = []
        next_url = None

        try:
            start_time = time.time()
            current_url = f"{self.base_url}{endpoint}"

            while True:
                 logger.debug(f"Fetching tickers page: {current_url}")
                 # Use session directly for pagination control
                 if self.connection_pool.session is None or self.connection_pool.session.closed:
                      await self.connection_pool.initialize()

                 async with self.connection_pool.session.get(current_url, params=params if not next_url else None) as response:
                      if response.status != 200:
                           error_text = await response.text()
                           logger.error(f"Error fetching tickers page: HTTP {response.status} - {error_text}")
                           if PROMETHEUS_AVAILABLE: API_ERROR_COUNT.labels(**metric_labels, error_type=f"HTTP{response.status}").inc()
                           # Decide whether to return partial results or None
                           break # Stop pagination on error

                      data = await response.json()
                      if data and "results" in data:
                           all_tickers.extend(data["results"])
                           next_url = data.get("next_url")
                           if next_url:
                                current_url = f"{next_url}&apiKey={self.api_key}" # Append API key to next_url
                                params = None # Clear params as they are in next_url
                           else:
                                break # No more pages
                      else:
                           logger.warning(f"Invalid response structure on tickers page: {data}")
                           break # Stop pagination

                 # Avoid overwhelming the API
                 await asyncio.sleep(0.2) # Small delay between pages

            if not all_tickers:
                 logger.warning("No tickers found.")
                 if use_cache: self.cache.store_dataframe(cache_key, pd.DataFrame(), ttl=3600) # Cache empty for 1h
                 return pd.DataFrame()

            tickers_df = pd.DataFrame(all_tickers)
            if use_cache:
                self.cache.store_dataframe(cache_key, tickers_df) # Cache the full result

            if PROMETHEUS_AVAILABLE:
                API_REQUEST_LATENCY.labels(**metric_labels).observe(time.time() - start_time)
                API_REQUEST_COUNT.labels(**metric_labels).inc() # Count as one logical request

            return tickers_df

        except Exception as e:
            logger.exception(f"Error listing tickers: {e}")
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(**metric_labels, error_type=type(e).__name__).inc()
            return None # Return None on failure


    async def get_last_quote(
        self,
        ticker: str,
        use_cache: bool = False # Quotes are volatile, default to no cache
    ) -> Optional[Dict[str, Any]]:
        """Get the last quote for a ticker."""
        cache_key = ["last_quote", ticker]
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                API_CACHE_HIT_COUNT.labels(client="polygon", cache_type="last_quote").inc()
                return cached
            API_CACHE_MISS_COUNT.labels(client="polygon", cache_type="last_quote").inc()

        endpoint = f"/v2/last/stocks/{ticker}" # Use v2 endpoint
        params = {"apiKey": self.api_key}
        metric_labels = {"client": "polygon", "endpoint": "get_last_quote", "method": "GET"}

        try:
            start_time = time.time()
            data = await self._make_request(endpoint, params)

            if data and "last" in data:
                 last_quote_data = data.get("last")
                 if use_cache:
                      self.cache.set(cache_key, last_quote_data, ttl=60) # Short TTL if cached
                 if PROMETHEUS_AVAILABLE:
                      API_REQUEST_LATENCY.labels(**metric_labels).observe(time.time() - start_time)
                      API_REQUEST_COUNT.labels(**metric_labels).inc()
                 return last_quote_data # Return the 'last' quote object
            else:
                 logger.warning(f"No 'last' quote data found for {ticker} in response: {data}")
                 if PROMETHEUS_AVAILABLE:
                      API_ERROR_COUNT.labels(**metric_labels, error_type="missing_data").inc()
                 return None

        except Exception as e:
            logger.exception(f"Error getting last quote for {ticker}: {e}")
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(**metric_labels, error_type=type(e).__name__).inc()
            return None

    # --- Placeholder methods for Trading ---
    # Removed as they belong in a brokerage client

    # --- End Placeholder methods ---


    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make API request using the connection pool."""
        url = f"{self.base_url}{endpoint}"

        try:
            # Use the connection pool's get method which includes retries
            response_data = await self.connection_pool.get(url, params=params, headers=headers)

            # Check if the pool returned an error structure
            if isinstance(response_data, dict) and response_data.get("status") == "ERROR":
                error_msg = response_data.get("error", "Unknown connection pool error")
                logger.error(f"Request failed via connection pool for {endpoint}: {error_msg}")
                # Increment error count based on the underlying error if possible
                # This might require the pool to return more error details
                if PROMETHEUS_AVAILABLE:
                     API_ERROR_COUNT.labels(
                          client="polygon", endpoint=endpoint.split('/')[1], method="GET", error_type="PoolError"
                     ).inc()
                return None # Return None on pool error

            # If successful, response_data should be the JSON dict
            return response_data

        except Exception as e:
            # Catch any unexpected errors during the pool interaction
            logger.exception(f"Unexpected error making request for {endpoint}: {e}")
            if PROMETHEUS_AVAILABLE:
                 API_ERROR_COUNT.labels(
                      client="polygon", endpoint=endpoint.split('/')[1], method="GET", error_type=type(e).__name__
                 ).inc()
            return None # Return None on unexpected error

    async def close(self) -> None:
        """Clean up resources."""
        self.running = False

        # Cancel scheduled tasks
        for task in self.scheduled_tasks.values():
            if not task.done():
                task.cancel()

        # Close connection pool
        await self.connection_pool.close()

        # GPU memory clearing removed - handled globally
        # self.gpu_accelerator.clear_memory()

        logger.info("Polygon REST client closed")
