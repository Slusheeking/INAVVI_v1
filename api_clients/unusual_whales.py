"""
Enhanced Unusual Whales API Client
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aiohttp
import numpy as np
import pandas as pd
from aiohttp import ClientSession, ClientTimeout
from utils.metrics_registry import PROMETHEUS_AVAILABLE

# Import relevant base components and config
from .base import (
    logger, # Use the logger from base
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY,
    API_ERROR_COUNT,
    API_CACHE_HIT_COUNT,
    API_CACHE_MISS_COUNT,
    RedisCache,
    AsyncConnectionPool,
    # GPUAccelerator removed
    UNUSUAL_WHALES_API_KEY,
    UNUSUAL_WHALES_CACHE_TTL,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    CONNECTION_TIMEOUT,
    MAX_POOL_SIZE,
    # GPU_PROCESSING_TIME removed
)
# Import centralized GPU utilities if needed (unlikely here)
# from utils.gpu_utils import is_gpu_available, process_array, clear_gpu_memory

class UnusualWhalesClient:
    """Enhanced Unusual Whales API client with caching and connection pooling."""

    def __init__(
        self,
        api_key: str = UNUSUAL_WHALES_API_KEY,
        cache: Optional[RedisCache] = None,
        # use_gpu parameter removed
        max_retries: int = MAX_RETRIES,
        backoff_factor: float = RETRY_BACKOFF_FACTOR,
        timeout: int = CONNECTION_TIMEOUT,
        max_pool_size: int = MAX_POOL_SIZE
    ) -> None:
        """Initialize the Unusual Whales client."""
        if not api_key:
             logger.warning("Unusual Whales API Key not provided. Client will be disabled.")
             # Set api_key to None or empty string to disable requests
             self.api_key = None
        else:
             self.api_key = api_key

        self.base_url = "https://api.unusualwhales.com/v1" # Verify base URL
        self.cache = cache or RedisCache(prefix="unusual_whales", ttl=UNUSUAL_WHALES_CACHE_TTL)
        # self.gpu_accelerator removed
        self.connection_pool = AsyncConnectionPool(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_pool_size=max_pool_size,
            timeout=timeout
        )
        self.running = True # Added running flag

    async def get_flow_alerts(
        self,
        ticker: str,
        limit: int = 100,
        use_cache: bool = True
    ) -> Optional[List[Dict[str, Any]]]: # Return List of alerts
        """Get flow alerts."""
        if not self.api_key: return [] # Return empty list if disabled

        cache_key = ["flow_alerts", ticker, str(limit)]
        metric_labels = {"client": "unusual_whales", "endpoint": "get_flow_alerts", "method": "GET"}

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                API_CACHE_HIT_COUNT.labels(client="unusual_whales", cache_type="flow_alerts").inc()
                return cached if isinstance(cached, list) else []

        API_CACHE_MISS_COUNT.labels(client="unusual_whales", cache_type="flow_alerts").inc()

        endpoint = f"/stock/{ticker}/flow-alerts" # Verify endpoint
        params = { "limit": limit } # API key likely passed in headers
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            start_time = time.time()
            response_data = await self._make_request(endpoint, params, headers=headers)

            alerts_list: Optional[List[Dict[str, Any]]] = None
            if isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                 alerts_list = response_data['data']
            elif isinstance(response_data, list): # If API returns list directly
                 alerts_list = response_data
            else:
                 logger.warning(f"Received unexpected format for flow alerts {ticker}: {type(response_data)}")
                 if PROMETHEUS_AVAILABLE: API_ERROR_COUNT.labels(**metric_labels, error_type="invalid_format").inc()
                 return []

            if alerts_list:
                # GPU processing block removed

                if use_cache:
                    await self.cache.set(cache_key, alerts_list) # Cache the extracted list

                if PROMETHEUS_AVAILABLE:
                    API_REQUEST_LATENCY.labels(**metric_labels).observe(time.time() - start_time)
                    API_REQUEST_COUNT.labels(**metric_labels).inc()

            return alerts_list if alerts_list is not None else []

        except Exception as e:
            logger.exception(f"Error getting flow alerts for {ticker}: {e}")
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(**metric_labels, error_type=type(e).__name__).inc()
            return [] # Return empty list on error

    # _process_flow_alerts_gpu method removed

    async def get_historical_flow(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Optional[List[Dict[str, Any]]]: # Return List of flow data
        """Get historical flow data."""
        if not self.api_key: return []

        cache_key = ["historical_flow", ticker, start_date, end_date]
        metric_labels = {"client": "unusual_whales", "endpoint": "get_historical_flow", "method": "GET"}

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                API_CACHE_HIT_COUNT.labels(client="unusual_whales", cache_type="historical_flow").inc()
                return cached if isinstance(cached, list) else []

        API_CACHE_MISS_COUNT.labels(client="unusual_whales", cache_type="historical_flow").inc()

        endpoint = f"/stock/{ticker}/historical-flow" # Verify endpoint
        params = { "start_date": start_date, "end_date": end_date }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            start_time = time.time()
            response_data = await self._make_request(endpoint, params, headers=headers)

            flow_list: Optional[List[Dict[str, Any]]] = None
            if isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                 flow_list = response_data['data']
            elif isinstance(response_data, list):
                 flow_list = response_data
            else:
                 logger.warning(f"Received unexpected format for historical flow {ticker}: {type(response_data)}")
                 if PROMETHEUS_AVAILABLE: API_ERROR_COUNT.labels(**metric_labels, error_type="invalid_format").inc()
                 return []

            if flow_list:
                if use_cache:
                    await self.cache.set(cache_key, flow_list)

                if PROMETHEUS_AVAILABLE:
                    API_REQUEST_LATENCY.labels(**metric_labels).observe(time.time() - start_time)
                    API_REQUEST_COUNT.labels(**metric_labels).inc()

            return flow_list if flow_list is not None else []

        except Exception as e:
            logger.exception(f"Error getting historical flow for {ticker}: {e}")
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(**metric_labels, error_type=type(e).__name__).inc()
            return []

    async def get_unusual_options(
        self,
        min_volume: Optional[int] = None,
        min_premium: Optional[float] = None,
        limit: int = 100,
        use_cache: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """Get unusual options activity data."""
        if not self.api_key: return []

        cache_key_parts = ["unusual_options", str(limit)]
        if min_volume is not None: cache_key_parts.append(f"vol_{min_volume}")
        if min_premium is not None: cache_key_parts.append(f"prem_{min_premium}")
        cache_key = cache_key_parts
        metric_labels = {"client": "unusual_whales", "endpoint": "get_unusual_options", "method": "GET"}

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                API_CACHE_HIT_COUNT.labels(client="unusual_whales", cache_type="unusual_options").inc()
                return cached if isinstance(cached, list) else []

        API_CACHE_MISS_COUNT.labels(client="unusual_whales", cache_type="unusual_options").inc()

        endpoint = "/options/unusual-activity" # Verify endpoint
        params = { "limit": limit }
        if min_volume is not None: params["min_volume"] = min_volume
        if min_premium is not None: params["min_premium"] = min_premium
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            start_time = time.time()
            response_data = await self._make_request(endpoint, params, headers=headers)

            options_list: Optional[List[Dict[str, Any]]] = None
            if isinstance(response_data, dict) and 'data' in response_data and isinstance(response_data['data'], list):
                 options_list = response_data['data']
            elif isinstance(response_data, list):
                 options_list = response_data
            else:
                 logger.warning(f"Received unexpected format for unusual options: {type(response_data)}")
                 if PROMETHEUS_AVAILABLE: API_ERROR_COUNT.labels(**metric_labels, error_type="invalid_format").inc()
                 return []

            if options_list:
                # GPU processing block removed

                if use_cache:
                    await self.cache.set(cache_key, options_list)

                if PROMETHEUS_AVAILABLE:
                    API_REQUEST_LATENCY.labels(**metric_labels).observe(time.time() - start_time)
                    API_REQUEST_COUNT.labels(**metric_labels).inc()

            return options_list if options_list is not None else []

        except Exception as e:
            logger.exception(f"Error getting unusual options activity: {e}")
            if PROMETHEUS_AVAILABLE:
                API_ERROR_COUNT.labels(**metric_labels, error_type=type(e).__name__).inc()
            return []

    # Optional GPU processing placeholder removed

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[Union[Dict[str, Any], List[Any]]]: # Allow list response
        """Make API request using the connection pool."""
        if not self.api_key:
             logger.error("API key not configured for Unusual Whales client.")
             return None

        url = f"{self.base_url}{endpoint}"
        # Ensure headers include Authorization if not already present
        request_headers = headers or {}
        if "Authorization" not in request_headers:
             request_headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response_data = await self.connection_pool.get(url, params=params, headers=request_headers)

            if isinstance(response_data, dict) and response_data.get("status") == "ERROR":
                error_msg = response_data.get("error", "Unknown connection pool error")
                logger.error(f"Request failed via connection pool for {endpoint}: {error_msg}")
                if PROMETHEUS_AVAILABLE:
                     API_ERROR_COUNT.labels(
                          client="unusual_whales", endpoint=endpoint.split('/')[1], method="GET", error_type="PoolError"
                     ).inc()
                return None

            # Return the raw JSON response (could be dict or list)
            return response_data

        except Exception as e:
            logger.exception(f"Unexpected error making request for {endpoint}: {e}")
            if PROMETHEUS_AVAILABLE:
                 API_ERROR_COUNT.labels(
                      client="unusual_whales", endpoint=endpoint.split('/')[1], method="GET", error_type=type(e).__name__
                 ).inc()
            return None

    async def close(self) -> None:
        """Clean up resources."""
        self.running = False
        await self.connection_pool.close()
        # GPU memory clearing removed
        logger.info("Unusual Whales client closed")
