#!/usr/bin/env python3
"""
Data Loading Module

Contains functionality for loading data from various sources including:
- Polygon.io REST API
- Polygon.io WebSocket
- Unusual Whales API
- Redis cache
- Local files
"""

import asyncio
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytz

from utils.logging_config import get_logger
from utils.metrics_registry import API_REQUEST_COUNT, API_ERROR_COUNT

logger = get_logger("data_pipeline.loading")

# DEFAULT_TICKERS constant removed

def load_from_cache(cache_key: str, redis_client: Any, expiry: int = 86400) -> Optional[Any]:
    """Load data from Redis cache

    Args:
        cache_key: Key to retrieve from cache
        redis_client: Redis client instance
        expiry: Cache expiry time in seconds

    Returns:
        Cached data or None if not found/expired
    """
    if not redis_client:
        return None

    try:
        # Check if key exists
        if not redis_client.exists(cache_key):
            return None

        # Get data from cache
        cached_data = redis_client.get(cache_key)
        if not cached_data:
            return None

        # Deserialize data
        data = pickle.loads(cached_data)

        # Check if data is expired
        if isinstance(data, dict) and "timestamp" in data:
            if time.time() - data["timestamp"] > expiry:
                return None

        return data
    except Exception as e:
        logger.error(f"Error loading from cache {cache_key}: {e}")
        return None

def save_to_cache(cache_key: str, data: Any, redis_client: Any, expiry: int = 86400) -> bool:
    """Save data to Redis cache

    Args:
        cache_key: Key to store data under
        data: Data to cache (must be serializable)
        redis_client: Redis client instance
        expiry: Cache expiry time in seconds

    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False

    try:
        # Add timestamp if not already present
        if isinstance(data, dict) and "timestamp" not in data:
            data["timestamp"] = time.time()

        # Serialize data
        serialized = pickle.dumps(data)

        # Save to cache
        redis_client.set(cache_key, serialized, ex=expiry)
        return True
    except Exception as e:
        logger.error(f"Error saving to cache {cache_key}: {e}")
        return False

async def load_from_polygon_rest(
    polygon_client: Any,
    endpoint: str,
    params: dict,
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> Optional[Union[dict, pd.DataFrame]]:
    """Load data from Polygon REST API with retry logic

    Args:
        polygon_client: Polygon REST API client
        endpoint: API endpoint to call
        params: Parameters for the API call
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff

    Returns:
        API response data or None if failed
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # Record API request
            API_REQUEST_COUNT.labels(api="polygon", endpoint=endpoint).inc()

            # Make API call
            response = await polygon_client.get(endpoint, params=params)

            if response.status_code != 200:
                raise ValueError(f"Polygon API error: {response.text}")

            # Parse response
            data = response.json()

            if "results" in data:
                return pd.DataFrame(data["results"])
            return data
        except Exception as e:
            attempt += 1
            API_ERROR_COUNT.labels(
                api="polygon",
                endpoint=endpoint,
                error_type=type(e).__name__
            ).inc()

            if attempt == max_retries:
                logger.error(f"Failed to load from Polygon REST API after {max_retries} attempts: {e}")
                return None

            # Exponential backoff
            wait_time = min(backoff_factor * (2 ** (attempt - 1)), 10)
            await asyncio.sleep(wait_time)

async def load_from_polygon_ws(
    polygon_ws: Any,
    message: dict,
    timeout: float = 10.0
) -> Optional[Union[dict, pd.DataFrame]]:
    """Load data from Polygon WebSocket

    Args:
        polygon_ws: Polygon WebSocket client
        message: Message to send to WebSocket
        timeout: Timeout in seconds

    Returns:
        WebSocket response data or None if failed
    """
    try:
        # Record API request
        API_REQUEST_COUNT.labels(api="polygon", endpoint="ws").inc()

        # Send message and wait for response
        response = await polygon_ws.send_and_wait(message, timeout=timeout)

        if isinstance(response, dict):
            if "results" in response:
                return pd.DataFrame(response["results"])
            return response
        return None
    except Exception as e:
        API_ERROR_COUNT.labels(
            api="polygon",
            endpoint="ws",
            error_type=type(e).__name__
        ).inc()
        logger.error(f"Error loading from Polygon WebSocket: {e}")
        return None

async def load_from_unusual_whales(
    unusual_whales_client: Any,
    endpoint: str,
    params: dict,
    max_retries: int = 3,
    backoff_factor: float = 1.0
) -> Optional[Union[dict, pd.DataFrame]]:
    """Load data from Unusual Whales API with retry logic

    Args:
        unusual_whales_client: Unusual Whales API client
        endpoint: API endpoint to call
        params: Parameters for the API call
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff

    Returns:
        API response data or None if failed
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # Record API request
            API_REQUEST_COUNT.labels(api="unusual_whales", endpoint=endpoint).inc()

            # Make API call
            response = await unusual_whales_client.get(endpoint, params=params)

            if response.status_code != 200:
                raise ValueError(f"Unusual Whales API error: {response.text}")

            # Parse response
            data = response.json()

            if "data" in data:
                return pd.DataFrame(data["data"])
            return data
        except Exception as e:
            attempt += 1
            API_ERROR_COUNT.labels(
                api="unusual_whales",
                endpoint=endpoint,
                error_type=type(e).__name__
            ).inc()

            if attempt == max_retries:
                logger.error(f"Failed to load from Unusual Whales API after {max_retries} attempts: {e}")
                return None

            # Exponential backoff
            wait_time = min(backoff_factor * (2 ** (attempt - 1)), 10)
            await asyncio.sleep(wait_time)

def load_from_file(file_path: str, file_type: str = "csv") -> Optional[Union[dict, pd.DataFrame]]:
    """Load data from local file

    Args:
        file_path: Path to file
        file_type: Type of file (csv, json, parquet, pickle)

    Returns:
        Loaded data or None if failed
    """
    try:
        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "json":
            return pd.read_json(file_path)
        elif file_type == "parquet":
            return pd.read_parquet(file_path)
        elif file_type == "pickle":
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error loading from file {file_path}: {e}")
        return None

def same_day(timestamp_str: str, timezone: str = "US/Eastern") -> bool:
    """
    Check if timestamp is from the same day (in specified timezone)

    Args:
        timestamp_str: Timestamp string
        timezone: Timezone to use for comparison

    Returns:
        Boolean indicating if timestamp is from today
    """
    try:
        # Parse timestamp
        timestamp = datetime.fromisoformat(timestamp_str)

        # Get current date in specified timezone
        now = datetime.now(pytz.timezone(timezone))

        # Check if same day
        return timestamp.date() == now.date()
    except Exception as e:
        logger.exception(f"Error checking same day: {e}")
        return False
