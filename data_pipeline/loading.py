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
import random # Import random module for jitter
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytz

from utils.logging_config import get_logger
from utils.metrics_registry import API_REQUEST_COUNT, API_ERROR_COUNT
from utils.config import Config # Import Config
from utils.exceptions import CacheError, SerializationError, APIError, APIConnectionError, APITimeoutError, RedisError # Import custom exceptions (added RedisError)

# Import Redis client wrapper type hint
if TYPE_CHECKING:
    from utils.redis_helpers import RedisClient
logger = get_logger("data_pipeline.loading")

# DEFAULT_TICKERS constant removed

async def load_from_cache(cache_key: str, redis_client: 'RedisClient', config: Config) -> Optional[Any]: # Make async
    """Load data from Redis cache using the RedisClient wrapper.

    Args:
        cache_key: Key to retrieve from cache.
        redis_client: Shared RedisClient instance.
        config: Centralized configuration object (used for potential future settings).

    Returns:
        Cached data or None if not found or error occurred.
    """
    # expiry is handled by Redis TTL, not checked here
    if not redis_client:
        return None

    try:
        # Use the get method from RedisClient wrapper
        cached_data_bytes = await redis_client.get(cache_key) # Assuming async get

        if cached_data_bytes is None:
            logger.debug(f"Cache miss for key: {cache_key}")
            return None

        # Deserialize data (handle potential errors)
        try:
            data = pickle.loads(cached_data_bytes)
            logger.debug(f"Cache hit for key: {cache_key}")
            return data
        except (pickle.UnpicklingError, TypeError, EOFError) as deser_err:
            logger.error(f"Deserialization error loading from cache {cache_key}: {deser_err}")
            # Optionally delete the corrupted key
            # await redis_client.delete(cache_key)
            raise SerializationError(f"Failed to deserialize cache data for {cache_key}") from deser_err

    except RedisError as redis_err: # Catch specific Redis errors from the wrapper
        logger.error(f"Redis error loading from cache {cache_key}: {redis_err}")
        raise CacheError(f"Redis error during cache load for {cache_key}") from redis_err
    except Exception as e: # Fallback for unexpected errors
        logger.exception(f"Unexpected error loading from cache {cache_key}: {e}")
        raise CacheError(f"Unexpected error during cache load for {cache_key}") from e

async def save_to_cache(cache_key: str, data: Any, redis_client: 'RedisClient', config: Config, ttl: Optional[int] = None) -> bool: # Make async
    """Save data to Redis cache using the RedisClient wrapper.

    Args:
        cache_key: Key to store data under.
        data: Data to cache (must be pickleable).
        redis_client: Shared RedisClient instance.
        config: Centralized configuration object.
        ttl: Cache expiry time in seconds (overrides default if provided).

    Returns:
        True if successful, False otherwise.
    """
    # Get default TTL from config if not provided
    if ttl is None:
         ttl = config.get_int("DEFAULT_CACHE_TTL", 86400)
    if not redis_client:
        return False

    try:
        # Serialize data (handle potential errors)
        try:
             # Add timestamp for potential debugging/info, though not used for expiry
             # if isinstance(data, dict) and "timestamp" not in data:
             #     data["_cache_saved_at"] = time.time()
             serialized = pickle.dumps(data)
        except (pickle.PicklingError, TypeError) as ser_err:
             logger.error(f"Serialization error saving to cache {cache_key}: {ser_err}")
             raise SerializationError(f"Failed to serialize data for cache key {cache_key}") from ser_err

        # Save to cache using RedisClient wrapper's set method
        await redis_client.set(cache_key, serialized, ex=ttl) # Assuming async set
        logger.debug(f"Saved data to cache key: {cache_key} with TTL: {ttl}s")
        return True

    except RedisError as redis_err: # Catch specific Redis errors from the wrapper
        logger.error(f"Redis error saving to cache {cache_key}: {redis_err}")
        raise CacheError(f"Redis error during cache save for {cache_key}") from redis_err
    except Exception as e: # Fallback for unexpected errors
        logger.exception(f"Unexpected error saving to cache {cache_key}: {e}")
        raise CacheError(f"Unexpected error during cache save for {cache_key}") from e

async def load_from_polygon_rest(
    polygon_client: Any, # Keep Any for now, ideally PolygonRESTClient base/protocol
    endpoint: str,
    params: dict,
    config: Config # Pass config object
) -> Optional[Union[dict, pd.DataFrame]]:
    """Load data from Polygon REST API with retry logic using settings from config.

    Args:
        polygon_client: Polygon REST API client instance.
        endpoint: API endpoint to call (relative path).
        params: Parameters for the API call.
        config: Centralized configuration object.

    Returns:
        API response data (dict or DataFrame) or None if failed after retries.

    Raises:
        APIError: If a non-retryable API error occurs or retries are exhausted.
    """
    # Load retry settings from config
    max_retries = config.get_int("MAX_RETRIES", 3)
    backoff_factor = config.get_float("RETRY_BACKOFF_FACTOR", 0.5)
    connection_timeout = config.get_int("CONNECTION_TIMEOUT", 15) # Get timeout from config
    attempt = 0
    while attempt < max_retries:
        try:
            # Record API request
            API_REQUEST_COUNT.labels(api="polygon", endpoint=endpoint).inc()

            # Make API call using the client's method (assuming .get)
            # Pass timeout from config if the client supports it
            response = await polygon_client.get(endpoint, params=params, timeout=connection_timeout)

            # Handle different status codes more specifically
            if response.status_code == 200:
                # Parse response
                try:
                     data = response.json()
                     # Convert 'results' to DataFrame if present
                     if "results" in data and isinstance(data["results"], list):
                          return pd.DataFrame(data["results"])
                     return data # Return raw dict otherwise
                except json.JSONDecodeError as json_err:
                     logger.error(f"JSON decode error for Polygon {endpoint}: {json_err}. Response text: {response.text[:500]}")
                     raise APIError(f"Invalid JSON response from Polygon {endpoint}") from json_err

            # --- Handle specific error codes ---
            elif response.status_code == 401: # Unauthorized
                 msg = f"Polygon API Unauthorized (Check API Key): {response.text[:200]}"
                 logger.error(msg)
                 API_ERROR_COUNT.labels(api="polygon", endpoint=endpoint, error_type="401").inc()
                 raise APIError(msg, code="401") # Non-retryable
            elif response.status_code == 404: # Not Found
                 msg = f"Polygon API Not Found for {endpoint} with params {params}: {response.text[:200]}"
                 logger.warning(msg) # Warning, might not be an error (e.g., no data for ticker)
                 API_ERROR_COUNT.labels(api="polygon", endpoint=endpoint, error_type="404").inc()
                 return None # Treat as no data found, don't raise APIError
            elif response.status_code == 429: # Rate Limit
                 msg = f"Polygon API Rate Limit Exceeded: {response.text[:200]}"
                 logger.warning(msg)
                 API_ERROR_COUNT.labels(api="polygon", endpoint=endpoint, error_type="429").inc()
                 # This IS retryable, let the retry logic handle it
                 raise APIConnectionError(msg, code="429") # Raise a retryable error type
            elif response.status_code >= 500: # Server Errors (Retryable)
                 msg = f"Polygon API Server Error ({response.status_code}): {response.text[:200]}"
                 logger.warning(msg)
                 API_ERROR_COUNT.labels(api="polygon", endpoint=endpoint, error_type=str(response.status_code)).inc()
                 raise APIConnectionError(msg, code=str(response.status_code)) # Raise a retryable error type
            else: # Other client errors (4xx) - likely non-retryable
                 msg = f"Polygon API Client Error ({response.status_code}): {response.text[:200]}"
                 logger.error(msg)
                 API_ERROR_COUNT.labels(api="polygon", endpoint=endpoint, error_type=str(response.status_code)).inc()
                 raise APIError(msg, code=str(response.status_code)) # Non-retryable
        # Catch specific potentially retryable errors (e.g., connection, timeout)
        # Assumes the client library might raise these or wraps them
        except (APIConnectionError, APITimeoutError, asyncio.TimeoutError) as retry_err: # Add specific client errors if known
            attempt += 1
            error_type_label = type(retry_err).__name__
            # Use code if available (e.g., "429", "503")
            if hasattr(retry_err, 'code') and retry_err.code:
                 error_type_label = f"{error_type_label}_{retry_err.code}"

            API_ERROR_COUNT.labels(api="polygon", endpoint=endpoint, error_type=error_type_label).inc()
            logger.warning(f"Retryable error on attempt {attempt}/{max_retries} for Polygon {endpoint}: {retry_err}")

            if attempt >= max_retries:
                logger.error(f"Failed to load from Polygon REST API after {max_retries} attempts: {retry_err}")
                raise APIError(f"Polygon API failed after {max_retries} retries: {retry_err}") from retry_err

            # Exponential backoff with jitter
            wait_time = min(backoff_factor * (2 ** (attempt - 1)), 10) + random.uniform(0, 0.1)
            logger.info(f"Retrying Polygon {endpoint} in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        except APIError as non_retry_api_err: # Catch non-retryable API errors raised above
             logger.error(f"Non-retryable API error for Polygon {endpoint}: {non_retry_api_err}")
             raise # Re-raise immediately
        except Exception as e: # Catch unexpected errors
            attempt += 1
            API_ERROR_COUNT.labels(api="polygon", endpoint=endpoint, error_type=type(e).__name__).inc()
            logger.exception(f"Unexpected error on attempt {attempt}/{max_retries} for Polygon {endpoint}: {e}")
            if attempt >= max_retries:
                 logger.error(f"Failed unexpectedly loading from Polygon REST API after {max_retries} attempts: {e}")
                 raise APIError(f"Polygon API failed unexpectedly after {max_retries} retries") from e
            # Exponential backoff for unexpected errors too? Or fail faster?
            wait_time = min(backoff_factor * (2 ** (attempt - 1)), 10) + random.uniform(0, 0.1)
            await asyncio.sleep(wait_time)

    # If loop finishes without returning/raising, something went wrong
    logger.error(f"Polygon REST API load loop completed without success for {endpoint}")
    return None # Should not be reached if max_retries > 0

# Removed load_from_polygon_ws - WebSocket interaction is typically stream-based
# and handled by the dedicated client (e.g., in trading_engine), not simple request-response.
# If a specific request-response pattern is needed, it should be implemented carefully
# based on the WebSocket client's capabilities.

async def load_from_unusual_whales(
    unusual_whales_client: Any, # Keep Any for now, ideally UW client base/protocol
    endpoint: str,
    params: dict,
    config: Config # Pass config object
) -> Optional[Union[dict, pd.DataFrame]]:
    """Load data from Unusual Whales API with retry logic using settings from config.

    Args:
        unusual_whales_client: Unusual Whales API client instance.
        endpoint: API endpoint to call.
        params: Parameters for the API call.
        config: Centralized configuration object.

    Returns:
        API response data (dict or DataFrame) or None if failed after retries.

    Raises:
        APIError: If a non-retryable API error occurs or retries are exhausted.
    """
    # Load retry settings from config
    max_retries = config.get_int("MAX_RETRIES", 3)
    backoff_factor = config.get_float("RETRY_BACKOFF_FACTOR", 0.5)
    connection_timeout = config.get_int("CONNECTION_TIMEOUT", 15)

    attempt = 0
    while attempt < max_retries:
        try:
            API_REQUEST_COUNT.labels(api="unusual_whales", endpoint=endpoint).inc()

            # Make API call
            response = await unusual_whales_client.get(endpoint, params=params, timeout=connection_timeout)

            if response.status_code == 200:
                 try:
                      data = response.json()
                      # UW API often wraps lists in a 'data' key
                      if "data" in data and isinstance(data["data"], list):
                           return pd.DataFrame(data["data"])
                      return data # Return raw dict otherwise
                 except json.JSONDecodeError as json_err:
                      logger.error(f"JSON decode error for UW {endpoint}: {json_err}. Response text: {response.text[:500]}")
                      raise APIError(f"Invalid JSON response from UW {endpoint}") from json_err

            # --- Handle specific error codes ---
            elif response.status_code == 401: # Unauthorized
                 msg = f"Unusual Whales API Unauthorized (Check API Key): {response.text[:200]}"
                 logger.error(msg)
                 API_ERROR_COUNT.labels(api="unusual_whales", endpoint=endpoint, error_type="401").inc()
                 raise APIError(msg, code="401") # Non-retryable
            elif response.status_code == 404: # Not Found
                 msg = f"Unusual Whales API Not Found for {endpoint} with params {params}: {response.text[:200]}"
                 logger.warning(msg)
                 API_ERROR_COUNT.labels(api="unusual_whales", endpoint=endpoint, error_type="404").inc()
                 return None # Treat as no data found
            elif response.status_code == 429: # Rate Limit
                 msg = f"Unusual Whales API Rate Limit Exceeded: {response.text[:200]}"
                 logger.warning(msg)
                 API_ERROR_COUNT.labels(api="unusual_whales", endpoint=endpoint, error_type="429").inc()
                 raise APIConnectionError(msg, code="429") # Retryable
            elif response.status_code >= 500: # Server Errors (Retryable)
                 msg = f"Unusual Whales API Server Error ({response.status_code}): {response.text[:200]}"
                 logger.warning(msg)
                 API_ERROR_COUNT.labels(api="unusual_whales", endpoint=endpoint, error_type=str(response.status_code)).inc()
                 raise APIConnectionError(msg, code=str(response.status_code)) # Retryable
            else: # Other client errors (4xx) - likely non-retryable
                 msg = f"Unusual Whales API Client Error ({response.status_code}): {response.text[:200]}"
                 logger.error(msg)
                 API_ERROR_COUNT.labels(api="unusual_whales", endpoint=endpoint, error_type=str(response.status_code)).inc()
                 raise APIError(msg, code=str(response.status_code)) # Non-retryable

        except (APIConnectionError, APITimeoutError, asyncio.TimeoutError) as retry_err:
            attempt += 1
            error_type_label = type(retry_err).__name__
            if hasattr(retry_err, 'code') and retry_err.code: error_type_label = f"{error_type_label}_{retry_err.code}"
            API_ERROR_COUNT.labels(api="unusual_whales", endpoint=endpoint, error_type=error_type_label).inc()
            logger.warning(f"Retryable error on attempt {attempt}/{max_retries} for UW {endpoint}: {retry_err}")
            if attempt >= max_retries:
                logger.error(f"Failed to load from UW API after {max_retries} attempts: {retry_err}")
                raise APIError(f"UW API failed after {max_retries} retries: {retry_err}") from retry_err
            wait_time = min(backoff_factor * (2 ** (attempt - 1)), 10) + random.uniform(0, 0.1)
            logger.info(f"Retrying UW {endpoint} in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        except APIError as non_retry_api_err:
             logger.error(f"Non-retryable API error for UW {endpoint}: {non_retry_api_err}")
             raise
        except Exception as e:
            attempt += 1
            API_ERROR_COUNT.labels(api="unusual_whales", endpoint=endpoint, error_type=type(e).__name__).inc()
            logger.exception(f"Unexpected error on attempt {attempt}/{max_retries} for UW {endpoint}: {e}")
            if attempt >= max_retries:
                 logger.error(f"Failed unexpectedly loading from UW API after {max_retries} attempts: {e}")
                 raise APIError(f"UW API failed unexpectedly after {max_retries} retries") from e
            wait_time = min(backoff_factor * (2 ** (attempt - 1)), 10) + random.uniform(0, 0.1)
            await asyncio.sleep(wait_time)

    logger.error(f"Unusual Whales API load loop completed without success for {endpoint}")
    return None

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
        logger.error(f"Error loading from file {file_path}: {e}", exc_info=True)
        # Consider raising a DataError here?
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
        logger.error(f"Error checking same day for '{timestamp_str}': {e}") # Use error level
        return False
