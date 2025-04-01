#!/usr/bin/env python3
"""
Base Data Pipeline Module

Contains the core DataPipeline class and initialization logic.
Relies on utils.gpu_utils for GPU management.
"""

import asyncio
import json
import os
import time
from typing import Any, Optional, Union, Dict, List # Added Dict, List
import numpy as np # Added numpy
import pandas as pd

# Import standardized utility modules
# Use the singleton gpu_manager and helper functions
from utils.gpu_utils import gpu_manager, get_device_info, is_gpu_available
from utils.logging_config import get_logger
from utils.config import Config # Import Config
from utils.exceptions import ConfigurationError, APIError, RedisError, DataError # Import custom exceptions
# Import functions from other pipeline modules
from .loading import load_from_cache, save_to_cache
from .processing import clean_market_data # Keep processing imports if needed by access methods

# Configure logging
logger = get_logger("data_pipeline.base") # More specific logger name

class DataPipeline:
    """
    Unified data pipeline for the trading system

    This class combines functionality from:
    - Data loading from APIs (via loading.py helpers)
    - Data preprocessing and caching
    - Feature engineering and technical indicators (via processing.py helpers)
    - ML data preparation
    - Market data analysis and filtering

    It provides both synchronous and asynchronous interfaces and leverages
    utils.gpu_utils for GPU acceleration when available.
    """

    def __init__(
        self,
        config: Config, # Require central Config object
        redis_client, # Require shared RedisClient
        # Clients are still injected for now, but could be initialized here based on config
        polygon_client=None,
        polygon_ws=None,
        unusual_whales_client=None,
        test_mode=False, # Keep test_mode flag
    ) -> None:
        """
        Initialize the data pipeline

        Args:
            config: Centralized configuration object.
            redis_client: Shared Redis client instance.
            polygon_client: Optional pre-initialized Polygon REST client.
            polygon_ws: Optional pre-initialized Polygon WebSocket client.
            unusual_whales_client: Optional pre-initialized Unusual Whales client.
            test_mode: Whether to use mock data for testing.
        """
        self.config = config
        self.logger = get_logger(__name__) # Use configured logger
        self.polygon = polygon_client
        self.polygon_ws = polygon_ws
        self.unusual_whales = unusual_whales_client
        self.redis = redis_client # Use injected shared client
        # Test mode for using synthetic data
        self.test_mode = test_mode

        # GPU status is now determined by the global gpu_manager
        self.use_gpu = is_gpu_available() # Check status from gpu_manager
        self.gpu_info = get_device_info() # Get detailed info if needed

        # Load settings from the central Config object
        # These keys should be defined in DEFAULT_CONFIG in utils/config.py
        self.cache_dir = self.config.get_path("CACHE_DIR", "./data/cache")
        self.data_dir = self.config.get_path("DATA_DIR", "./data")
        self.monitoring_dir = self.config.get_path("MONITORING_DIR", "./monitoring")
        # Use specific cache TTLs if defined, otherwise a general default
        self.polygon_cache_ttl = self.config.get_int("POLYGON_CACHE_TTL", 3600)
        self.uw_cache_ttl = self.config.get_int("UNUSUAL_WHALES_CACHE_TTL", 300)
        self.default_cache_ttl = self.config.get_int("DEFAULT_CACHE_TTL", 86400) # Fallback TTL

        # Load Redis keys from config (consistent with ml_engine)
        self.redis_notify_key = self.config.get("REDIS_KEY_NOTIFICATIONS", "frontend:notifications")
        self.redis_notify_limit = self.config.get_int("REDIS_LIMIT_NOTIFICATIONS", 100)
        self.redis_category_limit = self.config.get_int("REDIS_LIMIT_CATEGORY", 50)
        self.redis_status_key = self.config.get("REDIS_KEY_SYSTEM_STATUS", "frontend:system:status")

        # TODO: Consider loading rate_limit and retry_settings from config if needed by clients initialized here
        # self.rate_limit_config = self.config.get_dict("API_RATE_LIMITS", {"polygon": 5, "unusual_whales": 2})
        # self.retry_config = self.config.get_dict("API_RETRY_SETTINGS", {...})

        # Ensure directories exist
        self._ensure_directories() # Ensure directories based on loaded config paths

        # GPU initialization is handled globally by gpu_manager instance creation
        # No need to call self._initialize_gpu() here

        # Create event loop for async calls
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.logger.info("Data Pipeline initialized")

        # Send notification to frontend using info from gpu_manager
        # Send notification using configured keys
        gpu_details = get_device_info()
        self._send_frontend_notification(
            message="Data Pipeline initialized successfully",
            level="info",
            category="system_startup", # Use a specific category? "data_pipeline_startup"?
            details={
                "gpu_acceleration": "enabled" if gpu_details["use_gpu"] else "disabled",
                "gpu_device": gpu_details["device_name"],
                "gh200_optimizations": "enabled" if gpu_details["use_gh200"] else "disabled",
                "test_mode": self.test_mode,
                "cache_dir": str(self.cache_dir), # Use loaded Path object
                "timestamp": time.time()
            }
        )

    # Removed _update_config_recursive - config is handled by the central Config object

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist using loaded Path objects."""
        # Use the Path objects loaded from config
        for directory_path in [self.data_dir, self.monitoring_dir, self.cache_dir]:
            try:
                directory_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Ensured directory exists: {directory_path}")
            except OSError as e:
                # Raise a specific configuration error if directory creation fails
                msg = f"Failed to create required directory {directory_path}: {e}"
                self.logger.error(msg)
                raise ConfigurationError(msg) from e
            except Exception as e: # Catch any other unexpected errors
                self.logger.exception(f"Unexpected error creating directory {directory_path}: {e}")
                # Optionally re-raise or handle

    # Removed _initialize_gpu method - now handled by utils.gpu_utils.gpu_manager

    def _send_frontend_notification(self, message, level="info", category="data_pipeline", details=None):
        """Send notification to frontend via Redis."""
        # Assume self.redis is valid if initialization succeeded
        # if not self.redis:
        #     self.logger.debug(f"Redis not available, skipping notification: {message}")
        #     return

        try:
            notification = {
                "type": category, "message": message, "level": level,
                "timestamp": time.time(), "details": details or {}
            }
            notification_json = json.dumps(notification)
            # Use configured keys and limits
            category_key = f"frontend:{category}" # Keep prefix for now

            # Use Redis pipeline for atomic operations
            # Ensure RedisClient supports pipeline context manager or use client directly
            # Assuming RedisClient provides access to the underlying client or pipeline method
            if hasattr(self.redis, 'pipeline'):
                 with self.redis.pipeline() as pipe:
                     pipe.lpush(self.redis_notify_key, notification_json)
                     pipe.ltrim(self.redis_notify_key, 0, self.redis_notify_limit - 1)
                     pipe.lpush(category_key, notification_json)
                     pipe.ltrim(category_key, 0, self.redis_category_limit - 1)
                     pipe.execute()
            else: # Fallback if pipeline context manager not available on RedisClient wrapper
                 self.redis.lpush(self.redis_notify_key, notification_json)
                 self.redis.ltrim(self.redis_notify_key, 0, self.redis_notify_limit - 1)
                 self.redis.lpush(category_key, notification_json)
                 self.redis.ltrim(category_key, 0, self.redis_category_limit - 1)

            log_func = getattr(logger, level, logger.info)
            log_func(f"Frontend notification: {message}")

            if category in ["system_status", "data_system"]:
                try:
                    # Use configured status key
                    system_status = json.loads(self.redis.get(self.redis_status_key) or "{}")
                    system_status.update({
                        "last_update": time.time(),
                        "last_message": message,
                        "status": level
                    })
                    self.redis.set(self.redis_status_key, json.dumps(system_status))
                except (json.JSONDecodeError, RedisError) as e: # Catch specific errors
                    self.logger.error(f"Redis error updating system status: {e}")
                except Exception as e: # Fallback
                     self.logger.error(f"Unexpected error updating system status: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error sending frontend notification: {e}", exc_info=True)

    # Removed _process_dataframe_with_gpu - use gpu_utils.process_array or similar
    # Removed _process_with_pytorch - use gpu_utils.to_gpu/from_gpu
    # Removed _optimize_pytorch_model_with_tensorrt - use gpu_utils.optimize_model

    # --- Data Access Methods ---

    async def get_ticker_details(self, ticker: str) -> Optional[dict]:
        """Fetch detailed information for a specific ticker."""
        # Use appropriate TTL (e.g., default or a specific one for details)
        ttl = self.default_cache_ttl
        cache_key = f"ticker_details:{ticker}"
        cached = load_from_cache(cache_key, self.redis, ttl)
        if cached is not None: # Check explicitly for None, as empty dict is valid
            return cached

        if not self.polygon:
            logger.warning("Polygon client not available for get_ticker_details")
            return None

        try:
            if hasattr(self.polygon, 'get_ticker_details'):
                details = await self.polygon.get_ticker_details(ticker)
                if details:
                    save_to_cache(cache_key, details, self.redis, ttl)
                    return details
            else:
                logger.warning("Polygon client missing 'get_ticker_details' method")
            return None
        except Exception as e:
            # Use specific exceptions if possible (e.g., APIError from client)
            self.logger.error(f"Error fetching ticker details for {ticker}: {e}", exc_info=True)
            return None

    async def get_all_tickers(self, market: str = 'stocks', active: bool = True, limit: int = 50000) -> Optional[pd.DataFrame]:
        """Fetch a list of all available tickers."""
        # Use appropriate TTL (e.g., default or a specific one for tickers list)
        ttl = self.default_cache_ttl
        cache_key = f"all_tickers:{market}:{active}:{limit}"
        cached = load_from_cache(cache_key, self.redis, ttl)
        if cached is not None:
            # Ensure cached data is DataFrame (assuming save_to_cache handles serialization)
            return cached # Assuming load_from_cache returns DataFrame directly

        if not self.polygon:
            logger.warning("Polygon client not available for get_all_tickers")
            return None

        try:
            if hasattr(self.polygon, 'list_tickers'):
                tickers_df = await self.polygon.list_tickers(market=market, active=active, limit=limit)
                if tickers_df is not None and not tickers_df.empty:
                    save_to_cache(cache_key, tickers_df, self.redis, ttl)
                    return tickers_df
            else:
                 logger.warning("Polygon client missing 'list_tickers' method")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching all tickers: {e}", exc_info=True)
            return None

    async def get_latest_quote(self, ticker: str) -> Optional[dict]:
        """Fetch the latest quote for a ticker."""
        if not self.polygon:
            logger.warning("Polygon client not available for get_latest_quote")
            return None
        try:
            if hasattr(self.polygon, 'get_last_quote'):
                quote = await self.polygon.get_last_quote(ticker)
                if quote and 'ask_price' in quote and 'bid_price' in quote:
                     return quote
                else:
                     logger.warning(f"Received invalid quote for {ticker}: {quote}")
                     return None
            else:
                logger.warning("Polygon client missing 'get_last_quote' method")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching latest quote for {ticker}: {e}", exc_info=True)
            return None

    async def get_aggregates(
        self,
        ticker: str,
        timespan: str = 'minute',
        multiplier: int = 1,
        start_date: str = None,
        end_date: str = None,
        limit: int = 5000,
        apply_clean: bool = True # Option to skip cleaning if done elsewhere
    ) -> Optional[pd.DataFrame]:
        """Fetch historical aggregate bars for a ticker."""
        # Use Polygon-specific TTL
        ttl = self.polygon_cache_ttl
        cache_key = f"aggregates:{ticker}:{timespan}:{multiplier}:{start_date}:{end_date}:{limit}:{apply_clean}"
        cached = load_from_cache(cache_key, self.redis, ttl)
        if cached is not None:
             return cached # Assuming load_from_cache returns DataFrame

        if not self.polygon:
            logger.warning("Polygon client not available for get_aggregates")
            return None

        try:
            if hasattr(self.polygon, 'get_aggregates'):
                agg_df = await self.polygon.get_aggregates(
                    symbol=ticker, multiplier=multiplier, timespan=timespan,
                    from_date=start_date, to_date=end_date, limit=limit
                )

                if agg_df is not None and not agg_df.empty:
                    if apply_clean:
                        agg_df = clean_market_data(agg_df) # Use imported function

                    save_to_cache(cache_key, agg_df, self.redis, ttl)
                    return agg_df
                else:
                    logger.warning(f"No aggregate data returned for {ticker} with params: {timespan}, {start_date}, {end_date}")
                    # Cache the empty result to avoid repeated failed calls for a period
                    save_to_cache(cache_key, pd.DataFrame(), self.redis, 300) # Cache empty for 5 mins
                    return pd.DataFrame() # Return empty DataFrame consistently
            else:
                logger.warning("Polygon client missing 'get_aggregates' method")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching aggregates for {ticker}: {e}", exc_info=True)
            return None

    # --- Methods needed by trading_engine (Synchronous Wrappers) ---
    # These use asyncio.run which is generally discouraged in production servers.
    # Ideally, the calling code (trading_engine) should be async or manage its own loop/thread.

    def get_market_data(self, symbol: str) -> Optional[dict]:
        """Provides recent market data needed by trading_engine (SYNC WRAPPER)."""
        logger.warning("get_market_data called synchronously - consider async usage")
        try:
            now = pd.Timestamp.utcnow()
            start = (now - pd.Timedelta(minutes=60)).strftime('%Y-%m-%d')
            end = now.strftime('%Y-%m-%d')
            # Run the async method in the existing or a new loop
            df = asyncio.run_coroutine_threadsafe(
                self.get_aggregates(symbol, 'minute', 1, start, end, limit=60, apply_clean=True),
                self.loop
            ).result(timeout=10) # Add timeout

            if df is not None and not df.empty:
                return {
                    'close': df['close'].tolist(),
                    'volume': df['volume'].tolist(),
                    'timestamp': df['timestamp'].astype(str).tolist()
                }
            return None
        except Exception as e:
            self.logger.error(f"Error in sync get_market_data for {symbol}: {e}", exc_info=True)
            return None

    def get_intraday_data(self, symbol: str, minutes: Optional[int] = None) -> Optional[dict]:
        """Provides intraday data needed by trading_engine strategies (SYNC WRAPPER)."""
        logger.warning("get_intraday_data called synchronously - consider async usage")
        try:
            today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
            limit = minutes if minutes else 5000
            df = asyncio.run_coroutine_threadsafe(
                 self.get_aggregates(symbol, 'minute', 1, today, today, limit=limit, apply_clean=True),
                 self.loop
            ).result(timeout=10)

            if df is not None and not df.empty:
                return {
                    'close': df['close'].tolist(),
                    'volume': df['volume'].tolist(),
                    'timestamp': df['timestamp'].astype(str).tolist()
                }
            return None
        except Exception as e:
            self.logger.error(f"Error in sync get_intraday_data for {symbol}: {e}", exc_info=True)
            return None

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Provides the last known price for a symbol (SYNC WRAPPER)."""
        logger.warning("get_last_price called synchronously - consider async usage")
        try:
            quote = asyncio.run_coroutine_threadsafe(
                self.get_latest_quote(symbol),
                self.loop
            ).result(timeout=5)

            if quote:
                 # Prioritize last trade price if available
                 last_trade = quote.get('last_trade', {})
                 price = last_trade.get('p')
                 if price is not None:
                      return float(price)
                 # Fallback to mid-price
                 bid = quote.get('bid_price')
                 ask = quote.get('ask_price')
                 if bid is not None and ask is not None:
                      return (float(bid) + float(ask)) / 2.0
            return None # Return None if no usable price found
        except Exception as e:
            self.logger.error(f"Error in sync get_last_price for {symbol}: {e}", exc_info=True)
            return None
