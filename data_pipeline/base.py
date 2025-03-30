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

# Import functions from other pipeline modules
from .loading import load_from_cache, save_to_cache
from .processing import clean_market_data # Keep processing imports if needed by access methods

# Configure logging
logger = get_logger("data_pipeline")

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
        polygon_client=None,
        polygon_ws=None,
        unusual_whales_client=None,
        redis_client=None,
        config=None,
        # Removed use_gpu, use_gh200 - handled by gpu_manager
        test_mode=False,
    ) -> None:
        """
        Initialize the data pipeline

        Args:
            polygon_client: Polygon API client
            polygon_ws: Polygon WebSocket client
            unusual_whales_client: Unusual Whales API client
            redis_client: Redis client for caching
            config: Configuration parameters (dict or loaded from config.py)
            test_mode: Whether to use mock data for testing
        """
        self.polygon = polygon_client
        self.polygon_ws = polygon_ws
        self.unusual_whales = unusual_whales_client
        self.redis = redis_client

        # Test mode for using synthetic data
        self.test_mode = test_mode

        # GPU status is now determined by the global gpu_manager
        self.use_gpu = is_gpu_available() # Check status from gpu_manager
        self.gpu_info = get_device_info() # Get detailed info if needed

        # Default configuration
        self.default_config = {
            "cache_dir": os.environ.get("DATA_CACHE_DIR", "./data/cache"),
            "cache_expiry": 86400,  # 1 day in seconds
            "rate_limit": {
                "polygon": 5,  # requests per second
                "unusual_whales": 2,  # requests per second
            },
            "retry_settings": {
                "stop_max_attempt_number": 3,
                "wait_exponential_multiplier": 1000,
                "wait_exponential_max": 10000,
            },
            "data_dir": os.environ.get("DATA_DIR", "./data"),
            "monitoring_dir": os.environ.get("MONITORING_DIR", "./monitoring"),
            "min_samples": 1000,
            "lookback_days": 30,
            "monitoring": {"enabled": True, "drift_threshold": 0.05},
            "feature_selection": {
                "enabled": True,
                "method": "importance",
                "threshold": 0.01,
                "n_features": 20,
            },
            "time_series_cv": {"n_splits": 5, "embargo_size": 10},
            "watchlist": {
                "refresh_interval": 900,  # 15 minutes
                "max_size": 100,
                "min_price": 5.0,
                "min_volume": 500000,
            },
        }

        # Update with provided config
        self.config = self.default_config.copy()
        if config:
            self._update_config_recursive(self.config, config)

        # Ensure directories exist
        self._ensure_directories()

        # GPU initialization is handled globally by gpu_manager instance creation
        # No need to call self._initialize_gpu() here

        # Create event loop for async calls
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        logger.info("Data Pipeline initialized")

        # Send notification to frontend using info from gpu_manager
        if self.redis:
            gpu_details = get_device_info()
            self._send_frontend_notification(
                message="Data Pipeline initialized successfully",
                level="info",
                category="system_startup",
                details={
                    "gpu_acceleration": "enabled" if gpu_details["use_gpu"] else "disabled",
                    "gpu_device": gpu_details["device_name"],
                    "gh200_optimizations": "enabled" if gpu_details["use_gh200"] else "disabled",
                    "test_mode": self.test_mode,
                    "cache_dir": self.config["cache_dir"],
                    "timestamp": time.time()
                }
            )

    def _update_config_recursive(
        self, target: dict[str, Any], source: dict[str, Any],
    ) -> None:
        """Recursively update a nested dictionary."""
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._update_config_recursive(target[key], value)
                else:
                    target[key] = value
            else:
                target[key] = value

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [
            self.config["data_dir"],
            self.config["monitoring_dir"],
            self.config["cache_dir"],
        ]:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.exception(f"Error creating directory {directory}: {e!s}")

    # Removed _initialize_gpu method - now handled by utils.gpu_utils.gpu_manager

    def _send_frontend_notification(self, message, level="info", category="data_pipeline", details=None):
        """Send notification to frontend via Redis."""
        if not self.redis:
            logger.debug(f"Redis not available, skipping notification: {message}")
            return

        try:
            notification = {
                "type": category, "message": message, "level": level,
                "timestamp": time.time(), "details": details or {}
            }
            notification_json = json.dumps(notification)
            list_key = "frontend:notifications"
            category_key = f"frontend:{category}"

            with self.redis.pipeline() as pipe:
                pipe.lpush(list_key, notification_json)
                pipe.ltrim(list_key, 0, 99)
                pipe.lpush(category_key, notification_json)
                pipe.ltrim(category_key, 0, 49)
                pipe.execute()

            log_func = getattr(logger, level, logger.info)
            log_func(f"Frontend notification: {message}")

            if category in ["system_status", "data_system"]:
                try:
                    status_key = "frontend:system:status"
                    system_status = json.loads(self.redis.get(status_key) or "{}")
                    system_status.update({
                        "last_update": time.time(),
                        "last_message": message,
                        "status": level
                    })
                    self.redis.set(status_key, json.dumps(system_status))
                except Exception as e:
                    logger.error(f"Error updating system status: {e}")

        except Exception as e:
            logger.error(f"Error sending frontend notification: {e}")

    # Removed _process_dataframe_with_gpu - use gpu_utils.process_array or similar
    # Removed _process_with_pytorch - use gpu_utils.to_gpu/from_gpu
    # Removed _optimize_pytorch_model_with_tensorrt - use gpu_utils.optimize_model

    # --- Data Access Methods ---

    async def get_ticker_details(self, ticker: str) -> Optional[dict]:
        """Fetch detailed information for a specific ticker."""
        cache_key = f"ticker_details:{ticker}"
        cached = load_from_cache(cache_key, self.redis, self.config['cache_expiry'])
        if cached:
            return cached

        if not self.polygon:
            logger.warning("Polygon client not available for get_ticker_details")
            return None

        try:
            if hasattr(self.polygon, 'get_ticker_details'):
                details = await self.polygon.get_ticker_details(ticker)
                if details:
                    save_to_cache(cache_key, details, self.redis, self.config['cache_expiry'])
                    return details
            else:
                logger.warning("Polygon client missing 'get_ticker_details' method")
            return None
        except Exception as e:
            logger.error(f"Error fetching ticker details for {ticker}: {e}")
            return None

    async def get_all_tickers(self, market: str = 'stocks', active: bool = True, limit: int = 50000) -> Optional[pd.DataFrame]:
        """Fetch a list of all available tickers."""
        cache_key = f"all_tickers:{market}:{active}:{limit}"
        cached = load_from_cache(cache_key, self.redis, self.config['cache_expiry'])
        if cached is not None:
            # Ensure cached data is DataFrame
            return cached if isinstance(cached, pd.DataFrame) else pd.DataFrame(cached)

        if not self.polygon:
            logger.warning("Polygon client not available for get_all_tickers")
            return None

        try:
            if hasattr(self.polygon, 'list_tickers'):
                tickers_df = await self.polygon.list_tickers(market=market, active=active, limit=limit)
                if tickers_df is not None and not tickers_df.empty:
                    save_to_cache(cache_key, tickers_df, self.redis, self.config['cache_expiry'])
                    return tickers_df
            else:
                 logger.warning("Polygon client missing 'list_tickers' method")
            return None
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
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
            logger.error(f"Error fetching latest quote for {ticker}: {e}")
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
        cache_key = f"aggregates:{ticker}:{timespan}:{multiplier}:{start_date}:{end_date}:{limit}:{apply_clean}"
        cached = load_from_cache(cache_key, self.redis, self.config['cache_expiry'])
        if cached is not None:
             return cached if isinstance(cached, pd.DataFrame) else pd.DataFrame(cached)

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

                    save_to_cache(cache_key, agg_df, self.redis, self.config['cache_expiry'])
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
            logger.error(f"Error fetching aggregates for {ticker}: {e}")
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
            logger.error(f"Error in sync get_market_data for {symbol}: {e}")
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
            logger.error(f"Error in sync get_intraday_data for {symbol}: {e}")
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
            logger.error(f"Error in sync get_last_price for {symbol}: {e}")
            return None
