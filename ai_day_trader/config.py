"""
Configuration loading for the AI Day Trading Bot.
Merges general config with API client specific settings.
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from ai_day_trader.utils.exceptions import ConfigurationError # Use new utils path
from ai_day_trader.utils.config import Config as BaseConfig # Use new utils path
from ai_day_trader.utils.config import get_env_var, normalize_env_var_name # Use new utils path

# Define logger for this module
logger = logging.getLogger(__name__)

def load_ai_trader_config(env_file: Optional[str] = '.env') -> BaseConfig:
    """
    Loads configuration specifically for the AI Day Trader,
    including general settings and API client settings.

    Args:
        env_file: Path to the .env file. Defaults to '.env'.

    Returns:
        An instance of Config with all settings loaded.

    Raises:
        ConfigurationError: If required settings are missing or invalid.
    """
    config = BaseConfig(prefix="TRADING", load_env=True, env_file=env_file)

    # --- AI Trader Specific Settings ---
    config.SYMBOLS = config.get_list("SYMBOLS", ["AAPL", "MSFT", "NVDA"])
    config.STRATEGY_LOOP_INTERVAL_SECONDS = config.get_float("STRATEGY_LOOP_INTERVAL_SECONDS", 10.0)
    config.EOD_CLOSE_MINUTES_BEFORE = config.get_int("EOD_CLOSE_MINUTES_BEFORE", 15)
    config.STOP_LOSS_PCT = config.get_float("STOP_LOSS_PCT", 0.02)
    config.PEAK_EXIT_DROP_THRESHOLD = config.get_float("PEAK_EXIT_DROP_THRESHOLD", 0.01)
    config.ML_EXIT_THRESHOLD = config.get_float("ML_EXIT_THRESHOLD", 0.75)
    config.SIGNAL_CONFIDENCE_THRESHOLD = config.get_float("SIGNAL_CONFIDENCE_THRESHOLD", 0.6)
    config.PORTFOLIO_SIZE = config.get_float("PORTFOLIO_SIZE", 100000.0)
    # Risk Management - Allow fixed amount OR percentage
    # Use get() first, then parse to float if not None
    raw_max_daily_risk = config.get("MAX_DAILY_RISK_AMOUNT", None)
    config.MAX_DAILY_RISK_AMOUNT = float(raw_max_daily_risk) if raw_max_daily_risk is not None else None
    config.MAX_DAILY_RISK_PCT = config.get_float("MAX_DAILY_RISK_PCT", 0.01) # Used if fixed amount not set
    config.MAX_TRADE_RISK_PCT = config.get_float("MAX_TRADE_RISK_PCT", 0.005)

    # --- Execution Mode ---
    config.EXECUTION_MODE = config.get_str("EXECUTION_MODE", "live").lower() # 'live' or 'paper'
    if config.EXECUTION_MODE not in ['live', 'paper']:
        raise ConfigurationError(f"Invalid TRADING_EXECUTION_MODE: '{config.EXECUTION_MODE}'. Must be 'live' or 'paper'.")

    # --- Stock Selector Settings ---
    # Import defaults from stock_selector to avoid duplication
    from ai_day_trader.stock_selector import (
        DEFAULT_MIN_SHARE_PRICE, DEFAULT_MAX_SHARE_PRICE_BUFFER,
        DEFAULT_MIN_AVG_DOLLAR_VOLUME, DEFAULT_POLYGON_TICKERS_LIMIT,
        DEFAULT_CANDIDATE_CACHE_KEY, DEFAULT_CANDIDATE_CACHE_TTL_SECONDS
    )
    config.STOCK_SELECTOR_MIN_PRICE = config.get_float("STOCK_SELECTOR_MIN_PRICE", DEFAULT_MIN_SHARE_PRICE)
    config.STOCK_SELECTOR_MAX_PRICE_BUFFER = config.get_float("STOCK_SELECTOR_MAX_PRICE_BUFFER", DEFAULT_MAX_SHARE_PRICE_BUFFER)
    config.STOCK_SELECTOR_MIN_DOLLAR_VOLUME = config.get_float("STOCK_SELECTOR_MIN_DOLLAR_VOLUME", DEFAULT_MIN_AVG_DOLLAR_VOLUME)
    config.POLYGON_TICKERS_LIMIT = config.get_int("POLYGON_TICKERS_LIMIT", DEFAULT_POLYGON_TICKERS_LIMIT)
    config.STOCK_SELECTOR_CACHE_KEY = config.get_str("STOCK_SELECTOR_CACHE_KEY", DEFAULT_CANDIDATE_CACHE_KEY)
    config.STOCK_SELECTOR_CACHE_TTL_SECONDS = config.get_int("STOCK_SELECTOR_CACHE_TTL_SECONDS", DEFAULT_CANDIDATE_CACHE_TTL_SECONDS)


    # --- API Client Settings (Merged from ApiClientConfig) ---

    def clean_numeric_env_var(var_name: str, default_val: str, as_float: bool = False) -> Union[int, float]:
        """Clean and parse numeric environment variables that might contain comments"""
        val_str = config.get(var_name, default_val)
        # Add check for None before splitting
        if val_str is None:
             raise ConfigurationError(f"Value for {var_name} is unexpectedly None.")
        cleaned_val = str(val_str).split()[0] # Ensure it's a string before split
        try:
            return float(cleaned_val) if as_float else int(cleaned_val)
        except ValueError:
                 raise ConfigurationError(f"Invalid numeric value for {var_name}: '{val_str}'")

        # Polygon.io - polygon_api_key is loaded by BaseConfig.__init__
        # config.polygon_api_key = config.get("POLYGON_API_KEY", "") # REMOVED - Already loaded
        config.polygon_api_base_url = config.get("POLYGON_API_BASE_URL", "https://api.polygon.io")
        config.polygon_ws_url = config.get("POLYGON_WS_URL", "wss://socket.polygon.io/stocks")
        config.polygon_rate_limit = clean_numeric_env_var("POLYGON_RATE_LIMIT", "5")
        config.polygon_cache_ttl = clean_numeric_env_var("POLYGON_CACHE_TTL", "3600")

        # Unusual Whales - unusual_whales_api_key is loaded by BaseConfig.__init__
        # config.unusual_whales_api_key = config.get("UNUSUAL_WHALES_API_KEY", "") # REMOVED - Already loaded
        config.unusual_whales_api_base_url = config.get("UNUSUAL_WHALES_API_BASE_URL", "https://api.unusualwhales.com/api")
        config.unusual_whales_rate_limit = clean_numeric_env_var("UNUSUAL_WHALES_RATE_LIMIT", "2")
        config.unusual_whales_cache_ttl = clean_numeric_env_var("UNUSUAL_WHALES_CACHE_TTL", "300")

        # Alpaca - alpaca_api_key_id and alpaca_api_secret_key are loaded by BaseConfig.__init__
        # config.alpaca_api_key_id = config.get("APCA_API_KEY_ID", "") # REMOVED - Already loaded
        # config.alpaca_api_secret_key = config.get("APCA_API_SECRET_KEY", "") # REMOVED - Already loaded
        config.alpaca_api_base_url = config.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

        # Redis - redis_password is loaded by BaseConfig.__init__
        config.redis_host = config.get("REDIS_HOST", "localhost")
        config.redis_port = clean_numeric_env_var("REDIS_PORT", "6379")
        config.redis_db = clean_numeric_env_var("REDIS_DB", "0")
        # config.redis_password = config.get("REDIS_PASSWORD", None) # REMOVED - Already loaded
        config.redis_max_connections = int(clean_numeric_env_var("REDIS_MAX_CONNECTIONS", "50")) # Cast to int

        # Connection Pool & Retry
    config.max_retries = int(clean_numeric_env_var("DEFAULT_RETRY_COUNT", "3")) # Cast to int
    config.retry_backoff_factor = clean_numeric_env_var("DEFAULT_RETRY_DELAY", "1.0", as_float=True)
    config.connection_timeout = int(clean_numeric_env_var("DEFAULT_TIMEOUT", "30")) # Cast to int
    config.max_pool_size = int(clean_numeric_env_var("MAX_CONCURRENT_REQUESTS", "20")) # Cast to int

    # WebSocket Specific
    config.max_reconnect_attempts = int(clean_numeric_env_var("WS_RETRY_COUNT", "10")) # Cast to int
    config.reconnect_delay = clean_numeric_env_var("WS_RETRY_DELAY", "2.0", as_float=True)
    config.buffer_size = int(clean_numeric_env_var("WS_BUFFER_SIZE", "10000")) # Cast to int
    config.max_queue_size = int(clean_numeric_env_var("WS_MAX_QUEUE_SIZE", "50000")) # Cast to int

    # --- ML Model Configuration ---
    model_dir_path = config.get_path("MODEL_DIR", Path("./models"))
    if model_dir_path is None: # Add check for None, though default should prevent it
        raise ConfigurationError("MODEL_DIR configuration resulted in None.")
    config.MODEL_DIR = model_dir_path
    config.XGBOOST_USE_GPU = config.get_bool("XGBOOST_USE_GPU", True)
    # Cast result to string to satisfy type checker
    config.XGBOOST_TREE_METHOD = str(config.get("XGBOOST_TREE_METHOD", "gpu_hist" if config.XGBOOST_USE_GPU else "hist"))
    config.XGBOOST_GPU_ID = config.get_int("XGBOOST_GPU_ID", 0)


    # --- Validate Required API Keys ---
    if not config.polygon_api_key:
        raise ConfigurationError(f"{normalize_env_var_name('POLYGON_API_KEY', config.prefix)} environment variable is required.")
    if not config.alpaca_api_key_id or not config.alpaca_api_secret_key:
         logger.warning(f"{normalize_env_var_name('APCA_API_KEY_ID', config.prefix)} or {normalize_env_var_name('APCA_API_SECRET_KEY', config.prefix)} not set. Live execution/market clock disabled.")

    return config
