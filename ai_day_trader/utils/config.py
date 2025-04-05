"""
Configuration Management Utilities
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv

# Define logger for this module
logger = logging.getLogger("ai_day_trader.utils.config")

# --- Helper Functions ---

def normalize_env_var_name(var_name: str, prefix: Optional[str] = None) -> str:
    """Normalizes environment variable name (e.g., adds prefix)."""
    if prefix and not var_name.startswith(f"{prefix}_"):
        return f"{prefix}_{var_name}"
    return var_name

def get_env_var(
    var_name: str,
    default: Optional[str] = None,
    prefix: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """
    Retrieves an environment variable, optionally adding a prefix.

    Args:
        var_name: The base name of the environment variable.
        default: The default value if the variable is not found.
        prefix: An optional prefix to add to the variable name (e.g., "TRADING").
        required: If True, raises an error if the variable is not set.

    Returns:
        The value of the environment variable or the default.

    Raises:
        ValueError: If the variable is required but not set.
    """
    full_var_name = normalize_env_var_name(var_name, prefix)
    value = os.environ.get(full_var_name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{full_var_name}' is not set.")

    # Basic cleaning: remove potential leading/trailing whitespace/quotes
    if isinstance(value, str):
        value = value.strip().strip("'\"")

    return value

# --- Config Class ---

class Config:
    """
    Centralized configuration class. Loads settings from environment variables
    and .env files, providing typed access methods.
    """
    def __init__(
        self,
        prefix: Optional[str] = None,
        load_env: bool = True,
        env_file: Optional[str] = '.env'
    ):
        """
        Initializes the Config object.

        Args:
            prefix: Optional prefix for environment variables (e.g., "TRADING").
            load_env: If True, load variables from a .env file.
            env_file: Path to the .env file (default: '.env').
        """
        self.prefix = prefix
        self._config: Dict[str, Any] = {} # Store loaded values

        if load_env:
            env_path = Path(env_file) if env_file else Path('.env')
            if env_path.is_file():
                logger.info(f"Loading environment variables from: {env_path.resolve()}")
                load_dotenv(dotenv_path=env_path, override=True)
            else:
                logger.warning(f".env file not found at {env_path.resolve()}, relying solely on environment variables.")

        # --- Define Expected Attributes with Defaults ---
        # AI Trader Specific
        self.SYMBOLS: List[str] = ["AAPL", "MSFT", "NVDA"] # Provide a default list
        self.STRATEGY_LOOP_INTERVAL_SECONDS: float = 10.0
        self.EOD_CLOSE_MINUTES_BEFORE: int = 15
        self.STOP_LOSS_PCT: float = 0.02
        self.PEAK_EXIT_DROP_THRESHOLD: float = 0.01
        self.ML_EXIT_THRESHOLD: float = 0.75
        self.SIGNAL_CONFIDENCE_THRESHOLD: float = 0.6
        self.PORTFOLIO_SIZE: float = 100000.0
        self.MAX_DAILY_RISK_AMOUNT: Optional[float] = None
        self.MAX_DAILY_RISK_PCT: float = 0.01
        self.MAX_TRADE_RISK_PCT: float = 0.005

        # Polygon.io
        self.polygon_api_key: Optional[str] = None
        self.polygon_api_base_url: str = "https://api.polygon.io"
        self.polygon_ws_url: str = "wss://socket.polygon.io/stocks"
        self.polygon_rate_limit: int = 5
        self.polygon_cache_ttl: int = 3600

        # Unusual Whales
        self.unusual_whales_api_key: Optional[str] = None
        self.unusual_whales_api_base_url: str = "https://api.unusualwhales.com/api"
        self.unusual_whales_rate_limit: int = 2
        self.unusual_whales_cache_ttl: int = 300

        # Alpaca
        self.alpaca_api_key_id: Optional[str] = None
        self.alpaca_api_secret_key: Optional[str] = None
        self.alpaca_api_base_url: str = "https://paper-api.alpaca.markets"

        # Redis
        self.redis_host: str = "localhost"
        self.redis_port: int = 6379
        self.redis_db: int = 0
        self.redis_password: Optional[str] = None
        self.redis_max_connections: int = 50

        # Connection Pool & Retry
        self.max_retries: int = 3
        self.retry_backoff_factor: float = 1.0
        self.connection_timeout: int = 30
        self.max_pool_size: int = 20

        # WebSocket Specific
        self.max_reconnect_attempts: int = 10
        self.reconnect_delay: float = 2.0
        self.buffer_size: int = 10000
        self.max_queue_size: int = 50000

        # ML Model Configuration
        self.MODEL_DIR: Path = Path("./models")
        self.XGBOOST_USE_GPU: bool = True
        self.XGBOOST_TREE_METHOD: str = "hist"
        self.XGBOOST_GPU_ID: int = 0

        # Sentry (Optional)
        self.SENTRY_ENABLED: bool = True
        self.SENTRY_DSN: Optional[str] = None
        self.SENTRY_ENVIRONMENT: str = "development"
        self.SENTRY_TRACES_SAMPLE_RATE: float = 0.1
        self.SENTRY_PROFILES_SAMPLE_RATE: float = 0.1
        self.SENTRY_DEBUG: bool = False
        self.SENTRY_RELEASE: str = "1.0.0"

        # Logging
        self.LOG_LEVEL: str = "INFO"
        self.LOG_TO_CONSOLE: bool = True
        self.LOG_TO_FILE: bool = True
        self.LOG_DIR: str = "./logs"
        self.LOG_FILE: str = "trading_system.log"

        # --- Load values from environment ---
        # Iterate over defined attributes to load corresponding env vars
        default_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        for key, default_value in default_attrs.items():
            # Convert attribute name (e.g., polygon_api_key) to env var name (e.g., POLYGON_API_KEY)
            env_var_name = key.upper()
            env_value_str = get_env_var(env_var_name, None, self.prefix) # Use uppercase name
            final_value = default_value # Start with default

            if env_value_str is not None:
                try:
                    target_type = type(default_value)
                    if default_value is None and key in ['MAX_DAILY_RISK_AMOUNT']: # Handle Optional[float] specifically
                         target_type = float # Attempt to cast to float if provided
                    elif default_value is None: # Handle Optional[str]
                         target_type = str

                    if target_type == bool:
                        final_value = env_value_str.lower() in ['true', '1', 'yes', 'y']
                    elif target_type == int:
                        final_value = int(env_value_str.split()[0])
                    elif target_type == float:
                        final_value = float(env_value_str.split()[0])
                    elif target_type == Path:
                        final_value = Path(env_value_str)
                    elif target_type == list: # Assumes list of strings
                        final_value = [item.strip() for item in env_value_str.split(',') if item.strip()]
                    elif target_type == str:
                         final_value = env_value_str # Already a string
                    # Add other type conversions if needed

                    setattr(self, key, final_value)
                    self._config[key] = final_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not cast env var {key}='{env_value_str}' to type {target_type}. Using default '{default_value}'. Error: {e}")
                    setattr(self, key, default_value) # Fallback to default
                    self._config[key] = default_value
            else:
                # If env var not found, use the default value already set
                setattr(self, key, default_value)
                self._config[key] = default_value

        # Special handling after loading all values
        if self.XGBOOST_USE_GPU:
             self.XGBOOST_TREE_METHOD = os.environ.get(normalize_env_var_name("XGBOOST_TREE_METHOD", self.prefix), "gpu_hist")
        else:
             self.XGBOOST_TREE_METHOD = os.environ.get(normalize_env_var_name("XGBOOST_TREE_METHOD", self.prefix), "hist")
        self._config["XGBOOST_TREE_METHOD"] = self.XGBOOST_TREE_METHOD


    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Gets a configuration value."""
        # Use getattr which respects the defined attributes and defaults
        return getattr(self, key, default)

    def get_str(self, key: str, default: str = "") -> str:
        """Gets a configuration value as a string."""
        value = self.get(key, default)
        return str(value) if value is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        """Gets a configuration value as an integer."""
        value = self.get(key, default)
        try:
            # Check if value is already int
            if isinstance(value, int): return value
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            logger.warning(f"Config value for '{key}' ('{value}') is not a valid integer. Using default: {default}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Gets a configuration value as a float."""
        value = self.get(key, default)
        try:
             # Check if value is already float
             if isinstance(value, float): return value
             return float(value) if value is not None else default
        except (ValueError, TypeError):
            logger.warning(f"Config value for '{key}' ('{value}') is not a valid float. Using default: {default}")
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Gets a configuration value as a boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'y']
        try:
             return bool(value) if value is not None else default
        except Exception:
             logger.warning(f"Could not interpret config value for '{key}' ('{value}') as boolean. Using default: {default}")
             return default


    def get_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        """Gets a configuration value as a list of strings (comma-separated)."""
        value = self.get(key, None)
        actual_default = default if default is not None else []
        if value is None:
            return actual_default
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        logger.warning(f"Config value for '{key}' is not a string or list. Cannot parse as list. Using default.")
        return actual_default

    def get_path(self, key: str, default: Optional[Path] = None) -> Optional[Path]:
         """Gets a configuration value as a Path object."""
         value = self.get(key, None)
         if value is None:
             return default
         if isinstance(value, Path):
             return value
         if isinstance(value, str):
             try:
                  return Path(value)
             except TypeError:
                  logger.warning(f"Config value for '{key}' ('{value}') is not a valid path string. Using default.")
                  return default
         logger.warning(f"Config value for '{key}' ('{value}') is not a valid path string or Path object. Using default.")
         return default

    def as_dict(self) -> Dict[str, Any]:
         """Returns the configuration as a dictionary."""
         return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self) -> str:
        items = {k: v for k, v in self.as_dict().items() if k not in ['prefix']}
        return f"Config({items})"
