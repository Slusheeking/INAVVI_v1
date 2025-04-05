#!/usr/bin/env python3
"""
Logging Configuration Module

This module provides a standardized logging configuration for the application.
It supports both console and file logging, with optional JSON formatting.
"""

import logging
import logging.handlers
import os
import socket
import sys
from datetime import datetime
from typing import Dict, Optional, Union
from pythonjsonlogger import jsonlogger

# Try to import sentry_sdk
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Try to import config
try:
    # Import from the new relative path within ai_day_trader.utils
    from . import config
    from .config import get_env_var
except ImportError:
    config = None
    get_env_var = None

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default JSON log format
JSON_LOG_FORMAT = True

# Default log directory
DEFAULT_LOG_DIR = "./logs"

# Default log file name (not full path)
DEFAULT_LOG_FILE = "trading_system.log"

# Default max bytes for rotating file handler
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

# Default backup count for rotating file handler
DEFAULT_BACKUP_COUNT = 5

# Default log to console
DEFAULT_LOG_TO_CONSOLE = True

# Default log to file
DEFAULT_LOG_TO_FILE = True

# Default application name
DEFAULT_APP_NAME = "trading_system"

# Default environment
DEFAULT_ENVIRONMENT = "development"

# Default version
DEFAULT_VERSION = "1.0.0"

# Custom JSON formatter using python-json-logger
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Extended JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
        
        # Add hostname
        log_record['hostname'] = getattr(record, 'hostname', socket.gethostname())
        
        # Add app info
        log_record['app_name'] = getattr(record, 'app_name', DEFAULT_APP_NAME)
        log_record['environment'] = getattr(record, 'environment', DEFAULT_ENVIRONMENT)
        log_record['version'] = getattr(record, 'version', DEFAULT_VERSION)
        
        # Add standard fields
        log_record['level'] = record.levelname
        log_record['logger'] = record.name

class SystemInfoFilter(logging.Filter):
    """
    Filter that adds system info to log records
    """

    def __init__(
        self,
        app_name: str = DEFAULT_APP_NAME,
        environment: str = DEFAULT_ENVIRONMENT,
        version: str = DEFAULT_VERSION
    ):
        """
        Initialize the filter
        
        Args:
            app_name: Application name
            environment: Environment name
            version: Application version
        """
        super().__init__()
        self.hostname = socket.gethostname()
        self.app_name = app_name
        self.environment = environment
        self.version = version

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter the record
        
        Args:
            record: Log record
            
        Returns:
            True to include the record, False to exclude it
        """
        # Add system info
        record.hostname = self.hostname
        record.app_name = self.app_name
        record.environment = self.environment
        record.version = self.version
        
        # Always include the record
        return True


def get_log_level(level: Optional[str] = None) -> int:
    """
    Get log level from string
    
    Args:
        level: Log level string
        
    Returns:
        Log level integer
    """
    if level is None:
        # Try to get from config
        level = get_env_var("LOG_LEVEL", "INFO") if get_env_var else "INFO"
    
    # Convert to upper case
    level = level.upper()
    
    # Get log level
    if level == "DEBUG":
        return logging.DEBUG
    elif level == "INFO":
        return logging.INFO
    elif level == "WARNING":
        return logging.WARNING
    elif level == "ERROR":
        return logging.ERROR
    elif level == "CRITICAL":
        return logging.CRITICAL
    else:
        return logging.INFO


def get_log_to_console() -> bool:
    """
    Get log to console preference
    
    Returns:
        True if log to console, False otherwise
    """
    # Try to get from environment
    log_to_console = os.environ.get("LOG_TO_CONSOLE")
    if log_to_console is not None:
        return log_to_console.lower() == "true"
    
    # Try to get from config
    if config:
        log_to_console = config.get("LOG_TO_CONSOLE")
        if log_to_console is not None:
            return log_to_console.lower() == "true"
    
    # Default
    return DEFAULT_LOG_TO_CONSOLE


def get_log_to_file() -> bool:
    """
    Get log to file preference
    
    Returns:
        True if log to file, False otherwise
    """
    # Try to get from environment
    log_to_file = os.environ.get("LOG_TO_FILE")
    if log_to_file is not None:
        return log_to_file.lower() == "true"
    
    # Use get_env_var as fallback if available
    log_to_file = get_env_var("LOG_TO_FILE") if get_env_var else None
    if log_to_file is not None and isinstance(log_to_file, str):
        return log_to_file.lower() == "true"
    # Default
    return DEFAULT_LOG_TO_FILE


def get_log_dir() -> str:
    """
    Get log directory
    
    Returns:
        Log directory path
    """
    # Try to get from config
    log_dir = get_env_var("LOG_DIR", DEFAULT_LOG_DIR) if get_env_var else DEFAULT_LOG_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir


def get_log_file(filename: Optional[str] = None) -> str:
    """
    Get log file path
    
    Args:
        filename: Log filename
        
    Returns:
        Log file path
    """
    if filename is None:
        # Try to get from config
        filename = get_env_var("LOG_FILE", DEFAULT_LOG_FILE) if get_env_var else DEFAULT_LOG_FILE
    
    # Get log directory
    log_dir = get_log_dir()  # type: ignore
    
    # Create full path
    return os.path.join(log_dir, filename)


def configure_sentry(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    traces_sample_rate: Optional[float] = None,
    profiles_sample_rate: Optional[float] = None,
    debug: Optional[bool] = None,
    release: Optional[str] = None
) -> bool:
    """
    Configure Sentry for error tracking
    
    Args:
        dsn: Sentry DSN (if None, will try to get from environment)
        environment: Environment name (if None, will try to get from environment)
        traces_sample_rate: Traces sample rate
        profiles_sample_rate: Profiles sample rate
        
    Returns:
        True if Sentry was configured, False otherwise
    """
    if not SENTRY_AVAILABLE:
        return False
    
    # Try to import config
    try:
        # Import from the new relative path within ai_day_trader.utils
        from . import config
        config_available = True
    except ImportError:
        config_available = False
    
    # Check if Sentry is enabled
    if config_available and not config.get_bool("SENTRY_ENABLED", True):
        logger = logging.getLogger("sentry")
        logger.info("Sentry integration is disabled via configuration")
        return False
    
    # Get configuration values, with priority:
    # 1. Explicitly passed parameters
    # 2. Config module values
    # 3. Environment variables
    # 4. Default values
    
    # Get DSN
    if dsn is None and config_available:
        dsn = config.get("SENTRY_DSN")
    if dsn is None and get_env_var:
        dsn = get_env_var("SENTRY_DSN")
    
    # DSN is required - if not available, don't configure Sentry
    if not dsn:
        logger = logging.getLogger("sentry")
        logger.warning("Sentry DSN not provided, Sentry integration disabled")
        return False
    
    # Get environment
    if environment is None and config_available:
        environment = config.get("SENTRY_ENVIRONMENT")
    if environment is None and get_env_var:
        environment = os.environ.get("ENVIRONMENT") or get_env_var("ENVIRONMENT", DEFAULT_ENVIRONMENT) if get_env_var else DEFAULT_ENVIRONMENT
    
    # Get traces sample rate
    if traces_sample_rate is None and config_available:
        traces_sample_rate = config.get_float("SENTRY_TRACES_SAMPLE_RATE", 0.1)
    if traces_sample_rate is None:
        traces_sample_rate = 0.1
    
    # Get profiles sample rate
    if profiles_sample_rate is None and config_available:
        profiles_sample_rate = config.get_float("SENTRY_PROFILES_SAMPLE_RATE", 0.1)
    if profiles_sample_rate is None:
        profiles_sample_rate = 0.1
    
    # Get debug mode
    if debug is None and config_available:
        debug = config.get_bool("SENTRY_DEBUG", False)
    if debug is None:
        debug = False
    
    # Get release
    if release is None and config_available:
        release = config.get("SENTRY_RELEASE")
    if release is None and get_env_var:
        release = get_env_var("SENTRY_RELEASE", "1.0.0")
    
    # Configure Sentry logging integration
    logging_integration = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors as events
    )
    
    logger = logging.getLogger("sentry")
    logger.info(f"Initializing Sentry with environment: {environment}")
    
    # Initialize Sentry
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        debug=debug,
        release=release,
        integrations=[logging_integration],
        # Enable performance monitoring
        enable_tracing=True
    )
    
    logger.info("Sentry error tracking configured successfully")
    return True


def configure_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    use_json: Optional[bool] = None,
    console: Optional[bool] = None,
    file: Optional[bool] = None,
    app_name: Optional[str] = None,
    environment: Optional[str] = None,
    version: Optional[str] = None
) -> Dict[str, logging.Handler]:
    """
    Configure logging for the application
    
    Args:
        level: Log level
        log_file: Log file path
        log_format: Log format string
        use_json: Whether to use JSON formatting
        console: Whether to log to console
        file: Whether to log to file
        app_name: Application name
        environment: Environment name
        version: Application version
    """
    # Get log level
    if isinstance(level, str):
        level = get_log_level(level)  # type: ignore
    elif level is None:
        level = get_log_level()
    
    # Get log file
    if log_file is None:
        log_file = get_log_file()
    
    # Get log format
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT
    
    # Get JSON formatting preference
    if use_json is None:
        use_json = JSON_LOG_FORMAT
    
    # Create handlers
    handlers = {}
    
    # Console handler
    if console:
        if use_json:
            console_formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            console_formatter = logging.Formatter(log_format)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        handlers["console"] = console_handler
    
    # File handler
    if file and log_file:
        if use_json:
            file_formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            file_formatter = logging.Formatter(log_format)
        
        try:
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=DEFAULT_MAX_BYTES,
                backupCount=DEFAULT_BACKUP_COUNT
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            handlers["file"] = file_handler
        except Exception as e:
            print(f"Error creating file handler: {e}")
            # Continue without file handler
    
    # Create filters
    filters = [
        SystemInfoFilter(
            app_name=app_name or DEFAULT_APP_NAME,
            environment=environment or DEFAULT_ENVIRONMENT,
            version=version or DEFAULT_VERSION
        )
    ]
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers
    for handler in handlers.values():
        # Add filters
        for filter in filters:
            handler.addFilter(filter)
        
        # Add handler
        root_logger.addHandler(handler)
    
    return handlers


def get_logger(name: str) -> logging.Logger:
    """
    Get logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger
    """
    return logging.getLogger(name)


# Configure logging by default
configure_logging()

# Add alias for backward compatibility
# Some modules use configure_logger instead of configure_logging
configure_logger = configure_logging
