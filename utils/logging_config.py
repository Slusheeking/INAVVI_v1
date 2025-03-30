#!/usr/bin/env python3
"""
Logging Configuration Module

This module provides standardized logging configuration for the trading system:

1. Unified log formats
2. Common handlers and filters
3. Structured logging support
4. Log level management
5. Rotating file handlers with compression

All components in the trading system should use this module for logging
to ensure consistent formats and behavior.
"""

import atexit
import json
import logging
import logging.config
import logging.handlers
import os
import platform
import socket
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Import configuration module if available
try:
    from config import config, get_config
except ImportError:
    # Fallback if config module is not available
    config = None
    
    def get_config():
        return None


# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Default log directory
DEFAULT_LOG_DIR = "./logs"

# Default log file
DEFAULT_LOG_FILE = "trading_system.log"

# Maximum log file size (10 MB)
DEFAULT_MAX_BYTES = 10 * 1024 * 1024

# Maximum number of backup log files
DEFAULT_BACKUP_COUNT = 5

# Default log rotation interval (daily)
DEFAULT_ROTATION_INTERVAL = 'D'

# JSON log format
JSON_LOG_FORMAT = True


class StructuredLogRecord(logging.LogRecord):
    """
    Extended LogRecord class for structured logging
    
    This class adds additional fields to log records for structured logging,
    such as hostname, process ID, and thread ID.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.system = platform.system()
        self.python_version = platform.python_version()


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    
    This formatter outputs log records as JSON objects with additional
    metadata for easier parsing and analysis.
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = '%',
        validate: bool = True,
        *,
        defaults: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize JSON formatter"""
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.hostname = socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "path": record.pathname,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add hostname if available
        if hasattr(record, 'hostname'):
            log_data["hostname"] = record.hostname
        else:
            log_data["hostname"] = self.hostname
        
        # Add system info if available
        if hasattr(record, 'system'):
            log_data["system"] = record.system
        
        # Add Python version if available
        if hasattr(record, 'python_version'):
            log_data["python_version"] = record.python_version
        
        # Add exception info if available
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_data["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": traceback.format_exception(exc_type, exc_value, exc_tb)
            }
        
        # Add extra attributes
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_') and key not in (
                'args', 'exc_info', 'exc_text', 'stack_info', 'created',
                'msecs', 'relativeCreated', 'levelno', 'msg'
            ):
                try:
                    # Try to serialize value to JSON
                    json.dumps({key: value})
                    log_data[key] = value  # type: ignore
                except (TypeError, OverflowError):
                    # Skip values that can't be serialized
                    log_data[key] = str(value)  # type: ignore
        
        return json.dumps(log_data)


class ExtraFieldsFilter(logging.Filter):
    """
    Filter that adds extra fields to log records
    
    This filter adds additional fields to log records, such as
    application name, environment, and version.
    """
    
    def __init__(
        self,
        name: str = "",
        app_name: str = "trading_system",
        environment: str = "production",
        version: str = "1.0.0"
    ) -> None:
        """
        Initialize filter with extra fields
        
        Args:
            name: Filter name
            app_name: Application name
            environment: Environment name
            version: Application version
        """
        super().__init__(name)
        self.app_name = app_name
        self.environment = environment
        self.version = version
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add extra fields to log record"""
        record.app_name = self.app_name
        record.environment = self.environment
        record.version = self.version
        return True


class SensitiveDataFilter(logging.Filter):
    """
    Filter that masks sensitive data in log records
    
    This filter masks sensitive data such as API keys, passwords,
    and other credentials in log messages.
    """
    
    def __init__(
        self,
        name: str = "",
        patterns: Optional[List[str]] = None,
        replacement: str = "********"
    ) -> None:
        """
        Initialize filter with patterns to mask
        
        Args:
            name: Filter name
            patterns: List of patterns to mask
            replacement: Replacement string for masked data
        """
        super().__init__(name)
        self.patterns = patterns or [
            "api_key", "apikey", "password", "secret", "token",
            "credential", "auth", "access_key", "private_key"
        ]
        self.replacement = replacement
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Mask sensitive data in log message"""
        if isinstance(record.msg, str):
            for pattern in self.patterns:  # type: ignore
                # Look for pattern in query parameters
                record.msg = self._mask_pattern_in_url(record.msg, pattern)
                
                # Look for pattern in JSON or key-value pairs
                record.msg = self._mask_pattern_in_text(record.msg, pattern)
        
        # Also check args if they are strings
        if record.args:
            args = list(record.args)  # type: ignore
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    for pattern in self.patterns:  # type: ignore
                        args[i] = self._mask_pattern_in_url(arg, pattern)
                        args[i] = self._mask_pattern_in_text(arg, pattern)
            record.args = tuple(args)
        
        return True
    
    def _mask_pattern_in_url(self, text: str, pattern: str) -> str:
        """Mask pattern in URL query parameters"""
        import re
        
        # Match pattern in URL query parameters
        # e.g., api_key=abc123 or apikey=abc123
        regex = re.compile(
            f"({pattern}=)([^&\s]+)",
            re.IGNORECASE
        )
        
        return regex.sub(f"\\1{self.replacement}", text)
    
    def _mask_pattern_in_text(self, text: str, pattern: str) -> str:
        """Mask pattern in text (JSON or key-value pairs)"""
        import re
        
        # Match pattern in JSON or key-value pairs
        # e.g., "api_key": "abc123" or "apikey":"abc123"
        regex = re.compile(
            f"(['\"]?{pattern}['\"]?\\s*[:=]\\s*['\"]?)([^'\",\\s]+)(['\"]?)",
            re.IGNORECASE
        )
        
        return regex.sub(f"\\1{self.replacement}\\3", text)


def get_log_level(level_name: Optional[str] = None) -> int:
    """
    Get log level from name
    
    Args:
        level_name: Log level name
        
    Returns:
        Log level as integer
    """
    if level_name is None:
        # Try to get from config
        if config:
            level_name = config.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
        else:
            level_name = DEFAULT_LOG_LEVEL
    
    # Convert to uppercase
    level_name = str(level_name).upper()
    
    # Get log level
    level = getattr(logging, level_name, None)
    if level is None:
        # Default to INFO if invalid level
        level = logging.INFO
    
    return level


def get_log_dir() -> str:
    """
    Get log directory
    
    Returns:
        Log directory path
    """
    # Try to get from config
    if config:
        log_dir = config.get("LOG_DIR", DEFAULT_LOG_DIR)  # type: ignore
    else:
        log_dir = DEFAULT_LOG_DIR
    
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
        if config:
            filename = config.get("LOG_FILE", DEFAULT_LOG_FILE)
        else:
            filename = DEFAULT_LOG_FILE
    
    # Get log directory
    log_dir = get_log_dir()  # type: ignore
    
    # Create full path
    return os.path.join(log_dir, filename)


def configure_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    use_json: Optional[bool] = None,
    console: bool = True,
    file: bool = True,
    app_name: str = "trading_system",
    environment: str = "production",
    version: str = "1.0.0"
) -> None:
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
            console_formatter = JSONFormatter()
        else:
            console_formatter = logging.Formatter(log_format)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        handlers["console"] = console_handler
    
    # File handler
    if file and log_file:
        if use_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(log_format)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(  # type: ignore
            log_file,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        handlers["file"] = file_handler
    
    # Create filters
    filters = [
        ExtraFieldsFilter(
            app_name=app_name,
            environment=environment,
            version=version
        ),
        SensitiveDataFilter()
    ]
    
    # Apply filters to handlers
    for handler in handlers.values():
        for filter_obj in filters:
            handler.addFilter(filter_obj)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers.values():  # type: ignore
        root_logger.addHandler(handler)
    
    # Set custom log record factory
    logging.setLogRecordFactory(StructuredLogRecord)
    
    # Register atexit handler to flush logs
    atexit.register(logging.shutdown)
    
    # Log configuration
    logging.info(
        f"Logging configured: level={logging.getLevelName(level)}, "
        f"console={console}, file={file}, json={use_json}"
    )


def get_logger(
    name: str,
    level: Optional[Union[int, str]] = None,
    propagate: bool = True
) -> logging.Logger:
    """
    Get logger with standard configuration
    
    Args:
        name: Logger name
        level: Log level
        propagate: Whether to propagate to parent loggers
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Set level if specified
    if level is not None:
        if isinstance(level, str):
            level = get_log_level(level)
        logger.setLevel(level)
    
    # Set propagation
    logger.propagate = propagate
    
    return logger


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    level: int = logging.ERROR,
    message: Optional[str] = None
) -> None:
    """
    Log exception with traceback
    
    Args:
        logger: Logger to use
        exc: Exception to log
        level: Log level
        message: Optional message to include
    """
    if message:
        logger.log(level, message, exc_info=exc)
    else:
        logger.log(level, f"Exception: {exc}", exc_info=exc)


def log_dict(
    logger: logging.Logger,
    data: Dict[str, Any],
    level: int = logging.INFO,
    message: Optional[str] = None
) -> None:
    """
    Log dictionary as JSON
    
    Args:
        logger: Logger to use
        data: Dictionary to log
        level: Log level
        message: Optional message to include
    """
    json_str = json.dumps(data, indent=2, default=str)
    
    if message:
        logger.log(level, f"{message}: {json_str}")
    else:
        logger.log(level, json_str)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter with extra fields
    
    This adapter adds extra fields to log records, such as
    component name, request ID, and user ID.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize adapter with extra fields
        
        Args:
            logger: Logger to adapt
            extra: Extra fields to add to log records
        """
        super().__init__(logger, extra or {})
    
    def process(
        self,
        msg: Any,
        kwargs: Dict[str, Any]
    ) -> tuple[Any, Dict[str, Any]]:
        """Process log record with extra fields"""
        # Ensure extra dict exists
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add adapter extra fields
        for key, value in self.extra.items():
            kwargs['extra'][key] = value
        
        return msg, kwargs


def get_component_logger(
    component_name: str,
    level: Optional[Union[int, str]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> LoggerAdapter:
    """
    Get logger for a component with standard configuration
    
    Args:
        component_name: Component name
        level: Log level
        extra: Extra fields to add to log records
        
    Returns:
        Configured logger adapter
    """
    logger = get_logger(component_name, level)
    
    # Create extra fields
    extra_fields = {'component': component_name}
    if extra:
        extra_fields.update(extra)
    
    return LoggerAdapter(logger, extra_fields)


def get_request_logger(
    component_name: str,
    request_id: str,
    user_id: Optional[str] = None,
    level: Optional[Union[int, str]] = None
) -> LoggerAdapter:
    """
    Get logger for a request with standard configuration
    
    Args:
        component_name: Component name
        request_id: Request ID
        user_id: User ID
        level: Log level
        
    Returns:
        Configured logger adapter
    """
    logger = get_logger(component_name, level)
    
    # Create extra fields
    extra_fields = {
        'component': component_name,
        'request_id': request_id
    }
    
    if user_id:
        extra_fields['user_id'] = user_id
    
    return LoggerAdapter(logger, extra_fields)


# Configure logging with default settings
configure_logging()