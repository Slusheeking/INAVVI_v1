#!/usr/bin/env python3
"""
Configuration Management Module

This module provides standardized configuration management for the trading system:

1. Unified environment variable handling
2. Type conversion and validation
3. Default values management
4. Configuration hierarchies with overrides
5. Validation of required vs optional settings

All components in the trading system should use this module for configuration
to ensure consistent behavior and validation.
"""

import json
import logging
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

# Import exceptions module
try:
    from exceptions import (
        ConfigurationError,
        EnvironmentVariableError,
        InvalidConfigurationError,
        MissingConfigurationError,
    )
except ImportError:
    # Fallback if exceptions module is not available
    class ConfigurationError(Exception):
        """Base class for configuration errors"""
        pass
    
    class EnvironmentVariableError(ConfigurationError):
        """Error related to environment variables"""
        pass
    
    class InvalidConfigurationError(ConfigurationError):
        """Invalid configuration value or structure"""
        pass
    
    class MissingConfigurationError(ConfigurationError):
        """Required configuration is missing"""
        pass

# Configure logging
logger = logging.getLogger("config")

# Type variable for generic type hints
T = TypeVar('T')


class ConfigSource(Enum):
    """Enumeration of configuration sources with priority order"""
    DEFAULT = 0
    CONFIG_FILE = 1
    ENVIRONMENT = 2
    COMMAND_LINE = 3
    RUNTIME = 4


class ConfigValue:
    """
    Configuration value with metadata
    
    This class represents a configuration value with additional metadata such as
    source, description, and validation rules.
    """
    
    def __init__(
        self,
        value: Any,
        source: ConfigSource = ConfigSource.DEFAULT,
        description: Optional[str] = None,
        required: bool = False,
        secret: bool = False
    ) -> None:
        """
        Initialize configuration value
        
        Args:
            value: The configuration value
            source: Source of the configuration value
            description: Description of the configuration value
            required: Whether the configuration value is required
            secret: Whether the configuration value is sensitive and should be masked in logs
        """
        self.value = value
        self.source = source
        self.description = description
        self.required = required
        self.secret = secret
    
    def __str__(self) -> str:
        """String representation of configuration value"""
        if self.secret and self.value:
            # Mask sensitive values
            return "********"
        return str(self.value)
    
    def __repr__(self) -> str:
        """Detailed representation of configuration value"""
        value_str = "********" if self.secret and self.value else repr(self.value)
        return f"ConfigValue({value_str}, source={self.source}, required={self.required})"


class Config:
    """
    Configuration manager for the trading system
    
    This class provides a centralized configuration system with support for:
    - Multiple configuration sources with priority
    - Type conversion and validation
    - Default values
    - Required vs optional settings
    - Configuration hierarchies
    """
    
    def __init__(
        self,
        prefix: str = "TRADING",
        config_file: Optional[str] = None,
        load_env: bool = True,
        case_sensitive: bool = False
    ) -> None:
        """
        Initialize configuration manager
        
        Args:
            prefix: Prefix for environment variables
            config_file: Path to configuration file
            load_env: Whether to load environment variables
            case_sensitive: Whether keys are case-sensitive
        """
        self.prefix = prefix
        self.config_file = config_file
        self.case_sensitive = case_sensitive
        self._config: Dict[str, ConfigValue] = {}
        self._required_keys: Set[str] = set()
        
        # Load configuration from file if specified
        if config_file:
            self._load_config_file(config_file)
        
        # Load environment variables if requested
        if load_env:
            self._load_environment_variables()
        
        logger.info(f"Configuration initialized with prefix '{prefix}'")
    
    def _normalize_key(self, key: str) -> str:
        """
        Normalize configuration key based on case sensitivity
        
        Args:
            key: Configuration key
            
        Returns:
            Normalized key
        """
        if not self.case_sensitive:
            return key.lower()
        return key
    
    def _load_config_file(self, config_file: str) -> None:
        """
        Load configuration from file
        
        Args:
            config_file: Path to configuration file
        """
        try:
            path = Path(config_file)
            if not path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return
            
            # Determine file format from extension
            if path.suffix.lower() in ('.json', '.jsn'):
                with open(path, 'r') as f:
                    config_data = json.load(f)
            elif path.suffix.lower() in ('.py'):
                # Load Python module
                import importlib.util
                spec = importlib.util.spec_from_file_location("config_module", path)
                if spec and spec.loader:
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    # Extract uppercase variables as config
                    config_data = {
                        k: v for k, v in vars(config_module).items()
                        if k.isupper() and not k.startswith('_')
                    }
                else:
                    raise ConfigurationError(f"Could not load Python config file: {path}")
            else:
                # Assume simple key=value format
                config_data = {}
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, value = line.split('=', 1)
                            config_data[key.strip()] = value.strip()
            
            # Add configuration values
            for key, value in config_data.items():
                normalized_key = self._normalize_key(key)
                # Don't override existing values from higher priority sources
                if normalized_key not in self._config:
                    self._config[normalized_key] = ConfigValue(
                        value, source=ConfigSource.CONFIG_FILE
                    )
            
            logger.info(f"Loaded configuration from file: {config_file}")
        
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
            raise ConfigurationError(f"Error loading configuration file: {e}")
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables"""
        try:
            prefix_pattern = f"^{self.prefix}_"
            for key, value in os.environ.items():
                # Check if variable starts with prefix
                if re.match(prefix_pattern, key):
                    # Remove prefix and normalize
                    config_key = re.sub(prefix_pattern, "", key)
                    normalized_key = self._normalize_key(config_key)
                    
                    # Add to configuration with high priority
                    self._config[normalized_key] = ConfigValue(
                        value, source=ConfigSource.ENVIRONMENT
                    )
            
            logger.info(f"Loaded configuration from environment variables with prefix '{self.prefix}_'")
        
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
            raise EnvironmentVariableError(f"Error loading environment variables: {e}")
    
    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.RUNTIME,
        description: Optional[str] = None,
        required: bool = False,
        secret: bool = False
    ) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
            source: Source of the configuration value
            description: Description of the configuration value
            required: Whether the configuration value is required
            secret: Whether the configuration value is sensitive
        """
        normalized_key = self._normalize_key(key)
        
        # Check if value already exists with higher priority
        if normalized_key in self._config:
            existing = self._config[normalized_key]
            if existing.source.value >= source.value:
                logger.debug(
                    f"Not overriding configuration '{key}' from {existing.source} with {source}"
                )
                return
        
        # Set the value
        self._config[normalized_key] = ConfigValue(
            value, source=source, description=description, required=required, secret=secret
        )
        
        # Add to required keys if needed
        if required:
            self._required_keys.add(normalized_key)
        
        logger.debug(f"Set configuration '{key}' = {value} (source: {source})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        normalized_key = self._normalize_key(key)
        
        if normalized_key in self._config:
            return self._config[normalized_key].value
        
        return default
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get configuration value as integer
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value as integer or default
            
        Raises:
            InvalidConfigurationError: If value cannot be converted to integer
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        try:
            return int(value)
        except (ValueError, TypeError):
            raise InvalidConfigurationError(
                f"Configuration '{key}' value '{value}' cannot be converted to integer"
            )
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get configuration value as float
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value as float or default
            
        Raises:
            InvalidConfigurationError: If value cannot be converted to float
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            raise InvalidConfigurationError(
                f"Configuration '{key}' value '{value}' cannot be converted to float"
            )
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Get configuration value as boolean
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value as boolean or default
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', 'y', '1', 'on')
        
        return bool(value)
    
    def get_list(
        self,
        key: str,
        default: Optional[List[Any]] = None,
        item_type: Optional[Type[T]] = None
    ) -> Optional[List[Any]]:
        """
        Get configuration value as list
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            item_type: Type to convert list items to
            
        Returns:
            Configuration value as list or default
            
        Raises:
            InvalidConfigurationError: If value cannot be converted to list
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        # Already a list
        if isinstance(value, list):
            result = value
        # Convert string to list
        elif isinstance(value, str):
            # Handle JSON array
            if value.startswith('[') and value.endswith(']'):
                try:
                    result = json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to comma-separated values
                    result = [item.strip() for item in value.split(',')]
            else:
                # Comma-separated values
                result = [item.strip() for item in value.split(',')]
        # Convert other types to list
        else:
            try:
                result = list(value)
            except (TypeError, ValueError):
                result = [value]
        
        # Convert items to specified type
        if item_type is not None:
            try:
                result = [item_type(item) for item in result]
            except (ValueError, TypeError):
                raise InvalidConfigurationError(
                    f"Configuration '{key}' list items cannot be converted to {item_type.__name__}"
                )
        
        return result
    
    def get_dict(
        self,
        key: str,
        default: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get configuration value as dictionary
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value as dictionary or default
            
        Raises:
            InvalidConfigurationError: If value cannot be converted to dictionary
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        # Already a dictionary
        if isinstance(value, dict):
            return value
        
        # Try to parse JSON
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise InvalidConfigurationError(
                    f"Configuration '{key}' value '{value}' cannot be converted to dictionary"
                )
        
        raise InvalidConfigurationError(
            f"Configuration '{key}' value '{value}' cannot be converted to dictionary"
        )
    
    def get_path(
        self,
        key: str,
        default: Optional[str] = None,
        must_exist: bool = False
    ) -> Optional[Path]:
        """
        Get configuration value as Path
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            must_exist: Whether the path must exist
            
        Returns:
            Configuration value as Path or default
            
        Raises:
            InvalidConfigurationError: If path does not exist and must_exist is True
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        path = Path(value)
        
        if must_exist and not path.exists():
            raise InvalidConfigurationError(
                f"Configuration '{key}' path '{path}' does not exist"
            )
        
        return path
    
    def get_enum(
        self,
        key: str,
        enum_class: Type[Enum],
        default: Optional[Enum] = None
    ) -> Optional[Enum]:
        """
        Get configuration value as Enum
        
        Args:
            key: Configuration key
            enum_class: Enum class to convert to
            default: Default value if key is not found
            
        Returns:
            Configuration value as Enum or default
            
        Raises:
            InvalidConfigurationError: If value cannot be converted to Enum
        """
        value = self.get(key, default)
        
        if value is None:
            return None
        
        # Already an enum of the correct type
        if isinstance(value, enum_class):
            return value
        
        # Try to convert string to enum
        try:
            # Try by name
            if isinstance(value, str):
                return enum_class[value]
        except KeyError:
            pass
        
        try:
            # Try by value
            return enum_class(value)
        except (ValueError, TypeError):
            valid_values = [e.name for e in enum_class] + [e.value for e in enum_class]
            raise InvalidConfigurationError(
                f"Configuration '{key}' value '{value}' is not a valid {enum_class.__name__}. "
                f"Valid values: {valid_values}"
            )
    
    def get_with_meta(self, key: str) -> Optional[ConfigValue]:
        """
        Get configuration value with metadata
        
        Args:
            key: Configuration key
            
        Returns:
            ConfigValue object or None if key is not found
        """
        normalized_key = self._normalize_key(key)
        return self._config.get(normalized_key)
    
    def require(
        self,
        key: str,
        description: Optional[str] = None,
        secret: bool = False
    ) -> None:
        """
        Mark a configuration key as required
        
        Args:
            key: Configuration key
            description: Description of the configuration value
            secret: Whether the configuration value is sensitive
        """
        normalized_key = self._normalize_key(key)
        self._required_keys.add(normalized_key)
        
        # Update metadata if key exists
        if normalized_key in self._config:
            self._config[normalized_key].required = True
            if description:
                self._config[normalized_key].description = description
            if secret:
                self._config[normalized_key].secret = secret
        
        logger.debug(f"Marked configuration '{key}' as required")
    
    def validate(self) -> None:
        """
        Validate configuration
        
        Raises:
            MissingConfigurationError: If required configuration is missing
        """
        missing_keys = []
        
        for key in self._required_keys:
            if key not in self._config or self._config[key].value is None:
                # Convert normalized key back to original format for error message
                original_key = key.upper() if not self.case_sensitive else key
                missing_keys.append(original_key)
        
        if missing_keys:
            error_msg = f"Missing required configuration: {', '.join(missing_keys)}"
            logger.error(error_msg)
            raise MissingConfigurationError(error_msg)
        
        logger.info("Configuration validation successful")
    
    def as_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Args:
            include_secrets: Whether to include secret values
            
        Returns:
            Dictionary of configuration values
        """
        result = {}
        
        for key, config_value in self._config.items():
            if config_value.secret and not include_secrets:
                result[key] = "********"
            else:
                result[key] = config_value.value
        
        return result
    
    def load_defaults(self, defaults: Dict[str, Any]) -> None:
        """
        Load default configuration values
        
        Args:
            defaults: Dictionary of default values
        """
        for key, value in defaults.items():
            # Extract metadata if value is a dictionary with special keys
            if isinstance(value, dict) and '__meta__' in value:
                meta = value.pop('__meta__')
                actual_value = value.pop('value', None)
                
                description = meta.get('description')
                required = meta.get('required', False)
                secret = meta.get('secret', False)
                
                self.set(
                    key,
                    actual_value,
                    source=ConfigSource.DEFAULT,
                    description=description,
                    required=required,
                    secret=secret
                )
            else:
                # Simple value
                self.set(key, value, source=ConfigSource.DEFAULT)
        
        logger.info(f"Loaded {len(defaults)} default configuration values")
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key"""
        normalized_key = self._normalize_key(key)
        return normalized_key in self._config
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax"""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using dictionary syntax"""
        self.set(key, value)


# Create default configuration instance
config = Config()


# Default configuration values for the trading system
DEFAULT_CONFIG = {
    # API Keys
    "POLYGON_API_KEY": {
        "value": "",
        "__meta__": {
            "description": "Polygon.io API key",
            "required": True,
            "secret": True
        }
    },
    "UNUSUAL_WHALES_API_KEY": {
        "value": "",
        "__meta__": {
            "description": "Unusual Whales API key",
            "required": True,
            "secret": True
        }
    },
    
    # Redis Configuration
    "REDIS_HOST": {
        "value": "localhost",
        "__meta__": {
            "description": "Redis host"
        }
    },
    "REDIS_PORT": {
        "value": 6379,
        "__meta__": {
            "description": "Redis port"
        }
    },
    "REDIS_DB": {
        "value": 0,
        "__meta__": {
            "description": "Redis database number"
        }
    },
    "REDIS_USERNAME": {
        "value": "default",
        "__meta__": {
            "description": "Redis username"
        }
    },
    "REDIS_PASSWORD": {
        "value": "",
        "__meta__": {
            "description": "Redis password",
            "secret": True
        }
    },
    "REDIS_SSL": {
        "value": False,
        "__meta__": {
            "description": "Whether to use SSL for Redis connection"
        }
    },
    "REDIS_TIMEOUT": {
        "value": 5,
        "__meta__": {
            "description": "Redis connection timeout in seconds"
        }
    },
    
    # Connection Settings
    "MAX_RETRIES": {
        "value": 3,
        "__meta__": {
            "description": "Maximum number of retry attempts"
        }
    },
    "RETRY_BACKOFF_FACTOR": {
        "value": 0.5,
        "__meta__": {
            "description": "Backoff factor for retries"
        }
    },
    "CONNECTION_TIMEOUT": {
        "value": 15,
        "__meta__": {
            "description": "Connection timeout in seconds"
        }
    },
    "MAX_POOL_SIZE": {
        "value": 30,
        "__meta__": {
            "description": "Maximum connection pool size"
        }
    },
    "RECONNECT_DELAY": {
        "value": 2.0,
        "__meta__": {
            "description": "Delay between reconnection attempts in seconds"
        }
    },
    "MAX_RECONNECT_ATTEMPTS": {
        "value": 10,
        "__meta__": {
            "description": "Maximum number of reconnection attempts"
        }
    },
    
    # Cache Settings
    "POLYGON_CACHE_TTL": {
        "value": 3600,
        "__meta__": {
            "description": "Polygon.io cache TTL in seconds"
        }
    },
    "UNUSUAL_WHALES_CACHE_TTL": {
        "value": 300,
        "__meta__": {
            "description": "Unusual Whales cache TTL in seconds"
        }
    },
    
    # Processing Settings
    "USE_GPU": {
        "value": True,
        "__meta__": {
            "description": "Whether to use GPU acceleration if available"
        }
    },
    "USE_GH200": {
        "value": True,
        "__meta__": {
            "description": "Whether to use GH200-specific optimizations if available"
        }
    },
    "BUFFER_SIZE": {
        "value": 1000,
        "__meta__": {
            "description": "Buffer size for data processing"
        }
    },
    
    # Logging Settings
    "LOG_LEVEL": {
        "value": "INFO",
        "__meta__": {
            "description": "Logging level"
        }
    },
    "LOG_FILE": {
        "value": "",
        "__meta__": {
            "description": "Log file path (empty for stdout only)"
        }
    },
    
    # Metrics Settings
    "METRICS_ENABLED": {
        "value": True,
        "__meta__": {
            "description": "Whether to enable Prometheus metrics"
        }
    },
    "METRICS_PORT": {
        "value": 9090,
        "__meta__": {
            "description": "Port for Prometheus metrics server"
        }
    },
    
    # Directory Settings
    "DATA_DIR": {
        "value": "./data",
        "__meta__": {
            "description": "Directory for data files"
        }
    },
    "CACHE_DIR": {
        "value": "./cache",
        "__meta__": {
            "description": "Directory for cache files"
        }
    },
    "MODEL_DIR": {
        "value": "./models",
        "__meta__": {
            "description": "Directory for model files"
        }
    },
    "LOG_DIR": {
        "value": "./logs",
        "__meta__": {
            "description": "Directory for log files"
        }
    }
}

# Load default configuration
config.load_defaults(DEFAULT_CONFIG)


def get_config() -> Config:
    """
    Get the global configuration instance
    
    Returns:
        Global configuration instance
    """
    return config


def initialize_config(
    prefix: str = "TRADING",
    config_file: Optional[str] = None,
    defaults: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Initialize configuration with custom settings
    
    Args:
        prefix: Prefix for environment variables
        config_file: Path to configuration file
        defaults: Dictionary of default values
        
    Returns:
        Initialized configuration instance
    """
    global config
    
    # Create new configuration instance
    config = Config(prefix=prefix, config_file=config_file)
    
    # Load defaults
    if defaults:
        config.load_defaults(defaults)
    else:
        config.load_defaults(DEFAULT_CONFIG)
    
    logger.info(f"Initialized configuration with prefix '{prefix}'")
    return config