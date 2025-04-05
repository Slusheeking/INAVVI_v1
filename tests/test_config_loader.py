"""Tests for ai_day_trader.config loader"""
import pytest
import os
from ai_day_trader.config import load_ai_trader_config

def test_load_config_defaults():
    """Test loading config with default values (no .env)."""
    # Ensure no .env file exists for this test or mock os.environ
    config = load_ai_trader_config(env_file=".nonexistent")
    # Check attributes directly, as BaseConfig.__init__ sets them
    assert config.redis_host == "localhost"
    assert config.PORTFOLIO_SIZE == 100000.0 # This one is uppercase in the class definition
    # Cannot reliably assert polygon_api_key is None here, as it might be set in the global environment
    # Add more default checks

# TODO: Add tests for loading from a mock .env file
