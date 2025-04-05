"""Tests for ai_day_trader.utils.config"""
import pytest
import os
from unittest.mock import patch
from ai_day_trader.utils.config import Config
from ai_day_trader.config import load_ai_trader_config # Import the specific loader
from ai_day_trader.utils.exceptions import ConfigurationError # Import custom exception

# Dummy .env content for testing
DUMMY_ENV_CONTENT = """
TRADING_POLYGON_API_KEY="dummy_poly_key"
TRADING_APCA_API_KEY_ID="dummy_apca_key"
TRADING_APCA_API_SECRET_KEY="dummy_apca_secret"
# Other required vars can be added if load_ai_trader_config fails without them
"""

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to mock environment variables."""
    # Mock essential keys to avoid raising errors during basic load
    monkeypatch.setenv("TRADING_POLYGON_API_KEY", "dummy_poly_key")
    monkeypatch.setenv("TRADING_APCA_API_KEY_ID", "dummy_apca_key")
    monkeypatch.setenv("TRADING_APCA_API_SECRET_KEY", "dummy_apca_secret")
    yield monkeypatch # Allow test to use monkeypatch further

def test_load_ai_trader_config_defaults(mock_env_vars):
    """Test loading config with default execution mode."""
    # Ensure EXECUTION_MODE is not set
    mock_env_vars.delenv("TRADING_EXECUTION_MODE", raising=False)
    config = load_ai_trader_config()
    assert config.EXECUTION_MODE == "live" # Default is live

def test_load_ai_trader_config_mode_live(mock_env_vars):
    """Test loading config with execution mode set to 'live'."""
    mock_env_vars.setenv("TRADING_EXECUTION_MODE", "live")
    config = load_ai_trader_config()
    assert config.EXECUTION_MODE == "live"

def test_load_ai_trader_config_mode_paper(mock_env_vars):
    """Test loading config with execution mode set to 'paper'."""
    mock_env_vars.setenv("TRADING_EXECUTION_MODE", "paper")
    config = load_ai_trader_config()
    assert config.EXECUTION_MODE == "paper"

def test_load_ai_trader_config_mode_case_insensitive(mock_env_vars):
    """Test loading config with execution mode set case-insensitively."""
    mock_env_vars.setenv("TRADING_EXECUTION_MODE", "PaPeR")
    config = load_ai_trader_config()
    assert config.EXECUTION_MODE == "paper"

def test_load_ai_trader_config_invalid_mode(mock_env_vars):
    """Test loading config with an invalid execution mode."""
    mock_env_vars.setenv("TRADING_EXECUTION_MODE", "invalid_mode")
    with pytest.raises(ConfigurationError, match="Invalid TRADING_EXECUTION_MODE"):
        load_ai_trader_config()

# TODO: Add more tests for other config variables and edge cases
