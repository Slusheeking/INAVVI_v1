"""Tests for ai_day_trader.utils.logging_config"""
import logging
import pytest
from ai_day_trader.utils.logging_config import configure_logging, get_logger # Correct function name

def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"

# Add tests for setup_logging configuration effects (might need mocking)
