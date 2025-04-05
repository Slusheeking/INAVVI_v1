"""Tests for ai_day_trader.utils.exceptions"""
import pytest
# Import the base class as well
from ai_day_trader.utils.exceptions import TradingSystemError, TradingError, APIError

def test_exception_inheritance():
    """Test exception hierarchy."""
    try:
        raise APIError("Test API Error")
    # APIError inherits from TradingSystemError, not TradingError
    except TradingSystemError as e:
        # Optionally, assert it's the specific type we raised
        assert isinstance(e, APIError)
    except Exception:
        pytest.fail("Did not catch expected TradingSystemError")

# Add tests for specific exception properties or behaviors
