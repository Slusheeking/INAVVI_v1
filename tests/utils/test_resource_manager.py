"""Tests for ai_day_trader.utils.resource_manager"""
import pytest
from ai_day_trader.utils.resource_manager import ResourceManager, resource_managed

# TODO: Add tests for resource tracking and context manager
def test_resource_manager_init():
    """Test ResourceManager initialization."""
    manager = ResourceManager()
    assert manager is not None

@resource_managed("test_component")
def _dummy_function():
    return True

def test_resource_managed_decorator():
    """Test the resource_managed decorator."""
    assert _dummy_function() is True
