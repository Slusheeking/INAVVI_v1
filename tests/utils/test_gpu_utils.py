"""Tests for ai_day_trader.utils.gpu_utils"""
import pytest
from ai_day_trader.utils.gpu_utils import is_gpu_available, get_device_info

def test_gpu_availability_check():
    """Test the GPU availability check function."""
    # This test depends on the actual hardware environment
    available = is_gpu_available()
    assert isinstance(available, bool)
    print(f"GPU Available: {available}")

def test_get_device_info():
    """Test getting device info."""
    info = get_device_info()
    assert isinstance(info, dict)
    assert "use_gpu" in info
    assert "frameworks" in info
    print(f"Device Info: {info}")

# Add more tests, potentially mocking torch/cupy imports
