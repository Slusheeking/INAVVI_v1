"""Tests for ai_day_trader.peak_detector"""
import pytest
import logging
import numpy as np
from ai_day_trader.peak_detector import PeakDetector, PeakDetectionConfig
from ai_day_trader.config import load_ai_trader_config

@pytest.fixture
def peak_detector():
    """Provides an initialized PeakDetector instance."""
    config = load_ai_trader_config()
    logger_instance = logging.getLogger("test_peak_detector")
    # Ensure necessary config values exist or provide defaults
    # config._config['PEAK_DETECTION_USE_GPU'] = False # Force CPU for testing if needed
    detector = PeakDetector(config=config, logger_instance=logger_instance)
    return detector

def test_find_peaks_simple(peak_detector: PeakDetector):
    """Test finding peaks in a simple sine wave."""
    prices = np.sin(np.linspace(0, 4 * np.pi, 100)).tolist()
    # Expected peaks around indices 25 and 75
    peaks = peak_detector.find_peaks(prices, symbol="SINE")
    assert isinstance(peaks, list)
    assert len(peaks) > 0
    # Add more specific assertions based on expected indices

def test_find_troughs_simple(peak_detector: PeakDetector):
    """Test finding troughs in a simple sine wave."""
    prices = np.sin(np.linspace(0, 4 * np.pi, 100)).tolist()
    # Expected troughs around indices 50
    troughs = peak_detector.find_troughs(prices, symbol="SINE")
    assert isinstance(troughs, list)
    assert len(troughs) > 0
    # Add more specific assertions based on expected indices

# TODO: Add tests for edge cases (flat line, noisy data), config overrides, GPU usage
