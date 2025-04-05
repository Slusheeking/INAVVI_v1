"""Peak detection algorithms for price analysis."""
import logging
import time
import contextlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Callable

import numpy as np
import torch
from scipy.signal import find_peaks

# Import from new structure
from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.utils.metrics_registry import register_counter, register_gauge, register_histogram, MetricPrefix # Use new utils path
from ai_day_trader.utils.resource_manager import resource_managed # Use new utils path
from ai_day_trader.utils.exceptions import DataError, ModelInferenceError # Use new utils path
from ai_day_trader.utils.gpu_utils import is_gpu_available # Use new utils path

# Define metrics for peak detection
PEAK_DETECTION_TIME = register_histogram(
    MetricPrefix.TRADING,
    "peak_detection_time_seconds",
    "Time spent on peak detection",
    ["operation"]
)

PEAK_DETECTION_COUNT = register_counter(
    MetricPrefix.TRADING,
    "peak_detection_count",
    "Number of peaks detected",
    ["symbol", "type"]
)

PEAK_DETECTION_MEMORY = register_gauge(
    MetricPrefix.TRADING,
    "peak_detection_memory_usage_bytes",
    "Memory usage during peak detection",
    ["device"]
)

@dataclass
class PeakDetectionConfig:
    """Configuration for peak detection algorithms."""
    min_peak_height: float = 0.03
    min_trough_depth: float = 0.03
    smoothing_window: int = 5
    use_gpu: bool = True
    # cache_ttl_seconds: int = 300 # Caching removed for simplicity, can be added back if needed
    max_data_points: int = 10000
    prominence_threshold: float = 0.01
    distance_threshold: int = 5

class PeakDetector:
    """Detects peaks and troughs in price data with proper resource management."""

    def __init__(self, config: Config, logger_instance: logging.Logger):
        """
        Initialize the peak detector.

        Args:
            config: The application configuration object.
            logger_instance: The logger instance to use.
        """
        self.config_source = config # Store the main config source
        self.logger = logger_instance
        self.config = self._load_config() # Load specific config

        # Initialize device based on config and availability
        self.use_gpu = self.config.use_gpu and is_gpu_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.logger.info(f"Peak detector using device: {self.device}")

    def _load_config(self) -> PeakDetectionConfig:
        """Load configuration from the main config object."""
        try:
            return PeakDetectionConfig(
                min_peak_height=self.config_source.get_float('PEAK_DETECTION_MIN_HEIGHT', 0.03),
                min_trough_depth=self.config_source.get_float('PEAK_DETECTION_MIN_DEPTH', 0.03),
                smoothing_window=self.config_source.get_int('PEAK_DETECTION_SMOOTHING_WINDOW', 5),
                use_gpu=self.config_source.get_bool('PEAK_DETECTION_USE_GPU', True),
                # cache_ttl_seconds=self.config_source.get_int('PEAK_DETECTION_CACHE_TTL', 300),
                max_data_points=self.config_source.get_int('PEAK_DETECTION_MAX_DATA_POINTS', 10000),
                prominence_threshold=self.config_source.get_float('PEAK_DETECTION_PROMINENCE', 0.01),
                distance_threshold=self.config_source.get_int('PEAK_DETECTION_DISTANCE', 5)
            )
        except Exception as e:
            self.logger.warning(f"Error loading peak detection config: {e}. Using defaults.")
            return PeakDetectionConfig()

    @resource_managed("peak_detection")
    def find_peaks(self, prices: List[float], symbol: str = "unknown", min_height: Optional[float] = None) -> List[int]:
        """Find price peaks."""
        start_time = time.time()
        if not prices: raise DataError("Empty price list provided to find_peaks")
        if len(prices) > self.config.max_data_points:
            self.logger.warning(f"Truncating price data from {len(prices)} to {self.config.max_data_points} points for {symbol}")
            prices = prices[-self.config.max_data_points:]

        height_threshold = min_height if min_height is not None else self.config.min_peak_height

        try:
            with self._gpu_resource_context():
                prices_tensor = torch.tensor(prices, dtype=torch.float32).to(self.device)
                smoothed = self._smooth_prices(prices_tensor)
                smoothed_np = smoothed.cpu().numpy()

                # Calculate height relative to max price in the smoothed window
                max_price_in_window = np.max(smoothed_np)
                if max_price_in_window <= 0: max_price_in_window = 1e-6 # Avoid zero

                peaks, _ = find_peaks(
                    smoothed_np,
                    height=height_threshold * max_price_in_window,
                    prominence=self.config.prominence_threshold * max_price_in_window,
                    distance=self.config.distance_threshold
                )

                execution_time = time.time() - start_time
                PEAK_DETECTION_TIME.labels(operation="find_peaks").observe(execution_time)
                PEAK_DETECTION_COUNT.labels(symbol=symbol, type="peak").inc(len(peaks))
                self.logger.debug(f"Found {len(peaks)} peaks for {symbol} in {execution_time:.3f}s")
                return peaks.tolist()

        except Exception as e:
            self.logger.error(f"Error in peak detection for {symbol}: {str(e)}", exc_info=True)
            raise ModelInferenceError(f"Peak detection failed: {str(e)}") from e
        finally:
            if self.use_gpu: torch.cuda.empty_cache()

    @resource_managed("trough_detection")
    def find_troughs(self, prices: List[float], symbol: str = "unknown", min_depth: Optional[float] = None) -> List[int]:
        """Find price troughs."""
        start_time = time.time()
        if not prices: raise DataError("Empty price list provided to find_troughs")

        depth_threshold = min_depth if min_depth is not None else self.config.min_trough_depth

        try:
            # Invert prices to find troughs as peaks of the inverted series
            inverted_prices = [-x for x in prices]
            # Use the peak detection logic, passing the depth threshold as min_height for inverted data
            troughs = self.find_peaks(inverted_prices, symbol, depth_threshold)

            execution_time = time.time() - start_time
            PEAK_DETECTION_TIME.labels(operation="find_troughs").observe(execution_time)
            PEAK_DETECTION_COUNT.labels(symbol=symbol, type="trough").inc(len(troughs))
            self.logger.debug(f"Found {len(troughs)} troughs for {symbol} in {execution_time:.3f}s")
            return troughs

        except Exception as e:
            self.logger.error(f"Error in trough detection for {symbol}: {str(e)}", exc_info=True)
            raise ModelInferenceError(f"Trough detection failed: {str(e)}") from e

    def _smooth_prices(self, prices: torch.Tensor, window_size: Optional[int] = None) -> torch.Tensor:
        """Apply Gaussian smoothing to price data."""
        window = window_size if window_size is not None else self.config.smoothing_window
        if len(prices) < window: return prices

        kernel = torch.exp(-0.5 * (torch.arange(window, device=self.device) - window//2)**2 / 2)
        kernel = kernel / kernel.sum()

        # Ensure kernel and prices are float32 for conv1d
        kernel = kernel.to(dtype=torch.float32)
        prices = prices.to(dtype=torch.float32)

        # Unsqueeze to 3D (batch=1, channel=1, length) for padding and conv1d
        prices_3d = prices.unsqueeze(0).unsqueeze(0)
        # Pad the 3D tensor (only the last dimension)
        padding = (window // 2, window // 2)
        padded = torch.nn.functional.pad(prices_3d, padding, mode='reflect')

        # Kernel shape: (C_out, C_in/groups, kW) -> (1, 1, window)
        smoothed = torch.nn.functional.conv1d(
            padded,
            kernel.unsqueeze(0).unsqueeze(0)
        )
        # Squeeze back to 1D
        return smoothed.squeeze()

    @contextlib.contextmanager
    def _gpu_resource_context(self):
        """Context manager for GPU resource management."""
        initial_memory = 0
        if self.use_gpu:
            initial_memory = torch.cuda.memory_allocated()
            self.logger.debug(f"Initial GPU memory: {initial_memory / 1024 / 1024:.2f} MB")
        try:
            yield
        finally:
            if self.use_gpu:
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                self.logger.debug(f"Final GPU memory: {final_memory / 1024 / 1024:.2f} MB")
                if final_memory > initial_memory:
                    self.logger.warning(f"Potential memory leak in peak detection: {(final_memory - initial_memory) / 1024 / 1024:.2f} MB not released")
