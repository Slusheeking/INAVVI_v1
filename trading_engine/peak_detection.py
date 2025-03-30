"""Peak detection algorithms for price analysis."""
from typing import List, Tuple
import logging
import numpy as np
import torch
from scipy.signal import find_peaks

from .base import TradingEngine

class PeakDetector:
    """Detects peaks and troughs in price data."""
    
    def __init__(self, engine: TradingEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def find_peaks(self, prices: List[float], min_height: float = 0.03) -> List[int]:
        """Find price peaks using GPU-accelerated detection."""
        prices_tensor = torch.tensor(prices, dtype=torch.float32).to(self.device)
        
        # Smooth prices first
        smoothed = self._smooth_prices(prices_tensor)
        
        # Convert back to CPU for scipy (or implement custom GPU version)
        peaks, _ = find_peaks(
            smoothed.cpu().numpy(), 
            height=min_height * np.max(smoothed.cpu().numpy())
        )
        return peaks.tolist()
        
    def find_troughs(self, prices: List[float], min_depth: float = 0.03) -> List[int]:
        """Find price troughs using GPU-accelerated detection."""
        inverted = [-x for x in prices]
        return self.find_peaks(inverted, min_depth)
        
    def _smooth_prices(self, prices: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """Apply Gaussian smoothing to price data."""
        if len(prices) < window_size:
            return prices
            
        # Create Gaussian kernel
        kernel = torch.exp(-0.5 * (torch.arange(window_size) - window_size//2)**2 / 2)
        kernel = kernel.to(self.device) / kernel.sum()
        
        # Pad and convolve
        padded = torch.nn.functional.pad(prices, (window_size//2, window_size//2), mode='reflect')
        return torch.nn.functional.conv1d(
            padded.unsqueeze(0).unsqueeze(0), 
            kernel.unsqueeze(0).unsqueeze(0)
        ).squeeze()
