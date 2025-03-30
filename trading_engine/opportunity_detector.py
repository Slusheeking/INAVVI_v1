"""Market opportunity detection system."""
from typing import List, Dict, Optional
import logging
import numpy as np
import torch

from .base import TradingEngine

class OpportunityDetector:
    """Detects trading opportunities in market data."""
    
    def __init__(self, engine: TradingEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def scan_market(self, symbols: List[str]) -> List[Dict]:
        """Scan market for potential trading opportunities."""
        opportunities = []
        
        for symbol in symbols:
            score = self._score_symbol(symbol)
            if score > self.engine.config.get('min_score', 0.7):
                opportunities.append({
                    'symbol': symbol,
                    'score': score,
                    'reason': "High probability pattern detected"
                })
                
        return opportunities
        
    def _score_symbol(self, symbol: str) -> float:
        """Calculate opportunity score for a single symbol."""
        # Get recent market data
        data = self.engine.redis.get_market_data(symbol)
        
        # Convert to tensor and move to GPU if available
        prices = torch.tensor(data['close'], dtype=torch.float32).to(self.device)
        
        # Apply detection models
        pattern_score = self._detect_patterns(prices)
        volume_score = self._analyze_volume(data['volume'])
        
        return (pattern_score * 0.6) + (volume_score * 0.4)
        
    def _detect_patterns(self, prices: torch.Tensor) -> float:
        """Detect technical patterns in price data."""
        # Implementation would use trained models
        return torch.sigmoid(prices.mean()).item()
        
    def _analyze_volume(self, volume: List[float]) -> float:
        """Analyze volume anomalies."""
        avg_volume = np.mean(volume[-20:])
        current_volume = volume[-1]
        return min(current_volume / avg_volume, 1.0)
