"""Stock selection and ranking system."""
from typing import List, Dict, Optional
import logging
import numpy as np
import torch

from .base import TradingEngine

class StockSelector:
    """Selects and ranks stocks based on multiple factors."""
    
    def __init__(self, engine: TradingEngine):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def select_stocks(self, universe: List[str], n: int = 10) -> List[Dict]:
        """Select top n stocks from the universe."""
        scored = []
        
        for symbol in universe:
            try:
                score = self._score_stock(symbol)
                scored.append({
                    'symbol': symbol,
                    'score': score
                })
            except Exception as e:
                self.logger.warning(f"Failed to score {symbol}: {str(e)}")
                
        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:n]
        
    def _score_stock(self, symbol: str) -> float:
        """Calculate composite score for a stock."""
        # Get market data
        data = self.engine.redis.get_market_data(symbol)
        
        # Convert to tensors
        prices = torch.tensor(data['close'], dtype=torch.float32).to(self.device)
        volume = torch.tensor(data['volume'], dtype=torch.float32).to(self.device)
        
        # Calculate factors
        momentum = self._calculate_momentum(prices)
        volatility = self._calculate_volatility(prices)
        volume_score = self._calculate_volume_score(volume)
        
        # Combine factors
        return (0.4 * momentum) + (0.3 * (1 - volatility)) + (0.3 * volume_score)
        
    def _calculate_momentum(self, prices: torch.Tensor, lookback: int = 20) -> float:
        """Calculate momentum factor."""
        returns = prices[-1] / prices[-lookback] - 1
        return torch.sigmoid(returns * 10).item()
        
    def _calculate_volatility(self, prices: torch.Tensor, lookback: int = 20) -> float:
        """Calculate volatility factor."""
        returns = torch.log(prices[1:] / prices[:-1])
        return returns.std().item()
        
    def _calculate_volume_score(self, volume: torch.Tensor, lookback: int = 20) -> float:
        """Calculate volume anomaly score."""
        avg_volume = volume[-lookback:].mean()
        current_volume = volume[-1]
        return torch.sigmoid((current_volume - avg_volume) / avg_volume).item()
