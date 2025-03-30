"""
Stock Selection Package (:mod:`stock_selection`)
=================================================

Provides a modular system for stock selection, incorporating strategies for
core selection, GPU-accelerated processing, day trading, real-time data handling
via WebSockets, and market data acquisition.

Core Components:
  - :class:`~.base.StockSelectionBase`: Abstract base for selection systems.
  - :class:`~.base.TradingSystemBase`: Abstract base for trading execution systems.
  - :class:`~.core.StockSelectionCore`: CPU-based selection logic.
  - :class:`~.gpu_accelerated.GPUStockSelectionSystem`: GPU-accelerated selection logic.
  - :class:`~.day_trading.DayTradingSystem`: Intraday trading execution logic.
  - :class:`~.websocket.WebSocketEnhancedStockSelection`: Real-time WebSocket data integration.

Market Data Utilities:
  - :func:`~.market_data.get_historical_data`: Fetch historical data.
  - :func:`~.market_data.get_realtime_quotes`: Fetch real-time quotes.
  - :func:`~.market_data.preprocess_market_data`: Preprocess raw market data.
  - :func:`~.market_data.get_unusual_options_activity`: Fetch unusual options data.
  - :func:`~.market_data.setup_market_data_websocket`: Configure WebSocket connection.
"""
# Base classes
from stock_selection.base import StockSelectionBase, TradingSystemBase
# Core implementation
from stock_selection.core import StockSelectionCore
# GPU implementation
from stock_selection.gpu_accelerated import GPUStockSelectionSystem
# Market data utilities
from stock_selection.market_data import (
    get_historical_data,
    get_realtime_quotes,
    preprocess_market_data,
    get_unusual_options_activity, # Added missing function from market_data.py
    setup_market_data_websocket   # Added missing function from market_data.py
)
# Day trading implementation
from stock_selection.day_trading import DayTradingSystem
# WebSocket integration
from stock_selection.websocket import WebSocketEnhancedStockSelection

# Define the public API of the package
__all__ = [
    # Base Classes
    'StockSelectionBase',
    'TradingSystemBase',
    # Implementations
    'StockSelectionCore',
    'GPUStockSelectionSystem',
    'DayTradingSystem',
    'WebSocketEnhancedStockSelection',
    # Market Data Functions
    'get_historical_data',
    'get_realtime_quotes',
    'preprocess_market_data',
    'get_unusual_options_activity',
    'setup_market_data_websocket',
]
