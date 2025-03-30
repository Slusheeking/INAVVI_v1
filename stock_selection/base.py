#!/usr/bin/env python3
"""
Stock Selection Base Module

Contains base classes and interfaces for the stock selection system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, TypedDict, Literal, Optional # Added Optional

# Define more specific types for clarity if possible
# Example: Define what a 'Position' dictionary looks like
class PositionInfo(TypedDict):
    quantity: int
    entry_price: float
    current_price: Optional[float] # Might not always be available
    pnl: Optional[float]
    # Add other relevant position fields

# Example: Define what 'TradeExecutionDetails' looks like
class TradeExecutionDetails(TypedDict):
    order_id: str
    symbol: str
    quantity: int
    side: Literal['buy', 'sell']
    status: Literal['submitted', 'filled', 'partially_filled', 'cancelled', 'rejected', 'error']
    fill_price: Optional[float]
    fill_quantity: Optional[int]
    timestamp: float
    error_message: Optional[str]


logger = logging.getLogger("stock_selection_base")

class StockSelectionBase(ABC):
    """
    Abstract Base Class for stock selection components.

    Defines the essential interface that concrete stock selection
    implementations (e.g., core, GPU-accelerated) must adhere to.
    Ensures consistency across different selection strategies.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Initialize and start the stock selection system.
        This might involve loading initial data, connecting to services, etc.
        """
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """
        Clean up resources and gracefully stop the stock selection system.
        This might involve closing connections, saving state, etc.
        """
        raise NotImplementedError

    @abstractmethod
    async def build_universe(self) -> List[str]:
        """
        Build or refresh the universe of stocks to be considered for selection.

        This typically involves fetching a list of all tradable symbols from
        an exchange or applying broad market filters.

        Returns:
            A list of ticker symbols representing the current stock universe.
        """
        raise NotImplementedError

    @abstractmethod
    async def refresh_watchlist(self) -> List[str]:
        """
        Refresh the active watchlist based on current selection criteria.

        The watchlist is usually a subset of the universe, containing stocks
        that meet certain preliminary conditions (e.g., liquidity, volatility,
        initial score threshold).

        Returns:
            A list of ticker symbols currently in the active watchlist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_focused_list(self) -> List[str]:
        """
        Get the focused list of high-priority stocks for immediate attention.

        This is typically a smaller subset of the watchlist, containing stocks
        that rank highest according to the selection model or have triggered
        specific real-time signals.

        Returns:
            A list of ticker symbols in the focused list.
        """
        raise NotImplementedError


class TradingSystemBase(ABC):
    """
    Abstract Base Class for trading system components.

    Defines the essential interface for systems responsible for executing trades
    and managing positions, ensuring a consistent interaction pattern.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Initialize and start the trading system.
        May involve connecting to a brokerage API, loading account details, etc.
        """
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """
        Clean up resources and gracefully stop the trading system.
        May involve cancelling open orders, disconnecting from APIs, etc.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute_trade(
        self,
        symbol: str,
        quantity: int,
        side: Literal['buy', 'sell'],
        order_type: Literal['market', 'limit'] = 'market', # Example: Add order type
        limit_price: Optional[float] = None # Example: Add limit price
        # Add other relevant parameters like time_in_force, etc.
    ) -> TradeExecutionDetails:
        """
        Execute a trade order.

        Args:
            symbol: The stock ticker symbol to trade.
            quantity: The number of shares to buy or sell.
            side: The direction of the trade ('buy' or 'sell').
            order_type: The type of order (e.g., 'market', 'limit'). Defaults to 'market'.
            limit_price: The limit price if order_type is 'limit'.

        Returns:
            A dictionary containing details about the trade execution attempt
            (e.g., order ID, status, fill details). See TradeExecutionDetails.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_positions(self) -> Dict[str, PositionInfo]:
        """
        Retrieve the current portfolio positions.

        Returns:
            A dictionary where keys are ticker symbols and values are
            dictionaries containing details about each position (e.g., quantity,
            entry price). See PositionInfo.
        """
        raise NotImplementedError

    # Potential additional methods:
    # @abstractmethod
    # async def cancel_order(self, order_id: str) -> Dict[str, Any]:
    #     """Cancel an existing order."""
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def get_account_info(self) -> Dict[str, Any]:
    #     """Get account balance and status."""
    #     raise NotImplementedError
