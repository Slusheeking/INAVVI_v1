#!/usr/bin/env python3
"""
Day Trading System Module

Contains intraday trading strategies and execution logic, including risk management
and end-of-day position closing.
"""

import logging
import asyncio
import random
import time
from typing import Dict, List, Optional, Literal, Tuple
import redis # Import for type hint
from datetime import datetime
import pytz # For timezone handling

# Import base types and dependencies
from stock_selection.base import TradingSystemBase, TradeExecutionDetails, PositionInfo
from data_pipeline.base import DataPipeline # For getting current prices
from trading_engine.execution import ExecutionSystem # For actual execution
from utils.async_utils import async_retry, timeout_context
from utils.exceptions import TradingError, APIError, APITimeoutError, RiskError
from utils.metrics_registry import (
    TRADES_EXECUTED,
    TRADE_PNL,
    TRADE_LATENCY,
    OPEN_POSITIONS_VALUE, # New metric for portfolio value
    RISK_CHECKS_FAILED # New metric for failed risk checks
)

logger = logging.getLogger("day_trading")

# Configuration (consider moving to utils.config or injecting)
MAX_TOTAL_POSITION_VALUE = 5000.00
MARKET_CLOSE_TIME_ET = "15:50" # Time to start closing positions (ET)

class DayTradingSystem(TradingSystemBase):
    """
    Implements intraday trading logic, risk management, and execution.

    This system interacts with the DataPipeline for market data and the
    ExecutionSystem (from trading_engine) for order placement and position info.
    It enforces a maximum total portfolio value and ensures positions are closed
    before market end.
    """

    def __init__(
        self,
        data_pipeline: DataPipeline,
        execution_system: ExecutionSystem,
        redis_client: Optional[redis.Redis] = None,
    ):
        """
        Initialize the DayTradingSystem.

        Args:
            data_pipeline: Instance of DataPipeline for market data access.
            execution_system: Instance of ExecutionSystem for trade execution.
            redis_client: Optional Redis client instance for caching or state.
        """
        self.redis_client = redis_client
        self.data_pipeline = data_pipeline
        self.execution_system = execution_system
        self._validate_dependencies()
        self.market_timezone = pytz.timezone("US/Eastern")

    def _validate_dependencies(self):
        """Verify that required dependencies are provided."""
        if not self.data_pipeline:
            logger.error("DataPipeline dependency is missing.")
            raise TradingError("DataPipeline is required for DayTradingSystem.")
        if not self.execution_system:
            logger.error("ExecutionSystem dependency is missing.")
            raise TradingError("ExecutionSystem is required for DayTradingSystem.")

    async def start(self) -> None:
        """Initialize the day trading system."""
        logger.info("Starting Day Trading System...")
        # Add any specific startup logic here (e.g., load trading parameters)
        logger.info("Day Trading System started.")

    async def stop(self) -> None:
        """Clean up resources and stop the day trading system."""
        logger.info("Stopping Day Trading System...")
        # Ensure positions are closed before fully stopping if needed (though EOD logic should handle it)
        await self.close_all_positions_eod(force_close=True) # Force close on stop
        logger.info("Day Trading System stopped.")

    async def _get_current_portfolio_value(self) -> float:
        """Calculates the current market value of all open positions."""
        total_value = 0.0
        try:
            positions = await self.get_positions()
            if not positions:
                return 0.0

            # Fetch current prices for all positions concurrently
            price_tasks = {
                symbol: asyncio.create_task(self.data_pipeline.get_last_price(symbol))
                for symbol in positions.keys()
            }
            await asyncio.gather(*price_tasks.values(), return_exceptions=True)

            for symbol, position_info in positions.items():
                price_task = price_tasks[symbol]
                if price_task.exception():
                    logger.warning(f"Could not get price for {symbol} to calculate portfolio value: {price_task.exception()}")
                    # Fallback: use entry price? Or exclude? Excluding for now.
                    continue

                current_price = price_task.result()
                if current_price is not None and position_info.get('quantity') is not None:
                    total_value += abs(position_info['quantity'] * current_price) # Use absolute value

            OPEN_POSITIONS_VALUE.set(total_value)
            return total_value
        except Exception as e:
            logger.exception(f"Error calculating portfolio value: {e}")
            # Return a safe value (0) or raise? Returning 0 for now.
            OPEN_POSITIONS_VALUE.set(0) # Reset metric on error
            return 0.0

    async def _check_risk(self, symbol: str, quantity: int, side: Literal['buy', 'sell']) -> Tuple[bool, str]:
        """
        Perform pre-trade risk checks based on the $5000 total position value limit.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares.
            side: 'buy' or 'sell'.

        Returns:
            Tuple (risk_ok: bool, reason: str)
        """
        try:
            # 1. Check estimated cost of this trade
            current_price = await self.data_pipeline.get_last_price(symbol)
            if current_price is None:
                return False, f"Could not get current price for {symbol} to check risk."

            trade_value = abs(quantity * current_price)

            # Optional: Check if single trade exceeds limit (can be redundant with total check)
            # if trade_value > MAX_TOTAL_POSITION_VALUE:
            #     RISK_CHECKS_FAILED.labels(symbol=symbol, check="single_trade_limit").inc()
            #     return False, f"Trade value ${trade_value:.2f} exceeds max limit ${MAX_TOTAL_POSITION_VALUE:.2f}"

            # 2. Check total portfolio value *after* this trade
            current_portfolio_value = await self._get_current_portfolio_value()

            # Estimate portfolio value *if* this trade executes
            # This is an approximation - assumes we are adding to the position value.
            # A more precise check would consider if this trade reduces an existing opposite position.
            # For simplicity, we check if the *new* total value exceeds the limit.
            estimated_new_portfolio_value = current_portfolio_value + trade_value
            # TODO: Refine this logic if handling closing trades reducing exposure

            if estimated_new_portfolio_value > MAX_TOTAL_POSITION_VALUE:
                 RISK_CHECKS_FAILED.labels(symbol=symbol, check="total_portfolio_limit").inc()
                 return False, (f"Estimated portfolio value ${estimated_new_portfolio_value:.2f} "
                                f"(current ${current_portfolio_value:.2f} + trade ${trade_value:.2f}) "
                                f"would exceed max limit ${MAX_TOTAL_POSITION_VALUE:.2f}")

            return True, "Risk check passed."

        except Exception as e:
            logger.exception(f"Error during risk check for {symbol}: {e}")
            RISK_CHECKS_FAILED.labels(symbol=symbol, check="exception").inc()
            return False, f"Exception during risk check: {e}"

    @async_retry(max_retries=2, retry_delay=1.0) # Fewer retries for execution
    async def execute_trade(
        self,
        symbol: str,
        quantity: int,
        side: Literal['buy', 'sell'],
        order_type: Literal['market', 'limit'] = 'market',
        limit_price: Optional[float] = None
    ) -> TradeExecutionDetails:
        """
        Validates risk and executes a trade order via the ExecutionSystem.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares.
            side: 'buy' or 'sell'.
            order_type: Type of order ('market' or 'limit').
            limit_price: Required price for limit orders.

        Returns:
            Trade execution details dictionary from the ExecutionSystem.

        Raises:
            RiskError: If the trade violates risk limits.
            TradingError: If the ExecutionSystem is unavailable or execution fails.
            ValueError: If limit_price is missing for a limit order.
        """
        if order_type == 'limit' and limit_price is None:
            raise ValueError("limit_price is required for limit orders.")

        # 1. Perform Risk Check
        risk_ok, reason = await self._check_risk(symbol, quantity, side)
        if not risk_ok:
            logger.warning(f"Trade rejected for {symbol}: {reason}")
            raise RiskError(f"Trade rejected for {symbol}: {reason}")

        # 2. Execute via ExecutionSystem
        logger.info(f"Executing trade via ExecutionSystem: {side} {quantity} {symbol} @ {order_type} {limit_price or ''}")
        start_time = asyncio.get_event_loop().time()

        try:
            # Construct order details for ExecutionSystem
            order_details = {
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'order_type': order_type,
                'limit_price': limit_price,
                # Add other necessary fields like time_in_force if needed by ExecutionSystem
            }

            # Use timeout context for the execution call
            async with timeout_context(30.0): # Longer timeout for execution potentially
                execution_result = await self.execution_system.execute_order(order_details)

            # Record metrics based on actual result
            status = execution_result.get("status", "unknown")
            TRADES_EXECUTED.labels(
                symbol=symbol,
                side=side,
                order_type=order_type,
                status=status
            ).inc()

            latency = asyncio.get_event_loop().time() - start_time
            TRADE_LATENCY.labels(symbol=symbol, side=side, order_type=order_type).observe(latency)
            logger.info(f"Trade {status} for {symbol} via ExecutionSystem in {latency:.4f}s. Order ID: {execution_result.get('order_id', 'N/A')}")

            # Update portfolio value metric after trade confirmation (might be delayed)
            asyncio.create_task(self._get_current_portfolio_value())

            return execution_result

        except asyncio.TimeoutError:
            logger.error(f"Timeout executing trade via ExecutionSystem for {symbol}.")
            raise APITimeoutError(f"Timeout executing trade for {symbol}") from None
        except (APIError, TradingError) as e: # Catch errors from ExecutionSystem
             logger.error(f"ExecutionSystem error for {symbol}: {e}")
             # Re-raise as TradingError to signal execution failure
             raise TradingError(f"ExecutionSystem error for {symbol}: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during trade execution via ExecutionSystem for {symbol}: {e}")
            raise TradingError(f"Unexpected error executing trade for {symbol}: {e}") from e


    async def get_positions(self) -> Dict[str, PositionInfo]:
        """
        Retrieves current portfolio positions from the ExecutionSystem.

        Returns:
            Dictionary mapping symbols to PositionInfo objects.

        Raises:
            TradingError: If the ExecutionSystem call fails.
        """
        logger.debug("Fetching current positions via ExecutionSystem...")
        try:
            positions = await self.execution_system.get_positions()

            # Update PnL and value metrics
            total_value = 0.0
            if positions:
                 price_tasks = {
                    symbol: asyncio.create_task(self.data_pipeline.get_last_price(symbol))
                    for symbol in positions.keys()
                 }
                 await asyncio.gather(*price_tasks.values(), return_exceptions=True)

                 for symbol, position_info in positions.items():
                    # Calculate PnL if not provided by execution system
                    if position_info.get("pnl") is None:
                         price_task = price_tasks.get(symbol)
                         current_price = price_task.result() if price_task and not price_task.exception() else None
                         entry_price = position_info.get("entry_price")
                         quantity = position_info.get("quantity")
                         if current_price is not None and entry_price is not None and quantity is not None:
                              pnl = (current_price - entry_price) * quantity
                              position_info["pnl"] = pnl # Add calculated PnL
                              position_info["current_price"] = current_price # Add current price
                              TRADE_PNL.labels(ticker=symbol).set(pnl)
                         else:
                              logger.warning(f"Could not calculate PnL for {symbol}")
                    else:
                         TRADE_PNL.labels(ticker=symbol).set(position_info["pnl"])

                    # Calculate value
                    current_price = position_info.get("current_price") # Use if already calculated
                    if current_price is None: # Fetch if needed
                         price_task = price_tasks.get(symbol)
                         current_price = price_task.result() if price_task and not price_task.exception() else None

                    if current_price is not None and position_info.get('quantity') is not None:
                         total_value += abs(position_info['quantity'] * current_price)

            OPEN_POSITIONS_VALUE.set(total_value)
            logger.info(f"Retrieved {len(positions)} positions via ExecutionSystem.")
            return positions

        except (APIError, TradingError) as e:
             logger.error(f"ExecutionSystem error fetching positions: {e}")
             raise TradingError(f"ExecutionSystem error fetching positions: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error fetching positions via ExecutionSystem: {e}")
            raise TradingError(f"Unexpected error fetching positions: {e}") from e

    async def close_all_positions_eod(self, force_close: bool = False):
        """
        Checks if it's near market close and closes all open positions.
        """
        now_et = datetime.now(self.market_timezone)
        market_close_dt = now_et.replace(
            hour=int(MARKET_CLOSE_TIME_ET.split(':')[0]),
            minute=int(MARKET_CLOSE_TIME_ET.split(':')[1]),
            second=0,
            microsecond=0
        )

        if not force_close and now_et < market_close_dt:
            # Not time yet
            return

        logger.info(f"Market close time approaching ({MARKET_CLOSE_TIME_ET} ET) or force_close=True. Closing all positions...")

        try:
            positions = await self.get_positions()
            if not positions:
                logger.info("No open positions to close.")
                return

            close_tasks = []
            for symbol, position_info in positions.items():
                quantity = position_info.get('quantity')
                if quantity is None or quantity == 0:
                    continue

                side: Literal['buy', 'sell'] = 'sell' if quantity > 0 else 'buy'
                close_quantity = abs(quantity)

                logger.info(f"Submitting EOD closing order: {side} {close_quantity} {symbol}")
                # Use create_task to run closes concurrently
                close_tasks.append(
                    asyncio.create_task(
                        self.execute_trade(symbol, close_quantity, side, order_type='market')
                    )
                )

            results = await asyncio.gather(*close_tasks, return_exceptions=True)

            # Log results
            success_count = 0
            fail_count = 0
            for i, result in enumerate(results):
                symbol = list(positions.keys())[i] # Get symbol based on task order
                if isinstance(result, TradeExecutionDetails):
                    logger.info(f"Successfully submitted EOD close for {symbol}. Order ID: {result.get('order_id', 'N/A')}")
                    success_count += 1
                else:
                    logger.error(f"Failed to submit EOD close for {symbol}: {result}")
                    fail_count += 1

            logger.info(f"EOD position closing complete. Success: {success_count}, Failed: {fail_count}")

        except Exception as e:
            logger.exception(f"Error during EOD position closing: {e}")
