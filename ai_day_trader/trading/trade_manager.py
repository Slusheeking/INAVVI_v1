"""Trade execution and rollback management system."""
import asyncio
import time
import logging # Added logging import
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from ai_day_trader.utils.logging_config import get_logger # Use new utils path
from ai_day_trader.utils.metrics_registry import ( # Use new utils path
    MetricPrefix, register_counter, register_gauge, register_histogram
)
from ai_day_trader.utils.exceptions import TradingError, ExecutionError, OrderError # Use new utils path
from ai_day_trader.utils.resource_manager import resource_managed, ResourceContext # Use new utils path

# Define metrics for trade execution and rollback
TRADE_EXECUTION_COUNT = register_counter(
    MetricPrefix.TRADING,
    "trade_execution_count",
    "Number of trade executions",
    ["symbol", "status", "order_type"]
)

TRADE_ROLLBACK_COUNT = register_counter(
    MetricPrefix.TRADING,
    "trade_rollback_count",
    "Number of trade rollbacks",
    ["symbol", "reason"]
)

TRADE_EXECUTION_TIME = register_histogram(
    MetricPrefix.TRADING,
    "trade_execution_time_seconds",
    "Time spent on trade execution",
    ["symbol", "order_type"]
)

@dataclass
class TradeTransaction:
    """Class to track a trade transaction for potential rollback."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    status: str = "pending"
    execution_details: Dict = field(default_factory=dict)
    rollback_attempts: int = 0
    max_rollback_attempts: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class TradeManager:
    """
    Manages trade execution and provides automatic rollback capabilities for failed trades.
    This class acts as a layer between the trading engine and the execution system,
    adding transaction tracking and rollback functionality.
    """

    def __init__(self, execution_system: Any, logger: logging.Logger):
        """
        Initialize the trade manager.

        Args:
            execution_system: An initialized execution system instance.
            logger: A configured logger instance.
        """
        self.execution_system = execution_system # Store execution system directly
        self.logger = logger # Use passed logger
        self.active_transactions: Dict[str, TradeTransaction] = {}
        self.transaction_lock = asyncio.Lock()

    @resource_managed("trade_execution")
    async def execute_trade(self, order: Dict) -> Dict:
        """
        Execute a trade with automatic rollback capability for failed trades.
        
        Args:
            order: Order details including symbol, side, quantity, etc.
            
        Returns:
            Dictionary with execution details and status
            
        Raises:
            TradingError: If trade execution fails and cannot be rolled back
        """
        if not self.execution_system: # Check stored execution system
            raise TradingError("No execution system available in TradeManager")

        symbol = order.get('symbol')
        if not symbol:
            raise ValueError("Order missing required field: symbol")
            
        # Generate a transaction ID for tracking
        transaction_id = f"tx_{int(time.time()*1000)}_{symbol}"
        
        # Create transaction record
        transaction = TradeTransaction(
            order_id=transaction_id,
            symbol=symbol,
            side=order.get('side', 'unknown'),
            quantity=order.get('quantity', 0),
            order_type=order.get('order_type', 'market')
        )
        
        # Store transaction
        async with self.transaction_lock:
            self.active_transactions[transaction_id] = transaction
        
        start_time = time.time()
        try:
            # Execute the order
            self.logger.info(f"Executing trade via TradeManager for {symbol}: {order.get('side')} {order.get('quantity')} @ {order.get('order_type')}")
            execution_details = await self.execution_system.execute_order(order) # Use stored execution system

            # Update transaction record
            transaction.status = execution_details.get('status', 'unknown')
            transaction.execution_details = execution_details
            transaction.completed_at = datetime.now(timezone.utc)
            
            # Record metrics
            TRADE_EXECUTION_COUNT.labels(
                symbol=symbol, 
                status=transaction.status,
                order_type=order.get('order_type', 'market')
            ).inc()
            
            execution_time = time.time() - start_time
            TRADE_EXECUTION_TIME.labels(
                symbol=symbol,
                order_type=order.get('order_type', 'market')
            ).observe(execution_time)
            
            self.logger.info(f"Trade execution completed for {symbol} in {execution_time:.2f}s with status: {transaction.status}")
            return execution_details
            
        except Exception as e:
            # Handle execution failure with rollback
            execution_time = time.time() - start_time
            self.logger.error(f"Trade execution failed for {symbol}: {str(e)}")
            
            # Update transaction status
            transaction.status = "failed"
            
            # Attempt rollback
            try:
                rollback_result = await self._rollback_trade(transaction, str(e))
                if rollback_result:
                    self.logger.info(f"Successfully rolled back failed trade for {symbol}")
                else:
                    self.logger.error(f"Failed to roll back trade for {symbol}")
            except Exception as rollback_error:
                self.logger.error(f"Error during trade rollback for {symbol}: {str(rollback_error)}")
                
            # Re-raise the original exception
            raise
        finally:
            # Clean up transaction after some time
            if transaction.status in ["filled", "canceled", "rejected"]:
                asyncio.create_task(self._cleanup_transaction(transaction_id, delay=300))  # 5 minutes

    async def _rollback_trade(self, transaction: TradeTransaction, failure_reason: str) -> bool:
        """
        Roll back a failed trade.
        
        Args:
            transaction: The transaction to roll back
            failure_reason: The reason for the rollback
            
        Returns:
            True if rollback was successful, False otherwise
        """
        if not self.execution_system: # Check stored execution system
            self.logger.error("Cannot roll back trade: No execution system available in TradeManager")
            return False

        # Increment rollback attempts
        transaction.rollback_attempts += 1
        
        # Check if we've exceeded max rollback attempts
        if transaction.rollback_attempts > transaction.max_rollback_attempts:
            self.logger.error(f"Exceeded maximum rollback attempts for {transaction.symbol}")
            return False
            
        self.logger.info(f"Rolling back trade for {transaction.symbol} (attempt {transaction.rollback_attempts})")
        
        # Record rollback metric
        TRADE_ROLLBACK_COUNT.labels(symbol=transaction.symbol, reason=failure_reason[:50]).inc()
        
        # If the order was partially filled, create a counter-order
        if transaction.status == "partially_filled" and transaction.execution_details.get("fill_quantity", 0) > 0:
            # Create a counter-order to reverse the partial fill
            counter_side = "sell" if transaction.side == "buy" else "buy"
            fill_quantity = transaction.execution_details.get("fill_quantity", 0)
            
            if fill_quantity > 0:
                counter_order = {
                    "symbol": transaction.symbol,
                    "side": counter_side,
                    "quantity": fill_quantity,
                    "order_type": "market",
                    "time_in_force": "day",
                    "client_order_id": f"rollback_{transaction.order_id}"
                }
                
                try:
                    self.logger.info(f"Executing counter-order for {transaction.symbol}: {counter_side} {fill_quantity}")
                    await self.execution_system.execute_order(counter_order) # Use stored execution system
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to execute counter-order for {transaction.symbol}: {str(e)}")
                    return False
        
        # If the order was not filled or was just submitted, try to cancel it
        elif transaction.status in ["new", "accepted", "pending"]:
            order_id = transaction.execution_details.get("order_id")
            if order_id:
                try:
                    self.logger.info(f"Cancelling order {order_id} for {transaction.symbol}")
                    cancelled = await self.execution_system.cancel_order(order_id) # Use stored execution system
                    return cancelled
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order_id} for {transaction.symbol}: {str(e)}")
                    return False
        
        # No action needed for other statuses
        return True

    async def _cleanup_transaction(self, transaction_id: str, delay: int = 300) -> None:
        """
        Clean up a completed transaction after a delay.
        
        Args:
            transaction_id: ID of the transaction to clean up
            delay: Delay in seconds before cleanup
        """
        await asyncio.sleep(delay)
        async with self.transaction_lock:
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]
                self.logger.debug(f"Cleaned up transaction {transaction_id}")
                
    async def get_active_transactions(self) -> Dict[str, TradeTransaction]:
        """
        Get all active transactions.
        
        Returns:
            Dictionary of active transactions
        """
        async with self.transaction_lock:
            return self.active_transactions.copy()
            
    async def get_transaction(self, transaction_id: str) -> Optional[TradeTransaction]:
        """
        Get a specific transaction by ID.
        
        Args:
            transaction_id: ID of the transaction to get
            
        Returns:
            Transaction if found, None otherwise
        """
        async with self.transaction_lock:
            return self.active_transactions.get(transaction_id)
