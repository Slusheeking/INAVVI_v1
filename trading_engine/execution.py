"""Order execution system for the trading engine."""
import asyncio
import time
import random
from utils.logging_config import get_logger # Use configured logger
import os
import csv # For logging paper trades
from pathlib import Path # For creating log directory
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any # Added List, Any
from datetime import datetime # For PnL timestamp

# Import base class and potentially shared types/exceptions
# Need TradingEngine for type hint and access to redis/clients if passed directly
from trading_engine.base import TradingEngine
from stock_selection.base import TradeExecutionDetails, PositionInfo # Use shared types
from utils.exceptions import TradingError, APIError as TradingAPIError, APIConnectionError, APITimeoutError # Use shared exceptions
# from utils.config import config # Config is accessed via self.engine.config

# Import a specific brokerage library (e.g., Alpaca)
# Ensure this library is listed in requirements.txt
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError as AlpacaAPIError, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    AlpacaAPIError = Exception # Define as base Exception if unavailable
    tradeapi = None
    TimeFrame = None # Define TimeFrame as None if unavailable
    get_logger(__name__).warning("alpaca-trade-api not found. LiveExecution will not function.") # Use configured logger


# Define log directory and file path
LOG_DIR = Path("./logs")
PAPER_TRADE_LOG_FILE = LOG_DIR / "paper_trades.csv"
PAPER_TRADE_LOG_FIELDNAMES = [
    "timestamp", "order_id", "client_order_id", "symbol", "side",
    "quantity", "order_type", "limit_price", "stop_price",
    "status", "fill_quantity", "fill_price", "error_message"
]

def _setup_paper_trade_log():
    """Creates log directory and writes header if file doesn't exist."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not PAPER_TRADE_LOG_FILE.exists():
            with open(PAPER_TRADE_LOG_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=PAPER_TRADE_LOG_FIELDNAMES)
                writer.writeheader()
    except Exception as e:
        get_logger(__name__).error(f"Failed to setup paper trade log: {e}") # Use configured logger

_setup_paper_trade_log() # Ensure log file is ready on module load


class ExecutionSystem(ABC):
    """Abstract base class for order execution systems."""

    # Pass TradingEngine directly for easier access to components like redis
    def __init__(self, engine: TradingEngine, config_override: Optional[Dict] = None):
        self.engine = engine
        self.config = config_override or {} # Allow overriding config for testing/specific instances
        self.logger = get_logger(self.__class__.__name__) # Use configured logger

    @abstractmethod
    async def execute_order(self, order: Dict) -> TradeExecutionDetails:
        """Execute a trading order asynchronously."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order asynchronously."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, PositionInfo]:
        """Get current portfolio positions asynchronously."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[TradeExecutionDetails]:
         """Get the status of a specific order asynchronously."""
         pass

    async def initialize(self):
        """Perform any necessary async initialization."""
        self.logger.info("Initializing execution system...")
        # Default implementation does nothing, subclasses can override
        pass

    async def close(self):
         """Perform any necessary async cleanup."""
         self.logger.info("Closing execution system...")
         # Default implementation does nothing, subclasses can override
         pass


class LiveExecution(ExecutionSystem):
    """Live order execution implementation using Alpaca as an example."""

    def __init__(self, engine: TradingEngine, config_override: Optional[Dict] = None):
        super().__init__(engine, config_override)
        self.api: Optional[tradeapi.REST] = None
        self._load_config()

    def _load_config(self):
        """Load Alpaca credentials from the engine's centralized config."""
        # Access config via self.engine.config
        self.api_key = self.engine.config.get("APCA_API_KEY_ID")
        self.secret_key = self.engine.config.get("APCA_API_SECRET_KEY")
        self.base_url = self.engine.config.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets") # Default to paper
        # self.data_url = self.engine.config.get("APCA_API_DATA_URL") # Data URL often not needed for REST trading

        # Validate required keys were found in the central config
        if not self.api_key:
            self.logger.error("APCA_API_KEY_ID not found in configuration. LiveExecution disabled.")
        if not self.secret_key:
             self.logger.error("APCA_API_SECRET_KEY not found in configuration. LiveExecution disabled.")

        if self.api_key and self.secret_key:
             self.logger.info(f"Alpaca LiveExecution configured for base URL: {self.base_url}")
        else:
             # Ensure keys are None if missing, preventing partial initialization
             self.api_key, self.secret_key = None, None

    async def initialize(self):
        """Initialize the Alpaca API client."""
        await super().initialize()
        if not ALPACA_AVAILABLE: self.logger.error("Alpaca library unavailable."); return
        if not self.api_key or not self.secret_key: self.logger.error("Alpaca credentials missing."); return
        try:
            self.api = tradeapi.REST(key_id=self.api_key, secret_key=self.secret_key, base_url=self.base_url, api_version='v2')
            account_info = await asyncio.get_event_loop().run_in_executor(None, self.api.get_account)
            self.logger.info(f"Alpaca connection successful. Account Status: {account_info.status}")
        # Catch more specific connection errors if possible, map to custom exceptions
        except AlpacaAPIError as e:
             # Check for common connection/auth issues
             if "forbidden" in str(e).lower() or "unauthorized" in str(e).lower():
                  err_type = TradingAPIError # Use the alias
             else: # Assume other API errors for now
                  err_type = TradingAPIError
             self.logger.error(f"Alpaca API error during init: {e}", exc_info=True)
             self.api = None
             raise err_type(f"Alpaca init error: {e}") from e
        except Exception as e: # Catch potential network errors from underlying requests library
             # if isinstance(e, requests.exceptions.ConnectionError): # Example check
             #      raise APIConnectionError(f"Alpaca connection failed: {e}") from e
             self.logger.exception(f"Unexpected error initializing Alpaca: {e}")
             self.api = None
             raise TradingError(f"Unexpected Alpaca init error: {e}") from e

    async def execute_order(self, order: Dict) -> TradeExecutionDetails:
        """Execute order with Alpaca trading API."""
        if not self.api: raise TradingError("Alpaca API client not initialized.")
        symbol, qty, side = order.get('symbol'), order.get('quantity'), order.get('side')
        order_type = order.get('order_type', 'market')
        time_in_force = order.get('time_in_force', 'day')
        limit_price, stop_price = order.get('limit_price'), order.get('stop_price')
        client_order_id = order.get('client_order_id', f'auto_{int(time.time()*1000)}')
        if not all([symbol, qty, side]): raise ValueError("Order missing required fields: symbol, quantity, side")
        self.logger.info(f"Submitting live order: {side} {qty} {symbol} @ {order_type}")
        try:
            alpaca_order = await asyncio.get_event_loop().run_in_executor(None, lambda: self.api.submit_order(
                symbol=symbol, qty=qty, side=side, type=order_type, time_in_force=time_in_force,
                limit_price=limit_price, stop_price=stop_price, client_order_id=client_order_id
            ))
            self.logger.info(f"Alpaca order submitted: ID {alpaca_order.id}, Status {alpaca_order.status}")
            return {
                "order_id": alpaca_order.id, "client_order_id": alpaca_order.client_order_id,
                "symbol": alpaca_order.symbol, "quantity": float(alpaca_order.qty or 0),
                "side": alpaca_order.side, "order_type": alpaca_order.type,
                "time_in_force": alpaca_order.time_in_force,
                "limit_price": float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                "stop_price": float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                "status": alpaca_order.status,
                "fill_price": float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                "fill_quantity": float(alpaca_order.filled_qty or 0),
                "timestamp": alpaca_order.submitted_at.timestamp() if alpaca_order.submitted_at else time.time(),
                "error_message": None
            }
        except AlpacaAPIError as e:
             # Check for specific actionable errors like insufficient funds, invalid order, etc.
             msg = f"Alpaca API error executing order for {symbol}: {e}"
             self.logger.error(msg)
             raise TradingAPIError(msg) from e # Use alias
        except Exception as e:
             msg = f"Unexpected error executing order via Alpaca for {symbol}: {e}"
             self.logger.exception(msg)
             raise TradingError(msg) from e

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Alpaca trading API."""
        if not self.api: raise TradingError("Alpaca API client not initialized.")
        self.logger.info(f"Attempting to cancel Alpaca order: {order_id}")
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: self.api.cancel_order(order_id))
            self.logger.info(f"Alpaca cancel request successful for order: {order_id}")
            return True
        except AlpacaAPIError as e:
            if "order not found" in str(e).lower() or "already cancelled" in str(e).lower() or "cannot be cancelled" in str(e).lower():
                 self.logger.warning(f"Could not cancel Alpaca order {order_id} (may be final): {e}"); return False
            else:
                 msg = f"Alpaca API error cancelling order {order_id}: {e}"
                 self.logger.error(msg)
                 raise TradingAPIError(msg) from e # Use alias
        except Exception as e:
             msg = f"Unexpected error cancelling order via Alpaca for {order_id}: {e}"
             self.logger.exception(msg)
             raise TradingError(msg) from e

    async def get_positions(self) -> Dict[str, PositionInfo]:
        """Get current portfolio positions from Alpaca."""
        if not self.api: raise TradingError("Alpaca API client not initialized.")
        self.logger.debug("Fetching positions from Alpaca...")
        try:
            alpaca_positions = await asyncio.get_event_loop().run_in_executor(None, self.api.list_positions)
            positions: Dict[str, PositionInfo] = {}
            for pos in alpaca_positions:
                try:
                    positions[pos.symbol] = {
                        "symbol": pos.symbol, "quantity": float(pos.qty), "entry_price": float(pos.avg_entry_price),
                        "current_price": float(pos.current_price), "market_value": float(pos.market_value),
                        "cost_basis": float(pos.cost_basis), "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc), "lastday_price": float(pos.lastday_price),
                        "change_today": float(pos.change_today),
                    }
                except (ValueError, TypeError, AttributeError) as map_err: self.logger.error(f"Error mapping position {pos.symbol}: {map_err}. Data: {pos}"); continue
            self.logger.info(f"Fetched {len(positions)} positions from Alpaca.")
            return positions
        except AlpacaAPIError as e:
             msg = f"Alpaca API error fetching positions: {e}"
             self.logger.error(msg)
             raise TradingAPIError(msg) from e # Use alias
        except Exception as e:
             msg = f"Unexpected error fetching positions via Alpaca: {e}"
             self.logger.exception(msg)
             raise TradingError(msg) from e

    async def get_order_status(self, order_id: str) -> Optional[TradeExecutionDetails]:
         """Get the status of a specific order from Alpaca."""
         if not self.api: raise TradingError("Alpaca API client not initialized.")
         self.logger.debug(f"Fetching order status from Alpaca for order: {order_id}")
         try:
             alpaca_order = await asyncio.get_event_loop().run_in_executor(None, lambda: self.api.get_order(order_id))
             return {
                 "order_id": alpaca_order.id, "client_order_id": alpaca_order.client_order_id,
                 "symbol": alpaca_order.symbol, "quantity": float(alpaca_order.qty or 0),
                 "side": alpaca_order.side, "order_type": alpaca_order.type,
                 "time_in_force": alpaca_order.time_in_force,
                 "limit_price": float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                 "stop_price": float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                 "status": alpaca_order.status,
                 "fill_price": float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                 "fill_quantity": float(alpaca_order.filled_qty or 0),
                 "timestamp": alpaca_order.submitted_at.timestamp() if alpaca_order.submitted_at else None,
                 "filled_at": alpaca_order.filled_at.timestamp() if alpaca_order.filled_at else None,
                 "cancelled_at": alpaca_order.cancelled_at.timestamp() if alpaca_order.cancelled_at else None,
                 "expired_at": alpaca_order.expired_at.timestamp() if alpaca_order.expired_at else None,
                 "error_message": None
             }
         except AlpacaAPIError as e:
             if e.status_code == 404: self.logger.warning(f"Alpaca order not found: {order_id}"); return None
             else:
                  msg = f"Alpaca API error getting order {order_id}: {e}"
                  self.logger.error(msg)
                  raise TradingAPIError(msg) from e # Use alias
         except Exception as e:
              msg = f"Unexpected error getting order status via Alpaca for {order_id}: {e}"
              self.logger.exception(msg)
              raise TradingError(msg) from e

    async def close(self):
        await super().close()
        self.logger.info("LiveExecution (Alpaca) closed.")


class PaperExecution(ExecutionSystem):
    """Paper trading execution implementation with CSV logging."""

    def __init__(self, engine: TradingEngine, config_override: Optional[Dict] = None):
        super().__init__(engine, config_override)
        self.paper_positions: Dict[str, PositionInfo] = {}
        self.paper_orders: Dict[str, TradeExecutionDetails] = {}
        self.order_id_counter = 0
        self.trade_log_lock = asyncio.Lock() # Lock for writing to CSV

    async def _log_paper_trade(self, execution_details: TradeExecutionDetails):
        """Appends execution details to the CSV log file."""
        log_entry = {k: execution_details.get(k) for k in PAPER_TRADE_LOG_FIELDNAMES}
        # Format timestamp for CSV
        log_entry['timestamp'] = datetime.fromtimestamp(log_entry['timestamp']).isoformat() if log_entry.get('timestamp') else ''

        async with self.trade_log_lock:
            try:
                with open(PAPER_TRADE_LOG_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=PAPER_TRADE_LOG_FIELDNAMES)
                    writer.writerow(log_entry)
            except Exception as e:
                self.logger.error(f"Failed to write paper trade to log: {e}")

    async def execute_order(self, order: Dict) -> TradeExecutionDetails:
        """Simulate order execution, using live quotes, and log the result."""
        self.logger.info(f"Paper trading order received: {order}")
        self.order_id_counter += 1
        order_id = f"PAPER-{self.order_id_counter}"
        symbol, qty, side = order.get('symbol'), order.get('quantity', 0), order.get('side')
        order_type = order.get('order_type', 'market')
        status, fill_price, fill_qty, error_message = "accepted", None, 0, None

        if order_type == 'market':
            latest_tick = await self.engine.redis.get_latest_tick_data(symbol)
            if latest_tick:
                try:
                    price_str, bid_str, ask_str = latest_tick.get('price'), latest_tick.get('bid'), latest_tick.get('ask')
                    if side == 'buy': fill_price = float(ask_str) if ask_str else float(price_str) if price_str else None
                    elif side == 'sell': fill_price = float(bid_str) if bid_str else float(price_str) if price_str else None
                    if fill_price is not None: status, fill_qty = "filled", qty; self.logger.info(f"Paper simulating MARKET {side.upper()} fill for {symbol} at {fill_price:.2f}")
                    else: error_message = "No suitable price (ask/bid/last) available for market fill."
                except (ValueError, TypeError) as e: error_message = f"Error processing tick data: {e}"; self.logger.error(f"{error_message} Tick: {latest_tick}")
            else: error_message = "No tick data for market fill simulation."
            if error_message: status = "rejected"; self.logger.error(f"Paper trade failed for {symbol}: {error_message}")

        elif order_type == 'limit':
            limit_price = order.get('limit_price')
            if limit_price is None: status, error_message = "rejected", "Limit price missing."
            else: # Basic immediate fill check (more realistic simulation needs state)
                 latest_tick = await self.engine.redis.get_latest_tick_data(symbol)
                 if latest_tick:
                      try:
                           current_price = float(latest_tick.get('price', 'nan'))
                           if (side == 'buy' and current_price <= limit_price) or (side == 'sell' and current_price >= limit_price):
                                fill_price, status, fill_qty = limit_price, "filled", qty
                                self.logger.info(f"Paper simulating LIMIT {side.upper()} fill for {symbol} at {fill_price:.2f}")
                      except (ValueError, TypeError): self.logger.warning(f"Could not parse price for limit check: {latest_tick}")
        else: status, error_message = "rejected", f"Order type '{order_type}' not supported."

        execution_details: TradeExecutionDetails = {
            "order_id": order_id, "client_order_id": order.get('client_order_id'),
            "symbol": symbol, "quantity": qty, "side": side, "order_type": order_type,
            "time_in_force": order.get('time_in_force', 'day'),
            "limit_price": order.get('limit_price'), "stop_price": order.get('stop_price'),
            "status": status, "fill_price": fill_price if status == "filled" else None,
            "fill_quantity": fill_qty if status == "filled" else 0,
            "timestamp": time.time(), "error_message": error_message
        }
        self.paper_orders[order_id] = execution_details
        await self._log_paper_trade(execution_details) # Log the outcome

        if status == "filled": await self._update_paper_position(symbol, side, fill_qty, fill_price)
        return execution_details

    async def _update_paper_position(self, symbol: str, side: str, filled_qty: float, fill_price: float):
         """Helper to update simulated positions."""
         current_pos = self.paper_positions.get(symbol)
         change = filled_qty if side == 'buy' else -filled_qty
         if current_pos:
              current_qty, current_cost = current_pos.get('quantity', 0), current_pos.get('cost_basis', 0)
              new_qty = current_qty + change
              if abs(new_qty) < 1e-9:
                   if symbol in self.paper_positions: del self.paper_positions[symbol]
                   self.logger.info(f"Paper position closed for {symbol}")
              else:
                   new_cost = current_cost + (change * fill_price)
                   current_pos.update({
                        'quantity': new_qty, 'cost_basis': new_cost,
                        'entry_price': new_cost / new_qty if new_qty != 0 else 0,
                        'last_update_time': datetime.utcnow()
                   })
                   self.logger.info(f"Paper position updated for {symbol}: Qty {new_qty:.2f}, Avg Cost {current_pos['entry_price']:.2f}")
         elif abs(change) > 1e-9:
              self.paper_positions[symbol] = {
                   "symbol": symbol, "quantity": change, "entry_price": fill_price,
                   "cost_basis": change * fill_price, "market_value": change * fill_price,
                   "current_price": fill_price, "unrealized_pl": 0.0, "unrealized_plpc": 0.0,
                   "lastday_price": fill_price, "change_today": 0.0, "last_update_time": datetime.utcnow()
              }
              self.logger.info(f"Paper position opened for {symbol}: Qty {change:.2f} @ {fill_price:.2f}")

    async def cancel_order(self, order_id: str) -> bool:
        """Simulate order cancellation and log."""
        self.logger.info(f"Paper cancel order request: {order_id}")
        cancelled = False
        if order_id in self.paper_orders:
            order = self.paper_orders[order_id]
            if order['status'] in ['accepted', 'new', 'partially_filled']:
                order['status'] = 'canceled'
                order['cancelled_at'] = time.time()
                self.logger.info(f"Paper order {order_id} cancelled.")
                cancelled = True
                # Log cancellation attempt/result
                await self._log_paper_trade(order)
            else: self.logger.warning(f"Paper order {order_id} cannot be cancelled (status: {order['status']}).")
        else: self.logger.warning(f"Paper order {order_id} not found for cancellation.")
        return cancelled

    async def get_positions(self) -> Dict[str, PositionInfo]:
        """Get simulated portfolio positions with updated PnL."""
        self.logger.debug("Fetching and updating paper positions...")
        updated_positions = {}
        symbols_to_update = list(self.paper_positions.keys())
        latest_ticks = {}
        if symbols_to_update:
             tasks = [self.engine.redis.get_latest_tick_data(symbol) for symbol in symbols_to_update]
             results = await asyncio.gather(*tasks)
             for symbol, tick_data in zip(symbols_to_update, results):
                  if tick_data and 'price' in tick_data:
                       try: latest_ticks[symbol] = float(tick_data['price'])
                       except (ValueError, TypeError): self.logger.warning(f"Could not parse price for PnL {symbol}: {tick_data.get('price')}")
                  else: self.logger.warning(f"No current price tick data for PnL {symbol}")
        for symbol, pos in self.paper_positions.items():
            current_price = latest_ticks.get(symbol, pos.get('current_price'))
            quantity, cost_basis = pos.get('quantity', 0), pos.get('cost_basis', 0)
            pos['current_price'] = current_price
            pos['market_value'] = current_price * quantity
            pos['last_update_time'] = datetime.utcnow()
            if quantity != 0: pos['unrealized_pl'] = pos['market_value'] - cost_basis
            else: pos['unrealized_pl'] = 0.0
            if cost_basis != 0: pos['unrealized_plpc'] = pos['unrealized_pl'] / abs(cost_basis)
            else: pos['unrealized_plpc'] = 0.0
            updated_positions[symbol] = pos.copy()
        return updated_positions

    async def get_order_status(self, order_id: str) -> Optional[TradeExecutionDetails]:
         """Get the status of a specific simulated order."""
         self.logger.debug(f"Fetching paper order status for: {order_id}")
         # Basic status check, no live update simulation for limit orders here
         return self.paper_orders.get(order_id)

    async def initialize(self):
        await super().initialize()
        self.logger.info("PaperExecution initialized.")

    async def close(self):
        await super().close()
        self.logger.info("PaperExecution closed.")
