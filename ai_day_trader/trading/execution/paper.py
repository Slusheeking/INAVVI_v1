"""Paper trading execution implementation with CSV logging."""
import asyncio
import time
import csv
import math # Import math for ceiling function
from datetime import datetime, timezone, timedelta # Added timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

# Import base classes and circuit breaker from relative paths
from .base import ExecutionSystem, TradeExecutionDetails, PositionInfo
from .base import PAPER_TRADE_LOG_FILE, PAPER_TRADE_LOG_FIELDNAMES, _setup_paper_trade_log
from .circuit_breaker import execution_circuit_breaker_registry
# Import Polygon client for price fetching
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient

# Import Config class
from ai_day_trader.utils.config import Config

class PaperExecution(ExecutionSystem):
    """Paper trading execution implementation with CSV logging."""

    def __init__(self, config: Config, polygon_client: PolygonRESTClient, redis_client=None): # Add redis_client parameter
        super().__init__(config) # Pass config to base class
        self.polygon_client = polygon_client # Store Polygon client instance
        self.redis_client = redis_client # Store Redis client for NBBO data
        self.paper_positions: Dict[str, PositionInfo] = {}
        self.paper_orders: Dict[str, TradeExecutionDetails] = {}
        self.order_id_counter = 0
        self.trade_log_lock = asyncio.Lock()  # Lock for writing to CSV

        # Load simulation parameters from config
        self.slippage_pct = self.config.get_float("PAPER_SLIPPAGE_PCT", 0.0005) # 0.05% default slippage
        self.commission_per_share = self.config.get_float("PAPER_COMMISSION_PER_SHARE", 0.005)
        self.commission_min_per_order = self.config.get_float("PAPER_COMMISSION_MIN_PER_ORDER", 1.0)
        self.logger.info(f"Paper trading slippage: {self.slippage_pct*100:.3f}%, Commission/share: ${self.commission_per_share:.3f}, Min commission: ${self.commission_min_per_order:.2f}")


        # Create circuit breakers for each operation
        self.execute_order_cb = execution_circuit_breaker_registry.get_or_create("execute_order", "paper")
        self.cancel_order_cb = execution_circuit_breaker_registry.get_or_create("cancel_order", "paper")
        self.get_positions_cb = execution_circuit_breaker_registry.get_or_create("get_positions", "paper")
        self.get_order_status_cb = execution_circuit_breaker_registry.get_or_create("get_order_status", "paper")

    async def _log_paper_trade(self, execution_details: TradeExecutionDetails):
        """Appends execution details to the CSV log file."""
        # Ensure all required fields are present, defaulting if necessary
        log_entry = {
            field: execution_details.get(field) for field in PAPER_TRADE_LOG_FIELDNAMES
        }
        # Format timestamp for CSV
        ts = log_entry.get('timestamp')
        log_entry['timestamp'] = datetime.fromtimestamp(ts).isoformat() if ts else ''
        # Ensure numeric fields are formatted reasonably if None (though they shouldn't be None if filled)
        for field in ['fill_price', 'fill_quantity', 'commission']:
             if log_entry[field] is None: log_entry[field] = ''


        async with self.trade_log_lock:
            try:
                # Ensure log file exists (might be redundant if _setup_paper_trade_log works)
                _setup_paper_trade_log()
                with open(PAPER_TRADE_LOG_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=PAPER_TRADE_LOG_FIELDNAMES)
                    # Write header if file is empty (check size)
                    f.seek(0, 2) # Go to end of file
                    if f.tell() == 0:
                         writer.writeheader()
                    writer.writerow(log_entry)
            except Exception as e:
                self.logger.error(f"Failed to write paper trade to log: {e}")

    async def execute_order(self, order: Dict) -> TradeExecutionDetails:
        """Simulate order execution with circuit breaker protection."""
        return await self.execute_order_cb.execute(self._execute_order_impl, order)

    async def cancel_order(self, order_id: str) -> bool:
        """Simulate order cancellation with circuit breaker protection."""
        return await self.cancel_order_cb.execute(self._cancel_order_impl, order_id)

    async def get_positions(self) -> Dict[str, PositionInfo]:
        """Get simulated portfolio positions with circuit breaker protection."""
        return await self.get_positions_cb.execute(self._get_positions_impl)

    async def get_order_status(self, order_id: str) -> Optional[TradeExecutionDetails]:
        """Get the status of a specific simulated order with circuit breaker protection."""
        return await self.get_order_status_cb.execute(self._get_order_status_impl, order_id)

    async def _execute_order_impl(self, order: Dict) -> TradeExecutionDetails:
        """Simulate order execution and log the result."""
        self.logger.info(f"Paper trading order received: {order}")
        self.order_id_counter += 1
        order_id = f"PAPER-{self.order_id_counter}"
        symbol, qty, side = order.get('symbol'), order.get('quantity', 0), order.get('side')
        order_type = order.get('order_type', 'market')
        status, fill_price, fill_qty, error_message, commission = "accepted", None, 0, None, 0.0

        # --- Simulate Fill Logic using fetched NBBO ---
        nbbo = await self._get_current_nbbo(symbol) # Use NBBO method

        if nbbo is None:
            status, error_message = "rejected", f"Could not get current NBBO for {symbol}."
            self.logger.warning(f"Paper trade rejected for {symbol}: {error_message}")
        elif order_type == 'market':
            bid_price = nbbo['bid_price']  # Updated key name
            ask_price = nbbo['ask_price']  # Updated key name
            # Simulate slippage for market orders
            if side == 'buy':
                # Buy fills at ask + slippage
                base_price = ask_price
                slippage_amount = max(base_price * self.slippage_pct, 0.01) # Min $0.01 slippage
                fill_price = base_price + slippage_amount
                self.logger.info(f"Paper filling MARKET BUY for {symbol} at {fill_price:.4f} (Ask: {ask_price:.4f} + Slippage: {slippage_amount:.4f})")
            else: # side == 'sell'
                # Sell fills at bid - slippage
                base_price = bid_price
                slippage_amount = max(base_price * self.slippage_pct, 0.01) # Min $0.01 slippage
                fill_price = base_price - slippage_amount
                self.logger.info(f"Paper filling MARKET SELL for {symbol} at {fill_price:.4f} (Bid: {bid_price:.4f} - Slippage: {slippage_amount:.4f})")

            status, fill_qty = "filled", qty
            # Calculate commission for filled order
            commission = max(self.commission_min_per_order, abs(fill_qty) * self.commission_per_share)

        elif order_type == 'limit':
            limit_price = order.get('limit_price')
            if limit_price is None:
                status, error_message = "rejected", "Limit price missing."
            else:
                limit_price = float(limit_price) # Ensure float
                bid_price = nbbo['bid_price']  # Updated key name
                ask_price = nbbo['ask_price']  # Updated key name
                limit_met = False
                if side == 'buy' and limit_price >= ask_price: # Compare limit to NBBO ask
                    # Buy limit met if limit price is at or above current ask
                    limit_met = True
                    fill_price = limit_price # Fill at limit price
                    self.logger.info(f"Paper simulating LIMIT BUY fill for {symbol} at {fill_price:.4f} (Limit >= Ask: {ask_price:.4f})")
                elif side == 'sell' and limit_price <= bid_price: # Compare limit to NBBO bid
                    # Sell limit met if limit price is at or below current bid
                    limit_met = True
                    fill_price = limit_price # Fill at limit price
                    self.logger.info(f"Paper simulating LIMIT SELL fill for {symbol} at {fill_price:.4f} (Limit <= Bid: {bid_price:.4f})")

                if limit_met:
                    status, fill_qty = "filled", qty
                    # Calculate commission for filled order
                    commission = max(self.commission_min_per_order, abs(fill_qty) * self.commission_per_share)
                else:
                    # Limit not met, order remains accepted (no fill, no commission)
                    status = "accepted"
                    if side == 'buy':
                         self.logger.info(f"Paper LIMIT BUY for {symbol} at {limit_price:.4f} NOT filled (Limit < Ask: {ask_price:.4f})")
                    else: # side == 'sell'
                         self.logger.info(f"Paper LIMIT SELL for {symbol} at {limit_price:.4f} NOT filled (Limit > Bid: {bid_price:.4f})")
        else:
            status, error_message = "rejected", f"Order type '{order_type}' not supported."
            self.logger.warning(f"Paper trade rejected for {symbol}: {error_message}")

        # --- Create Execution Details ---
        execution_details: TradeExecutionDetails = {
            "order_id": order_id,
            "client_order_id": order.get('client_order_id'),
            "symbol": symbol,
            "quantity": qty,
            "side": side,
            "order_type": order_type,
            "time_in_force": order.get('time_in_force', 'day'),
            "limit_price": order.get('limit_price'),
            "stop_price": order.get('stop_price'),
            "status": status,
            "fill_price": fill_price if status == "filled" else None,
            "fill_quantity": float(fill_qty) if status == "filled" else 0.0, # Ensure float
            "timestamp": time.time(),
            "commission": commission if status == "filled" else 0.0, # Add commission field
            "error_message": error_message
        }
        self.paper_orders[order_id] = execution_details
        await self._log_paper_trade(execution_details) # Log includes commission if field added

        # --- Update Paper Position ---
        if status == "filled":
            # Pass commission to update function
            await self._update_paper_position(symbol, side, float(fill_qty), fill_price, commission)

        return execution_details

    async def _update_paper_position(self, symbol: str, side: str, filled_qty: float, fill_price: float, commission: float) -> None:
        """Helper to update simulated positions, including commission in cost basis."""
        current_pos = self.paper_positions.get(symbol)
        change_qty = filled_qty if side == 'buy' else -filled_qty
        # Commission always reduces value/increases cost basis
        cost_change = (change_qty * fill_price) + commission

        if current_pos:
            current_qty = float(current_pos.get('quantity', 0.0)) # Ensure float
            current_cost = float(current_pos.get('cost_basis', 0.0)) # Ensure float
            new_qty = current_qty + change_qty

            if abs(new_qty) < 1e-9: # Position closed
                # Log realized PnL (optional)
                proceeds = abs(change_qty * fill_price) - commission
                realized_pnl = proceeds - current_cost if side == 'sell' else current_cost - proceeds # Rough calc
                self.logger.info(f"Paper position closed for {symbol}. Approx Realized PnL: ${realized_pnl:.2f} (incl. ${commission:.2f} commission)")
                if symbol in self.paper_positions:
                    del self.paper_positions[symbol]
            else: # Position modified
                new_cost = current_cost + cost_change
                new_entry_price = new_cost / new_qty if new_qty != 0 else 0 # Avg price includes commission cost
                current_pos.update({
                    'quantity': str(new_qty), # Store as string like Alpaca
                    'cost_basis': str(new_cost), # Store as string
                    'avg_entry_price': str(new_entry_price), # Store as string
                    'last_update_time': datetime.now(timezone.utc) # Use timezone aware
                })
                self.logger.info(f"Paper position updated for {symbol}: Qty {new_qty:.2f}, Avg Cost {new_entry_price:.2f} (incl. ${commission:.2f} commission)")
        elif abs(change_qty) > 1e-9: # New position opened
            self.paper_positions[symbol] = {
                "symbol": symbol,
                "quantity": str(change_qty), # Store as string
                "avg_entry_price": str(fill_price), # Initial avg price is fill price
                "cost_basis": str(cost_change), # Cost basis includes commission
                "market_value": str(change_qty * fill_price), # Initial market value (pre-commission for PnL calc)
                "current_price": str(fill_price), # Initial current price
                "unrealized_pl": str(-commission), # Initial PnL is negative commission
                "unrealized_plpc": 0.0,
                "lastday_price": fill_price, # Use fill price as initial lastday price
                "change_today": 0.0,
                "last_update_time": datetime.now(timezone.utc) # Use timezone aware
            }
            self.logger.info(f"Paper position opened for {symbol}: Qty {change_qty:.2f} @ {fill_price:.2f}") # Use change_qty

    async def _cancel_order_impl(self, order_id: str) -> bool:
        """Simulate order cancellation and log."""
        self.logger.info(f"Paper cancel order request: {order_id}")
        cancelled = False
        if order_id in self.paper_orders:
            order = self.paper_orders[order_id]
            if order['status'] in ['accepted', 'new', 'partially_filled']: # Can cancel these states
                order['status'] = 'canceled'
                order['cancelled_at'] = time.time()
                self.logger.info(f"Paper order {order_id} cancelled.")
                cancelled = True
                await self._log_paper_trade(order) # Log the cancellation
            else:
                self.logger.warning(f"Paper order {order_id} cannot be cancelled (status: {order['status']}).")
        else:
            self.logger.warning(f"Paper order {order_id} not found for cancellation.")
        return cancelled

    async def _get_positions_impl(self) -> Dict[str, PositionInfo]:
        """Get simulated portfolio positions with updated PnL."""
        self.logger.debug("Fetching and updating paper positions...")
        updated_positions = {}
        symbols_to_update = list(self.paper_positions.keys()) # Get symbols before iterating

        for symbol in symbols_to_update:
            pos_info = self.paper_positions.get(symbol)
            if not pos_info: continue # Should not happen if iterating keys, but safe check

            nbbo = await self._get_current_nbbo(symbol) # Use NBBO method
            if nbbo is not None:
                # Mark longs to bid, shorts to ask for PnL
                qty = float(pos_info.get('quantity', 0.0))
                mark_price = nbbo['bid_price'] if qty > 0 else nbbo['ask_price']  # Updated key names
                if mark_price <= 0: # Handle invalid mark price
                     self.logger.warning(f"Invalid NBBO mark price ({mark_price}) for {symbol} PnL calc. Skipping update.")
                     updated_positions[symbol] = pos_info # Keep existing info
                     continue

                cost_basis = float(pos_info.get('cost_basis', 0.0))
                market_value = qty * mark_price
                unrealized_pl = market_value - cost_basis
                unrealized_plpc = (unrealized_pl / abs(cost_basis)) * 100 if cost_basis != 0 else 0.0 # Use abs(cost_basis)

                # Update the position info dictionary directly
                pos_info['current_price'] = str(mark_price) # Store as string
                pos_info['market_value'] = str(market_value) # Store as string
                pos_info['unrealized_pl'] = str(unrealized_pl) # Store as string
                pos_info['unrealized_plpc'] = str(unrealized_plpc) # Store as string
                # 'change_today' would require fetching open price, more complex
                pos_info['last_update_time'] = datetime.now(timezone.utc)
                updated_positions[symbol] = pos_info # Add updated info to new dict
                self.logger.debug(f"Updated PnL for {symbol}: MarkPrice={mark_price:.2f}, PnL={unrealized_pl:.2f}")
            else:
                # Could not fetch NBBO, return existing data without PnL update
                updated_positions[symbol] = pos_info # Keep existing info
                self.logger.warning(f"Could not fetch current NBBO for {symbol} to update PnL.")

        self.paper_positions = updated_positions # Update the stored positions
        return self.paper_positions.copy() # Return a copy

    async def _get_current_nbbo(self, symbol: str) -> Optional[Dict[str, float]]:
        """Fetches the latest NBBO quote (bid/ask) from Redis or fallback to Polygon client."""
        # First, try to get NBBO from Redis if available
        if self.redis_client:
            try:
                key = f"nbbo:{symbol}"
                nbbo_data_bytes = await self.redis_client.hgetall(key)
                if nbbo_data_bytes:
                    # Decode and convert to float, handle potential errors
                    nbbo_data = {}
                    required_keys = ['bid_price', 'ask_price', 'timestamp']
                    all_keys_present = True
                    
                    for k_bytes, v_bytes in nbbo_data_bytes.items():
                        k = k_bytes.decode('utf-8')
                        try:
                            # Handle potential empty strings before converting to float
                            value_str = v_bytes.decode('utf-8')
                            if value_str:
                                nbbo_data[k] = float(value_str)
                            else:
                                # Decide how to handle empty values - treat as missing?
                                self.logger.warning(f"Empty value for key '{k}' in NBBO data for {symbol}")
                                if k in required_keys: 
                                    all_keys_present = False 
                                    break
                                nbbo_data[k] = 0.0  # Default to 0 for sizes
                        except (ValueError, TypeError, UnicodeDecodeError) as e:
                            self.logger.warning(f"Invalid value format for key '{k}' in NBBO data for {symbol}: {v_bytes}. Error: {e}")
                            return None  # Invalidate cache entry if any value is bad

                    # Check if essential keys are present after decoding attempt
                    if not all_keys_present:
                        self.logger.warning(f"Missing required keys in NBBO data for {symbol} after decoding.")
                        return None
                        
                    for req_key in required_keys:
                        if req_key not in nbbo_data:
                            self.logger.warning(f"Missing required key '{req_key}' in NBBO data for {symbol}.")
                            return None  # Return None if essential keys are missing

                    # Check timestamp staleness
                    quote_timestamp_ms = nbbo_data.get('timestamp', 0)
                    current_time_ms = time.time() * 1000
                    # Allow some staleness, e.g., 15 seconds
                    if current_time_ms - quote_timestamp_ms > 15000:
                        self.logger.warning(f"Stale NBBO data for {symbol} ({(current_time_ms - quote_timestamp_ms)/1000:.1f}s old).")
                        return None  # Treat stale data as missing

                    # Ensure sizes are present, default to 0 if not
                    nbbo_data['bid_size'] = nbbo_data.get('bid_size', 0.0)
                    nbbo_data['ask_size'] = nbbo_data.get('ask_size', 0.0)

                    self.logger.debug(f"Using Redis NBBO for {symbol}: Bid={nbbo_data['bid_price']:.4f}, Ask={nbbo_data['ask_price']:.4f}")
                    return nbbo_data
            except Exception as e:
                self.logger.error(f"Error fetching NBBO data from Redis for {symbol}: {e}", exc_info=True)
                # Fall through to polygon client if Redis fails

        # Fallback: Use Polygon client if Redis is unavailable or failed
        try:
            self.logger.debug(f"Fetching current NBBO for {symbol} via Polygon client (fallback)...")
            quote_data = await self.polygon_client.get_last_quote(symbol)
            if quote_data and 'bid' in quote_data and 'ask' in quote_data:
                # Convert to the standard NBBO format we're using with Redis
                nbbo_data = {
                    'bid_price': float(quote_data['bid']),
                    'ask_price': float(quote_data['ask']),
                    'bid_size': float(quote_data.get('bidsize', 0)),
                    'ask_size': float(quote_data.get('asksize', 0)),
                    'timestamp': time.time() * 1000  # Current time in milliseconds
                }
                self.logger.debug(f"Using Polygon NBBO for {symbol}: Bid={nbbo_data['bid_price']:.4f}, Ask={nbbo_data['ask_price']:.4f}")
                return nbbo_data
            else:
                self.logger.warning(f"Polygon client returned no valid quote data for {symbol}.")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching quote from Polygon for {symbol}: {e}", exc_info=True)
            return None


    async def _get_order_status_impl(self, order_id: str) -> Optional[TradeExecutionDetails]:
        """Get the status of a specific simulated order, potentially filling open limit orders."""
        self.logger.debug(f"Fetching paper order status for: {order_id}")
        order = self.paper_orders.get(order_id)

        if not order:
            self.logger.warning(f"Order ID {order_id} not found in paper orders.")
            return None

        # Check if an open limit order can now be filled
        if order['status'] == 'accepted' and order['order_type'] == 'limit':
            symbol = order['symbol']
            side = order['side']
            limit_price = order.get('limit_price')
            qty = order.get('quantity')

            if limit_price is None or qty is None:
                 self.logger.error(f"Limit order {order_id} is missing price or quantity.")
                 return order # Return existing status

            limit_price = float(limit_price)
            qty = float(qty)

            self.logger.debug(f"Checking fill condition for open limit order {order_id} ({side} {qty} {symbol} @ {limit_price:.4f})")
            nbbo = await self._get_current_nbbo(symbol) # Use NBBO method

            if nbbo:
                bid_price = nbbo['bid_price']  # Updated key name
                ask_price = nbbo['ask_price']  # Updated key name
                limit_met = False
                fill_price = limit_price # Limit orders fill at the limit price

                if side == 'buy' and limit_price >= ask_price:
                    limit_met = True
                    self.logger.info(f"Open LIMIT BUY order {order_id} condition met (Limit {limit_price:.4f} >= Ask {ask_price:.4f})")
                elif side == 'sell' and limit_price <= bid_price:
                    limit_met = True
                    self.logger.info(f"Open LIMIT SELL order {order_id} condition met (Limit {limit_price:.4f} <= Bid {bid_price:.4f})")

                if limit_met:
                    self.logger.info(f"Simulating fill for previously accepted limit order {order_id}.")
                    # Calculate commission for the fill
                    commission = max(self.commission_min_per_order, abs(qty) * self.commission_per_share)
                    order['status'] = 'filled'
                    order['fill_price'] = fill_price
                    order['fill_quantity'] = qty
                    order['timestamp'] = time.time() # Update timestamp to fill time
                    order['commission'] = commission # Record commission

                    # Log the fill and update position, passing commission
                    await self._log_paper_trade(order)
                    await self._update_paper_position(symbol, side, qty, fill_price, commission)
            else:
                self.logger.warning(f"Could not get NBBO for {symbol} to check limit order {order_id} status.")

        return order # Return the potentially updated order

    async def initialize(self):
        await super().initialize()
        self.logger.info("PaperExecution initialized.")

    async def close(self):
        await super().close()
        self.logger.info("PaperExecution closed.")
