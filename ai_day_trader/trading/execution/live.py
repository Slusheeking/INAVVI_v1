"""Live order execution implementation using Alpaca."""
import asyncio
from typing import Dict, Optional, List, Any

# Import base classes and circuit breaker from relative paths
from .base import ExecutionSystem, TradeExecutionDetails, PositionInfo
from .circuit_breaker import execution_circuit_breaker_registry
from ai_day_trader.utils.exceptions import TradingError, APIError as TradingAPIError # Use new utils path
from ai_day_trader.utils.logging_config import get_logger # Use new utils path
from ai_day_trader.utils.config import Config # Import the main Config class
# Import Polygon client for quote fetching
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient

# Import a specific brokerage library (e.g., Alpaca)
# Ensure this library is listed in requirements.txt
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError as AlpacaAPIError, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    AlpacaAPIError = Exception  # Define as base Exception if unavailable
    tradeapi = None
    TimeFrame = None  # Define TimeFrame as None if unavailable
    get_logger(__name__).warning("alpaca-trade-api not found. LiveExecution will not function.")

class LiveExecution(ExecutionSystem):
    """Live order execution implementation using Alpaca as an example."""

    def __init__(self, config: Config, polygon_client: Optional[PolygonRESTClient] = None): # Accept Config object and optional Polygon client
        super().__init__(config) # Pass Config object to base class
        self.api: Optional[tradeapi.REST] = None
        # Load Alpaca specific config using the passed Config object
        self._load_config() # Call the renamed method

        # Store Polygon client for quote fetching (needed for slippage)
        # If not provided, try to create one (though ideally it's passed from main)
        if polygon_client:
            self.polygon_client = polygon_client
        else:
            self.logger.warning("PolygonRESTClient not provided to LiveExecution. Initializing internally. Slippage monitoring might be less efficient.")
            self.polygon_client = PolygonRESTClient(config=config)
            # Note: We assume the connection pool for this internal client will be initialized elsewhere or handle it lazily.

        # Store expected prices for slippage calculation
        self.expected_prices: Dict[str, float] = {} # Key: client_order_id

        # Create circuit breakers for each operation
        self.execute_order_cb = execution_circuit_breaker_registry.get_or_create("execute_order", "alpaca")
        self.cancel_order_cb = execution_circuit_breaker_registry.get_or_create("cancel_order", "alpaca")
        self.get_positions_cb = execution_circuit_breaker_registry.get_or_create("get_positions", "alpaca")
        self.get_order_status_cb = execution_circuit_breaker_registry.get_or_create("get_order_status", "alpaca")

    def _load_config(self):
        """Load Alpaca credentials from the main Config object."""
        # Use the stored self.config (which is a Config object)
        self.api_key = self.config.get_str("APCA_API_KEY_ID")
        self.secret_key = self.config.get_str("APCA_API_SECRET_KEY")
        self.base_url = self.config.get_str("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

        # Validate required keys were found in the central config
        if not self.api_key:
            self.logger.error("APCA_API_KEY_ID not found in configuration. LiveExecution disabled.")
        if not self.secret_key:
            self.logger.error("APCA_API_SECRET_KEY not found in configuration. LiveExecution disabled.")

        if self.api_key and self.secret_key:
            self.logger.info(f"Alpaca LiveExecution configured for base URL: {self.base_url}")
        else:
            # Ensure keys are None if missing, preventing partial initialization
            self.api_key = None
            self.secret_key = None

    async def initialize(self):
        """Initialize the Alpaca API client."""
        await super().initialize()
        if not ALPACA_AVAILABLE:
            self.logger.error("Alpaca library unavailable.")
            return
        if not self.api_key or not self.secret_key:
            self.logger.error("Alpaca credentials missing.")
            return
        try:
            self.api = tradeapi.REST(key_id=self.api_key, secret_key=self.secret_key, base_url=self.base_url, api_version='v2')
            account_info = await asyncio.get_event_loop().run_in_executor(None, self.api.get_account)
            self.logger.info(f"Alpaca connection successful. Account Status: {account_info.status}")
        except AlpacaAPIError as e:
            # Check for common connection/auth issues
            if "forbidden" in str(e).lower() or "unauthorized" in str(e).lower():
                err_type = TradingAPIError  # Use the alias
            else:
                # Assume other API errors for now
                err_type = TradingAPIError
            self.logger.error(f"Alpaca API error during init: {e}", exc_info=True)
            self.api = None
            raise err_type(f"Alpaca init error: {e}") from e
        except Exception as e:
            self.logger.exception(f"Unexpected error initializing Alpaca: {e}")
            self.api = None
            raise TradingError(f"Unexpected Alpaca init error: {e}") from e

    async def execute_order(self, order: Dict) -> TradeExecutionDetails:
        """Execute order with Alpaca trading API."""
        return await self.execute_order_cb.execute(self._execute_order_impl, order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with Alpaca trading API."""
        return await self.cancel_order_cb.execute(self._cancel_order_impl, order_id)

    async def get_positions(self) -> Dict[str, PositionInfo]:
        """Get current portfolio positions from Alpaca."""
        return await self.get_positions_cb.execute(self._get_positions_impl)

    async def get_order_status(self, order_id: str) -> Optional[TradeExecutionDetails]:
        """Get the status of a specific order from Alpaca."""
        return await self.get_order_status_cb.execute(self._get_order_status_impl, order_id)

    async def _execute_order_impl(self, order: Dict) -> TradeExecutionDetails:
        """Internal implementation of execute_order with circuit breaker protection."""
        if not self.api:
            raise TradingError("Alpaca API client not initialized.")
        symbol = order.get('symbol')
        qty = order.get('quantity')
        side = order.get('side')
        order_type = order.get('order_type', 'market')
        time_in_force = order.get('time_in_force', 'day')
        limit_price = order.get('limit_price')
        stop_price = order.get('stop_price')
        client_order_id = order.get('client_order_id', f'auto_{int(asyncio.get_event_loop().time()*1000)}')

        if not all([symbol, qty, side]):
            raise ValueError("Order missing required fields: symbol, quantity, side")

        self.logger.info(f"Submitting live order: {side} {qty} {symbol} @ {order_type}")

        # --- Slippage Monitoring: Get expected price ---
        expected_price = None
        if order_type == 'market': # Only relevant for market orders
            try:
                quote = await self.polygon_client.get_last_quote(symbol)
                if quote:
                    expected_price = quote['ask'] if side == 'buy' else quote['bid']
                    self.expected_prices[client_order_id] = expected_price # Store by client_order_id
                    self.logger.info(f"Slippage check: Expected price for {client_order_id} ({side} {symbol}) = {expected_price:.4f}")
                else:
                    self.logger.warning(f"Could not get quote for {symbol} before order submission. Cannot calculate slippage.")
            except Exception as e:
                self.logger.error(f"Error fetching quote for slippage check ({symbol}): {e}")
        # ---------------------------------------------

        try:
            alpaca_order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    client_order_id=client_order_id
                )
            )
            self.logger.info(f"Alpaca order submitted: ID {alpaca_order.id}, Status {alpaca_order.status}")

            return {
                "order_id": alpaca_order.id,
                "client_order_id": alpaca_order.client_order_id,
                "symbol": alpaca_order.symbol,
                "quantity": float(alpaca_order.qty or 0),
                "side": alpaca_order.side,
                "order_type": alpaca_order.type,
                "time_in_force": alpaca_order.time_in_force,
                "limit_price": float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                "stop_price": float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                "status": alpaca_order.status,
                "fill_price": float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                "fill_quantity": float(alpaca_order.filled_qty or 0),
                "timestamp": alpaca_order.submitted_at.timestamp() if alpaca_order.submitted_at else asyncio.get_event_loop().time(),
                "error_message": None
            }
        except AlpacaAPIError as e:
            msg = f"Alpaca API error executing order for {symbol}: {e}"
            self.logger.error(msg)
            raise TradingAPIError(msg) from e
        except Exception as e:
            msg = f"Unexpected error executing order via Alpaca for {symbol}: {e}"
            self.logger.exception(msg)
            raise TradingError(msg) from e

    async def _cancel_order_impl(self, order_id: str) -> bool:
        """Internal implementation of cancel_order with circuit breaker protection."""
        if not self.api:
            raise TradingError("Alpaca API client not initialized.")

        self.logger.info(f"Attempting to cancel Alpaca order: {order_id}")

        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: self.api.cancel_order(order_id))
            self.logger.info(f"Alpaca cancel request successful for order: {order_id}")
            return True
        except AlpacaAPIError as e:
            if "order not found" in str(e).lower() or "already cancelled" in str(e).lower() or "cannot be cancelled" in str(e).lower():
                self.logger.warning(f"Could not cancel Alpaca order {order_id} (may be final): {e}")
                return False
            else:
                msg = f"Alpaca API error cancelling order {order_id}: {e}"
                self.logger.error(msg)
                raise TradingAPIError(msg) from e
        except Exception as e:
            msg = f"Unexpected error cancelling order via Alpaca for {order_id}: {e}"
            self.logger.exception(msg)
            raise TradingError(msg) from e

    async def _get_positions_impl(self) -> Dict[str, PositionInfo]:
        """Get current portfolio positions from Alpaca."""
        if not self.api:
            raise TradingError("Alpaca API client not initialized.")

        self.logger.debug("Fetching positions from Alpaca...")

        try:
            alpaca_positions = await asyncio.get_event_loop().run_in_executor(None, self.api.list_positions)
            positions: Dict[str, PositionInfo] = {}

            for pos in alpaca_positions:
                try:
                    positions[pos.symbol] = {
                        "symbol": pos.symbol,
                        "quantity": float(pos.qty),
                        "entry_price": float(pos.avg_entry_price),
                        "current_price": float(pos.current_price),
                        "market_value": float(pos.market_value),
                        "cost_basis": float(pos.cost_basis),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                        "lastday_price": float(pos.lastday_price),
                        "change_today": float(pos.change_today),
                    }
                except (ValueError, TypeError, AttributeError) as map_err:
                    self.logger.error(f"Error mapping position {pos.symbol}: {map_err}. Data: {pos}")
                    continue

            self.logger.info(f"Fetched {len(positions)} positions from Alpaca.")
            return positions
        except AlpacaAPIError as e:
            msg = f"Alpaca API error fetching positions: {e}"
            self.logger.error(msg)
            raise TradingAPIError(msg) from e
        except Exception as e:
            msg = f"Unexpected error fetching positions via Alpaca: {e}"
            self.logger.exception(msg)
            raise TradingError(msg) from e

    async def _get_order_status_impl(self, order_id: str) -> Optional[TradeExecutionDetails]:
        """Get the status of a specific order from Alpaca."""
        if not self.api:
            raise TradingError("Alpaca API client not initialized.")
            
        self.logger.debug(f"Fetching order status from Alpaca for: {order_id}")
        
        try:
            alpaca_order = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.api.get_order(order_id)
            )

            # --- Slippage Monitoring: Calculate and log ---
            client_order_id = alpaca_order.client_order_id
            if alpaca_order.status == 'filled' and client_order_id in self.expected_prices:
                expected_price = self.expected_prices.pop(client_order_id) # Retrieve and remove
                filled_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None
                if filled_price is not None:
                    slippage = filled_price - expected_price if alpaca_order.side == 'buy' else expected_price - filled_price
                    slippage_pct = (slippage / expected_price) * 100 if expected_price != 0 else 0
                    self.logger.info(
                        f"Slippage Report for {client_order_id} ({alpaca_order.side} {alpaca_order.symbol}): "
                        f"Expected={expected_price:.4f}, Filled={filled_price:.4f}, "
                        f"Slippage=${slippage:.4f} ({slippage_pct:.4f}%)"
                    )
                else:
                    self.logger.warning(f"Order {client_order_id} filled but filled_avg_price is None. Cannot calculate slippage.")
            elif client_order_id in self.expected_prices and alpaca_order.status not in ['new', 'partially_filled', 'pending_cancel', 'pending_replace']:
                 # Clean up expected price if order reached a terminal state other than filled
                 del self.expected_prices[client_order_id]
            # ---------------------------------------------

            return {
                "order_id": alpaca_order.id,
                "client_order_id": alpaca_order.client_order_id,
                "symbol": alpaca_order.symbol,
                "quantity": float(alpaca_order.qty or 0),
                "side": alpaca_order.side,
                "order_type": alpaca_order.type,
                "time_in_force": alpaca_order.time_in_force,
                "limit_price": float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
                "stop_price": float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
                "status": alpaca_order.status,
                "fill_price": float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
                "fill_quantity": float(alpaca_order.filled_qty or 0),
                "timestamp": alpaca_order.submitted_at.timestamp() if alpaca_order.submitted_at else asyncio.get_event_loop().time(),
                "error_message": None
            }
        except AlpacaAPIError as e:
            if "order not found" in str(e).lower():
                self.logger.warning(f"Order {order_id} not found in Alpaca: {e}")
                return None
            msg = f"Alpaca API error fetching order status for {order_id}: {e}"
            self.logger.error(msg)
            raise TradingAPIError(msg) from e
        except Exception as e:
            msg = f"Unexpected error fetching order status via Alpaca for {order_id}: {e}"
            self.logger.exception(msg)
            raise TradingError(msg) from e
