"""
AI Day Trading Strategy Runner.

Orchestrates the trading process by:
- Running on a defined schedule (e.g., every N seconds).
- Checking market status and EOD conditions.
- Fetching signals from the SignalGenerator.
- Getting position sizing and risk checks from the RiskManager.
- Monitoring open positions for exit conditions.
- Interacting with the TradeManager/ExecutionSystem to place orders.
"""

import asyncio
import logging
import time
from collections import deque # Import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, TYPE_CHECKING, List
import numpy as np
import pandas as pd

# Import necessary components
from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.utils.exceptions import TradingError # Use new utils path
from ai_day_trader.trading.trade_manager import TradeManager # Use new trading path
# Import components specific to this AI trader implementation
from ai_day_trader.signal_generator import SignalGenerator
from ai_day_trader.risk_manager import RiskManager
from ai_day_trader.feature_calculator import FeatureCalculator
from ai_day_trader.peak_detector import PeakDetector
from ai_day_trader.stock_selector import StockSelector # Import new selector
# Import reused components for data handling
from ai_day_trader.clients.polygon_ws_client import PolygonWebSocketClient
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient

if TYPE_CHECKING:
    try:
        from redis.asyncio import Redis as RedisClient # Define alias inside TYPE_CHECKING
    except ImportError:
        RedisClient = Any
    # Import directly from execution subdirectory
    from ai_day_trader.trading.execution.base import PositionInfo, ExecutionSystem # Use new trading path
    AlpacaRESTClientType = Any
else:
    PositionInfo = Dict
    AlpacaRESTClientType = Any


logger = logging.getLogger(__name__)

class AIStrategyRunner:
    """Orchestrates the AI day trading strategy execution."""

    def __init__(
        self,
        config: Config,
        redis_client: Optional['RedisClient'], # Revert to string literal
        polygon_ws_client: PolygonWebSocketClient,
        polygon_rest_client: PolygonRESTClient,
        trade_manager: TradeManager,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        execution_system: 'ExecutionSystem',
        stock_selector: StockSelector, # Added stock_selector dependency
        ml_predictor: Optional[Any] = None,
        feature_calculator: Optional[FeatureCalculator] = None,
        alpaca_client: Optional[AlpacaRESTClientType] = None # Add alpaca_client parameter
    ):
        """Initialize the AIStrategyRunner."""
        self.config = config
        self.redis_client = redis_client
        self.ws_client = polygon_ws_client
        self.rest_client = polygon_rest_client
        self.trade_manager = trade_manager
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.execution_system = execution_system
        self.stock_selector = stock_selector # Store stock_selector instance
        self.ml_predictor = ml_predictor
        self.feature_calculator = feature_calculator
        self.logger = logger # Assign logger to self
        self.peak_detector = PeakDetector(config=config, logger_instance=self.logger) # Use self.logger
        self.alpaca_client = alpaca_client # Assign passed client

        # self.symbols = config.get_list("SYMBOLS", []) # Removed static symbols
        self.current_candidate_symbols: List[str] = [] # Store dynamic symbols

        self.loop_interval = config.get_float("STRATEGY_LOOP_INTERVAL_SECONDS", 10.0)
        self.eod_close_minutes_before = config.get_int("EOD_CLOSE_MINUTES_BEFORE", 15)
        self.stop_loss_pct = config.get_float("STOP_LOSS_PCT", 0.02)
        self.peak_exit_drop_threshold = config.get_float("PEAK_EXIT_DROP_THRESHOLD", 0.01)
        self.ml_exit_threshold = config.get_float("ML_EXIT_THRESHOLD", 0.75)
        self.peak_detection_window_minutes = config.get_int("PEAK_DETECTION_WINDOW_MINUTES", 60) # Configurable window

        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None
        self.loop = asyncio.get_event_loop()
        # Cache for recent minute bar close prices {symbol: deque([price1, price2,...], maxlen=window_minutes)}
        self._recent_prices_cache: Dict[str, deque] = {}

        # Removed logic trying to get alpaca_client from execution_system
        if not self.alpaca_client:
             self.logger.warning("Alpaca client not provided. Market clock checks unavailable.")

        if not self.feature_calculator:
             self.logger.error("FeatureCalculator not provided. Feature-based ML prediction will fail.")


        self.logger.info("AIStrategyRunner initialized.")

    async def start(self):
        """Starts the main strategy execution loop."""
        if self._running:
            self.logger.warning("Strategy runner is already running.")
            return

        self.logger.info("Starting AI Strategy Runner loop...")
        self._running = True
        if not self.ws_client or not self.ws_client.is_connected():
             self.logger.error("WebSocket client not connected. Cannot start strategy loop.")
             self._running = False
             return

        # Initial stock selection
        await self._update_candidate_symbols()

        self._main_loop_task = asyncio.create_task(self._run_loop(), name="AIStrategyLoop")
        self.logger.info(f"Strategy loop started with interval: {self.loop_interval}s")

    async def stop(self):
        """Stops the main strategy execution loop."""
        if not self._running:
            self.logger.info("Strategy runner is not running.")
            return

        self.logger.info("Stopping AI Strategy Runner loop...")
        self._running = False
        if self._main_loop_task and not self._main_loop_task.done():
            self._main_loop_task.cancel()
            try:
                await asyncio.wait_for(self._main_loop_task, timeout=5.0)
                self.logger.info("Strategy loop stopped.")
            except asyncio.CancelledError:
                self.logger.info("Strategy loop task already cancelled.")
            except asyncio.TimeoutError:
                self.logger.warning("Strategy loop task did not cancel within timeout.")
            except Exception as e:
                self.logger.error(f"Error stopping strategy loop task: {e}", exc_info=True)
        self._main_loop_task = None
        self.logger.info("AI Strategy Runner stopped.")

    async def _update_candidate_symbols(self, force_refresh: bool = False):
        """Fetches and updates the list of candidate symbols."""
        self.logger.info(f"Updating candidate symbols (force_refresh={force_refresh})...")
        try:
            self.current_candidate_symbols = await self.stock_selector.select_candidate_symbols(force_refresh=force_refresh)
            self.logger.info(f"Updated candidate symbols: {len(self.current_candidate_symbols)} tickers.")
            # TODO: Potentially subscribe/unsubscribe to WebSocket data based on new/removed symbols
        except Exception as e:
            self.logger.error(f"Failed to update candidate symbols: {e}", exc_info=True)
            # Keep using the old list if update fails? Or clear it? Clearing for safety.
            self.current_candidate_symbols = []

    async def _run_loop(self):
        """The main periodic loop for executing the trading strategy."""
        # Counter to refresh symbols less frequently than the main loop
        refresh_counter = 0
        symbol_refresh_interval = 360 # e.g., refresh symbols every 360 * 10s = 1 hour

        while self._running:
            start_time = time.monotonic()
            try:
                market_open = await self._is_market_open()
                if not market_open:
                    self.logger.debug("Market closed. Sleeping.")
                    await asyncio.sleep(60)
                    continue

                # Refresh candidate symbols periodically
                refresh_counter += 1
                if refresh_counter >= symbol_refresh_interval:
                    await self._update_candidate_symbols()
                    refresh_counter = 0 # Reset counter

                eod_initiated = await self._handle_eod_closure()
                if eod_initiated:
                    self.logger.info("EOD closure initiated. Halting regular strategy execution for this cycle.")
                    await asyncio.sleep(self.loop_interval)
                    continue

                if not self.current_candidate_symbols:
                     self.logger.warning("No candidate symbols available. Skipping trading cycle.")
                     await asyncio.sleep(self.loop_interval) # Wait before retrying symbol fetch
                     continue

                current_positions = await self._get_current_positions()
                current_portfolio_value = self._calculate_portfolio_value(current_positions)
                remaining_daily_limit = await self.risk_manager.get_remaining_daily_limit()

                # --- Calculate Features Once ---
                symbols_requiring_features = set(self.current_candidate_symbols) | set(current_positions.keys())
                latest_features_map: Dict[str, Optional[Dict[str, Any]]] = {}
                if self.feature_calculator and symbols_requiring_features:
                    self.logger.debug(f"Calculating features for {len(symbols_requiring_features)} symbols...")
                    feature_tasks = {
                        symbol: asyncio.create_task(self.get_latest_features(symbol))
                        for symbol in symbols_requiring_features
                    }
                    results = await asyncio.gather(*feature_tasks.values(), return_exceptions=True)
                    latest_features_map = {
                        symbol: result if not isinstance(result, Exception) else None
                        for symbol, result in zip(feature_tasks.keys(), results)
                    }
                    error_count = sum(1 for res in results if isinstance(res, Exception))
                    if error_count > 0:
                        self.logger.warning(f"Encountered {error_count} errors during feature calculation.")
                # --- End Feature Calculation ---


                # Pass the dynamic list of symbols and pre-calculated features
                await self._process_entries(
                    self.current_candidate_symbols,
                    current_positions,
                    current_portfolio_value,
                    remaining_daily_limit,
                    latest_features_map # Pass features map
                )

                # Refresh positions in case entries occurred
                current_positions_after_entry = await self._get_current_positions()
                await self._monitor_exits(current_positions_after_entry, latest_features_map) # Pass features map

            except asyncio.CancelledError:
                self.logger.info("Strategy loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in strategy execution cycle: {e}", exc_info=True)
                await asyncio.sleep(self.loop_interval * 2)

            elapsed_time = time.monotonic() - start_time
            sleep_duration = max(0, self.loop_interval - elapsed_time)
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

    # --- Helper Methods ---

    async def _is_market_open(self) -> bool:
        """Checks market open status using the Alpaca client."""
        if not self.alpaca_client: return False
        try:
            clock = await self.loop.run_in_executor(None, self.alpaca_client.get_clock)
            return clock.is_open
        except Exception as e:
            self.logger.error(f"Failed to check market clock: {e}", exc_info=True)
            return False

    async def _handle_eod_closure(self) -> bool:
        """Checks EOD time and initiates closure if needed. Returns True if EOD closure active."""
        try:
            if not self.alpaca_client: return False

            clock = await self.loop.run_in_executor(None, self.alpaca_client.get_clock)
            now_utc = datetime.now(timezone.utc)
            next_close_utc = clock.next_close.astimezone(timezone.utc)
            time_until_close = next_close_utc - now_utc

            if timedelta(minutes=0) < time_until_close <= timedelta(minutes=self.eod_close_minutes_before):
                self.logger.info(f"Market closing soon ({time_until_close}). Initiating EOD position closure.")
                positions = await self._get_current_positions()
                if not positions: return True

                self.logger.info(f"Closing {len(positions)} open positions for EOD...")
                close_tasks = []
                for symbol, pos_info in positions.items():
                    qty_str = pos_info.get('qty')
                    # Add check for None before converting to float
                    try: qty = abs(float(qty_str)) if qty_str is not None else 0.0
                    except (ValueError, TypeError): continue
                    if qty <= 0: continue

                    # Add another check before this float conversion
                    if qty_str is None:
                         self.logger.error(f"qty_str became None unexpectedly for {symbol} before side determination.")
                         continue
                    side = 'sell' if float(qty_str) > 0 else 'buy'
                    close_order = {'symbol': symbol, 'side': side, 'quantity': qty, 'order_type': 'market', 'time_in_force': 'ioc'}
                    close_tasks.append(self._execute_order_managed(close_order, "EOD Closure"))

                    await asyncio.gather(*close_tasks, return_exceptions=True)
                self.logger.info("EOD position closure process initiated.")
                return True

        except Exception as e:
             self.logger.error(f"Critical error during EOD check/closure: {e}", exc_info=True)
             return True # Halt trading on critical EOD error

        return False

    async def _get_current_positions(self) -> Dict[str, 'PositionInfo']:
        """Gets current positions via the execution system."""
        if not self.execution_system: return {}
        try:
            positions = await self.execution_system.get_positions()
            return positions if isinstance(positions, dict) else {}
        except Exception as e:
            self.logger.error(f"Failed to get current positions: {e}", exc_info=True)
            return {}

    def _calculate_portfolio_value(self, positions: Dict[str, 'PositionInfo']) -> float:
        """Calculates total market value of positions."""
        total_value = 0.0
        for pos in positions.values():
            try:
                mv_str = pos.get('market_value')
                if mv_str is not None: total_value += abs(float(mv_str))
            except (ValueError, TypeError): continue
        # Add cash balance if available from execution system? For now, just position value.
        return total_value

    async def _process_entries(self, symbols_to_scan: List[str], current_positions: Dict[str, 'PositionInfo'], current_portfolio_value: float, remaining_daily_limit: float, latest_features_map: Dict[str, Optional[Dict[str, Any]]]):
        """Generates signals for candidate symbols, checks risk, and executes entry orders."""
        # Pass the dynamic list of symbols and features map to the signal generator
        entry_signals = await self.signal_generator.generate_signals(symbols_to_scan, latest_features_map)
        if not entry_signals: return

        entry_tasks = []
        for signal in entry_signals:
            symbol = signal.get('symbol')
            side = signal.get('side')
            # Check if we already have a position in this symbol
            if not symbol or side not in ['buy', 'sell'] or symbol in current_positions:
                continue

            entry_tasks.append(
                self._evaluate_and_execute_entry(
                    symbol, side, current_portfolio_value, remaining_daily_limit, signal.get('source', 'Unknown'), latest_features_map.get(symbol) # Pass specific symbol's features
                )
            )
        if entry_tasks: await asyncio.gather(*entry_tasks)


    async def _evaluate_and_execute_entry(self, symbol: str, side: str, current_portfolio_value: float, remaining_limit: float, signal_source: str, latest_features: Optional[Dict[str, Any]]):
        """Evaluates risk/size and executes a single entry order, using pre-calculated features."""
        try:
            # Use NBBO ask/bid for more realistic entry price estimation
            nbbo = await self._get_latest_nbbo(symbol)
            entry_price_estimate = None
            if nbbo:
                entry_price_estimate = nbbo['ask_price'] if side == 'buy' else nbbo['bid_price']
                if entry_price_estimate <= 0: entry_price_estimate = None # Ensure positive price

            # Fallback to last trade price if NBBO is unavailable or invalid
            if entry_price_estimate is None:
                self.logger.warning(f"Using last trade price for entry estimate for {symbol} (NBBO unavailable/invalid).")
                entry_price_estimate = await self._get_latest_price(symbol) # _get_latest_price already falls back

            if entry_price_estimate is None:
                self.logger.error(f"Could not determine entry price estimate for {symbol}. Skipping entry.")
                return

            # Features are now passed in, no need to fetch again unless they are None and needed
            if self.risk_manager.stop_loss_type == "atr" and latest_features is None:
                 self.logger.warning(f"Features for {symbol} were None, ATR stop sizing will fallback to percentage.")

            sized_qty, entry_value, calculated_stop_price = await self.risk_manager.calculate_position_size(
                symbol, side, entry_price_estimate, current_portfolio_value, remaining_limit, latest_features # Pass features directly
            )

            if sized_qty is not None and entry_value is not None and sized_qty > 0 and calculated_stop_price is not None:
                entry_order = {'symbol': symbol, 'side': side, 'quantity': sized_qty, 'order_type': 'market', 'time_in_force': 'day'}
                order_submitted = await self._execute_order_managed(entry_order, f"Entry Signal ({signal_source})")
                if order_submitted:
                    # Use the stop price calculated by RiskManager to determine potential loss
                    potential_loss = abs(entry_price_estimate - calculated_stop_price) * sized_qty
                    self.logger.info(f"Updating daily risk limit by potential loss: ${potential_loss:.2f} for {symbol} entry.")
                    await self.risk_manager.update_daily_limit_used(potential_loss)
            else:
                 self.logger.debug(f"Skipping entry for {symbol}: Position sizing failed, quantity is zero, or stop price calculation failed.")

        except Exception as e:
            self.logger.error(f"Error evaluating/executing entry for {symbol}: {e}", exc_info=True)


    async def _monitor_exits(self, current_positions: Dict[str, 'PositionInfo'], latest_features_map: Dict[str, Optional[Dict[str, Any]]]):
        """Checks exit conditions for all open positions, using pre-calculated features."""
        if not current_positions: return
        self.logger.debug(f"Monitoring {len(current_positions)} positions for exits...")
        exit_tasks = [
            self._check_exit_conditions(symbol, pos_info, latest_features_map.get(symbol))
            for symbol, pos_info in current_positions.items()
        ]
        await asyncio.gather(*exit_tasks)

    async def _check_exit_conditions(self, symbol: str, pos_info: 'PositionInfo', latest_features: Optional[Dict[str, Any]]):
        """Checks all exit conditions (peak, ML, stop) for a single position, using pre-calculated features."""
        exit_reason = None
        try:
            pos_qty_str = pos_info.get('qty')
            pos_qty = float(pos_qty_str) if pos_qty_str is not None else 0.0
            if abs(pos_qty) < 1e-9: return # Position effectively closed
        except (ValueError, TypeError):
            self.logger.error(f"Invalid quantity '{pos_info.get('qty')}' for {symbol} in exit check.")
            return

        # Use NBBO bid/ask for more accurate exit checks
        nbbo = await self._get_latest_nbbo(symbol)
        current_price_for_check = None
        if nbbo:
            # If long, check against bid; if short, check against ask
            current_price_for_check = nbbo['bid_price'] if pos_qty > 0 else nbbo['ask_price']
            if current_price_for_check <= 0: current_price_for_check = None # Ensure positive price
        else:
            self.logger.warning(f"NBBO unavailable for {symbol} exit check, falling back to last trade price.")
            current_price_for_check = await self._get_latest_price(symbol) # Fallback

        if current_price_for_check is None:
            self.logger.error(f"Could not determine current price for exit check on {symbol}.")
            return

        # 1. Peak Detection Exit (Still uses mid-price or last trade from _get_latest_price for history)
        # Note: Peak detection inherently uses historical close prices, not live bid/ask.
        # We use the NBBO-derived price for the *current* price comparison against the detected peak/trough.
        exit_reason = await self._check_peak_exit(symbol, pos_qty, current_price_for_check)

        # 2. ML Model Exit
        if not exit_reason:
             # Pass position info and base features (ML model might use internal price or features)
             exit_reason = await self._check_ml_exit(symbol, pos_info, latest_features)

        # 3. Stop Loss Exit
        if not exit_reason:
            # Pass the NBBO-derived price for the check against the calculated stop level
            exit_reason = await self._check_stop_loss(symbol, pos_info, pos_qty, current_price_for_check, latest_features) # Pass features

        # Execute exit if reason found
        if exit_reason:
            qty_to_close = abs(pos_qty)
            side = 'sell' if pos_qty > 0 else 'buy'
            exit_order = {'symbol': symbol, 'side': side, 'quantity': qty_to_close, 'order_type': 'market', 'time_in_force': 'ioc'}
            await self._execute_order_managed(exit_order, exit_reason)

    async def _check_peak_exit(self, symbol: str, pos_qty: float, current_price: float) -> Optional[str]:
        """Checks peak/trough detection for exit signal using cached recent prices."""
        try:
            # Retrieve prices from the cache
            price_deque = self._recent_prices_cache.get(symbol)
            if not price_deque or len(price_deque) < self.peak_detector.config.distance_threshold:
                self.logger.debug(f"Not enough cached price data ({len(price_deque) if price_deque else 0} points) for peak detection on {symbol}.")
                # Optional: Fallback to REST API if cache is insufficient? For now, just skip.
                # recent_data = await self.get_historical_aggregates(symbol, timespan="minute", multiplier=1, minutes=self.peak_detection_window_minutes)
                # if recent_data is not None and not recent_data.empty and 'close' in recent_data.columns:
                #     prices = recent_data['close'].tolist()
                # else:
                #     return None # Still no data
                return None

            prices = list(price_deque) # Convert deque to list for detector

            if pos_qty > 0: # Long position, look for peaks
                peaks = self.peak_detector.find_peaks(prices, symbol=symbol)
                if peaks:
                    last_peak_index = peaks[-1]
                    # Ensure index is valid before accessing price
                    if 0 <= last_peak_index < len(prices):
                        last_peak_price = prices[last_peak_index]
                        if current_price < last_peak_price * (1 - self.peak_exit_drop_threshold):
                            return f"Peak ({last_peak_index}@{last_peak_price:.2f}) detected from cache, price dropped to {current_price:.2f}"
                    else:
                        self.logger.warning(f"Invalid peak index {last_peak_index} for prices length {len(prices)} on {symbol}")

            elif pos_qty < 0: # Short position, look for troughs (inverted peaks)
                troughs = self.peak_detector.find_troughs(prices, symbol=symbol)
                if troughs:
                    last_trough_index = troughs[-1]
                    # Ensure index is valid
                    if 0 <= last_trough_index < len(prices):
                        last_trough_price = prices[last_trough_index]
                        if current_price > last_trough_price * (1 + self.peak_exit_drop_threshold):
                            return f"Trough ({last_trough_index}@{last_trough_price:.2f}) detected from cache, price bounced to {current_price:.2f}"
                    else:
                        self.logger.warning(f"Invalid trough index {last_trough_index} for prices length {len(prices)} on {symbol}")

        except Exception as e:
            self.logger.error(f"Error during peak detection check using cache for {symbol}: {e}", exc_info=True)
        return None

    async def _check_ml_exit(self, symbol: str, pos_info: 'PositionInfo', latest_features: Optional[Dict[str, Any]]) -> Optional[str]:
        """Checks the ML model for an exit signal, using pre-calculated features."""
        if not self.ml_predictor or not hasattr(self.ml_predictor, 'predict_exit'):
            return None

        try:
            # Pass pre-calculated base features to predict_exit
            prediction_result = await self.ml_predictor.predict_exit(symbol, pos_info, base_features=latest_features)

            if isinstance(prediction_result, dict): # Check if result is a dict
                 pred = prediction_result.get('prediction')
                 prob = prediction_result.get('probability') # Assuming probability of EXIT
                 # Assuming 1 means exit signal
                 if pred == 1 and prob is not None and prob >= self.ml_exit_threshold:
                      return f"ML Exit Signal (Score: {prob:.3f})"
                 elif pred == 1 and prob is None: # Handle case where only prediction is returned
                      return "ML Exit Signal (Triggered)"
            # Handle older format if predict_exit returns float/int directly (less likely now)
            elif isinstance(prediction_result, (float, int)) and prediction_result >= self.ml_exit_threshold:
                 return f"ML Exit Signal (Score: {prediction_result:.3f})"
            elif isinstance(prediction_result, bool) and prediction_result:
                  return "ML Exit Signal (Triggered)"

        except Exception as e:
            self.logger.error(f"Error during ML exit check for {symbol}: {e}", exc_info=True)

        return None

    async def _check_stop_loss(self, symbol: str, pos_info: 'PositionInfo', pos_qty: float, current_price: float, latest_features: Optional[Dict[str, Any]]) -> Optional[str]:
        """Checks stop loss based on configured type (percentage or ATR), using pre-calculated features."""
        stop_price = None
        stop_type_used = "percentage" # Default
        try:
            entry_price_str = pos_info.get('avg_entry_price')
            if entry_price_str is None:
                self.logger.error(f"Missing avg_entry_price for {symbol}. Cannot check stop loss.")
                return None
            entry_price = float(entry_price_str)

            # --- ATR Stop Loss Logic ---
            if self.risk_manager.stop_loss_type == "atr":
                # Use pre-calculated features passed as argument
                if latest_features and self.risk_manager.atr_feature_name in latest_features:
                    atr_value = latest_features[self.risk_manager.atr_feature_name]
                    if atr_value is not None and atr_value > 0:
                        stop_price = entry_price - (atr_value * self.risk_manager.atr_stop_multiplier) if pos_qty > 0 else entry_price + (atr_value * self.risk_manager.atr_stop_multiplier)
                        stop_type_used = f"ATR ({atr_value:.4f} * {self.risk_manager.atr_stop_multiplier})"
                        self.logger.debug(f"Checking ATR stop for {symbol}. Entry: {entry_price:.2f}, Stop: {stop_price:.2f}")
                    else:
                        self.logger.warning(f"Invalid ATR value ({atr_value}) for {symbol} during stop check. Falling back to percentage.")
                else:
                    self.logger.warning(f"ATR feature '{self.risk_manager.atr_feature_name}' not found for {symbol} during stop check. Falling back to percentage.")

            # --- Percentage Stop Loss Logic (Fallback or Primary) ---
            if stop_price is None: # If ATR failed or type is percentage
                if self.stop_loss_pct > 0:
                    stop_price = entry_price * (1 - self.stop_loss_pct) if pos_qty > 0 else entry_price * (1 + self.stop_loss_pct)
                    stop_type_used = f"Percentage ({self.stop_loss_pct*100}%)"
                    self.logger.debug(f"Checking Percentage stop for {symbol}. Entry: {entry_price:.2f}, Stop: {stop_price:.2f}")
                else:
                    self.logger.warning(f"Stop loss percentage is zero or negative for {symbol}. Cannot check percentage stop.")
                    return None # No stop loss configured effectively

            # --- Check Trigger ---
            if stop_price is not None:
                # Use the NBBO-derived current_price for the check
                if (pos_qty > 0 and current_price <= stop_price) or \
                   (pos_qty < 0 and current_price >= stop_price):
                    return f"Stop Loss ({stop_type_used}) triggered at {current_price:.2f} (Stop: {stop_price:.2f})"

        except (ValueError, TypeError) as e:
             self.logger.error(f"Error calculating stop loss for {symbol}: Invalid entry price '{pos_info.get('avg_entry_price')}' or feature value. Error: {e}")
        except Exception as e:
            self.logger.error(f"Error during Stop Loss check for {symbol}: {e}", exc_info=True)
        return None


    async def _execute_order_managed(self, order: Dict, reason: str) -> bool:
         """Helper to execute order via TradeManager, returns True if submission attempted."""
         symbol = order.get('symbol', 'unknown')
         side = order.get('side', 'unknown')
         qty = order.get('quantity', 0)
         self.logger.info(f"Attempting order execution for {symbol} ({reason}): {side} {qty}")
         try:
              if self.trade_manager:
                   await self.trade_manager.execute_trade(order)
                   self.logger.info(f"Order submission successful for {symbol} ({reason}).")
                   return True
              else:
                   self.logger.error(f"Cannot execute order for {symbol}: TradeManager not available.")
                   return False
         except TradingError as te:
              self.logger.error(f"TradingError executing order for {symbol} ({reason}): {te}", exc_info=False)
              return False
         except Exception as e:
              self.logger.error(f"Unexpected failure executing order for {symbol} ({reason}): {e}", exc_info=True)
              return False

    # --- Proxy/Helper methods needed by components passed 'self' as engine ---

    async def get_historical_aggregates(self, *args, **kwargs):
         """Proxy method to access Polygon REST client."""
         if not self.rest_client: return None
         return await self.rest_client.get_aggregates(*args, **kwargs)

    async def get_latest_features(self, symbol: str) -> Optional[Dict[str, Any]]:
         """Gets latest features using the FeatureCalculator."""
         if not self.feature_calculator:
             self.logger.error("FeatureCalculator not available for get_latest_features.")
             return None
         try:
             lookback_minutes = 60
             if hasattr(self.feature_calculator, 'macd_slow_period'):
                 lookback_minutes = max(lookback_minutes, self.feature_calculator.macd_slow_period + 5)

             hist_data = await self.get_historical_aggregates(symbol, timespan="minute", multiplier=1, minutes=lookback_minutes)
             if hist_data is not None and not hist_data.empty:
                 return await self.feature_calculator.calculate_features(symbol, hist_data)
             else:
                 self.logger.warning(f"Could not get historical data for {symbol} to calculate latest features.")
                 return None
         except Exception as e:
              self.logger.error(f"Error getting latest features for {symbol}: {e}", exc_info=True)
              return None

    async def _get_latest_tick_data(self, symbol: str) -> Optional[Dict[str, str]]:
         """Gets latest tick data from Redis."""
         if not self.redis_client: return None
         key = f"tick:{symbol}"
         try:
             tick_data_bytes = await self.redis_client.hgetall(key)
             return {k.decode('utf-8'): v.decode('utf-8') for k, v in tick_data_bytes.items()} if tick_data_bytes else None
         except Exception as e:
             self.logger.error(f"Error getting latest tick data for {symbol} from Redis: {e}")
             return None

    async def _get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Gets the latest price, prioritizing NBBO mid-price, falling back to last trade.
        """
        # 1. Try getting NBBO quote
        nbbo_data = await self._get_latest_nbbo(symbol)
        if nbbo_data and 'bid_price' in nbbo_data and 'ask_price' in nbbo_data:
            bid = nbbo_data['bid_price']
            ask = nbbo_data['ask_price']
            if bid > 0 and ask > 0: # Ensure valid bid/ask
                mid_price = (bid + ask) / 2.0
                # self.logger.debug(f"Using NBBO mid-price for {symbol}: {mid_price:.4f}")
                return mid_price
            else:
                self.logger.warning(f"Invalid bid/ask prices in NBBO data for {symbol}: Bid={bid}, Ask={ask}")

        # 2. Fallback to last trade price if NBBO fails or is invalid
        self.logger.debug(f"NBBO unavailable or invalid for {symbol}, falling back to last trade price.")
        tick_data = await self._get_latest_tick_data(symbol)
        if tick_data and 'price' in tick_data:
              try:
                  return float(tick_data['price'])
              except (ValueError, TypeError):
                  self.logger.error(f"Invalid price format in tick data for {symbol}: {tick_data['price']}")
                  return None
        self.logger.warning(f"No latest price found in tick data for {symbol}.")
        # Fallback: Try getting last aggregate close if tick data fails?
        # aggregates = await self.get_historical_aggregates(symbol, timespan="minute", multiplier=1, limit=1)
        # if aggregates is not None and not aggregates.empty and 'close' in aggregates.columns:
        #     last_close = aggregates['close'].iloc[-1]
        #     self.logger.warning(f"Using last aggregate close price ({last_close}) for {symbol} as fallback.")
        #     return last_close
        return None

    # --- NBBO Helper ---
    async def _get_latest_nbbo(self, symbol: str) -> Optional[Dict[str, float]]:
        """Gets the latest NBBO quote data from Redis."""
        if not self.redis_client: return None
        key = f"nbbo:{symbol}"
        try:
            nbbo_data_bytes = await self.redis_client.hgetall(key)
            if not nbbo_data_bytes:
                # self.logger.debug(f"No NBBO data found in Redis for {symbol}.")
                return None

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
                         if k in required_keys: all_keys_present = False; break
                         nbbo_data[k] = 0.0 # Or np.nan? Defaulting to 0 for sizes
                except (ValueError, TypeError, UnicodeDecodeError) as e:
                    self.logger.warning(f"Invalid value format for key '{k}' in NBBO data for {symbol}: {v_bytes}. Error: {e}")
                    return None # Invalidate cache entry if any value is bad

            # Check if essential keys are present after decoding attempt
            if not all_keys_present:
                 self.logger.warning(f"Missing required keys in NBBO data for {symbol} after decoding.")
                 return None
            for req_key in required_keys:
                if req_key not in nbbo_data:
                    self.logger.warning(f"Missing required key '{req_key}' in NBBO data for {symbol}.")
                    return None # Return None if essential keys are missing

            # Optional: Check timestamp staleness
            quote_timestamp_ms = nbbo_data.get('timestamp', 0)
            current_time_ms = time.time() * 1000
            # Allow some staleness, e.g., 15 seconds (Redis expiry is 10s, this adds buffer)
            if current_time_ms - quote_timestamp_ms > 15000:
                 self.logger.warning(f"Stale NBBO data for {symbol} ({(current_time_ms - quote_timestamp_ms)/1000:.1f}s old).")
                 return None # Treat stale data as missing

            # Ensure sizes are present, default to 0 if not
            nbbo_data['bid_size'] = nbbo_data.get('bid_size', 0.0)
            nbbo_data['ask_size'] = nbbo_data.get('ask_size', 0.0)

            return nbbo_data

        except Exception as e:
            self.logger.error(f"Error getting latest NBBO data for {symbol} from Redis: {e}")
            return None
