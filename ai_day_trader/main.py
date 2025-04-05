"""
Main Entry Point for the AI Day Trading Bot.

Initializes all necessary components and starts the main trading loop,
managed by a central scheduler based on market state.
"""

import asyncio
import signal
import logging
from collections import deque # Import deque
from datetime import datetime, timezone
from typing import Optional, Any

from ai_day_trader.utils.logging_config import configure_logging # Use new utils path
from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.utils.exceptions import ConfigurationError, TradingError # Use new utils path, Import TradingError
# Import both execution systems
from ai_day_trader.trading.execution.live import LiveExecution # Use new trading path
from ai_day_trader.trading.execution.paper import PaperExecution # Use new trading path
from ai_day_trader.trading.trade_manager import TradeManager # Use new trading path
from ai_day_trader.trading.day_trading import DayTradingStrategies # Use new trading path
# Import new AI trader components
from ai_day_trader.strategy import AIStrategyRunner
from ai_day_trader.signal_generator import SignalGenerator
from ai_day_trader.risk_manager import RiskManager
from ai_day_trader.config import load_ai_trader_config
from ai_day_trader.feature_calculator import FeatureCalculator
from ai_day_trader.stock_selector import StockSelector
from ai_day_trader.scheduler import CentralScheduler
# Import new clients
from ai_day_trader.clients.redis_client import init_redis_pool, close_redis_pool, get_async_redis_client
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient
from ai_day_trader.clients.polygon_ws_client import PolygonWebSocketClient, shutdown_all_websocket_clients
# Import predictor from new location
from ai_day_trader.ml.predictor import Predictor as MLEnginePredictor

# Import Alpaca client if available
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.common import URL # Import URL type
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None
    URL = None # Define URL as None if import fails

# Setup logging first
configure_logging() # Correct function name
logger = logging.getLogger(__name__)

# --- Global State (Consider a dedicated state management class later) ---
strategy_runner: Optional[AIStrategyRunner] = None
scheduler: Optional[CentralScheduler] = None
shutdown_event = asyncio.Event()

# --- Scheduled Task Functions ---

async def run_pre_market_tasks():
    """Tasks to run before market open."""
    logger.info("Running pre-market tasks...")
    if strategy_runner and strategy_runner.stock_selector:
        await strategy_runner._update_candidate_symbols(force_refresh=True)
    logger.info("Pre-market tasks finished.")

async def start_trading_loop():
    """Starts the main trading loop when the market opens."""
    logger.info("Market opened. Starting main trading loop.")
    if strategy_runner:
        await strategy_runner.start()
    else:
        logger.error("Cannot start trading loop: Strategy Runner not initialized.")

async def run_eod_closure():
    """Runs the EOD closure logic."""
    logger.info("Running EOD closure task...")
    if strategy_runner:
        await strategy_runner._handle_eod_closure()
    else:
        logger.error("Cannot run EOD closure: Strategy Runner not initialized.")

async def run_post_market_tasks():
    """Tasks to run after market close."""
    logger.info("Running post-market tasks...")
    if strategy_runner:
        await strategy_runner.stop()
    logger.info("Post-market tasks finished.")

async def refresh_symbols_task():
    """Scheduled task to refresh candidate symbols."""
    logger.info("Running scheduled symbol refresh...")
    if strategy_runner:
        await strategy_runner._update_candidate_symbols(force_refresh=True)
    else:
        logger.error("Cannot refresh symbols: Strategy Runner not initialized.")


# --- Main Application ---

async def main_app():
    """Initializes components and runs the scheduler."""
    global strategy_runner, scheduler

    logger.info("Starting AI Day Trading Bot...")
    try:
        config = load_ai_trader_config()
    except ConfigurationError as e:
        logger.critical(f"Failed to load configuration: {e}", exc_info=True)
        return

    redis_client = None
    polygon_ws_client = None
    polygon_rest = None
    stock_selector = None
    alpaca_client = None # Holds the Alpaca client instance needed for scheduler/clock
    execution_system = None # Holds the selected execution system instance

    try:
        logger.info("Initializing core components...")
        await init_redis_pool(config)
        redis_client = await get_async_redis_client()
        logger.info("Redis connection pool initialized.")

        polygon_rest = PolygonRESTClient(config=config)
        await polygon_rest.connection_pool.initialize()

        # --- Initialize Execution System based on Config ---
        if config.EXECUTION_MODE == 'live':
            logger.info("Execution mode: LIVE (Alpaca)")
            if not ALPACA_AVAILABLE:
                raise ConfigurationError("Live execution mode selected, but 'alpaca-trade-api' library is not installed.")

            api_key = config.get("APCA_API_KEY_ID")
            secret_key = config.get("APCA_API_SECRET_KEY")
            if not api_key or not secret_key:
                raise ConfigurationError("Alpaca API Key ID or Secret Key missing for live execution mode.")

            try:
                # LiveExecution initializes its own Alpaca client
                execution_system = LiveExecution(config=config)
                await execution_system.initialize()
                # Get the client instance from the initialized system for scheduler/clock
                alpaca_client = getattr(execution_system, 'api', None)
                if not alpaca_client:
                    raise TradingError("LiveExecution initialized, but failed to get internal Alpaca API client instance.")
                logger.info(f"Initialized LiveExecution (Alpaca URL: {execution_system.base_url}).")
            except Exception as e:
                logger.error(f"Failed to initialize LiveExecution: {e}", exc_info=True)
                raise ConfigurationError(f"Failed to initialize Alpaca client/system for live mode: {e}") from e

        elif config.EXECUTION_MODE == 'paper':
            logger.info("Execution mode: PAPER (Internal Simulation with Live Quotes)")
            # PaperExecution needs the Polygon REST client
            execution_system = PaperExecution(config=config, polygon_client=polygon_rest)
            await execution_system.initialize()
            logger.info("Initialized PaperExecution.")

            # Need an Alpaca client *just* for the market clock if using paper mode
            if ALPACA_AVAILABLE:
                 api_key = config.get("APCA_API_KEY_ID")
                 secret_key = config.get("APCA_API_SECRET_KEY")
                 base_url = config.get_str("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
                 if api_key and secret_key and tradeapi and URL:
                      try:
                           # Initialize a separate Alpaca client instance just for clock/scheduler
                           alpaca_client = tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url=URL(base_url))
                           # Quick check to ensure client is valid
                           await asyncio.get_event_loop().run_in_executor(None, alpaca_client.get_account)
                           logger.info("Initialized separate Alpaca client for market clock checks in paper mode.")
                      except Exception as e:
                           logger.error(f"Failed to initialize Alpaca client for clock checks: {e}")
                           alpaca_client = None # Ensure it's None on failure
                 else:
                      logger.warning("Alpaca keys/library missing, market clock checks unavailable in paper mode.")
            else:
                 logger.warning("Alpaca library not available, market clock checks unavailable in paper mode.")
            # Note: We proceed even if clock checks are unavailable in paper mode, but log warnings.

        else:
             # This case should be caught by config validation, but as a safeguard:
             raise ConfigurationError(f"Unsupported execution mode: {config.EXECUTION_MODE}")

        # --- Continue Initialization with the selected execution_system ---

        # Instantiate TradeManager with the selected execution_system
        trade_manager = TradeManager(execution_system=execution_system, logger=logger)

        ws_api_key = config.get('POLYGON_API_KEY')
        if not ws_api_key: raise ConfigurationError("POLYGON_API_KEY missing.")
        polygon_ws_client = PolygonWebSocketClient(api_key=ws_api_key, config=config)
        await polygon_ws_client.connect()
        if not polygon_ws_client.is_connected(): raise ConnectionError("Failed to connect Polygon WebSocket.")
        logger.info("Polygon WebSocket connected.")

        async def ws_message_handler(msg):
             if msg.get('ev') == 'T' and redis_client:
                  symbol = msg.get('sym')
                  if symbol:
                       key = f"tick:{symbol}"
                       price = msg.get('p')
                       if price is not None:
                            try:
                                await redis_client.hset(key, mapping={'price': price}) # type: ignore
                            except Exception:
                                pass

        # Define the handler *before* initializing strategy_runner so it can be passed
        async def ws_message_handler(msg):
             event_type = msg.get('ev')
             symbol = msg.get('sym')

             if not symbol: return # Ignore messages without symbols

             # --- Update latest price in Redis (from Trades 'T') ---
             if event_type == 'T' and redis_client:
                  key = f"tick:{symbol}"
                  price = msg.get('p')
                  if price is not None:
                       try:
                            # Use hset for potential future fields
                            await redis_client.hset(key, mapping={'price': str(price)}) # Ensure price is string
                            # Optional: Set expiry for tick data?
                       except Exception as e:
                            logger.error(f"Failed to update Redis tick for {symbol}: {e}", exc_info=False) # Log error but don't crash

             # --- Update recent prices cache (from Minute Aggregates 'AM') ---
             elif event_type == 'AM' and strategy_runner: # Check if strategy_runner exists
                  close_price = msg.get('c')
                  if close_price is not None:
                       try:
                            price_float = float(close_price)
                            if symbol not in strategy_runner._recent_prices_cache:
                                # Initialize deque with maxlen if not present
                                strategy_runner._recent_prices_cache[symbol] = deque(maxlen=strategy_runner.peak_detection_window_minutes)
                            strategy_runner._recent_prices_cache[symbol].append(price_float)
                            # logger.debug(f"Updated price cache for {symbol}: {price_float} (Cache size: {len(strategy_runner._recent_prices_cache[symbol])})")
                       except (ValueError, TypeError) as e:
                            logger.warning(f"Could not parse close price '{close_price}' for {symbol} from AM message: {e}")
                       except Exception as e:
                            logger.error(f"Error updating price cache for {symbol}: {e}", exc_info=True)

             # --- Update NBBO quote in Redis (from Quotes 'Q') ---
             elif event_type == 'Q' and redis_client:
                  key = f"nbbo:{symbol}"
                  bid_price = msg.get('bp')
                  ask_price = msg.get('ap')
                  bid_size = msg.get('bs')
                  ask_size = msg.get('as')
                  timestamp_ms = msg.get('t') # Polygon quote timestamp (ms)

                  if bid_price is not None and ask_price is not None and timestamp_ms is not None:
                       nbbo_data = {
                            'bid_price': str(bid_price),
                            'ask_price': str(ask_price),
                            'bid_size': str(bid_size) if bid_size is not None else '0',
                            'ask_size': str(ask_size) if ask_size is not None else '0',
                            'timestamp': str(timestamp_ms)
                       }
                       try:
                            await redis_client.hset(key, mapping=nbbo_data)
                            # Set a short expiry (e.g., 10 seconds) as quotes can become stale quickly
                            await redis_client.expire(key, 10)
                       except Exception as e:
                            logger.error(f"Failed to update Redis NBBO for {symbol}: {e}", exc_info=False)

        polygon_ws_client.message_handler = ws_message_handler
        asyncio.create_task(polygon_ws_client.listen(), name="PolygonWSListener")

        logger.info("Initializing AI strategy components...")
        stock_selector = StockSelector(config=config, polygon_client=polygon_rest)
        feature_calculator = FeatureCalculator(config=config.as_dict())
        # Pass feature_calculator to RiskManager
        risk_manager = RiskManager(config=config, feature_calculator=feature_calculator, redis_client=redis_client)
        ml_predictor = MLEnginePredictor(config=config, polygon_rest_client=polygon_rest, feature_calculator=feature_calculator, redis_client=redis_client)
        await ml_predictor.load_models()
        day_trading_strategies = DayTradingStrategies(config=config, polygon_rest_client=polygon_rest, redis_client=redis_client, logger=logger) if execution_system else None
        signal_generator = SignalGenerator(config=config, day_trading_strategies=day_trading_strategies, redis_client=redis_client, ml_predictor=ml_predictor, feature_calculator=feature_calculator)

        # Pass the potentially separate alpaca_client for clock checks
        strategy_runner = AIStrategyRunner(
            config=config, redis_client=redis_client, polygon_ws_client=polygon_ws_client,
            polygon_rest_client=polygon_rest, trade_manager=trade_manager, signal_generator=signal_generator,
            risk_manager=risk_manager, execution_system=execution_system, stock_selector=stock_selector,
            ml_predictor=ml_predictor, feature_calculator=feature_calculator,
            alpaca_client=alpaca_client # Pass the client used for clock/scheduler
        )
        # Now strategy_runner exists and can be accessed by the ws_message_handler closure

        logger.info("Core components initialization complete.")

        # Scheduler requires an Alpaca client for market timing
        if not alpaca_client:
             logger.error("Alpaca client (for market clock) not available. Cannot start scheduler. Exiting.")
             # Clean up already initialized components before exiting
             await shutdown_all_websocket_clients()
             if polygon_rest and hasattr(polygon_rest, 'close'): await polygon_rest.close()
             await close_redis_pool()
             return

        scheduler = CentralScheduler(config=config, alpaca_client=alpaca_client)
        tasks_to_schedule = {
            "pre_market": run_pre_market_tasks,
            "market_open": start_trading_loop,
            "market_close": run_eod_closure,
            "post_market": run_post_market_tasks,
            "refresh_symbols": refresh_symbols_task
        }
        await scheduler.schedule_daily_tasks(tasks_to_schedule)

        scheduler.start()
        logger.info("Scheduler started. Waiting for shutdown signal...")

        await shutdown_event.wait()

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
    except ConnectionError as e:
         logger.error(f"Connection error during startup: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Critical error during initialization or runtime: {e}", exc_info=True)
    finally:
        logger.info("Initiating shutdown sequence...")

        if scheduler:
            await scheduler.stop()
        if strategy_runner:
            await strategy_runner.stop()

        await shutdown_all_websocket_clients()
        if polygon_rest and hasattr(polygon_rest, 'close'):
             await polygon_rest.close()
        await close_redis_pool()
        logger.info("Redis connection pool closed.")

        logger.info("AI Day Trading Bot shut down complete.")


# --- Signal Handling and Entry Point ---

def handle_shutdown_signal(sig, frame):
    """Sets the shutdown event when a signal is received."""
    logger.info(f"Received signal {sig}. Initiating graceful shutdown...")
    shutdown_event.set()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        asyncio.run(main_app())
    except KeyboardInterrupt:
        logger.info("Manual shutdown initiated.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main entry point: {e}", exc_info=True)
