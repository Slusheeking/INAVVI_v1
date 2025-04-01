"""Core trading engine implementation."""
import os
import asyncio
import time
from datetime import datetime, timedelta, timezone # Import timezone
from dotenv import load_dotenv # Keep for potential .env loading by Config
from utils.logging_config import get_logger # Use configured logger
from typing import Optional, Dict, List, Any, TYPE_CHECKING
import numpy as np
import pandas as pd # For timestamp conversion and aggregate handling
import torch

from utils.metrics_registry import MetricsRegistry
from utils.redis_helpers import RedisClient
from utils.gpu_utils import configure_gpu
from utils.config import Config, get_config, InvalidConfigurationError, MissingConfigurationError # Import Config system and specific config exceptions
from utils.exceptions import ( # Import custom exceptions
    ConfigurationError, APIConnectionError, APITimeoutError, APIError as TradingAPIError,
    WebSocketError, DataFormatError # Add missing exceptions
)
from api_clients.polygon_ws import PolygonWebSocketClient
from api_clients.polygon_rest import PolygonRESTClient

# Type hint for ExecutionSystem without circular import
if TYPE_CHECKING:
    from trading_engine.execution import ExecutionSystem
# Import Alpaca SDK if available for market clock
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError as AlpacaAPIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    AlpacaAPIError = Exception
    tradeapi = None
    # Initialize logger early for module-level warnings
    module_logger = get_logger(__name__) # Use configured logger
    module_logger.warning("alpaca-trade-api not found. Market clock functionality will be disabled.")


# load_dotenv() # Config class handles .env loading if needed via its initialization

class TradingEngine:
    """
    Core trading engine class handling market analysis, execution, real-time data feed,
    and operational scheduling based on market hours.
    """

    def __init__(
        self,
        config_obj: Optional[Config] = None, # Accept Config object
        redis_client: Optional[RedisClient] = None,
        polygon_rest_client: Optional[PolygonRESTClient] = None,
        alpaca_client_override: Optional[tradeapi.REST] = None,
        # Allow injecting execution system later or during init
        execution_system: Optional['ExecutionSystem'] = None
    ):
        """Initialize trading engine with optional configuration and clients."""
        self.config = config_obj or get_config() # Use provided or global config
        self.logger = get_logger(__name__) # Use configured logger
        self.metrics = MetricsRegistry()
        self.redis = redis_client or RedisClient(config=self.config) # Pass config to RedisClient
        self.gpu_enabled = configure_gpu(config=self.config) # Pass config to gpu_utils
        self.polygon_rest_client = polygon_rest_client # Keep optional for now
        self.alpaca_client = alpaca_client_override
        self.execution_system = execution_system # Store execution system

        self._validate_config()
        self._init_polygon_rest_client() # Init internal Polygon REST if needed
        self._init_alpaca_client_for_clock()
        self.ws_client: Optional[PolygonWebSocketClient] = None
        self._feed_processor_task: Optional[asyncio.Task] = None
        self._running = False
        self.loop = asyncio.get_running_loop()


    def _validate_config(self) -> None:
        """Validate engine configuration using the Config object."""
        try:
            # Define required keys for the engine itself
            self.config.require('POLYGON_API_KEY', secret=True, description="Polygon.io API Key for WebSocket/REST")
            self.config.require('SYMBOLS', description="List of symbols to trade/monitor")
            # Optional but needed for clock
            self.config.require('APCA_API_KEY_ID', required=False, secret=True, description="Alpaca API Key ID (optional, for market clock)")
            self.config.require('APCA_API_SECRET_KEY', required=False, secret=True, description="Alpaca API Secret Key (optional, for market clock)")

            # Perform validation
            self.config.validate()

            # Validate specific types/values
            symbols = self.config.get_list('SYMBOLS')
            if not symbols:
                raise InvalidConfigurationError("Config 'SYMBOLS' must be a non-empty list.")
            self.symbols = symbols # Store validated symbols

            self.logger.info(f"Trading engine configured for symbols: {self.symbols}")

        except (MissingConfigurationError, InvalidConfigurationError) as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ConfigurationError(f"TradingEngine configuration error: {e}") from e

    def _init_polygon_rest_client(self) -> None:
        """Initializes internal Polygon REST client if not provided."""
        if self.polygon_rest_client is None:
            api_key = self.config.get('POLYGON_API_KEY')
            if api_key:
                self.logger.info("Initializing internal PolygonRESTClient.")
                self.polygon_rest_client = PolygonRESTClient(api_key=api_key, config=self.config)
            else:
                # This case should be caught by _validate_config, but double-check
                self.logger.error("Cannot initialize internal PolygonRESTClient: POLYGON_API_KEY is missing.")
                # Decide whether to raise an error or allow running without REST features
                # raise ConfigurationError("POLYGON_API_KEY is required to initialize PolygonRESTClient.")
                self.logger.warning("Proceeding without PolygonRESTClient. Historical data fetching will be unavailable.")
        else:
            self.logger.info("Using provided PolygonRESTClient.")

    def _init_alpaca_client_for_clock(self) -> None:
        """Initialize a minimal Alpaca client if not passed, using Config."""
        if self.alpaca_client is None and ALPACA_AVAILABLE:
            api_key = self.config.get("APCA_API_KEY_ID")
            secret_key = self.config.get("APCA_API_SECRET_KEY")
            # Use paper URL for clock checks by default, less critical than trading endpoint
            base_url = self.config.get("APCA_API_BASE_URL_CLOCK", "https://paper-api.alpaca.markets")

            if api_key and secret_key:
                try:
                    # Consider adding retry logic here if needed using tenacity or similar
                    self.alpaca_client = tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url=base_url)
                    self.logger.info(f"Initialized separate Alpaca client for market clock (URL: {base_url}).")
                except Exception as e: # Catch potential API init errors
                    self.logger.error(f"Failed to initialize separate Alpaca client for clock: {e}", exc_info=True)
                    self.alpaca_client = None # Ensure it's None on failure
            else:
                self.logger.warning("Alpaca keys (APCA_API_KEY_ID, APCA_API_SECRET_KEY) not found in config. Market clock functionality disabled.")
        elif self.alpaca_client:
            self.logger.info("Using provided Alpaca client for market clock.")
        elif not ALPACA_AVAILABLE:
             self.logger.warning("alpaca-trade-api library not installed. Market clock functionality disabled.")


    async def is_market_open(self, extended_hours: bool = False) -> bool:
        """
        Checks if the market is currently open using Alpaca clock.

        Args:
            extended_hours: If True, considers extended hours (pre/post market).
                            Note: Basic implementation, relies on Alpaca's is_open flag primarily.

        Returns:
            True if the market is considered open, False otherwise.
        """
        if not self.alpaca_client:
            self.logger.warning("Alpaca client not available for market status check. Assuming market is closed.")
            return False
        try:
            # Run blocking call in executor
            clock = await self.loop.run_in_executor(
                None, self.alpaca_client.get_clock
            )
            is_open = clock.is_open

            # Basic extended hours check (Alpaca's is_open includes regular hours)
            # A more robust check would compare current time to clock.next_open/close/session times
            if extended_hours and not is_open:
                 # Check if we are in the pre-market or after-hours window based on timestamps
                 now = datetime.now(timezone.utc)
                 # Example: Check if between market close and next day's open (simplistic)
                 # This needs refinement based on actual session open/close times if precise
                 # extended hours trading is required.
                 if clock.next_close < now < clock.next_open:
                      # Potentially after-hours or weekend/holiday
                      pass # Keep is_open as False
                 elif clock.timestamp < clock.next_open:
                      # Potentially pre-market
                      # Check if it's reasonably close to market open
                      time_to_open = clock.next_open - now
                      if timedelta(hours=0) < time_to_open < timedelta(hours=2): # Example: 2 hours before open
                           self.logger.debug("Market is in pre-market hours (basic check).")
                           # Decide if pre-market counts as "open" for your logic
                           # return True # Uncomment if pre-market trading is intended
                           pass # Keep is_open as False for now

            status_log = f"Market open: {is_open}"
            if extended_hours: status_log += " (Extended hours check performed)"
            self.logger.debug(status_log)
            return is_open

        except (AlpacaAPIError, TradingAPIError) as e: # Catch Alpaca and potentially wrapped errors
            # Check if retryable (e.g., connection error, timeout)
            # from utils.exceptions import is_retryable_exception
            # if is_retryable_exception(e):
            #     self.logger.warning(f"Retryable API error checking market clock: {e}")
            # else:
            #     self.logger.error(f"Non-retryable API error checking market clock: {e}")
            self.logger.error(f"API error checking market clock: {e}")
            return False # Assume closed on error
        except Exception as e:
            self.logger.exception(f"Unexpected error checking market clock: {e}") # Use logger.exception for traceback
            return False # Assume closed on error


    async def start_realtime_feed(self) -> None:
        """Initializes, connects, subscribes, and starts the real-time data feed processor."""
        if self._running:
            self.logger.warning("Real-time feed already running.")
            return

        self._running = True
        self.logger.info("Starting real-time data feed...")
        await self.redis.ensure_initialized()

        try:
            ws_api_key = self.config.get('POLYGON_API_KEY') # Use config key
            if not ws_api_key:
                 # This should be caught by _validate_config, but raise defensively
                 raise ConfigurationError("Cannot start WebSocket feed: POLYGON_API_KEY is missing in config.")

            self.ws_client = PolygonWebSocketClient(api_key=ws_api_key)
            await self.ws_client.connect()

            if self.ws_client.is_connected():
                channels = []
                for symbol in self.symbols: # Use validated symbols list
                    channels.append(f"T.{symbol}") # Trades
                    channels.append(f"Q.{symbol}") # Quotes
                self.logger.info(f"Subscribing to {len(channels)} channels...")
                await self.ws_client.subscribe(channels)

                self._feed_processor_task = asyncio.create_task(
                    self._process_realtime_feed(),
                    name="TradingEngineFeedProcessor"
                )
                self.logger.info("Real-time feed processor task started.")
            else:
                self.logger.error("Failed to connect WebSocket client. Real-time feed not started.")
                self._running = False

        except ConfigurationError as e: # Catch config errors specifically
             self.logger.error(f"Configuration error preventing feed start: {e}")
             self._running = False
        except (APIConnectionError, APITimeoutError, TradingAPIError) as e: # Catch specific API errors
             self.logger.error(f"API error starting real-time feed: {e}", exc_info=True)
             self._running = False
             if self.ws_client: await self.ws_client.close() # Attempt cleanup
        except Exception as e:
            self.logger.exception(f"Unexpected error starting real-time feed: {e}") # Use logger.exception
            self._running = False
            if self.ws_client:
                await self.ws_client.close() # Attempt cleanup

    async def _process_realtime_feed(self) -> None:
        """Continuously processes messages from the WebSocket queue."""
        if not self.ws_client: return
        self.logger.info("Starting WebSocket message processing loop...")
        while self._running:
            try:
                message: Dict[str, Any] = await self.ws_client.receive()
                event_type = message.get('ev')
                symbol = message.get('sym')
                if not symbol: continue

                server_timestamp_ms = message.get('t') or message.get('s') or message.get('sip_timestamp')
                client_recv_time = message.get('_client_recv_time')
                fallback_ts = client_recv_time * 1000 if isinstance(client_recv_time, (int, float)) else time.time() * 1000
                timestamp_ms = server_timestamp_ms if server_timestamp_ms else fallback_ts
                timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True)

                price, volume, bid, ask = None, None, None, None

                if event_type == 'T':
                    price = message.get('p')
                    volume = message.get('s')
                    if price is not None:
                         await self.redis.update_last_tick_data(
                              symbol=symbol, price=float(price), timestamp=timestamp.to_pydatetime(),
                              volume=int(volume) if volume is not None else None, event_type='T'
                         )
                elif event_type == 'Q':
                    bid = message.get('bp')
                    ask = message.get('ap')
                    if bid is not None and ask is not None:
                         mid_price = (float(bid) + float(ask)) / 2.0
                         await self.redis.update_last_tick_data(
                              symbol=symbol, price=mid_price, timestamp=timestamp.to_pydatetime(),
                              bid=float(bid), ask=float(ask), event_type='Q'
                         )
            except asyncio.CancelledError:
                self.logger.info("Feed processing task cancelled.")
                break
            except (TradingAPIError, WebSocketError) as e: # Catch specific websocket/api errors if defined (using alias)
                 self.logger.error(f"WebSocket processing error: {e}", exc_info=True)
                 # Consider more robust error handling: reconnect? circuit breaker?
                 await asyncio.sleep(1) # Basic retry delay
            except Exception as e:
                self.logger.exception(f"Unexpected error processing WebSocket message: {e}")
                await asyncio.sleep(1) # Basic retry delay
        self.logger.info("WebSocket message processing loop stopped.")


    async def stop_realtime_feed(self) -> None:
        """Stops the feed processor task and closes the WebSocket connection."""
        if not self._running and not self._feed_processor_task and not self.ws_client:
            self.logger.info("Real-time feed is not running or already stopped.")
            return
        self.logger.info("Stopping real-time data feed...")
        self._running = False
        if self._feed_processor_task and not self._feed_processor_task.done():
            self._feed_processor_task.cancel()
            try: await asyncio.wait_for(self._feed_processor_task, timeout=5.0)
            # Use self.logger in except blocks
            except asyncio.CancelledError: self.logger.info("Feed processor task already cancelled.")
            except asyncio.TimeoutError: self.logger.warning("Feed processor task did not cancel within timeout.")
            except Exception as e: self.logger.exception(f"Error waiting for feed processor task cancellation: {e}") # Use logger.exception
        self._feed_processor_task = None
        if self.ws_client:
            await self.ws_client.close()
            self.ws_client = None
            self.logger.info("WebSocket client closed.")
        self.logger.info("Real-time data feed stopped.")

    async def get_historical_aggregates(
        self, symbol: str, timespan: str = "minute", multiplier: int = 1,
        minutes: Optional[int] = None, days: Optional[int] = None, limit: int = 5000
    ) -> Optional[pd.DataFrame]:
        """Fetches historical aggregate bars using the Polygon REST client."""
        if not self.polygon_rest_client:
            self.logger.error(f"Cannot fetch aggregates for {symbol}: PolygonRESTClient not available.")
            return None
        to_date = datetime.now(timezone.utc)
        if minutes: from_date = to_date - timedelta(minutes=minutes)
        elif days: from_date = to_date - timedelta(days=days)
        # Use self.logger in else block
        else: from_date = to_date - timedelta(days=1); self.logger.warning(f"Defaulting to last day for {symbol}.")
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        self.logger.debug(f"Fetching {multiplier} {timespan} aggregates for {symbol} from {from_date_str} to {to_date_str}")
        try:
            df = await self.polygon_rest_client.get_aggregates(
                symbol=symbol, multiplier=multiplier, timespan=timespan, from_date=from_date_str,
                to_date=to_date_str, limit=limit, adjusted=True, sort="asc"
            )
            # Use self.logger in error/warning checks
            if df is None: self.logger.error(f"Failed fetch aggregates for {symbol}."); return None
            if df.empty: self.logger.warning(f"No aggregate data for {symbol} in range."); return df
            if minutes or days:
                 from_date_aware = from_date.replace(tzinfo=timezone.utc) if from_date.tzinfo is None else from_date
                 df = df[df['timestamp'] >= from_date_aware]
            self.logger.info(f"Fetched {len(df)} aggregate bars for {symbol}.")
            return df
        except (APIConnectionError, APITimeoutError, TradingAPIError) as e:
             self.logger.error(f"API error fetching aggregates for {symbol}: {e}", exc_info=True)
             return None
        except Exception as e:
            self.logger.exception(f"Unexpected error fetching historical aggregates for {symbol}: {e}")
            return None

    async def get_latest_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches latest tick data and computes basic real-time features.
        Expand this based on ML model needs. Accuracy depends heavily on feature quality.
        Consider: Volatility, Momentum, Order Book Imbalance (if L2 data added), Options Flow (if UW client used).
        """
        self.logger.debug(f"Fetching latest features for {symbol}...")
        tick_data = await self.redis.get_latest_tick_data(symbol)
        if not tick_data: return None
        features = {}
        try:
            price_str, bid_str, ask_str = tick_data.get('price'), tick_data.get('bid'), tick_data.get('ask')
            features['price'] = float(price_str) if price_str is not None else np.nan
            features['bid'] = float(bid_str) if bid_str is not None else np.nan
            features['ask'] = float(ask_str) if ask_str is not None else np.nan
            features['volume'] = int(tick_data.get('volume', '0'))
            features['timestamp'] = datetime.fromisoformat(tick_data['timestamp']) if tick_data.get('timestamp') else None
            features['event_type'] = tick_data.get('event_type')
            if not np.isnan(features['bid']) and not np.isnan(features['ask']):
                 features['spread'] = features['ask'] - features['bid']
            else: features['spread'] = np.nan
            # Add more feature calculations here...
        except (ValueError, TypeError, KeyError) as e: # Add KeyError for missing dict keys
            # Consider using DataFormatError from utils.exceptions
            self.logger.error(f"Error processing tick data for features {symbol}: {e}", exc_info=True)
            return None
        except Exception as e:
             self.logger.exception(f"Unexpected error generating features for {symbol}: {e}")
             return None
        self.logger.debug(f"Generated features for {symbol}: {features}")
        return features


    async def start(self):
        """Starts the trading engine components, including the real-time feed."""
        self.logger.info("Starting Trading Engine...")
        await self.start_realtime_feed()
        self.logger.info("Trading Engine started.")

    async def stop(self):
         """Stops the trading engine components."""
         self.logger.info("Stopping Trading Engine...")
         await self.shutdown()
         self.logger.info("Trading Engine stopped.")

    async def run_strategies(self):
         """Runs strategies only if the market is open."""
         # Consider adding specific schedule checks (e.g., only run between 9:30 AM and 4:00 PM ET)
         # market_schedule = await self.get_market_schedule() # Needs implementation
         # if not is_within_schedule(market_schedule): return

         market_open = await self.is_market_open(extended_hours=False) # Check regular hours for strategies
         if not market_open:
              self.logger.debug("Market is closed. Skipping strategy execution.")
              return

         self.logger.info("Market is open. Running trading strategies (placeholder)...")
         # TODO: Implement actual strategy execution logic here.
         # This might involve:
         # 1. Getting signals from the ML inference loop (e.g., via Redis Pub/Sub or direct call)
         # 2. Iterating through symbols based on signals or a watchlist
         # 3. Calling specific strategy logic (e.g., from trading_engine.day_trading)
         # 4. Passing generated orders to the execution system
         pass

    async def shutdown(self):
         """Gracefully shuts down the trading engine."""
         self.logger.info("Shutting down trading engine...")
         await self.stop_realtime_feed()
         if self.polygon_rest_client and hasattr(self.polygon_rest_client, 'close'):
              await self.polygon_rest_client.close()
         # Alpaca REST client doesn't need explicit close
         await self.redis.close()
         self.logger.info("Trading engine shut down complete.")
