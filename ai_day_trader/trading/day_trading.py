"""Intraday trading strategies implementation."""
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
import pandas as pd

# Import necessary types for hints
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient # Use new client path
from ai_day_trader.utils.config import Config # Use new utils path
from .execution.base import ExecutionSystem # Use relative import

if TYPE_CHECKING:
    try:
        from redis.asyncio import Redis as RedisClient
    except ImportError:
        RedisClient = Any

class DayTradingStrategies:
    """Implements common day trading strategies using latest tick data and real historical aggregates."""

    def __init__(
        self,
        config: Config,
        polygon_rest_client: PolygonRESTClient,
        redis_client: Optional['RedisClient'],
        # execution: ExecutionSystem, # Removed execution as it wasn't used
        logger: logging.Logger
    ):
        """Initialize strategies with direct dependencies."""
        self.config = config
        self.polygon_rest = polygon_rest_client
        self.redis = redis_client
        # self.execution = execution # Removed
        self.logger = logger

    async def _get_historical_aggregates(self, *args, **kwargs):
         """Proxy to the polygon rest client."""
         if not self.polygon_rest:
              self.logger.error("Polygon REST client not available for historical data.")
              return None
         try:
             # Ensure limit is passed if provided, otherwise use a default
             kwargs.setdefault('limit', 5000)
             return await self.polygon_rest.get_aggregates(*args, **kwargs)
         except Exception as e:
              self.logger.error(f"Error fetching aggregates via proxy: {e}", exc_info=True)
              return None

    async def _get_latest_tick_data(self, symbol: str) -> Optional[Dict[str, Any]]:
         """Gets latest tick data from Redis."""
         if not self.redis:
              self.logger.error("Redis client not available for latest tick data.")
              return None
         key = f"tick:{symbol}"
         try:
             tick_data_bytes = await self.redis.hgetall(key)
             # Decode bytes to string
             return {k.decode('utf-8'): v.decode('utf-8') for k, v in tick_data_bytes.items()} if tick_data_bytes else None
         except Exception as e:
             self.logger.error(f"Error getting latest tick data for {symbol} from Redis: {e}")
             return None

    async def run_opening_range_breakout(self, symbol: str) -> Optional[Dict]:
        """Generates an Opening Range Breakout signal if conditions met."""
        # Define ORB period (e.g., first 30 minutes) - make configurable?
        orb_minutes = 30
        orb_min_bars = 5

        # --- Opening Range Calculation ---
        opening_df = await self._get_historical_aggregates(
            symbol=symbol,
            timespan="minute",
            multiplier=1,
            minutes=orb_minutes * 2 # Fetch more data to ensure coverage
        )

        if opening_df is None or opening_df.empty or len(opening_df) < orb_min_bars:
            self.logger.debug(f"ORB: Insufficient opening aggregate data for {symbol} ({len(opening_df) if opening_df is not None else 0} bars).")
            return None
        if 'h' not in opening_df.columns or 'l' not in opening_df.columns:
             self.logger.error(f"ORB: Missing 'h' or 'l' columns in aggregate data for {symbol}.")
             return None

        # TODO: More robustly define the actual opening range time window based on market open
        # For now, assume the latest data covers the relevant period
        opening_range_df = opening_df.tail(orb_minutes) # Approximate range
        if len(opening_range_df) < orb_min_bars:
             self.logger.debug(f"ORB: Not enough bars in approximated opening range for {symbol}.")
             return None

        high = opening_range_df['h'].max()
        low = opening_range_df['l'].min()
        range_size = high - low
        if range_size <= 1e-6:
             self.logger.debug(f"ORB: Negligible range size for {symbol}.")
             return None

        # --- Fetch Latest Tick Data ---
        latest_tick = await self._get_latest_tick_data(symbol)
        if not latest_tick or 'price' not in latest_tick:
            self.logger.debug(f"ORB: No latest tick data available for {symbol}.")
            return None

        try:
            current_price = float(latest_tick['price'])
        except (ValueError, KeyError, TypeError) as e:
            self.logger.error(f"ORB: Error parsing latest tick price for {symbol}: {e} - Data: {latest_tick}")
            return None

        # --- Check Breakout Conditions ---
        breakout_threshold_pct = self.config.get_float("ORB_BREAKOUT_THRESHOLD_PCT", 0.1) # Example: 10% of range
        breakout_threshold = breakout_threshold_pct * range_size
        signal = None
        if current_price > high + breakout_threshold:
            self.logger.info(f"ORB LONG signal generated for {symbol}: Price {current_price:.2f} > High {high:.2f} + Threshold {breakout_threshold:.2f}")
            signal = {'symbol': symbol, 'side': 'buy', 'source': 'ORB', 'trigger_price': current_price, 'confidence': 0.7}
        elif current_price < low - breakout_threshold:
            self.logger.info(f"ORB SHORT signal generated for {symbol}: Price {current_price:.2f} < Low {low:.2f} - Threshold {breakout_threshold:.2f}")
            signal = {'symbol': symbol, 'side': 'sell', 'source': 'ORB', 'trigger_price': current_price, 'confidence': 0.7}

        return signal

    async def run_vwap_reversion(self, symbol: str) -> Optional[Dict]:
        """Generates a VWAP reversion signal if conditions met."""
        # --- VWAP Calculation ---
        vwap_minutes = self.config.get_int("VWAP_LOOKBACK_MINUTES", 390) # Default to full day
        min_vwap_bars = 30

        historical_df = await self._get_historical_aggregates(
            symbol=symbol,
            timespan="minute",
            multiplier=1,
            minutes=vwap_minutes
        )

        if historical_df is None or historical_df.empty or len(historical_df) < min_vwap_bars:
            self.logger.debug(f"VWAP: Insufficient historical data for {symbol} ({len(historical_df) if historical_df is not None else 0} bars).")
            return None
        if 'c' not in historical_df.columns or 'v' not in historical_df.columns:
             self.logger.error(f"VWAP: Missing 'c' or 'v' columns in aggregate data for {symbol}.")
             return None

        prices = historical_df['c'].values
        volumes = historical_df['v'].values
        valid_indices = volumes > 0
        if not np.any(valid_indices):
             self.logger.debug(f"VWAP: No valid volume data for {symbol}.")
             return None
        vwap = np.sum(prices[valid_indices] * volumes[valid_indices]) / np.sum(volumes[valid_indices])
        if vwap <= 1e-6:
             self.logger.debug(f"VWAP: Calculated VWAP is zero or near-zero for {symbol}.")
             return None

        # --- Fetch Latest Tick Data ---
        latest_tick = await self._get_latest_tick_data(symbol)
        if not latest_tick or 'price' not in latest_tick:
            self.logger.debug(f"VWAP: No latest tick data available for {symbol}.")
            return None

        try:
            current_price = float(latest_tick['price'])
        except (ValueError, KeyError, TypeError) as e:
            self.logger.error(f"VWAP: Error parsing latest tick price for {symbol}: {e} - Data: {latest_tick}")
            return None

        # --- Check Reversion Conditions ---
        deviation = (current_price - vwap) / vwap
        reversion_threshold = self.config.get_float("VWAP_REVERSION_THRESHOLD", 0.01) # Default 1%
        signal = None
        if deviation > reversion_threshold:  # Price is significantly above VWAP -> Short signal
            self.logger.info(f"VWAP SELL signal generated for {symbol}: Price {current_price:.2f} > VWAP {vwap:.2f} by {deviation*100:.2f}%")
            signal = {'symbol': symbol, 'side': 'sell', 'source': 'VWAP_Reversion', 'trigger_price': current_price, 'confidence': 0.65}
        elif deviation < -reversion_threshold:  # Price is significantly below VWAP -> Long signal
            self.logger.info(f"VWAP BUY signal generated for {symbol}: Price {current_price:.2f} < VWAP {vwap:.2f} by {deviation*100:.2f}%")
            signal = {'symbol': symbol, 'side': 'buy', 'source': 'VWAP_Reversion', 'trigger_price': current_price, 'confidence': 0.65}

        return signal

    # Add other strategies here, adapting them to return signals
