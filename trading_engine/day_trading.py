"""Intraday trading strategies implementation."""
from typing import Dict, List, Optional, Any
import logging
import numpy as np
import pandas as pd # Import pandas for DataFrame handling
import torch
from datetime import datetime # Import datetime

# Assuming TradingEngine provides access to RedisClient via self.engine.redis
# and historical data via self.engine.get_historical_aggregates
from trading_engine.base import TradingEngine # Import for type hint
from .execution import ExecutionSystem

class DayTradingStrategies:
    """Implements common day trading strategies using latest tick data and real historical aggregates."""

    def __init__(self, engine: TradingEngine, execution: ExecutionSystem): # Use TradingEngine type hint
        self.engine = engine # engine provides access to self.engine.redis and self.engine.get_historical_aggregates
        self.execution = execution
        self.logger = logging.getLogger(__name__)
        # Ensure GPU device configuration is handled appropriately if needed here
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # If strategies don't use torch directly, remove device logic from here

    async def run_opening_range_breakout(self, symbol: str) -> Optional[Dict]:
        """Execute opening range breakout strategy using latest tick price."""
        # --- Opening Range Calculation (Uses real historical data fetch) ---
        # Fetch first 30 minutes of 1-minute bars for the current day
        # Note: Adjust timespan/multiplier/minutes as needed for ORB definition
        opening_df = await self.engine.get_historical_aggregates(
            symbol=symbol,
            timespan="minute",
            multiplier=1,
            minutes=30 # Fetch data for the last 30 minutes (adjust if market just opened)
            # Consider adding specific start/end times based on market open for more robustness
        )

        if opening_df is None or opening_df.empty or len(opening_df) < 5: # Need a few bars at least
            self.logger.debug(f"ORB: Insufficient opening aggregate data for {symbol}.")
            return None

        # Calculate opening range high/low from the DataFrame
        # Ensure columns 'h' (high) and 'l' (low) exist from get_aggregates
        if 'h' not in opening_df.columns or 'l' not in opening_df.columns:
             self.logger.error(f"ORB: Missing 'h' or 'l' columns in aggregate data for {symbol}.")
             return None

        high = opening_df['h'].max()
        low = opening_df['l'].min()
        range_size = high - low
        if range_size == 0: # Avoid division by zero
             self.logger.debug(f"ORB: Zero range size for {symbol}.")
             return None

        # --- Fetch Latest Tick Data for Trigger ---
        latest_tick = await self.engine.redis.get_latest_tick_data(symbol)
        if not latest_tick or 'price' not in latest_tick:
            self.logger.debug(f"ORB: No latest tick data available for {symbol}.")
            return None

        try:
            # Get current price from the latest tick data hash
            current_price = float(latest_tick['price'])
            # Optional: Staleness check
        except (ValueError, KeyError) as e:
            self.logger.error(f"ORB: Error parsing latest tick data for {symbol}: {e} - Data: {latest_tick}")
            return None

        # --- Check Breakout Conditions using Latest Price ---
        breakout_threshold = 0.2 * range_size
        if current_price > high + breakout_threshold:
            self.logger.info(f"ORB LONG signal for {symbol}: Price {current_price} > High {high:.2f} + Threshold {breakout_threshold:.2f}")
            # Ensure execute_order is async
            return await self.execution.execute_order({
                'symbol': symbol,
                'side': 'buy',
                'price': current_price, # Optional: pass trigger price
                'quantity': 100, # Example quantity - needs proper sizing logic
                'strategy': 'opening_range_breakout'
            })
        elif current_price < low - breakout_threshold:
            self.logger.info(f"ORB SHORT signal for {symbol}: Price {current_price} < Low {low:.2f} - Threshold {breakout_threshold:.2f}")
            return await self.execution.execute_order({
                'symbol': symbol,
                'side': 'sell',
                'price': current_price, # Optional: pass trigger price
                'quantity': 100, # Example quantity - needs proper sizing logic
                'strategy': 'opening_range_breakout'
            })

        # self.logger.debug(f"ORB: No signal for {symbol}. Price: {current_price}, Range: [{low:.2f}, {high:.2f}]")
        return None

    async def run_vwap_reversion(self, symbol: str) -> Optional[Dict]:
        """Execute VWAP reversion strategy using latest tick price."""
        # --- VWAP Calculation (Uses real historical data fetch) ---
        # Fetch minute bars for the current day (e.g., last 6.5 hours = 390 mins)
        historical_df = await self.engine.get_historical_aggregates(
            symbol=symbol,
            timespan="minute",
            multiplier=1,
            minutes=390 # Fetch data for typical trading day length
            # Consider fetching since market open for more accuracy
        )

        if historical_df is None or historical_df.empty or len(historical_df) < 30:
            self.logger.debug(f"VWAP: Insufficient historical aggregate data for {symbol}.")
            return None

        # Ensure required columns exist
        if 'c' not in historical_df.columns or 'v' not in historical_df.columns:
             self.logger.error(f"VWAP: Missing 'c' or 'v' columns in aggregate data for {symbol}.")
             return None

        # Calculate VWAP using pandas/numpy
        prices = historical_df['c'].values # Close prices
        volumes = historical_df['v'].values # Volumes
        valid_indices = volumes > 0
        if not np.any(valid_indices):
             self.logger.debug(f"VWAP: No valid volume data for {symbol}.")
             return None
        vwap = np.sum(prices[valid_indices] * volumes[valid_indices]) / np.sum(volumes[valid_indices])

        # --- Fetch Latest Tick Data for Trigger ---
        latest_tick = await self.engine.redis.get_latest_tick_data(symbol)
        if not latest_tick or 'price' not in latest_tick:
            self.logger.debug(f"VWAP: No latest tick data available for {symbol}.")
            return None

        try:
            current_price = float(latest_tick['price'])
            # Optional: Staleness check
        except (ValueError, KeyError) as e:
            self.logger.error(f"VWAP: Error parsing latest tick data for {symbol}: {e} - Data: {latest_tick}")
            return None

        # --- Check Reversion Conditions using Latest Price ---
        deviation = (current_price - vwap) / vwap

        reversion_threshold = 0.01 # 1% deviation threshold
        if deviation > reversion_threshold:  # Price is significantly above VWAP
            self.logger.info(f"VWAP SELL signal for {symbol}: Price {current_price} > VWAP {vwap:.2f} by {deviation*100:.2f}%")
            return await self.execution.execute_order({
                'symbol': symbol,
                'side': 'sell',
                'price': current_price,
                'quantity': 100, # Example quantity
                'strategy': 'vwap_reversion'
            })
        elif deviation < -reversion_threshold:  # Price is significantly below VWAP
            self.logger.info(f"VWAP BUY signal for {symbol}: Price {current_price} < VWAP {vwap:.2f} by {deviation*100:.2f}%")
            return await self.execution.execute_order({
                'symbol': symbol,
                'side': 'buy',
                'price': current_price,
                'quantity': 100, # Example quantity
                'strategy': 'vwap_reversion'
            })

        # self.logger.debug(f"VWAP: No signal for {symbol}. Price: {current_price}, VWAP: {vwap:.2f}, Deviation: {deviation*100:.2f}%")
        return None

    # Add other strategies here, adapting them to use get_latest_tick_data and get_historical_aggregates
