"""
Dynamic Stock Selector for AI Day Trader.

Selects candidate stocks based on criteria like price, volume, volatility,
and available trading capital.
"""
import logging
import asyncio
import random
import json # Added missing import
from typing import List, Dict, Optional, Any

import pandas as pd

from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient
from ai_day_trader.clients.redis_client import get_async_redis_client, set_redis_key, get_redis_key # For caching results

logger = logging.getLogger(__name__)

# Constants moved to config where possible

# Default values if not found in config
DEFAULT_MIN_SHARE_PRICE = 5.0
DEFAULT_MAX_SHARE_PRICE_BUFFER = 1.2
DEFAULT_MIN_AVG_DOLLAR_VOLUME = 5_000_000
DEFAULT_POLYGON_TICKERS_LIMIT = 1000
DEFAULT_CANDIDATE_CACHE_KEY = "stock_selector:candidates"
DEFAULT_CANDIDATE_CACHE_TTL_SECONDS = 60 * 60 * 4 # 4 hours

class StockSelector:
    """
    Selects candidate stocks for trading based on configurable criteria
    using efficient batch API calls (Polygon Snapshot).
    """

    def __init__(self, config: Config, polygon_client: PolygonRESTClient):
        self.config = config
        self.polygon_client = polygon_client
        self.logger = logger # Use module-level logger
        self.redis_client = None # Initialize as None, will be set in _get_redis

        # Load configuration for filtering and caching
        self.min_share_price = config.get_float("STOCK_SELECTOR_MIN_PRICE", DEFAULT_MIN_SHARE_PRICE)
        self.max_share_price_buffer = config.get_float("STOCK_SELECTOR_MAX_PRICE_BUFFER", DEFAULT_MAX_SHARE_PRICE_BUFFER)
        self.min_avg_dollar_volume = config.get_float("STOCK_SELECTOR_MIN_DOLLAR_VOLUME", DEFAULT_MIN_AVG_DOLLAR_VOLUME)
        self.polygon_tickers_limit = config.get_int("POLYGON_TICKERS_LIMIT", DEFAULT_POLYGON_TICKERS_LIMIT) # For fetching initial list
        self.candidate_cache_key = config.get_str("STOCK_SELECTOR_CACHE_KEY", DEFAULT_CANDIDATE_CACHE_KEY)
        self.candidate_cache_ttl = config.get_int("STOCK_SELECTOR_CACHE_TTL_SECONDS", DEFAULT_CANDIDATE_CACHE_TTL_SECONDS)

        # Needed for max affordable price calculation (consider moving this calc elsewhere later)
        self.max_trade_risk_pct = config.get_float("MAX_TRADE_RISK_PCT", 0.005)
        self.portfolio_size = config.get_float("PORTFOLIO_SIZE", 100000.0)
        self.stop_loss_pct = config.get_float("STOP_LOSS_PCT", 0.02)

        self.logger.info(f"StockSelector initialized with criteria: MinPrice=${self.min_share_price}, MinDollarVol=${self.min_avg_dollar_volume:,.0f}, CacheTTL={self.candidate_cache_ttl}s")

    async def _get_redis(self):
        """Lazy initialize Redis client."""
        if self.redis_client is None:
            self.redis_client = await get_async_redis_client()
        return self.redis_client

    async def _fetch_all_active_tickers(self) -> List[str]:
        """Fetches all active common stock tickers from Polygon."""
        tickers = []
        url = "/v3/reference/tickers"
        params = {
            "active": "true",
            "market": "stocks",
            "type": "CS", # Common Stock
            "limit": self.polygon_tickers_limit # Use configured limit
        }
        page_count = 0
        # Removed max_pages limit - fetch all available pages

        self.logger.info("Fetching active common stock tickers from Polygon...")
        while url: # Loop until no next_url
            page_count += 1
            self.logger.debug(f"Fetching ticker page {page_count}: {url}")
            try:
                # Use the client's generic get method which handles auth
                response_obj = await self.polygon_client.get(url.replace(self.polygon_client.base_url, ""), params=params)
                if response_obj.status_code != 200:
                     self.logger.error(f"Failed to fetch tickers page {page_count}: Status {response_obj.status_code} - {response_obj.text}")
                     break # Stop fetching on error

                data = response_obj.json()

                if data and "results" in data:
                    tickers.extend([t['ticker'] for t in data['results'] if 'ticker' in t])
                    url = data.get("next_url")
                    params = {} # next_url includes params and API key
                    if url:
                         # Add a small delay between pages
                         await asyncio.sleep(0.2)
                    else:
                         break # No more pages
                else:
                    self.logger.warning(f"No 'results' in tickers response page {page_count}.")
                    break
            except Exception as e:
                self.logger.error(f"Error fetching tickers page {page_count}: {e}", exc_info=True)
                break

        self.logger.info(f"Fetched {len(tickers)} active common stock tickers.")
        return tickers

    async def _filter_tickers_by_criteria(self, tickers_to_filter: List[str]) -> List[str]:
        """
        Filters tickers based on price and volume using Polygon's Snapshot API.

        Args:
            tickers_to_filter: A list of ticker symbols initially fetched (e.g., active common stocks).

        Returns:
            A list of ticker symbols that meet the configured criteria.
        """
        if not tickers_to_filter:
            self.logger.warning("No tickers provided to filter.")
            return []

        # Calculate max affordable price based on risk parameters (rough estimate)
        max_risk_per_trade_amt = self.portfolio_size * self.max_trade_risk_pct
        if self.stop_loss_pct > 0:
             max_affordable_price = (max_risk_per_trade_amt / self.stop_loss_pct) * self.max_share_price_buffer
        else:
             # If no stop loss %, allow up to portfolio size (less realistic but avoids division by zero)
             max_affordable_price = self.portfolio_size * self.max_share_price_buffer
             self.logger.warning("STOP_LOSS_PCT is zero or not configured, max affordable price calculation might be less meaningful.")

        self.logger.info(
            f"Filtering {len(tickers_to_filter)} tickers using Snapshot API. "
            f"Criteria: Price [${self.min_share_price:.2f} - ${max_affordable_price:.2f}], "
            f"Min Dollar Vol: ${self.min_avg_dollar_volume:,.0f}"
        )

        # Fetch snapshot data for all tickers in one go
        all_snapshots = await self.polygon_client.get_all_tickers_snapshot()

        if not all_snapshots:
            self.logger.error("Failed to fetch ticker snapshots from Polygon. Cannot filter.")
            return []

        self.logger.info(f"Received {len(all_snapshots)} snapshots from Polygon API.")

        # Use a set for efficient lookup of the initially fetched tickers
        tickers_set = set(tickers_to_filter)
        final_candidates = []
        processed_count = 0

        for snapshot in all_snapshots:
            ticker = snapshot.get('ticker')
            if not ticker:
                self.logger.debug("Snapshot missing ticker symbol, skipping.")
                continue

            processed_count += 1
            if processed_count % 5000 == 0: # Log progress less frequently for batch processing
                 self.logger.info(f"Filtering progress: {processed_count}/{len(all_snapshots)} snapshots processed...")

            # Only consider tickers that were in our initial fetch (e.g., active common stocks)
            if ticker not in tickers_set:
                # self.logger.debug(f"Ticker {ticker} from snapshot not in initial fetch list, skipping.")
                continue

            prev_day_data = snapshot.get('prevDay', {})
            price = prev_day_data.get('c') # Previous day's close price
            volume = prev_day_data.get('v') # Previous day's volume

            if price is not None and volume is not None and price > 0 and volume > 0:
                try:
                    price = float(price)
                    volume = float(volume)
                    dollar_volume = price * volume

                    # Apply filters using configured values
                    if (self.min_share_price <= price <= max_affordable_price and
                        dollar_volume >= self.min_avg_dollar_volume):
                        final_candidates.append(ticker)
                        self.logger.debug(f"Ticker {ticker} passed filters (Price: ${price:.2f}, Vol: {volume:,.0f}, Dollar Vol: ${dollar_volume:,.0f})")
                    # else: # Optional: Log failures if needed, can be verbose
                    #     self.logger.debug(f"Ticker {ticker} failed filters (Price: ${price:.2f}, Vol: {volume:,.0f}, Dollar Vol: ${dollar_volume:,.0f})")

                except (ValueError, TypeError) as e:
                     self.logger.warning(f"Could not parse price/volume for {ticker}: price={price}, volume={volume}. Error: {e}")
            # else: # Optional: Log missing data if needed
            #     self.logger.debug(f"Ticker {ticker}: Missing valid price ({price}) or volume ({volume}) in prevDay snapshot data.")


        self.logger.info(f"Selected {len(final_candidates)} candidate tickers after filtering {len(all_snapshots)} snapshots.")
        return final_candidates


    async def select_candidate_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        Selects and returns a list of candidate symbols for trading.
        Uses cached results if available and not forced to refresh.

        Args:
            force_refresh: If True, ignores cache and fetches fresh data.

        Returns:
            A list of candidate ticker symbols.
        """
        redis = await self._get_redis()
        if not redis:
             self.logger.error("Redis unavailable, cannot perform stock selection.")
             return [] # Cannot proceed without Redis for caching

        # Check cache first using configured key
        if not force_refresh:
            cached_data = await get_redis_key(self.candidate_cache_key)
            if cached_data:
                try:
                    candidates = json.loads(cached_data.decode('utf-8'))
                    if isinstance(candidates, list):
                        self.logger.info(f"Using cached candidate symbols ({len(candidates)} tickers).")
                        return candidates
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.warning(f"Failed to load candidates from cache, fetching fresh: {e}")

        # Fetch all active tickers
        all_tickers = await self._fetch_all_active_tickers()
        if not all_tickers:
            return []

        # Filter tickers based on criteria
        candidate_symbols = await self._filter_tickers_by_criteria(all_tickers)

        # Cache the results using configured key and TTL
        if candidate_symbols:
            try:
                await set_redis_key(
                    self.candidate_cache_key,
                    json.dumps(candidate_symbols),
                    expire=self.candidate_cache_ttl
                )
                self.logger.info(f"Cached {len(candidate_symbols)} candidate symbols for {self.candidate_cache_ttl} seconds.")
            except Exception as e:
                self.logger.error(f"Failed to cache candidate symbols: {e}")

        return candidate_symbols
