"""
Polygon.io REST API Client for AI Day Trader
"""

import asyncio
import json
import pandas as pd
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Import from new structure
from ai_day_trader.utils.config import Config # Import base Config for type hint
from ai_day_trader.config import load_ai_trader_config # Import loader
from ai_day_trader.clients.redis_cache import AsyncRedisCache
from ai_day_trader.clients.async_connection_pool import AsyncConnectionPool
from ai_day_trader.utils.logging_config import get_logger # Use new utils path
from ai_day_trader.clients.base import API_CIRCUIT_BREAKER, rate_limited


class PolygonRESTClient:
    """Enhanced Polygon.io REST API client with production features."""

    def __init__(
        self: Any,
        api_key: Optional[str] = None,
        cache: Optional[AsyncRedisCache] = None,
        connection_pool: Optional[AsyncConnectionPool] = None,
        config: Optional[Config] = None, # Use base Config for type hint
        batch_size: int = 10,
        batch_delay: float = 0.1,
    ) -> None:
        """Initialize the Polygon REST client with production features."""
        self.config = config or load_ai_trader_config() # Load config if not passed
        self.api_key = api_key or self.config.polygon_api_key
        self.logger = get_logger("ai_day_trader.clients.polygon_rest")

        if not self.api_key:
            self.logger.error("Polygon API Key is required. Set TRADING_POLYGON_API_KEY.")
            raise ValueError("Polygon API Key not provided.")

        self.base_url = self.config.polygon_api_base_url
        self.cache = cache
        if self.cache is None:
             self.logger.warning("No external cache provided, initializing internal Redis cache for Polygon REST.")
             self.cache = AsyncRedisCache(prefix="polygon_rest", config=self.config, ttl=self.config.polygon_cache_ttl)


        self.connection_pool = connection_pool or AsyncConnectionPool(config=self.config)
        self.running = True
        self.scheduled_tasks: Dict[str, asyncio.Task] = {}

        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.batch_queue = asyncio.Queue()
        self.batch_processor_task = asyncio.create_task(self._process_batches())
        self.logger.info(f"Polygon REST Client initialized for base URL: {self.base_url}")


    async def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate client configuration and dependencies."""
        errors = []
        if not self.api_key or len(self.api_key) < 8:
            errors.append("Invalid Polygon API key.")
        if not self.base_url:
            errors.append("Polygon API base URL missing.")
        return len(errors) == 0, errors

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        minutes: Optional[int] = None,
        limit: int = 5000,
        adjusted: bool = True,
        sort: str = "asc",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Get aggregate bars with production enhancements."""
        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Aggregates request for {symbol} ({timespan}x{multiplier})")

        if not symbol:
            raise ValueError("Symbol cannot be empty.")
        if not timespan or timespan not in ["minute", "hour", "day", "week", "month", "quarter", "year"]:
             raise ValueError(f"Invalid timespan: {timespan}")
        if not isinstance(multiplier, int) or multiplier <= 0:
             raise ValueError("Multiplier must be a positive integer.")

        if from_date and to_date:
             endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
             cache_key_dates = f"{from_date}_{to_date}"
        elif minutes is not None and isinstance(minutes, int) and minutes > 0:
             self.logger.warning(f"Fetching aggregates by 'minutes' lookback ({minutes}) is approximated by using limit={limit}.")
             endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/2000-01-01/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
             cache_key_dates = f"last_{minutes}m_approx"
        else:
             raise ValueError("Either from_date/to_date or minutes must be provided.")

        cache_key = [
            "aggregates", symbol, multiplier, timespan, cache_key_dates, str(adjusted), sort, limit
        ]

        if use_cache and self.cache:
            cached_df = await self.cache.get_dataframe(cache_key)
            if cached_df is not None:
                self.logger.debug(f"[TraceID: {trace_id}] Cache hit for {symbol} aggregates.")
                if minutes is not None and not cached_df.empty:
                    return cached_df.tail(minutes)
                return cached_df

        params = {
            "apiKey": self.api_key, "limit": limit,
            "adjusted": str(adjusted).lower(), "sort": sort,
        }

        try:
            data = await self._make_request(endpoint, params, trace_id=trace_id)

            if data and "results" in data and data["results"]:
                results_df = pd.DataFrame(data["results"])
                if "t" in results_df.columns:
                    results_df["timestamp"] = pd.to_datetime(results_df["t"], unit="ms", utc=True)
                    results_df = results_df.set_index("timestamp").drop(columns=["t"])
                else:
                     self.logger.warning(f"Timestamp column 't' missing in Polygon response for {symbol}")

                if use_cache and self.cache:
                    await self.cache.store_dataframe(cache_key, results_df)

                if minutes is not None and not results_df.empty:
                    return results_df.tail(minutes)
                return results_df
            elif data and "results" in data and not data["results"]:
                self.logger.info(f"[TraceID: {trace_id}] No aggregate results found for {symbol}")
                if use_cache and self.cache:
                    await self.cache.store_dataframe(cache_key, pd.DataFrame(), ttl=300)
                return pd.DataFrame()
            else:
                self.logger.error(f"[TraceID: {trace_id}] Invalid response structure for aggregates {symbol}: {data}")
                return None

        except Exception as e:
            self.logger.exception(f"[TraceID: {trace_id}] Error getting aggregates for {symbol}: {e}")
            return None

    async def batch_get_aggregates(
        self, requests: List[Dict[str, Any]]
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Batch process multiple aggregate requests."""
        results = {}
        futures = []
        for req in requests:
            future = asyncio.Future()
            await self.batch_queue.put((req, future))
            futures.append(future)
        await asyncio.gather(*futures, return_exceptions=True)
        for req, future in zip(requests, futures):
            try:
                results[req["symbol"]] = future.result()
            except Exception as e:
                 self.logger.error(f"Error in batch result for {req.get('symbol', 'unknown')}: {e}")
                 results[req["symbol"]] = None
        return results

    async def _process_batches(self) -> None:
        """Background task to process batched requests."""
        while self.running:
            batch = []
            try:
                first_item = await asyncio.wait_for(self.batch_queue.get(), timeout=self.batch_delay)
                batch.append(first_item)
                self.batch_queue.task_done()
                while len(batch) < self.batch_size:
                    try:
                        item = self.batch_queue.get_nowait()
                        batch.append(item)
                        self.batch_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                if batch:
                    self.logger.debug(f"Processing batch of {len(batch)} aggregate requests.")
                    await self._process_single_batch(batch)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                 self.logger.info("Batch processor task cancelled.")
                 break
            except Exception as e:
                 self.logger.error(f"Unexpected error in batch processor loop: {e}", exc_info=True)
                 await asyncio.sleep(1)

    async def _process_single_batch(self, batch: List[Tuple[Dict, asyncio.Future]]) -> None:
        """Process a single batch of requests."""
        tasks = [self._process_single_request(req, future) for req, future in batch]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_request(self, req: Dict, future: asyncio.Future) -> None:
        """Process a single request within a batch."""
        try:
            result = await self.get_aggregates(**req)
            future.set_result(result)
        except Exception as e:
            # Ensure logger call and future.set_exception are indented under except
            self.logger.error(f"Error processing single request for {req.get('symbol', 'unknown')}: {e}", exc_info=True)
            future.set_exception(e)

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def get_all_tickers_snapshot(self) -> List[Dict[str, Any]]:
        """
        Fetches the snapshot data for all US stock tickers.

        Uses the /v2/snapshot/locale/us/markets/stocks/tickers endpoint.
        Includes basic retry logic for transient errors (429, 5xx).

        Returns:
            List[Dict[str, Any]]: A list of ticker snapshot dictionaries,
                                  or an empty list if the request fails.
        """
        endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Requesting all tickers snapshot.")
        params = {"apiKey": self.api_key} # API key is the main param needed here

        max_attempts = 3
        base_delay = 5 # seconds

        for attempt in range(max_attempts):
            try:
                # Use the internal _make_request which uses the connection pool
                # Note: _make_request returns the parsed JSON dict or None on failure
                response_data = await self._make_request(endpoint, params, trace_id=trace_id)

                if response_data and isinstance(response_data, dict):
                    if response_data.get("status") == "OK" and "tickers" in response_data:
                        self.logger.info(f"[TraceID: {trace_id}] Successfully fetched snapshot for {len(response_data['tickers'])} tickers.")
                        return response_data.get("tickers", [])
                    elif response_data.get("status") != "OK":
                         # Handle Polygon specific error status if present
                         error_msg = response_data.get("error", "Unknown Polygon error status")
                         self.logger.error(f"[TraceID: {trace_id}] Polygon API error fetching snapshot: {error_msg} (Status: {response_data.get('status')})")
                         return [] # Non-retryable API error
                    else:
                        # OK status but missing 'tickers' key
                        self.logger.warning(f"[TraceID: {trace_id}] Snapshot response OK but missing 'tickers' key. Response: {response_data}")
                        return []
                elif response_data is None:
                    # _make_request already logged the specific connection pool error
                    self.logger.error(f"[TraceID: {trace_id}] Failed to get snapshot response from _make_request on attempt {attempt + 1}.")
                    # Check if circuit breaker is open (though _make_request might not return specific status)
                    # For simplicity, retry on None response assuming it might be transient
                    if attempt < max_attempts - 1:
                         delay = base_delay * (2 ** attempt)
                         self.logger.warning(f"[TraceID: {trace_id}] Retrying snapshot fetch after {delay}s...")
                         await asyncio.sleep(delay)
                         continue
                    else:
                         self.logger.error(f"[TraceID: {trace_id}] Max retries reached for snapshot fetch.")
                         return []
                else:
                    # Unexpected response type
                    self.logger.error(f"[TraceID: {trace_id}] Unexpected response type from _make_request: {type(response_data)}")
                    return []

            except Exception as e:
                # Catch any other unexpected errors during the process
                self.logger.exception(f"[TraceID: {trace_id}] Unexpected error fetching snapshot on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"[TraceID: {trace_id}] Retrying snapshot fetch after {delay}s due to unexpected error...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"[TraceID: {trace_id}] Max retries reached after unexpected error during snapshot fetch.")
                    return [] # Failed after retries

        # Should not be reached if logic is correct, but as a fallback
        self.logger.error(f"[TraceID: {trace_id}] Exited snapshot fetch loop unexpectedly.")
        return []


    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Enhanced request method using the connection pool."""
        url = f"{self.base_url}{endpoint}"
        trace_id = trace_id or str(uuid.uuid4())
        request_headers = headers or {}
        request_headers["X-Request-ID"] = trace_id

        self.logger.debug(f"[TraceID: {trace_id}] Making request to {url}")
        try:
            response_data = await self.connection_pool.get(
                url, params=params, headers=request_headers, trace_id=trace_id
            )
            return response_data
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Request failed: {e}", exc_info=True)
            return None

    async def close(self) -> None:
        """Clean up resources with enhanced shutdown."""
        self.running = False
        if self.batch_processor_task and not self.batch_processor_task.done():
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                self.logger.debug("Batch processor task successfully cancelled.")
            except Exception as e:
                self.logger.error(f"Error during batch processor task cancellation: {e}", exc_info=True)

        if self.connection_pool:
            await self.connection_pool.close()
        if self.cache:
            await self.cache.close()
        self.logger.info("Polygon REST client closed")

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any: # Corrected params hint, removed timeout
        """Generic GET method for Polygon REST API endpoints."""
        url = f"{self.base_url}{endpoint}"
        trace_id = str(uuid.uuid4())
        self.logger.debug(f"[TraceID: {trace_id}] Generic GET request to {url}")

        request_params = params or {} # Handle None case
        if "apiKey" not in request_params:
            request_params["apiKey"] = self.api_key
        # request_kwargs removed as timeout is handled by the pool

        try:
            response_data = await self.connection_pool.get(
                url, params=request_params, headers={"X-Request-ID": trace_id}, # Use request_params
                trace_id=trace_id # Removed timeout kwargs
            )
            class ResponseWrapper:
                def __init__(self, data, status_code=200):
                    self.data = data
                    self.status_code = status_code
                    self.text = json.dumps(data) if isinstance(data, dict) else str(data)
                def json(self): # Already fixed in previous step, ensuring it stays fixed
                    return self.data
            return ResponseWrapper(response_data)
        except Exception as e:
            self.logger.exception(f"[TraceID: {trace_id}] Generic GET request failed: {e}")
            error_message = str(e)
            class ErrorResponseWrapper:
                def __init__(self, error_msg, status_code=500):
                    self.error_msg = error_msg
                    self.status_code = status_code
                    self.text = json.dumps({"status": "ERROR", "error": error_msg})
                def json(self): # Fixing this instance
                    return {"status": "ERROR", "error": self.error_msg}
            return ErrorResponseWrapper(error_message)

    # Add get_previous_close method if it doesn't exist, or ensure it's correct
    # This was used by the old stock_selector logic, might still be useful elsewhere
    # or needed for testing comparisons.
    @API_CIRCUIT_BREAKER
    @rate_limited
    async def get_previous_close(self, symbol: str, adjusted: bool = True) -> Any:
        """Fetches the previous day's OHLCV data for a single ticker."""
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        params = {"adjusted": str(adjusted).lower(), "apiKey": self.api_key}
        trace_id = str(uuid.uuid4())
        self.logger.debug(f"[TraceID: {trace_id}] Requesting previous close for {symbol}")

        # Using the generic 'get' method which wraps the response
        response = await self.get(endpoint, params=params)

        # The generic 'get' method returns a wrapper object.
        # We return this wrapper directly, allowing callers to check status_code, etc.
        if response.status_code != 200:
             self.logger.warning(f"[TraceID: {trace_id}] Failed to get previous close for {symbol}: Status {response.status_code} - {response.text}")

        return response

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def get_last_quote(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, float]]:
        """
        Fetches the last quote (bid/ask) for a single ticker using the snapshot endpoint.

        Args:
            symbol: The stock ticker symbol.
            use_cache: Whether to use the Redis cache.

        Returns:
            A dictionary containing 'bid' and 'ask' prices, or None if fetching fails.
        """
        trace_id = str(uuid.uuid4())
        self.logger.debug(f"[TraceID: {trace_id}] Requesting last quote for {symbol}")

        cache_key = ["last_quote", symbol]
        cache_ttl = 15 # Short TTL for quote data (15 seconds)

        if use_cache and self.cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                self.logger.debug(f"[TraceID: {trace_id}] Cache hit for {symbol} last quote.")
                try:
                    # Assuming cached data is stored as JSON string
                    quote_data = json.loads(cached_data)
                    if isinstance(quote_data, dict) and 'bid' in quote_data and 'ask' in quote_data:
                         return quote_data
                    else:
                         self.logger.warning(f"[TraceID: {trace_id}] Invalid quote data format in cache for {symbol}.")
                         await self.cache.delete(cache_key) # Clear invalid cache entry
                except json.JSONDecodeError:
                    self.logger.warning(f"[TraceID: {trace_id}] Failed to decode cached quote data for {symbol}.")
                    await self.cache.delete(cache_key) # Clear invalid cache entry


        # Use the Ticker Snapshot endpoint
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apiKey": self.api_key}

        try:
            # Use the internal _make_request for connection pooling and error handling
            response_data = await self._make_request(endpoint, params, trace_id=trace_id)

            if response_data and response_data.get("status") == "OK" and "ticker" in response_data:
                ticker_data = response_data["ticker"]
                last_quote = ticker_data.get("lastQuote")

                if last_quote and 'P' in last_quote and 'S' in last_quote and 'p' in last_quote and 's' in last_quote:
                    # P = Bid Price, S = Bid Size, p = Ask Price, s = Ask Size
                    bid_price = float(last_quote['P'])
                    ask_price = float(last_quote['p'])
                    quote = {"bid": bid_price, "ask": ask_price}

                    self.logger.debug(f"[TraceID: {trace_id}] Fetched quote for {symbol}: Bid={bid_price}, Ask={ask_price}")

                    if use_cache and self.cache:
                        try:
                            await self.cache.store(cache_key, json.dumps(quote), ttl=cache_ttl)
                        except Exception as cache_err:
                             self.logger.error(f"[TraceID: {trace_id}] Failed to cache quote for {symbol}: {cache_err}")

                    return quote
                else:
                    self.logger.warning(f"[TraceID: {trace_id}] Last quote data missing or incomplete in snapshot for {symbol}. Response: {last_quote}")
                    return None
            elif response_data and response_data.get("status") != "OK":
                 error_msg = response_data.get("error", "Unknown Polygon error status")
                 self.logger.error(f"[TraceID: {trace_id}] Polygon API error fetching snapshot for {symbol}: {error_msg} (Status: {response_data.get('status')})")
                 return None
            else:
                # Handles None response from _make_request or unexpected structure
                self.logger.error(f"[TraceID: {trace_id}] Failed to get valid snapshot response for {symbol}. Response: {response_data}")
                return None

        except Exception as e:
            self.logger.exception(f"[TraceID: {trace_id}] Error getting last quote for {symbol}: {e}")
            return None
