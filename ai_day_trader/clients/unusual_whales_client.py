"""
Unusual Whales API Client for AI Day Trader
"""

import asyncio
import uuid
import json
from typing import Any, Dict, List, Optional, Union, Tuple, Set

# Import from new structure
from ai_day_trader.utils.config import Config # Import base Config for type hint
from ai_day_trader.config import load_ai_trader_config # Import loader
from ai_day_trader.clients.redis_cache import AsyncRedisCache # Use new placeholder cache
from ai_day_trader.clients.async_connection_pool import AsyncConnectionPool # Use new pool
from ai_day_trader.utils.logging_config import get_logger # Use new utils path
# Import decorators from new base
from ai_day_trader.clients.base import API_CIRCUIT_BREAKER, rate_limited


class UnusualWhalesClient:
    """Production-grade Unusual Whales API client with enhanced features."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[AsyncRedisCache] = None,
        connection_pool: Optional[AsyncConnectionPool] = None,
        config: Optional[Config] = None, # Use base Config for type hint
        batch_size: int = 10,
        health_check_interval: Optional[float] = None
    ) -> None:
        """Initialize with production configuration."""
        self.config = config or load_ai_trader_config() # Load config if not passed
        self.api_key = api_key or self.config.unusual_whales_api_key
        self.logger = get_logger("ai_day_trader.clients.unusual_whales")

        if not self.api_key:
            self.logger.warning("Unusual Whales API Key not provided. Client will be disabled.")

        self.base_url = self.config.unusual_whales_api_base_url
        self.cache = cache
        if self.cache is None:
             self.logger.warning("No external cache provided, initializing internal Redis cache for UW.")
             self.cache = AsyncRedisCache(prefix="unusual_whales", config=self.config, ttl=self.config.unusual_whales_cache_ttl)


        self.connection_pool = connection_pool or AsyncConnectionPool(config=self.config)
        self.running = True
        self.batch_size = batch_size
        self.health_check_interval = health_check_interval or 60.0
        self._health_task: Optional[asyncio.Task] = None
        self._trace_id: str = str(uuid.uuid4())
        self._tasks: Set[asyncio.Task] = set()

        if self.health_check_interval > 0:
             self._health_task = self._create_task(self._monitor_health(), name="UWHealthMonitor")

        self.logger.info(f"Unusual Whales Client initialized for base URL: {self.base_url}")


    async def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate client configuration."""
        errors = []
        if not self.api_key:
            errors.append("API key not configured")
        elif not isinstance(self.api_key, str) or len(self.api_key) < 32:
            errors.append("Invalid API key format")
        return len(errors) == 0, errors

    def _create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Create a tracked task."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def get_flow_alerts(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get flow alerts with production enhancements."""
        if not self.api_key:
            return []

        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Getting flow alerts for ticker: {ticker or 'all'}")

        cache_key_parts = ["flow_alerts", str(limit)]
        if ticker:
            cache_key_parts.append(f"ticker_{ticker}")
        cache_key = "_".join(cache_key_parts)

        if use_cache and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                try:
                    cached_data = json.loads(cached)
                    if isinstance(cached_data, list):
                        self.logger.debug(f"[TraceID: {trace_id}] Cache hit for flow alerts: {ticker or 'all'}")
                        return cached_data
                except json.JSONDecodeError:
                    self.logger.warning(f"[TraceID: {trace_id}] Failed to decode cached data for {cache_key}")

        endpoint = "/alerts/options-flow"
        # Ensure params values are strings as expected by HTTP query params
        params: Dict[str, Any] = {"limit": str(min(max(limit, 1), 200))}
        if ticker:
            params["ticker"] = str(ticker) # Ensure ticker is string

        headers = {"Authorization": f"Bearer {self.api_key}", "X-Request-ID": trace_id}

        try:
            response_data = await self._make_request(endpoint, params, headers=headers, trace_id=trace_id)

            alerts_list = []
            if isinstance(response_data, dict) and 'data' in response_data:
                alerts_list = response_data['data']
            elif isinstance(response_data, list):
                alerts_list = response_data

            if not isinstance(alerts_list, list):
                self.logger.warning(f"[TraceID: {trace_id}] Invalid alerts format received: {type(alerts_list)}")
                return []

            if use_cache and self.cache and alerts_list:
                await self.cache.set(cache_key, json.dumps(alerts_list))

            return alerts_list

        except Exception as e:
            self.logger.exception(f"[TraceID: {trace_id}] Error getting flow alerts: {e}")
            return []

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def get_historical_flow(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get historical flow data with production enhancements."""
        if not self.api_key:
            return []

        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Getting historical flow for {ticker} ({start_date} to {end_date})")

        cache_key = ["historical_flow", ticker, start_date, end_date]

        if use_cache and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                 try:
                     cached_data = json.loads(cached)
                     if isinstance(cached_data, list):
                         self.logger.debug(f"[TraceID: {trace_id}] Cache hit for historical flow: {ticker}")
                         return cached_data
                 except json.JSONDecodeError:
                     self.logger.warning(f"[TraceID: {trace_id}] Failed to decode cached data for {cache_key}")

        endpoint = "/history/options_flow"
        # Ensure params values are strings
        params: Dict[str, Any] = {"start_date": str(start_date), "end_date": str(end_date), "ticker": str(ticker)}
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Request-ID": trace_id}

        try:
            response_data = await self._make_request(endpoint, params, headers=headers, trace_id=trace_id)

            flow_list = []
            if isinstance(response_data, dict) and 'data' in response_data:
                flow_list = response_data['data']
            elif isinstance(response_data, list):
                flow_list = response_data

            if not isinstance(flow_list, list):
                self.logger.warning(f"[TraceID: {trace_id}] Invalid historical flow data format: {type(flow_list)}")
                return []

            if use_cache and self.cache and flow_list:
                await self.cache.set(cache_key, json.dumps(flow_list))

            return flow_list

        except Exception as e:
            self.logger.exception(f"[TraceID: {trace_id}] Error getting historical flow: {e}")
            return []

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def get_unusual_options(
        self,
        min_volume: Optional[int] = None,
        min_premium: Optional[float] = None,
        limit: int = 100,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get unusual options activity with production enhancements."""
        if not self.api_key:
            return []

        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Getting unusual options activity")

        cache_key_parts = ["unusual_options", str(limit)]
        if min_volume is not None:
            cache_key_parts.append(f"vol_{min_volume}")
        if min_premium is not None:
            cache_key_parts.append(f"prem_{min_premium}")
        cache_key = "_".join(cache_key_parts)

        if use_cache and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                 try:
                     cached_data = json.loads(cached)
                     if isinstance(cached_data, list):
                         self.logger.debug(f"[TraceID: {trace_id}] Cache hit for unusual options.")
                         return cached_data
                 except json.JSONDecodeError:
                     self.logger.warning(f"[TraceID: {trace_id}] Failed to decode cached data for {cache_key}")

        endpoint = "/options/unusual_activity"
        # Ensure params values are strings
        params: Dict[str, Any] = {"limit": str(limit)}
        if min_volume is not None:
            params["min_volume"] = str(min_volume) # Convert to string
        if min_premium is not None:
            params["min_premium"] = str(min_premium) # Convert to string

        headers = {"Authorization": f"Bearer {self.api_key}", "X-Request-ID": trace_id}

        try:
            response_data = await self._make_request(endpoint, params, headers=headers, trace_id=trace_id)

            options_list = []
            if isinstance(response_data, dict) and 'data' in response_data:
                options_list = response_data['data']
            elif isinstance(response_data, list):
                options_list = response_data

            if not isinstance(options_list, list):
                self.logger.warning(f"[TraceID: {trace_id}] Invalid unusual options format: {type(options_list)}")
                return []

            if use_cache and self.cache and options_list:
                await self.cache.set(cache_key, json.dumps(options_list))

            return options_list

        except Exception as e:
            self.logger.exception(f"[TraceID: {trace_id}] Error getting unusual options: {e}")
            return []

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None
    ) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Enhanced request method using the connection pool."""
        if not self.api_key:
            return None

        trace_id = trace_id or str(uuid.uuid4())
        url = f"{self.base_url}{endpoint}"

        request_headers = headers or {}
        request_headers['X-Request-ID'] = trace_id
        if "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {self.api_key}"
        if "Accept" not in request_headers:
             request_headers["Accept"] = "application/json"

        try:
            self.logger.debug(f"[TraceID: {trace_id}] Making request to {url} with params {params}")
            response_data = await self.connection_pool.get(
                url,
                params=params,
                headers=request_headers,
                trace_id=trace_id
            )
            if isinstance(response_data, dict) and response_data.get("success") is False:
                 error_msg = response_data.get("message", "Unusual Whales API error")
                 self.logger.error(f"[TraceID: {trace_id}] Unusual Whales API Error: {error_msg}")
                 return None

            return response_data
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Request failed: {e}", exc_info=True)
            return None

    async def _monitor_health(self) -> None:
        """Monitor API health status."""
        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Starting health monitoring for Unusual Whales API")
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                self.logger.debug(f"[TraceID: {trace_id}] Performing UW health check...")
                result = await self.get_flow_alerts(limit=1, use_cache=False)
                if result is None:
                     raise ConnectionError("Health check request failed")
                self.logger.debug(f"[TraceID: {trace_id}] UW health check successful.")
            except asyncio.CancelledError:
                 self.logger.info(f"[TraceID: {trace_id}] UW Health monitor cancelled.")
                 break
            except Exception as e:
                self.logger.error(f"[TraceID: {trace_id}] UW Health monitor error during check: {e}", exc_info=True)
                await asyncio.sleep(self.health_check_interval * 2)

    async def close(self) -> None:
        """Clean up resources with graceful shutdown."""
        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Closing Unusual Whales client")
        self.running = False

        tasks_to_cancel = [t for t in self._tasks if not t.done()]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            try:
                await asyncio.wait(tasks_to_cancel, timeout=5.0, return_when=asyncio.ALL_COMPLETED)
            except Exception as e:
                self.logger.error(f"[TraceID: {trace_id}] Error waiting for tasks during close: {e}")

        if self.connection_pool:
            await self.connection_pool.close()
        if self.cache:
            await self.cache.close()

        self._tasks.clear()
        self.logger.info(f"[TraceID: {trace_id}] Unusual Whales client closed")
