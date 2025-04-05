"""
Enhanced Asynchronous HTTP Connection Pool for AI Day Trader

This module provides an enhanced asynchronous HTTP connection pool:
- Connection pooling with proper resource management
- Retry logic with exponential backoff
- Circuit breaker integration (via decorators)
- Metrics collection (placeholders, can be re-added if needed)
- Graceful shutdown
"""

import asyncio
import random
import uuid
# import logging # Removed unused import
from typing import Any, Dict, Optional, Set

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector

# Import base Config class for type hinting and loader function
from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.config import load_ai_trader_config

from ai_day_trader.utils.logging_config import get_logger # Use new utils path

# Configure logger
logger = get_logger("ai_day_trader.clients.async_connection_pool")

class AsyncConnectionPool:
    """Enhanced asynchronous HTTP connection pool with retry logic"""

    def __init__(
        self,
        config: Optional[Config] = None, # Use Config from utils for type hint
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None,
        max_pool_size: Optional[int] = None,
        timeout: Optional[int] = None,
        user_agent: str = "AIDayTraderClient/1.0"
    ) -> None:
        """Initialize the connection pool"""
        # If no config is passed, load the default AI trader config
        self.config = config or load_ai_trader_config() # Use loader function
        # Access attributes directly from the loaded config object
        self.max_retries = max_retries or self.config.max_retries
        self.backoff_factor = backoff_factor or self.config.retry_backoff_factor
        self.max_pool_size = max_pool_size or self.config.max_pool_size
        self.timeout = timeout or self.config.connection_timeout
        self.user_agent = user_agent

        self.session: Optional[ClientSession] = None
        self.connector: Optional[TCPConnector] = None
        self._tasks: Set[asyncio.Task] = set()
        self._initialization_lock = asyncio.Lock()
        self._trace_id = str(uuid.uuid4())

    async def initialize(self) -> bool:
        """Initialize the aiohttp session"""
        if self.session is not None and not self.session.closed:
            return True

        async with self._initialization_lock:
            if self.session is not None and not self.session.closed:
                return True

            try:
                self.connector = TCPConnector(
                    limit=self.max_pool_size,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    ssl=False,
                    enable_cleanup_closed=True
                )

                self.session = ClientSession(
                    connector=self.connector,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "application/json",
                    },
                    timeout=ClientTimeout(total=self.timeout),
                    raise_for_status=False
                )

                logger.info(f"[TraceID: {self._trace_id}] Connection pool initialized with size {self.max_pool_size}")
                return True
            except Exception as e:
                logger.error(f"[TraceID: {self._trace_id}] Failed to initialize connection pool: {e}", exc_info=True)
                self.session = None
                self.connector = None
                return False

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make GET request with retry logic"""
        trace_id = trace_id or str(uuid.uuid4())

        if self.session is None or self.session.closed:
            logger.info(f"[TraceID: {trace_id}] Session not initialized or closed, attempting initialization.")
            if not await self.initialize():
                return {"status": "ERROR", "error": "Failed to initialize session"}

        last_error = None
        request_headers = headers or {}

        for attempt in range(self.max_retries + 1):
            try:
                # Add explicit check for self.session before using it
                if self.session is None:
                    raise aiohttp.ClientError("Session is None, cannot make request.")

                logger.debug(f"[TraceID: {trace_id}] GET attempt {attempt+1}/{self.max_retries+1} to {url}")
                # Revert: Pass timeout directly to session.get
                async with self.session.get(url, params=params, headers=request_headers, timeout=ClientTimeout(total=self.timeout)) as response:
                    if 200 <= response.status < 300:
                        try:
                            result = await response.json(content_type=None)
                            logger.debug(f"[TraceID: {trace_id}] Request successful (Status: {response.status})")
                            return result
                        except aiohttp.ContentTypeError as _: # Replaced unused 'e' with '_'
                             text_response = await response.text()
                             logger.warning(f"[TraceID: {trace_id}] Non-JSON success response (Status: {response.status}): {text_response[:100]}...")
                             return {"status": "SUCCESS_NON_JSON", "content": text_response}
                        except Exception as e:
                            last_error = f"JSON decode error: {e}"
                            logger.warning(f"[TraceID: {trace_id}] {last_error}")
                            return {"status": "ERROR", "error": last_error}

                    elif response.status == 429:
                        wait_time = (2**attempt) * self.backoff_factor + random.uniform(0, 1)
                        logger.warning(f"[TraceID: {trace_id}] Rate limited (Status: 429), retrying in {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    elif 400 <= response.status < 500:
                        error_text = await response.text()
                        last_error = f"HTTP Client Error {response.status}: {error_text}"
                        logger.error(f"[TraceID: {trace_id}] {last_error}")
                        return {"status": "ERROR", "error": last_error, "status_code": response.status}

                    else: # >= 500
                        error_text = await response.text()
                        last_error = f"HTTP Server Error {response.status}: {error_text}"
                        logger.warning(f"[TraceID: {trace_id}] {last_error}, retrying...")

            # Catch original exception types
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = f"Request connection error: {type(e).__name__}: {e!s}"
                logger.warning(f"[TraceID: {trace_id}] {last_error}, retrying...")
            except Exception as e:
                 last_error = f"Unexpected error during request: {type(e).__name__}: {e!s}"
                 logger.error(f"[TraceID: {trace_id}] {last_error}", exc_info=True)

            # Retry Logic
            if attempt < self.max_retries:
                wait_time = (2**attempt) * self.backoff_factor + random.uniform(0, 0.5)
                logger.info(f"[TraceID: {trace_id}] Waiting {wait_time:.2f}s before retry {attempt+2}")
                await asyncio.sleep(wait_time)
            else:
                 logger.error(f"[TraceID: {trace_id}] Max retries ({self.max_retries+1}) exceeded. Last error: {last_error}")

        return {"status": "ERROR", "error": last_error or "Max retries exceeded"}

    def create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Create and track an asyncio task."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def close(self) -> None:
        """Close the session and all connections with proper cleanup"""
        logger.info(f"[TraceID: {self._trace_id}] Closing connection pool")

        pending_tasks = [t for t in self._tasks if not t.done()]
        if pending_tasks:
            logger.info(f"[TraceID: {self._trace_id}] Cancelling {len(pending_tasks)} pending tasks")
            for task in pending_tasks:
                task.cancel()
            try:
                await asyncio.wait(pending_tasks, timeout=5.0, return_when=asyncio.ALL_COMPLETED)
            except asyncio.TimeoutError:
                logger.warning(f"[TraceID: {self._trace_id}] Timeout waiting for tasks to cancel")
            except Exception as e:
                 logger.error(f"[TraceID: {self._trace_id}] Error waiting for tasks during close: {e}")

        if self.session and not self.session.closed:
            try:
                await self.session.close()
                logger.info(f"[TraceID: {self._trace_id}] Session closed")
            except Exception as e:
                logger.error(f"[TraceID: {self._trace_id}] Error closing session: {e}")
        self.session = None
        self.connector = None

        self._tasks.clear()
        logger.info(f"[TraceID: {self._trace_id}] Connection pool closed.")
