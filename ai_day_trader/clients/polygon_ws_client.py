"""
Enhanced Polygon.io WebSocket Client for AI Day Trader
"""

import asyncio
import json
import time
import uuid
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Coroutine

import websockets
from websockets import connect # Keep connect here
# Removed legacy client imports as they were incorrect for v10.4
from websockets.exceptions import WebSocketException
# Removed State import as it causes issues with websockets v10.4; using integer comparison instead

# Import from new structure
from ai_day_trader.utils.config import Config # Import base Config for type hint
from ai_day_trader.config import load_ai_trader_config # Import loader
from ai_day_trader.clients.redis_cache import AsyncRedisCache # Use new placeholder cache

from ai_day_trader.utils.logging_config import get_logger # Use new utils path
# Import decorators from new base
from ai_day_trader.clients.base import API_CIRCUIT_BREAKER, rate_limited


# Global registry of active clients for graceful shutdown
_active_clients = weakref.WeakSet()

# Define our own ConnectionClosed class to handle the code parameter issue
class ConnectionClosed(WebSocketException):
    """WebSocket connection is closed."""
    def __init__(self, code=1006, reason="Connection closed", sent=False):
        self.code = code
        self.reason = reason
        self.sent = sent
        super().__init__(f"ConnectionClosed: code={code}, reason={reason}, sent={sent}")

async def shutdown_all_websocket_clients() -> None:
    """Gracefully shutdown all active WebSocket clients."""
    logger = get_logger("ai_day_trader.clients.polygon_ws_shutdown")
    logger.info(f"Shutting down {len(_active_clients)} Polygon WebSocket clients")
    clients_to_close = list(_active_clients)
    for client in clients_to_close:
        if hasattr(client, 'close') and asyncio.iscoroutinefunction(client.close):
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing client {client}: {e}")
    logger.info("All Polygon WebSocket clients shut down attempt complete.")

class PolygonWebSocketClient:
    """Production-grade Polygon.io WebSocket client with enhanced features."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        message_handler: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = None,
        cache: Optional[AsyncRedisCache] = None,
        config: Optional[Config] = None, # Use base Config for type hint
        buffer_size: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        warmup_timeout: Optional[float] = None,
        health_check_interval: Optional[float] = None
    ) -> None:
        """Initialize with production configuration."""
        self.config = config or load_ai_trader_config() # Load config if not passed
        self.api_key = api_key or self.config.polygon_api_key
        self.logger = get_logger("ai_day_trader.clients.polygon_ws")

        if not self.api_key:
            self.logger.error("Polygon API Key is required for PolygonWebSocketClient.")
            raise ValueError("Polygon API Key not provided.")

        self.base_url = self.config.polygon_ws_url
        self.message_handler = message_handler
        self.cache = cache
        if self.cache is None:
             self.logger.warning("No external cache provided, initializing internal Redis cache for Polygon WS.")
             self.cache = AsyncRedisCache(prefix="polygon_ws", config=self.config)

        # Use Any for connection type hint to bypass v10.4 type issues for now
        self.connection: Optional[Any] = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.config.max_reconnect_attempts
        self.reconnect_delay = self.config.reconnect_delay
        self.buffer_size = buffer_size or self.config.buffer_size
        self.max_queue_size = max_queue_size or self.config.max_queue_size
        self.subscribed_channels: Set[str] = set()
        self.running = False
        self._receive_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._listen_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self.warmup_timeout = warmup_timeout or 30.0
        self.health_check_interval = health_check_interval or 30.0
        self.last_message_time: float = 0
        self._trace_id: str = str(uuid.uuid4())
        self._tasks: Set[asyncio.Task] = set()

        _active_clients.add(self)
        self.logger.info(f"Polygon WebSocket Client initialized for URL: {self.base_url}")

    async def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate client configuration."""
        errors = []
        if not isinstance(self.api_key, str) or len(self.api_key) < 8:
            errors.append("Invalid API key format")
        if self.buffer_size <= 0:
            errors.append("Buffer size must be positive")
        if self.max_queue_size <= 0:
            errors.append("Max queue size must be positive")
        return len(errors) == 0, errors

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def connect(self) -> None:
        """Enhanced connect with warmup and health monitoring."""
        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Starting WebSocket connection")

        if self.running and self.connected:
            self.logger.info("WebSocket already connected.")
            return

        self.running = True
        self.reconnect_attempts = 0
        self._trace_id = trace_id

        try:
            self.logger.info(f"[TraceID: {trace_id}] Starting connection attempt with {self.warmup_timeout}s timeout")
            await asyncio.wait_for(
                self._connect_attempt(),
                timeout=self.warmup_timeout
            )

            if self._health_task is None or self._health_task.done():
                self._health_task = self._create_task(
                    self._monitor_health(),
                    name=f"PolygonWSHealth-{trace_id[:8]}"
                )

        except asyncio.TimeoutError:
            self.logger.error(f"[TraceID: {trace_id}] WebSocket connection timed out during warmup")
            await self._handle_connection_error(
                asyncio.TimeoutError("Connection warmup timeout")
            )
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Unexpected connection error: {e}")
            await self._handle_connection_error(e)

    async def _connect_attempt(self) -> None:
        """Single connection attempt with enhanced logging."""
        trace_id = self._trace_id
        self.logger.info(f"[TraceID: {trace_id}] Attempting WebSocket connection to {self.base_url}")

        try:
            self.connection = await asyncio.wait_for(
                websockets.connect(
                    self.base_url, ping_interval=30, ping_timeout=20,
                    close_timeout=10, open_timeout=20, max_size=None,
                    max_queue=None, compression=None
                ),
                timeout=20.0
            )
            self.logger.info(f"[TraceID: {trace_id}] WebSocket connection established successfully")
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] WebSocket connection failed: {e}")
            raise WebSocketException(f"Failed to establish WebSocket connection: {e}") from e

        # Authentication
        try:
            if not self.connection: # Add check
                raise WebSocketException("Connection object is None after successful connect attempt.")

            self.logger.info(f"[TraceID: {trace_id}] Waiting for initial connection message...")
            initial_response = await asyncio.wait_for(self.connection.recv(), timeout=15.0)
            initial_data = json.loads(initial_response)
            self.logger.info(f"[TraceID: {trace_id}] Initial connection response: {initial_data}")
            if isinstance(initial_data, list) and len(initial_data) > 0 and initial_data[0].get("status") == "connected":
                 self.logger.info(f"[TraceID: {trace_id}] Connection confirmed by server.")
            else:
                 self.logger.warning(f"[TraceID: {trace_id}] Unexpected initial connection response format.")


            self.logger.info(f"[TraceID: {trace_id}] Sending authentication request...")
            auth_message = {"action": "auth", "params": self.api_key}
            await self.connection.send(json.dumps(auth_message)) # Add check before send

            auth_response = await asyncio.wait_for(self.connection.recv(), timeout=10.0) # Add check before recv
            auth_data = json.loads(auth_response)

            if not isinstance(auth_data, list) or len(auth_data) == 0:
                raise WebSocketException(f"Invalid auth response format: {auth_data}")

            auth_status = auth_data[0].get("status")
            auth_message_text = auth_data[0].get("message", "Unknown error")

            if auth_status != "auth_success":
                raise WebSocketException(f"Authentication failed: {auth_message_text}")

            self.logger.info(f"[TraceID: {trace_id}] WebSocket authentication successful")
            self.connected = True
            self.reconnect_attempts = 0
            self.last_message_time = time.time()

            if self.subscribed_channels:
                await self._resubscribe()

            if self._listen_task is None or self._listen_task.done():
                self._listen_task = self._create_task(
                    self.listen(),
                    name=f"PolygonWSListener-{trace_id[:8]}"
                )

        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Authentication or post-auth step failed: {e}")
            if self.connection and not self.connection.closed:
                try:
                    await asyncio.shield(self.connection.close())
                except Exception:
                    pass
            await self._handle_connection_error(WebSocketException(f"Authentication failed: {e}"))

    async def _monitor_health(self) -> None:
        """Monitor connection health and message flow."""
        trace_id = self._trace_id
        self.logger.info(f"[TraceID: {trace_id}] Starting health monitoring")
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                if self.connected and time.time() - self.last_message_time > self.health_check_interval * 2:
                    self.logger.warning(f"[TraceID: {trace_id}] No messages received in {self.health_check_interval*2:.1f}s, sending heartbeat.")
                    await self._send_heartbeat()
                queue_size = self._receive_queue.qsize()
                if queue_size > self.max_queue_size * 0.8:
                    self.logger.warning(f"[TraceID: {trace_id}] Receive queue high watermark: {queue_size}/{self.max_queue_size}")
            except asyncio.CancelledError:
                 self.logger.info(f"[TraceID: {trace_id}] Health monitor cancelled.")
                 break
            except Exception as e:
                self.logger.error(f"[TraceID: {trace_id}] Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat (ping frame) to check connection."""
        trace_id = self._trace_id
        try:
            # Add explicit check for self.connection
            if self.is_connected() and self.connection:
                await self.connection.ping()
                self.logger.debug(f"[TraceID: {trace_id}] Sent WebSocket ping frame.")
            else:
                 self.logger.warning(f"[TraceID: {trace_id}] Cannot send heartbeat, not connected.")
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Heartbeat (ping) failed: {e}")
            await self._handle_connection_error(e)

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Enhanced message sending with connection checks."""
        if not self.is_connected():
            error_msg = "Cannot send message: WebSocket not connected"
            self.logger.error(error_msg)
            # Raise a generic WebSocketException instead of specific ConnectionClosed
            # to avoid parameter type mismatch issues with websockets v10.x
            raise WebSocketException(error_msg)

        trace_id = message.get("trace_id", self._trace_id)
        try:
            # Add explicit check for self.connection
            if not self.connection:
                 raise WebSocketException("Cannot send message: Connection is None.")
            start_time = time.time()
            await self.connection.send(json.dumps(message))
            self.logger.debug(f"[TraceID: {trace_id}] Sent message in {(time.time()-start_time)*1000:.2f}ms: {message.get('action', 'unknown')}")
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"[TraceID: {trace_id}] Connection closed while sending: {e.code} {e.reason}")
            await self._handle_connection_error(e)
            raise
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Error sending message: {e}")
            raise WebSocketException(f"Failed to send message: {e}") from e

    async def _handle_connection_error(self, error: Exception) -> None:
        """Enhanced error handling with tracing and reconnection logic."""
        self.connected = False
        self.reconnect_attempts += 1
        attempt = self.reconnect_attempts
        trace_id = self._trace_id
        error_type = type(error).__name__
        error_msg = str(error)

        close_code = getattr(error, 'code', None)
        close_reason = getattr(error, 'reason', None)
        log_msg = f"[TraceID: {trace_id}] Connection error (Attempt {attempt}/{self.max_reconnect_attempts}): {error_type}"
        if close_code:
            log_msg += f" Code={close_code}"
        if close_reason:
            log_msg += f" Reason='{close_reason}'"
        if error_msg and error_msg != close_reason:
            log_msg += f" Msg='{error_msg}'"
        self.logger.warning(log_msg)

        if attempt >= self.max_reconnect_attempts:
            self.logger.error(f"[TraceID: {trace_id}] Max reconnection attempts reached. Stopping.")
            self.running = False
            return

        delay = min(self.reconnect_delay * (1.5 ** (attempt - 1)), 60)
        self.logger.info(f"[TraceID: {trace_id}] Will attempt reconnect in {delay:.2f}s")
        await asyncio.sleep(delay)

        if self.running:
            self.logger.info(f"[TraceID: {trace_id}] Attempting reconnect...")
            if self.connection and not self.connection.closed:
                try:
                    await asyncio.shield(self.connection.close())
                except Exception:
                    pass
            self.connection = None
            self._create_task(self.connect(), name=f"ReconnectTask-{trace_id[:8]}")


    async def _resubscribe(self) -> None:
        """Resubscribe to stored channels after reconnection."""
        trace_id = self._trace_id
        if not self.is_connected() or not self.subscribed_channels:
            return

        self.logger.info(f"[TraceID: {trace_id}] Resubscribing to {len(self.subscribed_channels)} channels")
        try:
            batch_size = 100
            channel_list = list(self.subscribed_channels)
            for i in range(0, len(channel_list), batch_size):
                batch = channel_list[i:i+batch_size]
                await self._send_message({
                    "action": "subscribe",
                    "params": ",".join(batch),
                    "trace_id": trace_id
                })
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Resubscribe failed: {e}")

    @API_CIRCUIT_BREAKER
    @rate_limited
    async def subscribe(self, channels: List[str]) -> None:
        """Subscribe to WebSocket channels."""
        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Subscribe request for channels: {channels}")

        if not self.is_connected():
            self.logger.warning(f"[TraceID: {trace_id}] Not connected, attempting connection first.")
            await self.connect()
            if not self.is_connected():
                raise ConnectionError("Failed to connect before subscribing.")

        new_channels = [ch for ch in channels if ch not in self.subscribed_channels]
        if not new_channels:
            self.logger.info(f"[TraceID: {trace_id}] Already subscribed to all requested channels.")
            return

        self.logger.info(f"[TraceID: {trace_id}] Subscribing to {len(new_channels)} new channels.")
        try:
            batch_size = 100
            for i in range(0, len(new_channels), batch_size):
                batch = new_channels[i:i+batch_size]
                await self._send_message({
                    "action": "subscribe",
                    "params": ",".join(batch),
                    "trace_id": trace_id
                })
                self.subscribed_channels.update(batch)
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Subscribe failed: {e}")
            self.subscribed_channels.difference_update(new_channels)
            raise

    async def unsubscribe(self, channels: List[str]) -> None:
        """Unsubscribe from WebSocket channels."""
        trace_id = str(uuid.uuid4())
        self.logger.info(f"[TraceID: {trace_id}] Unsubscribe request for channels: {channels}")

        if not self.is_connected():
            self.logger.warning(f"[TraceID: {trace_id}] Not connected, cannot unsubscribe.")
            self.subscribed_channels.difference_update(channels)
            return

        to_unsubscribe = [ch for ch in channels if ch in self.subscribed_channels]
        if not to_unsubscribe:
            self.logger.info(f"[TraceID: {trace_id}] Not subscribed to any of the requested channels.")
            return

        self.logger.info(f"[TraceID: {trace_id}] Unsubscribing from {len(to_unsubscribe)} channels.")
        try:
            batch_size = 100
            for i in range(0, len(to_unsubscribe), batch_size):
                batch = to_unsubscribe[i:i+batch_size]
                await self._send_message({
                    "action": "unsubscribe",
                    "params": ",".join(batch),
                    "trace_id": trace_id
                })
                self.subscribed_channels.difference_update(batch)
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"[TraceID: {trace_id}] Unsubscribe failed: {e}")
            self.logger.warning(f"[TraceID: {trace_id}] State might be inconsistent after unsubscribe failure.")
            raise

    async def listen(self) -> None:
        """Listen for messages and put them into the queue."""
        trace_id = self._trace_id
        self.logger.info(f"[TraceID: {trace_id}] Listener starting...")

        if not self.is_connected():
            self.logger.error(f"[TraceID: {trace_id}] Cannot start listener, not connected.")
            return

        while self.running:
            try:
                # Add explicit check for self.connection
                if not self.connection:
                    self.logger.warning(f"[TraceID: {trace_id}] Connection lost during listen loop. Attempting reconnect.")
                    await self._handle_connection_error(WebSocketException("Connection is None in listen loop"))
                    continue # Skip rest of loop iteration

                message_str = await self.connection.recv()
                recv_time = time.time()
                self.last_message_time = recv_time

                try:
                    message_data = json.loads(message_str)
                    if not isinstance(message_data, list):
                        self.logger.warning(f"[TraceID: {trace_id}] Received non-list message: {message_data}")
                        continue

                    for event in message_data:
                        if not isinstance(event, dict):
                             self.logger.warning(f"[TraceID: {trace_id}] Received non-dict event in list: {event}")
                             continue

                        event_type = event.get('ev', 'unknown')
                        event['_recv_time'] = recv_time
                        event['_trace_id'] = trace_id

                        if self.cache and event_type in ['T', 'Q']:
                             symbol = event.get('sym')
                             if symbol:
                                  cache_key = f"latest_{event_type}:{symbol}"
                                  self._create_task(self.cache.set(cache_key, json.dumps(event), ttl=60))

                        try:
                            self._receive_queue.put_nowait(event)
                        except asyncio.QueueFull:
                            self.logger.warning(f"[TraceID: {trace_id}] Receive queue full, discarding message for {event_type}")

                        if self.message_handler:
                            try:
                                self._create_task(
                                    self.message_handler(event),
                                    name=f"MsgHandler-{event_type}-{trace_id[:8]}"
                                )
                            except Exception as handler_error:
                                self.logger.error(f"[TraceID: {trace_id}] Error in message handler: {handler_error}", exc_info=True)

                except json.JSONDecodeError:
                    self.logger.warning(f"[TraceID: {trace_id}] Received invalid JSON: {message_str[:200]}...")

            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"[TraceID: {trace_id}] Connection closed during listen: Code={e.code}, Reason='{e.reason}'. Listener stopping.")
                await self._handle_connection_error(e)
                break
            except asyncio.CancelledError:
                 self.logger.info(f"[TraceID: {trace_id}] Listener task cancelled.")
                 break
            except Exception as e:
                self.logger.error(f"[TraceID: {trace_id}] Unexpected error in listener loop: {e}", exc_info=True)
                await self._handle_connection_error(e)
                break

        self.logger.info(f"[TraceID: {trace_id}] Listener stopped.")

    def _validate_message(self, message: Any) -> bool:
        """Validate incoming message structure (basic check)."""
        return isinstance(message, list) and all(isinstance(item, dict) for item in message)

    async def receive(self) -> Dict[str, Any]:
        """Receive the next message from the internal queue."""
        try:
            return await self._receive_queue.get()
        except asyncio.CancelledError:
             self.logger.info("Receive task cancelled.")
             raise

    def _create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Create and track an asyncio task."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def close(self) -> None:
        """Gracefully close the WebSocket connection and associated tasks."""
        trace_id = self._trace_id
        self.logger.info(f"[TraceID: {trace_id}] Closing WebSocket client...")
        self.running = False

        tasks_to_cancel = []
        if self._listen_task and not self._listen_task.done():
            tasks_to_cancel.append(self._listen_task)
        if self._health_task and not self._health_task.done():
            tasks_to_cancel.append(self._health_task)
        # Removed check for non-existent self.batch_processor_task
        # if self.batch_processor_task and not self.batch_processor_task.done():
        #     tasks_to_cancel.append(self.batch_processor_task)

        for task in tasks_to_cancel:
            task.cancel()

        # Check state instead of close_code or closed
        # Use integer 1 for OPEN state check (compatible with websockets v10.x)
        if self.connection and self.connection.state == 1: # State.OPEN
            try:
                await asyncio.wait_for(self.connection.close(), timeout=5.0)
                self.logger.info(f"[TraceID: {trace_id}] WebSocket connection closed.")
            except asyncio.TimeoutError:
                self.logger.warning(f"[TraceID: {trace_id}] Timeout closing WebSocket connection.")
            except Exception as e:
                self.logger.error(f"[TraceID: {trace_id}] Error closing WebSocket connection: {e}")
        self.connection = None
        self.connected = False

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            self.logger.debug(f"[TraceID: {trace_id}] Listener, health, and batch tasks cancelled.")

        if self.cache:
            await self.cache.close()

        self._tasks.clear()
        _active_clients.discard(self)
        self.logger.info(f"[TraceID: {trace_id}] Polygon WebSocket client closed.")

    def is_connected(self) -> bool:
        """Check connection status."""
        # Check state instead of close_code or closed
        # Use integer 1 for OPEN state check (compatible with websockets v10.x)
        return self.connected and self.connection is not None and self.connection.state == 1 # State.OPEN
