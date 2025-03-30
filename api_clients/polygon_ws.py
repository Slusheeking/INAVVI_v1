"""
Enhanced Polygon.io WebSocket Client

This module provides a comprehensive WebSocket client for Polygon.io with:
- Automatic reconnection
- Message buffering via internal queue
- Connection state management
- Comprehensive metrics
- Advanced error handling
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor # Keep for potential future non-GPU offloading
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Import relevant base components and config
from .base import (
    logger, # Use the logger from base
    API_WEBSOCKET_RECONNECTS,
    API_WEBSOCKET_MESSAGES,
    API_ERROR_COUNT,
    RedisCache,
    # GPUAccelerator removed
    POLYGON_API_KEY,
    MAX_RECONNECT_ATTEMPTS,
    RECONNECT_DELAY,
    handle_exception, # Keep if used elsewhere, otherwise remove
    handle_asyncio_exception # Keep if used elsewhere, otherwise remove
)
# Import centralized GPU utilities if needed (unlikely for WS client itself)
# from utils.gpu_utils import is_gpu_available, clear_gpu_memory

class PolygonWebSocketClient:
    """Enhanced Polygon.io WebSocket client with reconnection and message queuing."""

    def __init__(
        self,
        api_key: str = POLYGON_API_KEY,
        message_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
        cache: Optional[RedisCache] = None,
        # use_gpu parameter removed
        max_reconnect_attempts: int = MAX_RECONNECT_ATTEMPTS,
        reconnect_delay: float = RECONNECT_DELAY,
        buffer_size: int = 1000 # Queue size
    ) -> None:
        """Initialize the WebSocket client."""
        if not api_key:
             logger.error("Polygon API Key is required for PolygonWebSocketClient.")
             raise ValueError("Polygon API Key not provided.")
        self.api_key = api_key
        self.base_url = "wss://socket.polygon.io/stocks"
        self.message_handler = message_handler
        self.cache = cache # Cache might still be useful for other things
        # self.gpu_accelerator removed
        self.connection: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        self.subscribed_channels: Set[str] = set()
        self.running = False
        # self.message_buffer removed - replaced by queue
        # self.thread_pool removed - was only for placeholder GPU processing
        self._receive_queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size * 2) # Queue for external consumers
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish WebSocket connection, authenticate, and start listener task."""
        if self.running and self.connected:
             logger.info("WebSocket already connected.")
             return

        self.running = True
        self.reconnect_attempts = 0 # Reset attempts on explicit connect call

        while self.running and not self.connected and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Attempting WebSocket connection (attempt {self.reconnect_attempts + 1})...")
                # Increase open timeout slightly
                self.connection = await websockets.connect(
                    self.base_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    open_timeout=15 # Added open timeout
                )

                # Authenticate
                await self._send_message({"action": "auth", "params": self.api_key})
                auth_response = await asyncio.wait_for(self.connection.recv(), timeout=10) # Increased auth timeout
                auth_data = json.loads(auth_response)
                if isinstance(auth_data, list) and auth_data[0].get("status") == "auth_success":
                     logger.info("WebSocket authentication successful.")
                     self.connected = True
                     self.reconnect_attempts = 0 # Reset on success

                     # Resubscribe to channels if needed
                     if self.subscribed_channels:
                          await self._resubscribe()

                     # Start the listener task if not already running
                     if self._listen_task is None or self._listen_task.done():
                          self._listen_task = asyncio.create_task(self.listen(), name="PolygonWSListener")
                          logger.info("WebSocket listener task started.")
                     break # Exit connection loop on success
                else:
                     logger.error(f"WebSocket authentication failed: {auth_data}")
                     await self.connection.close()
                     raise WebSocketException("Authentication failed")

            except (ConnectionClosed, WebSocketException, asyncio.TimeoutError, OSError) as e:
                await self._handle_connection_error(e)
            except Exception as e:
                logger.exception(f"Unexpected error during WebSocket connection attempt: {e}")
                await self._handle_connection_error(e)

        if not self.connected:
             logger.critical("Failed to connect to Polygon WebSocket after multiple attempts.")
             self.running = False


    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send JSON message with error handling"""
        if not self.is_connected(): # Use helper method
             logger.error("Cannot send message: WebSocket not connected.")
             raise ConnectionClosed("WebSocket not connected")

        try:
            await self.connection.send(json.dumps(message))
            logger.debug(f"Sent message: {message}")
        except ConnectionClosed as e:
            logger.warning(f"Connection closed while sending message: {e}")
            self.connected = False
            raise # Re-raise the specific error
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise WebSocketException(f"Failed to send message: {e}") from e


    async def _handle_connection_error(self, error: Exception) -> None:
        """Handles connection errors, increments attempts, and implements backoff delay."""
        self.connected = False
        self.reconnect_attempts += 1
        attempt = self.reconnect_attempts
        max_attempts = self.max_reconnect_attempts

        error_type = type(error).__name__
        logger.warning(
            f"WebSocket connection error (Attempt {attempt}/{max_attempts}): {error_type} - {error}"
        )
        API_ERROR_COUNT.labels(
            client="polygon", endpoint="websocket_connect", method="CONNECT", error_type=error_type
        ).inc()

        if attempt >= max_attempts:
            logger.error(f"Max reconnection attempts ({max_attempts}) reached. Stopping reconnection efforts.")
            self.running = False
            return

        delay = min(self.reconnect_delay * (2 ** (attempt - 1)), 60)
        logger.info(f"Will attempt reconnection in {delay:.2f} seconds...")
        await asyncio.sleep(delay)


    async def _resubscribe(self) -> None:
        """Resubscribe to tracked channels after a successful reconnection."""
        if self.is_connected() and self.subscribed_channels:
            try:
                logger.info(f"Resubscribing to {len(self.subscribed_channels)} channels...")
                batch_size = 100
                channel_list = list(self.subscribed_channels)
                for i in range(0, len(channel_list), batch_size):
                     batch = channel_list[i:i+batch_size]
                     await self._send_message({
                          "action": "subscribe",
                          "params": ",".join(batch)
                     })
                     logger.debug(f"Sent resubscribe request for batch: {batch}")
                     await asyncio.sleep(0.1)
                logger.info(f"Successfully sent resubscribe requests for all channels.")
            except Exception as e:
                 logger.error(f"Failed to resubscribe to channels after reconnect: {e}")
        elif not self.subscribed_channels:
             logger.info("No channels to resubscribe to.")


    async def subscribe(self, channels: List[str]) -> None:
        """Subscribes to a list of WebSocket channel names."""
        if not self.is_connected():
             logger.warning("Cannot subscribe: WebSocket not connected. Attempting to connect first.")
             await self.connect()
             if not self.is_connected():
                  raise ConnectionError("Failed to connect before subscribing.")

        new_channels = [ch for ch in channels if ch not in self.subscribed_channels]
        if not new_channels:
            logger.info(f"Already subscribed to requested channels: {channels}")
            return

        logger.info(f"Subscribing to new channels: {new_channels}")
        try:
            batch_size = 100
            for i in range(0, len(new_channels), batch_size):
                 batch = new_channels[i:i+batch_size]
                 await self._send_message({
                      "action": "subscribe",
                      "params": ",".join(batch)
                 })
                 logger.debug(f"Sent subscribe request for batch: {batch}")
                 self.subscribed_channels.update(batch)
                 await asyncio.sleep(0.1)
            logger.info(f"Successfully sent subscribe requests for {len(new_channels)} channels.")
        except Exception as e:
             logger.error(f"Failed to subscribe to channels {new_channels}: {e}")
             self.subscribed_channels.difference_update(new_channels)
             raise


    async def unsubscribe(self, channels: List[str]) -> None:
        """Unsubscribes from a list of WebSocket channel names."""
        if not self.is_connected():
            logger.warning("Cannot unsubscribe: WebSocket not connected.")
            return

        to_unsubscribe = [ch for ch in channels if ch in self.subscribed_channels]
        if not to_unsubscribe:
            logger.info(f"Not currently subscribed to requested channels for unsubscribe: {channels}")
            return

        logger.info(f"Unsubscribing from channels: {to_unsubscribe}")
        try:
            batch_size = 100
            for i in range(0, len(to_unsubscribe), batch_size):
                 batch = to_unsubscribe[i:i+batch_size]
                 await self._send_message({
                      "action": "unsubscribe",
                      "params": ",".join(batch)
                 })
                 logger.debug(f"Sent unsubscribe request for batch: {batch}")
                 self.subscribed_channels.difference_update(batch)
                 await asyncio.sleep(0.1)
            logger.info(f"Successfully sent unsubscribe requests for {len(to_unsubscribe)} channels.")
        except Exception as e:
             logger.error(f"Failed to unsubscribe from channels {to_unsubscribe}: {e}")
             raise


    async def listen(self) -> None:
        """Listens for incoming messages, puts them on the queue, and handles errors."""
        if not self.is_connected():
             logger.error("Cannot listen: WebSocket not connected.")
             return

        logger.info("WebSocket listener starting...")
        while self.running:
            try:
                message_str = await self.connection.recv()
                message_recv_time = time.time()

                try:
                    message_data = json.loads(message_str)
                    if isinstance(message_data, list):
                         for event in message_data:
                              event_type = event.get('ev', 'unknown')
                              API_WEBSOCKET_MESSAGES.labels(client="polygon", message_type=event_type).inc()
                              event['_client_recv_time'] = message_recv_time
                              try:
                                   self._receive_queue.put_nowait(event)
                              except asyncio.QueueFull:
                                   logger.warning(f"Receive queue full. Discarding message: {event_type}")
                                   API_ERROR_COUNT.labels(client="polygon", endpoint="websocket_recv_queue", method="PUT", error_type="QueueFull").inc()

                              if self.message_handler:
                                   try:
                                        # Run handler in background to avoid blocking listener
                                        asyncio.create_task(self.message_handler(event))
                                   except Exception as handler_exc:
                                        logger.error(f"Error in message handler callback: {handler_exc}", exc_info=True)
                    else:
                         logger.warning(f"Received non-list message: {message_data}")

                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON message: {message_str[:200]}...")
                    API_ERROR_COUNT.labels(client="polygon", endpoint="websocket_recv", method="RECV", error_type="JSONDecodeError").inc()
                except Exception as proc_err:
                     logger.exception(f"Error processing received message internally: {proc_err}")
                     API_ERROR_COUNT.labels(client="polygon", endpoint="websocket_recv_proc", method="RECV", error_type=type(proc_err).__name__).inc()


            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed unexpectedly: {e}. Attempting reconnect...")
                await self._handle_connection_error(e)
                if self.running:
                     await self.connect() # Try to re-establish connection
                else:
                     logger.info("Stopping listener loop due to connection closure and max retries.")
                     break
            except WebSocketException as e:
                 logger.error(f"WebSocket error during receive: {e}")
                 API_ERROR_COUNT.labels(client="polygon", endpoint="websocket_recv", method="RECV", error_type=type(e).__name__).inc()
                 await asyncio.sleep(1)
            except Exception as e:
                logger.exception(f"Unexpected error in WebSocket listen loop: {e}")
                API_ERROR_COUNT.labels(client="polygon", endpoint="websocket_listen_loop", method="RECV", error_type=type(e).__name__).inc()
                await asyncio.sleep(5)

        logger.info("WebSocket listener loop has stopped.")


    async def receive(self) -> Dict[str, Any]:
        """Retrieves the next message from the internal queue."""
        return await self._receive_queue.get()


    # _process_message_buffer removed
    # _process_messages_gpu removed


    async def close(self) -> None:
        """Cleanly close the WebSocket connection and associated tasks."""
        logger.info("Closing Polygon WebSocket client...")
        self.running = False

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await asyncio.wait_for(self._listen_task, timeout=2.0)
            except asyncio.CancelledError:
                logger.info("Listener task cancelled.")
            except asyncio.TimeoutError:
                logger.warning("Listener task did not finish cancelling within timeout.")
            except Exception as e:
                 logger.error(f"Error waiting for listener task cancellation: {e}")

        if self.connection:
            try:
                await self.connection.close()
                logger.info("WebSocket connection closed.")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            finally:
                self.connection = None
                self.connected = False

        # Thread pool removed

        # GPU memory clearing removed - handled globally

        logger.info("Polygon WebSocket client closed.")

    def is_connected(self) -> bool:
         """Returns the current connection status."""
         return self.connected and self.connection is not None and not self.connection.closed
