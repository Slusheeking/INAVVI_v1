"""
Base utilities for API clients, including decorators.
"""

import functools
import time
import asyncio
from typing import Optional, Callable, Any # Added missing imports
from ai_day_trader.utils.logging_config import get_logger # Use new utils path
# Import actual implementations
try:
    import pybreaker
    from asyncio_throttle import Throttler
    PYBREAKER_AVAILABLE = True
    ASYNCIO_THROTTLE_AVAILABLE = True
    # Use the specific error from pybreaker
    from pybreaker import CircuitBreakerError
except ImportError:
    PYBREAKER_AVAILABLE = False
    ASYNCIO_THROTTLE_AVAILABLE = False
    # Define a placeholder error if pybreaker is not installed
    class CircuitBreakerError(Exception): pass


logger = get_logger("ai_day_trader.clients.base")

# --- Circuit Breaker Implementation ---
if PYBREAKER_AVAILABLE:
    # Configure pybreaker listener for logging state changes
    class LoggingListener(pybreaker.CircuitBreakerListener):
        def state_changed(self, cb, old_state, new_state):
            logger.info(f"Circuit breaker '{cb.name}' state changed from {old_state.name} to {new_state.name}")
        def failure(self, cb, exc):
             logger.warning(f"Circuit breaker '{cb.name}' recorded failure: {exc}. Fail count: {cb.fail_counter}")
        def success(self, cb):
             if cb.current_state == pybreaker.STATE_HALF_OPEN:
                 logger.info(f"Circuit breaker '{cb.name}' succeeded in half-open state, resetting to CLOSED.")

    # Instance configured with defaults (can be overridden by config)
    # fail_max=5: Open after 5 failures
    # reset_timeout=30: Try to close after 30 seconds
    API_CIRCUIT_BREAKER = pybreaker.CircuitBreaker(
        fail_max=5,
        reset_timeout=30,
        listeners=[LoggingListener()],
        name="API_CircuitBreaker" # Give it a name for logging
    )
    # Decorator for async functions
    circuit_breaker_decorator = API_CIRCUIT_BREAKER # pybreaker handles async via the instance itself
else:
    logger.warning("pybreaker not installed. Circuit breaker functionality is disabled.")
    # Dummy decorator if pybreaker is not available
    def dummy_circuit_breaker_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    circuit_breaker_decorator = dummy_circuit_breaker_decorator
    API_CIRCUIT_BREAKER = None # No instance available


# --- Rate Limiter Implementation ---
if ASYNCIO_THROTTLE_AVAILABLE:
    # Instance configured with defaults (can be overridden by config)
    # Example: 5 calls per second
    API_RATE_LIMITER = Throttler(rate_limit=5, period=1.0)

    # Decorator using the throttler instance
    def rate_limited(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with API_RATE_LIMITER:
                logger.debug(f"Rate limiter acquired for {func.__name__}")
                return await func(*args, **kwargs)
        return wrapper
else:
    logger.warning("asyncio-throttle not installed. Rate limiting functionality is disabled.")
    # Dummy decorator if asyncio-throttle is not available
    def rate_limited(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    API_RATE_LIMITER = None # No instance available
