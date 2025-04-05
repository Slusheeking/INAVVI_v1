"""Circuit breaker functionality for execution operations."""
import time
import asyncio
from typing import Dict, Optional, List, Any, Callable, Awaitable, TypeVar

from ai_day_trader.utils.exceptions import CircuitBreakerError # Use new utils path
from ai_day_trader.utils.metrics_registry import register_counter, register_gauge, MetricPrefix # Use new utils path
from ai_day_trader.utils.logging_config import get_logger # Use new utils path

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Define circuit breaker metrics using the new metrics system
EXECUTION_CIRCUIT_BREAKER_STATE = register_gauge(
    MetricPrefix.TRADING,
    "execution_circuit_breaker_state",
    "Execution circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["broker", "operation"]
)

EXECUTION_CIRCUIT_BREAKER_FAILURE_COUNT = register_counter(
    MetricPrefix.TRADING,
    "execution_circuit_breaker_failure_count",
    "Execution circuit breaker failure count",
    ["broker", "operation"]
)

EXECUTION_CIRCUIT_BREAKER_SUCCESS_COUNT = register_counter(
    MetricPrefix.TRADING,
    "execution_circuit_breaker_success_count",
    "Execution circuit breaker success count",
    ["broker", "operation"]
)

class CircuitBreakerState:
    """Enum-like class for circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation, requests pass through
    OPEN = "OPEN"      # Circuit is open, requests fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing if the service is back online

class ExecutionCircuitBreaker:
    """Circuit breaker for execution operations to prevent cascading failures."""
    
    def __init__(
        self,
        name: str,
        broker: str,
        failure_threshold: int = 3,
        reset_timeout: int = 60,
        half_open_max_calls: int = 1,
        exclude_exceptions: Optional[List[type]] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the operation (e.g., "execute_order", "cancel_order")
            broker: Name of the broker (e.g., "alpaca", "paper")
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Seconds to wait before attempting to close the circuit
            half_open_max_calls: Maximum number of calls in half-open state
            exclude_exceptions: List of exception types that should not count as failures
        """
        self.name = name
        self.broker = broker
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exclude_exceptions = exclude_exceptions or []
        self.logger = get_logger(f"circuit_breaker_{name}")
        
        # State
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0
        self._consecutive_successes = 0
        
        # Update metrics
        self._update_state_metric()
        
    def _update_state_metric(self):
        """Update the circuit breaker state metric."""
        # Update state metric with numeric value for visualization
        state_value = 0  # CLOSED
        if self._state == CircuitBreakerState.HALF_OPEN:
            state_value = 1
        elif self._state == CircuitBreakerState.OPEN:
            state_value = 2
        EXECUTION_CIRCUIT_BREAKER_STATE.labels(broker=self.broker, operation=self.name).set(state_value)
        
    async def execute(self, func: Callable[..., Awaitable[R]], *args: Any, **kwargs: Any) -> R:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
            
        Raises:
            CircuitBreakerError: If the circuit is open
            Exception: Any exception raised by the function
        """
        # Check if circuit is open
        if self._state == CircuitBreakerState.OPEN:
            # Check if it's time to attempt reset
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                self.logger.info(f"Circuit breaker {self.name} attempting reset (HALF_OPEN)")
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_calls = 0
                self._consecutive_successes = 0
                self._update_state_metric()
            else:
                self.logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerError(f"Circuit breaker {self.name} is open, retry after {self.reset_timeout - elapsed:.1f}s")
                
        # If half-open, check if we've reached the max calls
        if self._state == CircuitBreakerState.HALF_OPEN and self._half_open_calls >= self.half_open_max_calls:
            self.logger.warning(f"Circuit breaker {self.name} is HALF_OPEN with max calls reached, failing fast")
            raise CircuitBreakerError(f"Circuit breaker {self.name} is half-open with max calls reached")
            
        # Increment half-open calls if in half-open state
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._half_open_calls += 1
            
        start_time = time.time()
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Track consecutive successes
            # Track success metrics
            self._consecutive_successes += 1
            EXECUTION_CIRCUIT_BREAKER_SUCCESS_COUNT.labels(broker=self.broker, operation=self.name).inc()
            self.logger.debug(f"Circuit breaker {self.name} operation succeeded")
            
            # If we have enough consecutive successes in half-open state, reset the circuit
            if self._state == CircuitBreakerState.HALF_OPEN and self._consecutive_successes >= 3:
                self.logger.info(f"Circuit breaker {self.name} reset to CLOSED state")
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._consecutive_successes = 0
                self._update_state_metric()
            
            return result
        except Exception as e:
            # Check if this exception should be excluded
            if any(isinstance(e, exc_type) for exc_type in self.exclude_exceptions):
                self.logger.debug(f"Circuit breaker {self.name} ignoring excluded exception: {type(e).__name__}")
                raise
                
            # Increment failure count
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._consecutive_successes = 0
            
            # Track failure metrics
            EXECUTION_CIRCUIT_BREAKER_FAILURE_COUNT.labels(broker=self.broker, operation=self.name).inc()
            self.logger.debug(f"Circuit breaker {self.name} operation failed: {type(e).__name__}")
            
            # Check if we should trip the circuit
            if self._state == CircuitBreakerState.CLOSED and self._failure_count >= self.failure_threshold:
                self.logger.warning(f"Circuit breaker {self.name} TRIPPED after {self._failure_count} failures")
                self._state = CircuitBreakerState.OPEN
                self._update_state_metric()
                
            # Re-raise the exception
            raise

class ExecutionCircuitBreakerRegistry:
    """Registry for execution circuit breakers."""
    
    def __init__(self):
        """Initialize the registry."""
        self.circuit_breakers: Dict[str, ExecutionCircuitBreaker] = {}
        self.logger = get_logger("execution_circuit_breaker_registry")
        
    def get_or_create(
        self,
        name: str,
        broker: str,
        failure_threshold: int = 3,
        reset_timeout: int = 60,
        half_open_max_calls: int = 1,
        exclude_exceptions: Optional[List[type]] = None
    ) -> ExecutionCircuitBreaker:
        """
        Get or create a circuit breaker.
        
        Args:
            name: Name of the operation
            broker: Name of the broker
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Seconds to wait before attempting to close the circuit
            half_open_max_calls: Maximum number of calls in half-open state
            exclude_exceptions: List of exception types that should not count as failures
            
        Returns:
            Circuit breaker instance
        """
        key = f"{broker}:{name}"
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = ExecutionCircuitBreaker(
                name=name,
                broker=broker,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                half_open_max_calls=half_open_max_calls,
                exclude_exceptions=exclude_exceptions
            )
        return self.circuit_breakers[key]

# Global registry instance
execution_circuit_breaker_registry = ExecutionCircuitBreakerRegistry()
