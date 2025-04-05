"""Tests for ai_day_trader.clients.base"""
import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock

# Import decorators and error class from the module under test
from ai_day_trader.clients import base as clients_base

# Conditionally import real implementations for type checking and state comparison
try:
    import pybreaker
    from asyncio_throttle import Throttler
    CircuitBreakerError = pybreaker.CircuitBreakerError
    PYBREAKER_INSTALLED = True
    ASYNCIO_THROTTLE_INSTALLED = True
except ImportError:
    CircuitBreakerError = clients_base.CircuitBreakerError # Use the placeholder error
    pybreaker = None
    Throttler = None
    PYBREAKER_INSTALLED = False
    ASYNCIO_THROTTLE_INSTALLED = False


# --- Circuit Breaker Tests ---

# Helper async function for testing
async def success_func():
    await asyncio.sleep(0.01)
    return "Success"

async def fail_func():
    await asyncio.sleep(0.01)
    raise ValueError("Simulated failure")

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.PYBREAKER_AVAILABLE', True) # Assume library is available
async def test_circuit_breaker_decorator_closed_state(mocker):
    """Test circuit breaker decorator allows calls in closed state."""
    if not PYBREAKER_INSTALLED: pytest.skip("pybreaker not installed")

    # Create a fresh breaker instance for the test
    test_cb = pybreaker.CircuitBreaker(fail_max=2, reset_timeout=1, name="test_closed")
    mocker.patch('ai_day_trader.clients.base.API_CIRCUIT_BREAKER', test_cb) # Patch the global instance

    protected_func = clients_base.circuit_breaker_decorator(success_func)

    assert await protected_func() == "Success"
    assert await protected_func() == "Success"
    assert test_cb.current_state == pybreaker.STATE_CLOSED

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.PYBREAKER_AVAILABLE', True)
async def test_circuit_breaker_decorator_opens_on_failures(mocker):
    """Test circuit breaker decorator opens after threshold failures."""
    if not PYBREAKER_INSTALLED: pytest.skip("pybreaker not installed")

    test_cb = pybreaker.CircuitBreaker(fail_max=2, reset_timeout=10, name="test_opens")
    mocker.patch('ai_day_trader.clients.base.API_CIRCUIT_BREAKER', test_cb)

    protected_func = clients_base.circuit_breaker_decorator(fail_func)

    with pytest.raises(ValueError): await protected_func()
    with pytest.raises(ValueError): await protected_func()
    assert test_cb.current_state == pybreaker.STATE_OPEN

    # Subsequent call should fail fast with CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        await protected_func()

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.PYBREAKER_AVAILABLE', True)
async def test_circuit_breaker_decorator_half_open_and_reset(mocker):
    """Test circuit breaker half-open state and reset on success."""
    if not PYBREAKER_INSTALLED: pytest.skip("pybreaker not installed")

    test_cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=0.1, name="test_half_open")
    mocker.patch('ai_day_trader.clients.base.API_CIRCUIT_BREAKER', test_cb)

    protected_fail = clients_base.circuit_breaker_decorator(fail_func)
    protected_success = clients_base.circuit_breaker_decorator(success_func)

    # Trip the breaker
    with pytest.raises(ValueError): await protected_fail()
    assert test_cb.current_state == pybreaker.STATE_OPEN

    # Wait for reset timeout
    await asyncio.sleep(0.15)
    assert test_cb.current_state == pybreaker.STATE_HALF_OPEN

    # Call success function - should reset the breaker
    assert await protected_success() == "Success"
    assert test_cb.current_state == pybreaker.STATE_CLOSED

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.PYBREAKER_AVAILABLE', True)
async def test_circuit_breaker_decorator_half_open_fail(mocker):
    """Test circuit breaker re-opens if call fails in half-open state."""
    if not PYBREAKER_INSTALLED: pytest.skip("pybreaker not installed")

    test_cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=0.1, name="test_half_fail")
    mocker.patch('ai_day_trader.clients.base.API_CIRCUIT_BREAKER', test_cb)

    protected_fail = clients_base.circuit_breaker_decorator(fail_func)

    # Trip the breaker
    with pytest.raises(ValueError): await protected_fail()
    assert test_cb.current_state == pybreaker.STATE_OPEN

    # Wait for reset timeout
    await asyncio.sleep(0.15)
    assert test_cb.current_state == pybreaker.STATE_HALF_OPEN

    # Call failing function again - should re-open the breaker
    with pytest.raises(ValueError): await protected_fail()
    assert test_cb.current_state == pybreaker.STATE_OPEN

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.PYBREAKER_AVAILABLE', False) # Test dummy decorator
async def test_circuit_breaker_decorator_disabled(mocker):
    """Test dummy circuit breaker decorator when library is unavailable."""
    # No need to patch API_CIRCUIT_BREAKER as it will be None

    protected_success = clients_base.circuit_breaker_decorator(success_func)
    protected_fail = clients_base.circuit_breaker_decorator(fail_func)

    # Calls should pass through without error (other than the original one)
    assert await protected_success() == "Success"
    with pytest.raises(ValueError): await protected_fail()


# --- Rate Limiter Tests ---

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.ASYNCIO_THROTTLE_AVAILABLE', True)
async def test_rate_limiter_decorator_allows_initial_calls(mocker):
    """Test rate limiter decorator allows calls within the rate."""
    if not ASYNCIO_THROTTLE_INSTALLED: pytest.skip("asyncio-throttle not installed")

    test_limiter = Throttler(rate_limit=3, period=1.0)
    mocker.patch('ai_day_trader.clients.base.API_RATE_LIMITER', test_limiter)

    call_count = 0
    @clients_base.rate_limited
    async def limited_func():
        nonlocal call_count
        call_count += 1

    tasks = [limited_func() for _ in range(3)]
    await asyncio.gather(*tasks)
    assert call_count == 3

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.ASYNCIO_THROTTLE_AVAILABLE', True)
async def test_rate_limiter_decorator_throttles_calls(mocker):
    """Test rate limiter decorator throttles calls exceeding the rate."""
    if not ASYNCIO_THROTTLE_INSTALLED: pytest.skip("asyncio-throttle not installed")

    test_limiter = Throttler(rate_limit=2, period=1.0) # 2 calls per second
    mocker.patch('ai_day_trader.clients.base.API_RATE_LIMITER', test_limiter)

    call_count = 0
    start_time = time.monotonic()

    @clients_base.rate_limited
    async def limited_func():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01) # Simulate some work

    tasks = [limited_func() for _ in range(4)] # Try 4 calls
    await asyncio.gather(*tasks)
    end_time = time.monotonic()

    assert call_count == 4
    # Expect execution time to be roughly > 1 second due to throttling
    # (2 calls immediately, wait ~1s, 2 more calls)
    assert end_time - start_time >= 1.0

@pytest.mark.asyncio
@patch('ai_day_trader.clients.base.ASYNCIO_THROTTLE_AVAILABLE', False) # Test dummy decorator
async def test_rate_limiter_decorator_disabled(mocker):
    """Test dummy rate limiter decorator when library is unavailable."""
    # No need to patch API_RATE_LIMITER as it will be None

    call_count = 0
    start_time = time.monotonic()

    @clients_base.rate_limited
    async def limited_func():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)

    tasks = [limited_func() for _ in range(10)] # Try many calls
    await asyncio.gather(*tasks)
    end_time = time.monotonic()

    assert call_count == 10
    # Should execute quickly as there's no throttling
    assert end_time - start_time < 0.5
