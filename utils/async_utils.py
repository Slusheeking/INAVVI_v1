#!/usr/bin/env python3
"""
Async Utilities Module

This module provides standardized async utilities for the trading system:

1. Common retry patterns with exponential backoff
2. Timeout handling with graceful cancellation
3. Concurrent task management
4. Rate limiting for API calls
5. Async context managers for resource management

All components in the trading system should use these utilities for async operations
to ensure consistent behavior and error handling.
"""

import asyncio
import functools
import inspect
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

# Import exceptions module if available
try:
    from exceptions import (
        TradingSystemError,
        APIError,
        APIRateLimitError,
        APITimeoutError,
        NetworkError,
        TimeoutError,
        is_retryable_exception
    )
except ImportError:
    # Fallback if exceptions module is not available
    class TradingSystemError(Exception):
        """Base exception class for all trading system errors"""
        pass
    
    class APIError(TradingSystemError):
        """Base class for API-related errors"""
        pass
    
    class APIRateLimitError(APIError):
        """API rate limit exceeded"""
        pass
    
    class APITimeoutError(APIError):
        """API request timed out"""
        pass
    
    class NetworkError(TradingSystemError):
        """Base class for network-related errors"""
        pass
    
    class TimeoutError(NetworkError):
        """Connection or operation timed out"""
        pass
    
    def is_retryable_exception(exc: Exception) -> bool:
        """Check if an exception is retryable"""
        error_msg = str(exc).lower()
        retryable_patterns = [
            "timeout", "connection reset", "connection refused",
            "temporarily unavailable", "retry", "try again",
            "too many requests", "service unavailable"
        ]
        return any(pattern in error_msg for pattern in retryable_patterns)

# Configure logging
logger = logging.getLogger("async_utils")

# Type variable for generic type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class RateLimiter:
    """
    Rate limiter for async operations
    
    This class implements a token bucket algorithm for rate limiting
    async operations, such as API calls.
    """
    
    def __init__(
        self,
        rate: float,
        burst: Optional[int] = None,
        *,
        per_second: bool = True
    ) -> None:
        """
        Initialize rate limiter
        
        Args:
            rate: Maximum rate (tokens per second or minute)
            burst: Maximum burst size (defaults to rate)
            per_second: Whether rate is per second (True) or per minute (False)
        """
        self.rate = rate
        self.burst = burst if burst is not None else rate
        self.per_second = per_second
        self.tokens = self.burst
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquire tokens from the bucket
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Delay in seconds before the operation should proceed
            
        Raises:
            APIRateLimitError: If tokens cannot be acquired
        """
        async with self.lock:
            # Refill tokens
            now = time.monotonic()
            elapsed = now - self.last_refill
            
            # Calculate refill rate based on per_second setting
            refill_rate = self.rate if self.per_second else self.rate / 60.0
            
            # Refill tokens
            self.tokens = min(self.burst, self.tokens + elapsed * refill_rate)
            self.last_refill = now
            
            # Check if we have enough tokens
            if tokens > self.burst:
                raise APIRateLimitError(
                    f"Requested tokens ({tokens}) exceed burst size ({self.burst})"
                )
            
            # Calculate delay if we don't have enough tokens
            if tokens > self.tokens:
                # Calculate how long to wait for enough tokens
                delay = (tokens - self.tokens) / refill_rate
                
                # If delay is too long, raise an error
                if delay > 60.0:  # Don't wait more than 60 seconds
                    raise APIRateLimitError(
                        f"Rate limit exceeded, would need to wait {delay:.2f}s"
                    )
                
                # Update tokens and last refill time
                self.tokens = 0
                self.last_refill = now + delay
                
                return delay
            
            # We have enough tokens, consume them
            self.tokens -= tokens
            return 0.0
    
    async def __aenter__(self) -> 'RateLimiter':
        """Enter async context manager"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager"""
        pass


def async_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_exceptions: Optional[List[Type[Exception]]] = None,
    retry_on: Optional[Callable[[Exception], bool]] = None
) -> Callable[[F], F]:
    """
    Decorator for retrying async functions with exponential backoff
    
    Args:
        retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delay
        retry_exceptions: List of exception types to retry on
        retry_on: Function to determine if an exception should be retried
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay
            
            # Use default retry_on function if not provided
            nonlocal retry_on
            if retry_on is None:
                retry_on = is_retryable_exception
            
            # Use default retry_exceptions if not provided
            nonlocal retry_exceptions
            if retry_exceptions is None:
                retry_exceptions = [
                    APIError, NetworkError, asyncio.TimeoutError, ConnectionError
                ]
            
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    should_retry = False
                    
                    # Check if exception is in retry_exceptions
                    if retry_exceptions and isinstance(e, tuple(retry_exceptions)):
                        should_retry = True
                    
                    # Check if retry_on function returns True
                    if retry_on and retry_on(e):
                        should_retry = True
                    
                    # Don't retry if this is the last attempt
                    if attempt >= retries:
                        should_retry = False
                    
                    if should_retry:
                        # Calculate delay with jitter
                        if jitter:
                            actual_delay = current_delay * (0.5 + random.random())
                        else:
                            actual_delay = current_delay
                        
                        # Log retry
                        logger.warning(
                            f"Retrying {func.__name__} in {actual_delay:.2f}s "
                            f"(attempt {attempt + 1}/{retries}) after error: {e}"
                        )
                        
                        # Wait before retrying
                        await asyncio.sleep(actual_delay)
                        
                        # Increase delay for next attempt
                        current_delay = min(current_delay * backoff, max_delay)
                    else:
                        # Don't retry, re-raise exception
                        raise
            
            # This should never happen, but just in case
            if last_exception:
                raise last_exception
            
            # This should never happen either
            raise RuntimeError(f"Unexpected error in {func.__name__}")
        
        return cast(F, wrapper)
    
    return decorator

@asynccontextmanager
async def async_timeout(
    timeout: float,
    timeout_exception: Optional[Type[Exception]] = None
):
    """
    Async context manager for timeout
    
    Args:
        timeout: Timeout in seconds
        timeout_exception: Exception to raise on timeout
        
    Yields:
        None
        
    Raises:
        timeout_exception: If context times out
    """
    # Use the existing timeout_context implementation
    async with timeout_context(timeout, timeout_exception):
        yield





async def with_timeout(
    coro: asyncio.coroutine,
    timeout: float,
    timeout_exception: Optional[Type[Exception]] = None
) -> Any:
    """
    Run coroutine with timeout
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        timeout_exception: Exception to raise on timeout
        
    Returns:
        Coroutine result
        
    Raises:
        timeout_exception: If coroutine times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if timeout_exception:
            raise timeout_exception(f"Operation timed out after {timeout}s")
        else:
            raise APITimeoutError(f"Operation timed out after {timeout}s")


@asynccontextmanager
async def timeout_context(
    timeout: float,
    timeout_exception: Optional[Type[Exception]] = None
):
    """
    Async context manager for timeout
    
    Args:
        timeout: Timeout in seconds
        timeout_exception: Exception to raise on timeout
        
    Yields:
        None
        
    Raises:
        timeout_exception: If context times out
    """
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("No current task")
    
    # Create a timeout task
    async def cancel_on_timeout():
        await asyncio.sleep(timeout)
        if not task.done():
            task.cancel()
    
    # Start timeout task
    timeout_task = asyncio.create_task(cancel_on_timeout())
    
    try:
        yield
    except asyncio.CancelledError:
        if timeout_exception:
            raise timeout_exception(f"Operation timed out after {timeout}s")
        else:
            raise APITimeoutError(f"Operation timed out after {timeout}s")
    finally:
        # Cancel timeout task if it's still running
        if not timeout_task.done():
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass


class TaskGroup:
    """
    Group of async tasks with management utilities
    
    This class provides utilities for managing a group of async tasks,
    including waiting for all tasks to complete, cancelling tasks,
    and handling exceptions.
    """
    
    def __init__(self) -> None:
        """Initialize task group"""
        self.tasks: Set[asyncio.Task] = set()
        self.results: Dict[asyncio.Task, Any] = {}
        self.exceptions: Dict[asyncio.Task, Exception] = {}
    
    def create_task(self, coro: asyncio.coroutine) -> asyncio.Task:
        """
        Create a new task in the group
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Created task
        """
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        
        # Add done callback to remove task from set
        task.add_done_callback(self._task_done)
        
        return task
    
    def _task_done(self, task: asyncio.Task) -> None:
        """
        Callback for when a task is done
        
        Args:
            task: Completed task
        """
        # Remove task from set
        self.tasks.discard(task)
        
        # Store result or exception
        if task.cancelled():
            self.exceptions[task] = asyncio.CancelledError()
        elif task.exception():
            self.exceptions[task] = task.exception()
        else:
            self.results[task] = task.result()
    
    async def wait(
        self,
        timeout: Optional[float] = None,
        return_when: str = asyncio.ALL_COMPLETED
    ) -> None:
        """
        Wait for tasks to complete
        
        Args:
            timeout: Timeout in seconds
            return_when: When to return (ALL_COMPLETED, FIRST_COMPLETED, FIRST_EXCEPTION)
        """
        if not self.tasks:
            return
        
        try:
            await asyncio.wait(
                self.tasks,
                timeout=timeout,
                return_when=return_when
            )
        except asyncio.TimeoutError:
            # Timeout is handled by returning before all tasks are done
            pass
    
    def cancel_all(self) -> None:
        """Cancel all tasks in the group"""
        for task in self.tasks:
            task.cancel()
    
    async def cancel_and_wait(self, timeout: Optional[float] = None) -> None:
        """
        Cancel all tasks and wait for them to complete
        
        Args:
            timeout: Timeout in seconds
        """
        self.cancel_all()
        await self.wait(timeout=timeout)
    
    def get_results(self) -> Dict[asyncio.Task, Any]:
        """
        Get results of completed tasks
        
        Returns:
            Dictionary mapping tasks to results
        """
        return self.results
    
    def get_exceptions(self) -> Dict[asyncio.Task, Exception]:
        """
        Get exceptions from failed tasks
        
        Returns:
            Dictionary mapping tasks to exceptions
        """
        return self.exceptions
    
    def raise_if_any_failed(self) -> None:
        """
        Raise exception if any task failed
        
        Raises:
            Exception: First exception from failed tasks
        """
        if self.exceptions:
            # Get first exception
            task, exc = next(iter(self.exceptions.items()))
            raise exc
    
    async def __aenter__(self) -> 'TaskGroup':
        """Enter async context manager"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager"""
        # Cancel all tasks if an exception occurred
        if exc_type is not None:
            self.cancel_all()
        
        # Wait for all tasks to complete
        await self.wait()


class AsyncBatch:
    """
    Batch processor for async operations
    
    This class provides utilities for processing items in batches,
    with concurrency control and error handling.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrency: int = 5,
        timeout: Optional[float] = None
    ) -> None:
        """
        Initialize batch processor
        
        Args:
            batch_size: Maximum batch size
            max_concurrency: Maximum number of concurrent batches
            timeout: Timeout for each batch in seconds
        """
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process(
        self,
        items: List[Any],
        processor: Callable[[List[Any]], asyncio.coroutine],
        on_batch_complete: Optional[Callable[[List[Any], Any], asyncio.coroutine]] = None,
        on_batch_error: Optional[Callable[[List[Any], Exception], asyncio.coroutine]] = None
    ) -> List[Any]:
        """
        Process items in batches
        
        Args:
            items: Items to process
            processor: Function to process each batch
            on_batch_complete: Callback for when a batch completes successfully
            on_batch_error: Callback for when a batch fails
            
        Returns:
            List of results
        """
        # Split items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Create task group
        async with TaskGroup() as group:
            # Process each batch
            for batch in batches:
                group.create_task(
                    self._process_batch(
                        batch, processor, on_batch_complete, on_batch_error
                    )
                )
            
            # Wait for all batches to complete
            await group.wait()
        
        # Raise if any batch failed
        group.raise_if_any_failed()
        
        # Return results
        results = []
        for task, result in group.get_results().items():
            results.extend(result)
        
        return results
    
    async def _process_batch(
        self,
        batch: List[Any],
        processor: Callable[[List[Any]], asyncio.coroutine],
        on_batch_complete: Optional[Callable[[List[Any], Any], asyncio.coroutine]],
        on_batch_error: Optional[Callable[[List[Any], Exception], asyncio.coroutine]]
    ) -> List[Any]:
        """
        Process a single batch
        
        Args:
            batch: Batch to process
            processor: Function to process batch
            on_batch_complete: Callback for when batch completes successfully
            on_batch_error: Callback for when batch fails
            
        Returns:
            Batch results
        """
        async with self.semaphore:
            try:
                # Process batch with timeout
                if self.timeout:
                    result = await with_timeout(
                        processor(batch),
                        self.timeout,
                        APITimeoutError
                    )
                else:
                    result = await processor(batch)
                
                # Call on_batch_complete callback
                if on_batch_complete:
                    await on_batch_complete(batch, result)
                
                return result
            
            except Exception as e:
                # Call on_batch_error callback
                if on_batch_error:
                    await on_batch_error(batch, e)
                
                # Re-raise exception
                raise


class AsyncLimiter:
    """
    Concurrency limiter for async operations
    
    This class provides utilities for limiting concurrency of async operations,
    with support for both maximum concurrency and rate limiting.
    """
    
    def __init__(
        self,
        max_concurrency: int = 10,
        rate_limit: Optional[float] = None,
        per_second: bool = True
    ) -> None:
        """
        Initialize concurrency limiter
        
        Args:
            max_concurrency: Maximum number of concurrent operations
            rate_limit: Maximum rate (operations per second or minute)
            per_second: Whether rate is per second (True) or per minute (False)
        """
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = (
            RateLimiter(rate_limit, per_second=per_second)
            if rate_limit is not None else None
        )
    
    async def acquire(self) -> None:
        """
        Acquire permission to perform an operation
        
        This method acquires both the semaphore and rate limiter tokens.
        """
        # Acquire semaphore
        await self.semaphore.acquire()
        
        # Acquire rate limiter tokens
        if self.rate_limiter:
            delay = await self.rate_limiter.acquire()
            if delay > 0:
                await asyncio.sleep(delay)
    
    def release(self) -> None:
        """Release permission after operation completes"""
        self.semaphore.release()
    
    async def __aenter__(self) -> None:
        """Enter async context manager"""
        await self.acquire()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager"""
        self.release()


def rate_limited(
    rate: float,
    per_second: bool = True,
    burst: Optional[int] = None
) -> Callable[[F], F]:
    """
    Decorator for rate limiting async functions
    
    Args:
        rate: Maximum rate (operations per second or minute)
        per_second: Whether rate is per second (True) or per minute (False)
        burst: Maximum burst size (defaults to rate)
        
    Returns:
        Decorated function
    """
    # Create rate limiter
    limiter = RateLimiter(rate, burst, per_second=per_second)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Acquire rate limiter tokens
            delay = await limiter.acquire()
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Call function
            return await func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def concurrency_limited(max_concurrency: int) -> Callable[[F], F]:
    """
    Decorator for limiting concurrency of async functions
    
    Args:
        max_concurrency: Maximum number of concurrent operations
        
    Returns:
        Decorated function
    """
    # Create semaphore
    semaphore = asyncio.Semaphore(max_concurrency)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Acquire semaphore
            async with semaphore:
                return await func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


@asynccontextmanager
async def async_timed(
    name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    threshold: Optional[float] = None
) -> None:
    """
    Async context manager for timing operations
    
    Args:
        name: Operation name
        logger: Logger to use
        level: Log level
        threshold: Threshold in seconds for logging
        
    Yields:
        None
    """
    start_time = time.monotonic()
    
    try:
        yield
    finally:
        elapsed = time.monotonic() - start_time
        
        # Log if elapsed time exceeds threshold
        if threshold is None or elapsed >= threshold:
            if logger:
                logger.log(level, f"{name} took {elapsed:.6f}s")


async def gather_with_concurrency(
    max_concurrency: int,
    *tasks: asyncio.coroutine,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Run coroutines with limited concurrency
    
    Args:
        max_concurrency: Maximum number of concurrent tasks
        *tasks: Coroutines to run
        return_exceptions: Whether to return exceptions instead of raising them
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_with_semaphore(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *(run_with_semaphore(task) for task in tasks),
        return_exceptions=return_exceptions
    )


async def wait_for_event(
    event: asyncio.Event,
    timeout: Optional[float] = None,
    timeout_exception: Optional[Type[Exception]] = None
) -> None:
    """
    Wait for event with timeout
    
    Args:
        event: Event to wait for
        timeout: Timeout in seconds
        timeout_exception: Exception to raise on timeout
        
    Raises:
        timeout_exception: If event times out
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        if timeout_exception:
            raise timeout_exception(f"Event wait timed out after {timeout}s")
        else:
            raise APITimeoutError(f"Event wait timed out after {timeout}s")


class AsyncCache:
    """
    Cache for async function results
    
    This class provides a cache for async function results,
    with support for TTL and size limits.
    """
    
    def __init__(
        self,
        ttl: float = 60.0,
        max_size: int = 1000,
        cleanup_interval: float = 60.0
    ) -> None:
        """
        Initialize async cache
        
        Args:
            ttl: Time-to-live in seconds
            max_size: Maximum cache size
            cleanup_interval: Cleanup interval in seconds
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        self.last_cleanup = time.monotonic()
    
    async def get(self, key: str) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        async with self.lock:
            # Check if key exists and is not expired
            if key in self.cache:
                entry = self.cache[key]
                if entry["expires"] > time.monotonic():
                    return entry["value"]
                
                # Remove expired entry
                del self.cache[key]
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (defaults to self.ttl)
        """
        async with self.lock:
            # Clean up if needed
            await self._cleanup()
            
            # Set value
            self.cache[key] = {
                "value": value,
                "expires": time.monotonic() + (ttl or self.ttl)
            }
    
    async def delete(self, key: str) -> None:
        """
        Delete value from cache
        
        Args:
            key: Cache key
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    async def clear(self) -> None:
        """Clear cache"""
        async with self.lock:
            self.cache.clear()
    
    async def _cleanup(self) -> None:
        """Clean up expired entries and enforce size limit"""
        now = time.monotonic()
        
        # Only clean up periodically
        if now - self.last_cleanup < self.cleanup_interval:
            # But still enforce size limit
            if len(self.cache) > self.max_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]["expires"]
                )
                for key in sorted_keys[:len(self.cache) - self.max_size]:
                    del self.cache[key]
            
            return
        
        # Clean up expired entries
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry["expires"] <= now
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # Enforce size limit
        if len(self.cache) > self.max_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]["expires"]
            )
            for key in sorted_keys[:len(self.cache) - self.max_size]:
                del self.cache[key]
        
        self.last_cleanup = now


def cached(
    ttl: float = 60.0,
    key_fn: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """
    Decorator for caching async function results
    
    Args:
        ttl: Time-to-live in seconds
        key_fn: Function to generate cache key from arguments
        
    Returns:
        Decorated function
    """
    # Create cache
    cache = AsyncCache(ttl=ttl)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default key is function name + args + kwargs
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = ":".join(key_parts)
            
            # Check cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                return cached_value
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(key, result)
            
            return result
        
        return cast(F, wrapper)
    
    return decorator