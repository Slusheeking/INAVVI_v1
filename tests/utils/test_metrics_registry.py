"""Tests for ai_day_trader.utils.metrics_registry"""
import pytest
import asyncio # Add missing import
from unittest.mock import patch, MagicMock, AsyncMock
import time

# Import the module to test
from ai_day_trader.utils import metrics_registry

# Conditionally import prometheus client for type checking if available
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry
    PROMETHEUS_INSTALLED = True
except ImportError:
    PROMETHEUS_INSTALLED = False
    # Define dummy types if not installed for type hints
    Counter = metrics_registry.DummyMetric
    Gauge = metrics_registry.DummyMetric
    Histogram = metrics_registry.DummyMetric
    Summary = metrics_registry.DummyMetric
    CollectorRegistry = object

# Fixture to reset the internal registry before each test
@pytest.fixture(autouse=True)
def reset_registry():
    """Clear the internal metrics registry before each test."""
    original_registry = metrics_registry._metrics_registry.copy()
    metrics_registry._metrics_registry.clear()
    yield
    # Restore original registry state if needed, though clearing is usually sufficient
    metrics_registry._metrics_registry = original_registry

# --- Test Metric Registration ---

@pytest.mark.parametrize("prometheus_available", [True, False])
def test_register_counter(prometheus_available):
    """Test registering a counter with and without prometheus."""
    with patch('ai_day_trader.utils.metrics_registry.PROMETHEUS_AVAILABLE', prometheus_available), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', prometheus_available): # Ensure METRICS_ENABLED reflects availability

        # Use a specific registry for isolation if prometheus is available
        reg = CollectorRegistry() if prometheus_available and PROMETHEUS_INSTALLED else None

        counter1 = metrics_registry.register_counter(
            "test", "counter1", "Desc 1", ["l1"], registry=reg
        )
        metric_name = "test_counter1"
        assert metric_name in metrics_registry._metrics_registry
        assert metrics_registry._metrics_registry[metric_name] == counter1

        if prometheus_available and PROMETHEUS_INSTALLED:
            assert isinstance(counter1, Counter)
            # Check if it's in the provided registry
            assert reg.get_sample_value(metric_name, labels={"l1": "v1"}) is None # Check initial value indirectly
        else:
            assert isinstance(counter1, metrics_registry.DummyMetric)

        # Test registering the same metric again returns the same instance
        counter2 = metrics_registry.register_counter("test", "counter1", "Desc 1", ["l1"], registry=reg)
        assert counter1 is counter2

@pytest.mark.parametrize("prometheus_available", [True, False])
def test_register_gauge(prometheus_available):
    """Test registering a gauge."""
    with patch('ai_day_trader.utils.metrics_registry.PROMETHEUS_AVAILABLE', prometheus_available), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', prometheus_available):
        reg = CollectorRegistry() if prometheus_available and PROMETHEUS_INSTALLED else None
        gauge = metrics_registry.register_gauge("test", "gauge1", "Desc G", registry=reg)
        metric_name = "test_gauge1"
        assert metric_name in metrics_registry._metrics_registry
        if prometheus_available and PROMETHEUS_INSTALLED:
            assert isinstance(gauge, Gauge)
        else:
            assert isinstance(gauge, metrics_registry.DummyMetric)
        assert metrics_registry.register_gauge("test", "gauge1", "Desc G", registry=reg) is gauge

@pytest.mark.parametrize("prometheus_available", [True, False])
def test_register_histogram(prometheus_available):
    """Test registering a histogram."""
    with patch('ai_day_trader.utils.metrics_registry.PROMETHEUS_AVAILABLE', prometheus_available), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', prometheus_available):
        reg = CollectorRegistry() if prometheus_available and PROMETHEUS_INSTALLED else None
        hist = metrics_registry.register_histogram("test", "hist1", "Desc H", buckets=(1, 5, 10), registry=reg)
        metric_name = "test_hist1"
        assert metric_name in metrics_registry._metrics_registry
        if prometheus_available and PROMETHEUS_INSTALLED:
            assert isinstance(hist, Histogram)
            # Check buckets were set (indirectly)
            assert hasattr(hist, '_upper_bounds')
            assert hist._upper_bounds == (1.0, 5.0, 10.0, float('inf'))
        else:
            assert isinstance(hist, metrics_registry.DummyMetric)
        assert metrics_registry.register_histogram("test", "hist1", "Desc H", registry=reg) is hist

@pytest.mark.parametrize("prometheus_available", [True, False])
def test_register_summary(prometheus_available):
    """Test registering a summary."""
    with patch('ai_day_trader.utils.metrics_registry.PROMETHEUS_AVAILABLE', prometheus_available), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', prometheus_available):
        reg = CollectorRegistry() if prometheus_available and PROMETHEUS_INSTALLED else None
        summary = metrics_registry.register_summary("test", "summary1", "Desc S", registry=reg)
        metric_name = "test_summary1"
        assert metric_name in metrics_registry._metrics_registry
        if prometheus_available and PROMETHEUS_INSTALLED:
            assert isinstance(summary, Summary)
        else:
            assert isinstance(summary, metrics_registry.DummyMetric)
        assert metrics_registry.register_summary("test", "summary1", "Desc S", registry=reg) is summary

# --- Test Decorators ---

@pytest.mark.asyncio
async def test_time_function_decorator():
    """Test the time_function decorator runs."""
    mock_hist = MagicMock(spec=Histogram)
    mock_hist.labels.return_value = mock_hist # Mock chaining

    @metrics_registry.time_function(metric=mock_hist, labels={"op": "test"})
    async def sample_async_func():
        await asyncio.sleep(0.01)
        return "done"

    # Test with metrics enabled
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True):
        result = await sample_async_func()
        assert result == "done"
        mock_hist.labels.assert_called_once_with(op="test")
        mock_hist.observe.assert_called_once()
        assert mock_hist.observe.call_args[0][0] > 0 # Check duration > 0

    # Test with metrics disabled
    mock_hist.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', False):
        result = await sample_async_func()
        assert result == "done"
        mock_hist.labels.assert_not_called()
        mock_hist.observe.assert_not_called()


@pytest.mark.asyncio
async def test_count_calls_decorator():
    """Test the count_calls decorator runs."""
    mock_counter = MagicMock(spec=Counter)
    mock_counter.labels.return_value = mock_counter

    @metrics_registry.count_calls(metric=mock_counter, labels={"call": "test"})
    async def sample_async_func(fail=False):
        if fail:
            raise ValueError("Test failure")
        return "ok"

    # Test success call with metrics enabled
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True):
        await sample_async_func()
        mock_counter.labels.assert_called_once_with(call="test")
        mock_counter.inc.assert_called_once()

    # Test failure call with metrics enabled (counting exceptions)
    mock_counter.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True):
         with pytest.raises(ValueError):
              await sample_async_func(fail=True)
         # Should still be called once for the exception
         mock_counter.labels.assert_called_once() # Labels might include error type now
         mock_counter.inc.assert_called_once()

    # Test success call with metrics disabled
    mock_counter.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', False):
        await sample_async_func()
        mock_counter.labels.assert_not_called()
        mock_counter.inc.assert_not_called()

@pytest.mark.asyncio
async def test_track_in_progress_decorator():
    """Test the track_in_progress decorator runs."""
    mock_gauge = MagicMock(spec=Gauge)
    mock_gauge.labels.return_value = mock_gauge

    @metrics_registry.track_in_progress(metric=mock_gauge, labels={"task": "test"})
    async def sample_async_func():
        await asyncio.sleep(0.01)
        return "ok"

    # Test with metrics enabled
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True):
        await sample_async_func()
        mock_gauge.labels.assert_called_with(task="test")
        mock_gauge.inc.assert_called_once()
        mock_gauge.dec.assert_called_once()

    # Test with metrics disabled
    mock_gauge.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', False):
        await sample_async_func()
        mock_gauge.labels.assert_not_called()
        mock_gauge.inc.assert_not_called()
        mock_gauge.dec.assert_not_called()


# --- Test Server/Gateway Functions ---

@patch('ai_day_trader.utils.metrics_registry.prometheus_push_to_gateway')
def test_push_metrics_to_gateway(mock_push):
    """Test pushing metrics to gateway."""
    # Test when enabled
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_PUSH_GATEWAY', 'http://fakegateway:9091'):
        result = metrics_registry.push_metrics_to_gateway(job="testjob")
        assert result is True
        mock_push.assert_called_once()
        # Check specific args if needed, e.g., mock_push.assert_called_with(gateway='http://fakegateway:9091', job='testjob', ...)

    # Test when disabled
    mock_push.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', False):
        result = metrics_registry.push_metrics_to_gateway()
        assert result is False
        mock_push.assert_not_called()

    # Test when enabled but no gateway configured
    mock_push.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_PUSH_GATEWAY', ''): # No gateway URL
        result = metrics_registry.push_metrics_to_gateway()
        assert result is False # Should return False if no gateway
        mock_push.assert_not_called()


@patch('ai_day_trader.utils.metrics_registry.prometheus_start_http_server')
@patch('ai_day_trader.utils.metrics_registry._metrics_server_started', False) # Ensure server starts as not started
def test_register_metrics_server_if_needed(mock_start_server):
    """Test starting the metrics server."""
    # Reset the global flag for safety between tests if needed, though patch should handle it
    metrics_registry._metrics_server_started = False

    # Test when enabled
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_SERVER_ENABLED', True):
        result = metrics_registry.register_metrics_server_if_needed(port=9999)
        assert result is True
        mock_start_server.assert_called_once_with(9999)
        assert metrics_registry._metrics_server_started is True # Check flag

        # Test calling again when already started
        mock_start_server.reset_mock()
        result2 = metrics_registry.register_metrics_server_if_needed()
        assert result2 is True
        mock_start_server.assert_not_called() # Should not start again

    # Test when disabled
    metrics_registry._metrics_server_started = False # Reset flag
    mock_start_server.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', False):
        result = metrics_registry.register_metrics_server_if_needed()
        assert result is False
        mock_start_server.assert_not_called()

    # Test when server specifically disabled
    metrics_registry._metrics_server_started = False # Reset flag
    mock_start_server.reset_mock()
    with patch('ai_day_trader.utils.metrics_registry.METRICS_ENABLED', True), \
         patch('ai_day_trader.utils.metrics_registry.METRICS_SERVER_ENABLED', False):
        result = metrics_registry.register_metrics_server_if_needed()
        assert result is False
        mock_start_server.assert_not_called()
