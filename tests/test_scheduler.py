"""Tests for ai_day_trader.scheduler"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import date, time, datetime
import pytz
from ai_day_trader.scheduler import CentralScheduler
from ai_day_trader.config import load_ai_trader_config

# Mock Alpaca calendar entry
class MockCalendarDay:
    def __init__(self, open_time, close_time):
        self.date = date.today()
        self.open = open_time
        self.close = close_time

@pytest.fixture
def mock_alpaca_client():
    client = MagicMock()
    # Mock get_calendar to return a valid entry for today
    open_time = time(9, 30)
    close_time = time(16, 0)
    client.get_calendar = MagicMock(return_value=[MockCalendarDay(open_time, close_time)])
    return client

@pytest.mark.asyncio
async def test_scheduler_init(mock_alpaca_client):
    """Test scheduler initialization."""
    config = load_ai_trader_config()
    scheduler = CentralScheduler(config, mock_alpaca_client)
    assert scheduler is not None
    assert scheduler.scheduler is not None
    await scheduler.stop() # Stop the background task

@pytest.mark.asyncio
async def test_schedule_daily_tasks(mock_alpaca_client):
    """Test scheduling of daily tasks."""
    config = load_ai_trader_config()
    scheduler = CentralScheduler(config, mock_alpaca_client)
    # Define a simple async function to pass to the scheduler
    async def mock_task_func():
        pass
    tasks = {"pre_market": mock_task_func, "refresh_symbols": mock_task_func}
    await scheduler.schedule_daily_tasks(tasks)
    assert "pre_market_task" in scheduler._jobs
    assert "refresh_symbols_daily" in scheduler._jobs
    await scheduler.stop()

# TODO: Add tests for job execution at correct times (requires time mocking or longer running tests)
