"""
Centralized Scheduler for the AI Day Trader.

Manages scheduled tasks based on market hours, holidays, and specific times.
Uses APScheduler and Alpaca market calendar/clock.
"""

import logging
import asyncio
from datetime import datetime, time, timedelta, date, timezone # Added timezone
from typing import Callable, Optional, Any, List, Dict # Added Dict
import pytz

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

# Assuming Alpaca client is available and configured
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError as AlpacaAPIError
    from alpaca_trade_api.common import URL
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None
    AlpacaAPIError = Exception # Placeholder

from ai_day_trader.utils.config import Config # Use new utils path

logger = logging.getLogger(__name__)

# Define timezones
NY_TZ = pytz.timezone("America/New_York")

class CentralScheduler:
    """Manages and runs scheduled tasks for the trading bot."""

    def __init__(self, config: Config, alpaca_client: Optional[Any]):
        self.config = config
        self.alpaca_client = alpaca_client
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            executors={'default': AsyncIOExecutor()},
            job_defaults={'coalesce': True, 'max_instances': 1, 'misfire_grace_time': 60*5}, # Grace time 5 mins
            timezone=NY_TZ # Schedule jobs based on New York time
        )
        self._jobs = {} # To keep track of scheduled jobs
        self._market_calendar: Optional[List[Any]] = None
        self._last_calendar_fetch: Optional[datetime] = None

        if not ALPACA_AVAILABLE or not self.alpaca_client:
            logger.error("Alpaca client not available. Scheduler cannot function properly.")
            # Consider raising an error or disabling scheduling

    async def _get_market_calendar(self, force_refresh: bool = False) -> Optional[List[Any]]:
        """Fetches and caches the market calendar for the next few days."""
        now = datetime.now(timezone.utc)
        if not force_refresh and self._market_calendar and self._last_calendar_fetch and \
           (now - self._last_calendar_fetch) < timedelta(hours=12):
            return self._market_calendar

        if not self.alpaca_client: return None

        try:
            start_date = date.today().isoformat()
            end_date = (date.today() + timedelta(days=7)).isoformat() # Fetch for next 7 days
            logger.info(f"Fetching market calendar from {start_date} to {end_date}...")
            # Run sync Alpaca call in executor
            calendar = await asyncio.get_running_loop().run_in_executor(
                None,
                self.alpaca_client.get_calendar,
                start_date,
                end_date
            )
            self._market_calendar = calendar
            self._last_calendar_fetch = now
            logger.info(f"Fetched {len(calendar)} calendar entries.")
            return calendar
        except AlpacaAPIError as e:
            logger.error(f"Failed to fetch market calendar: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching market calendar: {e}", exc_info=True)
        return None

    async def _is_market_day(self, target_date: date) -> bool:
        """Checks if a given date is a market trading day."""
        calendar = await self._get_market_calendar()
        if not calendar: return False # Assume not a market day if calendar fails

        for day in calendar:
            if day.date.date() == target_date:
                # Check if market is open based on calendar entry
                # Assuming calendar entry exists means it's potentially open
                # More robust check might look at day.open/day.close times if needed
                return True
        return False

    async def add_job(self, func: Callable, trigger: Any, job_id: str, **kwargs):
        """Adds or replaces a job in the scheduler."""
        if job_id in self._jobs:
            self.scheduler.remove_job(job_id)
            logger.debug(f"Removed existing job: {job_id}")

        job = self.scheduler.add_job(func, trigger=trigger, id=job_id, **kwargs)
        self._jobs[job_id] = job
        logger.info(f"Scheduled job '{job_id}' with trigger: {trigger}")

    def remove_job(self, job_id: str):
        """Removes a job from the scheduler."""
        if job_id in self._jobs:
            try:
                self.scheduler.remove_job(job_id)
                del self._jobs[job_id]
                logger.info(f"Removed scheduled job: {job_id}")
            except Exception as e:
                 logger.error(f"Error removing job {job_id}: {e}")
        else:
             logger.warning(f"Attempted to remove non-existent job: {job_id}")

    async def schedule_daily_tasks(self, tasks: Dict[str, Callable]):
        """
        Schedules tasks to run daily based on market open/close times.

        Args:
            tasks: A dictionary mapping task names (e.g., 'pre_market', 'market_open',
                   'market_close', 'post_market') to async functions to run.
        """
        calendar = await self._get_market_calendar(force_refresh=True)
        if not calendar:
            logger.error("Cannot schedule daily tasks without market calendar.")
            return

        today_date = date.today() # Get today's date once
        today_cal = next((day for day in calendar if day.date == today_date), None) # Compare date objects directly

        if not today_cal:
            logger.info(f"{today_date} is not a market day according to calendar. No daily tasks scheduled.")
            return

        # --- Schedule Pre-Market Task ---
        if 'pre_market' in tasks:
            pre_market_time_ny = time(8, 0, 0) # Example: 8:00 AM NY time
            trigger = CronTrigger(hour=pre_market_time_ny.hour, minute=pre_market_time_ny.minute, day_of_week='mon-fri', timezone=NY_TZ)
            # We could also use DateTrigger based on today_cal.open if more precision needed
            await self.add_job(tasks['pre_market'], trigger, 'pre_market_task')

        # --- Schedule Market Open Task ---
        if 'market_open' in tasks:
            market_open_time_ny = today_cal.open # Time object in NY time
            trigger = DateTrigger(run_date=datetime.combine(date.today(), market_open_time_ny, tzinfo=NY_TZ))
            await self.add_job(tasks['market_open'], trigger, 'market_open_task')

        # --- Schedule Market Close Task (EOD Liquidation) ---
        if 'market_close' in tasks:
            market_close_time_ny = today_cal.close # Time object in NY time
            # Schedule slightly before close based on config
            eod_minutes = self.config.get_int("EOD_CLOSE_MINUTES_BEFORE", 15)
            close_trigger_dt_ny = datetime.combine(date.today(), market_close_time_ny, tzinfo=NY_TZ) - timedelta(minutes=eod_minutes)
            trigger = DateTrigger(run_date=close_trigger_dt_ny)
            await self.add_job(tasks['market_close'], trigger, 'market_close_task')

        # --- Schedule Post-Market Task ---
        if 'post_market' in tasks:
            post_market_time_ny = time(17, 0, 0) # Example: 5:00 PM NY time
            trigger = CronTrigger(hour=post_market_time_ny.hour, minute=post_market_time_ny.minute, day_of_week='mon-fri', timezone=NY_TZ)
            await self.add_job(tasks['post_market'], trigger, 'post_market_task')

        # --- Schedule Daily Symbol Refresh ---
        if 'refresh_symbols' in tasks:
             # Run once early, e.g., 7 AM NY time
             refresh_time_ny = time(7, 0, 0)
             trigger = CronTrigger(hour=refresh_time_ny.hour, minute=refresh_time_ny.minute, day_of_week='mon-fri', timezone=NY_TZ)
             await self.add_job(tasks['refresh_symbols'], trigger, 'refresh_symbols_daily')


    def start(self):
        """Starts the scheduler."""
        try:
            if not self.scheduler.running:
                self.scheduler.start()
                logger.info("Scheduler started.")
            else:
                 logger.warning("Scheduler already running.")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}", exc_info=True)

    async def stop(self):
        """Stops the scheduler gracefully."""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True) # Wait for running jobs to complete
                logger.info("Scheduler shut down.")
            # Clear job tracking
            self._jobs = {}
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}", exc_info=True)
