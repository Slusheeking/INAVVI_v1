#!/usr/bin/env python3
"""
Stock Selection System Main Entry Point

This module serves as the main entry point for the stock selection system.
It initializes all components and manages the lifecycle of the application.
Refactored for improved modularity and readability.
"""

import asyncio
import logging
import os
import json
import time
import random # For dummy ML signal
import threading # To run Prometheus server
from typing import Dict, Any, Optional, List # Import List

import redis.asyncio as aioredis # Use async redis
from dotenv import load_dotenv
from prometheus_client import start_http_server # Import Prometheus server
from apscheduler.schedulers.asyncio import AsyncIOScheduler # Import APScheduler
from apscheduler.triggers.interval import IntervalTrigger

# --- Import System Components ---
# API Clients
from api_clients.polygon_rest import PolygonRESTClient
from api_clients.unusual_whales import UnusualWhalesClient

# Data Pipeline
from data_pipeline.base import DataPipeline

# Learning Engine (MLOps)
from learning_engine.registry import ModelRegistry
from learning_engine.deployer import ModelDeployer

# Trading Engine (Execution & Core)
from trading_engine.base import TradingEngine # Import TradingEngine
from trading_engine.execution import ExecutionSystem, PaperExecution # Using PaperExecution for now

# Stock Selection Components
from stock_selection.core import StockSelectionCore # Use the refactored core
from stock_selection.day_trading import DayTradingSystem

# Utilities
from utils.logging_config import get_logger
from utils import config as app_config # Renamed to avoid conflict
from utils.redis_helpers import RedisClient, get_async_redis_client, RedisCache, async_send_notification
from utils.exceptions import ConfigurationError

# Load environment variables from .env file if present
load_dotenv()

# Initialize logging using the shared configuration
logger = get_logger("stock_selection")

# --- Configuration Loading ---

# Prometheus metrics server configuration
PROMETHEUS_PORT = int(os.environ.get("PROMETHEUS_PORT", "8000"))
PROMETHEUS_HOST = os.environ.get("PROMETHEUS_HOST", "0.0.0.0") # Listen on all interfaces

def load_configuration() -> Dict[str, Any]:
    """Loads and validates configuration."""
    logger.info("Loading configuration...")
    try:
        use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
        polygon_api_key = os.environ.get("POLYGON_API_KEY")
        unusual_whales_api_key = os.environ.get("UNUSUAL_WHALES_API_KEY")
        symbols_str = os.environ.get("TRADING_SYMBOLS", "AAPL,MSFT,GOOG")
        trading_symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        models_dir = os.environ.get("MODELS_DIR", "./test_models")
        ml_interval = int(os.environ.get("ML_INFERENCE_INTERVAL_SECONDS", "10"))
        strategy_interval = int(os.environ.get("STRATEGY_INTERVAL_SECONDS", "60"))

        if not polygon_api_key: logger.warning("POLYGON_API_KEY not set.")
        if not unusual_whales_api_key: logger.warning("UNUSUAL_WHALES_API_KEY not set.")
        if not trading_symbols: raise ConfigurationError("TRADING_SYMBOLS not set.")

        logger.info(f"GPU usage: {use_gpu}")
        logger.info(f"Trading symbols: {trading_symbols}")
        logger.info(f"Models dir: {models_dir}")
        logger.info(f"ML Interval: {ml_interval}s")
        logger.info(f"Strategy Interval: {strategy_interval}s")

        return {
            "use_gpu": use_gpu, "polygon_api_key": polygon_api_key,
            "unusual_whales_api_key": unusual_whales_api_key,
            "trading_symbols": trading_symbols, "models_dir": models_dir,
            "ml_inference_interval": ml_interval,
            "strategy_execution_interval": strategy_interval,
        }
    except Exception as e:
        logger.exception(f"Configuration loading failed: {e}")
        raise

# --- Initialization Functions ---

async def initialize_redis() -> Optional[RedisClient]:
    """Initializes RedisClient."""
    logger.info(f"Connecting to Redis...")
    try:
        redis_client = RedisClient()
        await redis_client.ensure_initialized()
        if not redis_client.client: raise ConfigurationError("RedisClient failed init.")
        logger.info("Redis connection successful.")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}. Continuing without Redis.")
        return None

def initialize_api_clients(cfg: Dict[str, Any], redis_cache: RedisCache) -> Dict[str, Any]:
    """Initializes API clients."""
    logger.info("Initializing API clients...")
    clients = {"polygon_rest": None, "unusual_whales": None, "polygon_ws": None}
    if cfg.get("polygon_api_key"):
        clients["polygon_rest"] = PolygonRESTClient(api_key=cfg["polygon_api_key"], cache=redis_cache)
        logger.info("Polygon REST client initialized.")
    else: logger.warning("Polygon REST client not initialized (API key missing).")
    if cfg.get("unusual_whales_api_key"):
        clients["unusual_whales"] = UnusualWhalesClient(api_key=cfg["unusual_whales_api_key"], cache=redis_cache)
        logger.info("Unusual Whales client initialized.")
    else: logger.warning("Unusual Whales client not initialized (API key missing).")
    return clients

def initialize_systems(cfg: Dict[str, Any], redis_client: Optional[RedisClient], api_clients: Dict[str, Any]) -> Dict[str, Any]:
    """Initializes core application systems."""
    logger.info("Initializing application systems...")
    trading_engine_config = {"api_key": cfg.get("polygon_api_key"), "symbols": cfg.get("trading_symbols", [])}
    trading_engine = TradingEngine(config=trading_engine_config, redis_client=redis_client, polygon_rest_client=api_clients.get("polygon_rest"))
    data_pipeline = DataPipeline(polygon_client=api_clients.get("polygon_rest"), unusual_whales_client=api_clients.get("unusual_whales"), redis_client=redis_client, config=cfg, use_gpu=cfg["use_gpu"])
    model_registry = ModelRegistry(models_dir=cfg["models_dir"])
    model_deployer = ModelDeployer(model_registry=model_registry)
    execution_system = PaperExecution(engine=trading_engine) # Using Paper Trading
    stock_selection_core = StockSelectionCore(data_pipeline=data_pipeline, model_registry=model_registry, model_deployer=model_deployer, redis_client=redis_client)
    day_trading = DayTradingSystem(engine=trading_engine, execution_system=execution_system)
    logger.info("All core systems initialized.")
    return {
        "trading_engine": trading_engine, "stock_selection_core": stock_selection_core,
        "day_trading": day_trading, "data_pipeline": data_pipeline,
        "polygon_rest_client": api_clients.get("polygon_rest"),
        "unusual_whales_client": api_clients.get("unusual_whales"),
        "execution_system": execution_system, "model_registry": model_registry,
        "model_deployer": model_deployer,
    }

# --- Lifecycle Management Functions ---

async def start_systems(systems: Dict[str, Any]):
    """Starts all initialized systems concurrently."""
    logger.info("Starting all systems...")
    start_tasks = []
    for name, system in systems.items():
         if system and hasattr(system, 'start') and callable(system.start):
              if asyncio.iscoroutinefunction(system.start): start_tasks.append(asyncio.create_task(system.start(), name=f"{name}_start"))
              else: logger.warning(f"System '{name}' has sync start.")
         elif system and hasattr(system, 'initialize') and callable(system.initialize):
              if asyncio.iscoroutinefunction(system.initialize): start_tasks.append(asyncio.create_task(system.initialize(), name=f"{name}_initialize"))
              else: logger.warning(f"System '{name}' has sync initialize.")
         else: logger.debug(f"System '{name}' has no start/initialize method.")
    if start_tasks:
        results = await asyncio.gather(*start_tasks, return_exceptions=True)
        failed_tasks = [ (start_tasks[i].get_name(), r) for i, r in enumerate(results) if isinstance(r, Exception) ]
        if failed_tasks:
             for name, err in failed_tasks: logger.error(f"Error starting system task '{name}': {err}", exc_info=err)
             raise RuntimeError("System startup failed.")
        else: logger.info("All systems started successfully.")
    else: logger.warning("No systems found with start/initialize methods.")

async def stop_systems(systems: Dict[str, Any]):
    """Stops all initialized systems concurrently."""
    logger.info("Stopping all systems...")
    stop_tasks = []
    shutdown_order = ["day_trading", "stock_selection_core", "data_pipeline", "execution_system", "trading_engine"]
    processed = set()
    for name in shutdown_order:
        if name in systems and systems[name]:
            system = systems[name]
            method_name, method = None, None
            if hasattr(system, 'stop') and callable(system.stop): method_name, method = 'stop', system.stop
            elif hasattr(system, 'shutdown') and callable(system.shutdown): method_name, method = 'shutdown', system.shutdown
            elif hasattr(system, 'close') and callable(system.close): method_name, method = 'close', system.close
            if method:
                if asyncio.iscoroutinefunction(method): stop_tasks.append(asyncio.create_task(method(), name=f"{name}_{method_name}"))
                else: logger.warning(f"System '{name}' has sync {method_name}.")
            processed.add(name)
    for name, system in systems.items():
         if name in processed or not system: continue
         method_name, method = None, None
         if hasattr(system, 'stop') and callable(system.stop): method_name, method = 'stop', system.stop
         elif hasattr(system, 'shutdown') and callable(system.shutdown): method_name, method = 'shutdown', system.shutdown
         elif hasattr(system, 'close') and callable(system.close): method_name, method = 'close', system.close
         if method:
             if asyncio.iscoroutinefunction(method): stop_tasks.append(asyncio.create_task(method(), name=f"{name}_{method_name}"))
             else: logger.warning(f"System '{name}' has sync {method_name}.")
         else: logger.debug(f"System '{name}' has no stop/shutdown/close method.")
    if stop_tasks:
        try:
            results = await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=20.0)
            failed_tasks = [ (stop_tasks[i].get_name(), r) for i, r in enumerate(results) if isinstance(r, Exception) ]
            for name, err in failed_tasks: logger.error(f"Error stopping system task '{name}': {err}", exc_info=err)
            logger.info("All systems stop/shutdown/close sequence initiated.")
        except asyncio.TimeoutError: logger.warning("Timeout waiting for systems to stop.")
        except Exception as e: logger.error(f"Error during system stop gathering: {e}", exc_info=True)
    else: logger.warning("No systems found with stop/shutdown/close methods.")

# --- Notification Functions ---

async def update_redis_system_status(redis_client: Optional[RedisClient], running: bool, reason: Optional[str] = None):
    """Updates system status in Redis."""
    if not redis_client or not redis_client.client: return
    status_key = "frontend:system:status"
    try:
        status_json = await redis_client.client.get(status_key)
        system_status = json.loads(status_json) if status_json else {}
        system_status.update({"running": running, "timestamp": time.time()})
        if not running: system_status["shutdown_time"] = time.time(); system_status["shutdown_reason"] = reason
        else: system_status["startup_time"] = time.time(); system_status.pop("shutdown_time", None); system_status.pop("shutdown_reason", None)
        await redis_client.client.set(status_key, json.dumps(system_status))
        logger.info(f"Updated Redis system status: running={running}")
    except Exception as e: logger.error(f"Redis error/JSON error updating system status: {e}")

# --- Scheduled Job Functions ---

async def scheduled_ml_inference(trading_engine: TradingEngine):
    """Job function for ML inference, checks market hours."""
    if not await trading_engine.is_market_open(extended_hours=False):
        logger.debug("Market closed. Skipping scheduled ML inference.")
        return
    logger.debug("Running scheduled ML inference...")
    symbols = trading_engine.config.get('symbols', [])
    feature_tasks = [trading_engine.get_latest_features(symbol) for symbol in symbols]
    all_features = await asyncio.gather(*feature_tasks, return_exceptions=True)
    for symbol, features in zip(symbols, all_features):
        if isinstance(features, Exception): logger.error(f"Error fetching features for {symbol}: {features}"); continue
        if features is None: logger.debug(f"No features for {symbol} for inference."); continue
        # ** Placeholder for actual ML model inference **
        signal, confidence = random.choice(['buy', 'sell', 'hold']), random.random()
        prediction = {"signal": signal, "confidence": confidence}
        logger.info(f"ML Signal for {symbol}: {prediction['signal']} (Conf: {prediction['confidence']:.2f})")
        # TODO: Integrate signal (e.g., publish to Redis)
        signal_data = {"symbol": symbol, "signal": prediction['signal'], "confidence": prediction['confidence'], "timestamp": time.time()}
        if trading_engine.redis and trading_engine.redis.client:
             await async_send_notification(f"signals:{symbol}", signal_data, trading_engine.redis.client)


async def scheduled_strategy_execution(systems: Dict[str, Any]):
    """Job function for strategy execution, checks market hours."""
    trading_engine = systems.get("trading_engine")
    day_trading_system = systems.get("day_trading")
    if not trading_engine or not day_trading_system: logger.error("Missing engine/system for strategy execution."); return
    if not await trading_engine.is_market_open(extended_hours=False):
        logger.debug("Market closed. Skipping scheduled strategy execution.")
        return
    logger.info("Market open. Triggering scheduled strategy execution...")
    # --- Placeholder for actual strategy execution logic ---
    symbols_to_trade = trading_engine.config.get('symbols', [])
    strategy_tasks = []
    for symbol in symbols_to_trade:
         strategy_tasks.append(day_trading_system.run_opening_range_breakout(symbol))
         strategy_tasks.append(day_trading_system.run_vwap_reversion(symbol))
    if strategy_tasks:
         results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
         for result in results:
              if isinstance(result, Exception): logger.error(f"Error during strategy execution: {result}", exc_info=result)
              elif isinstance(result, dict) and result.get("order_id"): logger.info(f"Strategy submitted order: {result.get('order_id')} for {result.get('symbol')}")
    # --- End Placeholder ---

# --- Main Application Logic ---

async def main_entry() -> int:
    """Asynchronous main entry point orchestrating the application lifecycle."""
    exit_code = 0
    redis_client: Optional[RedisClient] = None
    systems: Dict[str, Any] = {}
    scheduler: Optional[AsyncIOScheduler] = None
    prometheus_thread = None
    default_api_cache = RedisCache(prefix="api_client_cache")

    try:
        # Start Prometheus metrics server
        try:
            prometheus_thread = threading.Thread(target=start_http_server, args=(PROMETHEUS_PORT, PROMETHEUS_HOST), daemon=True)
            prometheus_thread.start()
            logger.info(f"Prometheus metrics server started on http://{PROMETHEUS_HOST}:{PROMETHEUS_PORT}/metrics")
        except Exception as prom_e: logger.error(f"Failed to start Prometheus metrics server: {prom_e}")

        cfg = load_configuration()
        redis_client = await initialize_redis()

        # Initialize API clients cache
        default_api_cache.redis_client = redis_client.client if redis_client else None
        default_api_cache.redis_available = bool(redis_client and redis_client.client)
        api_clients = initialize_api_clients(cfg, default_api_cache)

        # Initialize core systems
        systems = initialize_systems(cfg, redis_client, api_clients)
        await start_systems(systems) # Starts TradingEngine feed etc.

        # Initialize and start scheduler
        scheduler = AsyncIOScheduler(timezone="UTC") # Use UTC timezone
        trading_engine = systems.get("trading_engine")
        if trading_engine:
            # Schedule ML Inference
            ml_interval = cfg.get("ml_inference_interval", 10)
            scheduler.add_job(
                scheduled_ml_inference,
                trigger=IntervalTrigger(seconds=ml_interval),
                args=[trading_engine],
                id="ml_inference_job",
                name="ML Inference Job",
                replace_existing=True,
                misfire_grace_time=5 # Allow 5s grace period
            )
            logger.info(f"Scheduled ML inference job every {ml_interval} seconds.")

            # Schedule Strategy Execution
            strategy_interval = cfg.get("strategy_execution_interval", 60)
            scheduler.add_job(
                scheduled_strategy_execution,
                # Example: Run only on weekdays during typical market hours (adjust timezone/hours)
                # trigger=CronTrigger(day_of_week='mon-fri', hour='9-16', minute='*', second='*/60', timezone='America/New_York'),
                # Using IntervalTrigger for simplicity now, relies on internal market check
                trigger=IntervalTrigger(seconds=strategy_interval),
                args=[systems],
                id="strategy_execution_job",
                name="Strategy Execution Job",
                replace_existing=True,
                misfire_grace_time=15 # Allow 15s grace period
            )
            logger.info(f"Scheduled strategy execution job every {strategy_interval} seconds.")

            scheduler.start()
            logger.info("Scheduler started.")
        else:
             logger.error("TradingEngine not found. Cannot start scheduler.")
             raise RuntimeError("TradingEngine failed to initialize.")

        # Send Startup Notification
        startup_details = {"components": [name for name, sys in systems.items() if sys is not None]}
        if trading_engine: startup_details.update({"gpu_enabled": trading_engine.gpu_enabled, "trading_symbols": trading_engine.config.get("symbols")})
        if redis_client:
            await async_send_notification(redis_client.client, "system_startup", "Stock Selection Engine started successfully", "success", details=startup_details, category_key="frontend:system_startup")
            await update_redis_system_status(redis_client, running=True)

        # Keep main process alive (scheduler runs in background)
        while True: await asyncio.sleep(3600)

    except (ConfigurationError, RuntimeError) as e:
        logger.critical(f"Failed to start: {e}")
        exit_code = 1
    except (KeyboardInterrupt, asyncio.CancelledError):
         logger.info("Shutdown signal received.")
         exit_code = 0 # Normal exit on signal
    except Exception as e:
        logger.critical(f"Critical error during runtime: {e}", exc_info=True)
        exit_code = 1
    finally:
        logger.info("Initiating shutdown sequence...")
        # Shutdown scheduler first
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False) # Don't wait for jobs to finish
            logger.info("Scheduler shutdown.")
        # Stop systems (includes TradingEngine shutdown which closes Redis)
        if systems: await stop_systems(systems)
        # Send Shutdown Notification
        shutdown_reason = "System initiated"
        shutdown_details = {"components": [name for name, sys in systems.items() if sys is not None], "shutdown_time": time.time(), "shutdown_reason": shutdown_reason}
        if redis_client and redis_client.client:
             try:
                  await redis_client.client.ping()
                  await async_send_notification(redis_client.client, "system_shutdown", "Stock Selection Engine stopped", "info", details=shutdown_details, category_key="frontend:system_shutdown")
                  await update_redis_system_status(redis_client, running=False, reason=shutdown_reason)
             except Exception as e: logger.error(f"Error sending shutdown notification: {e}")
        await default_api_cache.close()
        # Prometheus server thread is daemon
        logger.info(f"Shutdown complete. Exiting with code {exit_code}.")
    return exit_code

if __name__ == "__main__":
    try: loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)

    # Removed signal handling here, rely on KeyboardInterrupt/CancelledError in main_entry
    try:
        exit_status = loop.run_until_complete(main_entry())
        raise SystemExit(exit_status)
    except (KeyboardInterrupt, asyncio.CancelledError):
         logger.info("Main execution interrupted/cancelled.")
         # The finally block in main_entry handles cleanup
         raise SystemExit(1) # Indicate non-standard exit
    finally:
        # Final cleanup of any remaining tasks (should be handled by main_entry's finally)
        tasks = asyncio.all_tasks(loop=loop)
        tasks_to_cancel = [t for t in tasks if t is not asyncio.current_task(loop=loop) and not t.done()]
        if tasks_to_cancel:
             for task in tasks_to_cancel: task.cancel()
             loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
        logger.info("Event loop stopped.")
        # loop.close()
