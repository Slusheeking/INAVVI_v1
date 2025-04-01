#!/usr/bin/env python3
"""
ML Engine Main Module

This module serves as the entry point for the ML engine:
1. Initializes the ML engine components
2. Connects to Redis for caching and notifications
3. Loads data and trains models
4. Provides prediction functionality
5. Handles error reporting and monitoring

Usage:
    python -m ml_engine.main [--train] [--predict] [--optimize]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union
import redis # Keep for type hint if needed, but use RedisClient

from dotenv import load_dotenv

from ml_engine.base import MLModelTrainer
from data_pipeline.main import DataPipeline # Import DataPipeline at top level
from utils.logging_config import get_logger
from utils.config import get_config, Config # Import config system
from utils.redis_helpers import RedisClient # Import shared Redis client
from utils.exceptions import ConfigurationError, RedisConnectionError, DataError # Import custom exceptions

# Configure logging
logger = get_logger("ml_engine.main")
# load_dotenv() # Handled by Config initialization if needed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML Engine")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    return parser.parse_args()


# Removed initialize_redis function - use RedisClient directly


def initialize_data_loader(config: Config, redis_client: RedisClient) -> Optional['DataPipeline']:
    """Initialize data loader using Config and shared RedisClient."""
    # Assuming DataPipeline is refactored to accept config and redis_client
    try:
        # Import moved to top level

        # DataPipeline should ideally take config and redis_client
        # and initialize its own dependencies (like Polygon clients) based on config.
        data_loader = DataPipeline(config=config, redis_client=redis_client)
        # await data_loader.initialize() # If DataPipeline has async init

        logger.info("Data loader initialized successfully")
        return data_loader
    except ImportError as e:
         logger.error(f"Failed to import DataPipeline: {e}. Ensure data_pipeline module is correctly structured.")
         return None
    except Exception as e:
        logger.error(f"Error initializing data loader: {e}", exc_info=True)
        return None


def run_gpu_diagnostics(config: Config) -> Dict[str, Any]:
    """Run GPU diagnostics using config."""
    try:
        from utils.gpu_utils import run_diagnostics as run_gpu_diag_util

        # Pass config to the utility function if it accepts it
        # diagnostics_results = run_gpu_diag_util(config=config)
        diagnostics_results = run_gpu_diag_util() # Assuming it uses global config or env vars for now
        logger.info("GPU diagnostics completed")

        # Log diagnostics results
        for key, value in diagnostics_results.items():
            if isinstance(value, dict):
                logger.info(f"{key}: {json.dumps(value, indent=2)}")
            else:
                logger.info(f"{key}: {value}")

        return diagnostics_results
    except ImportError as e:
         logger.warning(f"Could not import GPU diagnostics utility: {e}")
         return {"error": "GPU diagnostics utility not found"}
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}", exc_info=True)
        return {"error": str(e)}


def main():
    """Main entry point"""
    try:
        # Get centralized configuration
        config = get_config()

        # Parse command line arguments
        args = parse_args()
        
        # Run diagnostics
        # Run diagnostics (passing config)
        diagnostics_results = run_gpu_diagnostics(config)

        # Initialize shared RedisClient using config
        try:
            redis_client = RedisClient(config=config)
            # Ensure connection works (optional, RedisClient might do lazy init)
            # await redis_client.ensure_initialized() # If RedisClient has async init
            # await redis_client.ping() # Or a simple ping
            logger.info("RedisClient initialized successfully.")
        except (RedisConnectionError, ConfigurationError) as e:
            logger.error(f"Failed to initialize RedisClient: {e}", exc_info=True)
            return 1
            
        # Initialize data loader
        # Initialize data loader (passing config and redis_client)
        data_loader = initialize_data_loader(config, redis_client)
        if not data_loader:
            logger.error("Failed to initialize data loader, exiting.")
            # Consider sending error notification here too
            return 1
            
        # Create model trainer
        # Create model trainer (assuming it's refactored to accept config, redis, data_loader)
        # model_trainer = MLModelTrainer(config=config, redis_client=redis_client, data_loader=data_loader)
        model_trainer = MLModelTrainer(redis_client, data_loader) # Keep original for now if not refactored
        
        # Send system startup notification to frontend
        try:
            # Create notification for frontend
            notification = {
                "type": "system_startup",
                "message": "ML Engine started successfully",
                "level": "success",
                "timestamp": time.time(),
                "details": {
                    "gpu_available": model_trainer.use_gpu if hasattr(model_trainer, "use_gpu") else False,
                    "device_name": model_trainer.device_name if hasattr(model_trainer, "device_name") else "CPU",
                    "diagnostics": diagnostics_results,
                    "startup_time": time.time()
                }
            }
            
            # Push to notifications list
            redis_client.lpush("frontend:notifications", json.dumps(notification))
            redis_client.ltrim("frontend:notifications", 0, 99)
            
            # Also store in system_startup category
            redis_client.lpush("frontend:system_startup", json.dumps(notification))
            redis_client.ltrim("frontend:system_startup", 0, 49)
            
            # Update system status
            system_status = {
                "running": True,
                "startup_time": time.time(),
                "last_update": time.time(),
                "status": "success",
                "last_message": "ML Engine started successfully"
            }
            redis_client.set("frontend:system:status", json.dumps(system_status))
            
            logger.info("Startup notification sent to frontend")
        except redis.RedisError as e: # Catch specific Redis errors
            logger.error(f"Redis error sending startup notification: {e}", exc_info=True)
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error sending startup notification: {e}", exc_info=True)
        
        # Process command line arguments
        if args.optimize:
            logger.info("Optimizing hyperparameters")
            os.environ["OPTIMIZE_HYPERPARAMS"] = "true"
            
        if args.train or not (args.predict or args.optimize):
            # Train all models (default behavior if no args specified)
            logger.info("Training all models")
            model_trainer.train_all_models()
            
        if args.predict:
            # Make predictions
            logger.info("Making predictions")
            # Load market data
            market_data = data_loader.load_latest_market_data()
            if market_data is not None and not market_data.empty:
                predictions = model_trainer.predict_signals(market_data)
                logger.info(f"Generated predictions for {len(predictions)} tickers")
            else:
                logger.error("No market data available for predictions")
        
        logger.info("ML Engine completed successfully")
        return 0
        
    except ConfigurationError as e:
         logger.error(f"Configuration error during execution: {e}", exc_info=True)
    except DataError as e:
         logger.error(f"Data processing error during execution: {e}", exc_info=True)
    except redis.RedisError as e:
         logger.error(f"Redis error during execution: {e}", exc_info=True)
    except Exception as e: # General fallback
        logger.error(f"Unexpected error in main execution: {e}", exc_info=True)
        
        # Send error notification to frontend
        if 'redis_client' in locals() and redis_client:
            try:
                # Create notification for frontend
                notification = {
                    "type": "system_error",
                    "message": f"ML Engine error: {str(e)}",
                    "level": "error",
                    "timestamp": time.time(),
                    "details": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": time.time()
                    }
                }
                
                # Push to notifications list
                redis_client.lpush("frontend:notifications", json.dumps(notification))
                redis_client.ltrim("frontend:notifications", 0, 99)
                
                # Also store in system_error category
                redis_client.lpush("frontend:system_error", json.dumps(notification))
                redis_client.ltrim("frontend:system_error", 0, 49)
                
                # Update system status
                system_status = json.loads(redis_client.get("frontend:system:status") or "{}")
                system_status["status"] = "error"
                system_status["last_error"] = str(e)
                system_status["last_update"] = time.time()
                redis_client.set("frontend:system:status", json.dumps(system_status))
                
                logger.info("Error notification sent to frontend")
            except redis.RedisError as notify_error: # Catch specific Redis errors
                logger.error(f"Redis error sending error notification: {notify_error}", exc_info=True)
            except Exception as notify_error: # Catch other unexpected errors
                logger.error(f"Unexpected error sending error notification: {notify_error}", exc_info=True)
                
        return 1


if __name__ == "__main__":
    sys.exit(main())