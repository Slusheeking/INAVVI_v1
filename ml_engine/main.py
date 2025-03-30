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

from dotenv import load_dotenv

from ml_engine.base import MLModelTrainer
from utils.logging_config import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger("ml_engine.main")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML Engine")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument("--optimize", action="store_true", help="Optimize hyperparameters")
    return parser.parse_args()


def initialize_redis():
    """Initialize Redis client"""
    try:
        import redis
        
        redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6380)),
            db=int(os.environ.get("REDIS_DB", 0)),
            username=os.environ.get("REDIS_USERNAME", "default"),
            password=os.environ.get("REDIS_PASSWORD", "trading_system_2025"),
        )
        
        # Test connection
        redis_client.ping()
        logger.info("Connected to Redis successfully")
        return redis_client
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return None


def initialize_data_loader(redis_client):
    """Initialize data loader"""
    try:
        from data_pipeline import DataPipeline
        
        data_loader = DataPipeline(
            redis_client=redis_client,
            polygon_client=None,  # Will be initialized by DataPipeline
            polygon_ws=None,      # Will be initialized by DataPipeline
            unusual_whales_client=None,  # Will be initialized by DataPipeline
            use_gpu=os.environ.get("USE_GPU", "true").lower() == "true",
        )
        
        logger.info("Data loader initialized successfully")
        return data_loader
    except Exception as e:
        logger.error(f"Error initializing data loader: {e}")
        return None


def run_diagnostics():
    """Run GPU diagnostics"""
    try:
        from utils.gpu_utils import run_diagnostics
        
        diagnostics_results = run_diagnostics()
        logger.info("GPU diagnostics completed")
        
        # Log diagnostics results
        for key, value in diagnostics_results.items():
            if isinstance(value, dict):
                logger.info(f"{key}: {json.dumps(value, indent=2)}")
            else:
                logger.info(f"{key}: {value}")
                
        return diagnostics_results
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
        return {}


def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Run diagnostics
        diagnostics_results = run_diagnostics()
        
        # Initialize Redis
        redis_client = initialize_redis()
        if not redis_client:
            logger.error("Failed to initialize Redis, exiting")
            return 1
            
        # Initialize data loader
        data_loader = initialize_data_loader(redis_client)
        if not data_loader:
            logger.error("Failed to initialize data loader, exiting")
            return 1
            
        # Create model trainer
        model_trainer = MLModelTrainer(redis_client, data_loader)
        
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
        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")
        
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
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        
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
            except Exception as notify_error:
                logger.error(f"Error sending error notification: {notify_error}")
                
        return 1


if __name__ == "__main__":
    sys.exit(main())