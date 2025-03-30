#!/usr/bin/env python3
"""
Stock Selection Core Module

Contains the fundamental stock selection logic and scoring algorithms.
This module focuses on the core selection logic without GPU acceleration,
integrating with the ML engine for scoring.
"""

import logging
import time
import asyncio
import random
import os
import redis
from typing import Dict, List, Optional, Set, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Assuming get_redis_client returns a redis.Redis compatible client
# Need RedisClient for async operations now
from utils.redis_helpers import RedisClient # Changed import
from utils.metrics_registry import (
    UNIVERSE_SIZE,
    API_ERROR_COUNT,
    WATCHLIST_SIZE,
    FOCUSED_LIST_SIZE,
    STOCK_SCORES,
    MODEL_PREDICTION_LATENCY
)
from utils.exceptions import RedisConnectionError, StockSelectionError, DataProcessingError, ModelError
from data_pipeline.base import DataPipeline
from data_pipeline.processing import clean_market_data, calculate_technical_indicators, add_time_features
from learning_engine.registry import ModelRegistry
from learning_engine.deployer import ModelDeployer
from ml_engine.xgboost_model import XGBoostModel # Keep for type hint if needed

logger = logging.getLogger("stock_selection_core")

# DEFAULT_UNIVERSE removed - rely on dynamic fetching
DEFAULT_MODEL_NAME = "signal_detection_model"

class StockSelectionCore:
    """
    Core stock selection system implementing ML-based scoring and filtering logic.
    """

    def __init__(
        self,
        data_pipeline: DataPipeline,
        model_registry: ModelRegistry,
        model_deployer: ModelDeployer,
        redis_client: Optional[RedisClient] = None, # Expecting async RedisClient
        scoring_model_name: str = DEFAULT_MODEL_NAME
    ):
        """
        Initialize the StockSelectionCore.

        Args:
            data_pipeline: An instance of the DataPipeline for data access.
            model_registry: Instance of ModelRegistry to load models.
            model_deployer: Instance of ModelDeployer to find deployed models.
            redis_client: Optional async RedisClient instance.
            scoring_model_name: Name of the ML model to use for scoring.
        """
        self.data_pipeline = data_pipeline
        self.model_registry = model_registry
        self.model_deployer = model_deployer
        self.scoring_model_name = scoring_model_name
        self.loaded_model: Optional[Any] = None
        self.loaded_model_version: Optional[int] = None
        self.model_feature_names: Optional[List[str]] = None
        self.current_universe: Set[str] = set() # Initialize as empty set
        self.redis_client = redis_client # Store the async client
        self.loop = asyncio.get_running_loop() # Get current event loop

        # Log initialization status
        redis_status = "available" if self.redis_client and self.redis_client.client else "unavailable"
        logger.info(f"StockSelectionCore initialized. Redis status: {redis_status}")


    async def _validate_redis_connection(self):
        """Verify Redis connection is working if a client is provided."""
        if not self.redis_client:
            logger.warning("Redis client not available for validation.")
            return # Allow operation without Redis if possible

        try:
            logger.info("Validating Redis connection...")
            # Use ensure_initialized which includes a ping check
            await self.redis_client.ensure_initialized()
            if not self.redis_client.client: # Check again after ensure_initialized
                 raise RedisConnectionError("Redis client failed initialization.")
            logger.info("Redis connection validated successfully.")
        except Exception as e:
            logger.error(f"Redis connection validation failed: {e}")
            # Do not disable client here, let start handle it if needed
            raise RedisConnectionError(f"Redis connection validation failed: {e}") from e


    async def start(self):
        """Initialize core selection system (e.g., build initial universe, load model)."""
        logger.info("Starting Stock Selection Core system...")
        redis_available = True
        try:
            await self._validate_redis_connection()
        except RedisConnectionError as e:
             logger.error(f"Redis connection failed during startup validation: {e}. Running without Redis.")
             # self.redis_client = None # Don't nullify, methods check client attribute
             redis_available = False

        try:
            await self.build_universe() # Build initial universe on start
            await self._load_scoring_model() # Load the scoring model
            logger.info(f"Stock Selection Core started with universe size: {len(self.current_universe)}. Redis available: {redis_available}")
        except Exception as e:
            logger.exception("Failed during startup (universe build or model load).")
            raise StockSelectionError("Failed during startup") from e


    async def stop(self):
        """Clean up resources specific to the Core system."""
        logger.info("Stock Selection Core stopping...")
        self.loaded_model = None
        self.loaded_model_version = None
        self.model_feature_names = None
        logger.info("Stock Selection Core stopped.")

    async def build_universe(self) -> List[str]:
        """
        Builds or refreshes the universe of stocks to consider using DataPipeline.

        Fetches active stock tickers, applies basic filtering, and updates the
        internal universe set. Returns empty list and sets empty universe on failure.

        Returns:
            List of ticker symbols in the current universe.
        """
        logger.info("Building/Refreshing stock universe using DataPipeline...")
        try:
            # Fetch tickers using DataPipeline
            tickers_df = await self.data_pipeline.get_all_tickers(market='stocks', active=True)

            if tickers_df is None or tickers_df.empty:
                logger.error("Failed to fetch tickers from DataPipeline or no active tickers found.")
                API_ERROR_COUNT.labels(api="data_pipeline", endpoint="get_all_tickers", error_type="NoData").inc()
                self.current_universe = set() # Set empty universe on failure
            else:
                # Extract ticker symbols
                if 'ticker' in tickers_df.columns:
                    potential_tickers = tickers_df['ticker'].astype(str).unique()
                    filtered_tickers = {
                        t for t in potential_tickers
                        if 1 <= len(t) <= 10 and t.isupper() and t.isalpha()
                    }
                    self.current_universe = filtered_tickers
                    logger.info(f"Fetched {len(potential_tickers)}, filtered to {len(self.current_universe)} valid tickers.")
                else:
                    logger.error("Ticker column not found in DataFrame from get_all_tickers.")
                    self.current_universe = set() # Set empty universe

            if not self.current_universe:
                 logger.warning("Stock universe is empty after build process!")

            UNIVERSE_SIZE.set(len(self.current_universe))
            logger.info(f"Universe built. Final size: {len(self.current_universe)}")
            return sorted(list(self.current_universe))

        except Exception as e:
            logger.exception(f"Critical error building stock universe: {e}")
            self.current_universe = set() # Clear universe on critical failure
            UNIVERSE_SIZE.set(0)
            # Do not re-raise, allow system to potentially continue without universe for this cycle
            return [] # Return empty list on error

    async def _load_scoring_model(self):
        """
        Loads the currently deployed scoring model using the registry and deployer.
        Runs potentially blocking file I/O in an executor thread.
        """
        logger.info(f"Attempting to load scoring model: {self.scoring_model_name}")
        try:
            # --- Run blocking operations in executor ---
            def _sync_load_operations():
                deployed_version = self.model_deployer.get_deployed_version(
                    self.scoring_model_name, environment="production"
                )
                if deployed_version is None:
                    return None, None, None # Return tuple indicating failure

                # Check if already loaded
                if self.loaded_model and self.loaded_model_version == deployed_version:
                    logger.info(f"Model {self.scoring_model_name} v{deployed_version} already loaded.")
                    return self.loaded_model, deployed_version, self.model_feature_names # Return existing

                logger.info(f"Loading deployed model {self.scoring_model_name} v{deployed_version}...")
                model_obj = self.model_registry.get_model(self.scoring_model_name, deployed_version)
                metadata = self.model_registry.get_model_metadata(self.scoring_model_name, deployed_version)
                feature_names = metadata.get('feature_names')

                if not feature_names:
                     logger.error(f"Feature names not found in metadata for model {self.scoring_model_name} v{deployed_version}. Cannot score.")
                     raise ModelError("Missing feature names in model metadata")

                return model_obj, deployed_version, feature_names

            # Execute the synchronous loading function in the default executor
            model_obj, deployed_version, feature_names = await self.loop.run_in_executor(
                None, _sync_load_operations
            )
            # --- End of executor block ---

            if model_obj is None and deployed_version is None: # Indicates no deployed version found
                logger.warning(f"No deployed version found for model '{self.scoring_model_name}'. Scoring will be disabled.")
                self.loaded_model = None
                self.loaded_model_version = None
                self.model_feature_names = None
                return

            # Check if model was already loaded (returned from executor)
            if self.loaded_model and self.loaded_model_version == deployed_version:
                 return # Already logged inside executor function

            # Update instance variables with newly loaded model
            self.loaded_model = model_obj
            self.loaded_model_version = deployed_version
            self.model_feature_names = feature_names

            logger.info(f"Successfully loaded model {self.scoring_model_name} v{deployed_version} with {len(self.model_feature_names)} features.")

        except ModelError as me: # Catch specific error from sync function
             logger.error(f"ModelError during loading: {me}")
             self.loaded_model = None
             self.loaded_model_version = None
             self.model_feature_names = None
             raise # Re-raise ModelError
        except Exception as e:
            logger.exception(f"Failed to load scoring model '{self.scoring_model_name}': {e}")
            self.loaded_model = None
            self.loaded_model_version = None
            self.model_feature_names = None
            raise ModelError(f"Failed to load scoring model '{self.scoring_model_name}'") from e


    async def _prepare_features_for_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetches intraday (minute) data and prepares the feature set for a single ticker.
        Adjusts lookback and indicator parameters for minute frequency.
        """
        if not self.model_feature_names:
            logger.warning(f"Cannot prepare features for {ticker}: Model feature names not loaded.")
            return None

        try:
            # Determine required lookback in minutes for intraday indicators
            lookback_minutes = 300 # Example: Fetch last 5 hours
            # Fetch MINUTE aggregate data using DataPipeline (which uses TradingEngine now)
            agg_df = await self.data_pipeline.get_aggregates( # Assuming DP still has this method
                ticker,
                timespan='minute',
                multiplier=1,
                minutes=lookback_minutes, # Use minutes parameter
                apply_clean=True
            )

            min_required_bars = 200 # Example
            if agg_df is None or agg_df.empty or len(agg_df) < min_required_bars:
                logger.warning(f"Insufficient MINUTE aggregate data ({len(agg_df) if agg_df is not None else 0} bars) for {ticker} to calculate features (need ~{min_required_bars}).")
                return None

            # Calculate indicators needed by the model
            try:
                 # These calculations might be CPU intensive - consider executor if they are slow
                 agg_df = calculate_technical_indicators(agg_df, window_sizes=[10, 20, 50, 100, 200])
                 agg_df = add_time_features(agg_df)
            except Exception as indicator_err:
                 logger.error(f"Error calculating intraday indicators for {ticker}: {indicator_err}")
                 return None

            # Select only the required features and handle missing values
            features_df = pd.DataFrame(index=agg_df.index)
            missing_req_features = []
            for feature in self.model_feature_names:
                 if feature in agg_df.columns:
                      features_df[feature] = agg_df[feature]
                 else:
                      missing_req_features.append(feature)
                      features_df[feature] = 0 # Or np.nan? Depends on model training

            if missing_req_features:
                 logger.warning(f"Required features missing for {ticker} and filled with 0: {missing_req_features}")

            # Forward fill, backward fill, then fill remaining NaNs with 0
            features_df = features_df.ffill().bfill().fillna(0)

            if not features_df.empty:
                # Return only the latest row with features in the correct order
                return features_df[self.model_feature_names].iloc[[-1]]
            else:
                return None

        except Exception as e:
            logger.error(f"Error preparing features for {ticker}: {e}", exc_info=True)
            return None


    async def score_stocks(self, tickers: List[str]) -> Dict[str, float]:
        """
        Scores a list of stocks using the loaded ML model.
        Fetches required data via DataPipeline, prepares features, and predicts scores.
        Runs prediction in an executor thread.

        Args:
            tickers: A list of ticker symbols to score.

        Returns:
            A dictionary mapping ticker symbols to their calculated scores.

        Raises:
            StockSelectionError: If a critical error occurs during scoring.
        """
        if not tickers:
            logger.warning("score_stocks called with empty ticker list.")
            return {}

        if not self.loaded_model or not self.model_feature_names:
            logger.warning("Scoring model not loaded. Attempting to load...")
            try:
                await self._load_scoring_model()
            except ModelError:
                 logger.error("Failed to load scoring model. Cannot score stocks.")
                 return {}
            if not self.loaded_model:
                 logger.error("Model still not loaded after attempt. Cannot score.")
                 return {}

        valid_tickers = [t for t in tickers if t in self.current_universe]
        if len(valid_tickers) != len(tickers):
             logger.warning(f"Scoring requested for {len(tickers)}, only {len(valid_tickers)} are in universe.")
        if not valid_tickers:
             return {}

        logger.info(f"Scoring {len(valid_tickers)} stocks using model {self.scoring_model_name} v{self.loaded_model_version}...")
        scores: Dict[str, float] = {}
        start_time_scoring = time.time()

        # Prepare features concurrently
        feature_tasks = {
            ticker: asyncio.create_task(self._prepare_features_for_ticker(ticker))
            for ticker in valid_tickers
        }
        await asyncio.gather(*feature_tasks.values())

        features_to_predict_list = []
        tickers_for_prediction = []
        for ticker in valid_tickers:
            task = feature_tasks[ticker]
            if task.exception():
                logger.error(f"Error preparing features for {ticker}: {task.exception()}")
                continue
            features_df = task.result()
            if features_df is not None and not features_df.empty:
                try:
                     # Ensure features are in the correct order expected by the model
                     features_df_ordered = features_df[self.model_feature_names]
                     features_to_predict_list.append(features_df_ordered)
                     tickers_for_prediction.append(ticker)
                except KeyError as ke:
                     logger.error(f"Feature mismatch for {ticker} when ordering: {ke}. Skipping.")
                except Exception as e:
                     logger.error(f"Unexpected error processing features for {ticker}: {e}. Skipping.")
            else:
                logger.warning(f"Could not prepare features for {ticker}. Skipping scoring.")

        if not features_to_predict_list:
            logger.warning("No features could be prepared for any valid tickers.")
            return {}

        try:
            combined_features = pd.concat(features_to_predict_list)
            combined_features = combined_features.astype(np.float32) # Ensure correct dtype
        except Exception as concat_err:
             logger.exception(f"Error combining features for prediction: {concat_err}")
             raise StockSelectionError("Failed to combine features for prediction") from concat_err

        # --- Run potentially blocking prediction in executor ---
        def _sync_predict(model, features):
            start_predict_time = time.time()
            if hasattr(model, 'predict_proba'):
                predicted_probas = model.predict_proba(features)
                # Assuming binary classification, take probability of positive class
                raw_scores = predicted_probas[:, 1] if predicted_probas.ndim > 1 and predicted_probas.shape[1] > 1 else predicted_probas
            elif hasattr(model, 'predict'):
                raw_scores = model.predict(features)
            else:
                 raise ModelError("Loaded model has no standard 'predict' or 'predict_proba' method.")
            predict_latency = time.time() - start_predict_time
            return raw_scores, predict_latency

        try:
            raw_scores, predict_latency = await self.loop.run_in_executor(
                None, _sync_predict, self.loaded_model, combined_features
            )
            MODEL_PREDICTION_LATENCY.labels(model_name=self.scoring_model_name, model_version=self.loaded_model_version).observe(predict_latency)

            if len(raw_scores) != len(tickers_for_prediction):
                 logger.error(f"Prediction output length ({len(raw_scores)}) mismatch with input feature count ({len(tickers_for_prediction)}).")
                 raise ModelError("Prediction output length mismatch")

            for i, ticker in enumerate(tickers_for_prediction):
                try:
                     score = float(raw_scores[i])
                     score = max(0.0, min(1.0, score)) # Clamp score between 0 and 1
                     scores[ticker] = score
                     STOCK_SCORES.labels(ticker=ticker).set(score)
                except (ValueError, TypeError) as cast_err:
                     logger.warning(f"Could not convert score '{raw_scores[i]}' to float for ticker {ticker}: {cast_err}")
                     scores[ticker] = 0.0
                     STOCK_SCORES.labels(ticker=ticker).set(0.0)

            total_latency = time.time() - start_time_scoring
            logger.info(f"Finished scoring {len(scores)} stocks in {total_latency:.4f}s (prediction took {predict_latency:.4f}s).")
            return scores

        except ModelError as me: # Catch specific error from sync function
             logger.error(f"ModelError during prediction: {me}")
             raise StockSelectionError("Failed to score stocks due to model prediction error") from me
        except Exception as e:
            logger.exception(f"Critical error during ML model prediction execution: {e}")
            raise StockSelectionError("Failed to score stocks due to model prediction error") from e


    async def refresh_watchlist(self) -> List[str]:
        """
        Refreshes the active watchlist based on selection criteria (e.g., top scores).

        Scores the current universe and filters based on score threshold and size limit.

        Returns:
            List of ticker symbols in the watchlist.
        """
        logger.info("Refreshing watchlist...")
        if not self.current_universe:
             # Attempt to build universe if empty
             await self.build_universe()
             if not self.current_universe:
                  logger.warning("Cannot refresh watchlist: Universe is empty.")
                  WATCHLIST_SIZE.set(0)
                  return []

        try:
            all_scores = await self.score_stocks(list(self.current_universe))
            if not all_scores:
                 logger.warning("Scoring returned no results. Watchlist will be empty.")
                 WATCHLIST_SIZE.set(0)
                 return []

            min_score_threshold = float(os.environ.get("WATCHLIST_MIN_SCORE", 0.7))
            potential_watchlist = {
                ticker: score for ticker, score in all_scores.items()
                if score >= min_score_threshold
            }

            watchlist_size_target = int(os.environ.get("WATCHLIST_SIZE_TARGET", 100))
            sorted_tickers = sorted(potential_watchlist, key=potential_watchlist.get, reverse=True)
            watchlist = sorted_tickers[:watchlist_size_target]

            WATCHLIST_SIZE.set(len(watchlist))
            logger.info(f"Watchlist refreshed. Size: {len(watchlist)}")
            return watchlist

        except Exception as e:
             logger.exception(f"Error refreshing watchlist: {e}")
             WATCHLIST_SIZE.set(0)
             return []


    async def get_focused_list(self) -> List[str]:
        """
        Gets a focused list of high-priority stocks (subset of watchlist).

        Currently uses top-M scoring from the watchlist as placeholder.

        Returns:
            List of ticker symbols in the focused list.
        """
        logger.info("Getting focused list...")

        try:
            current_watchlist = await self.refresh_watchlist()
            if not current_watchlist:
                 logger.warning("Cannot get focused list: Watchlist is empty.")
                 FOCUSED_LIST_SIZE.set(0)
                 return []

            # --- Placeholder Logic (Top M scores from watchlist) ---
            logger.warning("Using placeholder logic for get_focused_list (top M scores)!")
            # Re-score only the watchlist tickers to get their current scores
            watchlist_scores = await self.score_stocks(current_watchlist)

            if not watchlist_scores:
                 logger.warning("Scoring watchlist returned no results. Focused list will be empty.")
                 FOCUSED_LIST_SIZE.set(0)
                 return []

            focused_list_size_target = int(os.environ.get("FOCUSED_LIST_SIZE_TARGET", 10))
            # Sort the scored watchlist tickers
            sorted_tickers = sorted(watchlist_scores, key=watchlist_scores.get, reverse=True)
            focused_list = sorted_tickers[:focused_list_size_target]
            # --- End Placeholder Logic ---

            FOCUSED_LIST_SIZE.set(len(focused_list))
            logger.info(f"Focused list generated. Size: {len(focused_list)}")
            return focused_list

        except Exception as e:
             logger.exception(f"Error generating focused list: {e}")
             FOCUSED_LIST_SIZE.set(0)
             return []

    # Additional core selection methods can be added here
    # ...
