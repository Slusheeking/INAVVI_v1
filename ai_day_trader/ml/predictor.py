"""
ML Model Predictor for AI Day Trader.

Loads trained models and artifacts, generates features using FeatureCalculator,
and makes predictions for entry and exit signals.
"""

import logging
import time
import joblib
import json
# import numpy as np # Removed unused import
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

# Import from new locations within ai_day_trader
from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.feature_calculator import FeatureCalculator
from ai_day_trader.clients.polygon_rest_client import PolygonRESTClient
# Import XGBoost integration from within the new ml package
from .xgboost_integration import load_xgboost_model, predict_with_xgboost

# Define aliases, defaulting to Any
RedisClientTypeAlias = Any
PositionInfoTypeAlias = Any

if TYPE_CHECKING:
    try:
        # Attempt to import the specific type and reassign alias
        from redis.asyncio import Redis as RedisClientTypeAlias
    except ImportError:
        # If import fails, alias remains Any (already set)
        pass
    try:
        # Attempt to import the specific type and reassign alias
        from ai_day_trader.trading.execution.base import PositionInfo as PositionInfoTypeAlias
    except ImportError:
         # If import fails, alias remains Any (already set)
        pass

logger = logging.getLogger(__name__)

class Predictor:
    """Handles loading models and making predictions."""

    def __init__(
        self,
        config: Config,
        polygon_rest_client: PolygonRESTClient,
        feature_calculator: FeatureCalculator,
        redis_client: Optional[RedisClientTypeAlias] = None # Optional Redis for potential future use
    ):
        self.config = config
        self.rest_client = polygon_rest_client
        self.feature_calculator = feature_calculator
        self.redis_client = redis_client # Store if needed later
        # Ensure the path passed to Path is a string
        self.model_dir = Path(str(config.get("MODEL_DIR", "./models")))
        # Get lookback from config, default to 120
        self.lookback_minutes = int(config.get("PREDICTOR_LOOKBACK_MINUTES", 120))
        # Add a buffer to the limit, could be configurable too
        self.lookback_limit_buffer = int(config.get("PREDICTOR_LOOKBACK_BUFFER", 50))
        # Cache TTL in seconds
        self.cache_ttl_seconds = int(config.get("PREDICTOR_CACHE_TTL_SECONDS", 60))


        self.entry_model, self.entry_scaler, self.entry_features = None, None, []
        self.exit_model, self.exit_scaler, self.exit_features = None, None, []

        logger.info(
            f"Predictor initialized. Model directory: {self.model_dir}, "
            f"Lookback: {self.lookback_minutes} mins, Cache TTL: {self.cache_ttl_seconds}s"
        )

    def _load_model_artifacts(self, model_type: str) -> tuple[Any, Any, List[str]]:
        """Loads model, scaler, and features for a given type ('entry' or 'exit')."""
        model_name = f"{model_type}_signal_model.xgb"
        scaler_name = f"{model_type}_signal_scaler.pkl"
        features_name = f"{model_type}_signal_features.json"

        model_path = self.model_dir / model_name
        scaler_path = self.model_dir / scaler_name
        features_path = self.model_dir / features_name

        model, scaler, features = None, None, []

        if model_path.exists() and scaler_path.exists() and features_path.exists():
            try:
                # Convert Path to string for load_xgboost_model
                model = load_xgboost_model(str(model_path))
                scaler = joblib.load(scaler_path)
                with open(features_path, 'r') as f:
                    features = json.load(f)
                logger.info(f"{model_type.capitalize()} model loaded successfully. Features: {len(features)}")
            except Exception as e:
                 logger.error(f"Error loading {model_type} model artifacts from {self.model_dir}: {e}", exc_info=True)
                 # Ensure partial loads don't leave inconsistent state
                 model, scaler, features = None, None, []
        else:
            missing = []
            if not model_path.exists(): missing.append(model_name)
            if not scaler_path.exists(): missing.append(scaler_name)
            if not features_path.exists(): missing.append(features_name)
            logger.warning(f"{model_type.capitalize()} model artifacts not found ({', '.join(missing)}). {model_type.capitalize()} prediction disabled.")

        return model, scaler, features

    async def load_models(self):
        """Loads entry and exit models, scalers, and feature lists."""
        logger.info(f"Loading ML models and artifacts from {self.model_dir}...")
        try:
            self.entry_model, self.entry_scaler, self.entry_features = self._load_model_artifacts("entry")
            self.exit_model, self.exit_scaler, self.exit_features = self._load_model_artifacts("exit")

        except FileNotFoundError as e: # Keep top-level for catastrophic path issues maybe? Though handled inside now.
            logger.error(f"Model directory issue? Artifact not found: {e}. Ensure models are trained and placed in {self.model_dir}")
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            # Reset models on error
            self.entry_model, self.entry_scaler, self.entry_features = self._load_model_artifacts("entry")
            self.exit_model, self.exit_scaler, self.exit_features = self._load_model_artifacts("exit")

        except FileNotFoundError as e: # Keep top-level for catastrophic path issues maybe? Though handled inside now.
            logger.error(f"Model directory issue? Artifact not found: {e}. Ensure models are trained and placed in {self.model_dir}")
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            # Reset models on error - Ensure consistency
            self.entry_model, self.entry_scaler, self.entry_features = None, None, []
            self.exit_model, self.exit_scaler, self.exit_features = None, None, []


    async def _get_and_prepare_features(self, symbol: str, model_type: str, required_features: List[str], scaler: Any) -> Optional[pd.DataFrame]:
        """
        Fetches data, calculates features, selects required features, scales them,
        and caches the result. Returns the latest row of scaled features.
        """
        # Basic check for model artifacts being loaded for this prediction type
        if not required_features or not scaler:
             logger.error(f"Model features or scaler not loaded for {symbol} ({model_type}).")
             return None
        # self.feature_calculator and self.rest_client are assumed to be initialized

        # --- Cache Check ---
        cache_key = f"predictor:features:{symbol}:{model_type}"
        if self.redis_client and self.cache_ttl_seconds > 0:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    logger.debug(f"Cache hit for {cache_key}")
                    # Deserialize from JSON string stored in Redis
                    features_json = json.loads(cached_data)
                    # Convert back to DataFrame, handling potential timestamp index
                    cached_df = pd.read_json(features_json, orient='split')
                    # Ensure index is DatetimeIndex if it was originally
                    if 'index' in features_json and all(isinstance(i, int) for i in features_json['index']):
                         cached_df.index = pd.to_datetime(cached_df.index, unit='ms', utc=True) # Assuming ms timestamp stored
                    return cached_df
            except Exception as e:
                logger.warning(f"Redis cache GET error for key {cache_key}: {e}", exc_info=True)
        # --- End Cache Check ---

        try:
            # Determine precise lookback needed based on the required features for this model
            try:
                # Assume FeatureCalculator has this method; handle potential AttributeError
                lookback_needed = self.feature_calculator.get_required_lookback(required_features)
                # Use the greater of the calculated needed lookback or the configured minimum,
                # ensuring we fetch enough data but not less than a baseline if configured.
                lookback_to_fetch = max(lookback_needed, self.lookback_minutes)
                logger.debug(f"Required lookback for features: {lookback_needed}, Using fetch lookback: {lookback_to_fetch}")
            except AttributeError:
                logger.warning("FeatureCalculator does not have 'get_required_lookback' method. Falling back to configured lookback.")
                lookback_to_fetch = self.lookback_minutes
            except Exception as e:
                 logger.error(f"Error getting required lookback from FeatureCalculator: {e}. Falling back to configured lookback.", exc_info=True)
                 lookback_to_fetch = self.lookback_minutes

            limit_to_fetch = lookback_to_fetch + self.lookback_limit_buffer

            # Fetch historical data
            logger.debug(f"Fetching {limit_to_fetch} bars ({lookback_to_fetch} mins lookback) for {symbol}")
            hist_data = await self.rest_client.get_aggregates(
                symbol=symbol,
                multiplier=1,
                timespan="minute",
                # Use 'limit' to control the number of bars fetched
                # 'minutes' parameter might behave differently or not be supported by all clients/APIs
                limit=limit_to_fetch
             )

            # Check fetched data
            if not isinstance(hist_data, pd.DataFrame) or hist_data.empty:
                logger.warning(f"Could not fetch sufficient historical data (or data is not DataFrame) for {symbol} features.")
                return None

            # Calculate all features
            features_df = await self.feature_calculator.calculate_features(symbol, hist_data)

            # Add type check for features_df
            if not isinstance(features_df, pd.DataFrame) or features_df.empty:
                 logger.warning(f"Feature calculation failed (or result is not DataFrame) for {symbol}.")
                 return None

            # Select only the features required by the specific model
            missing_req_features = [f for f in required_features if f not in features_df.columns]
            if missing_req_features:
                 logger.error(f"Missing required features for {symbol} model: {missing_req_features}")
                 return None

            features_selected = features_df[required_features]

            # Scale features
            features_scaled = scaler.transform(features_selected)
            features_scaled_df = pd.DataFrame(features_scaled, index=features_selected.index, columns=required_features)

            # Return only the latest row of scaled features for prediction
            latest_features = features_scaled_df.iloc[[-1]]

            # --- Cache Store ---
            if self.redis_client and self.cache_ttl_seconds > 0:
                try:
                    # Serialize DataFrame to JSON string for Redis
                    # orient='split' is generally good for preserving index and dtypes
                    # Convert Timestamp index to integer (milliseconds since epoch) for JSON compatibility
                    features_to_cache = latest_features.copy()
                    if isinstance(features_to_cache.index, pd.DatetimeIndex):
                         features_to_cache.index = features_to_cache.index.astype(int) // 10**6 # Convert ns to ms

                    features_json = features_to_cache.to_json(orient='split')
                    await self.redis_client.set(cache_key, features_json, ex=self.cache_ttl_seconds)
                    logger.debug(f"Stored features in cache for {cache_key} with TTL {self.cache_ttl_seconds}s")
                except Exception as e:
                    logger.warning(f"Redis cache SET error for key {cache_key}: {e}", exc_info=True)
            # --- End Cache Store ---

            return latest_features

        except Exception as e:
            logger.error(f"Error getting/preparing features for {symbol} ({model_type}): {e}", exc_info=True)
            return None

    async def predict_entry(self, symbol: str, features: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Predicts entry signal for a symbol, optionally using pre-calculated features.

        Args:
            symbol: The stock symbol.
            features: Optional dictionary of pre-calculated features for the latest timestamp.

        Returns:
            Dictionary with 'prediction' and 'probability', or None on failure.
        """
        if not self.entry_model or not self.entry_features or not self.entry_scaler:
            # logger.debug(f"Entry model not loaded for {symbol}. Skipping prediction.")
            return None

        start_time = time.monotonic()
        features_scaled_df = None

        if features:
            logger.debug(f"Using pre-calculated features for {symbol} entry prediction.")
            try:
                # Convert dict to DataFrame for scaling
                features_df = pd.DataFrame([features]) # Create a single-row DataFrame
                # Select required features and ensure order
                missing_req_features = [f for f in self.entry_features if f not in features_df.columns]
                if missing_req_features:
                     logger.error(f"Pre-calculated features for {symbol} missing required entry features: {missing_req_features}")
                     return None
                features_selected = features_df[self.entry_features]
                # Scale features
                features_scaled = self.entry_scaler.transform(features_selected)
                features_scaled_df = pd.DataFrame(features_scaled, index=features_selected.index, columns=self.entry_features)
            except Exception as e:
                logger.error(f"Error processing pre-calculated features for {symbol} entry: {e}", exc_info=True)
                return None
        else:
            # Fallback: Fetch and prepare features if not provided (legacy path)
            logger.warning(f"No pre-calculated features provided for {symbol} entry. Fetching manually.")
            # Pass model_type for caching
            features_scaled_df = await self._get_and_prepare_features(symbol, "entry", self.entry_features, self.entry_scaler)

        if features_scaled_df is None or features_scaled_df.empty:
            logger.warning(f"Could not get or process features for entry prediction on {symbol}.")
            return None

        try:
            # Use XGBoost prediction function with the prepared scaled features
            prediction, probability = predict_with_xgboost(self.entry_model, features_scaled_df)

            # Add check for None before subscripting
            if prediction is None or probability is None or len(prediction) == 0 or len(probability) == 0:
                 logger.error(f"Prediction or probability result is invalid for {symbol}.")
                 return None

            # Assuming binary classification: 0=hold/sell, 1=buy
            # Or potentially: -1=sell, 0=hold, 1=buy
            # Adapt based on actual model training labels
            pred_label = int(prediction[0]) # Get the single prediction value
            prob_label = float(probability[0]) # Get the single probability

            processing_time = (time.monotonic() - start_time) * 1000
            logger.debug(f"Entry prediction for {symbol}: Label={pred_label}, Prob={prob_label:.4f} ({processing_time:.2f} ms)")

            return {"prediction": pred_label, "probability": prob_label}

        except Exception as e:
            logger.error(f"Error during entry prediction for {symbol}: {e}", exc_info=True)
            return None

    async def predict_exit(self, symbol: str, position_info: PositionInfoTypeAlias, base_features: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Predicts exit signal for an existing position, optionally using pre-calculated base features.

        Args:
            symbol: The stock symbol.
            position_info: Information about the current position.
            base_features: Optional dictionary of pre-calculated base features (before position-specific ones).

        Returns:
            Dictionary with 'prediction' and 'probability', or None on failure.
        """
        if not self.exit_model or not self.exit_features or not self.exit_scaler:
            # logger.debug(f"Exit model not loaded for {symbol}. Skipping prediction.")
            return None

        start_time = time.monotonic()
        features_scaled_df = None

        if base_features:
            logger.debug(f"Using pre-calculated base features for {symbol} exit prediction.")
            try:
                # Convert dict to DataFrame for scaling
                features_df = pd.DataFrame([base_features]) # Create a single-row DataFrame
                # Select required features and ensure order - Use self.exit_features
                missing_req_features = [f for f in self.exit_features if f not in features_df.columns and f not in ['holding_time_minutes', 'unrealized_pnl_pct']] # Exclude position-specific features for now
                if missing_req_features:
                     logger.error(f"Pre-calculated base features for {symbol} missing required exit features: {missing_req_features}")
                     return None
                # Select only the base features that are present and required for scaling
                base_features_to_scale = [f for f in self.exit_features if f in features_df.columns and f not in ['holding_time_minutes', 'unrealized_pnl_pct']]
                features_selected = features_df[base_features_to_scale]
                # Scale features
                features_scaled = self.exit_scaler.transform(features_selected)
                features_scaled_df = pd.DataFrame(features_scaled, index=features_selected.index, columns=base_features_to_scale)
            except Exception as e:
                logger.error(f"Error processing pre-calculated base features for {symbol} exit: {e}", exc_info=True)
                return None
        else:
            # Fallback: Fetch and prepare features if not provided
            logger.warning(f"No pre-calculated base features provided for {symbol} exit. Fetching manually.")
            features_scaled_df = await self._get_and_prepare_features(symbol, "exit", self.exit_features, self.exit_scaler) # This scales all required features

        if features_scaled_df is None or features_scaled_df.empty:
            logger.warning(f"Could not get or process base features for exit prediction on {symbol}.")
            return None

        try:
            # --- Add Position-Specific Features (if required by the model) ---
            # These features are calculated based on the current position and potentially the *unscaled* base features
            # or the latest price. They are typically NOT scaled with the base features.
            final_feature_vector = features_scaled_df.iloc[0].to_dict() # Start with scaled base features

            if 'holding_time_minutes' in self.exit_features:
                try:
                    # Use the timestamp from the latest feature data as 'now'
                    # Need the original unscaled data or fetch latest time if not available
                    # For simplicity, let's assume we can get entry_time from position_info
                    entry_time_str = getattr(position_info, 'entry_time', None) # Adapt attribute name
                    if entry_time_str:
                        entry_time = pd.to_datetime(entry_time_str, utc=True) # Assume stored as UTC string
                        now_utc = pd.Timestamp.utcnow() # Use current time
                        holding_duration_minutes = (now_utc - entry_time).total_seconds() / 60
                        final_feature_vector['holding_time_minutes'] = holding_duration_minutes
                        logger.debug(f"Added holding_time_minutes: {holding_duration_minutes:.2f} for {symbol}")
                    else:
                        logger.warning(f"'entry_time' not found in position_info for {symbol}. Cannot calculate 'holding_time_minutes'.")
                        final_feature_vector['holding_time_minutes'] = 0 # Default or handle as needed
                except Exception as e:
                    logger.error(f"Error calculating holding_time_minutes for {symbol}: {e}", exc_info=True)
                    final_feature_vector['holding_time_minutes'] = 0 # Default on error


            if 'unrealized_pnl_pct' in self.exit_features:
                try:
                    entry_price_str = getattr(position_info, 'avg_entry_price', None) # Adapt attribute name
                    current_price_str = getattr(position_info, 'current_price', None) # Adapt attribute name
                    if entry_price_str and current_price_str:
                        entry_price = float(entry_price_str)
                        current_price = float(current_price_str)
                        if entry_price != 0:
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                            final_feature_vector['unrealized_pnl_pct'] = pnl_pct
                            logger.debug(f"Added unrealized_pnl_pct: {pnl_pct:.2f}% for {symbol}")
                        else:
                            final_feature_vector['unrealized_pnl_pct'] = 0.0
                    else:
                        logger.warning(f"Missing entry_price or current_price in position_info for {symbol}. Cannot calculate 'unrealized_pnl_pct'.")
                        final_feature_vector['unrealized_pnl_pct'] = 0.0
                except Exception as e:
                    logger.error(f"Error calculating unrealized_pnl_pct for {symbol}: {e}", exc_info=True)
                    final_feature_vector['unrealized_pnl_pct'] = 0.0

            # Convert the final feature vector (dict) into a DataFrame for prediction
            # Ensure columns are in the order expected by the model (self.exit_features)
            try:
                final_features_df_pred = pd.DataFrame([final_feature_vector])[self.exit_features]
            except KeyError as e:
                 logger.error(f"Missing feature in final vector for exit prediction {symbol}: {e}. Vector: {final_feature_vector.keys()}")
                 return None
            # --------------------------------------------------------------------

            # Use XGBoost prediction function
            prediction, probability = predict_with_xgboost(self.exit_model, final_features_df_pred)

            # Add check for None before subscripting
            if prediction is None or probability is None or len(prediction) == 0 or len(probability) == 0:
                 logger.error(f"Prediction or probability result is invalid for exit signal on {symbol}.")
                 return None

            # Assuming binary classification: 0=hold, 1=exit
            pred_label = int(prediction[0])
            prob_label = float(probability[0])

            processing_time = (time.monotonic() - start_time) * 1000
            logger.debug(f"Exit prediction for {symbol}: Label={pred_label}, Prob={prob_label:.4f} ({processing_time:.2f} ms)")

            # Return probability of *exit* signal (assuming 1 means exit)
            return {"prediction": pred_label, "probability": prob_label}

        except Exception as e:
            logger.error(f"Error during exit prediction for {symbol}: {e}", exc_info=True)
            return None
