"""
Signal Generator for the AI Day Trading Bot.

Combines signals from rule-based strategies (DayTradingStrategies)
and ML models (MLEnginePredictor).
"""

import logging
import asyncio # Add asyncio back
from typing import List, Dict, Optional, Any, TYPE_CHECKING, Tuple # Import Tuple

# Import necessary components
from ai_day_trader.utils.config import Config # Use new utils path
from ai_day_trader.trading.day_trading import DayTradingStrategies # Use new trading path
# Import predictor from new location
from ai_day_trader.ml.predictor import Predictor as MLEnginePredictor
from ai_day_trader.feature_calculator import FeatureCalculator
from redis.asyncio import Redis # Import directly

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generates trading signals from various sources."""

    def __init__(
        self,
        config: Config,
        day_trading_strategies: Optional[DayTradingStrategies],
        redis_client: Optional[Redis], # Use Optional[Redis]
        ml_predictor: Optional[MLEnginePredictor],
        feature_calculator: Optional[FeatureCalculator] # Keep for potential future use or direct calls if needed
    ):
        """Initialize the SignalGenerator."""
        self.config = config
        self.day_trading_strategies = day_trading_strategies
        self.redis_client = redis_client
        self.ml_predictor = ml_predictor
        self.feature_calculator = feature_calculator # Store reference
        self.ml_confidence_threshold = config.get_float("SIGNAL_CONFIDENCE_THRESHOLD", 0.6)

        if not self.day_trading_strategies:
            logger.warning("DayTradingStrategies not provided. Rule-based signals disabled.")
        if not self.ml_predictor:
            logger.warning("MLEnginePredictor not provided. ML-based signals disabled.")
        # Feature calculator is needed by the predictor internally if features aren't passed
        # if not self.feature_calculator:
        #      logger.warning("FeatureCalculator not provided. ML-based signals may fail.")

        logger.info("SignalGenerator initialized.")

    async def generate_signals(self, symbols_to_scan: List[str], latest_features_map: Dict[str, Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Generates trading signals for the provided list of symbols, using pre-calculated features.

        Args:
            symbols_to_scan: List of ticker symbols to generate signals for.
            latest_features_map: Dictionary mapping symbols to their pre-calculated features.

        Returns:
            A list of signal dictionaries. Each dictionary should contain
            at least 'symbol', 'side' ('buy' or 'sell'), and 'source'.
            Optionally includes 'confidence', 'price', etc.
        """
        signals = []
        logger.info(f"Generating signals for {len(symbols_to_scan)} symbols...")

        # 1. Rule-Based Signals (if available)
        if self.day_trading_strategies:
            logger.debug(f"Generating rule-based signals for {len(symbols_to_scan)} symbols...")
            # Consider running these concurrently as well if they are slow
            rule_tasks = []
            symbols_for_rules = []
            for symbol in symbols_to_scan:
                 # Example: Add tasks for each rule-based strategy check
                 rule_tasks.append(self.day_trading_strategies.run_opening_range_breakout(symbol))
                 symbols_for_rules.append(symbol) # Track symbol for result mapping if needed
                 rule_tasks.append(self.day_trading_strategies.run_vwap_reversion(symbol))
                 symbols_for_rules.append(symbol) # Track symbol again

            if rule_tasks:
                rule_results = await asyncio.gather(*rule_tasks, return_exceptions=True)
                # Process results - need careful mapping back to symbols if multiple rules per symbol
                # Simplified: just append non-None, non-Exception results
                for result in rule_results:
                    if result and not isinstance(result, Exception):
                        signals.append(result)
                    elif isinstance(result, Exception):
                         logger.error(f"Error generating rule-based signal: {result}", exc_info=False)


        # 2. ML-Based Signals (if available)
        if self.ml_predictor:
            logger.debug(f"Generating ML-based signals for {len(symbols_to_scan)} symbols...")
            ml_tasks = []
            symbols_for_ml = []
            for symbol in symbols_to_scan:
                # Get pre-calculated features for the symbol
                features = latest_features_map.get(symbol)
                if features: # Only attempt prediction if features were successfully calculated
                    # Assuming predict_entry now accepts features
                    ml_tasks.append(self.ml_predictor.predict_entry(symbol, features))
                    symbols_for_ml.append(symbol)
                else:
                    logger.debug(f"Skipping ML prediction for {symbol}: Missing pre-calculated features.")

            if ml_tasks:
                ml_results = await asyncio.gather(*ml_tasks, return_exceptions=True)

                for i, symbol in enumerate(symbols_for_ml):
                    prediction_result = ml_results[i]
                    if isinstance(prediction_result, Exception):
                        logger.error(f"Error generating ML signal for {symbol}: {prediction_result}", exc_info=False)
                        continue
                    elif isinstance(prediction_result, dict):
                        pred = prediction_result.get('prediction')
                        conf = prediction_result.get('probability', prediction_result.get('confidence'))

                        if pred is not None and conf is not None and conf >= self.ml_confidence_threshold:
                            side = 'buy' if pred == 1 else ('sell' if pred == -1 else None)
                            if side:
                                signals.append({
                                    'symbol': symbol, 'side': side,
                                    'source': 'MLModel', 'confidence': float(conf)
                                })
                        elif pred is not None and conf is None: # Handle case where only prediction is returned
                             side = 'buy' if pred == 1 else ('sell' if pred == -1 else None)
                             if side:
                                 signals.append({
                                     'symbol': symbol, 'side': side,
                                     'source': 'MLModel', 'confidence': 1.0 # Assign default confidence?
                                 })

        # 3. Combine and Prioritize Signals
        final_signals = []
        grouped_signals: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

        # Group signals by (symbol, side)
        for signal in signals:
            symbol = signal.get('symbol')
            side = signal.get('side')
            if symbol and side:
                key = (symbol, side)
                if key not in grouped_signals:
                    grouped_signals[key] = []
                grouped_signals[key].append(signal)

        # Prioritize within each group
        for key, group in grouped_signals.items():
            symbol, side = key
            best_ml_signal: Optional[Dict[str, Any]] = None
            rule_signals: List[Dict[str, Any]] = []

            for signal in group:
                source = signal.get('source')
                if source == 'MLModel':
                    confidence = signal.get('confidence', 0.0)
                    if best_ml_signal is None or confidence > best_ml_signal.get('confidence', 0.0):
                        best_ml_signal = signal
                else: # Assume rule-based
                    rule_signals.append(signal)

            # Apply prioritization logic
            chosen_signal = None
            if best_ml_signal and best_ml_signal.get('confidence', 0.0) >= self.ml_confidence_threshold:
                chosen_signal = best_ml_signal
                logger.debug(f"Prioritizing high-confidence ML signal for {symbol}/{side}.")
            elif rule_signals:
                chosen_signal = rule_signals[0] # Take the first rule signal encountered
                logger.debug(f"Using rule-based signal for {symbol}/{side} (ML signal absent or below threshold).")
            elif best_ml_signal:
                 logger.debug(f"Discarding low-confidence ML signal for {symbol}/{side} (Confidence: {best_ml_signal.get('confidence', 0.0)} < {self.ml_confidence_threshold}).")
            # Else: No usable signal in this group

            if chosen_signal:
                final_signals.append(chosen_signal)


        if final_signals:
            logger.info(f"Generated {len(final_signals)} final signals after prioritization.")
            for i, sig in enumerate(final_signals[:5]): logger.debug(f"Final Signal {i+1}: {sig}")
        else:
             logger.info("No entry signals generated in this cycle.")

        return final_signals
