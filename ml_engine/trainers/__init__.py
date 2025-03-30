#!/usr/bin/env python3
"""
ML Model Trainers Package

This package provides specialized trainers for different model types:
1. SignalDetectionTrainer: For binary classification of trading signals
2. PricePredictionTrainer: For time series forecasting of price movements
3. RiskAssessmentTrainer: For regression of volatility and risk metrics
4. ExitStrategyTrainer: For regression of optimal exit points
5. MarketRegimeTrainer: For clustering of market states

All trainers inherit from BaseTrainer, which provides common functionality
for model training, evaluation, and persistence.
"""

from ml_engine.trainers.base import BaseTrainer
from ml_engine.trainers.signal_detection import SignalDetectionTrainer
from ml_engine.trainers.price_prediction import PricePredictionTrainer
from ml_engine.trainers.risk_assessment import RiskAssessmentTrainer
from ml_engine.trainers.exit_strategy import ExitStrategyTrainer
from ml_engine.trainers.market_regime import MarketRegimeTrainer

# Map of model types to trainer classes
TRAINER_MAP = {
    "signal_detection": SignalDetectionTrainer,
    "price_prediction": PricePredictionTrainer,
    "risk_assessment": RiskAssessmentTrainer,
    "exit_strategy": ExitStrategyTrainer,
    "market_regime": MarketRegimeTrainer,
}

# Version
__version__ = "1.0.0"