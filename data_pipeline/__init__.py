#!/usr/bin/env python3
"""
Data Pipeline Package

Provides a modular pipeline for financial data processing:
- Data loading from various sources
- Data preprocessing and feature engineering
- Technical indicator calculation
- Model training and evaluation
"""

from .base import DataPipeline
from .loading import (
    load_from_cache,
    save_to_cache,
    load_from_file,
    load_from_polygon_rest,
    load_from_polygon_ws,
    load_from_unusual_whales
)
from .processing import (
    clean_market_data,
    calculate_technical_indicators,
    normalize_features,
    add_time_features,
    detect_anomalies
)
from .modeling import (
    select_features_rfe,
    select_features_model_based,
    train_model,
    evaluate_model,
    create_time_series_splits
)

__all__ = [
    'DataPipeline',
    'load_from_cache',
    'save_to_cache',
    'load_from_file',
    'load_from_polygon_rest',
    'load_from_polygon_ws',
    'load_from_unusual_whales',
    'clean_market_data',
    'calculate_technical_indicators',
    'normalize_features',
    'add_time_features',
    'detect_anomalies',
    'select_features_rfe',
    'select_features_model_based',
    'train_model',
    'evaluate_model',
    'create_time_series_splits'
]
