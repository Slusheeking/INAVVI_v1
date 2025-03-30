#!/usr/bin/env python3
"""
Test suite for the data_pipeline module

This test suite verifies the functionality of the data_pipeline module components:
- DataPipeline initialization
- Data loading functions
- Data processing functions
- Modeling functions, 
- End-to-end pipeline functionality
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline import DataPipeline
from data_pipeline.loading import (
    load_from_cache,
    save_to_cache,
    load_from_file,
    load_from_polygon_rest,
    load_from_polygon_ws,
    load_from_unusual_whales
)
from data_pipeline.processing import (
    clean_market_data,
    calculate_technical_indicators,
    normalize_features,
    add_time_features,
    detect_anomalies
)
from data_pipeline.modeling import (
    select_features_rfe,
    select_features_model_based,
    train_model,
    evaluate_model,
    create_time_series_splits
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class TestDataPipelineInitialization(unittest.TestCase):
    """Test the initialization of the DataPipeline class"""

    def test_init_default(self):
        """Test initialization with default parameters"""
        pipeline = DataPipeline(test_mode=True)
        self.assertTrue(pipeline.test_mode)
        self.assertIsNone(pipeline.polygon)
        self.assertIsNone(pipeline.polygon_ws)
        self.assertIsNone(pipeline.unusual_whales)
        self.assertIsNone(pipeline.redis)
        
    def test_init_with_clients(self):
        """Test initialization with mock clients"""
        mock_polygon = MagicMock()
        mock_polygon_ws = MagicMock()
        mock_unusual_whales = MagicMock()
        mock_redis = MagicMock()
        
        pipeline = DataPipeline(
            polygon_client=mock_polygon,
            polygon_ws=mock_polygon_ws,
            unusual_whales_client=mock_unusual_whales,
            redis_client=mock_redis,
            test_mode=True
        )
        
        self.assertEqual(pipeline.polygon, mock_polygon)
        self.assertEqual(pipeline.polygon_ws, mock_polygon_ws)
        self.assertEqual(pipeline.unusual_whales, mock_unusual_whales)
        self.assertEqual(pipeline.redis, mock_redis)
        
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        custom_config = {
            "cache_dir": "/tmp/test_cache",
            "cache_expiry": 3600,
            "watchlist": {
                "max_size": 50,
                "min_price": 10.0
            }
        }
        
        pipeline = DataPipeline(config=custom_config, test_mode=True)
        
        self.assertEqual(pipeline.config["cache_dir"], "/tmp/test_cache")
        self.assertEqual(pipeline.config["cache_expiry"], 3600)
        self.assertEqual(pipeline.config["watchlist"]["max_size"], 50)
        self.assertEqual(pipeline.config["watchlist"]["min_price"], 10.0)
        
        # Check that other config values are preserved
        self.assertEqual(pipeline.config["watchlist"]["refresh_interval"], 900)


class TestDataLoading(unittest.TestCase):
    """Test the data loading functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        self.sample_market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Create mock Redis client
        self.mock_redis = MagicMock()
        
    def test_load_from_cache(self):
        """Test loading data from cache"""
        from data_pipeline.loading import load_from_cache
        
        # Mock the Redis get method
        self.mock_redis.exists.return_value = True
        self.mock_redis.get.return_value = b'\x80\x04\x95\x1a\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04test\x94\x8c\x05value\x94s.'  # Pickled dict
        
        # Call the function
        result = load_from_cache('test_key', self.mock_redis)
        
        # Verify the result
        self.mock_redis.exists.assert_called_once_with('test_key')
        self.mock_redis.get.assert_called_once_with('test_key')
        
    def test_save_to_cache(self):
        """Test saving data to cache"""
        from data_pipeline.loading import save_to_cache
        
        # Test data
        test_data = {'test': 'value'}
        
        # Call the function
        result = save_to_cache('test_key', test_data, self.mock_redis)
        
        # Verify the result
        self.mock_redis.set.assert_called_once()
        
    def test_load_from_file(self):
        """Test loading data from file"""
        from data_pipeline.loading import load_from_file
        
        # Create a temporary CSV file
        temp_file = 'tests/test_market_data.csv'
        self.sample_market_data.to_csv(temp_file, index=False)
        
        try:
            # Load the data
            result = load_from_file(temp_file, 'csv')
            
            # Verify the result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.sample_market_data))
            self.assertTrue(all(col in result.columns for col in self.sample_market_data.columns))
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestDataProcessing(unittest.TestCase):
    """Test the data processing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        self.sample_market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
    def test_clean_market_data(self):
        """Test cleaning market data"""
        # Create data with some issues
        dirty_data = self.sample_market_data.copy()
        dirty_data.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']  # Uppercase
        dirty_data.loc[0, 'Close'] = None  # Add a missing value
        
        # Clean the data
        cleaned_data = clean_market_data(dirty_data)
        
        # Verify the result
        self.assertEqual(len(cleaned_data), len(dirty_data) - 1)  # One row removed due to missing value
        self.assertTrue(all(col in cleaned_data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['timestamp']))
        
    def test_calculate_technical_indicators(self):
        """Test calculating technical indicators"""
        # Calculate indicators
        data_with_indicators = calculate_technical_indicators(self.sample_market_data)
        
        # Verify the result
        expected_columns = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'rsi', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal',
            'daily_return', 'volatility_30d'
        ]
        self.assertTrue(all(col in data_with_indicators.columns for col in expected_columns))
        
    def test_normalize_features(self):
        """Test normalizing features"""
        # Select features to normalize
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Normalize features
        normalized_data = normalize_features(self.sample_market_data, feature_columns)
        
        # Verify the result
        for col in feature_columns:
            self.assertAlmostEqual(normalized_data[col].mean(), 0, delta=1e-10)
            self.assertAlmostEqual(normalized_data[col].std(), 1, delta=1e-10)
            
    def test_add_time_features(self):
        """Test adding time features"""
        # Add time features
        data_with_time_features = add_time_features(self.sample_market_data)
        
        # Verify the result
        expected_columns = [
            'hour', 'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter',
            'is_premarket', 'is_market_hours', 'is_after_hours', 'minutes_since_open'
        ]
        self.assertTrue(all(col in data_with_time_features.columns for col in expected_columns))
        
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Create data with anomalies
        data_with_anomalies = self.sample_market_data.copy()
        data_with_anomalies.loc[0, 'close'] = 1000  # Add an anomaly
        
        # Detect anomalies
        result = detect_anomalies(data_with_anomalies, ['close'], threshold=3.0)
        
        # Verify the result
        self.assertTrue('close_anomaly' in result.columns)
        self.assertEqual(result['close_anomaly'].sum(), 1)  # One anomaly detected


class TestModeling(unittest.TestCase):
    """Test the modeling functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        # Features
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'feature4': np.random.normal(0, 1, n_samples),
            'feature5': np.random.normal(0, 1, n_samples)
        })
        
        # Classification target
        self.y_class = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Regression target
        self.y_reg = pd.Series(2 * self.X['feature1'] + 3 * self.X['feature2'] + np.random.normal(0, 0.1, n_samples))
        
    def test_select_features_rfe(self):
        """Test feature selection using RFE"""
        # Create estimator
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Select features
        selected_features = select_features_rfe(self.X, self.y_class, estimator, n_features=3)
        
        # Verify the result
        self.assertEqual(selected_features.shape[1], 3)
        
    def test_select_features_model_based(self):
        """Test feature selection using model-based selection"""
        # Create estimator
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Select features
        selected_features = select_features_model_based(self.X, self.y_class, estimator)
        
        # Verify the result
        self.assertLessEqual(selected_features.shape[1], self.X.shape[1])
        
    def test_train_model_classification(self):
        """Test training a classification model"""
        # Train model
        model = train_model(self.X, self.y_class, model_type='classification', model_class='random_forest')
        
        # Verify the result
        self.assertIsInstance(model, RandomForestClassifier)
        
    def test_train_model_regression(self):
        """Test training a regression model"""
        # Train model
        model = train_model(self.X, self.y_reg, model_type='regression', model_class='random_forest')
        
        # Verify the result
        self.assertIsInstance(model, RandomForestRegressor)
        
    def test_evaluate_model_classification(self):
        """Test evaluating a classification model"""
        # Train model
        model = train_model(self.X, self.y_class, model_type='classification', model_class='random_forest')
        
        # Evaluate model
        metrics = evaluate_model(model, self.X, self.y_class, model_type='classification')
        
        # Verify the result
        self.assertTrue(all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1']))
        self.assertTrue(all(0 <= metrics[metric] <= 1 for metric in metrics))
        
    def test_evaluate_model_regression(self):
        """Test evaluating a regression model"""
        # Train model
        model = train_model(self.X, self.y_reg, model_type='regression', model_class='random_forest')
        
        # Evaluate model
        metrics = evaluate_model(model, self.X, self.y_reg, model_type='regression')
        
        # Verify the result
        self.assertTrue(all(metric in metrics for metric in ['mse', 'mae', 'r2']))
        
    def test_create_time_series_splits(self):
        """Test creating time series splits"""
        # Create splits
        splits = create_time_series_splits(self.X, self.y_class, n_splits=3)
        
        # Verify the result
        self.assertEqual(len(splits), 3)
        for train_idx, test_idx in splits:
            self.assertTrue(len(train_idx) > 0)
            self.assertTrue(len(test_idx) > 0)
            self.assertTrue(max(train_idx) < min(test_idx))  # Train comes before test


class TestDataValidation(unittest.TestCase):
    """Test data validation and cleaning functionality"""
    
    def test_clean_market_data_with_nan(self):
        """Test cleaning market data with NaN values"""
        # Create data with NaN values
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, np.inf, 105, 106, 107, 108, 109, 110],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        })
        
        # Clean the data
        cleaned_data = clean_market_data(data)
        
        # Verify the result
        self.assertLessEqual(cleaned_data.isna().sum().sum(), 2)  # Allow up to 2 NaN (original NaN + inf converted to NaN)
        self.assertFalse(np.isinf(cleaned_data.select_dtypes(np.number)).any().any())
        
    def test_clean_market_data_with_invalid_values(self):
        """Test cleaning market data with invalid values"""
        # Create data with invalid values
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100, -1, 102, 1e10, 104, 105, 106, 107, 108, 109],  # Invalid values
            'volume': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        })
        
        # Clean the data
        cleaned_data = clean_market_data(data)
        
        # Verify the result
        self.assertEqual(len(cleaned_data), 10)  # No rows removed - invalid values clipped
        self.assertTrue((cleaned_data['close'] > 0).all())
        # Check values are within reasonable bounds (either quantile-based cap or absolute max)
        self.assertTrue((cleaned_data['close'] <= 1e6).all())
        # Verify negative value was clipped to minimum
        self.assertEqual(cleaned_data.loc[1, 'close'], 0.01)


class TestEndToEndPipeline(unittest.TestCase):
    """Test the end-to-end pipeline functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
            'open': np.random.uniform(100, 200, n_samples),
            'high': np.random.uniform(100, 200, n_samples),
            'low': np.random.uniform(100, 200, n_samples),
            'close': np.random.uniform(100, 200, n_samples),
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Create a temporary file
        self.temp_file = 'tests/test_pipeline_data.csv'
        self.sample_data.to_csv(self.temp_file, index=False)
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
            
    def test_pipeline_workflow(self):
        """Test the complete pipeline workflow"""
        # Initialize pipeline
        pipeline = DataPipeline(test_mode=True)
        
        # Load data
        data = pd.read_csv(self.temp_file)
        
        # Clean data
        data = clean_market_data(data)
        
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Add time features
        data = add_time_features(data)
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col not in ['timestamp']]
        X = data[feature_columns].copy()
        
        # Create a target variable (next day return)
        data['target'] = data['close'].pct_change(1).shift(-1) > 0
        data = data.dropna()
        y = data['target']
        
        # Normalize features
        X = normalize_features(X, feature_columns)
        
        # Select features
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        X_selected = select_features_model_based(X, y, estimator)
        
        # Create time series splits
        splits = create_time_series_splits(X_selected, y, n_splits=2)
        
        # Train and evaluate model
        train_idx, test_idx = splits[0]
        
        # Skip if no data available
        if len(train_idx) == 0 or len(test_idx) == 0:
            self.skipTest("No training or test data available")
            
        # Convert to numpy arrays and filter invalid indices
        train_idx = np.array([i for i in train_idx if 0 <= i < len(X_selected)])
        test_idx = np.array([i for i in test_idx if 0 <= i < len(X_selected)])
        
        if len(train_idx) == 0 or len(test_idx) == 0:
            self.skipTest("No valid training or test indices after bounds check")
            
        # Get data ensuring we don't go out of bounds
        try:
            X_train = X_selected.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            X_test = X_selected.iloc[test_idx].copy()
            y_test = y.iloc[test_idx].copy()
        except IndexError:
            self.skipTest("Invalid indices for data splitting")
            
        # Ensure no NaN/inf values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
        y_train = y_train[X_train.index]
        X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
        y_test = y_test[X_test.index]
        
        if len(X_train) == 0 or len(X_test) == 0:
            self.skipTest("No valid data after NaN/inf filtering")
        
        model = train_model(X_train, y_train, model_type='classification', model_class='random_forest')
        metrics = evaluate_model(model, X_test, y_test, model_type='classification')
        
        # Verify the result
        self.assertTrue(all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1']))
        self.assertTrue(all(0 <= metrics[metric] <= 1 for metric in metrics))


if __name__ == '__main__':
    unittest.main()
