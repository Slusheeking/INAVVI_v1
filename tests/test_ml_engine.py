#!/usr/bin/env python
"""
ML Engine Test Script

This script demonstrates how to use the ml_engine module:
1. Initializes the ML engine components
2. Loads sample data
3. Trains a simple model
4. Makes predictions
5. Evaluates results

This test suite ensures that the ml_engine module is production-ready by:
1. Validating model training and prediction functionality
2. Testing error handling and edge cases
3. Verifying model persistence and loading
4. Ensuring compatibility with GPU acceleration
"""

import json
import os
import sys
import shutil
import pandas as pd
import atexit
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Import ml_engine components
from ml_engine.base import MLModelTrainer
from ml_engine.data_processor import MLDataProcessor
from ml_engine.indicators import calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_rsi, calculate_atr, calculate_stochastic
from ml_engine.batch_processor import BatchProcessor
from data_pipeline.processing import normalize_features, add_time_features  # Import from data_pipeline instead of ml_engine
from ml_engine.explainability import ModelExplainer
from ml_engine.trainers.signal_detection import SignalDetectionTrainer
from ml_engine.trainers.market_regime import MarketRegimeTrainer
from ml_engine.trainers.price_prediction import PricePredictionTrainer
from ml_engine.trainers.risk_assessment import RiskAssessmentTrainer
from ml_engine.trainers.exit_strategy import ExitStrategyTrainer
from ml_engine.xgboost_model import XGBoostModel
from ml_engine.xgboost_integration import XGBoostIntegration
from ml_engine.model_cache import MODEL_CACHE, get_model, invalidate_model, get_model_metadata, get_cache_stats, cleanup_cache
from utils.logging_config import get_logger
from utils.exceptions import ModelTrainingError, DataProcessingError

# Configure logging
logger = get_logger("test_ml_engine")

# Test directories
TEST_MODELS_DIR = "./test_models"
TEST_MONITORING_DIR = "./test_monitoring"
TEST_DATA_DIR = "./test_data"

# Setup test directories and sample data
def test_directories():
    """Create and clean up test directories"""
    os.makedirs(TEST_MODELS_DIR, exist_ok=True)
    os.makedirs(TEST_MONITORING_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Create sample data
    sample_data, sample_data_path = create_sample_data(1000)

    # Create sample data
    sample_data, sample_data_path = create_sample_data(1000)
    
    return {
        "models_dir": TEST_MODELS_DIR,
        "monitoring_dir": TEST_MONITORING_DIR,
        "data_dir": TEST_DATA_DIR,
        "sample_data": sample_data,
        "sample_data_path": sample_data_path
    }

# Clean up test artifacts
atexit.register(lambda: MODEL_CACHE.clear())

# Test signal detection with different configurations
def test_signal_detection(test_directories: dict, use_gpu: bool, use_pytorch: bool):
    """Test signal detection model training and prediction"""
    logger.info(f"Testing signal detection model with GPU={use_gpu}, PyTorch={use_pytorch}")
    
    # Create sample data
    data = test_directories["sample_data"]
    logger.info(f"Created sample data with {len(data)} rows")
    
    # Create configuration
    config = {
        "models_dir": test_directories["models_dir"],
        "monitoring_dir": test_directories["monitoring_dir"],
        "data_dir": test_directories["data_dir"],
        "min_samples": 100,
        "lookback_days": 30,
        "feature_selection": {
            "enabled": True,
            "method": "importance",
            "threshold": 0.01,
            "n_features": 10,
        },
        "time_series_cv": {
            "enabled": True,
            "n_splits": 3,
            "embargo_size": 5,
        },
        "monitoring": {"enabled": True, "drift_threshold": 0.1},
        "test_size": 0.2,
        "random_state": 42,
        "model_configs": {
            "signal_detection": {
                "type": "xgboost",
                "params": {
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "n_estimators": 50,
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                },
            },
        },
    }
    
    try:
        # Create data processor
        data_processor = MLDataProcessor(data_loader=None, config=config)
    
        # Prepare data - this will include all available features
        features, target = data_processor.prepare_signal_detection_data(data)
        logger.info(f"Prepared data with {len(features)} samples and {features.shape[1]} features")
    
        # Filter features to match what was used during training
        # The model was trained with only these features
        required_features = [
            'close', 'open', 'high', 'low', 'volume', 'sma5', 'sma10', 'sma20', 
            'ema5', 'ema10', 'ema20', 'macd', 'macd_signal', 'macd_hist', 
            'price_rel_sma5', 'price_rel_sma10', 'price_rel_sma20', 'mom1', 
            'mom5', 'mom10', 'volatility', 'volume_ratio', 'rsi', 'bb_width', 'spy_close'
        ]
    
        # Keep only the features that were used during training
        features = features[required_features]
        logger.info(f"Filtered to {len(features)} samples and {features.shape[1]} features")

        # Create trainer
        trainer = SignalDetectionTrainer(config=config)
        
        # Train model
        success = trainer.train(features, target, data_processor)
        assert success, "Model training failed"
        logger.info("Model training succeeded")
    
        # Test model loading and prediction
        # Test using our new XGBoostModel class
        model_path = os.path.join(config["models_dir"], "signal_detection_model.json")
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        # Save model in JSON format if it doesn't exist
        if not os.path.exists(model_path):
            import joblib
            import xgboost as xgb
            
            # Load the XGBoost model
            xgb_model = xgb.Booster()
            xgb_model.load_model(os.path.join(config["models_dir"], "signal_detection_model.xgb"))
            
            # Save in JSON format
            xgb_model.save_model(model_path)
        
        # Create XGBoostModel instance
        xgb_model = XGBoostModel(
            model_path=model_path,
            model_type="classifier",
            use_gpu=use_gpu,
            use_pytorch=use_pytorch
        )
        
        # Make predictions
        predictions = xgb_model.predict(features)
        
        # Evaluate predictions
        
        y_pred = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(target, y_pred)
        precision = precision_score(target, y_pred)
        recall = recall_score(target, y_pred)
        f1 = f1_score(target, y_pred)
        auc = roc_auc_score(target, predictions)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC: {auc:.4f}")
        
        # Test XGBoostIntegration
        logger.info("Testing XGBoostIntegration")
        
        # Create a simple DataPipeline mock
        class MockDataPipeline:
            def load_price_data(self, **kwargs):
                return {'AAPL': data}
                
            def load_market_data(self, **kwargs):
                return data[['spy_close', 'vix_close', 'spy_change', 'vix_change']]
                
            def prepare_training_data(self, **kwargs):
                return data
                
            def prepare_signal_detection_data(self, data):
                return features, target
        
        # Create XGBoostIntegration instance
        integration = XGBoostIntegration(
            data_pipeline=MockDataPipeline(),
            model_path=model_path,
            model_type="classifier",
            use_gpu=use_gpu,
            create_if_missing=False,  # Model should already exist
            use_pytorch=True
        )
        
        # Test prediction
        end_date = pd.Timestamp('2023-01-31')
        start_date = pd.Timestamp('2023-01-01')
        predictions = integration.predict_for_tickers(
            tickers=['AAPL'],
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate predictions
        assert 'AAPL' in predictions, "No predictions returned for AAPL"
        assert len(predictions['AAPL']) > 0, "Empty predictions returned for AAPL"
        logger.info(f"XGBoostIntegration predictions: {predictions['AAPL'][:5]}...")
        
        # Validate metrics
        assert accuracy > 0.5, f"Accuracy too low: {accuracy}"
        assert precision > 0, f"Precision too low: {precision}"
        assert recall > 0, f"Recall too low: {recall}"
        assert f1 > 0, f"F1 score too low: {f1}"
        assert auc > 0.5, f"AUC too low: {auc}"
        
        # Test model persistence by reloading
        reloaded_model = XGBoostModel(
            model_path=model_path,
            model_type="classifier",
            use_gpu=use_gpu,
            use_pytorch=use_pytorch
        )
        reloaded_predictions = reloaded_model.predict(features)
        assert np.allclose(predictions, reloaded_predictions), "Reloaded model predictions don't match original"
        # Test model performance monitoring
        metrics = xgb_model.monitor_performance(
            features=features,
            target=target,
            metrics_path=os.path.join(config["monitoring_dir"], "signal_detection_metrics.json")
        )
        
        # Verify metrics were calculated
        assert "accuracy" in metrics, "Accuracy metric not calculated"
        assert "precision" in metrics, "Precision metric not calculated"
        assert "recall" in metrics, "Recall metric not calculated"
        assert "f1" in metrics, "F1 metric not calculated"
        assert "roc_auc" in metrics, "ROC AUC metric not calculated"
        assert "timestamp" in metrics, "Timestamp not included in metrics"
        
        # Verify metrics file was created
        metrics_path = os.path.join(config["monitoring_dir"], "signal_detection_metrics.json")
        assert os.path.exists(metrics_path), f"Metrics file not found at {metrics_path}"
        
        # Test model serialization/deserialization
        model_dict = xgb_model.to_dict()
        assert "model_type" in model_dict, "Model type not included in serialized model"
        assert "model_data" in model_dict, "Model data not included in serialized model"
        
        # Deserialize model
        deserialized_model = XGBoostModel.from_dict(model_dict)
        assert deserialized_model is not None, "Failed to deserialize model"
        
        # Test predictions with deserialized model
        deserialized_predictions = deserialized_model.predict(features)
        assert np.allclose(predictions, deserialized_predictions), "Deserialized model predictions don't match original"
    except Exception as e:
        logger.error(f"Error in signal detection test: {e!r}", exc_info=True)
        raise


# Function to clear model cache
def clear_model_cache():
    """Clear the model cache"""
    MODEL_CACHE.clear()


def create_sample_data(n_samples: int = 1000):
    """Create sample data for testing with specified number of samples"""
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    # Create price data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0, 1, n_samples)) * 0.1
    high = close + np.random.uniform(0, 1, n_samples)
    low = close - np.random.uniform(0, 1, n_samples)
    open_price = close - np.random.uniform(-0.5, 0.5, n_samples)
    volume = np.random.randint(1000, 10000, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ticker': 'AAPL',
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Add technical indicators
    df['sma5'] = df['close'].rolling(5).mean()
    df['sma10'] = df['close'].rolling(10).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['ema5'] = calculate_ema(df['close'].values, 5)
    df['ema10'] = calculate_ema(df['close'].values, 10)
    df['ema20'] = calculate_ema(df['close'].values, 20)
    
    macd_line, signal_line, histogram = calculate_macd(df['close'].values)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = histogram
    
    upper, middle, lower = calculate_bollinger_bands(df['close'].values)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle
    
    # Add price relative to moving averages
    df['price_rel_sma5'] = df['close'] / df['sma5'] - 1
    df['price_rel_sma10'] = df['close'] / df['sma10'] - 1
    df['price_rel_sma20'] = df['close'] / df['sma20'] - 1
    
    # Add momentum indicators
    df['mom1'] = df['close'].pct_change(1)
    df['mom5'] = df['close'].pct_change(5)
    df['mom10'] = df['close'].pct_change(10)
    
    # Add volatility
    df['volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
    
    # Add volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    
    # Add RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Add Stochastic Oscillator
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'].values, df['low'].values, df['close'].values)
    
    # Add target variable (simple signal based on future returns)
    future_return = df['close'].shift(-10) / df['close'] - 1
    df['signal_target'] = (future_return > 0).astype(int)
    df['future_return_5min'] = df['close'].shift(-5) / df['close'] - 1
    df['future_return_10min'] = future_return
    df['future_return_30min'] = df['close'].shift(-30) / df['close'] - 1
    
    # Add risk target
    df['atr_pct'] = (df['high'] - df['low']) / df['close']
    
    # Add exit target
    df['optimal_exit'] = df['high'].rolling(10).max() / df['close'] - 1
    
    # Add market data
    df['spy_close'] = 400 + np.cumsum(np.random.normal(0, 1, n_samples)) * 0.05
    df['vix_close'] = 15 + np.random.normal(0, 1, n_samples)
    df['spy_change'] = df['spy_close'].pct_change()
    df['vix_change'] = df['vix_close'].pct_change()
    
    # Drop NaN values
    df = df.dropna()
    
    # Save to CSV for persistence tests
    sample_data_path = os.path.join(TEST_DATA_DIR, "sample_data.csv")
    df.to_csv(sample_data_path)
    
    return df, sample_data_path


# Test XGBoost model with different configurations
def test_xgboost_model(test_directories: dict, use_gpu: bool, use_pytorch: bool):
    """Test XGBoostModel directly"""
    logger.info(f"Testing XGBoostModel directly with GPU={use_gpu}, PyTorch={use_pytorch}")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Prepare features and target
    features = data[['sma5', 'sma10', 'sma20', 'ema5', 'ema10', 'ema20', 
                     'macd', 'macd_signal', 'macd_hist', 
                     'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                     'price_rel_sma5', 'price_rel_sma10', 'price_rel_sma20',
                     'mom1', 'mom5', 'mom10', 'volatility', 'volume_ratio', 'rsi']]
    target = data['signal_target']
    
    # Create model
    model = XGBoostModel(
        model_type="classifier",
        use_gpu=use_gpu,
        use_pytorch=use_pytorch
    )
    
    try:
        # Train model with cross-validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        eval_sets = []
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            eval_sets.append((X_test, y_test))
        
        # Train with early stopping and evaluation sets
        success = model.train(
            features, 
            target, 
            early_stopping_rounds=10,
            eval_set=eval_sets,
            verbose=False
        )
        assert success, "Model training failed"
        logger.info("Model training succeeded")
        
        # Make predictions
        predictions = model.predict(features)
        assert len(predictions) == len(features), "Prediction length mismatch"
        
        # Evaluate predictions        
        
        y_pred = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(target, y_pred)
        precision = precision_score(target, y_pred)
        recall = recall_score(target, y_pred)
        f1 = f1_score(target, y_pred)
        auc = roc_auc_score(target, predictions)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC: {auc:.4f}")
        
        # Save and load model
        model_path = os.path.join(test_directories["models_dir"], "xgboost_test_model.json")
        model.save_model(model_path)
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        
        # Load model
        loaded_model = XGBoostModel(
            model_path=model_path,
            model_type="classifier",
            use_gpu=use_gpu,
            use_pytorch=use_pytorch
        )
        
        # Make predictions with loaded model
        loaded_predictions = loaded_model.predict(features)
        assert len(loaded_predictions) == len(features), "Loaded model prediction length mismatch"
        
        loaded_y_pred = (loaded_predictions > 0.5).astype(int)
        loaded_accuracy = accuracy_score(target, loaded_y_pred)
        
        logger.info(f"Loaded model accuracy: {loaded_accuracy:.4f}")
        
        # Check if predictions match
        match = np.allclose(predictions, loaded_predictions, rtol=1e-5, atol=1e-5)
        logger.info(f"Predictions match: {match}")
        assert match, "Loaded model predictions don't match original"
        
        # Test feature importance
        feature_importance = model.get_feature_importance()
        assert len(feature_importance) > 0, "No feature importance returned"
        logger.info(f"Top 5 features by importance: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        # Test model serialization with different formats
        for format in ['json', 'binary']:
            format_path = os.path.join(test_directories["models_dir"], f"xgboost_test_model.{format}")
            model.save_model(format_path, format=format)
            assert os.path.exists(format_path), f"Model file not found at {format_path}"
            
            # Load and verify
            format_model = XGBoostModel(
                model_path=format_path,
                model_type="classifier",
                use_gpu=use_gpu,
                use_pytorch=use_pytorch
            )
            format_predictions = format_model.predict(features)
            format_match = np.allclose(predictions, format_predictions, rtol=1e-5, atol=1e-5)
            assert format_match, f"Model saved in {format} format predictions don't match original"
    except Exception as e:
        logger.error(f"Error in XGBoost model test: {e}", exc_info=True)
        raise


# Test model with missing features
def test_model_with_missing_features(test_directories: dict, use_gpu: bool):
    """Test model robustness with missing features"""
    logger.info(f"Testing model with missing features, GPU={use_gpu}")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Prepare features and target
    features = data[['sma5', 'sma10', 'sma20', 'ema5', 'ema10', 'ema20']]
    target = data['signal_target']
    
    # Create and train model
    model = XGBoostModel(
        model_type="classifier",
        use_gpu=use_gpu,
        use_pytorch=False
    )
    
    success = model.train(features, target)
    assert success, "Model training failed"
    
    # Test with missing features
    test_features = features.copy()
    test_features['sma5'] = np.nan  # Introduce missing values
    
    # Model should handle missing values gracefully
    try:
        predictions = model.predict(test_features)
        assert len(predictions) == len(test_features), "Prediction length mismatch with missing features"
        logger.info("Model successfully handled missing features")
    except Exception as e:
        logger.error(f"Model failed to handle missing features: {e}")
        raise


# Test error handling
def test_error_handling(test_directories: dict):
    """Test error handling in ML engine components"""
    logger.info("Testing error handling")
    
    # Test with invalid model path
    try:
        model = XGBoostModel(
            model_path="nonexistent_model.json",
            model_type="classifier"
        )
        # Should not raise exception but log error and create default model
        assert model.model is not None, "Model should be created even with invalid path"
        logger.info("Successfully handled invalid model path")
    except Exception as e:
        logger.error(f"Failed to handle invalid model path: {e}")
        raise
    
    # Test with invalid feature data
    try:
        model = XGBoostModel(model_type="classifier")
        # Empty DataFrame should be handled gracefully
        predictions = model.predict(pd.DataFrame())
        assert isinstance(predictions, np.ndarray), "Should return empty array for empty input"
        assert len(predictions) == 0, "Should return empty array for empty input"
        logger.info("Successfully handled empty feature data")
    except Exception as e:
        logger.error(f"Failed to handle empty feature data: {e}")
        raise


# Test batch processor
def test_batch_processor(test_directories: dict):
    """Test batch processor functionality"""
    logger.info("Testing batch processor")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Create batch processor
    batch_processor = BatchProcessor(
        batch_size=100,
        num_workers=2,
        use_gpu=False
    )
    
    # Prepare features and target
    features = data[['sma5', 'sma10', 'sma20', 'ema5', 'ema10', 'ema20']]
    target = data['signal_target']
    
    # Test batch processing
    try:
        # Process features in batches
        processed_features = batch_processor.process_features(
            features,
            processor_fn=lambda x: x * 2  # Simple transformation
        )
        
        # Verify results
        assert len(processed_features) == len(features), "Processed features length mismatch"
        assert np.allclose(processed_features.values, features.values * 2), "Batch processing transformation incorrect"
        
        # Test batch prediction
        model = XGBoostModel(model_type="classifier")
        model.train(features, target)
        
        # Make batch predictions
        batch_predictions = batch_processor.predict(
            model=model,
            features=features
        )
        
        # Make regular predictions
        regular_predictions = model.predict(features)
        
        # Verify results
        assert len(batch_predictions) == len(regular_predictions), "Batch predictions length mismatch"
        assert np.allclose(batch_predictions, regular_predictions), "Batch predictions don't match regular predictions"
        
        logger.info("Batch processor tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in batch processor test: {e}", exc_info=True)
        raise


# Test model explainability
def test_model_explainability(test_directories: dict):
    """Test model explainability functionality"""
    logger.info("Testing model explainability")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Prepare features and target
    features = data[['sma5', 'sma10', 'sma20', 'ema5', 'ema10', 'ema20']]
    target = data['signal_target']
    
    # Create and train model
    model = XGBoostModel(
        model_type="classifier",
        use_gpu=False,
        use_pytorch=False
    )
    
    model.train(features, target)
    
    # Create explainer
    explainer = ModelExplainer(model)
    
    try:
        # Get feature importance
        importance = explainer.get_feature_importance(features)
        assert len(importance) == features.shape[1], "Feature importance length mismatch"
        logger.info("Model explainability tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in model explainability test: {e}", exc_info=True)
        raise


# Test model cache
def test_model_cache(test_directories: dict):
    """Test model cache functionality"""
    logger.info("Testing model cache")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Prepare features and target
    features = data[['sma5', 'sma10', 'sma20', 'ema5', 'ema10', 'ema20']]
    target = data['signal_target']
    
    # Create and train model
    model = XGBoostModel(
        model_type="classifier",
        use_gpu=False,
        use_pytorch=False
    )
    
    model.train(features, target)
    
    # Define loader function for get_model
    def loader_fn():
        return model
    
    try:
        # Test get_model
        cached_model = get_model("test_model", loader_fn)
        assert cached_model is not None, "Failed to get model from cache"
        
        # Test get_model_metadata
        metadata = get_model_metadata("test_model")
        assert metadata is not None, "Failed to get model metadata"
        
        # Test get_cache_stats
        stats = get_cache_stats()
        assert stats is not None, "Failed to get cache stats"
        assert "size" in stats, "Cache stats missing size"
        
        # Test invalidate_model
        success = invalidate_model("test_model")
        assert success, "Failed to invalidate model"
        
        logger.info("Model cache tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in model cache test: {e}", exc_info=True)
        raise


# Test market regime trainer
def test_market_regime_trainer(test_directories: dict):
    """Test market regime trainer"""
    logger.info("Testing market regime trainer")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Create configuration
    config = {
        "models_dir": test_directories["models_dir"],
        "monitoring_dir": test_directories["monitoring_dir"],
        "data_dir": test_directories["data_dir"],
        "test_size": 0.2,
        "random_state": 42,
    }
    
    try:
        # Create trainer
        trainer = MarketRegimeTrainer(config=config)
        
        # Prepare data
        features = data[['spy_close', 'vix_close', 'spy_change', 'vix_change']]
        target = (data['spy_change'] > 0).astype(int)  # Simple regime classification
        
        # Train model
        success = trainer.train(features, target)
        assert success, "Market regime model training failed"
        
        # Test prediction
        predictions = trainer.predict(features)
        assert len(predictions) == len(features), "Market regime prediction length mismatch"
        
        logger.info("Market regime trainer tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in market regime trainer test: {e}", exc_info=True)
        raise


# Test price prediction trainer
def test_price_prediction_trainer(test_directories: dict):
    """Test price prediction trainer"""
    logger.info("Testing price prediction trainer")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Create configuration
    config = {
        "models_dir": test_directories["models_dir"],
        "monitoring_dir": test_directories["monitoring_dir"],
        "data_dir": test_directories["data_dir"],
        "test_size": 0.2,
        "random_state": 42,
    }
    
    try:
        # Create trainer
        trainer = PricePredictionTrainer(config=config)
        
        # Prepare data
        features = data[['sma5', 'sma10', 'sma20', 'ema5', 'ema10', 'ema20']]
        target = data['future_return_10min']  # Predict future returns
        
        # Train model
        success = trainer.train(features, target)
        assert success, "Price prediction model training failed"
        
        # Test prediction
        predictions = trainer.predict(features)
        assert len(predictions) == len(features), "Price prediction length mismatch"
        
        logger.info("Price prediction trainer tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in price prediction trainer test: {e}", exc_info=True)
        raise


# Test risk assessment trainer
def test_risk_assessment_trainer(test_directories: dict):
    """Test risk assessment trainer"""
    logger.info("Testing risk assessment trainer")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Create configuration
    config = {
        "models_dir": test_directories["models_dir"],
        "monitoring_dir": test_directories["monitoring_dir"],
        "data_dir": test_directories["data_dir"],
        "test_size": 0.2,
        "random_state": 42,
    }
    
    try:
        # Create trainer
        trainer = RiskAssessmentTrainer(config=config)
        
        # Prepare data
        features = data[['volatility', 'atr', 'rsi', 'bb_width']]
        target = data['atr_pct']  # Predict risk (ATR as percentage of price)
        
        # Train model
        success = trainer.train(features, target)
        assert success, "Risk assessment model training failed"
        
        # Test prediction
        predictions = trainer.predict(features)
        assert len(predictions) == len(features), "Risk assessment prediction length mismatch"
        
        logger.info("Risk assessment trainer tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in risk assessment trainer test: {e}", exc_info=True)
        raise


# Test exit strategy trainer
def test_exit_strategy_trainer(test_directories: dict):
    """Test exit strategy trainer"""
    logger.info("Testing exit strategy trainer")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    # Create configuration
    config = {
        "models_dir": test_directories["models_dir"],
        "monitoring_dir": test_directories["monitoring_dir"],
        "data_dir": test_directories["data_dir"],
        "test_size": 0.2,
        "random_state": 42,
    }
    
    try:
        # Create trainer
        trainer = ExitStrategyTrainer(config=config)
        
        # Prepare data
        features = data[['sma5', 'sma10', 'sma20', 'volatility', 'rsi']]
        target = data['optimal_exit']  # Predict optimal exit point
        
        # Train model
        success = trainer.train(features, target)
        assert success, "Exit strategy model training failed"
        
        # Test prediction
        predictions = trainer.predict(features)
        assert len(predictions) == len(features), "Exit strategy prediction length mismatch"
        
        logger.info("Exit strategy trainer tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in exit strategy trainer test: {e}", exc_info=True)
        raise


# Test utils module
def test_utils(test_directories: dict):
    """Test utils module functionality"""
    logger.info("Testing utils module")
    
    # Create sample data
    data = test_directories["sample_data"]
    
    try:
        # Test normalize_features
        features = data[['sma5', 'sma10', 'sma20']]
        normalized = normalize_features(features)
        assert normalized.shape == features.shape, "Normalized features shape mismatch"
        
        # Test create_time_features
        # Create a DataFrame with timestamp column
        df_with_timestamp = pd.DataFrame({'timestamp': data.index})
        # Use add_time_features from data_pipeline.processing
        time_features = add_time_features(df_with_timestamp)
        assert len(time_features) == len(data), "Time features length mismatch"
        
        logger.info("Utils module tests passed")
        return True
    except Exception as e:
        logger.error(f"Error in utils module test: {e}", exc_info=True)
        raise


def main():
    """Main entry point"""
    logger.info("Starting ML Engine test")
    
    # Create test directories and sample data
    test_dirs = test_directories()
    
    # Run tests with different configurations
    configs = [
        (True, True),   # GPU + PyTorch
        (True, False),  # GPU only
        (False, True),  # PyTorch only
        (False, False)  # CPU only
    ]
    
    success = True
    
    # Run signal detection tests
    for use_gpu, use_pytorch in configs:
        try:
            logger.info(f"Running tests with GPU={use_gpu}, PyTorch={use_pytorch}")
            test_signal_detection(test_dirs, use_gpu, use_pytorch)
            test_xgboost_model(test_dirs, use_gpu, use_pytorch)
        except Exception as e:
            logger.error(f"Test failed with GPU={use_gpu}, PyTorch={use_pytorch}: {e}")
            success = False
    
    # Run additional tests
    test_model_with_missing_features(test_dirs, True)
    test_error_handling(test_dirs)
    test_batch_processor(test_dirs)
    test_model_explainability(test_dirs)
    test_model_cache(test_dirs)
    test_market_regime_trainer(test_dirs)
    test_price_prediction_trainer(test_dirs)
    test_risk_assessment_trainer(test_dirs)
    test_exit_strategy_trainer(test_dirs)
    test_utils(test_dirs)
    
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
