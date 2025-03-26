#!/usr/bin/env python3
"""
XGBoost Integration Module

This module demonstrates how to integrate the XGBoost model with the data pipeline:
1. Loading data using the data pipeline
2. Preprocessing data for XGBoost
3. Training or loading a pretrained XGBoost model
4. Making predictions with the model
5. Evaluating model performance

The module is designed to work with the existing data pipeline and supports
GPU acceleration when available.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
import packaging
from typing import Dict, List, Optional, Tuple

# Import the data pipeline and XGBoost model
from data_pipeline import DataPipeline
from xgboost_model import XGBoostModel, create_pretrained_model, load_pretrained_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("xgboost_integration")


class XGBoostIntegration:
    """
    Integration class for XGBoost models with the data pipeline
    """
    
    def __init__(
        self,
        data_pipeline: Optional[DataPipeline] = None,
        model_path: Optional[str] = None,
        model_type: str = "classifier",
        use_gpu: bool = True,
        create_if_missing: bool = True,
        use_pytorch: bool = True,
    ) -> None:
        """Initialize the XGBoost integration"""
        # Initialize data pipeline if not provided
        self.data_pipeline = data_pipeline or DataPipeline(test_mode=True)
        
        # Set model parameters
        self.model_path = model_path or 'models/xgboost/pretrained_model.json'
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.use_pytorch = use_pytorch
        
        # Check for PyTorch availability
        try:
            import torch
            self.torch_available = True
            self.cuda_available = torch.cuda.is_available()
            logger.info(f"PyTorch is available (CUDA: {self.cuda_available})")
        except ImportError:
            self.torch_available = False
            self.cuda_available = False
            logger.warning("PyTorch is not available. Install with 'pip install torch'")
        
        # Check for XGBoost 3.0.0+ availability
        try:
            import xgboost as xgb
            self.xgboost_version = packaging.version.parse(xgb.__version__)
            self.xgboost_3_available = self.xgboost_version >= packaging.version.parse("3.0.0")
            if self.xgboost_3_available:
                logger.info(f"XGBoost {xgb.__version__} supports PyTorch integration")
            else:
                logger.warning(f"XGBoost {xgb.__version__} detected. Version 3.0.0 or later recommended for PyTorch integration")
        except ImportError:
            self.xgboost_3_available = False
            logger.warning("XGBoost is not available. Install with 'pip install xgboost>=3.0.0'")
        
        # Determine if PyTorch integration should be used
        self.use_pytorch_integration = self.use_pytorch and self.torch_available and self.xgboost_3_available
        
        # Load or create the model
        if os.path.exists(self.model_path):
            logger.info(f"Loading pretrained model from {self.model_path}")
            self.model = load_pretrained_model(
                self.model_path,
                use_pytorch=self.use_pytorch_integration
            )
        elif create_if_missing:
            logger.info(f"Creating pretrained model at {self.model_path}")
            create_pretrained_model(
                self.model_path,
                use_pytorch=self.use_pytorch_integration
            )
            self.model = load_pretrained_model(
                self.model_path,
                use_pytorch=self.use_pytorch_integration
            )
        else:
            logger.info("Initializing new model")
            self.model = XGBoostModel(
                model_type=model_type,
                use_gpu=use_gpu,
                use_pytorch=self.use_pytorch_integration
            )
    
    def load_and_prepare_data(
        self,
        tickers: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        target_column: str = 'signal_target',
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for the XGBoost model
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            target_column: Target column name
            
        Returns:
            Tuple of (features, target)
        """
        try:
            # Load price data
            price_data = self.data_pipeline.load_price_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d"
            )
            
            # Load market data
            market_data = self.data_pipeline.load_market_data(
                start_date=start_date,
                end_date=end_date
            )
            
            # Prepare training data
            combined_data = self.data_pipeline.prepare_training_data(
                price_data=price_data,
                market_data=market_data
            )
            
            # Prepare features and target
            if target_column in combined_data.columns:
                features, target = self.data_pipeline.prepare_signal_detection_data(combined_data)
                return features, target
            else:
                logger.error(f"Target column '{target_column}' not found in data")
                return pd.DataFrame(), pd.Series()
        
        except Exception as e:
            logger.error(f"Error loading and preparing data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_model(
        self,
        tickers: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        target_column: str = 'signal_target',
        tune_hyperparameters: bool = False,
        param_grid: Dict = None,
        cv: int = 5,
        time_series_cv: bool = True,
        **kwargs
    ) -> bool:
        """
        Train the XGBoost model with data from the pipeline
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for training data
            end_date: End date for training data
            target_column: Target column name
            tune_hyperparameters: Whether to perform hyperparameter tuning
            param_grid: Dictionary of hyperparameters to search (for tuning)
            cv: Number of cross-validation folds
            time_series_cv: Whether to use time series cross-validation
            **kwargs: Additional arguments to pass to the model's train method
            
        Returns:
            Boolean indicating success
        """
        try:
            # Load and prepare data
            features, target = self.load_and_prepare_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                target_column=target_column
            )
            
            if features.empty or len(target) == 0:
                logger.error("No data available for training")
                return False
            
            # Perform hyperparameter tuning if requested
            if tune_hyperparameters:
                logger.info("Performing hyperparameter tuning...")
                best_params = self.model.tune_hyperparameters(
                    features=features,
                    target=target,
                    param_grid=param_grid,
                    cv=cv,
                    time_series=time_series_cv
                )
                logger.info(f"Best parameters found: {best_params}")
            
            # Train the model with PyTorch integration if available
            if self.use_pytorch_integration:
                logger.info("Training model with PyTorch integration")
                
                # Add enable_amp parameter for automatic mixed precision if CUDA is available
                if self.cuda_available:
                    logger.info("Enabling automatic mixed precision (AMP) for training")
                    kwargs['enable_amp'] = True
                
                # Train the model
                success = self.model.train(features, target, **kwargs)
            else:
                # Standard training
                logger.info("Training model with standard XGBoost")
                success = self.model.train(features, target, **kwargs)
            
            if success:
                # Evaluate the model
                logger.info("Evaluating model performance...")
                metrics = self.model.evaluate(features, target)
                logger.info(f"Model evaluation metrics: {metrics}")
                
                # Perform cross-validation
                logger.info("Performing cross-validation...")
                cv_results = self.model.cross_validate(
                    features=features,
                    target=target,
                    cv=cv,
                    time_series=time_series_cv
                )
                logger.info(f"Cross-validation results: {cv_results}")
                
                # Save the trained model
                self.model.save_model(self.model_path)
                logger.info(f"Model trained and saved to {self.model_path}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            features: Features to make predictions on
            
        Returns:
            Numpy array of predictions
        """
        return self.model.predict(features)
    
    def predict_for_tickers(
        self,
        tickers: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for a list of tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary of ticker -> predictions
        """
        try:
            # Load price data
            price_data = self.data_pipeline.load_price_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d"
            )
            
            # Load market data
            market_data = self.data_pipeline.load_market_data(
                start_date=start_date,
                end_date=end_date
            )
            
            # Prepare data
            combined_data = self.data_pipeline.prepare_training_data(
                price_data=price_data,
                market_data=market_data
            )
            
            # Make predictions for each ticker
            results = {}
            
            for ticker in tickers:
                if ticker not in price_data:
                    logger.warning(f"No data available for {ticker}")
                    continue
                
                # Get ticker-specific data
                ticker_data = combined_data[combined_data['ticker'] == ticker]
                
                if ticker_data.empty:
                    logger.warning(f"No processed data available for {ticker}")
                    continue
                
                # Prepare features
                features, _ = self.data_pipeline.prepare_signal_detection_data(ticker_data)
                
                if features.empty:
                    logger.warning(f"No features available for {ticker}")
                    continue
                
                # Make predictions
                predictions = self.predict(features)
                results[ticker] = predictions
            
            return results
        
        except Exception as e:
            logger.error(f"Error making predictions for tickers: {e}")
            return {}
    def create_ensemble_model(
        self,
        tickers: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        target_column: str = 'signal_target',
        ensemble_type: str = 'voting',
        include_models: List[str] = None
    ):
        """
        Create an ensemble model that combines XGBoost with other sklearn models
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            target_column: Target column name
            ensemble_type: Type of ensemble ('voting' or 'stacking')
            include_models: List of models to include in the ensemble
                            Options: 'rf' (Random Forest), 'gb' (Gradient Boosting),
                                    'lr' (Logistic Regression), 'svm' (Support Vector Machine)
            
        Returns:
            Trained ensemble model
        """
        try:
            # Import required sklearn models
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            # Load and prepare data
            features, target = self.load_and_prepare_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                target_column=target_column
            )
            
            if features.empty or len(target) == 0:
                logger.error("No data available for creating ensemble model")
                return None
            
            # Default to all models if not specified
            if include_models is None:
                include_models = ['rf', 'gb', 'lr']
            
            # Create base models
            models = []
            
            # Always include XGBoost
            models.append(('xgb', self.model.model))
            
            # Add other models based on include_models
            if 'rf' in include_models:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                models.append(('rf', rf))
                
            if 'gb' in include_models:
                gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                models.append(('gb', gb))
                
            if 'lr' in include_models:
                lr = LogisticRegression(max_iter=1000, random_state=42)
                models.append(('lr', lr))
                
            if 'svm' in include_models:
                svm = SVC(probability=True, random_state=42)
                models.append(('svm', svm))
            
            # Create ensemble model
            ensemble = None
            if ensemble_type == 'voting':
                ensemble = VotingClassifier(
                    estimators=models,
                    voting='soft'
                )
            else:  # stacking
                ensemble = StackingClassifier(
                    estimators=models[:-1],  # All except the last one
                    final_estimator=models[-1][1],  # Use the last model as final estimator
                    cv=5
                )
            
            # Train the ensemble model
            ensemble.fit(features, target)
            
            logger.info(f"Created and trained {ensemble_type} ensemble model with {len(models)} base models")
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            return None


def main():
    """Main function to demonstrate XGBoost integration with enhanced sklearn functionality"""
    try:
        # Initialize the integration with PyTorch support
        integration = XGBoostIntegration(
            create_if_missing=True,
            use_gpu=True,
            use_pytorch=True
        )
        
        # Log PyTorch integration status
        if integration.use_pytorch_integration:
            logger.info("Using XGBoost with PyTorch integration")
        else:
            logger.info("Using standard XGBoost without PyTorch integration")
        
        # Define date range and tickers
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)  # 1 year of data
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # 1. Basic training without hyperparameter tuning
        logger.info("1. Basic model training")
        logger.info(f"Training model with data for {tickers}")
        integration.train_model(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            early_stopping_rounds=10,
            verbose=True
        )
        
        # 2. Training with hyperparameter tuning
        logger.info("\n2. Training with hyperparameter tuning")
        # Define a smaller parameter grid for demonstration
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        integration.train_model(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            tune_hyperparameters=True,
            param_grid=param_grid,
            cv=3,
            time_series_cv=True,
            early_stopping_rounds=5,
            verbose=True
        )
        
        # 3. Make predictions
        logger.info("\n3. Making predictions")
        predictions = integration.predict_for_tickers(
            tickers=tickers,
            start_date=end_date - datetime.timedelta(days=30),  # Last 30 days
            end_date=end_date
        )
        
        # Print predictions
        for ticker, preds in predictions.items():
            logger.info(f"Predictions for {ticker}: {preds[:5]}...")
        
        # 4. Create a preprocessing pipeline
        logger.info("\n4. Creating preprocessing pipeline")
        features, target = integration.load_and_prepare_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add a categorical feature for demonstration
        if not features.empty:
            features['market_condition'] = np.random.choice(
                ['bullish', 'bearish', 'neutral'],
                size=len(features)
            )
            
            # Create pipeline
            pipeline = integration.model.create_pipeline(
                numeric_features=[col for col in features.columns if col != 'market_condition'],
                categorical_features=['market_condition']
            )
            
            if pipeline:
                logger.info("Pipeline created successfully")
                # Fit the pipeline
                pipeline.fit(features, target)
                logger.info("Pipeline fitted successfully")
        # 5. Create and use ensemble model
        logger.info("\n5. Creating ensemble model")
        ensemble = integration.create_ensemble_model(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            ensemble_type='voting',
            include_models=['rf', 'gb', 'lr']
        )
        
        if ensemble and not features.empty:
            # Make predictions with ensemble
            ensemble_predictions = ensemble.predict(features)
            logger.info(f"Ensemble predictions: {ensemble_predictions[:5]}...")
            
            # Compare with XGBoost predictions
            xgb_predictions = integration.model.predict(features)
            
            # Calculate agreement percentage
            agreement = np.mean(ensemble_predictions == xgb_predictions) * 100
            logger.info(f"Agreement between XGBoost and ensemble: {agreement:.2f}%")
        
        logger.info("\nXGBoost integration demonstration with enhanced sklearn functionality completed successfully")
        logger.info("\nXGBoost integration demonstration with enhanced sklearn functionality completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()