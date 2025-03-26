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
from typing import Dict, List, Optional, Tuple, Union

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
    ) -> None:
        """Initialize the XGBoost integration"""
        # Initialize data pipeline if not provided
        self.data_pipeline = data_pipeline or DataPipeline(test_mode=True)
        
        # Set model parameters
        self.model_path = model_path or 'models/xgboost/pretrained_model.json'
        self.model_type = model_type
        self.use_gpu = use_gpu
        
        # Load or create the model
        if os.path.exists(self.model_path):
            logger.info(f"Loading pretrained model from {self.model_path}")
            self.model = load_pretrained_model(self.model_path)
        elif create_if_missing:
            logger.info(f"Creating pretrained model at {self.model_path}")
            create_pretrained_model(self.model_path)
            self.model = load_pretrained_model(self.model_path)
        else:
            logger.info("Initializing new model")
            self.model = XGBoostModel(model_type=model_type, use_gpu=use_gpu)
    
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
        **kwargs
    ) -> bool:
        """
        Train the XGBoost model with data from the pipeline
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for training data
            end_date: End date for training data
            target_column: Target column name
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
            
            # Train the model
            success = self.model.train(features, target, **kwargs)
            
            if success:
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


def main():
    """Main function to demonstrate XGBoost integration"""
    try:
        # Initialize the integration
        integration = XGBoostIntegration(create_if_missing=True)
        
        # Define date range and tickers
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)  # 1 year of data
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # Train the model
        logger.info(f"Training model with data for {tickers}")
        integration.train_model(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Make predictions
        logger.info(f"Making predictions for {tickers}")
        predictions = integration.predict_for_tickers(
            tickers=tickers,
            start_date=end_date - datetime.timedelta(days=30),  # Last 30 days
            end_date=end_date
        )
        
        # Print predictions
        for ticker, preds in predictions.items():
            logger.info(f"Predictions for {ticker}: {preds[:5]}...")
        
        logger.info("XGBoost integration demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()