#!/usr/bin/env python3
"""
XGBoost Model Module

This module provides functionality for loading and using pretrained XGBoost models:
1. Loading pretrained models from disk
2. Making predictions with loaded models
3. Evaluating model performance
4. Integration with the data pipeline

The module is designed to work with the existing data pipeline and supports
GPU acceleration when available.
"""

import os
import pickle
import logging
import numpy as np
import packaging
import pandas as pd
from typing import Dict, List, Optional, Union

# Import sklearn components for enhanced functionality
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("xgboost_model")

# Check if XGBoost is available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info(f"XGBoost version {xgb.__version__} is available")
    
    # Check if we're using XGBoost 3.0.0 or later
    XGBOOST_VERSION = packaging.version.parse(xgb.__version__)
    XGBOOST_3_AVAILABLE = XGBOOST_VERSION >= packaging.version.parse("3.0.0")
    if XGBOOST_3_AVAILABLE:
        logger.info("Using XGBoost 3.0.0 or later with PyTorch integration")
    else:
        logger.warning(f"XGBoost version {xgb.__version__} detected. Version 3.0.0 or later recommended for PyTorch integration")
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_3_AVAILABLE = False
    logger.warning("XGBoost is not available. Install with 'pip install xgboost>=3.0.0'")

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch version {torch.__version__} is available")
    
    # Check if CUDA is available
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA is not available for PyTorch")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    logger.warning("PyTorch is not available. Install with 'pip install torch'")


class XGBoostModel:
    """
    XGBoost model wrapper for loading and using pretrained models
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "classifier",
        use_gpu: bool = True,
        feature_names: Optional[List[str]] = None,
        use_pytorch: bool = True,
    ) -> None:
        """Initialize the XGBoost model wrapper"""
        self.model_path = model_path
        self.model_type = model_type
        self.use_gpu = use_gpu and XGBOOST_AVAILABLE
        self.feature_names = feature_names
        self.model = None
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE and XGBOOST_3_AVAILABLE
        self.device = torch.device("cuda" if CUDA_AVAILABLE and self.use_gpu else "cpu") if TORCH_AVAILABLE else None
        
        # Check if XGBoost is available
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost is not available. Install with 'pip install xgboost>=3.0.0'")
            return
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.info("No model path provided or model file not found")
            
            # Create a default model
            if model_type == "classifier":
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Use PyTorch integration for XGBoost 3.0.0+
                    self.model = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method='gpu_hist' if use_gpu else 'hist',
                        eval_metric='logloss',
                        device='cuda' if use_gpu and CUDA_AVAILABLE else 'cpu',
                        enable_categorical=True,
                        objective='binary:logistic'
                    )
                    logger.info("Created XGBoost classifier with PyTorch integration")
                else:
                    # Use standard XGBoost
                    self.model = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method='gpu_hist' if use_gpu else 'hist',
                        eval_metric='logloss'
                    )
                    logger.info("Created standard XGBoost classifier")
            else:
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Use PyTorch integration for XGBoost 3.0.0+
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method='gpu_hist' if use_gpu else 'hist',
                        eval_metric='rmse',
                        device='cuda' if use_gpu and CUDA_AVAILABLE else 'cpu',
                        enable_categorical=True,
                        objective='reg:squarederror'
                    )
                    logger.info("Created XGBoost regressor with PyTorch integration")
                else:
                    # Use standard XGBoost
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        tree_method='gpu_hist' if use_gpu else 'hist',
                        eval_metric='rmse'
                    )
                    logger.info("Created standard XGBoost regressor")
            
            logger.info(f"Created default {model_type} model")
    
    def load_model(self, model_path: str) -> bool:
        """Load a pretrained XGBoost model from disk"""
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost is not available. Cannot load model.")
                return False
            
            # Check if the file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the model
            if model_path.endswith('.json'):
                # Load from JSON format
                if self.model_type == "classifier":
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBClassifier(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True
                        )
                    else:
                        self.model = xgb.XGBClassifier()
                else:
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBRegressor(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True
                        )
                    else:
                        self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
            elif model_path.endswith('.pkl'):
                # Load from pickle format
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
                # If using PyTorch with XGBoost 3.0.0+, update the device
                if self.use_pytorch and XGBOOST_3_AVAILABLE and hasattr(self.model, 'set_params'):
                    self.model.set_params(
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu'
                    )
            else:
                # Try to load as binary format
                if self.model_type == "classifier":
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBClassifier(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True
                        )
                    else:
                        self.model = xgb.XGBClassifier()
                else:
                    if self.use_pytorch and XGBOOST_3_AVAILABLE:
                        self.model = xgb.XGBRegressor(
                            device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                            enable_categorical=True
                        )
                    else:
                        self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
            
            logger.info(f"Successfully loaded model from {model_path}")
            
            # Log PyTorch integration status
            if self.use_pytorch and XGBOOST_3_AVAILABLE:
                device = getattr(self.model, 'device', 'unknown')
                logger.info(f"Model loaded with PyTorch integration (device: {device})")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: str, format: str = 'json') -> bool:
        """Save the XGBoost model to disk"""
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot save model.")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            if format == 'json':
                # For PyTorch integration with XGBoost 3.0.0+, we need to handle device
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Get current device
                    current_device = getattr(self.model, 'device', None)
                    
                    # Temporarily set to CPU for saving if needed
                    if current_device == 'cuda':
                        logger.info("Temporarily setting device to CPU for saving model")
                        self.model.set_params(device='cpu')
                    
                    # Save the model
                    self.model.save_model(model_path)
                    
                    # Restore device setting
                    if current_device == 'cuda':
                        self.model.set_params(device='cuda')
                else:
                    # Standard save for non-PyTorch models
                    self.model.save_model(model_path)
            elif format == 'pkl':
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
            elif format == 'binary':
                # For PyTorch integration with XGBoost 3.0.0+, we need to handle device
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    # Get current device
                    current_device = getattr(self.model, 'device', None)
                    
                    # Temporarily set to CPU for saving if needed
                    if current_device == 'cuda':
                        logger.info("Temporarily setting device to CPU for saving model")
                        self.model.set_params(device='cpu')
                    
                    # Save the model
                    self.model.save_model(model_path)
                    
                    # Restore device setting
                    if current_device == 'cuda':
                        self.model.set_params(device='cuda')
                else:
                    # Standard save for non-PyTorch models
                    self.model.save_model(model_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Successfully saved model to {model_path}")
            
            # Log PyTorch integration status
            if self.use_pytorch and XGBOOST_3_AVAILABLE:
                device = getattr(self.model, 'device', 'unknown')
                logger.info(f"Model saved with PyTorch integration (device: {device})")
                
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with the loaded model"""
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot make predictions.")
                return np.array([])
            
            # For PyTorch integration with XGBoost 3.0.0+
            if self.use_pytorch and XGBOOST_3_AVAILABLE:
                # Convert input to PyTorch tensor if needed
                if TORCH_AVAILABLE and isinstance(features, np.ndarray):
                    # Log the conversion
                    logger.debug("Converting numpy array to PyTorch tensor for prediction")
                    
                    # Make predictions directly with the model
                    predictions = self.model.predict(features)
                    
                    # Log device information
                    device = getattr(self.model, 'device', 'unknown')
                    logger.debug(f"Made predictions with PyTorch integration (device: {device})")
                else:
                    # Make predictions with DataFrame or other format
                    predictions = self.model.predict(features)
            else:
                # Standard prediction for non-PyTorch models
                predictions = self.model.predict(features)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def train(self, features: pd.DataFrame, target: pd.Series, eval_set=None, **kwargs) -> bool:
        """Train the XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot train model.")
                return False
            
            # Store feature names
            self.feature_names = features.columns.tolist()
            
            # For PyTorch integration with XGBoost 3.0.0+
            if self.use_pytorch and XGBOOST_3_AVAILABLE:
                # Log PyTorch integration
                device = getattr(self.model, 'device', 'unknown')
                logger.info(f"Training model with PyTorch integration (device: {device})")
                
                # Check if we need to enable mixed precision training
                if TORCH_AVAILABLE and CUDA_AVAILABLE and kwargs.get('enable_amp', False):
                    logger.info("Enabling automatic mixed precision (AMP) for training")
                    # Remove our custom parameter so it doesn't interfere with XGBoost
                    kwargs.pop('enable_amp')
                    
                    # Train with mixed precision
                    with torch.cuda.amp.autocast():
                        self.model.fit(features, target, eval_set=eval_set, **kwargs)
                else:
                    # Standard training with PyTorch backend
                    self.model.fit(features, target, eval_set=eval_set, **kwargs)
            else:
                # Standard training
                self.model.fit(features, target, eval_set=eval_set, **kwargs)
            
            logger.info("Successfully trained model")
            return True
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def tune_hyperparameters(self, features: pd.DataFrame, target: pd.Series,
                            param_grid: Dict = None, cv: int = 5,
                            method: str = 'grid', n_iter: int = 10,
                            time_series: bool = False) -> Dict:
        """
        Tune hyperparameters using GridSearchCV or RandomizedSearchCV
        
        Args:
            features: DataFrame of features
            target: Series of target values
            param_grid: Dictionary of hyperparameters to search
            cv: Number of cross-validation folds
            method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            n_iter: Number of iterations for RandomizedSearchCV
            time_series: Whether to use TimeSeriesSplit for cross-validation
            
        Returns:
            Dictionary of best parameters
        """
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost is not available. Cannot tune hyperparameters.")
                return {}
            
            # Default parameter grid if not provided
            if param_grid is None:
                if self.model_type == "classifier":
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2]
                    }
                else:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2]
                    }
            
            # Create cross-validation strategy
            if time_series:
                cv_strategy = TimeSeriesSplit(n_splits=cv)
            else:
                cv_strategy = cv
            
            # Create base model
            if self.model_type == "classifier":
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    base_model = xgb.XGBClassifier(
                        tree_method='gpu_hist' if self.use_gpu else 'hist',
                        eval_metric='logloss',
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        enable_categorical=True,
                        objective='binary:logistic'
                    )
                    logger.info("Using XGBoost classifier with PyTorch integration for hyperparameter tuning")
                else:
                    base_model = xgb.XGBClassifier(
                        tree_method='gpu_hist' if self.use_gpu else 'hist',
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    logger.info("Using standard XGBoost classifier for hyperparameter tuning")
            else:
                if self.use_pytorch and XGBOOST_3_AVAILABLE:
                    base_model = xgb.XGBRegressor(
                        tree_method='gpu_hist' if self.use_gpu else 'hist',
                        eval_metric='rmse',
                        device='cuda' if self.use_gpu and CUDA_AVAILABLE else 'cpu',
                        enable_categorical=True,
                        objective='reg:squarederror'
                    )
                    logger.info("Using XGBoost regressor with PyTorch integration for hyperparameter tuning")
                else:
                    base_model = xgb.XGBRegressor(
                        tree_method='gpu_hist' if self.use_gpu else 'hist',
                        eval_metric='rmse'
                    )
                    logger.info("Using standard XGBoost regressor for hyperparameter tuning")
            
            # Create search object
            if method == 'grid':
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring='accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error',
                    verbose=1,
                    n_jobs=-1
                )
            else:  # random search
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_strategy,
                    scoring='accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error',
                    verbose=1,
                    n_jobs=-1
                )
            
            # Fit the search
            search.fit(features, target)
            
            # Update model with best parameters
            self.model = search.best_estimator_
            
            logger.info(f"Best parameters found: {search.best_params_}")
            logger.info(f"Best score: {search.best_score_}")
            
            return search.best_params_
            
        except Exception as e:
            logger.error(f"Error tuning hyperparameters: {e}")
            return {}
    
    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Evaluate the model performance
        
        Args:
            features: DataFrame of features
            target: Series of target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot evaluate model.")
                return {}
            
            # Make predictions
            predictions = self.predict(features)
            
            # Calculate metrics based on model type
            metrics = {}
            if self.model_type == "classifier":
                # Classification metrics
                metrics['accuracy'] = accuracy_score(target, predictions)
                metrics['precision'] = precision_score(target, predictions, average='weighted')
                metrics['recall'] = recall_score(target, predictions, average='weighted')
                metrics['f1'] = f1_score(target, predictions, average='weighted')
                
                # ROC AUC if binary classification
                if len(np.unique(target)) == 2:
                    try:
                        # Get probability predictions
                        prob_predictions = self.model.predict_proba(features)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(target, prob_predictions)
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC score: {e}")
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(target, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(target, predictions)
                metrics['r2'] = r2_score(target, predictions)
            
            # Feature importance
            try:
                importance = self.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                metrics['feature_importance'] = feature_importance.to_dict('records')
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {e}")
            
            logger.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    def create_pipeline(self, numeric_features=None, categorical_features=None):
        """
        Create a scikit-learn pipeline with preprocessing steps and the XGBoost model
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            
        Returns:
            sklearn Pipeline object
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot create pipeline.")
                return None
            
            # If feature lists not provided, use all features as numeric
            if numeric_features is None and categorical_features is None:
                if self.feature_names:
                    numeric_features = self.feature_names
                else:
                    logger.error("No feature names available. Please provide feature lists.")
                    return None
            
            # Create preprocessing steps for numeric features
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Create preprocessing steps for categorical features if provided
            preprocessor = None
            if categorical_features:
                from sklearn.preprocessing import OneHotEncoder
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                # Combine preprocessing steps
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])
            else:
                # Only numeric features
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features)
                    ])
            
            # Create the full pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', self.model)
            ])
            
            logger.info("Created scikit-learn pipeline with preprocessing steps")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            return None
    
    def cross_validate(self, features: pd.DataFrame, target: pd.Series,
                      cv: int = 5, time_series: bool = False,
                      scoring: str = None) -> Dict:
        """
        Perform cross-validation on the model
        
        Args:
            features: DataFrame of features
            target: Series of target values
            cv: Number of cross-validation folds
            time_series: Whether to use TimeSeriesSplit for cross-validation
            scoring: Scoring metric to use
            
        Returns:
            Dictionary of cross-validation results
        """
        try:
            if not XGBOOST_AVAILABLE or self.model is None:
                logger.error("XGBoost is not available or no model loaded. Cannot perform cross-validation.")
                return {}
            
            # Set default scoring based on model type
            if scoring is None:
                scoring = 'accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error'
            
            # Create cross-validation strategy
            if time_series:
                cv_strategy = TimeSeriesSplit(n_splits=cv)
            else:
                cv_strategy = cv
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                self.model,
                features,
                target,
                cv=cv_strategy,
                scoring=scoring
            )
            
            # Calculate statistics
            results = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'min_score': np.min(cv_scores),
                'max_score': np.max(cv_scores),
                'all_scores': cv_scores.tolist()
            }
            
            logger.info(f"Cross-validation results: Mean={results['mean_score']:.4f}, Std={results['std_score']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error performing cross-validation: {e}")
            return {}


def create_pretrained_model(save_path: str = 'models/xgboost/pretrained_model.json', use_pytorch: bool = True):
    """Create and save a pretrained XGBoost model for demonstration purposes"""
    try:
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost is not available. Cannot create pretrained model.")
            return False
        
        # Check if PyTorch integration is available
        use_pytorch_integration = use_pytorch and TORCH_AVAILABLE and XGBOOST_3_AVAILABLE
        
        # Create a simple dataset
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Create and train a model
        model = XGBoostModel(
            model_type='classifier',
            use_pytorch=use_pytorch,
            use_gpu=CUDA_AVAILABLE
        )
        
        # Log PyTorch integration status
        if use_pytorch_integration:
            logger.info("Creating pretrained model with PyTorch integration")
        else:
            logger.info("Creating standard pretrained model without PyTorch integration")
        
        # Train the model
        model.train(pd.DataFrame(X, columns=feature_names), pd.Series(y))
        
        # Save the model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_model(save_path)
        
        logger.info(f"Created and saved pretrained model to {save_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating pretrained model: {e}")
        return False


def load_pretrained_model(model_path: str = 'models/xgboost/pretrained_model.json', use_pytorch: bool = True):
    """Load a pretrained XGBoost model"""
    # Check if PyTorch integration is available
    use_pytorch_integration = use_pytorch and TORCH_AVAILABLE and XGBOOST_3_AVAILABLE
    
    if use_pytorch_integration:
        logger.info(f"Loading pretrained model with PyTorch integration from {model_path}")
    else:
        logger.info(f"Loading standard pretrained model from {model_path}")
    
    model = XGBoostModel(
        model_path=model_path,
        use_pytorch=use_pytorch,
        use_gpu=CUDA_AVAILABLE
    )
    return model


if __name__ == "__main__":
    # Create a pretrained model with PyTorch integration
    create_pretrained_model(use_pytorch=True)
    
    # Load the pretrained model with PyTorch integration
    model = load_pretrained_model(use_pytorch=True)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
    
    print("1. Basic prediction with pretrained model:")
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions[:5]}...")
    
    print("\n2. Hyperparameter tuning:")
    # Define a smaller parameter grid for demonstration
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    best_params = model.tune_hyperparameters(X_train, y_train, param_grid=param_grid, cv=3)
    print(f"Best parameters: {best_params}")
    
    print("\n3. Model evaluation:")
    metrics = model.evaluate(X_test, y_test)
    print(f"Evaluation metrics: {metrics}")
    
    print("\n4. Cross-validation:")
    cv_results = model.cross_validate(X_df, y_series, cv=3)
    print(f"Cross-validation results: {cv_results}")
    
    print("\n5. Creating a preprocessing pipeline:")
    # Create numeric and categorical features for demonstration
    X_with_cat = X_df.copy()
    X_with_cat['categorical_feature'] = np.random.choice(['A', 'B', 'C'], size=len(X_df))
    
    pipeline = model.create_pipeline(
        numeric_features=feature_names,
        categorical_features=['categorical_feature']
    )
    
    if pipeline:
        print("Pipeline created successfully")
        # Fit the pipeline
        pipeline.fit(X_with_cat, y_series)
        print("Pipeline fitted successfully")
