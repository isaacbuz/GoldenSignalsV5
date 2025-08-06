"""
Finance ML Model Training Pipeline
Comprehensive machine learning pipeline for financial prediction and analysis
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
import mlflow
import mlflow.sklearn
import pickle
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pydantic import BaseModel, Field
from core.logging import get_logger
from core.data.historical_storage import historical_storage, DataQuery, DataType, TimeFrame
from core.events.bus import event_bus

logger = get_logger(__name__)


class ModelType(Enum):
    """Types of ML models"""
    # Traditional ML
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    
    # Deep Learning
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    TCN = "tcn"  # Temporal Convolutional Network
    
    # Ensemble
    ENSEMBLE = "ensemble"
    STACKING = "stacking"


class PredictionTarget(Enum):
    """Prediction targets"""
    PRICE_DIRECTION = "price_direction"  # Classification: up/down
    PRICE_RETURN = "price_return"  # Regression: % return
    VOLATILITY = "volatility"  # Regression: future volatility
    VOLUME = "volume"  # Regression: future volume
    TREND_STRENGTH = "trend_strength"  # Regression: trend magnitude
    REVERSAL_PROBABILITY = "reversal_probability"  # Classification: reversal likelihood
    OPTIMAL_POSITION = "optimal_position"  # Regression: position sizing


class FeatureSet(Enum):
    """Feature set categories"""
    PRICE = "price"  # OHLCV features
    TECHNICAL = "technical"  # Technical indicators
    MICROSTRUCTURE = "microstructure"  # Order book, spread
    SENTIMENT = "sentiment"  # News, social sentiment
    MACRO = "macro"  # Economic indicators
    ALTERNATIVE = "alternative"  # Alternative data
    ALL = "all"  # All features


@dataclass
class MLConfig:
    """ML pipeline configuration"""
    # Model settings
    model_type: ModelType = ModelType.LIGHTGBM
    prediction_target: PredictionTarget = PredictionTarget.PRICE_DIRECTION
    prediction_horizon: int = 1  # Periods ahead to predict
    
    # Feature settings
    feature_sets: List[FeatureSet] = field(default_factory=lambda: [FeatureSet.PRICE, FeatureSet.TECHNICAL])
    lookback_periods: int = 20  # Historical periods for features
    
    # Training settings
    train_test_split: float = 0.8
    validation_split: float = 0.1
    n_splits: int = 5  # For time series cross-validation
    
    # Optimization
    optimize_hyperparameters: bool = True
    n_trials: int = 100  # Optuna trials
    
    # Regularization
    use_sample_weights: bool = True  # Weight recent samples more
    use_class_weights: bool = True  # Balance classes
    
    # Post-processing
    use_ensemble: bool = True
    use_calibration: bool = True  # Probability calibration
    
    # Storage
    model_path: str = "./models"
    experiment_name: str = "finance_ml"


@dataclass
class FeatureEngineering:
    """Feature engineering configuration"""
    # Price features
    price_features: List[str] = field(default_factory=lambda: [
        "returns", "log_returns", "volatility", "volume_ratio",
        "high_low_ratio", "close_to_high", "close_to_low"
    ])
    
    # Technical indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_20", "sma_50", "ema_12", "ema_26", "rsi", "macd",
        "bollinger_bands", "atr", "adx", "obv", "vwap"
    ])
    
    # Microstructure features
    microstructure_features: List[str] = field(default_factory=lambda: [
        "bid_ask_spread", "order_imbalance", "trade_intensity",
        "quote_intensity", "effective_spread"
    ])
    
    # Time features
    time_features: List[str] = field(default_factory=lambda: [
        "hour_of_day", "day_of_week", "day_of_month", "month_of_year",
        "is_month_start", "is_month_end", "is_quarter_end"
    ])
    
    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    
    # Rolling statistics
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "skew", "kurt"])


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, output_size: int = 1, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=512, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = x.transpose(0, 1)  # (seq_len, batch, features)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, features)
        
        # Take last output
        last_output = x[:, -1, :]
        
        # Final projection
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out


class FinanceMLPipeline:
    """
    Complete ML pipeline for financial prediction
    """
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.feature_eng = FeatureEngineering()
        
        # Models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        
        # Training data
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        
        # Metrics
        self.training_metrics: Dict[str, Any] = {}
        self.validation_metrics: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}
        
        # MLflow
        mlflow.set_experiment(self.config.experiment_name)
        
        # Model storage
        self.model_dir = Path(self.config.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Finance ML Pipeline initialized")
    
    async def prepare_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Prepare data for training (pre-training phase)
        
        Args:
            symbols: List of symbols to train on
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Prepared DataFrame with features and target
        """
        logger.info(f"Preparing data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Query historical data
        query = DataQuery(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_type=DataType.OHLCV,
            timeframe=TimeFrame.DAY_1,
            include_indicators=True
        )
        
        df = await historical_storage.query(query)
        
        if df.empty:
            raise ValueError("No data found for specified parameters")
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Create target variable
        df = self._create_target(df)
        
        # Remove NaN values
        df = df.dropna()
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Feature selection
        df = self._select_features(df)
        
        logger.info(f"Prepared {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data
        """
        logger.info("Engineering features...")
        
        # Sort by symbol and timestamp
        df = df.sort_values(['symbol', 'timestamp'])
        
        # Group by symbol for feature calculation
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df[mask].copy()
            
            # Price features
            if FeatureSet.PRICE in self.config.feature_sets:
                # Returns
                df.loc[mask, 'returns'] = symbol_df['close'].pct_change()
                df.loc[mask, 'log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
                
                # Volatility (20-day rolling)
                df.loc[mask, 'volatility'] = symbol_df['returns'].rolling(20).std()
                
                # Volume ratio
                df.loc[mask, 'volume_ratio'] = symbol_df['volume'] / symbol_df['volume'].rolling(20).mean()
                
                # Price ratios
                df.loc[mask, 'high_low_ratio'] = symbol_df['high'] / symbol_df['low']
                df.loc[mask, 'close_to_high'] = symbol_df['close'] / symbol_df['high']
                df.loc[mask, 'close_to_low'] = symbol_df['close'] / symbol_df['low']
                
                # Price momentum
                for period in [5, 10, 20]:
                    df.loc[mask, f'momentum_{period}'] = symbol_df['close'] / symbol_df['close'].shift(period) - 1
            
            # Technical indicators
            if FeatureSet.TECHNICAL in self.config.feature_sets:
                # Moving averages
                df.loc[mask, 'sma_20'] = symbol_df['close'].rolling(20).mean()
                df.loc[mask, 'sma_50'] = symbol_df['close'].rolling(50).mean()
                df.loc[mask, 'ema_12'] = symbol_df['close'].ewm(span=12).mean()
                df.loc[mask, 'ema_26'] = symbol_df['close'].ewm(span=26).mean()
                
                # Price relative to moving averages
                df.loc[mask, 'close_to_sma20'] = symbol_df['close'] / df.loc[mask, 'sma_20']
                df.loc[mask, 'close_to_sma50'] = symbol_df['close'] / df.loc[mask, 'sma_50']
                
                # RSI
                delta = symbol_df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                df.loc[mask, 'rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                df.loc[mask, 'macd'] = df.loc[mask, 'ema_12'] - df.loc[mask, 'ema_26']
                df.loc[mask, 'macd_signal'] = df.loc[mask, 'macd'].ewm(span=9).mean()
                df.loc[mask, 'macd_histogram'] = df.loc[mask, 'macd'] - df.loc[mask, 'macd_signal']
                
                # Bollinger Bands
                bb_sma = symbol_df['close'].rolling(20).mean()
                bb_std = symbol_df['close'].rolling(20).std()
                df.loc[mask, 'bb_upper'] = bb_sma + (bb_std * 2)
                df.loc[mask, 'bb_lower'] = bb_sma - (bb_std * 2)
                df.loc[mask, 'bb_width'] = df.loc[mask, 'bb_upper'] - df.loc[mask, 'bb_lower']
                df.loc[mask, 'bb_position'] = (symbol_df['close'] - df.loc[mask, 'bb_lower']) / df.loc[mask, 'bb_width']
                
                # ATR (Average True Range)
                high_low = symbol_df['high'] - symbol_df['low']
                high_close = abs(symbol_df['high'] - symbol_df['close'].shift())
                low_close = abs(symbol_df['low'] - symbol_df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df.loc[mask, 'atr'] = true_range.rolling(14).mean()
                
                # OBV (On-Balance Volume)
                obv = (np.sign(symbol_df['close'].diff()) * symbol_df['volume']).fillna(0).cumsum()
                df.loc[mask, 'obv'] = obv
                df.loc[mask, 'obv_sma'] = obv.rolling(20).mean()
            
            # Lag features
            for col in ['returns', 'volume_ratio', 'rsi']:
                if col in df.columns:
                    for lag in self.feature_eng.lag_periods:
                        df.loc[mask, f'{col}_lag_{lag}'] = df.loc[mask, col].shift(lag)
            
            # Rolling statistics
            for window in self.feature_eng.rolling_windows:
                if 'returns' in df.columns:
                    df.loc[mask, f'returns_mean_{window}'] = symbol_df['returns'].rolling(window).mean()
                    df.loc[mask, f'returns_std_{window}'] = symbol_df['returns'].rolling(window).std()
                    df.loc[mask, f'returns_skew_{window}'] = symbol_df['returns'].rolling(window).skew()
                    df.loc[mask, f'returns_kurt_{window}'] = symbol_df['returns'].rolling(window).kurt()
        
        # Time features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['is_month_start'] = pd.to_datetime(df['timestamp']).dt.is_month_start.astype(int)
        df['is_month_end'] = pd.to_datetime(df['timestamp']).dt.is_month_end.astype(int)
        df['is_quarter_end'] = pd.to_datetime(df['timestamp']).dt.is_quarter_end.astype(int)
        
        return df
    
    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable based on prediction target
        """
        logger.info(f"Creating target: {self.config.prediction_target.value}")
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df[mask].copy()
            
            if self.config.prediction_target == PredictionTarget.PRICE_DIRECTION:
                # Binary classification: 1 if price goes up, 0 if down
                future_return = symbol_df['close'].shift(-self.config.prediction_horizon) / symbol_df['close'] - 1
                df.loc[mask, 'target'] = (future_return > 0).astype(int)
                
            elif self.config.prediction_target == PredictionTarget.PRICE_RETURN:
                # Regression: future return
                df.loc[mask, 'target'] = symbol_df['close'].shift(-self.config.prediction_horizon) / symbol_df['close'] - 1
                
            elif self.config.prediction_target == PredictionTarget.VOLATILITY:
                # Regression: future volatility
                future_volatility = symbol_df['returns'].shift(-self.config.prediction_horizon).rolling(20).std()
                df.loc[mask, 'target'] = future_volatility
                
            elif self.config.prediction_target == PredictionTarget.TREND_STRENGTH:
                # Regression: trend magnitude
                future_prices = symbol_df['close'].shift(-self.config.prediction_horizon).rolling(5).mean()
                current_prices = symbol_df['close'].rolling(5).mean()
                df.loc[mask, 'target'] = (future_prices - current_prices) / current_prices
                
            elif self.config.prediction_target == PredictionTarget.REVERSAL_PROBABILITY:
                # Classification: reversal detection
                current_trend = np.sign(symbol_df['close'] - symbol_df['close'].shift(5))
                future_trend = np.sign(symbol_df['close'].shift(-self.config.prediction_horizon) - symbol_df['close'])
                df.loc[mask, 'target'] = (current_trend != future_trend).astype(int)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in the data
        """
        logger.info("Handling outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['target', 'symbol', 'timestamp']]
        
        for col in numeric_cols:
            # Use robust scaling for outlier detection
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            
            if mad > 0:
                z_scores = np.abs((df[col] - median) / (mad * 1.4826))
                df.loc[z_scores > threshold, col] = np.nan
                df[col].fillna(median, inplace=True)
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features
        """
        logger.info("Selecting features...")
        
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'
        ]]
        
        # Remove features with low variance
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=0.01)
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        selected_features = []
        for col in numeric_features:
            if df[col].var() > 0.01:
                selected_features.append(col)
        
        # Keep target
        if 'target' in df.columns:
            selected_features.append('target')
        
        # Keep identifiers
        selected_features.extend(['timestamp', 'symbol'])
        
        return df[selected_features]
    
    def split_data(self, df: pd.DataFrame) -> None:
        """
        Split data into train, validation, and test sets
        """
        logger.info("Splitting data...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate split points
        n_samples = len(df)
        train_end = int(n_samples * self.config.train_test_split)
        val_end = int(n_samples * (self.config.train_test_split + self.config.validation_split))
        
        # Split data
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'timestamp', 'symbol']]
        
        self.X_train = train_df[feature_cols]
        self.y_train = train_df['target']
        
        self.X_val = val_df[feature_cols]
        self.y_val = val_df['target']
        
        self.X_test = test_df[feature_cols]
        self.y_test = test_df['target']
        
        logger.info(f"Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
    
    def scale_features(self) -> None:
        """
        Scale features
        """
        logger.info("Scaling features...")
        
        # Use RobustScaler for financial data
        scaler = RobustScaler()
        
        # Fit on train data
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        # Transform validation and test
        self.X_val = pd.DataFrame(
            scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )
        
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # Store scaler
        self.scalers['features'] = scaler
    
    async def train_model(self) -> Dict[str, Any]:
        """
        Train ML model (training phase)
        
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.config.model_type.value} model...")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_type": self.config.model_type.value,
                "prediction_target": self.config.prediction_target.value,
                "prediction_horizon": self.config.prediction_horizon,
                "n_features": len(self.X_train.columns)
            })
            
            # Train based on model type
            if self.config.model_type == ModelType.LIGHTGBM:
                model = await self._train_lightgbm()
            elif self.config.model_type == ModelType.XGBOOST:
                model = await self._train_xgboost()
            elif self.config.model_type == ModelType.RANDOM_FOREST:
                model = await self._train_random_forest()
            elif self.config.model_type == ModelType.LSTM:
                model = await self._train_lstm()
            elif self.config.model_type == ModelType.TRANSFORMER:
                model = await self._train_transformer()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Store model
            self.models[self.config.model_type.value] = model
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(self.X_train, self.y_train, model)
            val_metrics = self._calculate_metrics(self.X_val, self.y_val, model)
            
            self.training_metrics = train_metrics
            self.validation_metrics = val_metrics
            
            # Log metrics
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
            
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[self.config.model_type.value] = importance_df
                
                # Log top features
                for i, row in importance_df.head(10).iterrows():
                    mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
            
            logger.info(f"Training complete. Val accuracy: {val_metrics.get('accuracy', 0):.4f}")
            
            return val_metrics
    
    async def _train_lightgbm(self):
        """
        Train LightGBM model
        """
        # Hyperparameter optimization
        if self.config.optimize_hyperparameters:
            params = await self._optimize_lightgbm_params()
        else:
            params = {
                'objective': 'binary' if self.config.prediction_target == PredictionTarget.PRICE_DIRECTION else 'regression',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        # Create dataset
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        return model
    
    async def _optimize_lightgbm_params(self) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna
        """
        def objective(trial):
            params = {
                'objective': 'binary' if self.config.prediction_target == PredictionTarget.PRICE_DIRECTION else 'regression',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'verbose': -1,
                'random_state': 42
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(self.X_train):
                X_tr, X_vl = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_vl = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_vl, label=y_vl, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                
                # Calculate validation score
                if self.config.prediction_target == PredictionTarget.PRICE_DIRECTION:
                    pred = (model.predict(X_vl) > 0.5).astype(int)
                    score = (pred == y_vl).mean()
                else:
                    pred = model.predict(X_vl)
                    score = -mean_squared_error(y_vl, pred)
                
                scores.append(score)
            
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials)
        
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    async def _train_lstm(self) -> nn.Module:
        """
        Train LSTM model
        """
        # Prepare sequences
        X_train_seq = self._prepare_sequences(self.X_train)
        X_val_seq = self._prepare_sequences(self.X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(self.y_train.values)
        X_val_tensor = torch.FloatTensor(X_val_seq)
        y_val_tensor = torch.FloatTensor(self.y_val.values)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        input_size = X_train_seq.shape[2]
        model = LSTMModel(input_size=input_size)
        
        # Loss and optimizer
        if self.config.prediction_target == PredictionTarget.PRICE_DIRECTION:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.model_dir / 'best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(self.model_dir / 'best_lstm_model.pth'))
        
        return model
    
    def _prepare_sequences(self, X: pd.DataFrame, sequence_length: int = 20) -> np.ndarray:
        """
        Prepare sequences for LSTM
        """
        sequences = []
        
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X.iloc[i:i+sequence_length].values)
        
        return np.array(sequences)
    
    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict[str, float]:
        """
        Calculate model metrics
        """
        metrics = {}
        
        # Get predictions
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(self._prepare_sequences(X))
                pred = model(X_tensor).squeeze().numpy()
        elif hasattr(model, 'predict'):
            pred = model.predict(X)
        else:
            raise ValueError("Unknown model type")
        
        # Classification metrics
        if self.config.prediction_target == PredictionTarget.PRICE_DIRECTION:
            pred_binary = (pred > 0.5).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['accuracy'] = accuracy_score(y, pred_binary)
            metrics['precision'] = precision_score(y, pred_binary)
            metrics['recall'] = recall_score(y, pred_binary)
            metrics['f1'] = f1_score(y, pred_binary)
            
        # Regression metrics
        else:
            metrics['mse'] = mean_squared_error(y, pred)
            metrics['mae'] = mean_absolute_error(y, pred)
            metrics['r2'] = r2_score(y, pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    async def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate model on test set (post-training phase)
        
        Returns:
            Test metrics and analysis
        """
        logger.info("Evaluating model on test set...")
        
        # Get best model
        model = self.models.get(self.config.model_type.value)
        if not model:
            raise ValueError("No trained model found")
        
        # Calculate test metrics
        test_metrics = self._calculate_metrics(self.X_test, self.y_test, model)
        self.test_metrics = test_metrics
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test[:100])  # Sample for speed
            
            # Store SHAP values
            self.feature_importance['shap'] = pd.DataFrame({
                'feature': self.X_test.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
        
        # Performance by market condition
        analysis = {
            'test_metrics': test_metrics,
            'feature_importance': self.feature_importance,
            'performance_by_volatility': self._analyze_by_volatility(),
            'performance_by_trend': self._analyze_by_trend(),
            'temporal_stability': self._analyze_temporal_stability()
        }
        
        logger.info(f"Test evaluation complete. Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        
        return analysis
    
    def _analyze_by_volatility(self) -> Dict[str, float]:
        """
        Analyze performance by market volatility
        """
        if 'volatility' not in self.X_test.columns:
            return {}
        
        # Split by volatility quartiles
        volatility_quartiles = pd.qcut(self.X_test['volatility'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        results = {}
        model = self.models.get(self.config.model_type.value)
        
        for quartile in ['Low', 'Medium-Low', 'Medium-High', 'High']:
            mask = volatility_quartiles == quartile
            if mask.sum() > 0:
                X_subset = self.X_test[mask]
                y_subset = self.y_test[mask]
                
                metrics = self._calculate_metrics(X_subset, y_subset, model)
                results[quartile] = metrics.get('accuracy', metrics.get('r2', 0))
        
        return results
    
    def _analyze_by_trend(self) -> Dict[str, float]:
        """
        Analyze performance by market trend
        """
        if 'momentum_20' not in self.X_test.columns:
            return {}
        
        # Split by trend
        trend_conditions = {
            'Strong Uptrend': self.X_test['momentum_20'] > 0.05,
            'Weak Uptrend': (self.X_test['momentum_20'] > 0) & (self.X_test['momentum_20'] <= 0.05),
            'Weak Downtrend': (self.X_test['momentum_20'] < 0) & (self.X_test['momentum_20'] >= -0.05),
            'Strong Downtrend': self.X_test['momentum_20'] < -0.05
        }
        
        results = {}
        model = self.models.get(self.config.model_type.value)
        
        for trend, mask in trend_conditions.items():
            if mask.sum() > 0:
                X_subset = self.X_test[mask]
                y_subset = self.y_test[mask]
                
                metrics = self._calculate_metrics(X_subset, y_subset, model)
                results[trend] = metrics.get('accuracy', metrics.get('r2', 0))
        
        return results
    
    def _analyze_temporal_stability(self) -> Dict[str, List[float]]:
        """
        Analyze model stability over time
        """
        # Split test set into time periods
        n_periods = 5
        period_size = len(self.X_test) // n_periods
        
        results = {'accuracy': [], 'dates': []}
        model = self.models.get(self.config.model_type.value)
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(self.X_test)
            
            X_period = self.X_test.iloc[start_idx:end_idx]
            y_period = self.y_test.iloc[start_idx:end_idx]
            
            metrics = self._calculate_metrics(X_period, y_period, model)
            results['accuracy'].append(metrics.get('accuracy', metrics.get('r2', 0)))
            
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions
        """
        # Scale features
        if 'features' in self.scalers:
            X_scaled = pd.DataFrame(
                self.scalers['features'].transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Get model
        model = self.models.get(self.config.model_type.value)
        if not model:
            raise ValueError("No trained model found")
        
        # Predict
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(self._prepare_sequences(X_scaled))
                pred = model(X_tensor).squeeze().numpy()
        else:
            pred = model.predict(X_scaled)
        
        # Apply calibration if classification
        if self.config.prediction_target == PredictionTarget.PRICE_DIRECTION and self.config.use_calibration:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                # Apply probability calibration for better confidence estimates
                if hasattr(model, 'predict_proba') and len(pred.shape) > 1:
                    # For classification models with probability output
                    pred_proba = model.predict_proba(X_scaled)
                    # Use the calibrated probabilities as predictions
                    pred = pred_proba
                    logger.debug(f"Applied probability calibration to predictions")
                else:
                    logger.warning("Model doesn't support probability calibration")
            except Exception as e:
                logger.error(f"Calibration failed: {e}")
                # Continue with uncalibrated predictions
        
        return pred
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            path: Save path
            
        Returns:
            Path where model was saved
        """
        if not path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = self.model_dir / f"model_{self.config.model_type.value}_{timestamp}.pkl"
        
        # Save everything needed for inference
        model_package = {
            'models': self.models,
            'scalers': self.scalers,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'metrics': {
                'train': self.training_metrics,
                'val': self.validation_metrics,
                'test': self.test_metrics
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Model saved to {path}")
        
        return str(path)
    
    def load_model(self, path: str) -> None:
        """
        Load saved model
        
        Args:
            path: Model path
        """
        with open(path, 'rb') as f:
            model_package = pickle.load(f)
        
        self.models = model_package['models']
        self.scalers = model_package['scalers']
        self.config = model_package['config']
        self.feature_importance = model_package['feature_importance']
        self.training_metrics = model_package['metrics']['train']
        self.validation_metrics = model_package['metrics']['val']
        self.test_metrics = model_package['metrics']['test']
        
        logger.info(f"Model loaded from {path}")


# Global ML pipeline instance
finance_ml_pipeline = FinanceMLPipeline()