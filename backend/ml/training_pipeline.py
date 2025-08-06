"""
ML Training Pipeline for GoldenSignalsAI V5
Comprehensive training system with LSTM, Transformer, and ensemble models
Migrated and enhanced from archive with production improvements
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

from core.logging import get_logger
from core.config import settings

logger = get_logger(__name__)


# ==================== Model Architectures ====================

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, 
                 input_dim: int = 10, 
                 hidden_dim: int = 128, 
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use last output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerModel(nn.Module):
    """Transformer model for financial time series"""
    
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use last output
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EnsembleModel(nn.Module):
    """Ensemble of LSTM and Transformer models"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average of predictions
        stacked = torch.stack(outputs, dim=0)
        weights = self.weights.to(x.device).view(-1, 1, 1)
        weighted = stacked * weights
        
        return torch.sum(weighted, dim=0)


# ==================== Feature Engineering ====================

class FeatureEngineer:
    """Advanced feature engineering for trading ML models"""
    
    @staticmethod
    def extract_price_features(prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Extract features from price series"""
        features = []
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        features.append(np.mean(returns[-window:]))
        features.append(np.std(returns[-window:]))
        
        # Moving averages
        ma_5 = np.mean(prices[-5:])
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma_20
        
        features.append(prices[-1] / ma_5 - 1)  # Price relative to MA5
        features.append(prices[-1] / ma_20 - 1)  # Price relative to MA20
        features.append(ma_5 / ma_20 - 1)  # MA crossover signal
        
        # Volatility
        features.append(np.std(returns[-window:]) * np.sqrt(252))  # Annualized volatility
        
        # Momentum
        features.append((prices[-1] / prices[-window] - 1) if len(prices) >= window else 0)
        
        # RSI
        gains = returns[returns > 0][-window:] if len(returns) > 0 else [0]
        losses = -returns[returns < 0][-window:] if len(returns) > 0 else [0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)  # Normalized RSI
        
        return np.array(features)
    
    @staticmethod
    def extract_volume_features(volumes: np.ndarray, window: int = 20) -> np.ndarray:
        """Extract features from volume series"""
        features = []
        
        # Volume ratios
        features.append(volumes[-1] / np.mean(volumes[-window:]))
        features.append(np.std(volumes[-window:]) / np.mean(volumes[-window:]))
        
        # Volume trend
        if len(volumes) >= window:
            volume_trend = np.polyfit(np.arange(window), volumes[-window:], 1)[0]
            features.append(volume_trend)
        else:
            features.append(0)
        
        return np.array(features)
    
    @staticmethod
    def extract_options_features(options_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from options flow"""
        features = []
        
        # Put/Call ratio
        pc_ratio = options_data.get('put_volume', 0) / max(options_data.get('call_volume', 1), 1)
        features.append(pc_ratio)
        
        # Smart money score
        features.append(options_data.get('smart_money_score', 50) / 100)
        
        # Implied volatility
        features.append(options_data.get('implied_volatility', 0.2))
        
        # Options flow sentiment
        sentiment_map = {
            'bullish': 1.0,
            'bearish': -1.0,
            'neutral': 0.0
        }
        sentiment = sentiment_map.get(options_data.get('sentiment', 'neutral'), 0.0)
        features.append(sentiment)
        
        return np.array(features)
    
    @classmethod
    def create_feature_vector(cls,
                            price_data: np.ndarray,
                            volume_data: np.ndarray,
                            options_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Create complete feature vector"""
        price_features = cls.extract_price_features(price_data)
        volume_features = cls.extract_volume_features(volume_data)
        
        if options_data:
            options_features = cls.extract_options_features(options_data)
            return np.concatenate([price_features, volume_features, options_features])
        
        return np.concatenate([price_features, volume_features])


# ==================== Dataset Classes ====================

class TradingDataset(Dataset):
    """Custom dataset for trading data"""
    
    def __init__(self, 
                 features: np.ndarray, 
                 labels: np.ndarray,
                 sequence_length: int = 60):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        X = self.features[idx:idx + self.sequence_length]
        # Get corresponding label
        y = self.labels[idx + self.sequence_length]
        
        return X, y


# ==================== Training Pipeline ====================

class MLTrainingPipeline:
    """
    Comprehensive ML training pipeline for GoldenSignalsAI V5
    """
    
    def __init__(self,
                 model_type: str = 'lstm',
                 model_dir: str = 'models',
                 data_dir: str = 'data'):
        """
        Initialize training pipeline
        
        Args:
            model_type: Type of model ('lstm', 'transformer', 'ensemble')
            model_dir: Directory to save models
            data_dir: Directory containing training data
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_engineer = FeatureEngineer()
        
        # Training metrics
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'best_metrics': {}
        }
        
        logger.info(f"Initialized {model_type.upper()} training pipeline")
    
    def create_model(self, input_dim: int, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create model based on type and configuration"""
        if config is None:
            config = {}
        
        if self.model_type == 'lstm':
            model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('num_layers', 3),
                dropout=config.get('dropout', 0.2)
            )
        elif self.model_type == 'transformer':
            model = TransformerModel(
                input_dim=input_dim,
                hidden_dim=config.get('hidden_dim', 128),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 3),
                dropout=config.get('dropout', 0.2)
            )
        elif self.model_type == 'ensemble':
            # Create both models
            lstm = LSTMModel(input_dim=input_dim, **config.get('lstm', {}))
            transformer = TransformerModel(input_dim=input_dim, **config.get('transformer', {}))
            model = EnsembleModel([lstm, transformer])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model.to(self.device)
    
    def prepare_data(self,
                    data: pd.DataFrame,
                    target_col: str = 'target',
                    sequence_length: int = 60,
                    test_size: float = 0.2,
                    val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data for training
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Extract features and labels
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = TradingDataset(X_train_scaled, y_train, sequence_length)
        val_dataset = TradingDataset(X_val_scaled, y_val, sequence_length)
        test_dataset = TradingDataset(X_test_scaled, y_test, sequence_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 100,
             learning_rate: float = 0.001,
             early_stopping_patience: int = 10) -> Dict[str, Any]:
        """
        Train the model with advanced techniques
        """
        if self.model is None:
            # Infer input dimension from data
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch[0].shape[-1]
            self.model = self.create_model(input_dim)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            # Calculate average losses
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            # Update training history
            self.training_history['train_losses'].append(avg_train_loss)
            self.training_history['val_losses'].append(avg_val_loss)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                self.save_model(f'{self.model_type}_best.pth')
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={avg_val_loss:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Store best metrics
        self.training_history['best_metrics'] = {
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'epochs_trained': epoch + 1
        }
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
        return self.training_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                
                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Directional accuracy (for classification of up/down)
        direction_correct = np.sum(np.sign(predictions) == np.sign(targets))
        direction_accuracy = direction_correct / len(targets)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy)
        }
        
        logger.info(f"Test Metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Direction={direction_accuracy:.2%}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        self.model.eval()
        
        # Scale features if scaler is fitted
        try:
            features_scaled = self.scaler.transform(features)
        except:
            # If scaler not fitted, use raw features (for testing)
            logger.warning("Scaler not fitted, using raw features")
            features_scaled = features
        
        # Convert to tensor
        X = torch.FloatTensor(features_scaled).to(self.device)
        
        # Add batch dimension if needed
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filename: str):
        """Save model and training artifacts"""
        save_path = self.model_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model from file"""
        load_path = self.model_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Recreate model architecture
        # You need to know the input dimension
        # This should be stored in checkpoint for production
        self.model_type = checkpoint['model_type']
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {load_path}")
        
        return checkpoint


# ==================== Model Manager ====================

class ModelManager:
    """Manage multiple trained models"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.pipelines = {}
    
    def register_model(self, name: str, pipeline: MLTrainingPipeline):
        """Register a trained model"""
        self.pipelines[name] = pipeline
        logger.info(f"Model '{name}' registered")
    
    def get_model(self, name: str) -> MLTrainingPipeline:
        """Get a registered model"""
        if name not in self.pipelines:
            raise ValueError(f"Model '{name}' not found")
        return self.pipelines[name]
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.pipelines.keys())
    
    def ensemble_predict(self, features: np.ndarray, model_names: Optional[List[str]] = None) -> np.ndarray:
        """Make ensemble predictions using multiple models"""
        if model_names is None:
            model_names = list(self.pipelines.keys())
        
        predictions = []
        for name in model_names:
            if name in self.pipelines:
                pred = self.pipelines[name].predict(features)
                predictions.append(pred)
        
        # Average predictions
        return np.mean(predictions, axis=0)


# Global model manager instance
model_manager = ModelManager()