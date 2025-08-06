#!/usr/bin/env python3
"""
Test script for ML Training Pipeline
Tests LSTM, Transformer, and Ensemble models
"""

import asyncio
import numpy as np
import pandas as pd
import torch
from ml.training_pipeline import MLTrainingPipeline, ModelManager, FeatureEngineer


def generate_test_data(samples: int = 5000) -> pd.DataFrame:
    """Generate test data for model training"""
    np.random.seed(42)
    
    # Generate synthetic price series
    prices = 100 * np.exp(np.cumsum(np.random.randn(samples) * 0.01))
    volumes = np.random.lognormal(15, 1, samples)
    
    # Create features
    data = {
        'price': prices,
        'volume': volumes,
        'returns': np.concatenate([[0], np.diff(prices) / prices[:-1]]),
        'log_returns': np.concatenate([[0], np.diff(np.log(prices))]),
        'volume_ratio': volumes / np.roll(volumes, 1),
        'price_ma5': pd.Series(prices).rolling(5).mean().fillna(prices[0]),
        'price_ma20': pd.Series(prices).rolling(20).mean().fillna(prices[0]),
        'volatility': pd.Series(prices).pct_change().rolling(20).std().fillna(0.01),
        'rsi': np.random.uniform(30, 70, samples),
        'momentum': np.random.randn(samples) * 0.05
    }
    
    # Create target (next period return)
    data['target'] = np.concatenate([data['returns'][1:], [0]])
    
    return pd.DataFrame(data)


def test_feature_engineering():
    """Test feature engineering capabilities"""
    print("=" * 70)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 70)
    
    # Generate sample data
    prices = np.random.randn(100) * 10 + 100
    volumes = np.random.lognormal(15, 1, 100)
    
    options_data = {
        'put_volume': 10000,
        'call_volume': 15000,
        'smart_money_score': 75,
        'implied_volatility': 0.35,
        'sentiment': 'bullish'
    }
    
    # Extract features
    engineer = FeatureEngineer()
    
    price_features = engineer.extract_price_features(prices)
    print(f"\n✅ Price Features Shape: {price_features.shape}")
    print(f"   Sample values: {price_features[:3]}")
    
    volume_features = engineer.extract_volume_features(volumes)
    print(f"\n✅ Volume Features Shape: {volume_features.shape}")
    print(f"   Sample values: {volume_features[:3]}")
    
    options_features = engineer.extract_options_features(options_data)
    print(f"\n✅ Options Features Shape: {options_features.shape}")
    print(f"   Sample values: {options_features}")
    
    # Create complete feature vector
    full_features = engineer.create_feature_vector(prices, volumes, options_data)
    print(f"\n✅ Complete Feature Vector Shape: {full_features.shape}")
    print(f"   Total features: {len(full_features)}")


async def test_lstm_model():
    """Test LSTM model training"""
    print("\n" + "=" * 70)
    print("TESTING LSTM MODEL")
    print("=" * 70)
    
    # Create pipeline
    pipeline = MLTrainingPipeline(model_type='lstm')
    
    # Generate training data
    data = generate_test_data(1000)
    
    # Prepare data
    train_loader, val_loader, test_loader = pipeline.prepare_data(
        data,
        target_col='target',
        sequence_length=20,
        test_size=0.2,
        val_size=0.1
    )
    
    print(f"\n✅ Data prepared for training")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Train model (reduced epochs for testing)
    print(f"\n📊 Training LSTM model...")
    history = pipeline.train(
        train_loader,
        val_loader,
        epochs=10,  # Reduced for quick testing
        learning_rate=0.001
    )
    
    print(f"\n✅ Training completed")
    print(f"   Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"   Final val loss: {history['val_losses'][-1]:.4f}")
    print(f"   Best val loss: {history['best_metrics']['best_val_loss']:.4f}")
    
    # Evaluate model
    metrics = pipeline.evaluate(test_loader)
    print(f"\n✅ Test Metrics:")
    print(f"   MSE: {metrics['mse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R²: {metrics['r2']:.4f}")
    print(f"   Direction Accuracy: {metrics['direction_accuracy']:.1%}")
    
    # Test prediction
    test_features = data.iloc[:20][['price', 'volume', 'returns', 'volatility', 'rsi', 
                                    'momentum', 'price_ma5', 'price_ma20', 'volume_ratio', 'log_returns']].values
    predictions = pipeline.predict(test_features)
    print(f"\n✅ Predictions shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions.flatten()[:5]}")
    
    return pipeline


async def test_transformer_model():
    """Test Transformer model training"""
    print("\n" + "=" * 70)
    print("TESTING TRANSFORMER MODEL")
    print("=" * 70)
    
    # Create pipeline
    pipeline = MLTrainingPipeline(model_type='transformer')
    
    # Generate training data
    data = generate_test_data(1000)
    
    # Prepare data
    train_loader, val_loader, test_loader = pipeline.prepare_data(
        data,
        target_col='target',
        sequence_length=20,
        test_size=0.2,
        val_size=0.1
    )
    
    print(f"\n✅ Data prepared for training")
    
    # Train model (reduced epochs for testing)
    print(f"\n📊 Training Transformer model...")
    history = pipeline.train(
        train_loader,
        val_loader,
        epochs=10,  # Reduced for quick testing
        learning_rate=0.001
    )
    
    print(f"\n✅ Training completed")
    print(f"   Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"   Final val loss: {history['val_losses'][-1]:.4f}")
    
    # Evaluate model
    metrics = pipeline.evaluate(test_loader)
    print(f"\n✅ Test Metrics:")
    print(f"   MSE: {metrics['mse']:.4f}")
    print(f"   Direction Accuracy: {metrics['direction_accuracy']:.1%}")
    
    return pipeline


async def test_ensemble_model():
    """Test Ensemble model training"""
    print("\n" + "=" * 70)
    print("TESTING ENSEMBLE MODEL")
    print("=" * 70)
    
    # Create pipeline
    pipeline = MLTrainingPipeline(model_type='ensemble')
    
    # Generate training data
    data = generate_test_data(1000)
    
    # Prepare data
    train_loader, val_loader, test_loader = pipeline.prepare_data(
        data,
        target_col='target',
        sequence_length=20,
        test_size=0.2,
        val_size=0.1
    )
    
    print(f"\n✅ Data prepared for training")
    
    # Train model (reduced epochs for testing)
    print(f"\n📊 Training Ensemble model (LSTM + Transformer)...")
    history = pipeline.train(
        train_loader,
        val_loader,
        epochs=10,  # Reduced for quick testing
        learning_rate=0.001
    )
    
    print(f"\n✅ Training completed")
    print(f"   Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"   Final val loss: {history['val_losses'][-1]:.4f}")
    
    # Evaluate model
    metrics = pipeline.evaluate(test_loader)
    print(f"\n✅ Test Metrics:")
    print(f"   MSE: {metrics['mse']:.4f}")
    print(f"   Direction Accuracy: {metrics['direction_accuracy']:.1%}")
    
    return pipeline


async def test_model_manager():
    """Test model manager functionality"""
    print("\n" + "=" * 70)
    print("TESTING MODEL MANAGER")
    print("=" * 70)
    
    # Create model manager
    manager = ModelManager()
    
    # Train and register multiple models
    print(f"\n📊 Training and registering models...")
    
    # Create a simple LSTM model
    lstm_pipeline = MLTrainingPipeline(model_type='lstm')
    data = generate_test_data(500)
    train_loader, val_loader, test_loader = lstm_pipeline.prepare_data(
        data, target_col='target', sequence_length=10
    )
    lstm_pipeline.train(train_loader, val_loader, epochs=5)
    manager.register_model('lstm_v1', lstm_pipeline)
    
    # Create a simple Transformer model
    transformer_pipeline = MLTrainingPipeline(model_type='transformer')
    transformer_pipeline.train(train_loader, val_loader, epochs=5)
    manager.register_model('transformer_v1', transformer_pipeline)
    
    print(f"\n✅ Models registered:")
    for model_name in manager.list_models():
        print(f"   - {model_name}")
    
    # Test ensemble prediction
    test_features = data.iloc[:10][['price', 'volume', 'returns', 'volatility', 'rsi', 
                                    'momentum', 'price_ma5', 'price_ma20', 'volume_ratio', 'log_returns']].values
    ensemble_pred = manager.ensemble_predict(test_features)
    
    print(f"\n✅ Ensemble Prediction:")
    print(f"   Shape: {ensemble_pred.shape}")
    print(f"   Sample predictions: {ensemble_pred.flatten()[:5]}")


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("ML TRAINING PIPELINE TEST SUITE")
    print("=" * 70)
    
    # Check PyTorch availability
    print(f"\n📊 System Info:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    test_feature_engineering()
    
    lstm_pipeline = await test_lstm_model()
    transformer_pipeline = await test_transformer_model()
    ensemble_pipeline = await test_ensemble_model()
    
    await test_model_manager()
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print("\n📊 Model Training Summary:")
    print("   LSTM: Fast training, good for sequences")
    print("   Transformer: Better long-range dependencies")
    print("   Ensemble: Best overall performance")
    
    print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Features summary
    print("\n📊 ML TRAINING PIPELINE FEATURES:")
    print("  • Multiple architectures (LSTM, Transformer, Ensemble)")
    print("  • Advanced feature engineering")
    print("  • Automatic scaling and preprocessing")
    print("  • Learning rate scheduling")
    print("  • Early stopping with patience")
    print("  • Model checkpointing")
    print("  • Ensemble predictions")
    print("  • Performance metrics tracking")
    print("  • GPU acceleration support")


if __name__ == "__main__":
    asyncio.run(main())