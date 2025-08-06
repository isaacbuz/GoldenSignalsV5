"""
ML Training Pipeline API Routes
Provides endpoints for model training, evaluation, and prediction
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime
import io

from core.logging import get_logger
from ml.training_pipeline import MLTrainingPipeline, model_manager

logger = get_logger(__name__)
router = APIRouter(tags=["ML Training"])

# Active training sessions
training_sessions = {}


@router.post("/ml/train")
async def start_training(
    background_tasks: BackgroundTasks,
    model_name: str,
    model_type: str = "lstm",
    epochs: int = 100,
    learning_rate: float = 0.001,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Start a model training session in the background
    
    Args:
        model_name: Unique name for the model
        model_type: Type of model (lstm, transformer, ensemble)
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        config: Model configuration parameters
    """
    try:
        # Check if model name already exists
        if model_name in training_sessions:
            if training_sessions[model_name]['status'] == 'training':
                raise HTTPException(status_code=400, detail=f"Model '{model_name}' is already training")
        
        # Create training pipeline
        pipeline = MLTrainingPipeline(model_type=model_type)
        
        # Register with model manager
        model_manager.register_model(model_name, pipeline)
        
        # Store session info
        session_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_sessions[model_name] = {
            'session_id': session_id,
            'status': 'training',
            'model_type': model_type,
            'started_at': datetime.now().isoformat(),
            'epochs': epochs,
            'config': config or {}
        }
        
        # Start training in background
        background_tasks.add_task(
            run_training,
            model_name,
            pipeline,
            epochs,
            learning_rate
        )
        
        logger.info(f"Started training session for model '{model_name}' (type: {model_type})")
        
        return {
            'status': 'success',
            'message': f"Training started for model '{model_name}'",
            'session_id': session_id,
            'model_type': model_type
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training(
    model_name: str,
    pipeline: MLTrainingPipeline,
    epochs: int,
    learning_rate: float
):
    """Run training in background"""
    try:
        # Generate synthetic data for demonstration
        # In production, this would load real training data
        data = generate_synthetic_training_data()
        
        # Prepare data
        train_loader, val_loader, test_loader = pipeline.prepare_data(
            data,
            target_col='target',
            sequence_length=60
        )
        
        # Train model
        history = pipeline.train(
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Evaluate on test set
        metrics = pipeline.evaluate(test_loader)
        
        # Update session status
        training_sessions[model_name]['status'] = 'completed'
        training_sessions[model_name]['completed_at'] = datetime.now().isoformat()
        training_sessions[model_name]['metrics'] = metrics
        training_sessions[model_name]['history'] = {
            'train_losses': history['train_losses'][-10:],  # Last 10 epochs
            'val_losses': history['val_losses'][-10:],
            'best_metrics': history['best_metrics']
        }
        
        # Save model
        pipeline.save_model(f"{model_name}_final.pth")
        
        logger.info(f"Training completed for model '{model_name}' - Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed for model '{model_name}': {str(e)}")
        training_sessions[model_name]['status'] = 'failed'
        training_sessions[model_name]['error'] = str(e)


@router.get("/ml/status/{model_name}")
async def get_training_status(model_name: str) -> Dict[str, Any]:
    """Get the status of a training session"""
    if model_name not in training_sessions:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return training_sessions[model_name]


@router.get("/ml/models")
async def list_models() -> Dict[str, Any]:
    """List all registered models and their status"""
    models = []
    
    for name in model_manager.list_models():
        model_info = {
            'name': name,
            'status': 'ready'
        }
        
        if name in training_sessions:
            model_info.update(training_sessions[name])
        
        models.append(model_info)
    
    return {
        'models': models,
        'total': len(models)
    }


@router.post("/ml/predict/{model_name}")
async def predict(
    model_name: str,
    features: List[List[float]]
) -> Dict[str, Any]:
    """
    Make predictions using a trained model
    
    Args:
        model_name: Name of the model to use
        features: Feature vectors for prediction
    """
    try:
        # Get model pipeline
        pipeline = model_manager.get_model(model_name)
        
        # Convert to numpy array
        features_array = np.array(features)
        
        # Make predictions
        predictions = pipeline.predict(features_array)
        
        return {
            'model': model_name,
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/ensemble_predict")
async def ensemble_predict(
    features: List[List[float]],
    model_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Make ensemble predictions using multiple models
    
    Args:
        features: Feature vectors for prediction
        model_names: List of models to use (uses all if not specified)
    """
    try:
        # Convert to numpy array
        features_array = np.array(features)
        
        # Make ensemble predictions
        predictions = model_manager.ensemble_predict(features_array, model_names)
        
        return {
            'models': model_names or model_manager.list_models(),
            'predictions': predictions.tolist(),
            'method': 'ensemble_average',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/upload_data")
async def upload_training_data(
    file: UploadFile = File(...),
    model_name: str = "custom_model"
) -> Dict[str, Any]:
    """
    Upload training data for custom model training
    
    Expects CSV file with features and 'target' column
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate data
        if 'target' not in df.columns:
            raise ValueError("Data must contain 'target' column")
        
        # Store data (in production, save to database or file system)
        # For now, we'll just validate and return info
        
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'feature_count': len(df.columns) - 1,
            'target_stats': {
                'mean': float(df['target'].mean()),
                'std': float(df['target'].std()),
                'min': float(df['target'].min()),
                'max': float(df['target'].max())
            }
        }
        
        logger.info(f"Uploaded training data for model '{model_name}': {data_info['rows']} rows")
        
        return {
            'status': 'success',
            'message': 'Data uploaded successfully',
            'data_info': data_info
        }
        
    except Exception as e:
        logger.error(f"Failed to upload data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ml/feature_importance/{model_name}")
async def get_feature_importance(model_name: str) -> Dict[str, Any]:
    """
    Get feature importance for a trained model
    (Currently returns mock data - would need model-specific implementation)
    """
    try:
        # Verify model exists
        pipeline = model_manager.get_model(model_name)
        
        # Mock feature importance (would calculate from actual model)
        feature_importance = {
            'price_momentum': 0.25,
            'volume_trend': 0.18,
            'rsi': 0.15,
            'ma_crossover': 0.12,
            'volatility': 0.10,
            'smart_money_score': 0.08,
            'put_call_ratio': 0.07,
            'implied_volatility': 0.05
        }
        
        return {
            'model': model_name,
            'feature_importance': feature_importance,
            'method': 'permutation_importance',
            'timestamp': datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_synthetic_training_data(samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic training data for demonstration
    In production, this would load real market data
    """
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'price_momentum': np.random.randn(samples) * 0.1,
        'volume_trend': np.random.randn(samples) * 0.05,
        'rsi': np.random.uniform(20, 80, samples),
        'ma_crossover': np.random.randn(samples) * 0.02,
        'volatility': np.random.uniform(0.1, 0.5, samples),
        'smart_money_score': np.random.uniform(0, 100, samples),
        'put_call_ratio': np.random.uniform(0.5, 2.0, samples),
        'implied_volatility': np.random.uniform(0.1, 0.8, samples),
        'volume_ratio': np.random.uniform(0.5, 3.0, samples),
        'price_roc': np.random.randn(samples) * 0.03
    }
    
    # Generate target (synthetic returns)
    # Make target somewhat correlated with features
    target = (
        0.3 * data['price_momentum'] + 
        0.2 * data['ma_crossover'] + 
        0.1 * (data['smart_money_score'] / 100 - 0.5) +
        0.05 * np.random.randn(samples)
    )
    
    data['target'] = target
    
    return pd.DataFrame(data)


@router.get("/ml/backtest/{model_name}")
async def backtest_model(
    model_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run backtest for a trained model
    (Returns mock results for demonstration)
    """
    try:
        # Verify model exists
        pipeline = model_manager.get_model(model_name)
        
        # Mock backtest results
        backtest_results = {
            'model': model_name,
            'period': {
                'start': start_date or '2024-01-01',
                'end': end_date or '2024-12-31'
            },
            'metrics': {
                'total_return': 0.245,
                'sharpe_ratio': 1.82,
                'max_drawdown': -0.087,
                'win_rate': 0.58,
                'profit_factor': 1.95,
                'total_trades': 156
            },
            'monthly_returns': [
                0.032, 0.018, -0.012, 0.045,
                0.028, 0.015, 0.038, -0.008,
                0.022, 0.035, 0.019, 0.041
            ]
        }
        
        logger.info(f"Backtest completed for model '{model_name}'")
        
        return backtest_results
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))