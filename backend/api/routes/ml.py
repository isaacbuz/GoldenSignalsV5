"""
Machine Learning API Routes
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from datetime import datetime, timedelta
import pandas as pd
import json

from ml.finance_ml_pipeline import FinanceMLPipeline, ModelType, FeatureConfig
from core.auth import get_current_user
from core.logging import get_logger
from core.events.bus import event_bus

logger = get_logger(__name__)

router = APIRouter(prefix="/api/ml", tags=["machine-learning"])

# Global ML pipeline instance
ml_pipeline = FinanceMLPipeline()


@router.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    symbols: List[str] = Body(...),
    model_type: ModelType = Body(ModelType.LIGHTGBM),
    start_date: Optional[str] = Body(None),
    end_date: Optional[str] = Body(None),
    feature_config: Optional[Dict[str, Any]] = Body(None),
    hyperparameters: Optional[Dict[str, Any]] = Body(None),
    optimize_hyperparams: bool = Body(False),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Train a new ML model
    
    Args:
        symbols: List of symbols to train on
        model_type: Type of model to train
        start_date: Training data start date
        end_date: Training data end date
        feature_config: Feature engineering configuration
        hyperparameters: Model hyperparameters
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Training job ID and initial status
    """
    try:
        # Set dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Configure pipeline
        ml_pipeline.model_type = model_type
        if feature_config:
            ml_pipeline.feature_config = FeatureConfig(**feature_config)
        if hyperparameters:
            ml_pipeline.model_params = hyperparameters
        ml_pipeline.optimize_hyperparams = optimize_hyperparams
        
        # Start training in background
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        async def train_task():
            try:
                # Prepare data
                logger.info(f"Preparing data for {symbols} from {start_date} to {end_date}")
                await ml_pipeline.prepare_data(symbols, start_date, end_date)
                
                # Train model
                logger.info(f"Training {model_type.value} model")
                metrics = await ml_pipeline.train_model()
                
                # Evaluate model
                logger.info("Evaluating model")
                evaluation = await ml_pipeline.evaluate_model()
                
                # Save model
                model_path = await ml_pipeline.save_model(f"models/{job_id}.pkl")
                
                # Publish completion event
                await event_bus.publish(
                    "ml.training.completed",
                    data={
                        "job_id": job_id,
                        "model_type": model_type.value,
                        "metrics": metrics,
                        "evaluation": evaluation,
                        "model_path": model_path,
                        "symbols": symbols
                    }
                )
                
                logger.info(f"Training job {job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Training job {job_id} failed: {str(e)}")
                await event_bus.publish(
                    "ml.training.failed",
                    data={
                        "job_id": job_id,
                        "error": str(e)
                    }
                )
        
        background_tasks.add_task(train_task)
        
        return {
            "job_id": job_id,
            "status": "started",
            "model_type": model_type.value,
            "symbols": symbols,
            "date_range": f"{start_date} to {end_date}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start training")


@router.post("/predict")
async def predict(
    symbol: str = Body(...),
    model_path: Optional[str] = Body(None),
    horizon: int = Body(1),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Make predictions using trained model
    
    Args:
        symbol: Symbol to predict
        model_path: Path to saved model (uses latest if not provided)
        horizon: Prediction horizon in periods
        
    Returns:
        Predictions and confidence intervals
    """
    try:
        # Load model if path provided
        if model_path:
            await ml_pipeline.load_model(model_path)
        
        if not ml_pipeline.model:
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Get latest data for symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        # Prepare features
        await ml_pipeline.prepare_data(
            [symbol],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if ml_pipeline.data.empty:
            raise HTTPException(status_code=404, detail="No data available for symbol")
        
        # Get latest features
        latest_features = ml_pipeline.data.iloc[-1][ml_pipeline.feature_columns]
        
        # Make prediction
        prediction = await ml_pipeline.predict(latest_features.values.reshape(1, -1))
        
        # Calculate confidence based on recent model performance
        confidence = ml_pipeline.calculate_prediction_confidence(prediction[0])
        
        return {
            "symbol": symbol,
            "prediction": float(prediction[0]),
            "confidence": confidence,
            "horizon": horizon,
            "timestamp": datetime.now().isoformat(),
            "features_used": len(ml_pipeline.feature_columns),
            "model_type": ml_pipeline.model_type.value if ml_pipeline.model_type else "unknown"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/batch-predict")
async def batch_predict(
    symbols: List[str] = Body(...),
    model_path: Optional[str] = Body(None),
    current_user=Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Make predictions for multiple symbols
    
    Args:
        symbols: List of symbols to predict
        model_path: Path to saved model
        
    Returns:
        List of predictions
    """
    predictions = []
    
    for symbol in symbols:
        try:
            result = await predict(
                symbol=symbol,
                model_path=model_path,
                current_user=current_user
            )
            predictions.append(result)
        except Exception as e:
            logger.warning(f"Failed to predict for {symbol}: {str(e)}")
            predictions.append({
                "symbol": symbol,
                "error": str(e)
            })
    
    return predictions


@router.get("/feature-importance")
async def get_feature_importance(
    model_path: Optional[str] = Query(None),
    top_n: int = Query(20, ge=1, le=100),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get feature importance from trained model
    
    Args:
        model_path: Path to saved model
        top_n: Number of top features to return
        
    Returns:
        Feature importance rankings
    """
    try:
        # Load model if path provided
        if model_path:
            await ml_pipeline.load_model(model_path)
        
        if not ml_pipeline.model:
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Get feature importance
        importance = ml_pipeline.get_feature_importance()
        
        if not importance:
            return {"message": "Feature importance not available for this model type"}
        
        # Sort and limit
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return {
            "features": [
                {"name": name, "importance": float(value)}
                for name, value in sorted_importance
            ],
            "total_features": len(importance),
            "model_type": ml_pipeline.model_type.value if ml_pipeline.model_type else "unknown"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get feature importance")


@router.get("/models")
async def list_models(
    current_user=Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List available trained models
    
    Returns:
        List of model metadata
    """
    try:
                import pickle
        
        models = []
        model_dir = "models"
        
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(model_dir, filename)
                    
                    # Try to load metadata
                    try:
                        with open(filepath, 'rb') as f:
                            model_data = pickle.load(f)
                            
                        models.append({
                            "filename": filename,
                            "path": filepath,
                            "size": os.path.getsize(filepath),
                            "created": datetime.fromtimestamp(
                                os.path.getctime(filepath)
                            ).isoformat(),
                            "model_type": model_data.get('model_type', 'unknown')
                        })
                    except:
                        # If can't load, just add basic info
                        models.append({
                            "filename": filename,
                            "path": filepath,
                            "size": os.path.getsize(filepath),
                            "created": datetime.fromtimestamp(
                                os.path.getctime(filepath)
                            ).isoformat()
                        })
        
        return sorted(models, key=lambda x: x['created'], reverse=True)
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/backtest")
async def backtest_model(
    background_tasks: BackgroundTasks,
    model_path: str = Body(...),
    symbols: List[str] = Body(...),
    start_date: str = Body(...),
    end_date: str = Body(...),
    initial_capital: float = Body(100000),
    position_size: float = Body(0.1),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Backtest a trained model
    
    Args:
        model_path: Path to saved model
        symbols: Symbols to backtest
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        position_size: Position size as fraction of capital
        
    Returns:
        Backtest job ID and status
    """
    try:
        job_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        async def backtest_task():
            try:
                # Load model
                await ml_pipeline.load_model(model_path)
                
                # Prepare data
                await ml_pipeline.prepare_data(symbols, start_date, end_date)
                
                # Run backtest
                results = await ml_pipeline.backtest(
                    initial_capital=initial_capital,
                    position_size=position_size
                )
                
                # Publish results
                await event_bus.publish(
                    "ml.backtest.completed",
                    data={
                        "job_id": job_id,
                        "results": results,
                        "symbols": symbols,
                        "date_range": f"{start_date} to {end_date}"
                    }
                )
                
                logger.info(f"Backtest job {job_id} completed")
                
            except Exception as e:
                logger.error(f"Backtest job {job_id} failed: {str(e)}")
                await event_bus.publish(
                    "ml.backtest.failed",
                    data={
                        "job_id": job_id,
                        "error": str(e)
                    }
                )
        
        background_tasks.add_task(backtest_task)
        
        return {
            "job_id": job_id,
            "status": "started",
            "symbols": symbols,
            "date_range": f"{start_date} to {end_date}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start backtest: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start backtest")


@router.get("/performance-metrics")
async def get_performance_metrics(
    model_path: Optional[str] = Query(None),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get performance metrics for a model
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Model performance metrics
    """
    try:
        # Load model if path provided
        if model_path:
            await ml_pipeline.load_model(model_path)
        
        if not ml_pipeline.model:
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Get stored metrics
        metrics = ml_pipeline.model_metrics
        
        if not metrics:
            return {"message": "No metrics available for this model"}
        
        return {
            "metrics": metrics,
            "model_type": ml_pipeline.model_type.value if ml_pipeline.model_type else "unknown",
            "feature_count": len(ml_pipeline.feature_columns) if ml_pipeline.feature_columns else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@router.post("/optimize-hyperparameters")
async def optimize_hyperparameters(
    background_tasks: BackgroundTasks,
    symbols: List[str] = Body(...),
    model_type: ModelType = Body(ModelType.LIGHTGBM),
    n_trials: int = Body(50),
    start_date: Optional[str] = Body(None),
    end_date: Optional[str] = Body(None),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Optimize model hyperparameters
    
    Args:
        symbols: Symbols to train on
        model_type: Type of model
        n_trials: Number of optimization trials
        start_date: Training data start date
        end_date: Training data end date
        
    Returns:
        Optimization job ID and status
    """
    try:
        # Set dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        job_id = f"optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        async def optimize_task():
            try:
                # Configure pipeline
                ml_pipeline.model_type = model_type
                ml_pipeline.optimize_hyperparams = True
                ml_pipeline.optuna_trials = n_trials
                
                # Prepare data
                await ml_pipeline.prepare_data(symbols, start_date, end_date)
                
                # Run optimization
                best_params = await ml_pipeline.optimize_with_optuna()
                
                # Train final model with best params
                ml_pipeline.model_params = best_params
                metrics = await ml_pipeline.train_model()
                
                # Save optimized model
                model_path = await ml_pipeline.save_model(f"models/{job_id}_optimized.pkl")
                
                # Publish results
                await event_bus.publish(
                    "ml.optimization.completed",
                    data={
                        "job_id": job_id,
                        "best_params": best_params,
                        "metrics": metrics,
                        "model_path": model_path,
                        "n_trials": n_trials
                    }
                )
                
                logger.info(f"Optimization job {job_id} completed")
                
            except Exception as e:
                logger.error(f"Optimization job {job_id} failed: {str(e)}")
                await event_bus.publish(
                    "ml.optimization.failed",
                    data={
                        "job_id": job_id,
                        "error": str(e)
                    }
                )
        
        background_tasks.add_task(optimize_task)
        
        return {
            "job_id": job_id,
            "status": "started",
            "model_type": model_type.value,
            "n_trials": n_trials,
            "symbols": symbols
        }
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start optimization")


@router.ws("/ws")
async def ml_websocket(websocket):
    """
    WebSocket endpoint for real-time ML updates
    """
    await websocket.accept()
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to ML pipeline stream",
            "timestamp": datetime.now().isoformat()
        })
        
        # Subscribe to ML events
        async def on_training_update(event):
            await websocket.send_json({
                "type": "training_update",
                "data": event.data,
                "timestamp": datetime.now().isoformat()
            })
        
        async def on_prediction(event):
            await websocket.send_json({
                "type": "prediction",
                "data": event.data,
                "timestamp": datetime.now().isoformat()
            })
        
        await event_bus.subscribe("ml.training.progress", on_training_update)
        await event_bus.subscribe("ml.prediction.made", on_prediction)
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()