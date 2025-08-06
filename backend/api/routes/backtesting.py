"""
Backtesting API Routes
Endpoints for running and managing backtests
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4
import asyncio

from pydantic import BaseModel, Field

from backtesting.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    BacktestMetrics
)
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Backtesting"])

# Store running backtests
running_backtests: Dict[str, BacktestEngine] = {}
backtest_results: Dict[str, Dict[str, Any]] = {}


class BacktestRequest(BaseModel):
    """Request model for starting a backtest"""
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    symbols: List[str] = Field(..., description="Symbols to test")
    initial_capital: float = Field(100000, description="Starting capital")
    timeframe: str = Field("1d", description="Data timeframe")
    mode: str = Field("historical", description="Backtest mode")
    
    # Strategy parameters
    agents: List[str] = Field(default_factory=list, description="Agents to test")
    agent_weights: Dict[str, float] = Field(default_factory=dict, description="Agent weights")
    
    # Risk parameters
    max_position_size: float = Field(0.1, description="Max position size as fraction")
    stop_loss: Optional[float] = Field(0.02, description="Stop loss percentage")
    take_profit: Optional[float] = Field(0.05, description="Take profit percentage")
    
    # Execution parameters
    commission: float = Field(0.001, description="Commission rate")
    slippage: float = Field(0.001, description="Slippage rate")
    allow_shorting: bool = Field(False, description="Allow short positions")


@router.post("/backtest/run")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Start a new backtest
    
    Runs asynchronously and returns a backtest ID
    """
    try:
        # Create backtest config
        config = BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            symbols=request.symbols,
            timeframe=request.timeframe,
            mode=BacktestMode[request.mode.upper()],
            agents_to_test=request.agents,
            agent_weights=request.agent_weights,
            max_position_size=request.max_position_size,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            commission=request.commission,
            slippage=request.slippage,
            allow_shorting=request.allow_shorting
        )
        
        # Create backtest engine
        engine = BacktestEngine(config)
        
        # Generate backtest ID
        backtest_id = str(uuid4())
        running_backtests[backtest_id] = engine
        
        # Run backtest in background
        background_tasks.add_task(
            _run_backtest_async,
            backtest_id,
            engine
        )
        
        return {
            "backtest_id": backtest_id,
            "status": "running",
            "config": {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "symbols": request.symbols,
                "initial_capital": request.initial_capital
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to start backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}/status")
async def get_backtest_status(backtest_id: str) -> Dict[str, Any]:
    """
    Get status of a running backtest
    """
    if backtest_id in running_backtests:
        return {
            "backtest_id": backtest_id,
            "status": "running",
            "progress": "In progress..."
        }
    elif backtest_id in backtest_results:
        result = backtest_results[backtest_id]
        return {
            "backtest_id": backtest_id,
            "status": "completed",
            "metrics": result.get("metrics", {})
        }
    else:
        raise HTTPException(status_code=404, detail="Backtest not found")


@router.get("/backtest/{backtest_id}/results")
async def get_backtest_results(backtest_id: str) -> Dict[str, Any]:
    """
    Get detailed results of a completed backtest
    """
    if backtest_id not in backtest_results:
        if backtest_id in running_backtests:
            return {
                "backtest_id": backtest_id,
                "status": "still_running",
                "message": "Backtest is still running"
            }
        else:
            raise HTTPException(status_code=404, detail="Backtest not found")
    
    return backtest_results[backtest_id]


@router.post("/backtest/{backtest_id}/stop")
async def stop_backtest(backtest_id: str) -> Dict[str, Any]:
    """
    Stop a running backtest
    """
    if backtest_id not in running_backtests:
        raise HTTPException(status_code=404, detail="Backtest not found or already completed")
    
    # In a real implementation, we would signal the engine to stop
    engine = running_backtests.pop(backtest_id)
    
    return {
        "backtest_id": backtest_id,
        "status": "stopped",
        "message": "Backtest has been stopped"
    }


@router.get("/backtest/list")
async def list_backtests() -> Dict[str, Any]:
    """
    List all backtests (running and completed)
    """
    running = [
        {
            "id": bid,
            "status": "running"
        }
        for bid in running_backtests.keys()
    ]
    
    completed = [
        {
            "id": bid,
            "status": "completed",
            "metrics": result.get("metrics", {})
        }
        for bid, result in backtest_results.items()
    ]
    
    return {
        "running": running,
        "completed": completed,
        "total": len(running) + len(completed)
    }


@router.post("/backtest/compare")
async def compare_backtests(backtest_ids: List[str]) -> Dict[str, Any]:
    """
    Compare multiple backtest results
    """
    if len(backtest_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 backtests to compare")
    
    comparisons = {}
    
    for bid in backtest_ids:
        if bid in backtest_results:
            metrics = backtest_results[bid].get("metrics", {})
            comparisons[bid] = {
                "total_return": metrics.get("total_return", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "win_rate": metrics.get("win_rate", 0),
                "total_trades": metrics.get("total_trades", 0)
            }
        else:
            comparisons[bid] = {"error": "Not found or still running"}
    
    # Find best performing
    best_return = max(
        backtest_ids,
        key=lambda x: comparisons.get(x, {}).get("total_return", float('-inf'))
    )
    best_sharpe = max(
        backtest_ids,
        key=lambda x: comparisons.get(x, {}).get("sharpe_ratio", float('-inf'))
    )
    
    return {
        "comparisons": comparisons,
        "best_return": best_return,
        "best_sharpe": best_sharpe
    }


@router.post("/backtest/optimize")
async def optimize_strategy(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run parameter optimization for a strategy
    
    Tests multiple parameter combinations to find optimal settings
    """
    try:
        optimization_id = str(uuid4())
        
        # Create parameter grid
        param_grid = {
            "stop_loss": [0.01, 0.02, 0.03],
            "take_profit": [0.03, 0.05, 0.07],
            "max_position_size": [0.05, 0.1, 0.15]
        }
        
        # Run optimization in background
        background_tasks.add_task(
            _run_optimization_async,
            optimization_id,
            request,
            param_grid
        )
        
        return {
            "optimization_id": optimization_id,
            "status": "running",
            "parameter_grid": param_grid,
            "total_combinations": 27  # 3x3x3
        }
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_backtest_async(backtest_id: str, engine: BacktestEngine):
    """Run backtest asynchronously"""
    try:
        logger.info(f"Starting backtest {backtest_id}")
        
        # Run backtest
        metrics = await engine.run()
        
        # Store results
        backtest_results[backtest_id] = {
            "backtest_id": backtest_id,
            "status": "completed",
            "metrics": metrics.to_dict(),
            "results": engine.get_results(),
            "completed_at": datetime.now().isoformat()
        }
        
        # Remove from running
        running_backtests.pop(backtest_id, None)
        
        logger.info(f"Backtest {backtest_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {str(e)}")
        
        # Store error
        backtest_results[backtest_id] = {
            "backtest_id": backtest_id,
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
        
        # Remove from running
        running_backtests.pop(backtest_id, None)


async def _run_optimization_async(
    optimization_id: str,
    request: BacktestRequest,
    param_grid: Dict[str, List[float]]
):
    """Run parameter optimization asynchronously"""
    try:
        logger.info(f"Starting optimization {optimization_id}")
        
        best_params = None
        best_metric = float('-inf')
        all_results = []
        
        # Test all parameter combinations
        for stop_loss in param_grid["stop_loss"]:
            for take_profit in param_grid["take_profit"]:
                for position_size in param_grid["max_position_size"]:
                    # Update request with parameters
                    request.stop_loss = stop_loss
                    request.take_profit = take_profit
                    request.max_position_size = position_size
                    
                    # Create config
                    config = BacktestConfig(
                        start_date=request.start_date,
                        end_date=request.end_date,
                        initial_capital=request.initial_capital,
                        symbols=request.symbols,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        max_position_size=position_size
                    )
                    
                    # Run backtest
                    engine = BacktestEngine(config)
                    metrics = await engine.run()
                    
                    # Track results
                    result = {
                        "params": {
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "max_position_size": position_size
                        },
                        "sharpe_ratio": metrics.sharpe_ratio
                    }
                    all_results.append(result)
                    
                    # Check if best
                    if metrics.sharpe_ratio > best_metric:
                        best_metric = metrics.sharpe_ratio
                        best_params = result["params"]
        
        # Store optimization results
        backtest_results[optimization_id] = {
            "optimization_id": optimization_id,
            "status": "completed",
            "best_params": best_params,
            "best_sharpe_ratio": best_metric,
            "all_results": all_results,
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Optimization {optimization_id} completed")
        
    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {str(e)}")
        
        backtest_results[optimization_id] = {
            "optimization_id": optimization_id,
            "status": "failed",
            "error": str(e)
        }