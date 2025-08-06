"""
Performance Monitoring API Routes
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from monitoring.agent_performance import (
    performance_monitor,
    AgentMetrics,
    PerformanceMetric
)
from core.auth import get_current_user
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/performance", tags=["performance"])


class PerformanceThresholds(BaseModel):
    """Performance threshold configuration"""
    min_accuracy: Optional[float] = Field(None, ge=0, le=1)
    min_sharpe: Optional[float] = None
    max_drawdown: Optional[float] = Field(None, ge=0, le=1)
    min_signals: Optional[int] = Field(None, ge=1)


class AgentPerformanceResponse(BaseModel):
    """Agent performance response"""
    agent_id: str
    agent_type: str
    
    # Key metrics
    accuracy: float
    win_rate: float
    sharpe_ratio: float
    total_pnl: float
    max_drawdown: float
    
    # Confidence
    avg_confidence: float
    confidence_calibration: float
    overconfidence_ratio: float
    
    # Trend
    performance_trend: str
    recent_performance: float
    
    # Meta
    total_signals: int
    sample_size: int
    last_updated: datetime


class PerformanceAlert(BaseModel):
    """Performance alert"""
    agent_id: str
    alert_type: str
    message: str
    timestamp: datetime
    severity: str = "warning"


class PerformanceComparison(BaseModel):
    """Agent performance comparison"""
    agents: List[str]
    metric: str
    values: Dict[str, float]
    best_performer: str
    worst_performer: str
    average: float


@router.get("/agents/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    current_user=Depends(get_current_user)
) -> AgentPerformanceResponse:
    """
    Get current performance metrics for specific agent
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Current performance metrics
    """
    try:
        metrics = performance_monitor.get_agent_metrics(agent_id)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics found for agent {agent_id}"
            )
        
        return AgentPerformanceResponse(
            agent_id=metrics.agent_id,
            agent_type=metrics.agent_type,
            accuracy=metrics.accuracy,
            win_rate=metrics.win_rate,
            sharpe_ratio=metrics.sharpe_ratio,
            total_pnl=metrics.total_pnl,
            max_drawdown=metrics.max_drawdown,
            avg_confidence=metrics.avg_confidence,
            confidence_calibration=metrics.confidence_calibration,
            overconfidence_ratio=metrics.overconfidence_ratio,
            performance_trend=metrics.performance_trend,
            recent_performance=metrics.recent_performance,
            total_signals=metrics.total_signals,
            sample_size=metrics.sample_size,
            last_updated=metrics.last_updated
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve agent metrics"
        )


@router.get("/agents")
async def get_all_agent_metrics(
    current_user=Depends(get_current_user)
) -> List[AgentPerformanceResponse]:
    """
    Get performance metrics for all agents
    
    Returns:
        List of agent performance metrics
    """
    try:
        all_metrics = performance_monitor.get_all_metrics()
        
        return [
            AgentPerformanceResponse(
                agent_id=metrics.agent_id,
                agent_type=metrics.agent_type,
                accuracy=metrics.accuracy,
                win_rate=metrics.win_rate,
                sharpe_ratio=metrics.sharpe_ratio,
                total_pnl=metrics.total_pnl,
                max_drawdown=metrics.max_drawdown,
                avg_confidence=metrics.avg_confidence,
                confidence_calibration=metrics.confidence_calibration,
                overconfidence_ratio=metrics.overconfidence_ratio,
                performance_trend=metrics.performance_trend,
                recent_performance=metrics.recent_performance,
                total_signals=metrics.total_signals,
                sample_size=metrics.sample_size,
                last_updated=metrics.last_updated
            )
            for agent_id, metrics in all_metrics.items()
        ]
        
    except Exception as e:
        logger.error(f"Failed to get all agent metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve agent metrics"
        )


@router.get("/agents/{agent_id}/history")
async def get_agent_history(
    agent_id: str,
    days: int = Query(7, ge=1, le=365, description="Number of days of history"),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get historical performance metrics for an agent
    
    Args:
        agent_id: Agent identifier
        days: Number of days of history to retrieve
        
    Returns:
        Historical performance data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        history_df = await performance_monitor.get_historical_metrics(
            agent_id,
            start_date,
            end_date
        )
        
        if history_df.empty:
            return {
                "agent_id": agent_id,
                "period": f"{days} days",
                "data": []
            }
        
        # Convert to JSON-serializable format
        history_data = history_df.to_dict(orient="records")
        
        # Calculate summary statistics
        summary = {
            "avg_accuracy": history_df["accuracy"].mean(),
            "avg_win_rate": history_df["win_rate"].mean(),
            "avg_sharpe": history_df["sharpe_ratio"].mean(),
            "total_pnl": history_df["total_pnl"].iloc[-1] if not history_df.empty else 0,
            "max_drawdown": history_df["max_drawdown"].max(),
            "performance_improvement": (
                (history_df["accuracy"].iloc[-1] - history_df["accuracy"].iloc[0])
                if len(history_df) > 1 else 0
            )
        }
        
        return {
            "agent_id": agent_id,
            "period": f"{days} days",
            "summary": summary,
            "data": history_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve agent history"
        )


@router.post("/thresholds")
async def update_thresholds(
    thresholds: PerformanceThresholds,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update performance monitoring thresholds
    
    Args:
        thresholds: New threshold values
        
    Returns:
        Confirmation message
    """
    try:
        # Filter out None values
        threshold_updates = {
            k: v for k, v in thresholds.dict().items()
            if v is not None
        }
        
        if threshold_updates:
            performance_monitor.set_thresholds(threshold_updates)
            
            return {
                "message": "Thresholds updated successfully",
                "updated": list(threshold_updates.keys())
            }
        else:
            return {"message": "No thresholds to update"}
        
    except Exception as e:
        logger.error(f"Failed to update thresholds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update thresholds"
        )


@router.get("/comparison")
async def compare_agents(
    metric: PerformanceMetric = Query(..., description="Metric to compare"),
    agent_ids: Optional[List[str]] = Query(None, description="Specific agents to compare"),
    current_user=Depends(get_current_user)
) -> PerformanceComparison:
    """
    Compare performance across multiple agents
    
    Args:
        metric: Performance metric to compare
        agent_ids: Optional list of specific agents to compare
        
    Returns:
        Comparison results
    """
    try:
        all_metrics = performance_monitor.get_all_metrics()
        
        # Filter agents if specified
        if agent_ids:
            all_metrics = {
                k: v for k, v in all_metrics.items()
                if k in agent_ids
            }
        
        if not all_metrics:
            raise HTTPException(
                status_code=404,
                detail="No agents found for comparison"
            )
        
        # Extract metric values
        metric_values = {}
        for agent_id, metrics in all_metrics.items():
            if metric == PerformanceMetric.ACCURACY:
                metric_values[agent_id] = metrics.accuracy
            elif metric == PerformanceMetric.WIN_RATE:
                metric_values[agent_id] = metrics.win_rate
            elif metric == PerformanceMetric.SHARPE_RATIO:
                metric_values[agent_id] = metrics.sharpe_ratio
            elif metric == PerformanceMetric.PROFIT_FACTOR:
                metric_values[agent_id] = metrics.profit_factor
            elif metric == PerformanceMetric.MAX_DRAWDOWN:
                metric_values[agent_id] = metrics.max_drawdown
            elif metric == PerformanceMetric.AVG_RETURN:
                metric_values[agent_id] = metrics.total_pnl / max(metrics.total_signals, 1)
            elif metric == PerformanceMetric.CONFIDENCE_CALIBRATION:
                metric_values[agent_id] = metrics.confidence_calibration
            else:
                metric_values[agent_id] = 0
        
        # Find best and worst
        if metric == PerformanceMetric.MAX_DRAWDOWN:
            # Lower is better for drawdown
            best_agent = min(metric_values, key=metric_values.get)
            worst_agent = max(metric_values, key=metric_values.get)
        else:
            # Higher is better for other metrics
            best_agent = max(metric_values, key=metric_values.get)
            worst_agent = min(metric_values, key=metric_values.get)
        
        # Calculate average
        avg_value = sum(metric_values.values()) / len(metric_values) if metric_values else 0
        
        return PerformanceComparison(
            agents=list(metric_values.keys()),
            metric=metric.value,
            values=metric_values,
            best_performer=best_agent,
            worst_performer=worst_agent,
            average=avg_value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare agents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to compare agent performance"
        )


@router.get("/leaderboard")
async def get_leaderboard(
    metric: PerformanceMetric = Query(PerformanceMetric.SHARPE_RATIO, description="Metric to rank by"),
    limit: int = Query(10, ge=1, le=50, description="Number of agents to show"),
    current_user=Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get agent performance leaderboard
    
    Args:
        metric: Metric to rank agents by
        limit: Number of top agents to return
        
    Returns:
        Ranked list of agents
    """
    try:
        all_metrics = performance_monitor.get_all_metrics()
        
        # Create leaderboard entries
        leaderboard = []
        for agent_id, metrics in all_metrics.items():
            # Get metric value
            if metric == PerformanceMetric.ACCURACY:
                value = metrics.accuracy
            elif metric == PerformanceMetric.WIN_RATE:
                value = metrics.win_rate
            elif metric == PerformanceMetric.SHARPE_RATIO:
                value = metrics.sharpe_ratio
            elif metric == PerformanceMetric.PROFIT_FACTOR:
                value = metrics.profit_factor
            elif metric == PerformanceMetric.MAX_DRAWDOWN:
                value = -metrics.max_drawdown  # Negative so lower is ranked higher
            elif metric == PerformanceMetric.AVG_RETURN:
                value = metrics.total_pnl / max(metrics.total_signals, 1)
            else:
                value = 0
            
            leaderboard.append({
                "rank": 0,  # Will be set after sorting
                "agent_id": agent_id,
                "agent_type": metrics.agent_type,
                "metric_value": abs(value) if metric == PerformanceMetric.MAX_DRAWDOWN else value,
                "win_rate": metrics.win_rate,
                "total_pnl": metrics.total_pnl,
                "total_signals": metrics.total_signals,
                "trend": metrics.performance_trend
            })
        
        # Sort by metric value (descending)
        leaderboard.sort(key=lambda x: x["metric_value"], reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(leaderboard[:limit]):
            entry["rank"] = i + 1
        
        return leaderboard[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve leaderboard"
        )


@router.post("/monitoring/start")
async def start_monitoring(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Start performance monitoring
    
    Returns:
        Status message
    """
    try:
        await performance_monitor.start_monitoring()
        return {"message": "Performance monitoring started"}
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start monitoring"
        )


@router.post("/monitoring/stop")
async def stop_monitoring(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Stop performance monitoring
    
    Returns:
        Status message
    """
    try:
        await performance_monitor.stop_monitoring()
        return {"message": "Performance monitoring stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to stop monitoring"
        )


@router.ws("/ws/performance")
async def performance_websocket(websocket):
    """
    WebSocket endpoint for real-time performance updates
    """
    await websocket.accept()
    
    try:
        # Subscribe to performance events
        async def send_metrics(event):
            await websocket.send_json({
                "type": "metrics",
                "data": event.data
            })
        
        async def send_alert(event):
            await websocket.send_json({
                "type": "alert",
                "data": event.data
            })
        
        from core.events.bus import event_bus
        await event_bus.subscribe("performance.metrics", send_metrics)
        await event_bus.subscribe("performance.alert", send_alert)
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()