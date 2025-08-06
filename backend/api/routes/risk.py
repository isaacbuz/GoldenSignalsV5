"""
Risk Management API Routes
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime, timedelta
import numpy as np
import uuid

from core.risk_manager import (
    risk_manager,
    RiskLimits,
    RiskMetrics,
    RiskAlert,
    RiskControl,
    RiskLevel,
    RiskMetricType,
    RiskAction
)
from core.auth import get_current_user
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/check/{portfolio_id}")
async def check_risk_limits(
    portfolio_id: int,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check current risk limits for portfolio
    
    Args:
        portfolio_id: Portfolio ID
        
    Returns:
        Risk check results
    """
    try:
        is_allowed, violations = await risk_manager.check_risk_limits(portfolio_id)
        
        return {
            "portfolio_id": portfolio_id,
            "is_allowed": is_allowed,
            "violations": violations,
            "trading_frozen": risk_manager.trading_frozen,
            "frozen_until": risk_manager.frozen_until,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to check risk limits: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check risk limits")


@router.post("/check/{portfolio_id}/trade")
async def check_trade_risk(
    portfolio_id: int,
    trade: Dict[str, Any],
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if proposed trade violates risk limits
    
    Args:
        portfolio_id: Portfolio ID
        trade: Proposed trade details
        
    Returns:
        Risk check results for trade
    """
    try:
        is_allowed, violations = await risk_manager.check_risk_limits(
            portfolio_id,
            proposed_trade=trade
        )
        
        return {
            "portfolio_id": portfolio_id,
            "trade": trade,
            "is_allowed": is_allowed,
            "violations": violations,
            "recommendation": "Proceed" if is_allowed else "Block trade",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to check trade risk: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check trade risk")


@router.get("/metrics/{portfolio_id}")
async def get_risk_metrics(
    portfolio_id: int,
    current_user=Depends(get_current_user)
) -> RiskMetrics:
    """
    Get comprehensive risk metrics for portfolio
    
    Args:
        portfolio_id: Portfolio ID
        
    Returns:
        Risk metrics
    """
    try:
        metrics = await risk_manager.calculate_risk_metrics(portfolio_id)
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate risk metrics")


@router.get("/summary/{portfolio_id}")
async def get_risk_summary(
    portfolio_id: int,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get risk summary for portfolio
    
    Args:
        portfolio_id: Portfolio ID
        
    Returns:
        Risk summary
    """
    try:
        summary = risk_manager.get_risk_summary(portfolio_id)
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get risk summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get risk summary")


@router.get("/limits")
async def get_risk_limits(
    current_user=Depends(get_current_user)
) -> RiskLimits:
    """
    Get current risk limits configuration
    
    Returns:
        Risk limits
    """
    return risk_manager.risk_limits


@router.put("/limits")
async def update_risk_limits(
    limits: RiskLimits,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update risk limits configuration
    
    Args:
        limits: New risk limits
        
    Returns:
        Confirmation message
    """
    try:
        risk_manager.set_risk_limits(limits)
        return {"message": "Risk limits updated successfully"}
        
    except Exception as e:
        logger.error(f"Failed to update risk limits: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update risk limits")


@router.get("/controls")
async def get_risk_controls(
    current_user=Depends(get_current_user)
) -> List[RiskControl]:
    """
    Get all risk control rules
    
    Returns:
        List of risk controls
    """
    return risk_manager.risk_controls


@router.post("/controls")
async def add_risk_control(
    control: RiskControl,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Add new risk control rule
    
    Args:
        control: Risk control configuration
        
    Returns:
        Confirmation message
    """
    try:
        risk_manager.add_risk_control(control)
        return {
            "message": f"Risk control '{control.name}' added successfully",
            "rule_id": control.rule_id
        }
        
    except Exception as e:
        logger.error(f"Failed to add risk control: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add risk control")


@router.put("/controls/{rule_id}")
async def update_risk_control(
    rule_id: str,
    enabled: Optional[bool] = None,
    auto_execute: Optional[bool] = None,
    threshold: Optional[float] = None,
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update risk control rule
    
    Args:
        rule_id: Rule identifier
        enabled: Enable/disable rule
        auto_execute: Auto-execute actions
        threshold: New threshold value
        
    Returns:
        Confirmation message
    """
    try:
        # Find control
        control = next((c for c in risk_manager.risk_controls if c.rule_id == rule_id), None)
        
        if not control:
            raise HTTPException(status_code=404, detail="Risk control not found")
        
        # Update fields
        if enabled is not None:
            control.enabled = enabled
        if auto_execute is not None:
            control.auto_execute = auto_execute
        if threshold is not None:
            control.threshold = threshold
        
        return {"message": f"Risk control '{control.name}' updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update risk control: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update risk control")


@router.post("/controls/{rule_id}/execute")
async def execute_risk_control(
    rule_id: str,
    portfolio_id: int,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Manually execute a risk control action
    
    Args:
        rule_id: Rule identifier
        portfolio_id: Portfolio ID
        
    Returns:
        Execution result
    """
    try:
        # Find control
        control = next((c for c in risk_manager.risk_controls if c.rule_id == rule_id), None)
        
        if not control:
            raise HTTPException(status_code=404, detail="Risk control not found")
        
        # Get current metric value
        metrics = await risk_manager.calculate_risk_metrics(portfolio_id)
        metric_value = risk_manager._get_metric_value(control.metric_type, metrics)
        
        # Execute control
        success = await risk_manager.execute_risk_control(
            control,
            portfolio_id,
            metric_value
        )
        
        return {
            "success": success,
            "control": control.name,
            "action": control.action.value,
            "portfolio_id": portfolio_id,
            "message": f"Risk control executed: {control.action.value}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute risk control: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute risk control")


@router.get("/alerts")
async def get_risk_alerts(
    portfolio_id: Optional[int] = None,
    risk_level: Optional[RiskLevel] = None,
    limit: int = Query(100, ge=1, le=1000),
    current_user=Depends(get_current_user)
) -> List[RiskAlert]:
    """
    Get risk alerts
    
    Args:
        portfolio_id: Filter by portfolio
        risk_level: Filter by risk level
        limit: Maximum alerts to return
        
    Returns:
        List of risk alerts
    """
    alerts = risk_manager.alert_history[-limit:]
    
    # Filter if requested
    if portfolio_id:
        alerts = [a for a in alerts if a.portfolio_id == portfolio_id]
    
    if risk_level:
        alerts = [a for a in alerts if a.risk_level == risk_level]
    
    return alerts


@router.post("/freeze")
async def freeze_trading(
    duration_minutes: int = Query(60, ge=1, le=1440),
    reason: str = Query(..., description="Reason for freezing"),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Manually freeze all trading
    
    Args:
        duration_minutes: How long to freeze (max 24 hours)
        reason: Reason for freeze
        
    Returns:
        Freeze details
    """
    try:
        risk_manager.trading_frozen = True
        risk_manager.frozen_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Send alert
        await risk_manager._send_alert(
            RiskLevel.CRITICAL,
            RiskMetricType.LEVERAGE,  # Generic
            0,
            0,
            f"Trading manually frozen: {reason}",
            RiskAction.FREEZE_TRADING
        )
        
        return {
            "frozen": True,
            "frozen_until": risk_manager.frozen_until,
            "duration_minutes": duration_minutes,
            "reason": reason,
            "message": f"Trading frozen until {risk_manager.frozen_until}"
        }
        
    except Exception as e:
        logger.error(f"Failed to freeze trading: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to freeze trading")


@router.post("/unfreeze")
async def unfreeze_trading(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Manually unfreeze trading
    
    Returns:
        Confirmation message
    """
    risk_manager.trading_frozen = False
    risk_manager.frozen_until = None
    
    return {"message": "Trading unfrozen"}


@router.get("/var/{portfolio_id}")
async def calculate_var(
    portfolio_id: int,
    confidence_level: float = Query(0.95, ge=0.9, le=0.99),
    time_horizon: int = Query(1, ge=1, le=30, description="Days"),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Calculate Value at Risk for portfolio
    
    Args:
        portfolio_id: Portfolio ID
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        time_horizon: Time horizon in days
        
    Returns:
        VaR calculation results
    """
    try:
        # Get positions
        from core.position_manager import position_manager
        positions = await position_manager.get_portfolio_positions(portfolio_id)
        
        # Calculate VaR
        var_95, var_99, cvar_95 = await risk_manager._calculate_var(
            portfolio_id,
            positions
        )
        
        # Scale for time horizon
        if time_horizon > 1:
            scale_factor = np.sqrt(time_horizon)
            var_95 *= scale_factor
            var_99 *= scale_factor
            cvar_95 *= scale_factor
        
        # Get portfolio value
        portfolio_summary = await position_manager.get_portfolio_summary(portfolio_id)
        
        return {
            "portfolio_id": portfolio_id,
            "portfolio_value": portfolio_summary.total_value,
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon,
            "var_95": {
                "percentage": var_95,
                "amount": var_95 * portfolio_summary.total_value
            },
            "var_99": {
                "percentage": var_99,
                "amount": var_99 * portfolio_summary.total_value
            },
            "cvar_95": {
                "percentage": cvar_95,
                "amount": cvar_95 * portfolio_summary.total_value
            },
            "interpretation": f"With {confidence_level*100}% confidence, "
                            f"the portfolio will not lose more than "
                            f"${var_95 * portfolio_summary.total_value:.2f} "
                            f"over the next {time_horizon} day(s)"
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate VaR: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate VaR")


@router.get("/stress-test/{portfolio_id}")
async def stress_test(
    portfolio_id: int,
    scenario: str = Query(..., description="Stress scenario"),
    magnitude: float = Query(0.2, ge=0.1, le=0.5, description="Shock magnitude"),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Run stress test on portfolio
    
    Args:
        portfolio_id: Portfolio ID
        scenario: Stress scenario (market_crash, volatility_spike, etc.)
        magnitude: Magnitude of shock (0.2 = 20%)
        
    Returns:
        Stress test results
    """
    try:
        # Get positions
        from core.position_manager import position_manager
        positions = await position_manager.get_portfolio_positions(portfolio_id)
        portfolio_summary = await position_manager.get_portfolio_summary(portfolio_id)
        
        # Apply stress scenario
        stressed_pnl = 0
        position_impacts = []
        
        for position in positions:
            # Apply shock based on scenario
            if scenario == "market_crash":
                price_change = -magnitude
            elif scenario == "volatility_spike":
                price_change = np.random.normal(0, magnitude)
            elif scenario == "correlation_breakdown":
                price_change = np.random.uniform(-magnitude, magnitude)
            else:
                price_change = -magnitude
            
            # Calculate impact
            position_impact = position.position_value * price_change
            stressed_pnl += position_impact
            
            position_impacts.append({
                "symbol": position.symbol,
                "current_value": position.position_value,
                "stressed_value": position.position_value * (1 + price_change),
                "impact": position_impact,
                "impact_pct": price_change
            })
        
        # Calculate stressed metrics
        stressed_portfolio_value = portfolio_summary.total_value + stressed_pnl
        stressed_drawdown = abs(stressed_pnl) / portfolio_summary.total_value if portfolio_summary.total_value > 0 else 0
        
        return {
            "portfolio_id": portfolio_id,
            "scenario": scenario,
            "magnitude": magnitude,
            "current_value": portfolio_summary.total_value,
            "stressed_value": stressed_portfolio_value,
            "total_impact": stressed_pnl,
            "impact_percentage": stressed_pnl / portfolio_summary.total_value if portfolio_summary.total_value > 0 else 0,
            "stressed_drawdown": stressed_drawdown,
            "position_impacts": position_impacts,
            "risk_assessment": "CRITICAL" if stressed_drawdown > 0.3 else "HIGH" if stressed_drawdown > 0.2 else "MEDIUM"
        }
        
    except Exception as e:
        logger.error(f"Failed to run stress test: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to run stress test")


@router.post("/monitoring/start")
async def start_risk_monitoring(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Start risk monitoring
    
    Returns:
        Status message
    """
    try:
        await risk_manager.start_monitoring()
        return {"message": "Risk monitoring started"}
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")


@router.post("/monitoring/stop")
async def stop_risk_monitoring(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Stop risk monitoring
    
    Returns:
        Status message
    """
    try:
        await risk_manager.stop_monitoring()
        return {"message": "Risk monitoring stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")


@router.ws("/ws/risk")
async def risk_websocket(websocket):
    """
    WebSocket endpoint for real-time risk updates
    """
    await websocket.accept()
    
    try:
        # Subscribe to risk events
        from core.events.bus import event_bus
        
        async def send_risk_alert(event):
            await websocket.send_json({
                "type": "risk_alert",
                "data": event.data
            })
        
        async def send_control_executed(event):
            await websocket.send_json({
                "type": "control_executed",
                "data": event.data
            })
        
        await event_bus.subscribe("risk.alert", send_risk_alert)
        await event_bus.subscribe("risk.control_executed", send_control_executed)
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "trading_frozen": risk_manager.trading_frozen,
                "frozen_until": risk_manager.frozen_until.isoformat() if risk_manager.frozen_until else None,
                "monitoring_active": risk_manager._monitoring_active
            }
        })
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()