"""
Position Management API Routes
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime

from core.position_manager import (
    position_manager,
    PositionRequest,
    PositionUpdate,
    PositionInfo,
    PortfolioSummary,
    OrderType,
    OrderSide,
    PositionStatus
)
from core.auth import get_current_user
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/positions", tags=["positions"])


@router.post("/open")
async def open_position(
    portfolio_id: int,
    request: PositionRequest,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Open a new position
    
    Args:
        portfolio_id: Portfolio to open position in
        request: Position details
        
    Returns:
        Created position information
    """
    try:
        position = await position_manager.open_position(portfolio_id, request)
        
        return {
            "success": True,
            "position": {
                "position_id": position.position_id,
                "symbol": position.symbol,
                "type": position.position_type.value,
                "quantity": position.quantity,
                "entry_price": position.avg_entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "position_value": position.position_value,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "opened_at": position.opened_at
            },
            "message": f"Position opened for {request.symbol}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to open position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to open position")


@router.post("/{position_id}/close")
async def close_position(
    position_id: str,
    quantity: Optional[float] = None,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Optional[float] = None,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Close a position (fully or partially)
    
    Args:
        position_id: Position to close
        quantity: Quantity to close (None for full close)
        order_type: Order type for closing
        limit_price: Limit price if applicable
        
    Returns:
        Close execution details
    """
    try:
        result = await position_manager.close_position(
            position_id,
            quantity,
            order_type,
            limit_price
        )
        
        return {
            "success": True,
            "result": result,
            "message": "Position closed successfully" if result["fully_closed"] else "Position partially closed"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to close position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to close position")


@router.patch("/{position_id}")
async def update_position(
    position_id: str,
    update: PositionUpdate,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update position parameters
    
    Args:
        position_id: Position to update
        update: Update parameters
        
    Returns:
        Updated position information
    """
    try:
        position = await position_manager.update_position(position_id, update)
        
        return {
            "success": True,
            "position": {
                "position_id": position.position_id,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "trailing_stop": position.trailing_stop_distance,
                "quantity": position.quantity,
                "notes": position.notes,
                "tags": position.tags,
                "last_updated": position.last_updated
            },
            "message": "Position updated successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update position")


@router.get("/{position_id}")
async def get_position(
    position_id: str,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get position details
    
    Args:
        position_id: Position ID
        
    Returns:
        Position information
    """
    try:
        position = await position_manager.get_position(position_id)
        
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        return {
            "position_id": position.position_id,
            "portfolio_id": position.portfolio_id,
            "symbol": position.symbol,
            "type": position.position_type.value,
            "quantity": position.quantity,
            "entry_price": position.avg_entry_price,
            "current_price": position.current_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
            "total_pnl": position.total_pnl,
            "pnl_percentage": position.pnl_percentage,
            "position_value": position.position_value,
            "exposure": position.exposure,
            "margin_required": position.margin_required,
            "leverage": position.leverage,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "trailing_stop": position.trailing_stop_distance,
            "max_profit": position.max_profit,
            "max_loss": position.max_loss,
            "opened_at": position.opened_at,
            "holding_period": str(position.holding_period),
            "tags": position.tags,
            "notes": position.notes,
            "source_signal_id": position.source_signal_id,
            "source_agent_id": position.source_agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve position")


@router.get("/portfolio/{portfolio_id}")
async def get_portfolio_positions(
    portfolio_id: int,
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user=Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get all positions for a portfolio
    
    Args:
        portfolio_id: Portfolio ID
        status: Optional status filter
        
    Returns:
        List of positions
    """
    try:
        status_enum = PositionStatus(status) if status else None
        positions = await position_manager.get_portfolio_positions(
            portfolio_id,
            status_enum
        )
        
        return [
            {
                "position_id": p.position_id,
                "symbol": p.symbol,
                "type": p.position_type.value,
                "quantity": p.quantity,
                "entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "total_pnl": p.total_pnl,
                "pnl_percentage": p.pnl_percentage,
                "position_value": p.position_value,
                "opened_at": p.opened_at,
                "holding_period": str(p.holding_period)
            }
            for p in positions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get portfolio positions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve positions")


@router.get("/portfolio/{portfolio_id}/summary")
async def get_portfolio_summary(
    portfolio_id: int,
    current_user=Depends(get_current_user)
) -> PortfolioSummary:
    """
    Get portfolio summary with metrics
    
    Args:
        portfolio_id: Portfolio ID
        
    Returns:
        Portfolio summary
    """
    try:
        summary = await position_manager.get_portfolio_summary(portfolio_id)
        return summary
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio summary")


@router.post("/batch/close")
async def close_multiple_positions(
    position_ids: List[str],
    order_type: OrderType = OrderType.MARKET,
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Close multiple positions at once
    
    Args:
        position_ids: List of position IDs to close
        order_type: Order type for closing
        
    Returns:
        Results for each position
    """
    results = {}
    errors = {}
    
    for position_id in position_ids:
        try:
            result = await position_manager.close_position(
                position_id,
                order_type=order_type
            )
            results[position_id] = result
        except Exception as e:
            errors[position_id] = str(e)
    
    return {
        "success": len(errors) == 0,
        "closed": results,
        "errors": errors,
        "message": f"Closed {len(results)} positions" + 
                  (f", {len(errors)} failed" if errors else "")
    }


@router.post("/hedge/{position_id}")
async def hedge_position(
    position_id: str,
    hedge_ratio: float = Query(1.0, ge=0.1, le=2.0, description="Hedge ratio"),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a hedge position
    
    Args:
        position_id: Position to hedge
        hedge_ratio: Ratio of hedge (1.0 = full hedge)
        
    Returns:
        Hedge position details
    """
    try:
        # Get original position
        original = await position_manager.get_position(position_id)
        if not original:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Create opposite position
        hedge_request = PositionRequest(
            symbol=original.symbol,
            side=OrderSide.SELL if original.position_type.value == "long" else OrderSide.BUY,
            quantity=original.quantity * hedge_ratio,
            order_type=OrderType.MARKET,
            notes=f"Hedge for position {position_id}",
            tags=["hedge", f"original_{position_id}"]
        )
        
        hedge = await position_manager.open_position(
            original.portfolio_id,
            hedge_request
        )
        
        return {
            "success": True,
            "original_position": position_id,
            "hedge_position": hedge.position_id,
            "hedge_ratio": hedge_ratio,
            "message": f"Hedge position created for {original.symbol}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to hedge position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create hedge")


@router.get("/analytics/pnl")
async def get_pnl_analytics(
    portfolio_id: Optional[int] = None,
    days: int = Query(30, ge=1, le=365, description="Days of history"),
    current_user=Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get P&L analytics
    
    Args:
        portfolio_id: Optional portfolio filter
        days: Number of days to analyze
        
    Returns:
        P&L analytics
    """
    try:
        # This would query historical data
        # For now, return current snapshot
        
        if portfolio_id:
            summary = await position_manager.get_portfolio_summary(portfolio_id)
            positions = await position_manager.get_portfolio_positions(portfolio_id)
        else:
            # Get all portfolios (simplified)
            positions = list(position_manager.positions_cache.values())
            
            total_unrealized = sum(p.unrealized_pnl for p in positions)
            total_realized = sum(p.realized_pnl for p in positions)
            
            summary = {
                "total_unrealized_pnl": total_unrealized,
                "total_realized_pnl": total_realized,
                "total_pnl": total_unrealized + total_realized
            }
        
        # Calculate analytics
        winning_positions = [p for p in positions if p.total_pnl > 0]
        losing_positions = [p for p in positions if p.total_pnl < 0]
        
        analytics = {
            "period": f"{days} days",
            "portfolio_id": portfolio_id,
            "total_pnl": summary.get("total_pnl", 
                                     summary.get("total_unrealized_pnl", 0) + 
                                     summary.get("total_realized_pnl", 0)),
            "unrealized_pnl": summary.get("total_unrealized_pnl", 0),
            "realized_pnl": summary.get("total_realized_pnl", 0),
            "winning_positions": len(winning_positions),
            "losing_positions": len(losing_positions),
            "win_rate": len(winning_positions) / max(len(positions), 1),
            "avg_win": sum(p.total_pnl for p in winning_positions) / max(len(winning_positions), 1),
            "avg_loss": sum(p.total_pnl for p in losing_positions) / max(len(losing_positions), 1),
            "best_position": max((p for p in positions), key=lambda x: x.total_pnl, default=None),
            "worst_position": min((p for p in positions), key=lambda x: x.total_pnl, default=None),
            "positions_by_symbol": {},  # Would aggregate by symbol
            "daily_pnl": []  # Would calculate daily P&L series
        }
        
        # Format best/worst positions
        if analytics["best_position"]:
            p = analytics["best_position"]
            analytics["best_position"] = {
                "symbol": p.symbol,
                "pnl": p.total_pnl,
                "pnl_percentage": p.pnl_percentage
            }
        
        if analytics["worst_position"]:
            p = analytics["worst_position"]
            analytics["worst_position"] = {
                "symbol": p.symbol,
                "pnl": p.total_pnl,
                "pnl_percentage": p.pnl_percentage
            }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get P&L analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate analytics")


@router.post("/monitoring/start")
async def start_position_monitoring(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Start position monitoring (stop loss, take profit checks)
    
    Returns:
        Status message
    """
    try:
        await position_manager.start_monitoring()
        return {"message": "Position monitoring started"}
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start monitoring")


@router.post("/monitoring/stop")
async def stop_position_monitoring(
    current_user=Depends(get_current_user)
) -> Dict[str, str]:
    """
    Stop position monitoring
    
    Returns:
        Status message
    """
    try:
        await position_manager.stop_monitoring()
        return {"message": "Position monitoring stopped"}
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop monitoring")


@router.ws("/ws/positions")
async def positions_websocket(websocket):
    """
    WebSocket endpoint for real-time position updates
    """
    await websocket.accept()
    
    try:
        # Subscribe to position events
        from core.events.bus import event_bus
        
        async def send_position_update(event):
            await websocket.send_json({
                "type": "position_update",
                "data": event.data
            })
        
        await event_bus.subscribe(EventTypes.POSITION_OPENED, send_position_update)
        await event_bus.subscribe(EventTypes.POSITION_CLOSED, send_position_update)
        await event_bus.subscribe("position.auto_closed", send_position_update)
        
        # Send current positions
        positions = list(position_manager.positions_cache.values())
        await websocket.send_json({
            "type": "initial_positions",
            "data": [
                {
                    "position_id": p.position_id,
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "pnl": p.total_pnl
                }
                for p in positions
            ]
        })
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()