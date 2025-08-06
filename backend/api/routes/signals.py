"""
Signal API Routes
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from services.signal_service import signal_service

router = APIRouter(prefix="/signals", tags=["signals"])


class CreateSignalRequest(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    reasoning: str
    agents_consensus: dict
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_level: str = "medium"
    timeframe: str = "1d"
    expires_in_hours: int = 24


class UpdateSignalStatusRequest(BaseModel):
    status: str  # active, executed, expired, cancelled
    execution_price: Optional[float] = None


@router.post("/")
async def create_signal(request: CreateSignalRequest):
    """Create a new trading signal"""
    try:
        signal = await signal_service.create_signal(
            symbol=request.symbol.upper(),
            action=request.action,
            confidence=request.confidence,
            price=request.price,
            reasoning=request.reasoning,
            agents_consensus=request.agents_consensus,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            risk_level=request.risk_level,
            timeframe=request.timeframe,
            expires_in_hours=request.expires_in_hours
        )
        
        return signal.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/")
async def get_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status"),
    action: Optional[str] = Query(None, description="Filter by action"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
    days: Optional[int] = Query(None, ge=1, le=365, description="Filter signals from last N days"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """Get signals with filtering and pagination"""
    try:
        # Calculate since date if days provided
        since = None
        if days:
            since = datetime.now() - timedelta(days=days)
        
        signals = await signal_service.get_signals(
            symbol=symbol,
            status=status,
            action=action,
            min_confidence=min_confidence,
            since=since,
            limit=limit,
            offset=offset
        )
        
        return {
            "signals": [signal.to_dict() for signal in signals],
            "count": len(signals),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analytics")
async def get_signal_analytics(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get signal analytics and performance metrics"""
    try:
        analytics = await signal_service.get_signal_analytics(symbol, days)
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{signal_id}")
async def get_signal(signal_id: str):
    """Get a specific signal by ID"""
    signal = await signal_service.get_signal_by_id(signal_id)
    
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal not found: {signal_id}")
    
    return signal.to_dict()


@router.patch("/{signal_id}/status")
async def update_signal_status(signal_id: str, request: UpdateSignalStatusRequest):
    """Update signal status"""
    try:
        signal = await signal_service.update_signal_status(
            signal_id,
            request.status,
            request.execution_price
        )
        
        if not signal:
            raise HTTPException(status_code=404, detail=f"Signal not found: {signal_id}")
        
        return signal.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{signal_id}/performance")
async def update_signal_performance(signal_id: str, current_price: float):
    """Update signal performance based on current price"""
    try:
        signal = await signal_service.update_signal_performance(signal_id, current_price)
        
        if not signal:
            raise HTTPException(status_code=404, detail=f"Signal not found: {signal_id}")
        
        return signal.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cleanup")
async def cleanup_expired_signals():
    """Clean up expired signals"""
    try:
        count = await signal_service.cleanup_expired_signals()
        return {"cleaned_up": count, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Import datetime for the days filter
from datetime import timedelta