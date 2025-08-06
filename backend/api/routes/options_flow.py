"""
API Routes for Options Flow Intelligence
Provides endpoints for institutional options flow analysis
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from agents.options_flow_intelligence import (
    options_flow_intelligence,
    FlowType,
    InstitutionType,
    PositionIntent
)
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/options-flow", tags=["Options Flow"])


class OptionsFlowRequest(BaseModel):
    """Request model for analyzing options flow"""
    symbol: str = Field(..., description="Stock symbol")
    underlying_price: float = Field(..., description="Current price of underlying")
    strike: float = Field(..., description="Strike price")
    days_to_expiry: int = Field(..., description="Days until expiration")
    call_put: str = Field(..., pattern="^[CP]$", description="Call (C) or Put (P)")
    side: str = Field(..., pattern="^(BUY|SELL)$", description="Buy or Sell side")
    size: int = Field(..., gt=0, description="Number of contracts")
    price: float = Field(..., gt=0, description="Option price")
    implied_volatility: float = Field(0.3, ge=0, le=5, description="Implied volatility")
    delta: float = Field(0.5, ge=-1, le=1, description="Option delta")
    gamma: float = Field(0.01, ge=0, description="Option gamma")
    flow_type: str = Field("block", description="Type of flow (sweep, block, etc)")
    aggressive_order: bool = Field(False, description="Was this an aggressive order")
    volume_ratio: float = Field(1.0, gt=0, description="Volume relative to average")
    days_to_event: Optional[int] = Field(None, description="Days to earnings/event")


class UnusualActivityRequest(BaseModel):
    """Request model for detecting unusual activity"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    timeframe: str = Field("1d", description="Timeframe for analysis")


@router.post("/analyze")
async def analyze_options_flow(request: OptionsFlowRequest) -> Dict[str, Any]:
    """
    Analyze options flow for institutional patterns
    
    Returns smart money score, institution type, and trading signals
    """
    try:
        flow_data = {
            'symbol': request.symbol,
            'underlying_price': request.underlying_price,
            'strike': request.strike,
            'days_to_expiry': request.days_to_expiry,
            'call_put': request.call_put,
            'side': request.side,
            'size': request.size,
            'price': request.price,
            'notional': request.size * request.price * 100,
            'implied_volatility': request.implied_volatility,
            'delta': request.delta,
            'gamma': request.gamma,
            'flow_type': request.flow_type,
            'aggressive_order': request.aggressive_order,
            'volume_ratio': request.volume_ratio,
            'days_to_event': request.days_to_event,
            'moneyness': ((request.strike / request.underlying_price) - 1) * 100
        }
        
        result = await options_flow_intelligence.analyze_options_flow(flow_data)
        
        logger.info(f"Analyzed options flow for {request.symbol}: {result['flow_analysis']['institution_type']}")
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing options flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unusual-activity")
async def detect_unusual_activity(request: UnusualActivityRequest) -> Dict[str, Any]:
    """
    Detect unusual options activity for a symbol
    
    Returns unusual flows, smart money detection, and recommendations
    """
    try:
        result = await options_flow_intelligence.detect_unusual_activity(
            request.symbol,
            request.timeframe
        )
        
        logger.info(f"Detected unusual activity for {request.symbol}: {result['overall_bias']}")
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error detecting unusual activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/smart-money/{symbol}")
async def get_smart_money_flows(symbol: str) -> Dict[str, Any]:
    """
    Get all smart money flows for a symbol
    
    Returns flows with high smart money scores
    """
    try:
        # Get all flows for the symbol
        symbol_flows = [
            flow for flow in options_flow_intelligence.options_flows
            if flow.symbol == symbol and flow.smart_money_score > 70
        ]
        
        # Sort by smart money score
        symbol_flows.sort(key=lambda x: x.smart_money_score, reverse=True)
        
        # Convert to dict format
        flows_data = [flow.to_dict() for flow in symbol_flows[:10]]
        
        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "smart_money_flows": flows_data,
                "total_flows": len(flows_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting smart money flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flow-types")
async def get_flow_types() -> Dict[str, Any]:
    """Get available flow types and their descriptions"""
    return {
        "status": "success",
        "data": {
            "flow_types": [
                {
                    "value": ft.value,
                    "name": ft.name,
                    "description": {
                        "sweep": "Aggressive multi-exchange sweep",
                        "block": "Large single block trade",
                        "split": "Order split across strikes/expiries",
                        "repeat": "Repeated similar orders",
                        "unusual": "Unusual size/strike/expiry"
                    }.get(ft.value, "")
                }
                for ft in FlowType
            ]
        }
    }


@router.get("/institution-types")
async def get_institution_types() -> Dict[str, Any]:
    """Get institution types and their characteristics"""
    return {
        "status": "success",
        "data": {
            "institution_types": [
                {
                    "value": it.value,
                    "name": it.name,
                    "description": {
                        "hedge_fund": "Aggressive, directional trades",
                        "market_maker": "Delta neutral, short-dated",
                        "proprietary_trading": "Prop trading desks",
                        "insurance_company": "Long-dated protective positions",
                        "pension_fund": "Conservative, long-term",
                        "retail_aggregate": "Aggregated retail flow",
                        "unknown": "Unidentified institution"
                    }.get(it.value, "")
                }
                for it in InstitutionType
            ]
        }
    }


@router.get("/position-intents")
async def get_position_intents() -> Dict[str, Any]:
    """Get position intent types and their meanings"""
    return {
        "status": "success",
        "data": {
            "position_intents": [
                {
                    "value": pi.value,
                    "name": pi.name,
                    "description": {
                        "directional_bullish": "Betting on price increase",
                        "directional_bearish": "Betting on price decrease",
                        "hedge": "Protective position",
                        "volatility_long": "Betting on increased volatility",
                        "volatility_short": "Betting on decreased volatility",
                        "income_generation": "Selling premium for income",
                        "spread_strategy": "Complex spread position"
                    }.get(pi.value, "")
                }
                for pi in PositionIntent
            ]
        }
    }


@router.get("/demo")
async def get_demo_analysis() -> Dict[str, Any]:
    """
    Get a demonstration of options flow analysis
    
    Shows example of analyzing a bullish call sweep
    """
    try:
        # Example bullish flow
        demo_flow = {
            'symbol': 'AAPL',
            'underlying_price': 195.0,
            'strike': 200.0,
            'days_to_expiry': 30,
            'call_put': 'C',
            'side': 'BUY',
            'size': 3000,
            'price': 4.0,
            'notional': 1200000,
            'implied_volatility': 0.32,
            'delta': 0.45,
            'gamma': 0.02,
            'flow_type': 'sweep',
            'aggressive_order': True,
            'volume_ratio': 5.0,
            'moneyness': 2.56
        }
        
        result = await options_flow_intelligence.analyze_options_flow(demo_flow)
        
        return {
            "status": "success",
            "message": "Demo analysis of bullish AAPL call sweep",
            "input": demo_flow,
            "analysis": result
        }
        
    except Exception as e:
        logger.error(f"Error in demo analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for options flow intelligence system"""
    try:
        flow_count = len(options_flow_intelligence.options_flows)
        return {
            "status": "healthy",
            "service": "Options Flow Intelligence",
            "flows_loaded": flow_count,
            "analyzer_ready": options_flow_intelligence.flow_analyzer is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }