"""
Agent API Routes
Endpoints for agent management and signal generation
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from core.orchestrator import unified_orchestrator as AgentOrchestrator
from services.market_data_unified import unified_market_service as MarketDataService
from services.signal_service import SignalService
from core.database import get_db
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# Global orchestrator instance (in production, use dependency injection)
orchestrator = AgentOrchestrator()


@router.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    signal_service = SignalService()
    orchestrator.signal_service = signal_service
    await orchestrator.initialize_default_agents()
    await orchestrator.start()
    logger.info("Agent system initialized")


@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown agents gracefully"""
    await orchestrator.shutdown()
    logger.info("Agent system shutdown")


@router.post("/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    market_service: MarketDataService = Depends(lambda: MarketDataService())
) -> Dict[str, Any]:
    """
    Analyze a symbol using all active agents
    
    Args:
        symbol: Stock symbol to analyze
        
    Returns:
        Analysis results and signal if generated
    """
    try:
        # Get current market data
        quote = await market_service.get_quote(symbol)
        if not quote:
            raise HTTPException(status_code=404, detail=f"No market data for {symbol}")
        
        # Get technical indicators
        indicators = await market_service.get_technical_indicators(symbol)
        
        # Prepare market data for agents
        market_data = {
            "symbol": symbol,
            "price": quote.get("price", 0),
            "volume": quote.get("volume", 0),
            "open": quote.get("open", quote.get("price", 0)),
            "high": quote.get("high", quote.get("price", 0)),
            "low": quote.get("low", quote.get("price", 0)),
            "close": quote.get("price", 0),
            "indicators": indicators
        }
        
        # Run analysis
        signal = await orchestrator.analyze_market(market_data)
        
        response = {
            "symbol": symbol,
            "timestamp": quote.get("timestamp"),
            "market_data": {
                "price": quote.get("price"),
                "change": quote.get("change"),
                "change_percent": quote.get("change_percent"),
                "volume": quote.get("volume")
            },
            "signal": None
        }
        
        if signal:
            response["signal"] = {
                "action": signal.action.value,
                "confidence": signal.confidence,
                "strength": signal.strength.value,
                "reasoning": signal.reasoning,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "features": signal.features
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/performance")
async def get_agent_performance() -> Dict[str, Any]:
    """Get performance metrics for all agents"""
    try:
        return orchestrator.get_agent_performance_summary()
    except Exception as e:
        logger.error(f"Failed to get agent performance: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")


@router.post("/performance/rebalance")
async def rebalance_agent_weights() -> Dict[str, Any]:
    """Rebalance agent weights based on performance"""
    try:
        await orchestrator.rebalance_weights()
        return {
            "status": "success",
            "message": "Agent weights rebalanced",
            "performance": orchestrator.get_agent_performance_summary()
        }
    except Exception as e:
        logger.error(f"Failed to rebalance weights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to rebalance: {str(e)}")


@router.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents"""
    agents = []
    for agent_id, agent in orchestrator.agents.items():
        agents.append({
            "id": agent_id,
            "name": agent.config.name,
            "enabled": agent.config.enabled,
            "weight": agent.config.weight,
            "confidence_threshold": agent.config.confidence_threshold,
            "performance": agent.get_current_performance()
        })
    
    return {
        "total": len(agents),
        "agents": agents
    }


@router.patch("/agents/{agent_id}")
async def update_agent_config(
    agent_id: str,
    config_update: Dict[str, Any]
) -> Dict[str, Any]:
    """Update agent configuration"""
    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = orchestrator.agents[agent_id]
    
    # Update allowed fields
    if "enabled" in config_update:
        agent.config.enabled = bool(config_update["enabled"])
    if "weight" in config_update:
        agent.adjust_weight(float(config_update["weight"]))
    if "confidence_threshold" in config_update:
        agent.config.confidence_threshold = max(0.0, min(1.0, float(config_update["confidence_threshold"])))
    
    return {
        "id": agent_id,
        "name": agent.config.name,
        "config": {
            "enabled": agent.config.enabled,
            "weight": agent.config.weight,
            "confidence_threshold": agent.config.confidence_threshold
        }
    }


@router.post("/feedback/{signal_id}")
async def submit_signal_feedback(
    signal_id: str,
    feedback: Dict[str, Any]
) -> Dict[str, Any]:
    """Submit feedback for a signal to update agent performance"""
    try:
        await orchestrator.update_agent_performance(signal_id, feedback)
        return {
            "status": "success",
            "message": "Feedback processed",
            "signal_id": signal_id
        }
    except Exception as e:
        logger.error(f"Failed to process feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")