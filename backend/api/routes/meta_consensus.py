"""
API Routes for Meta Signal Orchestrator
Byzantine fault-tolerant consensus endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from agents.meta_signal_orchestrator import (
    meta_orchestrator,
    ConsensusMethod,
    SignalAction,
    ConflictLevel
)
from agents.options_flow_intelligence import options_flow_intelligence
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/meta-consensus", tags=["Meta Consensus"])


class AgentSignal(BaseModel):
    """Model for individual agent signals"""
    agent_name: str = Field(..., description="Name of the agent")
    agent_type: str = Field(..., description="Type of agent (technical, sentiment, etc)")
    action: str = Field(..., description="Trading action (buy, sell, hold)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ConsensusRequest(BaseModel):
    """Request model for consensus generation"""
    symbol: str = Field(..., description="Trading symbol")
    agent_signals: List[AgentSignal] = Field(..., description="List of agent signals")
    market_context: Optional[Dict[str, Any]] = Field(None, description="Market context data")
    include_options_flow: bool = Field(False, description="Include options flow analysis")
    consensus_methods: Optional[List[str]] = Field(None, description="Specific methods to use")


class PerformanceUpdate(BaseModel):
    """Model for updating agent performance"""
    agent_name: str = Field(..., description="Agent name")
    predicted_action: str = Field(..., description="What the agent predicted")
    actual_outcome: str = Field(..., description="What actually happened")
    confidence: float = Field(..., ge=0, le=1, description="Agent's confidence")


@router.post("/orchestrate")
async def orchestrate_signals(request: ConsensusRequest) -> Dict[str, Any]:
    """
    Orchestrate signals from multiple agents with Byzantine fault tolerance
    
    Features:
    - Multiple consensus methods (weighted voting, Bayesian, Byzantine)
    - Conflict detection and resolution
    - Options flow integration
    - ML ensemble optimization
    """
    try:
        # Convert agent signals to dict format
        agent_signals = [signal.dict() for signal in request.agent_signals]
        
        # Get options flow if requested
        options_flow = None
        if request.include_options_flow:
            try:
                activity = await options_flow_intelligence.detect_unusual_activity(request.symbol)
                if activity.get('smart_money_detected'):
                    options_flow = {
                        'smart_money_score': max(
                            flow.get('smart_money_score', 0) 
                            for flow in activity.get('unusual_flows', [])
                        ) if activity.get('unusual_flows') else 50,
                        'position_intent': activity.get('overall_bias', 'neutral')
                    }
            except Exception as e:
                logger.warning(f"Could not get options flow: {e}")
        
        # Orchestrate signals
        result = await meta_orchestrator.orchestrate_signals(
            agent_signals=agent_signals,
            market_context=request.market_context,
            options_flow=options_flow
        )
        
        logger.info(
            f"Consensus for {request.symbol}: {result['action']} "
            f"(confidence: {result['confidence']:.2%}, type: {result['consensus_type']})"
        )
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "consensus": result
        }
        
    except Exception as e:
        logger.error(f"Error orchestrating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weighted-voting")
async def weighted_voting(request: ConsensusRequest) -> Dict[str, Any]:
    """
    Get consensus using weighted voting method only
    """
    try:
        agent_signals = [signal.dict() for signal in request.agent_signals]
        result = meta_orchestrator.weighted_voting_consensus(agent_signals)
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "consensus": result
        }
        
    except Exception as e:
        logger.error(f"Weighted voting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bayesian")
async def bayesian_consensus(request: ConsensusRequest) -> Dict[str, Any]:
    """
    Get consensus using Bayesian updating method
    """
    try:
        agent_signals = [signal.dict() for signal in request.agent_signals]
        result = meta_orchestrator.bayesian_consensus(agent_signals)
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "consensus": result
        }
        
    except Exception as e:
        logger.error(f"Bayesian consensus error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/byzantine")
async def byzantine_consensus(request: ConsensusRequest) -> Dict[str, Any]:
    """
    Get Byzantine fault-tolerant consensus
    
    Tolerates up to f faulty agents where n >= 3f + 1
    """
    try:
        agent_signals = [signal.dict() for signal in request.agent_signals]
        result = meta_orchestrator.byzantine_fault_tolerant_consensus(agent_signals)
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "consensus": result,
            "byzantine_requirements": {
                "total_agents": len(agent_signals),
                "max_faulty_tolerated": int(len(agent_signals) * meta_orchestrator.byzantine_threshold),
                "min_agents_required": 3 * int(len(agent_signals) * meta_orchestrator.byzantine_threshold) + 1
            }
        }
        
    except Exception as e:
        logger.error(f"Byzantine consensus error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-conflicts")
async def detect_conflicts(request: ConsensusRequest) -> Dict[str, Any]:
    """
    Detect and analyze conflicts between agent signals
    """
    try:
        agent_signals = [signal.dict() for signal in request.agent_signals]
        result = meta_orchestrator.detect_signal_conflicts(agent_signals)
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "conflict_analysis": result
        }
        
    except Exception as e:
        logger.error(f"Conflict detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-performance")
async def update_performance(update: PerformanceUpdate) -> Dict[str, Any]:
    """
    Update agent performance for adaptive weighting
    """
    try:
        meta_orchestrator.update_agent_performance(
            agent_name=update.agent_name,
            predicted_action=update.predicted_action,
            actual_outcome=update.actual_outcome,
            confidence=update.confidence
        )
        
        # Get updated performance metrics
        performance = meta_orchestrator.agent_performance[update.agent_name]
        ensemble_weight = meta_orchestrator.ensemble_weights[update.agent_name]
        
        return {
            "status": "success",
            "agent_name": update.agent_name,
            "performance": {
                "total_predictions": performance['total_predictions'],
                "correct_predictions": performance['correct_predictions'],
                "recent_accuracy": performance['recent_accuracy'],
                "confidence_calibration": performance['confidence_calibration'],
                "ensemble_weight": ensemble_weight
            }
        }
        
    except Exception as e:
        logger.error(f"Performance update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-weights")
async def get_agent_weights() -> Dict[str, Any]:
    """
    Get current agent weights and performance metrics
    """
    try:
        # Compile all agent information
        agents_info = {}
        
        for agent_name, performance in meta_orchestrator.agent_performance.items():
            agents_info[agent_name] = {
                "performance": {
                    "total_predictions": performance['total_predictions'],
                    "accuracy": (
                        performance['correct_predictions'] / performance['total_predictions']
                        if performance['total_predictions'] > 0 else 0.5
                    ),
                    "recent_accuracy": performance['recent_accuracy'],
                    "confidence_calibration": performance['confidence_calibration']
                },
                "ensemble_weight": meta_orchestrator.ensemble_weights[agent_name]
            }
        
        return {
            "status": "success",
            "agent_type_weights": meta_orchestrator.agent_type_weights,
            "agent_performance": agents_info,
            "total_agents": len(agents_info)
        }
        
    except Exception as e:
        logger.error(f"Error getting agent weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods")
async def get_consensus_methods() -> Dict[str, Any]:
    """Get available consensus methods"""
    return {
        "status": "success",
        "methods": [
            {
                "name": method.value,
                "description": {
                    "weighted_voting": "Weighted voting based on agent performance and confidence",
                    "bayesian": "Bayesian probability updating for consensus",
                    "confidence_weighted": "Confidence-weighted averaging",
                    "byzantine_fault_tolerant": "Tolerates up to 33% faulty agents",
                    "ml_ensemble": "Machine learning ensemble optimization"
                }.get(method.value, "")
            }
            for method in ConsensusMethod
        ]
    }


@router.get("/conflict-levels")
async def get_conflict_levels() -> Dict[str, Any]:
    """Get conflict level definitions"""
    return {
        "status": "success",
        "conflict_levels": [
            {
                "level": level.value,
                "description": {
                    "no_conflict": "All agents agree",
                    "low_conflict": "Minor disagreements, strong majority",
                    "moderate_conflict": "Some disagreement, clear majority",
                    "high_conflict": "Significant disagreement, weak majority",
                    "extreme_conflict": "Severe disagreement, no clear consensus"
                }.get(level.value, "")
            }
            for level in ConflictLevel
        ]
    }


@router.get("/demo")
async def demo_consensus() -> Dict[str, Any]:
    """
    Demonstration of consensus with sample agent signals
    """
    try:
        # Create sample agent signals
        sample_signals = [
            {
                "agent_name": "TechnicalAgent",
                "agent_type": "technical",
                "action": "buy",
                "confidence": 0.75,
                "reasoning": "RSI oversold, MACD bullish crossover"
            },
            {
                "agent_name": "SentimentAgent",
                "agent_type": "sentiment",
                "action": "buy",
                "confidence": 0.65,
                "reasoning": "Positive social media sentiment"
            },
            {
                "agent_name": "OptionsFlowAgent",
                "agent_type": "options_flow",
                "action": "buy",
                "confidence": 0.85,
                "reasoning": "Smart money accumulating calls"
            },
            {
                "agent_name": "MacroAgent",
                "agent_type": "macro",
                "action": "hold",
                "confidence": 0.60,
                "reasoning": "Mixed economic indicators"
            },
            {
                "agent_name": "VolumeAgent",
                "agent_type": "volume",
                "action": "buy",
                "confidence": 0.70,
                "reasoning": "Increasing volume on upward moves"
            }
        ]
        
        # Run orchestration
        result = await meta_orchestrator.orchestrate_signals(
            agent_signals=sample_signals,
            market_context={"volatility": "normal"},
            options_flow={"smart_money_score": 80, "position_intent": "directional_bullish"}
        )
        
        return {
            "status": "success",
            "message": "Demo consensus with 5 agents",
            "input_signals": sample_signals,
            "consensus_result": result
        }
        
    except Exception as e:
        logger.error(f"Demo consensus error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for meta consensus system"""
    return {
        "status": "healthy",
        "service": "Meta Signal Orchestrator",
        "byzantine_threshold": meta_orchestrator.byzantine_threshold,
        "min_agents_required": meta_orchestrator.min_agents_for_consensus,
        "consensus_methods": len(ConsensusMethod),
        "tracked_agents": len(meta_orchestrator.agent_performance)
    }