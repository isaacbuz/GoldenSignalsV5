"""
Unified Orchestrator Facade
Single entry point for all orchestration needs
Routes to LangGraph orchestrator for actual implementation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from core.langgraph_orchestrator import langgraph_orchestrator, LangGraphOrchestrator
from agents.base import BaseAgent, Signal
from core.logging import get_logger
from services.websocket_manager import ws_manager

logger = get_logger(__name__)


class UnifiedOrchestrator:
    """
    Unified orchestrator that consolidates all orchestration functionality
    Acts as a facade to the LangGraph orchestrator with backward compatibility
    """
    
    def __init__(self):
        self.core_orchestrator = langgraph_orchestrator
        self.initialized = False
        
        # Track active analyses for backward compatibility
        self.active_analyses = {}
        
        logger.info("Unified Orchestrator initialized")
    
    async def initialize(self):
        """Initialize the orchestrator and all dependencies"""
        if self.initialized:
            return
        
        try:
            # Initialize core services
            await self._initialize_agents()
            
            self.initialized = True
            logger.info("Unified Orchestrator fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize and register default agents"""
        # Import agents
        from agents.technical_analysis_agent import TechnicalAnalysisAgent
        from agents.sentiment_analysis_agent import SentimentAnalysisAgent
        from agents.pattern_recognition_agent import PatternRecognitionAgent
        from agents.risk_management_agent import RiskManagementAgent
        from agents.market_dynamics_agent import MarketDynamicsAgent
        from agents.ml_prediction_agent import MLPredictionAgent
        
        # Create agent instances
        agents = [
            TechnicalAnalysisAgent(),
            SentimentAnalysisAgent(),
            PatternRecognitionAgent(),
            RiskManagementAgent(),
            MarketDynamicsAgent(),
            MLPredictionAgent()
        ]
        
        # Register with core orchestrator
        for agent in agents:
            self.core_orchestrator.register_agent(agent)
            logger.info(f"Registered agent: {agent.config.name}")
    
    # ==================== Main Analysis Methods ====================
    
    async def analyze(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        broadcast: bool = True
    ) -> Optional[Signal]:
        """
        Main analysis method - unified interface for all analysis
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Optional pre-fetched market data
            broadcast: Whether to broadcast via WebSocket
            
        Returns:
            Signal object or None
        """
        analysis_id = f"{symbol}_{datetime.now().timestamp()}"
        
        try:
            # Track analysis
            self.active_analyses[analysis_id] = {
                "symbol": symbol,
                "started_at": datetime.now(),
                "status": "running"
            }
            
            # Broadcast start if enabled
            if broadcast:
                await ws_manager.broadcast_agent_update(
                    symbol=symbol,
                    agent_name="Orchestrator",
                    signal="ANALYZING",
                    confidence=0.0
                )
            
            # Run analysis through LangGraph orchestrator
            signal = await self.core_orchestrator.analyze(
                symbol=symbol,
                thread_id=analysis_id
            )
            
            # Broadcast result if enabled
            if broadcast and signal:
                await ws_manager.broadcast_signal({
                    "symbol": symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "price": signal.current_price,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Update tracking
            self.active_analyses[analysis_id]["status"] = "completed"
            
            return signal
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            self.active_analyses[analysis_id]["status"] = "failed"
            
            if broadcast:
                await ws_manager.broadcast_decision(
                    symbol=symbol,
                    decision={"action": "ERROR", "error": str(e)}
                )
            
            return None
        
        finally:
            # Clean up after delay
            asyncio.create_task(self._cleanup_analysis(analysis_id))
    
    async def _cleanup_analysis(self, analysis_id: str):
        """Clean up analysis record after delay"""
        await asyncio.sleep(300)  # Keep for 5 minutes
        self.active_analyses.pop(analysis_id, None)
    
    # ==================== Backward Compatibility Methods ====================
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Backward compatible method for market analysis
        Used by old code expecting market_data as primary input
        """
        symbol = market_data.get("symbol")
        if not symbol:
            logger.error("No symbol in market_data")
            return None
        
        return await self.analyze(symbol, market_data)
    
    async def orchestrate_agents(self, symbol: str) -> Dict[str, Any]:
        """
        Backward compatible method returning detailed results
        Used by old code expecting detailed agent results
        """
        signal = await self.analyze(symbol, broadcast=False)
        
        if signal:
            return {
                "signal": signal,
                "consensus": {
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value
                },
                "metadata": signal.metadata or {},
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "signal": None,
            "consensus": None,
            "error": "No signal generated",
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_meta_analysis(
        self,
        symbol: str,
        agent_signals: Dict[str, Signal]
    ) -> Dict[str, Any]:
        """
        Backward compatible method for meta analysis
        The new orchestrator handles this internally
        """
        # New orchestrator handles meta analysis in God AI phase
        signal = await self.analyze(symbol)
        
        return {
            "meta_signal": signal,
            "confidence": signal.confidence if signal else 0,
            "recommendation": signal.action.value if signal else "HOLD"
        }
    
    # ==================== Agent Management ====================
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.core_orchestrator.register_agent(agent)
        logger.info(f"Registered agent: {agent.config.name}")
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names"""
        return [
            agent.config.name 
            for agent in self.core_orchestrator.agents.values()
        ]
    
    def enable_agent(self, agent_id: str):
        """Enable a specific agent"""
        if agent_id in self.core_orchestrator.agents:
            self.core_orchestrator.agents[agent_id].config.enabled = True
            logger.info(f"Enabled agent: {agent_id}")
    
    def disable_agent(self, agent_id: str):
        """Disable a specific agent"""
        if agent_id in self.core_orchestrator.agents:
            self.core_orchestrator.agents[agent_id].config.enabled = False
            logger.info(f"Disabled agent: {agent_id}")
    
    # ==================== Status and Metrics ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "initialized": self.initialized,
            "active_analyses": len(self.active_analyses),
            "registered_agents": len(self.core_orchestrator.agents),
            "enabled_agents": len([
                a for a in self.core_orchestrator.agents.values()
                if a.config.enabled
            ]),
            "metrics": self.core_orchestrator.get_metrics(),
            "current_analyses": list(self.active_analyses.keys())
        }
    
    async def get_conversation_history(self, thread_id: str) -> List[Any]:
        """Get conversation history for a thread"""
        return await self.core_orchestrator.get_conversation_history(thread_id)
    
    async def visualize_workflow(self) -> str:
        """Get workflow visualization"""
        return await self.core_orchestrator.visualize_workflow()
    
    # ==================== WebSocket Integration ====================
    
    async def analyze_with_websocket(
        self,
        symbol: str,
        client_id: str
    ) -> Optional[Signal]:
        """
        Analyze with WebSocket updates for specific client
        
        Args:
            symbol: Stock symbol
            client_id: WebSocket client ID
            
        Returns:
            Signal or None
        """
        # Use client_id as thread_id for conversation continuity
        signal = await self.core_orchestrator.analyze(
            symbol=symbol,
            thread_id=client_id
        )
        
        # Broadcasting is handled internally by LangGraph orchestrator
        return signal


# ==================== Singleton Instances ====================

# Primary unified orchestrator
unified_orchestrator = UnifiedOrchestrator()

# Compatibility aliases for backward compatibility
orchestrator = unified_orchestrator  # Generic alias
agent_orchestrator = unified_orchestrator  # For agent-specific code
trading_orchestrator = unified_orchestrator  # For trading-specific code
websocket_orchestrator = unified_orchestrator  # For WebSocket-specific code


# ==================== Helper Functions ====================

async def analyze_symbol(symbol: str, **kwargs) -> Optional[Signal]:
    """
    Helper function for quick analysis
    
    Args:
        symbol: Stock symbol
        **kwargs: Additional arguments passed to analyze
        
    Returns:
        Signal or None
    """
    if not unified_orchestrator.initialized:
        await unified_orchestrator.initialize()
    
    return await unified_orchestrator.analyze(symbol, **kwargs)


async def get_orchestrator_status() -> Dict[str, Any]:
    """Get current orchestrator status"""
    return unified_orchestrator.get_status()


# ==================== Initialization ====================

async def initialize_orchestrator():
    """Initialize the unified orchestrator"""
    await unified_orchestrator.initialize()
    logger.info("Orchestrator initialization complete")