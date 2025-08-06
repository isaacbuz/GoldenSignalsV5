"""
Main Orchestrator - Integrates RAG, MCP, and Agents
Central hub for AI-powered trading operations
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

from .rag.rag_system import RAGSystem
from .mcp.mcp_server import MCPServer, MCPClient
from ..models.signal import Signal
from ..models.agent import Agent

logger = logging.getLogger(__name__)

class TradingAgent:
    """Base class for trading agents"""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.rag_system = None
        self.mcp_client = None
        
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on market data"""
        raise NotImplementedError
        
    def set_rag_system(self, rag_system: RAGSystem):
        """Set RAG system for context-aware analysis"""
        self.rag_system = rag_system
        
    def set_mcp_client(self, mcp_client: MCPClient):
        """Set MCP client for tool access"""
        self.mcp_client = mcp_client

class TradingOrchestrator:
    """
    Main orchestrator that coordinates:
    - RAG system for market intelligence
    - MCP server for standardized AI interactions
    - Trading agents for analysis
    - Signal generation and execution
    """
    
    def __init__(self):
        # Initialize core systems
        self.rag_system = RAGSystem()
        self.mcp_server = MCPServer()
        self.mcp_client = MCPClient(self.mcp_server)
        
        # Agent registry
        self.agents: Dict[str, TradingAgent] = {}
        
        # System state
        self.is_running = False
        
    async def initialize(self):
        """Initialize the orchestrator and all subsystems"""
        logger.info("Initializing Trading Orchestrator...")
        
        # Register MCP tools specific to orchestration
        self._register_orchestration_tools()
        
        # Initialize agents
        await self._initialize_agents()
        
        # Load initial market knowledge into RAG
        await self._load_initial_knowledge()
        
        self.is_running = True
        logger.info("Trading Orchestrator initialized successfully")
        
    def _register_orchestration_tools(self):
        """Register orchestration-specific MCP tools"""
        # Custom tools can be added here
        pass
        
    async def _initialize_agents(self):
        """Initialize and register trading agents"""
        # Agents will be loaded from the agents directory
        # For now, we'll create placeholder
        logger.info("Initializing trading agents...")
        
    async def _load_initial_knowledge(self):
        """Load initial knowledge into RAG system"""
        initial_data = [
            {
                "id": "market_rules_001",
                "content": "Trading hours: 9:30 AM - 4:00 PM EST. Pre-market: 4:00 AM - 9:30 AM.",
                "metadata": {"type": "market_rules"}
            },
            {
                "id": "risk_rules_001",
                "content": "Maximum position size: 10% of portfolio. Stop loss: 2% max loss per trade.",
                "metadata": {"type": "risk_management"}
            }
        ]
        
        await self.rag_system.ingest_market_data(initial_data)
        
    async def process_market_update(self, symbol: str, data: Dict[str, Any]) -> Signal:
        """
        Process market update through the full pipeline:
        1. Enrich with RAG context
        2. Run through agents via MCP
        3. Generate consolidated signal
        """
        logger.info(f"Processing market update for {symbol}")
        
        # Step 1: Get market context from RAG
        query = f"What is the current market context and trading setup for {symbol}?"
        rag_insights = await self.rag_system.process_query(query)
        
        # Step 2: Fetch additional data via MCP tools
        market_data = await self.mcp_client.execute_tool(
            "get_market_data",
            {"symbol": symbol, "timeframe": "1h"}
        )
        
        # Step 3: Run through agents (if available)
        agent_analyses = {}
        for agent_name, agent in self.agents.items():
            try:
                analysis = await agent.analyze(symbol, market_data)
                agent_analyses[agent_name] = analysis
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                
        # Step 4: Generate signal via MCP
        signal_params = {
            "symbol": symbol,
            "market_data": market_data,
            "rag_insights": rag_insights,
            "agent_analyses": agent_analyses
        }
        
        signal_data = await self.mcp_client.execute_tool(
            "generate_signal",
            signal_params
        )
        
        # Step 5: Create signal object
        signal = Signal(
            symbol=symbol,
            action=signal_data.get("signal", "HOLD"),
            confidence=signal_data.get("confidence", 0.5),
            price=market_data.get("price"),
            timestamp=datetime.now(),
            metadata={
                "rag_insights": rag_insights,
                "agent_analyses": agent_analyses,
                "mcp_tools_used": ["get_market_data", "generate_signal"]
            }
        )
        
        logger.info(f"Generated signal for {symbol}: {signal.action} (confidence: {signal.confidence})")
        return signal
        
    async def execute_trading_strategy(self, strategy_name: str, symbols: List[str]):
        """Execute a complete trading strategy"""
        logger.info(f"Executing strategy: {strategy_name} for symbols: {symbols}")
        
        signals = []
        
        for symbol in symbols:
            try:
                # Get strategy context from RAG
                strategy_query = f"What are the rules and parameters for {strategy_name} strategy?"
                strategy_context = await self.rag_system.process_query(strategy_query)
                
                # Process each symbol
                signal = await self.process_market_update(symbol, {})
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                
        return signals
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            "orchestrator": {
                "is_running": self.is_running,
                "timestamp": datetime.now().isoformat()
            },
            "rag_system": {
                "documents_count": len(self.rag_system.vector_store.documents),
                "status": "active"
            },
            "mcp_server": {
                "tools_count": len(self.mcp_server.tools),
                "tools": list(self.mcp_server.tools.keys())
            },
            "agents": {
                "count": len(self.agents),
                "active_agents": list(self.agents.keys())
            }
        }
        
        return status

# Global orchestrator instance
orchestrator = TradingOrchestrator()

async def initialize_orchestrator():
    """Initialize the global orchestrator"""
    await orchestrator.initialize()
    return orchestrator