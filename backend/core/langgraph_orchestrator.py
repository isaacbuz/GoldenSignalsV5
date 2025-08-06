"""
LangGraph-based God AI Orchestrator
Implements advanced agent orchestration with state management and decision flow
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from datetime import datetime
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.base import BaseAgent, Signal, SignalAction, SignalStrength
from core.ai_brain import CentralAIBrain
from core.pattern_recognition import PatternRecognitionService
from services.market_data_unified import UnifiedMarketDataService
from services.rag_service import RAGService
from core.logging import get_logger

logger = get_logger(__name__)


class MarketState(TypedDict):
    """State for market analysis workflow"""
    symbol: str
    market_data: Dict[str, Any]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_signals: Dict[str, Signal]
    pattern_results: Dict[str, Any]
    rag_context: List[Dict[str, Any]]
    god_ai_decision: Optional[Dict[str, Any]]
    final_signal: Optional[Signal]
    risk_parameters: Dict[str, float]
    execution_plan: Optional[Dict[str, Any]]
    errors: List[str]
    timestamp: str


class ExecutionPhase(Enum):
    """Execution phases in the orchestration flow"""
    DATA_GATHERING = "data_gathering"
    PATTERN_DETECTION = "pattern_detection"
    AGENT_ANALYSIS = "agent_analysis"
    RAG_ENRICHMENT = "rag_enrichment"
    GOD_AI_DECISION = "god_ai_decision"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTION_PLANNING = "execution_planning"
    SIGNAL_GENERATION = "signal_generation"


class LangGraphOrchestrator:
    """
    Advanced orchestrator using LangGraph for complex agent coordination
    Implements the God AI architecture with sophisticated decision flow
    """
    
    def __init__(self):
        self.graph = None
        self.app = None
        self.checkpointer = MemorySaver()
        
        # Core services
        self.god_ai = CentralAIBrain()
        self.pattern_service = PatternRecognitionService()
        self.market_service = UnifiedMarketDataService()
        self.rag_service = RAGService()
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        
        # Performance metrics
        self.metrics = {
            "total_analyses": 0,
            "successful_signals": 0,
            "failed_analyses": 0,
            "average_confidence": 0.0,
            "phase_timings": {}
        }
        
        # Build the graph
        self._build_graph()
        
        logger.info("LangGraph Orchestrator initialized")
    
    def _build_graph(self):
        """Build the LangGraph state machine"""
        workflow = StateGraph(MarketState)
        
        # Add nodes for each phase
        workflow.add_node("gather_data", self._gather_market_data)
        workflow.add_node("detect_patterns", self._detect_patterns)
        workflow.add_node("run_agents", self._run_agent_analysis)
        workflow.add_node("enrich_rag", self._enrich_with_rag)
        workflow.add_node("god_ai_decide", self._god_ai_decision)
        workflow.add_node("assess_risk", self._assess_risk)
        workflow.add_node("plan_execution", self._plan_execution)
        workflow.add_node("generate_signal", self._generate_final_signal)
        
        # Define the flow
        workflow.set_entry_point("gather_data")
        
        # Add edges with conditions
        workflow.add_edge("gather_data", "detect_patterns")
        workflow.add_edge("detect_patterns", "run_agents")
        workflow.add_edge("run_agents", "enrich_rag")
        workflow.add_edge("enrich_rag", "god_ai_decide")
        
        # Conditional routing based on God AI decision
        workflow.add_conditional_edges(
            "god_ai_decide",
            self._should_generate_signal,
            {
                "continue": "assess_risk",
                "no_signal": END
            }
        )
        
        workflow.add_edge("assess_risk", "plan_execution")
        workflow.add_edge("plan_execution", "generate_signal")
        workflow.add_edge("generate_signal", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("LangGraph workflow compiled successfully")
    
    async def _gather_market_data(self, state: MarketState) -> MarketState:
        """Phase 1: Gather comprehensive market data"""
        start_time = datetime.now()
        
        try:
            # Fetch multi-source market data
            market_data = await self.market_service.get_comprehensive_data(
                state["symbol"],
                include_news=True,
                include_options=True,
                include_fundamentals=True
            )
            
            if not market_data:
                state["errors"].append(f"Failed to fetch market data for {state['symbol']}")
                return state
            
            state["market_data"] = market_data
            state["messages"].append(
                SystemMessage(content=f"Market data gathered for {state['symbol']}")
            )
            
            # Update metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["data_gathering"] = elapsed
            
            logger.info(f"Market data gathered for {state['symbol']} in {elapsed:.2f}s")
            
        except Exception as e:
            state["errors"].append(f"Data gathering error: {str(e)}")
            logger.error(f"Error gathering market data: {e}")
        
        return state
    
    async def _detect_patterns(self, state: MarketState) -> MarketState:
        """Phase 2: Detect technical patterns"""
        start_time = datetime.now()
        
        try:
            if not state.get("market_data"):
                state["errors"].append("No market data available for pattern detection")
                return state
            
            # Run pattern detection
            patterns = await self.pattern_service.detect_patterns(
                state["market_data"],
                min_confidence=60
            )
            
            state["pattern_results"] = patterns
            state["messages"].append(
                AIMessage(content=f"Detected {len(patterns)} patterns")
            )
            
            # Log significant patterns
            for pattern in patterns[:3]:  # Top 3 patterns
                logger.info(f"Pattern detected: {pattern['pattern_name']} "
                          f"(confidence: {pattern['confidence']:.1f}%)")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["pattern_detection"] = elapsed
            
        except Exception as e:
            state["errors"].append(f"Pattern detection error: {str(e)}")
            logger.error(f"Error in pattern detection: {e}")
        
        return state
    
    async def _run_agent_analysis(self, state: MarketState) -> MarketState:
        """Phase 3: Run parallel agent analysis"""
        start_time = datetime.now()
        
        try:
            if not state.get("market_data"):
                state["errors"].append("No market data for agent analysis")
                return state
            
            # Prepare agent tasks
            agent_tasks = []
            for agent_id, agent in self.agents.items():
                if agent.config.enabled:
                    agent_tasks.append(
                        self._run_single_agent(agent, state["market_data"])
                    )
            
            # Execute agents in parallel
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process results
            agent_signals = {}
            for agent_id, result in zip(self.agents.keys(), results):
                if isinstance(result, Signal):
                    agent_signals[agent_id] = result
                    state["messages"].append(
                        AIMessage(content=f"Agent {agent_id}: {result.action.value} "
                                        f"(confidence: {result.confidence:.2f})")
                    )
                elif isinstance(result, Exception):
                    logger.error(f"Agent {agent_id} failed: {result}")
            
            state["agent_signals"] = agent_signals
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["agent_analysis"] = elapsed
            
            logger.info(f"Completed {len(agent_signals)} agent analyses in {elapsed:.2f}s")
            
        except Exception as e:
            state["errors"].append(f"Agent analysis error: {str(e)}")
            logger.error(f"Error in agent analysis: {e}")
        
        return state
    
    async def _run_single_agent(self, agent: BaseAgent, market_data: Dict[str, Any]) -> Signal:
        """Run a single agent with monitoring"""
        try:
            return await agent.execute_with_monitoring(market_data)
        except Exception as e:
            logger.error(f"Agent {agent.agent_id} execution failed: {e}")
            raise
    
    async def _enrich_with_rag(self, state: MarketState) -> MarketState:
        """Phase 4: Enrich analysis with RAG context"""
        start_time = datetime.now()
        
        try:
            # Build query from current state
            query_parts = [
                f"Analysis for {state['symbol']}",
                f"Current price: {state['market_data'].get('price', 'N/A')}",
            ]
            
            # Add pattern insights
            if state.get("pattern_results"):
                top_pattern = state["pattern_results"][0] if state["pattern_results"] else None
                if top_pattern:
                    query_parts.append(f"Pattern detected: {top_pattern['pattern_name']}")
            
            # Add agent consensus
            if state.get("agent_signals"):
                actions = [s.action.value for s in state["agent_signals"].values()]
                query_parts.append(f"Agent signals: {', '.join(actions)}")
            
            query = " | ".join(query_parts)
            
            # Query RAG system
            rag_context = await self.rag_service.query(
                query,
                k=5,
                filters={"symbol": state["symbol"]}
            )
            
            state["rag_context"] = rag_context
            state["messages"].append(
                SystemMessage(content=f"Retrieved {len(rag_context)} relevant contexts from RAG")
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["rag_enrichment"] = elapsed
            
        except Exception as e:
            state["errors"].append(f"RAG enrichment error: {str(e)}")
            logger.error(f"Error in RAG enrichment: {e}")
        
        return state
    
    async def _god_ai_decision(self, state: MarketState) -> MarketState:
        """Phase 5: God AI makes the strategic decision"""
        start_time = datetime.now()
        
        try:
            # Prepare comprehensive context for God AI
            context = {
                "symbol": state["symbol"],
                "market_data": state["market_data"],
                "patterns": state.get("pattern_results", []),
                "agent_signals": {
                    agent_id: {
                        "action": signal.action.value,
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning[:2]
                    }
                    for agent_id, signal in state.get("agent_signals", {}).items()
                },
                "rag_insights": state.get("rag_context", [])
            }
            
            # God AI strategic decision
            decision = await self.god_ai.think(
                state["symbol"],
                state["market_data"],
                state.get("agent_signals", {})
            )
            
            state["god_ai_decision"] = decision
            
            # Log decision
            if decision.get("should_trade"):
                state["messages"].append(
                    AIMessage(content=f"God AI Decision: {decision['action']} "
                                    f"(confidence: {decision['confidence']:.2f})")
                )
                logger.info(f"God AI recommends: {decision['action']} for {state['symbol']}")
            else:
                state["messages"].append(
                    AIMessage(content="God AI Decision: No trade signal")
                )
                logger.info(f"God AI: No signal for {state['symbol']}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["god_ai_decision"] = elapsed
            
        except Exception as e:
            state["errors"].append(f"God AI decision error: {str(e)}")
            logger.error(f"Error in God AI decision: {e}")
        
        return state
    
    def _should_generate_signal(self, state: MarketState) -> str:
        """Determine if we should generate a trading signal"""
        if not state.get("god_ai_decision"):
            return "no_signal"
        
        decision = state["god_ai_decision"]
        if decision.get("should_trade") and decision.get("confidence", 0) >= 0.6:
            return "continue"
        
        return "no_signal"
    
    async def _assess_risk(self, state: MarketState) -> MarketState:
        """Phase 6: Comprehensive risk assessment"""
        start_time = datetime.now()
        
        try:
            decision = state.get("god_ai_decision", {})
            market_data = state.get("market_data", {})
            
            # Calculate risk parameters
            current_price = market_data.get("price", 0)
            volatility = market_data.get("volatility", 0.02)  # Default 2%
            
            # ATR-based stop loss
            atr = market_data.get("atr", current_price * 0.02)
            
            risk_params = {
                "stop_loss": current_price - (2 * atr) if decision.get("action") == "BUY" else current_price + (2 * atr),
                "take_profit": current_price + (3 * atr) if decision.get("action") == "BUY" else current_price - (3 * atr),
                "position_size": self._calculate_position_size(volatility, decision.get("confidence", 0.5)),
                "max_risk_percent": 2.0,  # Max 2% portfolio risk
                "risk_reward_ratio": 1.5,
                "volatility": volatility,
                "correlation_risk": await self._assess_correlation_risk(state["symbol"])
            }
            
            state["risk_parameters"] = risk_params
            state["messages"].append(
                SystemMessage(content=f"Risk assessed: Stop @ {risk_params['stop_loss']:.2f}, "
                                    f"Target @ {risk_params['take_profit']:.2f}")
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["risk_assessment"] = elapsed
            
        except Exception as e:
            state["errors"].append(f"Risk assessment error: {str(e)}")
            logger.error(f"Error in risk assessment: {e}")
        
        return state
    
    def _calculate_position_size(self, volatility: float, confidence: float) -> float:
        """Calculate position size using Kelly Criterion"""
        # Simplified Kelly formula
        win_prob = confidence
        win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk
        
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Apply safety factor (use 25% of Kelly)
        safe_fraction = kelly_fraction * 0.25
        
        # Adjust for volatility
        vol_adjusted = safe_fraction * (1 - volatility)
        
        # Cap at 10% max position
        return min(max(vol_adjusted, 0.01), 0.10)
    
    async def _assess_correlation_risk(self, symbol: str) -> float:
        """Assess portfolio correlation risk"""
        # Simplified correlation assessment
        # In production, would check against existing positions
        return 0.3  # Placeholder
    
    async def _plan_execution(self, state: MarketState) -> MarketState:
        """Phase 7: Create execution plan"""
        start_time = datetime.now()
        
        try:
            decision = state.get("god_ai_decision", {})
            risk_params = state.get("risk_parameters", {})
            
            execution_plan = {
                "symbol": state["symbol"],
                "action": decision.get("action"),
                "order_type": "LIMIT",  # Default to limit orders
                "price": state["market_data"].get("price"),
                "quantity": self._calculate_quantity(
                    state["market_data"].get("price", 0),
                    risk_params.get("position_size", 0.01)
                ),
                "time_in_force": "GTC",  # Good Till Cancelled
                "stop_loss": risk_params.get("stop_loss"),
                "take_profit": risk_params.get("take_profit"),
                "execution_strategy": "TWAP",  # Time-Weighted Average Price
                "slices": 5,  # Split into 5 orders
                "interval_minutes": 2,  # 2 minutes between slices
                "conditions": {
                    "max_slippage": 0.002,  # 0.2% max slippage
                    "min_liquidity": 100000,  # Min volume requirement
                    "avoid_news": True  # Don't execute during news events
                }
            }
            
            state["execution_plan"] = execution_plan
            state["messages"].append(
                SystemMessage(content=f"Execution plan created: {execution_plan['action']} "
                                    f"{execution_plan['quantity']} shares via {execution_plan['execution_strategy']}")
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["execution_planning"] = elapsed
            
        except Exception as e:
            state["errors"].append(f"Execution planning error: {str(e)}")
            logger.error(f"Error in execution planning: {e}")
        
        return state
    
    def _calculate_quantity(self, price: float, position_size_pct: float) -> int:
        """Calculate order quantity based on position size"""
        # Assume $100,000 portfolio (would be dynamic in production)
        portfolio_value = 100000
        position_value = portfolio_value * position_size_pct
        
        if price > 0:
            return int(position_value / price)
        return 0
    
    async def _generate_final_signal(self, state: MarketState) -> MarketState:
        """Phase 8: Generate the final trading signal"""
        start_time = datetime.now()
        
        try:
            decision = state.get("god_ai_decision", {})
            risk_params = state.get("risk_parameters", {})
            execution_plan = state.get("execution_plan", {})
            
            # Map string action to enum
            action_map = {
                "BUY": SignalAction.BUY,
                "SELL": SignalAction.SELL,
                "HOLD": SignalAction.HOLD
            }
            
            # Determine signal strength
            confidence = decision.get("confidence", 0.5)
            if confidence >= 0.8:
                strength = SignalStrength.STRONG
            elif confidence >= 0.6:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Create final signal
            signal = Signal(
                symbol=state["symbol"],
                action=action_map.get(decision.get("action", "HOLD"), SignalAction.HOLD),
                confidence=confidence,
                strength=strength,
                reasoning=decision.get("reasoning", []),
                current_price=state["market_data"].get("price", 0),
                target_price=risk_params.get("take_profit"),
                stop_loss=risk_params.get("stop_loss"),
                indicators={
                    "patterns": len(state.get("pattern_results", [])),
                    "agent_consensus": len([s for s in state.get("agent_signals", {}).values() 
                                           if s.action.value == decision.get("action")]),
                    "risk_reward": risk_params.get("risk_reward_ratio", 0),
                    "position_size": risk_params.get("position_size", 0)
                },
                metadata={
                    "orchestrator": "LangGraph",
                    "god_ai_confidence": decision.get("confidence"),
                    "execution_plan": execution_plan,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            state["final_signal"] = signal
            state["messages"].append(
                AIMessage(content=f"Final Signal: {signal.action.value} {state['symbol']} "
                                f"@ {signal.current_price:.2f} (confidence: {signal.confidence:.2f})")
            )
            
            # Update metrics
            self.metrics["total_analyses"] += 1
            self.metrics["successful_signals"] += 1
            self.metrics["average_confidence"] = (
                (self.metrics["average_confidence"] * (self.metrics["successful_signals"] - 1) + confidence) /
                self.metrics["successful_signals"]
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["phase_timings"]["signal_generation"] = elapsed
            
            logger.info(f"Generated signal: {signal.action.value} for {state['symbol']} "
                       f"with confidence {confidence:.2f}")
            
        except Exception as e:
            state["errors"].append(f"Signal generation error: {str(e)}")
            logger.error(f"Error in signal generation: {e}")
            self.metrics["failed_analyses"] += 1
        
        return state
    
    async def analyze(self, symbol: str, thread_id: Optional[str] = None) -> Optional[Signal]:
        """
        Main entry point for analysis using LangGraph orchestration
        
        Args:
            symbol: Stock symbol to analyze
            thread_id: Optional thread ID for conversation continuity
            
        Returns:
            Signal object or None
        """
        # Initialize state
        initial_state: MarketState = {
            "symbol": symbol,
            "market_data": {},
            "messages": [HumanMessage(content=f"Analyze {symbol}")],
            "agent_signals": {},
            "pattern_results": {},
            "rag_context": [],
            "god_ai_decision": None,
            "final_signal": None,
            "risk_parameters": {},
            "execution_plan": None,
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Configure thread
        config = RunnableConfig(
            configurable={"thread_id": thread_id or f"{symbol}_{datetime.now().timestamp()}"}
        )
        
        try:
            # Run the graph
            logger.info(f"Starting LangGraph analysis for {symbol}")
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Check for errors
            if final_state.get("errors"):
                logger.error(f"Analysis completed with errors: {final_state['errors']}")
            
            # Return final signal
            return final_state.get("final_signal")
            
        except Exception as e:
            logger.error(f"LangGraph orchestration failed: {e}")
            self.metrics["failed_analyses"] += 1
            return None
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.config.name} ({agent.agent_id})")
    
    async def get_conversation_history(self, thread_id: str) -> List[BaseMessage]:
        """Retrieve conversation history for a thread"""
        config = RunnableConfig(configurable={"thread_id": thread_id})
        checkpoint = self.checkpointer.get(config)
        
        if checkpoint and "messages" in checkpoint:
            return checkpoint["messages"]
        return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        return {
            **self.metrics,
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.config.enabled]),
            "god_ai_status": "active" if self.god_ai else "inactive"
        }
    
    async def visualize_workflow(self) -> str:
        """Generate workflow visualization"""
        try:
            # Get graph structure
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            logger.error(f"Failed to generate workflow visualization: {e}")
            return "Visualization not available"


# Singleton instance
langgraph_orchestrator = LangGraphOrchestrator()