"""
Agent Orchestrator
Manages multiple trading agents and aggregates their signals
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import statistics

from agents.base import BaseAgent, Signal, SignalAction, SignalStrength
from agents.technical.technical_analyst import TechnicalAnalystAgent
from agents.sentiment_aggregator import sentiment_aggregator_agent
from agents.economic_indicator_agent import economic_indicator_agent
from agents.llm.fingpt_agent import fingpt_agent
from core.logging import get_logger
from services.signal_service import SignalService
from core.database import get_db
from rag.core import RAGEngine, DocumentType

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple trading agents
    
    Features:
    - Parallel agent execution
    - Signal aggregation and consensus
    - Performance-based weighting
    - Dynamic agent management
    """
    
    def __init__(self, signal_service: Optional[SignalService] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.signal_service = signal_service
        self._is_running = False
        self._stop_event = asyncio.Event()
        
        # Consensus parameters
        self.min_agents_for_signal = 1  # Minimum agents agreeing for signal
        self.consensus_threshold = 0.6  # Minimum weighted agreement
        
        # RAG engine for context augmentation
        self.rag_engine = RAGEngine()
        self.use_rag = True  # Enable RAG by default
        
        logger.info("Agent Orchestrator initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.config.name} (ID: {agent.agent_id})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            logger.info(f"Unregistered agent: {agent.config.name}")
    
    async def initialize_default_agents(self) -> None:
        """Initialize default set of agents and RAG engine"""
        # Initialize RAG engine
        if self.use_rag:
            await self.rag_engine.initialize()
            logger.info("RAG engine initialized")
        
        # FinGPT Agent - Primary LLM agent that consolidates analysis
        await fingpt_agent.initialize()
        self.register_agent(fingpt_agent)
        logger.info("FinGPT agent registered - replacing multiple sentiment/analysis agents")
        
        # Technical Analysis Agent
        technical_agent = TechnicalAnalystAgent()
        await technical_agent.initialize()
        self.register_agent(technical_agent)
        
        # Sentiment Analysis Agent (can be disabled if FinGPT handles it)
        # await sentiment_aggregator_agent.initialize()
        # self.register_agent(sentiment_aggregator_agent)
        
        # Economic Indicator Agent
        await economic_indicator_agent.initialize()
        self.register_agent(economic_indicator_agent)
        
        logger.info(f"Initialized {len(self.agents)} default agents")
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Analyze market data using all active agents
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Aggregated signal or None
        """
        if not self.agents:
            logger.warning("No agents registered")
            return None
        
        # Get enabled agents
        active_agents = [
            agent for agent in self.agents.values() 
            if agent.config.enabled
        ]
        
        if not active_agents:
            logger.warning("No active agents available")
            return None
        
        # Execute agents in parallel
        tasks = [
            agent.execute_with_monitoring(market_data)
            for agent in active_agents
        ]
        
        # Gather results with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max(agent.config.timeout for agent in active_agents) + 5
            )
        except asyncio.TimeoutError:
            logger.error("Agent orchestration timed out")
            return None
        
        # Filter valid signals
        valid_signals: List[tuple[BaseAgent, Signal]] = []
        for agent, result in zip(active_agents, results):
            if isinstance(result, Signal):
                valid_signals.append((agent, result))
            elif isinstance(result, Exception):
                logger.error(f"Agent {agent.config.name} failed: {str(result)}")
        
        if not valid_signals:
            logger.debug("No valid signals generated")
            return None
        
        # Aggregate signals
        aggregated_signal = await self._aggregate_signals(valid_signals, market_data)
        
        if aggregated_signal and self.signal_service:
            # Store signal in database
            async for db in get_db():
                stored_signal = await self.signal_service.create_signal(
                    db=db,
                    symbol=aggregated_signal.symbol,
                    action=aggregated_signal.action.value,
                    confidence=aggregated_signal.confidence,
                    source="Orchestrator",
                    agent_id=None,
                    metadata=aggregated_signal.dict(exclude={'symbol', 'action', 'confidence', 'source'})
                )
                logger.info(f"Stored aggregated signal: {stored_signal.id}")
                break
        
        return aggregated_signal
    
    async def _aggregate_signals(
        self, 
        agent_signals: List[tuple[BaseAgent, Signal]], 
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Aggregate multiple agent signals into consensus signal
        
        Args:
            agent_signals: List of (agent, signal) tuples
            market_data: Original market data
            
        Returns:
            Aggregated signal or None
        """
        if len(agent_signals) < self.min_agents_for_signal:
            return None
        
        # Group signals by action
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        total_weight = 0
        
        for agent, signal in agent_signals:
            weight = agent.config.weight * signal.confidence
            total_weight += agent.config.weight
            
            weighted_signal = (agent, signal, weight)
            
            if signal.action == SignalAction.BUY:
                buy_signals.append(weighted_signal)
            elif signal.action == SignalAction.SELL:
                sell_signals.append(weighted_signal)
            else:
                hold_signals.append(weighted_signal)
        
        # Calculate weighted consensus
        buy_weight = sum(w for _, _, w in buy_signals)
        sell_weight = sum(w for _, _, w in sell_signals)
        hold_weight = sum(w for _, _, w in hold_signals)
        
        total_signal_weight = buy_weight + sell_weight + hold_weight
        
        if total_signal_weight == 0:
            return None
        
        # Determine consensus action
        if buy_weight > sell_weight and buy_weight > hold_weight:
            consensus_action = SignalAction.BUY
            consensus_signals = buy_signals
            consensus_weight = buy_weight / total_weight
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            consensus_action = SignalAction.SELL
            consensus_signals = sell_signals
            consensus_weight = sell_weight / total_weight
        else:
            consensus_action = SignalAction.HOLD
            consensus_signals = hold_signals
            consensus_weight = hold_weight / total_weight
        
        # Check consensus threshold
        if consensus_weight < self.consensus_threshold:
            logger.debug(
                f"Consensus weight {consensus_weight:.2f} below threshold {self.consensus_threshold}"
            )
            return None
        
        # Aggregate signal properties
        all_reasoning = []
        all_features = {}
        all_indicators = {}
        confidences = []
        
        for agent, signal, weight in consensus_signals:
            all_reasoning.extend([f"[{agent.config.name}] {r}" for r in signal.reasoning])
            all_features[f"{agent.config.name}_features"] = signal.features
            all_indicators.update(signal.indicators)
            confidences.append(signal.confidence)
        
        # Calculate aggregated confidence
        avg_confidence = statistics.mean(confidences)
        weighted_confidence = sum(s.confidence * w for _, s, w in consensus_signals) / sum(w for _, _, w in consensus_signals)
        
        # Determine signal strength based on consensus
        if consensus_weight >= 0.8:
            strength = SignalStrength.STRONG
        elif consensus_weight >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Calculate aggregated targets
        if consensus_signals:
            # Use weighted average for targets
            target_prices = [s.target_price for _, s, _ in consensus_signals if s.target_price]
            stop_losses = [s.stop_loss for _, s, _ in consensus_signals if s.stop_loss]
            
            target_price = statistics.mean(target_prices) if target_prices else None
            stop_loss = statistics.mean(stop_losses) if stop_losses else None
        else:
            target_price = None
            stop_loss = None
        
        # Create aggregated signal
        aggregated_signal = Signal(
            symbol=market_data['symbol'],
            action=consensus_action,
            confidence=min(weighted_confidence, 0.95),
            strength=strength,
            source="Orchestrator",
            current_price=market_data['price'],
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=all_reasoning[:10],  # Limit reasoning items
            features={
                "consensus_weight": consensus_weight,
                "total_agents": len(agent_signals),
                "agreeing_agents": len(consensus_signals),
                "agent_details": all_features
            },
            indicators=all_indicators,
            market_conditions={
                "timestamp": datetime.utcnow().isoformat(),
                "consensus_type": consensus_action.value,
                "signal_distribution": {
                    "buy": len(buy_signals),
                    "sell": len(sell_signals),
                    "hold": len(hold_signals)
                }
            }
        )
        
        logger.info(
            f"Generated consensus signal: {consensus_action.value} "
            f"with confidence {weighted_confidence:.2f} "
            f"({len(consensus_signals)}/{len(agent_signals)} agents agree)"
        )
        
        # Apply RAG augmentation if enabled
        if self.use_rag:
            try:
                # Create query from signal context
                query = f"{market_data['symbol']} {consensus_action.value} signal technical analysis"
                
                # Retrieve relevant context
                rag_context = await self.rag_engine.retrieve_context(
                    query=query,
                    k=5,
                    doc_types=[DocumentType.SIGNAL_HISTORY, DocumentType.TECHNICAL_PATTERN]
                )
                
                # Augment signal with RAG context
                signal_dict = aggregated_signal.dict()
                augmented_dict = await self.rag_engine.generate_augmented_signal(
                    signal_dict, 
                    rag_context
                )
                
                # Update signal confidence and features
                aggregated_signal.confidence = augmented_dict.get("confidence", aggregated_signal.confidence)
                aggregated_signal.features["rag_context"] = augmented_dict.get("rag_context", {})
                aggregated_signal.features["rag_insights"] = augmented_dict.get("rag_insights", [])
                
                if augmented_dict.get("rag_reasoning"):
                    aggregated_signal.reasoning.append(f"[RAG] {augmented_dict['rag_reasoning']}")
                
                logger.info("Applied RAG augmentation to consensus signal")
                
            except Exception as e:
                logger.error(f"RAG augmentation failed: {e}", exc_info=True)
                # Continue with original signal if RAG fails
        
        return aggregated_signal
    
    async def update_agent_performance(self, signal_id: str, outcome: Dict[str, Any]) -> None:
        """
        Update agent performance based on signal outcome
        
        Args:
            signal_id: ID of the signal
            outcome: Dictionary with outcome details (was_correct, profit, etc.)
        """
        # This would need to track which agents contributed to each signal
        # For now, update all agents (in production, track signal -> agent mapping)
        for agent in self.agents.values():
            agent.update_performance_feedback(
                signal_id,
                outcome.get('was_correct', False),
                outcome.get('profit')
            )
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        summary = {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.config.enabled),
            "agents": {}
        }
        
        for agent_id, agent in self.agents.items():
            summary["agents"][agent.config.name] = agent.get_current_performance()
        
        return summary
    
    async def rebalance_weights(self) -> None:
        """Rebalance agent weights based on performance"""
        if not self.agents:
            return
        
        # Get performance metrics
        performances = []
        for agent in self.agents.values():
            if agent.performance.total_signals >= 10:  # Minimum signals
                performances.append((
                    agent,
                    agent.performance.accuracy * agent.performance.avg_confidence
                ))
        
        if not performances:
            return
        
        # Normalize weights based on performance
        total_score = sum(score for _, score in performances)
        
        if total_score > 0:
            for agent, score in performances:
                new_weight = score / total_score
                agent.adjust_weight(new_weight)
        
        logger.info("Rebalanced agent weights based on performance")
    
    async def start(self) -> None:
        """Start the orchestrator and all agents"""
        self._is_running = True
        self._stop_event.clear()
        
        # Start all agents
        await asyncio.gather(*[
            agent.start() for agent in self.agents.values()
        ])
        
        logger.info("Agent Orchestrator started")
    
    async def stop(self) -> None:
        """Stop the orchestrator and all agents"""
        self._is_running = False
        self._stop_event.set()
        
        # Stop all agents
        await asyncio.gather(*[
            agent.stop() for agent in self.agents.values()
        ])
        
        logger.info("Agent Orchestrator stopped")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator"""
        await self.stop()
        
        # Shutdown all agents
        await asyncio.gather(*[
            agent.shutdown() for agent in self.agents.values()
        ])
        
        logger.info("Agent Orchestrator shutdown complete")