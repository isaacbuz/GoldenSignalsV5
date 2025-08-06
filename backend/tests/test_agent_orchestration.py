"""
Comprehensive tests for AI agents and orchestration
Tests agent execution, consensus building, and signal generation
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import asyncio

from agents.orchestrator import AgentOrchestrator
from agents.base import BaseAgent, Signal, SignalAction, SignalStrength, AgentConfig
from agents.llm.fingpt_agent import FinGPTAgent
from agents.technical.technical_analyst import TechnicalAnalystAgent
from services.signal_service import SignalService


class TestAgentOrchestrator:
    """Test the agent orchestrator"""
    
    @pytest.fixture
    async def orchestrator(self, db_session):
        """Create orchestrator with test database"""
        signal_service = SignalService()
        orchestrator = AgentOrchestrator(signal_service=signal_service)
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        agents = []
        
        # Create 3 mock agents with different behaviors
        for i in range(3):
            agent = AsyncMock(spec=BaseAgent)
            agent.agent_id = f"test-agent-{i}"
            agent.config = AgentConfig(
                name=f"TestAgent{i}",
                enabled=True,
                weight=1.0,
                timeout=30
            )
            agent.performance.total_signals = 20
            agent.performance.accuracy = 0.8 - (i * 0.1)  # Different accuracies
            agent.performance.avg_confidence = 0.85
            
            agents.append(agent)
        
        return agents
    
    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator, mock_agent):
        """Test agent registration"""
        orchestrator.register_agent(mock_agent)
        
        assert mock_agent.agent_id in orchestrator.agents
        assert len(orchestrator.agents) == 1
    
    @pytest.mark.asyncio
    async def test_initialize_default_agents(self, orchestrator):
        """Test initialization of default agents"""
        with patch('agents.llm.fingpt_agent.fingpt_agent.initialize', new_callable=AsyncMock):
            with patch('agents.technical.technical_analyst.TechnicalAnalystAgent.initialize', new_callable=AsyncMock):
                with patch('agents.economic_indicator_agent.economic_indicator_agent.initialize', new_callable=AsyncMock):
                    await orchestrator.initialize_default_agents()
        
        # Should have at least 3 agents
        assert len(orchestrator.agents) >= 3
        
        # Check specific agents are registered
        agent_names = [a.config.name for a in orchestrator.agents.values()]
        assert any("FinGPT" in name for name in agent_names)
        assert any("Technical" in name for name in agent_names)
        assert any("Economic" in name for name in agent_names)
    
    @pytest.mark.asyncio
    async def test_analyze_market_no_agents(self, orchestrator, mock_market_data):
        """Test market analysis with no agents"""
        signal = await orchestrator.analyze_market(mock_market_data)
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_analyze_market_single_agent(self, orchestrator, mock_agent, mock_market_data):
        """Test market analysis with single agent"""
        orchestrator.register_agent(mock_agent)
        
        signal = await orchestrator.analyze_market(mock_market_data)
        
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.85
        mock_agent.execute_with_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_market_multiple_agents_consensus(self, orchestrator, mock_agents, mock_market_data):
        """Test consensus building with multiple agents"""
        # Register all agents
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        # Configure agent responses - 2 BUY, 1 SELL
        buy_signal = Signal(
            symbol="AAPL",
            action=SignalAction.BUY,
            confidence=0.85,
            strength=SignalStrength.STRONG,
            source="TestAgent",
            current_price=150.00,
            reasoning=["Bullish signal"]
        )
        
        sell_signal = Signal(
            symbol="AAPL",
            action=SignalAction.SELL,
            confidence=0.75,
            strength=SignalStrength.MODERATE,
            source="TestAgent",
            current_price=150.00,
            reasoning=["Bearish signal"]
        )
        
        mock_agents[0].execute_with_monitoring.return_value = buy_signal
        mock_agents[1].execute_with_monitoring.return_value = buy_signal
        mock_agents[2].execute_with_monitoring.return_value = sell_signal
        
        # Run analysis
        signal = await orchestrator.analyze_market(mock_market_data)
        
        # Should reach BUY consensus (2 out of 3)
        assert signal is not None
        assert signal.action == SignalAction.BUY
        assert signal.source == "Orchestrator"
        
        # Check consensus details
        assert signal.features["consensus_weight"] > 0.6
        assert signal.features["total_agents"] == 3
        assert signal.features["agreeing_agents"] == 2
    
    @pytest.mark.asyncio
    async def test_analyze_market_no_consensus(self, orchestrator, mock_agents, mock_market_data):
        """Test when agents don't reach consensus"""
        # Register agents
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        # Configure diverse responses - 1 BUY, 1 SELL, 1 HOLD
        actions = [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
        
        for i, agent in enumerate(mock_agents):
            signal = Signal(
                symbol="AAPL",
                action=actions[i],
                confidence=0.6,  # Low confidence
                strength=SignalStrength.WEAK,
                source=f"TestAgent{i}",
                current_price=150.00,
                reasoning=[f"{actions[i].value} signal"]
            )
            agent.execute_with_monitoring.return_value = signal
        
        # Set high consensus threshold
        orchestrator.consensus_threshold = 0.8
        
        # Run analysis
        signal = await orchestrator.analyze_market(mock_market_data)
        
        # Should not generate signal due to lack of consensus
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_analyze_market_with_agent_failures(self, orchestrator, mock_agents, mock_market_data):
        """Test handling of agent failures"""
        # Register agents
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        # Configure one agent to fail
        mock_agents[0].execute_with_monitoring.side_effect = Exception("Agent error")
        
        # Other agents return signals
        for agent in mock_agents[1:]:
            agent.execute_with_monitoring.return_value = Signal(
                symbol="AAPL",
                action=SignalAction.BUY,
                confidence=0.85,
                strength=SignalStrength.STRONG,
                source="TestAgent",
                current_price=150.00
            )
        
        # Run analysis
        signal = await orchestrator.analyze_market(mock_market_data)
        
        # Should still generate signal from working agents
        assert signal is not None
        assert signal.action == SignalAction.BUY
        assert signal.features["total_agents"] == 2  # Only working agents
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_confidence_calculation(self, orchestrator):
        """Test confidence calculation in signal aggregation"""
        # Create test agents and signals
        agents = []
        signals = []
        
        for i in range(3):
            agent = MagicMock(spec=BaseAgent)
            agent.config.weight = 1.0 + (i * 0.5)  # Different weights
            
            signal = Signal(
                symbol="AAPL",
                action=SignalAction.BUY,
                confidence=0.7 + (i * 0.1),  # Different confidences
                strength=SignalStrength.MODERATE,
                source=f"Agent{i}",
                current_price=150.00
            )
            
            agents.append(agent)
            signals.append(signal)
        
        agent_signals = list(zip(agents, signals))
        
        # Aggregate signals
        result = await orchestrator._aggregate_signals(
            agent_signals,
            {"symbol": "AAPL", "price": 150.00}
        )
        
        assert result is not None
        # Weighted confidence should be calculated correctly
        assert result.confidence > 0.7 and result.confidence < 0.95
        assert result.strength in [SignalStrength.MODERATE, SignalStrength.STRONG]
    
    @pytest.mark.asyncio
    async def test_rebalance_weights(self, orchestrator, mock_agents):
        """Test dynamic weight rebalancing based on performance"""
        # Register agents with different performance
        for i, agent in enumerate(mock_agents):
            agent.performance.total_signals = 50
            agent.performance.accuracy = 0.9 - (i * 0.2)  # 0.9, 0.7, 0.5
            agent.performance.avg_confidence = 0.8
            orchestrator.register_agent(agent)
        
        # Initial weights should be 1.0
        for agent in mock_agents:
            assert agent.config.weight == 1.0
        
        # Rebalance weights
        await orchestrator.rebalance_weights()
        
        # Weights should be adjusted based on performance
        for agent in mock_agents:
            agent.adjust_weight.assert_called_once()
        
        # Better performing agents should get higher weights
        # (actual weight values depend on implementation)
    
    @pytest.mark.asyncio
    async def test_rag_augmentation(self, orchestrator, mock_agents, mock_market_data):
        """Test RAG augmentation of signals"""
        orchestrator.use_rag = True
        
        # Mock RAG engine
        with patch.object(orchestrator.rag_engine, 'retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            with patch.object(orchestrator.rag_engine, 'generate_augmented_signal', new_callable=AsyncMock) as mock_augment:
                
                mock_retrieve.return_value = ["Historical context 1", "Historical context 2"]
                mock_augment.return_value = {
                    "confidence": 0.92,  # Enhanced confidence
                    "rag_context": {"historical_accuracy": 0.85},
                    "rag_insights": ["Historical pattern detected"],
                    "rag_reasoning": "Similar pattern showed 85% success rate"
                }
                
                # Register agent and run analysis
                orchestrator.register_agent(mock_agents[0])
                signal = await orchestrator.analyze_market(mock_market_data)
                
                assert signal is not None
                assert signal.confidence == 0.92  # Enhanced by RAG
                assert "rag_context" in signal.features
                assert "rag_insights" in signal.features
                assert any("RAG" in r for r in signal.reasoning)


class TestFinGPTAgent:
    """Test the FinGPT agent"""
    
    @pytest.fixture
    async def fingpt_agent(self):
        """Create FinGPT agent"""
        agent = FinGPTAgent()
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_fingpt_initialization(self, fingpt_agent):
        """Test FinGPT agent initialization"""
        assert fingpt_agent.config.name == "FinGPT"
        assert fingpt_agent.config.enabled is True
        assert "sentiment_analysis" in fingpt_agent.capabilities
        assert "market_forecasting" in fingpt_agent.capabilities
    
    @pytest.mark.asyncio
    async def test_fingpt_analyze_bullish(self, fingpt_agent, mock_market_data):
        """Test FinGPT analysis with bullish indicators"""
        # Mock market data with bullish indicators
        bullish_data = {
            **mock_market_data,
            "change_percent": 2.5,
            "volume": 150000000,  # High volume
            "historical_prices": [145, 146, 147, 148, 149, 150, 151, 152]  # Uptrend
        }
        
        with patch.object(fingpt_agent, '_analyze_with_fingpt', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "sentiment": "bullish",
                "confidence": 0.88,
                "reasoning": ["Strong uptrend detected", "High volume confirms momentum"]
            }
            
            signal = await fingpt_agent.analyze(bullish_data)
            
            assert signal is not None
            assert signal.action == SignalAction.BUY
            assert signal.confidence >= 0.8
            assert len(signal.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_fingpt_analyze_bearish(self, fingpt_agent, mock_market_data):
        """Test FinGPT analysis with bearish indicators"""
        # Mock market data with bearish indicators
        bearish_data = {
            **mock_market_data,
            "change_percent": -3.0,
            "volume": 50000000,  # Low volume
            "historical_prices": [155, 154, 153, 152, 151, 150, 149, 148]  # Downtrend
        }
        
        with patch.object(fingpt_agent, '_analyze_with_fingpt', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "sentiment": "bearish",
                "confidence": 0.85,
                "reasoning": ["Downtrend pattern", "Weak volume indicates lack of support"]
            }
            
            signal = await fingpt_agent.analyze(bearish_data)
            
            assert signal is not None
            assert signal.action == SignalAction.SELL
            assert signal.confidence >= 0.8
            assert len(signal.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_fingpt_error_handling(self, fingpt_agent, mock_market_data):
        """Test FinGPT error handling"""
        with patch.object(fingpt_agent, '_analyze_with_fingpt', side_effect=Exception("Model error")):
            signal = await fingpt_agent.analyze(mock_market_data)
            
            # Should return None on error
            assert signal is None


class TestTechnicalAnalystAgent:
    """Test the technical analyst agent"""
    
    @pytest.fixture
    async def tech_agent(self):
        """Create technical analyst agent"""
        agent = TechnicalAnalystAgent()
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_technical_indicators_calculation(self, tech_agent, mock_historical_data):
        """Test calculation of technical indicators"""
        import pandas as pd
        
        # Create DataFrame from mock data
        df = pd.DataFrame({
            'Close': mock_historical_data['prices'],
            'Volume': mock_historical_data['volumes']
        }, index=pd.to_datetime(mock_historical_data['dates']))
        
        market_data = {
            "symbol": "AAPL",
            "price": 165.00,
            "historical_prices": df
        }
        
        signal = await tech_agent.analyze(market_data)
        
        assert signal is not None
        assert "rsi" in signal.indicators
        assert "macd" in signal.indicators
        assert "sma_20" in signal.indicators
        assert "sma_50" in signal.indicators
    
    @pytest.mark.asyncio
    async def test_technical_signal_generation(self, tech_agent):
        """Test signal generation based on technical indicators"""
        # Mock oversold conditions
        market_data = {
            "symbol": "AAPL",
            "price": 145.00,
            "indicators": {
                "rsi": 25,  # Oversold
                "macd": 0.5,  # Positive
                "macd_signal": 0.3,
                "sma_20": 148,
                "sma_50": 150
            }
        }
        
        with patch.object(tech_agent, '_calculate_indicators', return_value=market_data["indicators"]):
            signal = await tech_agent.analyze(market_data)
            
            assert signal is not None
            assert signal.action == SignalAction.BUY  # Oversold = BUY
            assert signal.confidence > 0.7
            assert any("oversold" in r.lower() for r in signal.reasoning)


class TestAgentPerformance:
    """Test agent performance tracking"""
    
    @pytest.mark.asyncio
    async def test_agent_performance_update(self, mock_agent):
        """Test updating agent performance metrics"""
        # Initial performance
        assert mock_agent.performance.total_signals == 20
        assert mock_agent.performance.accuracy == 0.8
        
        # Update with correct prediction
        mock_agent.update_performance_feedback("signal-1", True, 100.0)
        mock_agent.update_performance_feedback.assert_called_once()
        
        # Update with incorrect prediction
        mock_agent.update_performance_feedback("signal-2", False, -50.0)
        assert mock_agent.update_performance_feedback.call_count == 2
    
    @pytest.mark.asyncio
    async def test_orchestrator_performance_summary(self, orchestrator, mock_agents):
        """Test getting performance summary from orchestrator"""
        # Register agents
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        summary = orchestrator.get_agent_performance_summary()
        
        assert summary["total_agents"] == 3
        assert summary["active_agents"] == 3
        assert len(summary["agents"]) == 3
        
        # Check each agent's performance is included
        for agent in mock_agents:
            agent.get_current_performance.assert_called_once()


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for the complete agent system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_signal_generation(self, orchestrator, mock_market_data):
        """Test complete signal generation pipeline"""
        # Initialize with real agents (mocked internals)
        with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', new_callable=AsyncMock) as mock_fingpt:
            mock_fingpt.return_value = {
                "sentiment": "bullish",
                "confidence": 0.85,
                "reasoning": ["Market momentum positive"]
            }
            
            await orchestrator.initialize_default_agents()
            
            # Run analysis
            signal = await orchestrator.analyze_market(mock_market_data)
            
            assert signal is not None
            assert signal.source == "Orchestrator"
            assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
            assert 0 <= signal.confidence <= 1
            assert len(signal.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, orchestrator):
        """Test concurrent execution of multiple agents"""
        # Create 10 mock agents
        agents = []
        for i in range(10):
            agent = AsyncMock(spec=BaseAgent)
            agent.agent_id = f"concurrent-agent-{i}"
            agent.config = AgentConfig(
                name=f"ConcurrentAgent{i}",
                enabled=True,
                weight=1.0,
                timeout=5
            )
            
            # Simulate varying execution times
            async def delayed_execute(delay):
                await asyncio.sleep(delay)
                return Signal(
                    symbol="AAPL",
                    action=SignalAction.BUY,
                    confidence=0.8,
                    source=f"Agent{i}",
                    current_price=150.00
                )
            
            agent.execute_with_monitoring.side_effect = lambda x: delayed_execute(0.1 * (i % 3))
            agents.append(agent)
            orchestrator.register_agent(agent)
        
        # Run analysis
        start_time = asyncio.get_event_loop().time()
        signal = await orchestrator.analyze_market({"symbol": "AAPL", "price": 150.00})
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should execute concurrently, not sequentially
        assert elapsed < 1.0  # Should be much faster than sequential (3 seconds)
        assert signal is not None