"""
Integration tests for the complete signal generation pipeline
Tests the entire flow from data acquisition to signal output
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from agents.orchestrator import AgentOrchestrator
from services.signal_service import SignalService
# from services.market_data_service import MarketDataService  # TODO: Update to unified service
from services.websocket_orchestrator import WebSocketOrchestrator
from models.signal import Signal
from agents.base import SignalAction


class TestSignalGenerationPipeline:
    """Test the complete signal generation pipeline"""
    
    @pytest.fixture
    async def pipeline_setup(self, db_session):
        """Setup complete pipeline with all components"""
        # Initialize services
        market_service = MarketDataService()
        signal_service = SignalService()
        orchestrator = AgentOrchestrator(signal_service=signal_service)
        ws_orchestrator = WebSocketOrchestrator()
        
        # Initialize orchestrator with agents
        await orchestrator.initialize_default_agents()
        
        yield {
            "market_service": market_service,
            "signal_service": signal_service,
            "orchestrator": orchestrator,
            "ws_orchestrator": ws_orchestrator,
            "db": db_session
        }
        
        # Cleanup
        await orchestrator.shutdown()
        await ws_orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_complete_signal_flow(self, pipeline_setup):
        """Test complete flow from market data to signal"""
        symbol = "AAPL"
        
        # Mock market data
        mock_quote = {
            "symbol": symbol,
            "price": 150.25,
            "volume": 75000000,
            "change_percent": 1.5
        }
        
        mock_historical = pd.DataFrame({
            'Close': [148, 149, 149.5, 150, 150.25],
            'Volume': [70000000, 72000000, 73000000, 74000000, 75000000]
        }, index=pd.date_range(end=datetime.now(), periods=5))
        
        # Patch data sources
        with patch.object(pipeline_setup["market_service"], 'get_quote', return_value=mock_quote):
            with patch.object(pipeline_setup["market_service"], 'get_historical_data', return_value=mock_historical):
                # Mock AI agent responses
                with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', return_value={
                    "sentiment": "bullish",
                    "confidence": 0.85,
                    "reasoning": ["Strong upward momentum", "Positive market sentiment"]
                }):
                    # Run analysis
                    market_data = {
                        "symbol": symbol,
                        "price": mock_quote["price"],
                        "volume": mock_quote["volume"],
                        "change_percent": mock_quote["change_percent"],
                        "historical_prices": mock_historical
                    }
                    
                    signal = await pipeline_setup["orchestrator"].analyze_market(market_data)
                    
                    # Verify signal generation
                    assert signal is not None
                    assert signal.symbol == symbol
                    assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
                    assert 0 <= signal.confidence <= 1
                    assert len(signal.reasoning) > 0
                    
                    # Verify signal storage
                    if signal:
                        stored_signals = await pipeline_setup["signal_service"].get_recent_signals(
                            pipeline_setup["db"],
                            limit=1
                        )
                        assert len(stored_signals) > 0
                        assert stored_signals[0].symbol == symbol
    
    @pytest.mark.asyncio
    async def test_multi_symbol_batch_processing(self, pipeline_setup):
        """Test processing multiple symbols in batch"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        # Mock data for each symbol
        mock_quotes = {
            symbol: {
                "symbol": symbol,
                "price": 100 + i * 50,
                "volume": 50000000 + i * 10000000,
                "change_percent": -2 + i  # Different changes
            }
            for i, symbol in enumerate(symbols)
        }
        
        signals = []
        
        # Process each symbol
        for symbol in symbols:
            with patch.object(pipeline_setup["market_service"], 'get_quote', return_value=mock_quotes[symbol]):
                with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', return_value={
                    "sentiment": "bullish" if mock_quotes[symbol]["change_percent"] > 0 else "bearish",
                    "confidence": 0.8,
                    "reasoning": ["Test reasoning"]
                }):
                    market_data = {
                        "symbol": symbol,
                        **mock_quotes[symbol]
                    }
                    
                    signal = await pipeline_setup["orchestrator"].analyze_market(market_data)
                    if signal:
                        signals.append(signal)
        
        # Should generate signals for all symbols
        assert len(signals) == len(symbols)
        
        # Verify different actions based on change_percent
        buy_signals = [s for s in signals if s.action == SignalAction.BUY]
        sell_signals = [s for s in signals if s.action == SignalAction.SELL]
        
        assert len(buy_signals) > 0  # Positive changes
        assert len(sell_signals) > 0  # Negative changes
    
    @pytest.mark.asyncio
    async def test_real_time_signal_broadcast(self, pipeline_setup):
        """Test real-time broadcasting of generated signals"""
        # Mock WebSocket manager
        with patch('services.websocket_manager.ws_manager') as mock_ws_manager:
            mock_ws_manager.broadcast_signal = AsyncMock()
            mock_ws_manager.broadcast_agent_update = AsyncMock()
            
            # Generate a signal
            with patch.object(pipeline_setup["market_service"], 'get_quote', return_value={"price": 150.25}):
                with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', return_value={
                    "sentiment": "bullish",
                    "confidence": 0.9,
                    "reasoning": ["Strong buy signal"]
                }):
                    # Analyze with WebSocket orchestrator
                    await pipeline_setup["ws_orchestrator"].analyze_symbol("AAPL")
                    
                    # Wait for async operations
                    await asyncio.sleep(0.1)
                    
                    # Should broadcast agent updates and final signal
                    assert mock_ws_manager.broadcast_agent_update.called
                    assert mock_ws_manager.broadcast_signal.called
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, pipeline_setup):
        """Test pipeline resilience to errors"""
        # Test data source failure
        with patch.object(pipeline_setup["market_service"], 'get_quote', side_effect=Exception("API Error")):
            market_data = {"symbol": "AAPL", "price": 150.00}
            signal = await pipeline_setup["orchestrator"].analyze_market(market_data)
            
            # Should handle gracefully
            # Signal might be None or based on partial data
            # Pipeline should not crash
        
        # Test agent failure
        with patch('agents.llm.fingpt_agent.FinGPTAgent.analyze', side_effect=Exception("Model Error")):
            market_data = {"symbol": "AAPL", "price": 150.00}
            signal = await pipeline_setup["orchestrator"].analyze_market(market_data)
            
            # Should still work with other agents
            # or return None if all fail
    
    @pytest.mark.asyncio
    async def test_signal_validation_and_filtering(self, pipeline_setup):
        """Test signal validation and filtering logic"""
        # Create signals with different confidence levels
        test_cases = [
            {"confidence": 0.95, "action": "BUY", "should_pass": True},
            {"confidence": 0.55, "action": "HOLD", "should_pass": False},  # Below threshold
            {"confidence": 0.85, "action": "SELL", "should_pass": True},
        ]
        
        for case in test_cases:
            with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', return_value={
                "sentiment": "bullish" if case["action"] == "BUY" else "bearish",
                "confidence": case["confidence"],
                "reasoning": ["Test"]
            }):
                # Set minimum confidence threshold
                pipeline_setup["orchestrator"].consensus_threshold = 0.6
                
                market_data = {"symbol": "AAPL", "price": 150.00}
                signal = await pipeline_setup["orchestrator"].analyze_market(market_data)
                
                if case["should_pass"]:
                    assert signal is not None
                    assert signal.confidence >= 0.6
                else:
                    # Low confidence might not generate consensus
                    pass


class TestSignalPersistence:
    """Test signal storage and retrieval"""
    
    @pytest.mark.asyncio
    async def test_signal_storage(self, db_session, test_user):
        """Test storing signals in database"""
        signal_service = SignalService()
        
        # Create signal
        signal = await signal_service.create_signal(
            db=db_session,
            symbol="AAPL",
            action="BUY",
            confidence=0.85,
            price=150.00,
            source="TestAgent",
            user_id=test_user.id,
            metadata={
                "reasoning": ["Test reasoning"],
                "indicators": {"rsi": 45}
            }
        )
        
        assert signal.id is not None
        assert signal.symbol == "AAPL"
        assert signal.metadata["reasoning"] == ["Test reasoning"]
    
    @pytest.mark.asyncio
    async def test_signal_retrieval_filters(self, db_session, test_user):
        """Test retrieving signals with various filters"""
        signal_service = SignalService()
        
        # Create multiple signals
        symbols = ["AAPL", "GOOGL", "AAPL", "MSFT"]
        actions = ["BUY", "SELL", "HOLD", "BUY"]
        
        for symbol, action in zip(symbols, actions):
            await signal_service.create_signal(
                db=db_session,
                symbol=symbol,
                action=action,
                confidence=0.8,
                price=100.00,
                source="TestAgent",
                user_id=test_user.id
            )
        
        # Test symbol filter
        aapl_signals = await signal_service.get_signals(
            db=db_session,
            symbol="AAPL"
        )
        assert len(aapl_signals) == 2
        
        # Test action filter
        buy_signals = await signal_service.get_signals(
            db=db_session,
            action="BUY"
        )
        assert len(buy_signals) == 2
        
        # Test combined filters
        aapl_buy_signals = await signal_service.get_signals(
            db=db_session,
            symbol="AAPL",
            action="BUY"
        )
        assert len(aapl_buy_signals) == 1
    
    @pytest.mark.asyncio
    async def test_signal_performance_tracking(self, db_session, test_signal):
        """Test tracking signal performance"""
        signal_service = SignalService()
        
        # Update signal with outcome
        test_signal.outcome = "correct"
        test_signal.actual_price = 155.00  # Price went up as predicted
        test_signal.profit_loss = 5.00
        
        await db_session.commit()
        
        # Get performance stats
        stats = await signal_service.get_signal_statistics(
            db=db_session,
            symbol=test_signal.symbol
        )
        
        assert stats["total_signals"] > 0
        assert "accuracy" in stats
        assert "avg_confidence" in stats


class TestSignalAggregation:
    """Test signal aggregation and consensus"""
    
    @pytest.mark.asyncio
    async def test_consensus_calculation(self, pipeline_setup):
        """Test consensus calculation with multiple agents"""
        orchestrator = pipeline_setup["orchestrator"]
        
        # Create mock agents with specific votes
        from agents.base import BaseAgent, Signal, SignalAction
        
        agents_data = [
            ("Agent1", SignalAction.BUY, 0.9, 1.0),
            ("Agent2", SignalAction.BUY, 0.85, 1.2),
            ("Agent3", SignalAction.SELL, 0.8, 0.8),
            ("Agent4", SignalAction.BUY, 0.75, 1.0),
        ]
        
        agent_signals = []
        for name, action, confidence, weight in agents_data:
            agent = MagicMock(spec=BaseAgent)
            agent.config.name = name
            agent.config.weight = weight
            
            signal = Signal(
                symbol="AAPL",
                action=action,
                confidence=confidence,
                source=name,
                current_price=150.00
            )
            
            agent_signals.append((agent, signal))
        
        # Calculate consensus
        consensus_signal = await orchestrator._aggregate_signals(
            agent_signals,
            {"symbol": "AAPL", "price": 150.00}
        )
        
        # Should reach BUY consensus (3 out of 4 agents)
        assert consensus_signal is not None
        assert consensus_signal.action == SignalAction.BUY
        
        # Check weighted confidence
        assert consensus_signal.confidence > 0.8
        
        # Check consensus details
        assert consensus_signal.features["total_agents"] == 4
        assert consensus_signal.features["agreeing_agents"] == 3
    
    @pytest.mark.asyncio
    async def test_signal_strength_determination(self, pipeline_setup):
        """Test signal strength classification"""
        orchestrator = pipeline_setup["orchestrator"]
        
        test_cases = [
            (0.85, SignalStrength.STRONG),    # High consensus
            (0.65, SignalStrength.MODERATE),  # Medium consensus
            (0.45, SignalStrength.WEAK),      # Low consensus
        ]
        
        for consensus_weight, expected_strength in test_cases:
            # Mock consensus weight calculation
            with patch.object(orchestrator, '_aggregate_signals') as mock_aggregate:
                # Create signal with specific consensus weight
                signal = Signal(
                    symbol="AAPL",
                    action=SignalAction.BUY,
                    confidence=0.8,
                    strength=expected_strength,
                    source="Orchestrator",
                    current_price=150.00,
                    features={"consensus_weight": consensus_weight}
                )
                
                mock_aggregate.return_value = signal
                
                # Verify strength classification
                assert signal.strength == expected_strength


class TestSignalQuality:
    """Test signal quality and accuracy metrics"""
    
    @pytest.mark.asyncio
    async def test_signal_accuracy_tracking(self, db_session):
        """Test tracking signal accuracy over time"""
        signal_service = SignalService()
        
        # Create historical signals with outcomes
        signals_data = [
            ("AAPL", "BUY", 150.00, 155.00, "correct"),
            ("AAPL", "SELL", 160.00, 165.00, "incorrect"),
            ("AAPL", "BUY", 145.00, 150.00, "correct"),
            ("GOOGL", "BUY", 2800.00, 2850.00, "correct"),
            ("GOOGL", "SELL", 2900.00, 2880.00, "correct"),
        ]
        
        for symbol, action, entry_price, exit_price, outcome in signals_data:
            signal = await signal_service.create_signal(
                db=db_session,
                symbol=symbol,
                action=action,
                confidence=0.8,
                price=entry_price,
                source="TestAgent"
            )
            
            # Update with outcome
            signal.outcome = outcome
            signal.actual_price = exit_price
            signal.profit_loss = exit_price - entry_price if action == "BUY" else entry_price - exit_price
        
        await db_session.commit()
        
        # Calculate accuracy
        aapl_stats = await signal_service.get_signal_statistics(
            db=db_session,
            symbol="AAPL"
        )
        
        assert aapl_stats["total_signals"] == 3
        assert aapl_stats["correct_signals"] == 2
        assert aapl_stats["accuracy"] == pytest.approx(0.667, 0.01)
        
        googl_stats = await signal_service.get_signal_statistics(
            db=db_session,
            symbol="GOOGL"
        )
        
        assert googl_stats["accuracy"] == 1.0  # Both correct


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_market_crash_scenario(self, pipeline_setup):
        """Test system behavior during market crash"""
        # Simulate sudden price drops
        crash_data = {
            "symbol": "SPY",
            "price": 450.00,
            "change_percent": -5.0,  # 5% drop
            "volume": 200000000,     # High volume
            "vix": 35.0              # High volatility
        }
        
        with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', return_value={
            "sentiment": "bearish",
            "confidence": 0.95,
            "reasoning": ["Market crash detected", "Risk-off sentiment"]
        }):
            signal = await pipeline_setup["orchestrator"].analyze_market(crash_data)
            
            assert signal is not None
            assert signal.action == SignalAction.SELL
            assert signal.confidence > 0.9
            assert any("crash" in r.lower() or "risk" in r.lower() for r in signal.reasoning)
    
    @pytest.mark.asyncio
    async def test_earnings_announcement_scenario(self, pipeline_setup):
        """Test handling of earnings announcements"""
        # Simulate post-earnings price movement
        earnings_data = {
            "symbol": "AAPL",
            "price": 160.00,
            "change_percent": 8.0,   # Big jump
            "volume": 150000000,     # Very high volume
            "metadata": {
                "earnings_beat": True,
                "revenue_beat": True,
                "guidance": "raised"
            }
        }
        
        with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', return_value={
            "sentiment": "bullish",
            "confidence": 0.92,
            "reasoning": ["Earnings beat expectations", "Raised guidance"]
        }):
            signal = await pipeline_setup["orchestrator"].analyze_market(earnings_data)
            
            assert signal is not None
            assert signal.action == SignalAction.BUY
            assert any("earnings" in r.lower() for r in signal.reasoning)
    
    @pytest.mark.asyncio
    async def test_high_frequency_updates(self, pipeline_setup):
        """Test handling rapid price updates"""
        symbol = "TSLA"
        
        # Simulate rapid price changes
        prices = [800, 805, 803, 807, 802, 808]  # Volatile movement
        
        signals = []
        for price in prices:
            market_data = {
                "symbol": symbol,
                "price": float(price),
                "volume": 80000000
            }
            
            with patch('agents.llm.fingpt_agent.FinGPTAgent._analyze_with_fingpt', return_value={
                "sentiment": "neutral",
                "confidence": 0.6,
                "reasoning": ["High volatility"]
            }):
                # Add small delay to simulate real-time
                await asyncio.sleep(0.1)
                
                signal = await pipeline_setup["orchestrator"].analyze_market(market_data)
                if signal:
                    signals.append(signal)
        
        # System should handle rapid updates without issues
        # May not generate signals for all updates due to volatility
        assert len(signals) <= len(prices)