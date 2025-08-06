"""
Signal Generation Integration Tests
End-to-end testing of AI signal generation pipeline
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.volatility_agent import VolatilityAgent
from services.enhanced_data_aggregator import EnhancedDataAggregator
from services.websocket_manager import ws_manager
from core.orchestrator import SignalOrchestrator


@pytest.mark.asyncio
class TestSignalGenerationPipeline:
    """Integration tests for the complete signal generation pipeline"""
    
    async def test_end_to_end_signal_generation(self):
        """Test complete signal generation from data ingestion to WebSocket broadcast"""
        # Mock data aggregator
        with patch.object(EnhancedDataAggregator, 'get_market_data') as mock_market_data, \
             patch.object(EnhancedDataAggregator, 'get_sentiment_data') as mock_sentiment, \
             patch.object(EnhancedDataAggregator, 'get_volatility_metrics') as mock_volatility:
            
            # Mock market data
            mock_market_data.return_value = {
                "symbol": "AAPL",
                "price": 150.0,
                "volume": 1000000,
                "change_percent": 2.5,
                "technical_indicators": {
                    "rsi": 65.0,
                    "macd": 0.5,
                    "bollinger_upper": 155.0,
                    "bollinger_lower": 145.0
                },
                "timestamp": datetime.now()
            }
            
            # Mock sentiment data
            mock_sentiment.return_value = {
                "overall_sentiment": 0.7,
                "news_sentiment": 0.6,
                "social_sentiment": 0.8,
                "analyst_sentiment": 0.75,
                "confidence": 0.85
            }
            
            # Mock volatility data
            mock_volatility.return_value = {
                "implied_volatility": 0.25,
                "historical_volatility": 0.22,
                "volatility_rank": 0.6,
                "volatility_trend": "increasing"
            }
            
            # Mock AI agents
            with patch.object(TechnicalAnalysisAgent, 'analyze') as mock_tech, \
                 patch.object(SentimentAnalysisAgent, 'analyze') as mock_sent, \
                 patch.object(VolatilityAgent, 'analyze') as mock_vol:
                
                # Mock agent responses
                mock_tech.return_value = {
                    "signal": "BUY",
                    "confidence": 0.8,
                    "reasoning": "Bullish technical pattern",
                    "indicators": {"rsi": "oversold_recovery", "macd": "bullish_crossover"}
                }
                
                mock_sent.return_value = {
                    "signal": "BUY", 
                    "confidence": 0.7,
                    "reasoning": "Positive sentiment across all sources",
                    "sentiment_score": 0.7
                }
                
                mock_vol.return_value = {
                    "signal": "HOLD",
                    "confidence": 0.6,
                    "reasoning": "Moderate volatility increase",
                    "volatility_assessment": "moderate_risk"
                }
                
                # Mock WebSocket manager
                with patch.object(ws_manager, 'broadcast_signal') as mock_broadcast:
                    # Create orchestrator and generate signal
                    orchestrator = SignalOrchestrator()
                    result = await orchestrator.generate_signal("AAPL", "1d")
                    
                    # Verify signal generation
                    assert result["symbol"] == "AAPL"
                    assert result["consensus_signal"] == "BUY"  # Majority BUY
                    assert result["consensus_confidence"] > 0.0
                    assert "agent_signals" in result
                    
                    # Verify agents were called
                    mock_tech.assert_called_once()
                    mock_sent.assert_called_once() 
                    mock_vol.assert_called_once()
                    
                    # Verify WebSocket broadcast
                    mock_broadcast.assert_called_once()
    
    async def test_signal_generation_with_real_agents(self):
        """Test signal generation with actual agent instances (mocked data)"""
        with patch('agents.base.BaseAgent.get_market_data') as mock_market_data:
            # Mock market data for all agents
            mock_market_data.return_value = {
                "symbol": "TSLA",
                "current_price": 800.0,
                "volume": 2000000,
                "price_history": [795.0, 798.0, 802.0, 800.0],
                "volume_history": [1800000, 1900000, 2100000, 2000000],
                "technical_indicators": {
                    "rsi": 58.0,
                    "macd": 1.2,
                    "sma_20": 795.0,
                    "sma_50": 785.0
                }
            }
            
            # Create real agent instances
            tech_agent = TechnicalAnalysisAgent("technical_agent")
            sentiment_agent = SentimentAnalysisAgent("sentiment_agent")
            volatility_agent = VolatilityAgent("volatility_agent")
            
            # Generate signals from each agent
            tech_result = await tech_agent.analyze("TSLA", "1d")
            sentiment_result = await sentiment_agent.analyze("TSLA", "1d")
            volatility_result = await volatility_agent.analyze("TSLA", "1d")
            
            # Verify all agents returned valid signals
            for result in [tech_result, sentiment_result, volatility_result]:
                assert result["symbol"] == "TSLA"
                assert result["signal"] in ["BUY", "SELL", "HOLD"]
                assert 0.0 <= result["confidence"] <= 1.0
                assert "reasoning" in result
                assert isinstance(result["timestamp"], datetime)
    
    async def test_consensus_signal_calculation(self):
        """Test consensus signal calculation with different agent outputs"""
        # Test case 1: All agents agree
        agent_signals = {
            "technical": {"signal": "BUY", "confidence": 0.8},
            "sentiment": {"signal": "BUY", "confidence": 0.7},
            "volatility": {"signal": "BUY", "confidence": 0.9}
        }
        
        orchestrator = SignalOrchestrator()
        consensus = orchestrator._calculate_consensus(agent_signals)
        
        assert consensus["signal"] == "BUY"
        assert consensus["confidence"] > 0.7  # High confidence when all agree
        
        # Test case 2: Mixed signals
        agent_signals = {
            "technical": {"signal": "BUY", "confidence": 0.8},
            "sentiment": {"signal": "SELL", "confidence": 0.6},
            "volatility": {"signal": "HOLD", "confidence": 0.7}
        }
        
        consensus = orchestrator._calculate_consensus(agent_signals)
        
        # Should be more conservative with mixed signals
        assert consensus["signal"] in ["BUY", "SELL", "HOLD"]
        assert consensus["confidence"] < 0.8  # Lower confidence with disagreement
        
        # Test case 3: Majority rule
        agent_signals = {
            "technical": {"signal": "SELL", "confidence": 0.9},
            "sentiment": {"signal": "SELL", "confidence": 0.8},
            "volatility": {"signal": "BUY", "confidence": 0.6}
        }
        
        consensus = orchestrator._calculate_consensus(agent_signals)
        
        assert consensus["signal"] == "SELL"  # Majority SELL
    
    async def test_signal_persistence_and_history(self):
        """Test that generated signals are properly persisted and retrievable"""
        with patch('core.database.get_session') as mock_session, \
             patch('models.signal.Signal') as mock_signal_model:
            
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            # Mock signal creation
            mock_signal = Mock()
            mock_signal.id = 1
            mock_signal_model.return_value = mock_signal
            
            mock_session_instance.add = Mock()
            mock_session_instance.commit = AsyncMock()
            mock_session_instance.refresh = AsyncMock()
            
            # Generate and persist signal
            orchestrator = SignalOrchestrator()
            
            with patch.object(orchestrator, '_generate_agent_signals') as mock_generate:
                mock_generate.return_value = {
                    "technical": {"signal": "BUY", "confidence": 0.8}
                }
                
                result = await orchestrator.generate_signal("AAPL", "1d", persist=True)
                
                # Verify signal was persisted
                mock_session_instance.add.assert_called_once()
                mock_session_instance.commit.assert_called_once()
                
                assert "signal_id" in result
    
    async def test_signal_generation_error_handling(self):
        """Test error handling in signal generation pipeline"""
        orchestrator = SignalOrchestrator()
        
        # Test with failing agent
        with patch.object(TechnicalAnalysisAgent, 'analyze') as mock_tech:
            mock_tech.side_effect = Exception("Agent failed")
            
            # Should handle gracefully and continue with other agents
            with patch.object(SentimentAnalysisAgent, 'analyze') as mock_sent:
                mock_sent.return_value = {
                    "signal": "HOLD", 
                    "confidence": 0.5,
                    "reasoning": "Backup signal"
                }
                
                result = await orchestrator.generate_signal("AAPL", "1d")
                
                # Should still generate signal from working agents
                assert result["symbol"] == "AAPL"
                assert "agent_signals" in result
                assert "technical" not in result["agent_signals"]  # Failed agent excluded
                assert "sentiment" in result["agent_signals"]  # Working agent included
    
    async def test_concurrent_signal_generation(self):
        """Test handling of concurrent signal generation requests"""
        with patch.object(EnhancedDataAggregator, 'get_market_data') as mock_data:
            mock_data.return_value = {"symbol": "TEST", "price": 100.0}
            
            orchestrator = SignalOrchestrator()
            
            # Generate signals concurrently for different symbols
            symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
            
            with patch.object(orchestrator, '_generate_agent_signals') as mock_generate:
                mock_generate.return_value = {
                    "technical": {"signal": "BUY", "confidence": 0.7}
                }
                
                # Create concurrent tasks
                tasks = [
                    orchestrator.generate_signal(symbol, "1d") 
                    for symbol in symbols
                ]
                
                results = await asyncio.gather(*tasks)
                
                # All should complete successfully
                assert len(results) == len(symbols)
                for i, result in enumerate(results):
                    assert result["symbol"] == symbols[i]
    
    async def test_signal_generation_performance(self):
        """Test signal generation performance under load"""
        import time
        
        orchestrator = SignalOrchestrator()
        
        with patch.object(orchestrator, '_generate_agent_signals') as mock_generate:
            mock_generate.return_value = {
                "technical": {"signal": "BUY", "confidence": 0.8},
                "sentiment": {"signal": "BUY", "confidence": 0.7}
            }
            
            start_time = time.time()
            
            # Generate multiple signals
            tasks = [
                orchestrator.generate_signal("AAPL", "1d")
                for _ in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert total_time < 5.0  # 5 seconds for 10 signals
            assert len(results) == 10
            
            # Calculate average time per signal
            avg_time = total_time / len(results)
            assert avg_time < 1.0  # Less than 1 second per signal


@pytest.mark.integration
class TestSignalGenerationWithExternalData:
    """Integration tests with external data sources"""
    
    @pytest.mark.asyncio
    async def test_signal_generation_with_real_market_data(self):
        """Test signal generation with real market data sources"""
        # This test would use real market data APIs
        pytest.skip("Requires real market data APIs - run manually")
    
    @pytest.mark.asyncio
    async def test_signal_generation_with_real_news_data(self):
        """Test signal generation with real news and sentiment data"""
        # This test would use real news APIs
        pytest.skip("Requires real news APIs - run manually")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])