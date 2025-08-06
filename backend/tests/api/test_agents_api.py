"""
Agents API Tests
Comprehensive testing of AI agents endpoints
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import status

from app import app


client = TestClient(app)


class TestAgentsAPI:
    """Test suite for AI agents API endpoints"""
    
    def test_get_agents_list_success(self):
        """Test successful agents list retrieval"""
        response = client.get("/api/v1/agents")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
    
    def test_get_agent_status_success(self):
        """Test successful agent status retrieval"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.get_status.return_value = {
                "name": "technical_analysis_agent",
                "status": "active",
                "last_analysis": datetime.now().isoformat(),
                "signals_generated": 150,
                "accuracy": 0.78
            }
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.get("/api/v1/agents/technical_analysis_agent/status")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["name"] == "technical_analysis_agent"
            assert data["status"] == "active"
    
    def test_get_agent_status_not_found(self):
        """Test agent status for non-existent agent"""
        response = client.get("/api/v1/agents/nonexistent_agent/status")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_agent_analyze_success(self):
        """Test successful agent analysis request"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.analyze = AsyncMock(return_value={
                "symbol": "AAPL",
                "signal": "BUY",
                "confidence": 0.85,
                "reasoning": "Strong bullish momentum with volume confirmation",
                "price_target": 160.0,
                "stop_loss": 145.0
            })
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.post("/api/v1/agents/technical_analysis_agent/analyze", 
                                 json={"symbol": "AAPL", "timeframe": "1d"})
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["symbol"] == "AAPL"
            assert data["signal"] == "BUY"
            assert data["confidence"] == 0.85
    
    def test_agent_analyze_invalid_params(self):
        """Test agent analysis with invalid parameters"""
        response = client.post("/api/v1/agents/technical_analysis_agent/analyze", 
                             json={})  # Missing required symbol
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_agent_analyze_agent_not_found(self):
        """Test agent analysis for non-existent agent"""
        response = client.post("/api/v1/agents/nonexistent_agent/analyze",
                             json={"symbol": "AAPL"})
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_agent_analyze_server_error(self):
        """Test agent analysis with server error"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.analyze = AsyncMock(side_effect=Exception("Analysis failed"))
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.post("/api/v1/agents/technical_analysis_agent/analyze",
                                 json={"symbol": "AAPL"})
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    def test_get_agent_config_success(self):
        """Test successful agent configuration retrieval"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.get_config.return_value = {
                "name": "technical_analysis_agent",
                "parameters": {
                    "rsi_period": 14,
                    "ma_short": 9,
                    "ma_long": 21,
                    "volume_threshold": 1.5
                },
                "enabled": True,
                "update_interval": 300
            }
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.get("/api/v1/agents/technical_analysis_agent/config")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["name"] == "technical_analysis_agent"
            assert "parameters" in data
    
    def test_update_agent_config_success(self):
        """Test successful agent configuration update"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.update_config = AsyncMock(return_value=True)
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            new_config = {
                "parameters": {
                    "rsi_period": 21,
                    "ma_short": 10,
                    "ma_long": 30
                },
                "enabled": False
            }
            
            response = client.put("/api/v1/agents/technical_analysis_agent/config",
                                json=new_config)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "updated"
    
    def test_agent_enable_success(self):
        """Test successful agent enable"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.enable = AsyncMock(return_value=True)
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.post("/api/v1/agents/technical_analysis_agent/enable")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "enabled"
    
    def test_agent_disable_success(self):
        """Test successful agent disable"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.disable = AsyncMock(return_value=True)
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.post("/api/v1/agents/technical_analysis_agent/disable")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "disabled"
    
    def test_get_agent_metrics_success(self):
        """Test successful agent metrics retrieval"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.get_metrics.return_value = {
                "total_analyses": 1000,
                "successful_analyses": 950,
                "average_confidence": 0.72,
                "signals_generated": {
                    "BUY": 450,
                    "SELL": 300,
                    "HOLD": 250
                },
                "performance": {
                    "accuracy": 0.78,
                    "precision": 0.82,
                    "recall": 0.75
                }
            }
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.get("/api/v1/agents/technical_analysis_agent/metrics")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_analyses"] == 1000
            assert "performance" in data
    
    def test_get_agent_history_success(self):
        """Test successful agent analysis history retrieval"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.get_analysis_history = AsyncMock(return_value=[
                {
                    "id": "analysis_1",
                    "symbol": "AAPL",
                    "timestamp": datetime.now().isoformat(),
                    "signal": "BUY",
                    "confidence": 0.85,
                    "price": 150.0
                },
                {
                    "id": "analysis_2",
                    "symbol": "TSLA",
                    "timestamp": datetime.now().isoformat(),
                    "signal": "SELL",
                    "confidence": 0.72,
                    "price": 800.0
                }
            ])
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.get("/api/v1/agents/technical_analysis_agent/history?limit=10")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2
            assert data[0]["symbol"] == "AAPL"
    
    def test_agent_bulk_analyze_success(self):
        """Test successful bulk analysis"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.bulk_analyze = AsyncMock(return_value={
                "AAPL": {"signal": "BUY", "confidence": 0.85},
                "TSLA": {"signal": "SELL", "confidence": 0.72},
                "GOOGL": {"signal": "HOLD", "confidence": 0.65}
            })
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            symbols = ["AAPL", "TSLA", "GOOGL"]
            response = client.post("/api/v1/agents/technical_analysis_agent/bulk-analyze",
                                 json={"symbols": symbols, "timeframe": "1d"})
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "AAPL" in data
            assert data["AAPL"]["signal"] == "BUY"
    
    def test_agent_bulk_analyze_too_many_symbols(self):
        """Test bulk analysis with too many symbols"""
        symbols = [f"SYMBOL{i}" for i in range(101)]  # 101 symbols
        response = client.post("/api/v1/agents/technical_analysis_agent/bulk-analyze",
                             json={"symbols": symbols})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_consensus_signals_success(self):
        """Test successful consensus signals retrieval"""
        with patch('api.routes.agents.generate_consensus_signals') as mock_consensus:
            mock_consensus.return_value = {
                "AAPL": {
                    "consensus_signal": "BUY",
                    "consensus_confidence": 0.78,
                    "agent_signals": {
                        "technical_analysis_agent": {"signal": "BUY", "confidence": 0.85},
                        "sentiment_analysis_agent": {"signal": "BUY", "confidence": 0.72},
                        "volatility_agent": {"signal": "HOLD", "confidence": 0.65}
                    }
                }
            }
            
            response = client.post("/api/v1/agents/consensus",
                                 json={"symbols": ["AAPL"], "timeframe": "1d"})
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "AAPL" in data
            assert data["AAPL"]["consensus_signal"] == "BUY"
    
    def test_agent_backtest_success(self):
        """Test successful agent backtesting"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.backtest = AsyncMock(return_value={
                "symbol": "AAPL",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.68,
                "total_trades": 45
            })
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.post("/api/v1/agents/technical_analysis_agent/backtest",
                                 json={
                                     "symbol": "AAPL",
                                     "start_date": "2023-01-01",
                                     "end_date": "2023-12-31"
                                 })
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["symbol"] == "AAPL"
            assert data["total_return"] == 0.15
    
    def test_agent_optimize_parameters_success(self):
        """Test successful parameter optimization"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.optimize_parameters = AsyncMock(return_value={
                "optimized_parameters": {
                    "rsi_period": 18,
                    "ma_short": 12,
                    "ma_long": 26
                },
                "performance": {
                    "sharpe_ratio": 1.45,
                    "total_return": 0.22,
                    "win_rate": 0.71
                },
                "optimization_method": "grid_search"
            })
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            response = client.post("/api/v1/agents/technical_analysis_agent/optimize",
                                 json={
                                     "symbol": "AAPL",
                                     "optimization_period": "1y",
                                     "method": "grid_search"
                                 })
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "optimized_parameters" in data
            assert data["performance"]["sharpe_ratio"] == 1.45


@pytest.mark.asyncio
class TestAgentsAPIAsync:
    """Async tests for agents API"""
    
    async def test_concurrent_agent_analyses(self):
        """Test handling concurrent analysis requests"""
        with patch('api.routes.agents.get_agent_registry') as mock_registry:
            mock_agent = Mock()
            mock_agent.analyze = AsyncMock(return_value={
                "signal": "BUY", "confidence": 0.8
            })
            mock_registry.return_value = {"technical_analysis_agent": mock_agent}
            
            # Simulate concurrent requests
            import asyncio
            tasks = []
            for _ in range(10):
                task = asyncio.create_task(
                    asyncio.to_thread(
                        client.post,
                        "/api/v1/agents/technical_analysis_agent/analyze",
                        json={"symbol": "AAPL"}
                    )
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == status.HTTP_200_OK


@pytest.mark.integration  
class TestAgentsAPIIntegration:
    """Integration tests with real agents"""
    
    def test_real_agent_integration(self):
        """Test with real agent instances"""
        pytest.skip("Requires real agent instances - run manually")
    
    def test_agent_performance_benchmarks(self):
        """Test agent performance under load"""
        pytest.skip("Performance test - run manually")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])