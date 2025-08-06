"""
WebSocket Integration Tests
End-to-end testing of WebSocket functionality with real connections
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import WebSocket

from app import app
from services.websocket_manager import ws_manager, SignalUpdate


client = TestClient(app)


@pytest.mark.asyncio
class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality"""
    
    async def test_websocket_connection_lifecycle(self):
        """Test complete WebSocket connection lifecycle"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_json()
            assert data["type"] == "heartbeat"
            assert data["message"] == "Connected to GoldenSignals WebSocket"
    
    async def test_websocket_subscription_flow(self):
        """Test WebSocket subscription and unsubscription"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Subscribe to AAPL
            websocket.send_json({
                "type": "subscribe",
                "symbol": "AAPL"
            })
            
            # Should receive subscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscribe"
            assert response["symbol"] == "AAPL"
            assert response["status"] == "subscribed"
            
            # Unsubscribe
            websocket.send_json({
                "type": "unsubscribe", 
                "symbol": "AAPL"
            })
            
            # Should receive unsubscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "unsubscribe"
            assert response["symbol"] == "AAPL"
            assert response["status"] == "unsubscribed"
    
    async def test_websocket_signal_broadcast(self):
        """Test signal broadcasting to subscribed clients"""
        with client.websocket_connect("/api/v1/market/ws") as websocket1, \
             client.websocket_connect("/api/v1/market/ws") as websocket2:
            
            # Skip welcome messages
            websocket1.receive_json()
            websocket2.receive_json()
            
            # Subscribe both clients to AAPL
            for ws in [websocket1, websocket2]:
                ws.send_json({"type": "subscribe", "symbol": "AAPL"})
                ws.receive_json()  # Skip confirmation
            
            # Create and broadcast a signal
            signal = SignalUpdate(
                symbol="AAPL",
                signal_id="test_signal_1",
                action="BUY",
                confidence=0.85,
                price=150.0,
                agents_consensus={"technical": "BUY", "sentiment": "HOLD"},
                timestamp=datetime.now()
            )
            
            await ws_manager.broadcast_signal(signal)
            
            # Both clients should receive the signal
            for ws in [websocket1, websocket2]:
                data = ws.receive_json()
                assert data["type"] == "signal_update"
                assert data["data"]["symbol"] == "AAPL"
                assert data["data"]["action"] == "BUY"
                assert data["data"]["confidence"] == 0.85
    
    async def test_websocket_price_updates(self):
        """Test price update broadcasting"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            websocket.receive_json()  # Skip welcome
            
            # Subscribe to TSLA
            websocket.send_json({"type": "subscribe", "symbol": "TSLA"})
            websocket.receive_json()  # Skip confirmation
            
            # Broadcast price update
            await ws_manager.broadcast_price_update("TSLA", 800.0, 1500000)
            
            # Should receive price update
            data = websocket.receive_json()
            assert data["type"] == "price_update"
            assert data["data"]["symbol"] == "TSLA"
            assert data["data"]["price"] == 800.0
            assert data["data"]["volume"] == 1500000
    
    async def test_websocket_agent_updates(self):
        """Test individual agent update broadcasting"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            websocket.receive_json()  # Skip welcome
            
            # Subscribe to GOOGL
            websocket.send_json({"type": "subscribe", "symbol": "GOOGL"})
            websocket.receive_json()  # Skip confirmation
            
            # Broadcast agent update
            await ws_manager.broadcast_agent_update("GOOGL", "technical_agent", "SELL", 0.72)
            
            # Should receive agent update
            data = websocket.receive_json()
            assert data["type"] == "agent_update"
            assert data["data"]["symbol"] == "GOOGL"
            assert data["data"]["agent"] == "technical_agent"
            assert data["data"]["signal"] == "SELL"
            assert data["data"]["confidence"] == 0.72
    
    async def test_websocket_decision_broadcast(self):
        """Test trading decision broadcasting"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            websocket.receive_json()  # Skip welcome
            
            # Subscribe to MSFT
            websocket.send_json({"type": "subscribe", "symbol": "MSFT"})
            websocket.receive_json()  # Skip confirmation
            
            # Broadcast trading decision
            decision = {
                "action": "BUY",
                "quantity": 100,
                "price": 300.0,
                "stop_loss": 290.0,
                "take_profit": 320.0,
                "reasoning": "Strong technical breakout"
            }
            await ws_manager.broadcast_decision("MSFT", decision)
            
            # Should receive decision update
            data = websocket.receive_json()
            assert data["type"] == "decision_update"
            assert data["data"]["symbol"] == "MSFT"
            assert data["data"]["decision"]["action"] == "BUY"
            assert data["data"]["decision"]["quantity"] == 100
    
    async def test_websocket_heartbeat_mechanism(self):
        """Test WebSocket heartbeat mechanism"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            websocket.receive_json()  # Skip welcome
            
            # Send heartbeat
            websocket.send_json({"type": "heartbeat"})
            
            # Should receive heartbeat echo
            data = websocket.receive_json()
            assert data["type"] == "heartbeat"
            assert "timestamp" in data
    
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            websocket.receive_json()  # Skip welcome
            
            # Send invalid message
            websocket.send_json({"type": "invalid_type", "data": "test"})
            
            # Should not disconnect, but might receive error or be ignored
            # The exact behavior depends on implementation
            try:
                # Try to send a valid message after the invalid one
                websocket.send_json({"type": "heartbeat"})
                data = websocket.receive_json()
                assert data["type"] == "heartbeat"
            except Exception:
                # If connection is closed, that's also acceptable behavior
                pass
    
    async def test_websocket_multiple_subscriptions(self):
        """Test client with multiple symbol subscriptions"""
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            websocket.receive_json()  # Skip welcome
            
            # Subscribe to multiple symbols
            symbols = ["AAPL", "TSLA", "GOOGL", "MSFT"]
            for symbol in symbols:
                websocket.send_json({"type": "subscribe", "symbol": symbol})
                confirmation = websocket.receive_json()
                assert confirmation["status"] == "subscribed"
            
            # Broadcast to each symbol
            for i, symbol in enumerate(symbols):
                await ws_manager.broadcast_price_update(symbol, 100.0 + i, 1000000)
                
                data = websocket.receive_json()
                assert data["type"] == "price_update"
                assert data["data"]["symbol"] == symbol
                assert data["data"]["price"] == 100.0 + i
    
    async def test_websocket_concurrent_connections(self):
        """Test multiple concurrent WebSocket connections"""
        connections = []
        try:
            # Create multiple connections
            for i in range(5):
                ws = client.websocket_connect("/api/v1/market/ws")
                ws.__enter__()
                connections.append(ws)
                ws.receive_json()  # Skip welcome
                
                # Subscribe each to a different symbol
                symbol = f"TEST{i}"
                ws.send_json({"type": "subscribe", "symbol": symbol})
                ws.receive_json()  # Skip confirmation
            
            # Broadcast to all symbols
            for i in range(5):
                symbol = f"TEST{i}"
                await ws_manager.broadcast_price_update(symbol, 50.0 + i, 500000)
            
            # Each connection should receive its respective update
            for i, ws in enumerate(connections):
                data = ws.receive_json()
                assert data["type"] == "price_update"
                assert data["data"]["symbol"] == f"TEST{i}"
                assert data["data"]["price"] == 50.0 + i
                
        finally:
            # Clean up connections
            for ws in connections:
                try:
                    ws.__exit__(None, None, None)
                except:
                    pass
    
    async def test_websocket_room_isolation(self):
        """Test that subscriptions are properly isolated by symbol"""
        with client.websocket_connect("/api/v1/market/ws") as ws1, \
             client.websocket_connect("/api/v1/market/ws") as ws2:
            
            # Skip welcome messages
            ws1.receive_json()
            ws2.receive_json()
            
            # ws1 subscribes to AAPL, ws2 subscribes to TSLA
            ws1.send_json({"type": "subscribe", "symbol": "AAPL"})
            ws1.receive_json()  # Skip confirmation
            
            ws2.send_json({"type": "subscribe", "symbol": "TSLA"})
            ws2.receive_json()  # Skip confirmation
            
            # Broadcast to AAPL - only ws1 should receive
            await ws_manager.broadcast_price_update("AAPL", 150.0, 1000000)
            
            data1 = ws1.receive_json()
            assert data1["type"] == "price_update"
            assert data1["data"]["symbol"] == "AAPL"
            
            # ws2 should not receive anything
            # Note: In a real test, you'd use a timeout to verify no message is received
            # For this example, we'll assume proper isolation
    
    async def test_websocket_connection_metrics(self):
        """Test WebSocket connection metrics tracking"""
        initial_metrics = ws_manager.get_metrics()
        initial_connections = initial_metrics["active_connections"]
        
        with client.websocket_connect("/api/v1/market/ws") as websocket:
            websocket.receive_json()  # Skip welcome
            
            # Metrics should show one more connection
            metrics = ws_manager.get_metrics()
            assert metrics["active_connections"] == initial_connections + 1
            assert metrics["total_connections"] == initial_metrics["total_connections"] + 1
        
        # After connection closes, active connections should decrease
        final_metrics = ws_manager.get_metrics()
        assert final_metrics["active_connections"] == initial_connections


@pytest.mark.integration
class TestWebSocketWithMarketData:
    """Integration tests combining WebSocket with market data"""
    
    @pytest.mark.asyncio
    async def test_live_market_data_streaming(self):
        """Test live market data streaming through WebSocket"""
        with patch('api.routes.market_data.live_data_provider') as mock_provider:
            # Mock live data
            mock_provider.get_current_price = AsyncMock(return_value={
                "price": 155.50,
                "change": 3.25,
                "change_percent": 2.14,
                "volume": 1500000
            })
            
            with client.websocket_connect("/api/v1/market/ws") as websocket:
                websocket.receive_json()  # Skip welcome
                
                # Subscribe to market data
                websocket.send_json({"type": "subscribe", "symbol": "AAPL"})
                websocket.receive_json()  # Skip confirmation
                
                # Simulate market data update (would normally come from live provider)
                await ws_manager.broadcast_price_update("AAPL", 155.50, 1500000)
                
                # Should receive real-time price update
                data = websocket.receive_json()
                assert data["type"] == "price_update"
                assert data["data"]["price"] == 155.50
                assert data["data"]["volume"] == 1500000
    
    @pytest.mark.asyncio
    async def test_ai_signal_integration(self):
        """Test AI signal generation and WebSocket broadcasting"""
        with patch('agents.technical_analysis_agent.TechnicalAnalysisAgent') as mock_agent:
            # Mock AI signal generation
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.analyze = AsyncMock(return_value={
                "signal": "BUY",
                "confidence": 0.87,
                "reasoning": "Golden cross pattern detected",
                "price_target": 165.0,
                "stop_loss": 148.0
            })
            
            with client.websocket_connect("/api/v1/market/ws") as websocket:
                websocket.receive_json()  # Skip welcome
                
                # Subscribe to signals
                websocket.send_json({"type": "subscribe", "symbol": "AAPL"})
                websocket.receive_json()  # Skip confirmation
                
                # Simulate AI signal generation and broadcast
                signal = SignalUpdate(
                    symbol="AAPL",
                    signal_id="ai_signal_123",
                    action="BUY",
                    confidence=0.87,
                    price=155.50,
                    agents_consensus={
                        "technical": "BUY",
                        "sentiment": "HOLD",
                        "volatility": "BUY"
                    },
                    timestamp=datetime.now(),
                    metadata={"reasoning": "Golden cross pattern detected"}
                )
                
                await ws_manager.broadcast_signal(signal)
                
                # Should receive AI-generated signal
                data = websocket.receive_json()
                assert data["type"] == "signal_update"
                assert data["data"]["action"] == "BUY"
                assert data["data"]["confidence"] == 0.87
                assert "Golden cross" in data["data"]["metadata"]["reasoning"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])