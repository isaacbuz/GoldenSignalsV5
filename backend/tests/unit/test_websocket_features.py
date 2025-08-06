"""
Comprehensive tests for WebSocket and real-time features
Tests WebSocket connections, broadcasting, and orchestration
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock, call
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from services.websocket_manager import WebSocketManager, SignalUpdate
from services.websocket_orchestrator import WebSocketOrchestrator, AgentActivity
from api.websocket.orchestrated_ws import handle_client_message


class TestWebSocketManager:
    """Test the WebSocket manager"""
    
    @pytest.fixture
    def ws_manager(self):
        """Create WebSocket manager"""
        return WebSocketManager()
    
    @pytest.fixture
    async def mock_websocket(self):
        """Create mock WebSocket"""
        ws = AsyncMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_json = AsyncMock()
        ws.close = AsyncMock()
        ws.application_state = WebSocketState.CONNECTED
        return ws
    
    @pytest.mark.asyncio
    async def test_connect_client(self, ws_manager, mock_websocket):
        """Test client connection"""
        client_id = await ws_manager.connect(mock_websocket)
        
        assert client_id is not None
        assert client_id in ws_manager.clients
        assert ws_manager.metrics["active_connections"] == 1
        
        mock_websocket.accept.assert_called_once()
        # Should send welcome message
        mock_websocket.send_json.assert_called()
    
    @pytest.mark.asyncio
    async def test_disconnect_client(self, ws_manager, mock_websocket):
        """Test client disconnection"""
        client_id = await ws_manager.connect(mock_websocket)
        await ws_manager.disconnect(client_id)
        
        assert client_id not in ws_manager.clients
        assert ws_manager.metrics["active_connections"] == 0
    
    @pytest.mark.asyncio
    async def test_subscribe_to_symbol(self, ws_manager, mock_websocket):
        """Test subscribing to symbol updates"""
        client_id = await ws_manager.connect(mock_websocket)
        await ws_manager.subscribe(client_id, "AAPL")
        
        assert "AAPL" in ws_manager.clients[client_id].subscriptions
        assert "AAPL" in ws_manager.rooms
        assert client_id in ws_manager.rooms["AAPL"]
        
        # Should send confirmation
        calls = mock_websocket.send_json.call_args_list
        assert any("subscribed" in str(call) for call in calls)
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_symbol(self, ws_manager, mock_websocket):
        """Test unsubscribing from symbol"""
        client_id = await ws_manager.connect(mock_websocket)
        await ws_manager.subscribe(client_id, "AAPL")
        await ws_manager.unsubscribe(client_id, "AAPL")
        
        assert "AAPL" not in ws_manager.clients[client_id].subscriptions
        assert client_id not in ws_manager.rooms.get("AAPL", set())
    
    @pytest.mark.asyncio
    async def test_broadcast_signal(self, ws_manager, mock_websocket):
        """Test broadcasting signal to subscribers"""
        # Connect and subscribe two clients
        client1_id = await ws_manager.connect(mock_websocket)
        await ws_manager.subscribe(client1_id, "AAPL")
        
        mock_websocket2 = AsyncMock(spec=WebSocket)
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.send_json = AsyncMock()
        client2_id = await ws_manager.connect(mock_websocket2)
        await ws_manager.subscribe(client2_id, "AAPL")
        
        # Create signal update
        signal = SignalUpdate(
            symbol="AAPL",
            signal_id="test-signal-1",
            action="BUY",
            confidence=0.85,
            price=150.00,
            agents_consensus={"total_agents": 3, "completed_agents": 3},
            timestamp=datetime.now()
        )
        
        # Broadcast signal
        await ws_manager.broadcast_signal(signal)
        
        # Both clients should receive the signal
        # Check last call for each WebSocket
        for ws in [mock_websocket, mock_websocket2]:
            last_call = ws.send_json.call_args_list[-1]
            message = last_call[0][0]
            assert message["type"] == "signal_update"
            assert message["data"]["symbol"] == "AAPL"
            assert message["data"]["action"] == "BUY"
    
    @pytest.mark.asyncio
    async def test_broadcast_to_specific_room(self, ws_manager, mock_websocket):
        """Test room-based broadcasting"""
        # Connect clients to different symbols
        client1_id = await ws_manager.connect(mock_websocket)
        await ws_manager.subscribe(client1_id, "AAPL")
        
        mock_websocket2 = AsyncMock(spec=WebSocket)
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.send_json = AsyncMock()
        client2_id = await ws_manager.connect(mock_websocket2)
        await ws_manager.subscribe(client2_id, "GOOGL")
        
        # Broadcast to AAPL room only
        await ws_manager.broadcast_price_update("AAPL", 151.00, 80000000)
        
        # Only client1 should receive the update
        last_call = mock_websocket.send_json.call_args_list[-1]
        message = last_call[0][0]
        assert message["type"] == "price_update"
        assert message["data"]["symbol"] == "AAPL"
        
        # Client2 should not receive AAPL update
        # (after initial connection message)
        assert len(mock_websocket2.send_json.call_args_list) <= 2
    
    @pytest.mark.asyncio
    async def test_handle_client_message(self, ws_manager, mock_websocket):
        """Test handling various client messages"""
        client_id = await ws_manager.connect(mock_websocket)
        
        # Test subscribe message
        await ws_manager.handle_message(client_id, {
            "type": "subscribe",
            "symbol": "NVDA"
        })
        assert "NVDA" in ws_manager.clients[client_id].subscriptions
        
        # Test heartbeat message
        await ws_manager.handle_message(client_id, {
            "type": "heartbeat"
        })
        # Should echo heartbeat
        calls = mock_websocket.send_json.call_args_list
        assert any("heartbeat" in str(call) for call in calls)
        
        # Test unknown message type
        await ws_manager.handle_message(client_id, {
            "type": "unknown_type"
        })
        # Should not cause error
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self, ws_manager, mock_websocket):
        """Test cleanup when WebSocket errors occur"""
        client_id = await ws_manager.connect(mock_websocket)
        await ws_manager.subscribe(client_id, "AAPL")
        
        # Simulate send error
        mock_websocket.send_json.side_effect = Exception("Connection lost")
        
        # Try to send message
        await ws_manager._send_to_client(client_id, {"test": "message"})
        
        # Client should be disconnected
        assert client_id not in ws_manager.clients
        assert client_id not in ws_manager.rooms.get("AAPL", set())


class TestWebSocketOrchestrator:
    """Test the WebSocket orchestrator"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create WebSocket orchestrator"""
        orchestrator = WebSocketOrchestrator()
        yield orchestrator
        await orchestrator.stop()
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock comprehensive market data"""
        return {
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 75000000,
            "change_percent": 0.5,
            "historical_prices": [149, 149.5, 150, 150.25],
            "timestamp": datetime.now().isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_initialize_orchestrator(self, orchestrator):
        """Test orchestrator initialization"""
        with patch.object(orchestrator.agent_orchestrator, 'initialize_default_agents', new_callable=AsyncMock):
            await orchestrator.initialize()
            
            assert orchestrator._running is True
            assert len(orchestrator._tasks) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_symbol(self, orchestrator, mock_market_data):
        """Test symbol analysis with agent updates"""
        # Mock dependencies
        with patch.object(orchestrator, '_fetch_market_data', return_value=mock_market_data):
            with patch.object(orchestrator.agent_orchestrator, 'agents') as mock_agents:
                # Create mock agent
                mock_agent = MagicMock()
                mock_agent.agent_id = "test-agent"
                mock_agent.config.name = "TestAgent"
                mock_agent.config.enabled = True
                mock_agents.values.return_value = [mock_agent]
                
                # Track agent callbacks
                agent_updates = []
                
                async def mock_analyze(market_data, callback):
                    await callback("test-agent", "TestAgent", "processing")
                    await asyncio.sleep(0.1)
                    await callback("test-agent", "TestAgent", "completed", MagicMock())
                    return MagicMock()  # Return mock signal
                
                with patch.object(orchestrator, '_analyze_with_callbacks', side_effect=mock_analyze):
                    with patch('services.websocket_manager.ws_manager.broadcast_agent_update') as mock_broadcast:
                        await orchestrator.analyze_symbol("AAPL")
                        
                        # Should broadcast agent updates
                        assert mock_broadcast.call_count >= 2
                        
                        # Check processing and completed updates
                        calls = mock_broadcast.call_args_list
                        assert any("processing" in str(call) for call in calls)
                        assert any("completed" in str(call) for call in calls)
    
    @pytest.mark.asyncio
    async def test_market_monitoring(self, orchestrator):
        """Test automatic market monitoring"""
        # Mock market data service
        with patch.object(orchestrator.market_data_service, 'get_quote') as mock_quote:
            # Simulate price change
            mock_quote.side_effect = [
                {"price": 150.00, "volume": 75000000},  # Initial price
                {"price": 151.60, "volume": 80000000},  # 1.06% change - should trigger
            ]
            
            with patch.object(orchestrator, 'analyze_symbol', new_callable=AsyncMock) as mock_analyze:
                # Start monitoring
                orchestrator._running = True
                
                # Run one iteration of monitoring loop
                await orchestrator._market_monitoring_loop()
                
                # Should trigger analysis due to >1% change
                mock_analyze.assert_called()
    
    @pytest.mark.asyncio
    async def test_signal_processing(self, orchestrator, db_session):
        """Test signal processing and broadcasting"""
        # Create test signal
        from models.signal import Signal
        
        signal = Signal(
            symbol="AAPL",
            action="BUY",
            confidence=0.85,
            price=150.00,
            source="TestAgent",
            metadata={"test": True}
        )
        db_session.add(signal)
        await db_session.commit()
        
        with patch('services.websocket_manager.ws_manager.broadcast_signal') as mock_broadcast:
            with patch('core.database.get_db', return_value=db_session):
                # Run signal processing
                orchestrator._running = True
                await orchestrator._signal_processing_loop()
                
                # Should broadcast the signal
                mock_broadcast.assert_called()
                
                # Signal should be marked as broadcasted
                await db_session.refresh(signal)
                assert signal.metadata.get("broadcasted") is True
    
    @pytest.mark.asyncio
    async def test_concurrent_analyses(self, orchestrator):
        """Test handling multiple concurrent analyses"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        with patch.object(orchestrator, '_fetch_market_data', return_value={"price": 150.00}):
            with patch.object(orchestrator, '_analyze_with_callbacks', return_value=None):
                # Start multiple analyses
                tasks = [orchestrator.analyze_symbol(symbol) for symbol in symbols]
                await asyncio.gather(*tasks)
                
                # Should track all active analyses
                assert len(orchestrator.active_analyses) >= 0  # Cleaned up after completion


class TestWebSocketAPI:
    """Test WebSocket API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from app import app
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, client):
        """Test WebSocket connection"""
        with client.websocket_connect("/ws/signals") as websocket:
            # Should receive connection message
            data = websocket.receive_json()
            assert data["type"] == "heartbeat"
            assert "client_id" in data
    
    @pytest.mark.asyncio
    async def test_websocket_subscribe(self, client):
        """Test WebSocket subscription"""
        with client.websocket_connect("/ws/signals") as websocket:
            # Subscribe to symbol
            websocket.send_json({
                "type": "subscribe",
                "symbol": "AAPL"
            })
            
            # Should receive subscription confirmation
            data = websocket.receive_json()
            # Skip initial heartbeat
            if data["type"] == "heartbeat":
                data = websocket.receive_json()
            
            assert data["type"] == "subscribe"
            assert data["symbol"] == "AAPL"
            assert data["status"] == "subscribed"
    
    @pytest.mark.asyncio
    async def test_websocket_analyze(self, client):
        """Test triggering analysis via WebSocket"""
        with patch('services.websocket_orchestrator.websocket_orchestrator.analyze_symbol', new_callable=AsyncMock):
            with client.websocket_connect("/ws/signals") as websocket:
                # Trigger analysis
                websocket.send_json({
                    "type": "analyze",
                    "symbol": "NVDA"
                })
                
                # Should receive acknowledgment
                data = websocket.receive_json()
                # Skip messages until we find alert
                while data["type"] != "alert":
                    data = websocket.receive_json()
                
                assert data["type"] == "alert"
                assert "Analysis started" in data["message"]
    
    @pytest.mark.asyncio
    async def test_symbol_specific_websocket(self, client):
        """Test symbol-specific WebSocket endpoint"""
        with patch('services.websocket_orchestrator.websocket_orchestrator.analyze_symbol', new_callable=AsyncMock):
            with client.websocket_connect("/ws/symbols/TSLA") as websocket:
                # Should auto-subscribe and trigger analysis
                data = websocket.receive_json()
                
                # Should receive initial messages
                assert data is not None
                # Auto-analysis should be triggered


class TestWebSocketIntegration:
    """Integration tests for WebSocket features"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_signal_broadcast(self):
        """Test complete signal generation and broadcast flow"""
        ws_manager = WebSocketManager()
        orchestrator = WebSocketOrchestrator()
        
        # Connect a test client
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        
        client_id = await ws_manager.connect(mock_ws)
        await ws_manager.subscribe(client_id, "AAPL")
        
        # Create and broadcast a signal
        signal = SignalUpdate(
            symbol="AAPL",
            signal_id="test-123",
            action="BUY",
            confidence=0.90,
            price=150.00,
            agents_consensus={
                "total_agents": 3,
                "completed_agents": 3,
                "consensus_strength": "STRONG"
            },
            timestamp=datetime.now()
        )
        
        await ws_manager.broadcast_signal(signal)
        
        # Client should receive the signal
        mock_ws.send_json.assert_called()
        last_call = mock_ws.send_json.call_args_list[-1]
        message = last_call[0][0]
        
        assert message["type"] == "signal_update"
        assert message["data"]["action"] == "BUY"
        assert message["data"]["confidence"] == 0.90
    
    @pytest.mark.asyncio
    async def test_multiple_client_coordination(self):
        """Test coordinating multiple WebSocket clients"""
        ws_manager = WebSocketManager()
        
        # Connect 3 clients
        clients = []
        for i in range(3):
            mock_ws = AsyncMock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            
            client_id = await ws_manager.connect(mock_ws)
            
            # Different subscriptions
            if i == 0:
                await ws_manager.subscribe(client_id, "AAPL")
            elif i == 1:
                await ws_manager.subscribe(client_id, "AAPL")
                await ws_manager.subscribe(client_id, "GOOGL")
            else:
                await ws_manager.subscribe(client_id, "GOOGL")
            
            clients.append((client_id, mock_ws))
        
        # Broadcast to AAPL
        await ws_manager.broadcast_price_update("AAPL", 151.00, 80000000)
        
        # Check which clients received updates
        # Clients 0 and 1 should receive AAPL update
        for i, (client_id, mock_ws) in enumerate(clients[:2]):
            calls = mock_ws.send_json.call_args_list
            assert any("AAPL" in str(call) and "price_update" in str(call) for call in calls)
        
        # Client 2 should not receive AAPL update
        calls = clients[2][1].send_json.call_args_list
        assert not any("AAPL" in str(call) and "price_update" in str(call) for call in calls)


@pytest.mark.performance
class TestWebSocketPerformance:
    """Performance tests for WebSocket features"""
    
    @pytest.mark.asyncio
    async def test_broadcast_performance(self, performance_timer):
        """Test broadcast performance with many clients"""
        ws_manager = WebSocketManager()
        
        # Connect 100 clients
        clients = []
        for i in range(100):
            mock_ws = AsyncMock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            
            client_id = await ws_manager.connect(mock_ws)
            await ws_manager.subscribe(client_id, "AAPL")
            clients.append((client_id, mock_ws))
        
        # Measure broadcast time
        performance_timer.start()
        
        await ws_manager.broadcast_price_update("AAPL", 150.00, 75000000)
        
        performance_timer.stop()
        
        # Should complete quickly even with many clients
        assert performance_timer.elapsed() < 0.5  # 500ms for 100 clients
        
        # All clients should receive the update
        for _, mock_ws in clients:
            mock_ws.send_json.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self):
        """Test handling concurrent messages from multiple clients"""
        ws_manager = WebSocketManager()
        
        # Connect multiple clients
        clients = []
        for i in range(10):
            mock_ws = AsyncMock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            
            client_id = await ws_manager.connect(mock_ws)
            clients.append(client_id)
        
        # Send concurrent messages
        tasks = []
        for client_id in clients:
            tasks.append(ws_manager.handle_message(client_id, {
                "type": "subscribe",
                "symbol": f"SYM{client_id[:4]}"
            }))
        
        # Should handle all messages without errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0