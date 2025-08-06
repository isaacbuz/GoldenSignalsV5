"""
Orchestrated WebSocket API
Real-time signal generation with agent activity streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Optional, Dict, Any
import json
import asyncio

from services.websocket_manager import ws_manager
from services.websocket_orchestrator import websocket_orchestrator
from core.logging import get_logger
from core.auth import get_current_user_ws

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/signals")
async def signals_websocket(
    websocket: WebSocket,
    token: Optional[str] = None
):
    """
    Main WebSocket endpoint for real-time signals and agent activities
    
    Supports:
    - Real-time price updates
    - Agent processing activities
    - Signal generation updates
    - Symbol subscriptions
    - On-demand analysis
    """
    client_id = None
    
    try:
        # Authenticate if token provided
        user = None
        if token:
            try:
                user = await get_current_user_ws(token)
            except Exception as e:
                await websocket.close(code=1008, reason="Authentication failed")
                return
        
        # Connect client
        client_id = await ws_manager.connect(
            websocket,
            client_metadata={
                "user_id": user.id if user else None,
                "username": user.username if user else "anonymous"
            }
        )
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Handle messages
        while True:
            # Receive message
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError:
                await ws_manager.send_alert(
                    client_id,
                    "error",
                    "Invalid JSON format"
                )
                continue
            
            # Process message
            await handle_client_message(client_id, message)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if client_id:
            await ws_manager.disconnect(client_id)


async def handle_client_message(client_id: str, message: Dict[str, Any]):
    """Handle incoming client messages"""
    try:
        msg_type = message.get("type")
        
        if msg_type == "subscribe":
            # Subscribe to symbol updates
            symbol = message.get("symbol")
            if symbol:
                await ws_manager.subscribe(client_id, symbol)
                logger.info(f"Client {client_id} subscribed to {symbol}")
        
        elif msg_type == "unsubscribe":
            # Unsubscribe from symbol
            symbol = message.get("symbol")
            if symbol:
                await ws_manager.unsubscribe(client_id, symbol)
                logger.info(f"Client {client_id} unsubscribed from {symbol}")
        
        elif msg_type == "analyze":
            # Trigger real-time analysis
            symbol = message.get("symbol")
            if symbol:
                # Ensure client is subscribed
                await ws_manager.subscribe(client_id, symbol)
                
                # Trigger analysis
                asyncio.create_task(
                    websocket_orchestrator.analyze_symbol(symbol, client_id)
                )
                
                # Send acknowledgment
                await ws_manager.send_alert(
                    client_id,
                    "info",
                    f"Analysis started for {symbol}",
                    severity="info"
                )
                
                logger.info(f"Analysis triggered for {symbol} by client {client_id}")
        
        elif msg_type == "get_status":
            # Get orchestrator status
            status = websocket_orchestrator.get_status()
            await ws_manager._send_to_client(
                client_id,
                {
                    "type": "status_update",
                    "data": status
                }
            )
        
        elif msg_type == "heartbeat":
            # Handle heartbeat
            await ws_manager.handle_message(client_id, message)
        
        else:
            # Unknown message type
            await ws_manager.send_alert(
                client_id,
                "warning",
                f"Unknown message type: {msg_type}",
                severity="warning"
            )
            
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        await ws_manager.send_alert(
            client_id,
            "error",
            f"Error processing message: {str(e)}",
            severity="error"
        )


@router.websocket("/ws/symbols/{symbol}")
async def symbol_websocket(
    websocket: WebSocket,
    symbol: str,
    token: Optional[str] = None
):
    """
    Symbol-specific WebSocket endpoint
    Automatically subscribes to the specified symbol
    """
    client_id = None
    
    try:
        # Authenticate if token provided
        user = None
        if token:
            try:
                user = await get_current_user_ws(token)
            except:
                await websocket.close(code=1008, reason="Authentication failed")
                return
        
        # Connect client
        client_id = await ws_manager.connect(
            websocket,
            client_metadata={
                "user_id": user.id if user else None,
                "username": user.username if user else "anonymous",
                "auto_symbol": symbol
            }
        )
        
        # Auto-subscribe to symbol
        await ws_manager.subscribe(client_id, symbol)
        
        # Trigger initial analysis
        asyncio.create_task(
            websocket_orchestrator.analyze_symbol(symbol, client_id)
        )
        
        logger.info(f"Symbol WebSocket connected: {client_id} -> {symbol}")
        
        # Keep connection alive
        while True:
            # Wait for messages (mainly heartbeats)
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "analyze":
                    # Re-analyze on demand
                    asyncio.create_task(
                        websocket_orchestrator.analyze_symbol(symbol, client_id)
                    )
                else:
                    await ws_manager.handle_message(client_id, message)
                    
            except json.JSONDecodeError:
                continue
                
    except WebSocketDisconnect:
        logger.info(f"Symbol WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Symbol WebSocket error: {e}", exc_info=True)
    finally:
        if client_id:
            await ws_manager.disconnect(client_id)


@router.get("/ws/status")
async def get_websocket_status():
    """Get WebSocket service status"""
    return {
        "websocket_manager": ws_manager.get_metrics(),
        "orchestrator": websocket_orchestrator.get_status()
    }


# WebSocket message format documentation
WEBSOCKET_MESSAGE_FORMATS = {
    "client_to_server": {
        "subscribe": {
            "type": "subscribe",
            "symbol": "AAPL"
        },
        "unsubscribe": {
            "type": "unsubscribe",
            "symbol": "AAPL"
        },
        "analyze": {
            "type": "analyze",
            "symbol": "AAPL"
        },
        "get_status": {
            "type": "get_status"
        },
        "heartbeat": {
            "type": "heartbeat"
        }
    },
    "server_to_client": {
        "price_update": {
            "type": "price_update",
            "data": {
                "symbol": "AAPL",
                "price": 150.00,
                "volume": 1000000,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        },
        "agent_update": {
            "type": "agent_update",
            "data": {
                "symbol": "AAPL",
                "agent": "TechnicalAnalyst",
                "signal": "BUY",
                "confidence": 0.85,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        },
        "signal_update": {
            "type": "signal_update",
            "data": {
                "symbol": "AAPL",
                "signal_id": "uuid",
                "action": "BUY",
                "confidence": 0.90,
                "price": 150.00,
                "agents_consensus": {
                    "total_agents": 5,
                    "completed_agents": 5,
                    "consensus_strength": "STRONG"
                },
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {}
            }
        },
        "decision_update": {
            "type": "decision_update",
            "data": {
                "symbol": "AAPL",
                "decision": {
                    "action": "BUY",
                    "reason": "Strong technical and sentiment indicators"
                },
                "timestamp": "2024-01-01T00:00:00Z"
            }
        },
        "alert": {
            "type": "alert",
            "alert_type": "info",
            "message": "Analysis started",
            "severity": "info",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }
}