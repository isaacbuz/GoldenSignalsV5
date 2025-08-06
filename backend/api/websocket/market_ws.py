"""
Market Data WebSocket Handler
Provides real-time market data and predictions
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
import json
import logging
from datetime import datetime

from services.market_data_service import market_data_service
from services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket manager instance
manager = WebSocketManager()

# Active subscriptions
subscriptions: Dict[str, Set[str]] = {}


@router.websocket("/ws/market/{client_id}")
async def market_websocket(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time market data"""
    await manager.connect(websocket, client_id)
    subscriptions[client_id] = set()
    
    try:
        # Send initial connection message
        await manager.send_personal_message(
            {"type": "connected", "message": "Connected to market data stream"},
            websocket
        )
        
        # Start background task for price updates
        update_task = asyncio.create_task(
            send_price_updates(websocket, client_id)
        )
        
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                symbol = data.get("symbol", "SPY")
                subscriptions[client_id].add(symbol)
                await manager.send_personal_message(
                    {"type": "subscribed", "symbol": symbol},
                    websocket
                )
                
            elif data.get("action") == "unsubscribe":
                symbol = data.get("symbol")
                if symbol in subscriptions[client_id]:
                    subscriptions[client_id].remove(symbol)
                    await manager.send_personal_message(
                        {"type": "unsubscribed", "symbol": symbol},
                        websocket
                    )
                    
    except WebSocketDisconnect:
        update_task.cancel()
        manager.disconnect(websocket)
        if client_id in subscriptions:
            del subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        update_task.cancel()
        manager.disconnect(websocket)
        if client_id in subscriptions:
            del subscriptions[client_id]


async def send_price_updates(websocket: WebSocket, client_id: str):
    """Send periodic price updates for subscribed symbols"""
    while True:
        try:
            await asyncio.sleep(2)  # Update every 2 seconds
            
            if client_id not in subscriptions or not subscriptions[client_id]:
                continue
                
            for symbol in subscriptions[client_id]:
                # Get latest quote
                quote = await market_data_service.get_quote(symbol)
                
                if quote:
                    # Send price update
                    await manager.send_personal_message({
                        "type": "price_update",
                        "symbol": symbol,
                        "data": {
                            "price": quote.get("price"),
                            "change": quote.get("change"),
                            "changePercent": quote.get("changePercent"),
                            "volume": quote.get("volume"),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }, websocket)
                    
        except Exception as e:
            logger.error(f"Error sending price updates: {e}")
            break


@router.websocket("/ws/predictions/{symbol}")
async def prediction_websocket(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time prediction updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send prediction updates every 5 seconds
            await asyncio.sleep(5)
            
            # Mock prediction update (replace with actual prediction logic)
            prediction_update = {
                "type": "prediction_update",
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": {
                    "price": 475.50 + (asyncio.get_event_loop().time() % 10) * 0.1,
                    "confidence": 0.85,
                    "trend": "bullish"
                }
            }
            
            await websocket.send_json(prediction_update)
            
    except WebSocketDisconnect:
        logger.info(f"Prediction WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"Prediction WebSocket error: {e}")