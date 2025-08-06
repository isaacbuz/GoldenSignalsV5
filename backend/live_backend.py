"""
Live Backend with WebSocket Support
"""

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import Dict, Set
import logging

# Import services
from services.market_data_service import market_data_service
from api.routes import market_data, signals, ai_analysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            for symbol, subscribers in self.subscriptions.items():
                subscribers.discard(client_id)
            logger.info(f"Client {client_id} disconnected")
            
    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def broadcast(self, message: str, symbol: str = None):
        if symbol and symbol in self.subscriptions:
            for client_id in self.subscriptions[symbol]:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_text(message)
                    except:
                        pass
        else:
            for client_id, connection in self.active_connections.items():
                try:
                    await connection.send_text(message)
                except:
                    pass
                    
    def subscribe(self, client_id: str, symbol: str):
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
        self.subscriptions[symbol].add(client_id)
        logger.info(f"Client {client_id} subscribed to {symbol}")
        
    def unsubscribe(self, client_id: str, symbol: str):
        if symbol in self.subscriptions:
            self.subscriptions[symbol].discard(client_id)

manager = ConnectionManager()

# Background task for live data
async def market_data_broadcaster():
    while True:
        try:
            all_symbols = list(manager.subscriptions.keys())
            
            if all_symbols:
                quotes = await market_data_service.get_quotes(all_symbols)
                
                for symbol, quote in quotes.items():
                    if quote:
                        message = json.dumps({
                            "type": "price_update",
                            "data": quote
                        })
                        await manager.broadcast(message, symbol)
                        
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in broadcaster: {e}")
            await asyncio.sleep(5)

# Create app
app = FastAPI(title="GoldenSignalsAI Live Backend")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(market_data.router, prefix="/api/v1")
app.include_router(signals.router, prefix="/api/v1")
app.include_router(ai_analysis.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(market_data_broadcaster())
    logger.info("Started market data broadcaster")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "GoldenSignalsAI Live"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                symbol = message.get("symbol", "").upper()
                if symbol:
                    manager.subscribe(client_id, symbol)
                    # Send subscription confirmation
                    await manager.send_message(
                        json.dumps({
                            "type": "subscribed",
                            "symbol": symbol,
                            "timestamp": datetime.now().isoformat()
                        }),
                        client_id
                    )
                    # Send immediate quote
                    quote = await market_data_service.get_quote(symbol)
                    if quote:
                        await manager.send_message(
                            json.dumps({
                                "type": "price_update",
                                "data": quote
                            }),
                            client_id
                        )
                        
            elif message.get("type") == "unsubscribe":
                symbol = message.get("symbol", "").upper()
                if symbol:
                    manager.unsubscribe(client_id, symbol)
                    
            elif message.get("type") == "heartbeat":
                await manager.send_message(
                    json.dumps({"type": "heartbeat"}),
                    client_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)