"""
Universal Market Data MCP Server for GoldenSignalsAI V5
MCP interface for the Universal Market Data Service
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime

from services.universal_market_data import (
    universal_market_data,
    AssetClass,
    DataSource
)
from core.logging import get_logger

logger = get_logger(__name__)


class UniversalDataMCPServer:
    """
    MCP Server wrapper for Universal Market Data
    Provides standardized MCP interface for all market data
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Universal Market Data MCP Server",
            description="Production-grade market data aggregation with MCP interface",
            version="2.0.0"
        )
        self.websocket_clients: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # client_id -> symbols
        self._setup_routes()
        logger.info("Universal Data MCP Server initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes for MCP interface"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Universal Market Data MCP",
                "version": "2.0.0",
                "status": "active",
                "capabilities": [
                    "real-time prices",
                    "historical data",
                    "order book",
                    "market status",
                    "multi-source failover",
                    "rate limiting",
                    "caching"
                ]
            }
        
        @self.app.get("/tools")
        async def list_tools():
            """List available MCP tools"""
            return {
                "tools": [
                    {
                        "name": "get_price",
                        "description": "Get current price with automatic failover",
                        "parameters": {
                            "symbol": "string (required)",
                            "asset_class": f"string (optional, options: {', '.join([a.value for a in AssetClass])})"
                        }
                    },
                    {
                        "name": "get_quote",
                        "description": "Get detailed quote information",
                        "parameters": {
                            "symbol": "string (required)"
                        }
                    },
                    {
                        "name": "get_historical",
                        "description": "Get historical price data with indicators",
                        "parameters": {
                            "symbol": "string (required)",
                            "start_date": "string (YYYY-MM-DD)",
                            "end_date": "string (YYYY-MM-DD)",
                            "interval": "string (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)"
                        }
                    },
                    {
                        "name": "get_order_book",
                        "description": "Get order book data",
                        "parameters": {
                            "symbol": "string (required)",
                            "depth": "integer (optional, default: 10)"
                        }
                    },
                    {
                        "name": "get_market_status",
                        "description": "Get global market open/close status",
                        "parameters": {}
                    },
                    {
                        "name": "get_multi_price",
                        "description": "Get prices for multiple symbols",
                        "parameters": {
                            "symbols": "array of strings (required)"
                        }
                    }
                ]
            }
        
        @self.app.post("/call")
        async def call_tool(request: Dict[str, Any]):
            """Execute an MCP tool call"""
            tool_name = request.get("tool")
            params = request.get("parameters", {})
            
            try:
                logger.info(f"MCP tool call: {tool_name} with params: {params}")
                
                if tool_name == "get_price":
                    symbol = params.get("symbol")
                    if not symbol:
                        raise ValueError("Symbol is required")
                    
                    asset_class = AssetClass(params.get("asset_class", "equity"))
                    result = await universal_market_data.get_price(symbol, asset_class)
                    
                elif tool_name == "get_quote":
                    symbol = params.get("symbol")
                    if not symbol:
                        raise ValueError("Symbol is required")
                    
                    result = await universal_market_data.get_quote(symbol)
                    
                elif tool_name == "get_historical":
                    symbol = params.get("symbol")
                    start_date = params.get("start_date")
                    end_date = params.get("end_date")
                    interval = params.get("interval", "1d")
                    
                    if not all([symbol, start_date, end_date]):
                        raise ValueError("Symbol, start_date, and end_date are required")
                    
                    result = await universal_market_data.get_historical(
                        symbol, start_date, end_date, interval
                    )
                    
                elif tool_name == "get_order_book":
                    symbol = params.get("symbol")
                    if not symbol:
                        raise ValueError("Symbol is required")
                    
                    depth = params.get("depth", 10)
                    result = await universal_market_data.get_order_book(symbol, depth)
                    
                elif tool_name == "get_market_status":
                    result = await universal_market_data.get_market_status()
                    
                elif tool_name == "get_multi_price":
                    symbols = params.get("symbols")
                    if not symbols:
                        raise ValueError("Symbols array is required")
                    
                    results = {}
                    for symbol in symbols:
                        try:
                            price_data = await universal_market_data.get_price(symbol)
                            results[symbol] = price_data
                        except Exception as e:
                            results[symbol] = {"error": str(e)}
                    
                    result = {"symbols": results}
                    
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}")
                return {
                    "success": False,
                    "tool": tool_name,
                    "error": str(e)
                }
        
        @self.app.websocket("/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket endpoint for real-time streaming"""
            await websocket.accept()
            client_id = f"client_{datetime.now().timestamp()}"
            self.websocket_clients[client_id] = websocket
            self.subscriptions[client_id] = []
            
            logger.info(f"WebSocket client connected: {client_id}")
            
            try:
                # Start background task for streaming
                stream_task = None
                
                while True:
                    data = await websocket.receive_text()
                    request = json.loads(data)
                    
                    action = request.get('action')
                    
                    if action == 'subscribe':
                        symbols = request.get('symbols', [])
                        self.subscriptions[client_id] = symbols
                        
                        # Cancel existing stream task
                        if stream_task:
                            stream_task.cancel()
                        
                        # Start new streaming task
                        stream_task = asyncio.create_task(
                            self._stream_to_client(client_id, symbols)
                        )
                        
                        await websocket.send_json({
                            'type': 'subscribed',
                            'symbols': symbols,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    elif action == 'unsubscribe':
                        symbols = request.get('symbols', [])
                        
                        # Remove symbols from subscription
                        for symbol in symbols:
                            if symbol in self.subscriptions[client_id]:
                                self.subscriptions[client_id].remove(symbol)
                        
                        await websocket.send_json({
                            'type': 'unsubscribed',
                            'symbols': symbols,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    elif action == 'ping':
                        await websocket.send_json({
                            'type': 'pong',
                            'timestamp': datetime.now().isoformat()
                        })
                        
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {client_id}")
            except Exception as e:
                logger.error(f"WebSocket error for {client_id}: {e}")
            finally:
                # Cleanup
                if stream_task:
                    stream_task.cancel()
                del self.websocket_clients[client_id]
                del self.subscriptions[client_id]
        
        @self.app.get("/status")
        async def get_status():
            """Get server status and statistics"""
            return {
                "status": "active",
                "connected_clients": len(self.websocket_clients),
                "active_subscriptions": sum(len(symbols) for symbols in self.subscriptions.values()),
                "cache_status": "active",
                "rate_limiter_status": "active",
                "data_sources": [source.value for source in DataSource],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Test data fetching
                test_price = await universal_market_data.get_price("AAPL")
                
                return {
                    "status": "healthy",
                    "service": "Universal Market Data MCP",
                    "test_fetch": "success",
                    "test_symbol": "AAPL",
                    "test_price": test_price.get("price")
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }
    
    async def _stream_to_client(self, client_id: str, symbols: List[str]):
        """Stream real-time data to a specific client"""
        websocket = self.websocket_clients.get(client_id)
        if not websocket:
            return
        
        while client_id in self.websocket_clients:
            try:
                if not symbols:
                    await asyncio.sleep(1)
                    continue
                
                # Fetch prices for all symbols
                updates = {}
                for symbol in symbols:
                    try:
                        price_data = await universal_market_data.get_price(symbol)
                        updates[symbol] = {
                            'price': price_data['price'],
                            'change': price_data.get('change', 0),
                            'change_percent': price_data.get('change_percent', 0),
                            'volume': price_data.get('volume', 0),
                            'source': price_data.get('source', 'unknown')
                        }
                    except Exception as e:
                        logger.error(f"Error fetching {symbol}: {e}")
                        updates[symbol] = {'error': str(e)}
                
                # Send update to client
                await websocket.send_json({
                    'type': 'price_update',
                    'data': updates,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Stream every second
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Streaming error for {client_id}: {e}")
                break
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the MCP server"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# Create server instance
mcp_server = UniversalDataMCPServer()

# Export the FastAPI app for ASGI servers
app = mcp_server.app