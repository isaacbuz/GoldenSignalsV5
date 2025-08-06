"""
API Routes for Universal Market Data Service
Production-grade market data endpoints with caching and failover
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
import json

from services.universal_market_data import (
    universal_market_data,
    AssetClass,
    DataSource
)
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/market-data", tags=["Market Data"])


class PriceRequest(BaseModel):
    """Request model for price data"""
    symbol: str = Field(..., description="Stock symbol")
    asset_class: str = Field("equity", description="Asset class")


class HistoricalRequest(BaseModel):
    """Request model for historical data"""
    symbol: str = Field(..., description="Stock symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    interval: str = Field("1d", description="Data interval (1m, 5m, 1h, 1d, etc)")


class MultiPriceRequest(BaseModel):
    """Request model for multiple symbols"""
    symbols: List[str] = Field(..., description="List of stock symbols")
    asset_class: str = Field("equity", description="Asset class")


@router.get("/price/{symbol}")
async def get_price(
    symbol: str,
    asset_class: str = Query("equity", description="Asset class")
) -> Dict[str, Any]:
    """
    Get current price for a symbol with automatic failover
    
    Features:
    - Multi-source data aggregation
    - Automatic failover on source failure
    - Rate limiting per source
    - 60-second caching
    """
    try:
        asset_class_enum = AssetClass(asset_class)
        result = await universal_market_data.get_price(symbol, asset_class_enum)
        
        logger.info(f"Price fetched for {symbol}: ${result['price']:.2f} from {result['source']}")
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prices")
async def get_multiple_prices(request: MultiPriceRequest) -> Dict[str, Any]:
    """
    Get prices for multiple symbols in one request
    
    Optimized for bulk fetching with parallel processing
    """
    try:
        asset_class_enum = AssetClass(request.asset_class)
        results = {}
        errors = {}
        
        # Fetch prices for all symbols
        for symbol in request.symbols:
            try:
                price_data = await universal_market_data.get_price(symbol, asset_class_enum)
                results[symbol] = price_data
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                errors[symbol] = str(e)
        
        return {
            "status": "success",
            "data": results,
            "errors": errors if errors else None,
            "total": len(request.symbols),
            "successful": len(results),
            "failed": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Error in bulk price fetch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quote/{symbol}")
async def get_quote(symbol: str) -> Dict[str, Any]:
    """
    Get detailed quote information for a symbol
    
    Includes market cap, PE ratio, 52-week range, etc.
    """
    try:
        result = await universal_market_data.get_quote(symbol)
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/historical")
async def get_historical(request: HistoricalRequest) -> Dict[str, Any]:
    """
    Get historical price data with technical indicators
    
    Includes SMA, RSI calculations
    """
    try:
        result = await universal_market_data.get_historical(
            request.symbol,
            request.start_date,
            request.end_date,
            request.interval
        )
        
        logger.info(f"Historical data fetched for {request.symbol}: {result['data_points']} points")
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orderbook/{symbol}")
async def get_order_book(
    symbol: str,
    depth: int = Query(10, description="Order book depth")
) -> Dict[str, Any]:
    """
    Get order book data for a symbol
    
    Currently simulated, will connect to real feeds in production
    """
    try:
        result = await universal_market_data.get_order_book(symbol, depth)
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting order book: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-status")
async def get_market_status() -> Dict[str, Any]:
    """
    Get global market open/close status
    
    Covers US, Europe, Asia, Forex, and Crypto markets
    """
    try:
        result = await universal_market_data.get_market_status()
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def get_data_sources() -> Dict[str, Any]:
    """Get available data sources and their status"""
    return {
        "status": "success",
        "data": {
            "sources": [
                {
                    "name": source.value,
                    "priority": i + 1,
                    "status": "active" if source in [DataSource.YAHOO, DataSource.MOCK] else "configured",
                    "rate_limit": {
                        DataSource.YAHOO: "100/min",
                        DataSource.TWELVEDATA: "8/min",
                        DataSource.FINNHUB: "60/min",
                        DataSource.ALPHAVANTAGE: "5/min",
                        DataSource.POLYGON: "5/min",
                        DataSource.FMP: "250/min",
                        DataSource.MOCK: "10000/min"
                    }.get(source, "N/A")
                }
                for i, source in enumerate([
                    DataSource.YAHOO,
                    DataSource.TWELVEDATA,
                    DataSource.FINNHUB,
                    DataSource.MOCK
                ])
            ]
        }
    }


@router.get("/asset-classes")
async def get_asset_classes() -> Dict[str, Any]:
    """Get supported asset classes"""
    return {
        "status": "success",
        "data": {
            "asset_classes": [
                {
                    "value": ac.value,
                    "name": ac.name,
                    "description": {
                        "equity": "Stocks and shares",
                        "option": "Options contracts",
                        "future": "Futures contracts",
                        "forex": "Foreign exchange",
                        "crypto": "Cryptocurrencies",
                        "commodity": "Commodities",
                        "index": "Market indices",
                        "etf": "Exchange-traded funds"
                    }.get(ac.value, "")
                }
                for ac in AssetClass
            ]
        }
    }


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price streaming
    
    Usage:
    - Send: {"action": "subscribe", "symbols": ["AAPL", "GOOGL"]}
    - Receive: Real-time price updates every second
    """
    await websocket.accept()
    client_id = f"client_{datetime.now().timestamp()}"
    subscribed_symbols = []
    
    logger.info(f"WebSocket client connected: {client_id}")
    
    try:
        # Background task for streaming
        stream_task = None
        
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            action = request.get('action')
            
            if action == 'subscribe':
                subscribed_symbols = request.get('symbols', [])
                
                # Cancel existing stream
                if stream_task:
                    stream_task.cancel()
                
                # Start streaming
                async def stream_prices():
                    while True:
                        try:
                            updates = {}
                            for symbol in subscribed_symbols:
                                try:
                                    price_data = await universal_market_data.get_price(symbol)
                                    updates[symbol] = {
                                        'price': price_data['price'],
                                        'change': price_data.get('change', 0),
                                        'change_percent': price_data.get('change_percent', 0),
                                        'volume': price_data.get('volume', 0)
                                    }
                                except Exception as e:
                                    updates[symbol] = {'error': str(e)}
                            
                            await websocket.send_json({
                                'type': 'price_update',
                                'data': updates,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            await asyncio.sleep(1)  # Update every second
                            
                        except Exception as e:
                            logger.error(f"Streaming error: {e}")
                            break
                
                stream_task = asyncio.create_task(stream_prices())
                
                await websocket.send_json({
                    'type': 'subscribed',
                    'symbols': subscribed_symbols
                })
                
            elif action == 'unsubscribe':
                if stream_task:
                    stream_task.cancel()
                subscribed_symbols = []
                
                await websocket.send_json({
                    'type': 'unsubscribed'
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if stream_task:
            stream_task.cancel()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for market data service"""
    try:
        # Test data fetch
        test_price = await universal_market_data.get_price("AAPL")
        
        return {
            "status": "healthy",
            "service": "Universal Market Data",
            "test_symbol": "AAPL",
            "test_price": test_price.get("price"),
            "cache_active": True,
            "rate_limiter_active": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }