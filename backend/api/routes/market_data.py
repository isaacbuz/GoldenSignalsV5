"""
Market Data API Routes V2
Enhanced market data endpoints with multi-provider support
"""

from fastapi import APIRouter, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

from pydantic import BaseModel, Field

from services.market_data_aggregator import (
    market_data_aggregator,
    DataStrategy
)
from core.market_data.base import (
    market_data_registry,
    MarketDataContext,
    DataQuality,
    ProviderStatus
)
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Market Data V2"])


class MarketDataRequest(BaseModel):
    """Request model for market data"""
    symbols: List[str] = Field(..., description="List of symbols")
    timeframe: str = Field("1d", description="Timeframe (1m, 5m, 15m, 30m, 1h, 1d)")
    start_date: Optional[datetime] = Field(None, description="Start date for historical data")
    end_date: Optional[datetime] = Field(None, description="End date for historical data")
    data_types: List[str] = Field(default_factory=lambda: ["ohlcv"], description="Data types to fetch")
    quality: str = Field("minute", description="Data quality level")
    strategy: str = Field("fallback", description="Data fetching strategy")
    include_extended: bool = Field(False, description="Include extended hours data")
    use_cache: bool = Field(True, description="Use cached data if available")


class ProviderHealthResponse(BaseModel):
    """Response model for provider health"""
    provider_name: str
    status: str
    uptime_percentage: float
    avg_latency_ms: float
    error_rate: float
    quality_score: float
    last_success: Optional[datetime]


@router.post("/data/fetch")
async def fetch_market_data(request: MarketDataRequest) -> Dict[str, Any]:
    """
    Fetch market data with intelligent routing
    
    Uses multiple providers with automatic fallback and quality control
    """
    try:
        # Convert string enums
        strategy = DataStrategy[request.strategy.upper()]
        quality = DataQuality[request.quality.upper()]
        
        # Fetch data
        data = await market_data_aggregator.fetch_data(
            symbols=request.symbols,
            timeframe=request.timeframe,
            strategy=strategy,
            use_cache=request.use_cache,
            start_date=request.start_date,
            end_date=request.end_date,
            data_types=request.data_types,
            quality=quality,
            include_extended=request.include_extended
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Convert DataFrame to dict for JSON response
        response = {
            "success": True,
            "data": data.to_dict(orient="records"),
            "metadata": {
                "symbols": request.symbols,
                "timeframe": request.timeframe,
                "rows": len(data),
                "columns": list(data.columns)
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/quote/{symbol}")
async def get_latest_quote(symbol: str) -> Dict[str, Any]:
    """
    Get latest quote for a symbol
    
    Returns best bid/ask from available providers
    """
    try:
        context = MarketDataContext(
            symbols=[symbol],
            timeframe="1s",
            data_types=["quotes"]
        )
        
        # Try to get quote from primary provider first
        primary = market_data_registry.get_primary_provider()
        if primary:
            response = await primary.fetch_quotes(context)
            
            if response and not response.data.empty:
                quote = response.data.iloc[0].to_dict()
                return {
                    "symbol": symbol,
                    "bid": quote.get("bid_price"),
                    "ask": quote.get("ask_price"),
                    "bid_size": quote.get("bid_size"),
                    "ask_size": quote.get("ask_size"),
                    "spread": quote.get("ask_price", 0) - quote.get("bid_price", 0),
                    "timestamp": quote.get("timestamp"),
                    "provider": response.provider
                }
        
        # Fallback to other providers
        response = await market_data_registry.fetch_with_fallback(context)
        
        if response and not response.data.empty:
            quote = response.data.iloc[0].to_dict()
            return {
                "symbol": symbol,
                "bid": quote.get("bid_price"),
                "ask": quote.get("ask_price"),
                "bid_size": quote.get("bid_size"),
                "ask_size": quote.get("ask_size"),
                "spread": quote.get("ask_price", 0) - quote.get("bid_price", 0),
                "timestamp": quote.get("timestamp"),
                "provider": response.provider
            }
        
        raise HTTPException(status_code=404, detail="Quote not available")
        
    except Exception as e:
        logger.error(f"Error fetching quote: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/snapshot")
async def get_market_snapshot(
    symbols: List[str] = Query(..., description="List of symbols")
) -> Dict[str, Any]:
    """
    Get market snapshot for multiple symbols
    
    Returns current price, change, volume for each symbol
    """
    try:
        # Fetch latest data for all symbols
        data = await market_data_aggregator.fetch_data(
            symbols=symbols,
            timeframe="1d",
            strategy=DataStrategy.FASTEST,
            use_cache=True
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Process snapshot data
        snapshot = {}
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol] if 'symbol' in data.columns else data
            
            if not symbol_data.empty:
                latest = symbol_data.iloc[-1]
                
                # Calculate change
                if len(symbol_data) > 1:
                    prev_close = symbol_data.iloc[-2]['close']
                    change = latest['close'] - prev_close
                    change_pct = (change / prev_close) * 100
                else:
                    change = 0
                    change_pct = 0
                
                snapshot[symbol] = {
                    "price": latest['close'],
                    "open": latest['open'],
                    "high": latest['high'],
                    "low": latest['low'],
                    "volume": latest['volume'],
                    "change": change,
                    "change_percent": change_pct,
                    "timestamp": latest.get('timestamp')
                }
        
        return {
            "snapshot": snapshot,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market snapshot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers/health")
async def get_providers_health() -> Dict[str, Any]:
    """
    Get health status of all data providers
    
    Returns detailed health metrics for each provider
    """
    try:
        health_report = market_data_registry.get_health_report()
        
        response = {
            "providers": [],
            "summary": {
                "total": len(health_report),
                "connected": 0,
                "disconnected": 0,
                "errors": 0
            }
        }
        
        for name, health in health_report.items():
            provider_health = ProviderHealthResponse(
                provider_name=health.provider_name,
                status=health.status.value,
                uptime_percentage=health.uptime_percentage,
                avg_latency_ms=health.avg_latency_ms,
                error_rate=health.error_rate,
                quality_score=health.quality_score,
                last_success=health.last_success
            )
            
            response["providers"].append(provider_health.dict())
            
            # Update summary
            if health.status == ProviderStatus.CONNECTED:
                response["summary"]["connected"] += 1
            elif health.status == ProviderStatus.DISCONNECTED:
                response["summary"]["disconnected"] += 1
            elif health.status == ProviderStatus.ERROR:
                response["summary"]["errors"] += 1
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting provider health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/providers/{provider_name}/connect")
async def connect_provider(provider_name: str) -> Dict[str, Any]:
    """
    Connect to a specific data provider
    """
    try:
        provider = market_data_registry.get_provider(provider_name)
        
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        success = await provider.connect()
        
        return {
            "provider": provider_name,
            "connected": success,
            "status": provider.status.value
        }
        
    except Exception as e:
        logger.error(f"Error connecting provider: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/providers/{provider_name}/disconnect")
async def disconnect_provider(provider_name: str) -> Dict[str, Any]:
    """
    Disconnect from a specific data provider
    """
    try:
        provider = market_data_registry.get_provider(provider_name)
        
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        success = await provider.disconnect()
        
        return {
            "provider": provider_name,
            "disconnected": success,
            "status": provider.status.value
        }
        
    except Exception as e:
        logger.error(f"Error disconnecting provider: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/providers/primary")
async def set_primary_provider(provider_name: str) -> Dict[str, Any]:
    """
    Set the primary data provider
    """
    try:
        provider = market_data_registry.get_provider(provider_name)
        
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
        
        market_data_registry.set_primary_provider(provider_name)
        
        return {
            "primary_provider": provider_name,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error setting primary provider: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics for the market data service
    """
    try:
        return market_data_aggregator.get_performance_report()
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/stream/{symbol}")
async def stream_market_data(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for streaming real-time market data
    """
    await websocket.accept()
    client_id = f"ws_{datetime.now().timestamp()}"
    
    try:
        logger.info(f"WebSocket client {client_id} connected for {symbol}")
        
        # Define callback for streaming data
        async def send_data(response):
            try:
                # Convert DataFrame to dict
                data = response.data.to_dict(orient="records") if not response.data.empty else []
                
                await websocket.send_json({
                    "type": "data",
                    "symbol": symbol,
                    "data": data,
                    "provider": response.provider,
                    "timestamp": response.timestamp.isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending WebSocket data: {str(e)}")
        
        # Start streaming
        await market_data_aggregator.stream_realtime(
            symbols=[symbol],
            callback=send_data
        )
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()


@router.get("/supported-symbols")
async def get_supported_symbols(
    provider: Optional[str] = Query(None, description="Specific provider")
) -> Dict[str, Any]:
    """
    Get list of supported symbols
    """
    try:
        if provider:
            provider_obj = market_data_registry.get_provider(provider)
            if not provider_obj:
                raise HTTPException(status_code=404, detail=f"Provider {provider} not found")
            
            symbols = await provider_obj.get_supported_symbols()
            
            return {
                "provider": provider,
                "symbols": symbols,
                "count": len(symbols)
            }
        else:
            # Get from all providers
            all_symbols = set()
            providers_symbols = {}
            
            for name, provider_obj in market_data_registry.get_all_providers().items():
                try:
                    symbols = await provider_obj.get_supported_symbols()
                    providers_symbols[name] = symbols
                    all_symbols.update(symbols)
                except:
                    providers_symbols[name] = []
            
            return {
                "all_symbols": sorted(list(all_symbols)),
                "by_provider": providers_symbols,
                "total_count": len(all_symbols)
            }
            
    except Exception as e:
        logger.error(f"Error getting supported symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))