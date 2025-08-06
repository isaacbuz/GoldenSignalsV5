"""
Market Data API Routes
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from services.market_data_service import market_data_service
import pandas as pd
import numpy as np

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote for a symbol"""
    quote = await market_data_service.get_quote(symbol.upper())
    
    if not quote:
        raise HTTPException(status_code=404, detail=f"Quote not found for symbol: {symbol}")
    
    return quote


from pydantic import BaseModel

class QuotesRequest(BaseModel):
    symbols: List[str]

@router.post("/quotes")
async def get_quotes(request: QuotesRequest):
    """Get quotes for multiple symbols"""
    symbols = request.symbols
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")
    
    if len(symbols) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 symbols allowed per request")
    
    # Convert to uppercase
    symbols = [s.upper() for s in symbols]
    
    quotes = await market_data_service.get_quotes(symbols)
    return quotes


@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    period: str = Query("1d", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"),
    interval: str = Query("1m", description="Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
):
    """Get historical price data for a symbol"""
    hist_data = await market_data_service.get_historical_data(symbol.upper(), period, interval)
    
    if hist_data is None or hist_data.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for symbol: {symbol}")
    
    # Convert DataFrame to dict for JSON response
    # Clean data: Replace NaN/inf values with None for JSON serialization
    hist_data = hist_data.replace([np.inf, -np.inf], np.nan)
    hist_data = hist_data.where(pd.notnull(hist_data), None)
    
    # Convert to records and clean
    data_records = []
    for record in hist_data.to_dict('records'):
        # Ensure all numeric fields are clean
        clean_record = {}
        for key, value in record.items():
            if isinstance(value, (float, np.float64)) and (np.isnan(value) or np.isinf(value)):
                clean_record[key] = None
            else:
                clean_record[key] = value
        data_records.append(clean_record)
    
    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "data": data_records
    }


@router.get("/market-status")
async def get_market_status():
    """Get current market status and major indices"""
    return await market_data_service.get_market_status()

@router.get("/status")
async def get_status():
    """Alias for market-status endpoint"""
    return await market_data_service.get_market_status()


@router.get("/search")
async def search_symbols(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return")
):
    """Search for symbols by name or ticker"""
    if not query or len(query) < 1:
        raise HTTPException(status_code=400, detail="Search query must be at least 1 character")
    
    results = await market_data_service.search_symbols(query, limit)
    return {"query": query, "results": results, "count": len(results)}


@router.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    return market_data_service.get_cache_stats()