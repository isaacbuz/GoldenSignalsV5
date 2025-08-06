"""
Market data Pydantic schemas for validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TimeFrame(str, Enum):
    """Valid timeframes for historical data"""
    ONE_MIN = "1m"
    TWO_MIN = "2m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    SIXTY_MIN = "60m"
    NINETY_MIN = "90m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAY = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTH = "3mo"


class Period(str, Enum):
    """Valid periods for historical data"""
    ONE_DAY = "1d"
    FIVE_DAY = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTH = "3mo"
    SIX_MONTH = "6mo"
    ONE_YEAR = "1y"
    TWO_YEAR = "2y"
    FIVE_YEAR = "5y"
    TEN_YEAR = "10y"
    YTD = "ytd"
    MAX = "max"


class QuoteRequest(BaseModel):
    """Request model for getting a quote"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()


class MultiQuoteRequest(BaseModel):
    """Request model for getting multiple quotes"""
    symbols: List[str] = Field(..., min_items=1, max_items=50, description="List of stock symbols")
    
    @validator('symbols')
    def uppercase_symbols(cls, v):
        return [s.upper() for s in v]


class HistoricalDataRequest(BaseModel):
    """Request model for historical data"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    period: Period = Field(Period.ONE_MONTH, description="Time period")
    interval: TimeFrame = Field(TimeFrame.ONE_DAY, description="Data interval")
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()


class SearchRequest(BaseModel):
    """Request model for symbol search"""
    query: str = Field(..., min_length=1, max_length=50, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum results")


class QuoteResponse(BaseModel):
    """Response model for quote data"""
    symbol: str
    price: float
    previousClose: float
    change: float
    changePercent: float
    volume: int
    marketCap: Optional[float] = None
    dayHigh: Optional[float] = None
    dayLow: Optional[float] = None
    fiftyTwoWeekHigh: Optional[float] = None
    fiftyTwoWeekLow: Optional[float] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "price": 150.25,
                "previousClose": 149.50,
                "change": 0.75,
                "changePercent": 0.5,
                "volume": 75000000,
                "marketCap": 2500000000000,
                "dayHigh": 151.00,
                "dayLow": 149.00,
                "fiftyTwoWeekHigh": 180.00,
                "fiftyTwoWeekLow": 120.00,
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class PriceBar(BaseModel):
    """Model for OHLCV price bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-01T09:30:00Z",
                "open": 150.00,
                "high": 151.50,
                "low": 149.50,
                "close": 150.25,
                "volume": 1000000
            }
        }


class HistoricalDataResponse(BaseModel):
    """Response model for historical data"""
    symbol: str
    period: str
    interval: str
    data: List[PriceBar]
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "period": "1mo",
                "interval": "1d",
                "data": [
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "open": 150.00,
                        "high": 151.50,
                        "low": 149.50,
                        "close": 150.25,
                        "volume": 75000000
                    }
                ]
            }
        }


class MarketStatus(BaseModel):
    """Market status model"""
    market_open: bool
    current_time: datetime
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    indices: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "market_open": True,
                "current_time": "2024-01-01T14:30:00Z",
                "next_open": "2024-01-02T09:30:00Z",
                "next_close": "2024-01-01T16:00:00Z",
                "indices": {
                    "S&P 500": {"price": 4500.00, "change": 25.00, "changePercent": 0.56},
                    "Dow Jones": {"price": 35000.00, "change": 150.00, "changePercent": 0.43},
                    "NASDAQ": {"price": 14000.00, "change": 75.00, "changePercent": 0.54}
                }
            }
        }


class SearchResult(BaseModel):
    """Search result model"""
    symbol: str
    name: str
    exchange: Optional[str] = None
    type: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "type": "EQUITY"
            }
        }


class SearchResponse(BaseModel):
    """Response model for symbol search"""
    query: str
    results: List[SearchResult]
    count: int
    
    class Config:
        schema_extra = {
            "example": {
                "query": "apple",
                "results": [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "exchange": "NASDAQ",
                        "type": "EQUITY"
                    }
                ],
                "count": 1
            }
        }