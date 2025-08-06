"""
Market Data Service - Fetches and processes real-time market data

This service provides:
- Real-time quotes and historical data
- Technical indicator calculations
- Multi-tier caching
- Rate limiting
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import numpy as np
import yfinance as yf
from functools import lru_cache
import redis
import json

from core.config import settings
from services.data_simulator import simulator

logger = logging.getLogger(__name__)

# Constants
DEFAULT_QUOTE_TTL = 300  # 5 minutes
DEFAULT_HISTORICAL_TTL = 600  # 10 minutes
MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests
MAX_CACHE_SIZE = 1000

# Common symbols for demo/search
COMMON_SYMBOLS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla Inc."},
    {"symbol": "META", "name": "Meta Platforms Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "WMT", "name": "Walmart Inc."},
    {"symbol": "PG", "name": "Procter & Gamble Co."},
    {"symbol": "MA", "name": "Mastercard Inc."},
    {"symbol": "UNH", "name": "UnitedHealth Group Inc."},
    {"symbol": "HD", "name": "The Home Depot Inc."},
    {"symbol": "DIS", "name": "The Walt Disney Company"},
    {"symbol": "BAC", "name": "Bank of America Corp."},
    {"symbol": "ADBE", "name": "Adobe Inc."},
    {"symbol": "CRM", "name": "Salesforce Inc."},
    {"symbol": "NFLX", "name": "Netflix Inc."},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Average ETF"},
]


class MarketDataService:
    """Service for fetching and processing market data"""

    def __init__(self):
        # Redis connection for caching
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using in-memory cache only.")
            self.redis_client = None

        # In-memory cache as fallback
        self.memory_cache = {}
        
        # Rate limiting
        self.last_request_time: Dict[str, float] = {}
        self.min_request_interval = MIN_REQUEST_INTERVAL

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol"""
        try:
            # Check cache first
            cached_data = await self._get_from_cache(f"quote:{symbol}")
            if cached_data:
                return cached_data

            # Apply rate limiting
            await self._apply_rate_limit(symbol)

            # Try live data provider first
            try:
                from services.live_data_provider import live_data_provider
                live_quote = await live_data_provider.get_live_quote(symbol)
                
                if live_quote:
                    quote_data = {
                        "symbol": live_quote.symbol,
                        "price": live_quote.price,
                        "change": live_quote.change,
                        "changePercent": live_quote.change_percent,
                        "volume": live_quote.volume,
                        "bid": live_quote.bid,
                        "ask": live_quote.ask,
                        "dayHigh": live_quote.high,
                        "dayLow": live_quote.low,
                        "open": live_quote.open,
                        "previousClose": live_quote.previous_close,
                        "timestamp": live_quote.timestamp.isoformat(),
                        "source": live_quote.source
                    }
                    
                    # Cache the result
                    await self._set_cache(f"quote:{symbol}", quote_data, DEFAULT_QUOTE_TTL)
                    logger.debug(f"Quote for {symbol} from {live_quote.source}: ${quote_data.get('price', 0):.2f}")
                    return quote_data
            except Exception as live_error:
                logger.warning(f"Live data provider error for {symbol}: {live_error}, falling back to yfinance")
            
            # Try yfinance as fallback
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info and info.get("currentPrice") is not None:
                    quote_data = {
                        "symbol": symbol,
                        "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                        "change": info.get("regularMarketChange", 0),
                        "changePercent": info.get("regularMarketChangePercent", 0),
                        "volume": info.get("regularMarketVolume", 0),
                        "marketCap": info.get("marketCap", 0),
                        "dayHigh": info.get("dayHigh", 0),
                        "dayLow": info.get("dayLow", 0),
                        "open": info.get("open", 0),
                        "previousClose": info.get("previousClose", 0),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "yfinance"
                    }
                    
                    # Cache the result
                    await self._set_cache(f"quote:{symbol}", quote_data, DEFAULT_QUOTE_TTL)
                    logger.debug(f"Quote for {symbol} from yfinance: ${quote_data.get('price', 0):.2f}")
                    return quote_data
            except Exception as yf_error:
                logger.warning(f"YFinance error for {symbol}: {yf_error}, falling back to simulator")
            
            # Fall back to simulator for demo data
            quote_data = simulator.get_quote(symbol)
            
            # Cache the result
            await self._set_cache(f"quote:{symbol}", quote_data, DEFAULT_QUOTE_TTL)
            
            logger.debug(f"Quote for {symbol} from simulator: ${quote_data.get('price', 0):.2f}")
            return quote_data

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            # Last resort - return simulator data
            try:
                return simulator.get_quote(symbol)
            except:
                return None

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols with batch processing"""
        try:
            # Fetch quotes concurrently
            tasks = [self.get_quote(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            quotes = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {symbol}: {result}")
                    quotes[symbol] = None
                else:
                    quotes[symbol] = result
                    
            return quotes

        except Exception as e:
            logger.error(f"Error in batch quote fetch: {e}")
            return {}

    async def get_historical_data(
        self, symbol: str, period: str = "1d", interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """Get historical price data for a symbol"""
        cache_key = f"hist:{symbol}:{period}:{interval}"
        
        try:
            # Check cache
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                return pd.DataFrame(cached_data)

            # Apply rate limiting
            await self._apply_rate_limit(symbol)

            # Try yfinance first
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period=period, interval=interval)
                
                if not hist_data.empty:
                    # Add technical indicators
                    hist_data = self._add_technical_indicators(hist_data)
                    
                    # Cache the result (as dict for JSON serialization)
                    await self._set_cache(cache_key, hist_data.to_dict('records'), DEFAULT_HISTORICAL_TTL)
                    
                    logger.debug(f"Retrieved {len(hist_data)} records for {symbol} from yfinance")
                    return hist_data
            except Exception as yf_error:
                logger.warning(f"YFinance historical error for {symbol}: {yf_error}, falling back to simulator")
            
            # Fall back to simulator
            sim_data = simulator.get_historical_data(symbol, period, interval)
            
            # Convert to DataFrame
            df = pd.DataFrame(sim_data)
            if not df.empty:
                # Convert time to datetime index
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Add technical indicators
                df = self._add_technical_indicators(df)
                
                # Clean data before caching - replace NaN/inf with None
                df_clean = df.replace([np.inf, -np.inf], np.nan)
                df_clean = df_clean.where(pd.notnull(df_clean), None)
                
                # Cache the result
                await self._set_cache(cache_key, df_clean.reset_index().to_dict('records'), DEFAULT_HISTORICAL_TTL)
                
                logger.debug(f"Retrieved {len(df)} records for {symbol} from simulator")
                return df
            
            return None

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            # Last resort - return simulator data
            try:
                sim_data = simulator.get_historical_data(symbol, period, interval)
                df = pd.DataFrame(sim_data)
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    return df
            except:
                pass
            return None

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        if df.empty or len(df) < 20:
            return df

        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean() if len(df) >= 50 else None
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Replace NaN values with None for JSON serialization
            df = df.where(pd.notnull(df), None)
            
            return df

        except Exception as e:
            logger.warning(f"Could not add indicators: {e}")
            return df

    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and major indices"""
        try:
            now = datetime.now(timezone.utc)

            # NYSE market hours (EST/EDT)
            market_open = now.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 AM EST
            market_close = now.replace(hour=21, minute=0, second=0, microsecond=0)  # 4:00 PM EST

            is_weekday = now.weekday() < 5
            is_market_hours = market_open <= now <= market_close
            is_open = is_weekday and is_market_hours

            # Fetch major indices
            indices = await self.get_quotes(["SPY", "QQQ", "DIA", "IWM"])

            return {
                "is_open": is_open,
                "status": "open" if is_open else "closed",
                "current_time": now.isoformat(),
                "market_open": market_open.isoformat(),
                "market_close": market_close.isoformat(),
                "indices": indices,
            }

        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {"is_open": False, "status": "error"}

    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for symbols matching the query"""
        if not query:
            return []

        try:
            query_lower = query.lower()

            # Filter symbols by query
            results = [
                symbol_info
                for symbol_info in COMMON_SYMBOLS
                if query_lower in symbol_info["symbol"].lower()
                or query_lower in symbol_info["name"].lower()
            ]

            return results[:limit]

        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

    async def _apply_rate_limit(self, symbol: str) -> None:
        """Apply rate limiting for a symbol"""
        import time
        
        current_time = time.time()
        last_request = self.last_request_time.get(symbol, 0)
        
        time_since_last = current_time - last_request
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time[symbol] = time.time()

    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis first, then memory)"""
        try:
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            
            # Fall back to memory cache
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                if datetime.now() < expiry:
                    return value
                else:
                    del self.memory_cache[key]
                    
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            
        return None

    async def _set_cache(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL"""
        try:
            # Store in Redis
            if self.redis_client:
                self.redis_client.setex(key, ttl, json.dumps(value))
            
            # Also store in memory cache
            expiry = datetime.now() + pd.Timedelta(seconds=ttl)
            self.memory_cache[key] = (value, expiry)
            
            # Limit memory cache size
            if len(self.memory_cache) > MAX_CACHE_SIZE:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k][1]
                )[:len(self.memory_cache) - MAX_CACHE_SIZE]
                
                for k in oldest_keys:
                    del self.memory_cache[k]
                    
        except Exception as e:
            logger.debug(f"Cache set error: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "redis_available": self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_memory_used"] = info.get("used_memory_human", "N/A")
                stats["redis_connected_clients"] = info.get("connected_clients", 0)
            except:
                pass
                
        return stats


# Global service instance
market_data_service = MarketDataService()