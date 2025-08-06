"""
Unified Market Data Service
Consolidates all market data fetching into a single, robust service
Supports multiple data providers with automatic fallback
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from cachetools import TTLCache
import aiohttp

from core.config import get_settings
from core.logging import get_logger
from core.cache import get_cache_manager

logger = get_logger(__name__)
settings = get_settings()


class DataProvider(Enum):
    """Available market data providers"""
    POLYGON = "polygon"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    TWELVE_DATA = "twelve_data"
    FMP = "fmp"
    YAHOO = "yahoo"


class UnifiedMarketDataService:
    """
    Unified service for all market data operations
    Consolidates functionality from multiple duplicate services
    """
    
    def __init__(self):
        # Provider API keys
        self.api_keys = {
            DataProvider.POLYGON: settings.POLYGON_API_KEY,
            DataProvider.ALPHA_VANTAGE: settings.ALPHA_VANTAGE_API_KEY,
            DataProvider.FINNHUB: settings.FINNHUB_API_KEY,
            DataProvider.TWELVE_DATA: settings.TWELVEDATA_API_KEY,
            DataProvider.FMP: settings.FMP_API_KEY
        }
        
        # Provider endpoints
        self.endpoints = {
            DataProvider.POLYGON: "https://api.polygon.io",
            DataProvider.ALPHA_VANTAGE: "https://www.alphavantage.co/query",
            DataProvider.FINNHUB: "https://finnhub.io/api/v1",
            DataProvider.TWELVE_DATA: "https://api.twelvedata.com",
            DataProvider.FMP: "https://financialmodelingprep.com/api/v3",
            DataProvider.YAHOO: "https://query2.finance.yahoo.com"
        }
        
        # Cache configuration
        self.cache_manager = get_cache_manager()
        self.local_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min local cache
        
        # Provider priority (order of fallback)
        self.provider_priority = [
            DataProvider.POLYGON,
            DataProvider.TWELVE_DATA,
            DataProvider.FINNHUB,
            DataProvider.ALPHA_VANTAGE,
            DataProvider.FMP,
            DataProvider.YAHOO  # Free fallback
        ]
        
        # Rate limiting
        self.rate_limits = {
            DataProvider.POLYGON: {"calls": 5, "period": 1},  # 5 calls/sec
            DataProvider.ALPHA_VANTAGE: {"calls": 5, "period": 60},  # 5 calls/min
            DataProvider.FINNHUB: {"calls": 30, "period": 1},  # 30 calls/sec
            DataProvider.TWELVE_DATA: {"calls": 8, "period": 60},  # 8 calls/min
            DataProvider.FMP: {"calls": 300, "period": 60},  # 300 calls/min
            DataProvider.YAHOO: {"calls": 100, "period": 60}  # No official limit
        }
        
        self.last_calls = {provider: [] for provider in DataProvider}
        
        logger.info("Unified Market Data Service initialized")
    
    async def get_quote(self, symbol: str, provider: Optional[DataProvider] = None) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            provider: Optional specific provider to use
            
        Returns:
            Quote data dictionary
        """
        cache_key = f"quote:{symbol}"
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Try providers in order
        providers = [provider] if provider else self.provider_priority
        
        for prov in providers:
            if not self._can_use_provider(prov):
                continue
            
            try:
                quote = await self._fetch_quote(symbol, prov)
                if quote:
                    # Normalize data
                    normalized = self._normalize_quote(quote, prov)
                    
                    # Cache result
                    await self._cache_result(cache_key, normalized, ttl=60)
                    
                    return normalized
            except Exception as e:
                logger.warning(f"Provider {prov.value} failed for quote: {e}")
                continue
        
        logger.error(f"All providers failed for quote: {symbol}")
        return {}
    
    async def _fetch_quote(self, symbol: str, provider: DataProvider) -> Dict[str, Any]:
        """Fetch quote from specific provider"""
        await self._rate_limit(provider)
        
        async with aiohttp.ClientSession() as session:
            if provider == DataProvider.POLYGON:
                url = f"{self.endpoints[provider]}/v2/aggs/ticker/{symbol}/prev"
                params = {"apiKey": self.api_keys[provider]}
                
            elif provider == DataProvider.FINNHUB:
                url = f"{self.endpoints[provider]}/quote"
                params = {"symbol": symbol, "token": self.api_keys[provider]}
                
            elif provider == DataProvider.TWELVE_DATA:
                url = f"{self.endpoints[provider]}/quote"
                params = {"symbol": symbol, "apikey": self.api_keys[provider]}
                
            elif provider == DataProvider.ALPHA_VANTAGE:
                url = self.endpoints[provider]
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.api_keys[provider]
                }
                
            elif provider == DataProvider.FMP:
                url = f"{self.endpoints[provider]}/quote/{symbol}"
                params = {"apikey": self.api_keys[provider]}
                
            elif provider == DataProvider.YAHOO:
                url = f"{self.endpoints[provider]}/v8/finance/quote"
                params = {"symbols": symbol}
            
            else:
                return {}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"{provider.value} returned {response.status}")
                    return {}
    
    def _normalize_quote(self, data: Dict[str, Any], provider: DataProvider) -> Dict[str, Any]:
        """Normalize quote data from different providers"""
        normalized = {
            "symbol": "",
            "price": 0.0,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "volume": 0,
            "timestamp": datetime.now().isoformat(),
            "change": 0.0,
            "change_percent": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "bid_size": 0,
            "ask_size": 0,
            "market_cap": 0,
            "pe_ratio": 0.0,
            "provider": provider.value
        }
        
        try:
            if provider == DataProvider.POLYGON:
                if "results" in data and data["results"]:
                    result = data["results"][0]
                    normalized.update({
                        "symbol": result.get("T", ""),
                        "open": result.get("o", 0),
                        "high": result.get("h", 0),
                        "low": result.get("l", 0),
                        "close": result.get("c", 0),
                        "price": result.get("c", 0),
                        "volume": result.get("v", 0)
                    })
            
            elif provider == DataProvider.FINNHUB:
                normalized.update({
                    "price": data.get("c", 0),
                    "open": data.get("o", 0),
                    "high": data.get("h", 0),
                    "low": data.get("l", 0),
                    "close": data.get("pc", 0),
                    "change": data.get("d", 0),
                    "change_percent": data.get("dp", 0)
                })
            
            elif provider == DataProvider.TWELVE_DATA:
                normalized.update({
                    "symbol": data.get("symbol", ""),
                    "price": float(data.get("close", 0)),
                    "open": float(data.get("open", 0)),
                    "high": float(data.get("high", 0)),
                    "low": float(data.get("low", 0)),
                    "close": float(data.get("previous_close", 0)),
                    "volume": int(data.get("volume", 0)),
                    "change": float(data.get("change", 0)),
                    "change_percent": float(data.get("percent_change", 0))
                })
            
            elif provider == DataProvider.ALPHA_VANTAGE:
                if "Global Quote" in data:
                    quote = data["Global Quote"]
                    normalized.update({
                        "symbol": quote.get("01. symbol", ""),
                        "price": float(quote.get("05. price", 0)),
                        "open": float(quote.get("02. open", 0)),
                        "high": float(quote.get("03. high", 0)),
                        "low": float(quote.get("04. low", 0)),
                        "close": float(quote.get("08. previous close", 0)),
                        "volume": int(quote.get("06. volume", 0)),
                        "change": float(quote.get("09. change", 0)),
                        "change_percent": float(quote.get("10. change percent", "0").replace("%", ""))
                    })
            
            elif provider == DataProvider.FMP:
                if data and isinstance(data, list) and len(data) > 0:
                    quote = data[0]
                    normalized.update({
                        "symbol": quote.get("symbol", ""),
                        "price": quote.get("price", 0),
                        "open": quote.get("open", 0),
                        "high": quote.get("dayHigh", 0),
                        "low": quote.get("dayLow", 0),
                        "close": quote.get("previousClose", 0),
                        "volume": quote.get("volume", 0),
                        "change": quote.get("change", 0),
                        "change_percent": quote.get("changesPercentage", 0),
                        "market_cap": quote.get("marketCap", 0),
                        "pe_ratio": quote.get("pe", 0)
                    })
            
            elif provider == DataProvider.YAHOO:
                if "quoteResponse" in data and "result" in data["quoteResponse"]:
                    if data["quoteResponse"]["result"]:
                        quote = data["quoteResponse"]["result"][0]
                        normalized.update({
                            "symbol": quote.get("symbol", ""),
                            "price": quote.get("regularMarketPrice", 0),
                            "open": quote.get("regularMarketOpen", 0),
                            "high": quote.get("regularMarketDayHigh", 0),
                            "low": quote.get("regularMarketDayLow", 0),
                            "close": quote.get("regularMarketPreviousClose", 0),
                            "volume": quote.get("regularMarketVolume", 0),
                            "change": quote.get("regularMarketChange", 0),
                            "change_percent": quote.get("regularMarketChangePercent", 0),
                            "bid": quote.get("bid", 0),
                            "ask": quote.get("ask", 0),
                            "bid_size": quote.get("bidSize", 0),
                            "ask_size": quote.get("askSize", 0),
                            "market_cap": quote.get("marketCap", 0),
                            "pe_ratio": quote.get("trailingPE", 0)
                        })
        
        except Exception as e:
            logger.error(f"Error normalizing {provider.value} data: {e}")
        
        return normalized
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
        provider: Optional[DataProvider] = None
    ) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Stock symbol
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
            interval: Time interval (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1mo)
            provider: Optional specific provider
            
        Returns:
            DataFrame with OHLCV data
        """
        # Default dates
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=365)
        
        cache_key = f"historical:{symbol}:{start_date.date()}:{end_date.date()}:{interval}"
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return pd.DataFrame(cached)
        
        # Try providers
        providers = [provider] if provider else self.provider_priority
        
        for prov in providers:
            if not self._can_use_provider(prov):
                continue
            
            try:
                data = await self._fetch_historical(symbol, start_date, end_date, interval, prov)
                if data is not None and not data.empty:
                    # Cache as dict for serialization
                    await self._cache_result(cache_key, data.to_dict(), ttl=3600)
                    return data
            except Exception as e:
                logger.warning(f"Provider {prov.value} failed for historical: {e}")
                continue
        
        logger.error(f"All providers failed for historical data: {symbol}")
        return pd.DataFrame()
    
    async def _fetch_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        provider: DataProvider
    ) -> pd.DataFrame:
        """Fetch historical data from specific provider"""
        await self._rate_limit(provider)
        
        async with aiohttp.ClientSession() as session:
            if provider == DataProvider.POLYGON:
                # Convert interval to Polygon format
                multiplier, timespan = self._parse_interval_polygon(interval)
                url = f"{self.endpoints[provider]}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.date()}/{end_date.date()}"
                params = {"apiKey": self.api_keys[provider], "sort": "asc"}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "results" in data:
                            df = pd.DataFrame(data["results"])
                            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                            df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
                            df.set_index("timestamp", inplace=True)
                            return df[["open", "high", "low", "close", "volume"]]
            
            elif provider == DataProvider.TWELVE_DATA:
                url = f"{self.endpoints[provider]}/time_series"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "apikey": self.api_keys[provider]
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "values" in data:
                            df = pd.DataFrame(data["values"])
                            df["datetime"] = pd.to_datetime(df["datetime"])
                            df.set_index("datetime", inplace=True)
                            for col in ["open", "high", "low", "close", "volume"]:
                                df[col] = pd.to_numeric(df[col])
                            return df[["open", "high", "low", "close", "volume"]]
            
            elif provider == DataProvider.YAHOO:
                # Use yfinance for Yahoo data
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    return df[["open", "high", "low", "close", "volume"]]
        
        return pd.DataFrame()
    
    def _parse_interval_polygon(self, interval: str) -> Tuple[int, str]:
        """Parse interval for Polygon API"""
        mappings = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "1h": (1, "hour"),
            "1d": (1, "day"),
            "1w": (1, "week"),
            "1mo": (1, "month")
        }
        return mappings.get(interval, (1, "day"))
    
    async def get_comprehensive_data(
        self,
        symbol: str,
        include_news: bool = True,
        include_options: bool = True,
        include_fundamentals: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive market data for analysis
        
        Args:
            symbol: Stock symbol
            include_news: Include news sentiment
            include_options: Include options data
            include_fundamentals: Include fundamental data
            
        Returns:
            Comprehensive data dictionary
        """
        tasks = []
        
        # Basic quote
        tasks.append(self.get_quote(symbol))
        
        # Historical data
        tasks.append(self.get_historical_data(symbol, interval="1d"))
        tasks.append(self.get_historical_data(symbol, interval="1h", 
                                             start_date=datetime.now() - timedelta(days=5)))
        
        # Technical indicators
        tasks.append(self.get_technical_indicators(symbol))
        
        # Optional data
        if include_news:
            tasks.append(self.get_news_sentiment(symbol))
        
        if include_options:
            tasks.append(self.get_options_flow(symbol))
        
        if include_fundamentals:
            tasks.append(self.get_fundamentals(symbol))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        comprehensive_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "quote": results[0] if not isinstance(results[0], Exception) else {},
            "daily_history": results[1].to_dict() if not isinstance(results[1], Exception) and not results[1].empty else {},
            "hourly_history": results[2].to_dict() if not isinstance(results[2], Exception) and not results[2].empty else {},
            "technical": results[3] if not isinstance(results[3], Exception) else {},
        }
        
        idx = 4
        if include_news:
            comprehensive_data["news"] = results[idx] if not isinstance(results[idx], Exception) else {}
            idx += 1
        
        if include_options:
            comprehensive_data["options"] = results[idx] if not isinstance(results[idx], Exception) else {}
            idx += 1
        
        if include_fundamentals:
            comprehensive_data["fundamentals"] = results[idx] if not isinstance(results[idx], Exception) else {}
        
        return comprehensive_data
    
    async def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            # Get historical data
            df = await self.get_historical_data(symbol, interval="1d")
            
            if df.empty:
                return {}
            
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            indicators["sma_20"] = df["close"].rolling(window=20).mean().iloc[-1]
            indicators["sma_50"] = df["close"].rolling(window=50).mean().iloc[-1]
            indicators["sma_200"] = df["close"].rolling(window=200).mean().iloc[-1]
            indicators["ema_12"] = df["close"].ewm(span=12).mean().iloc[-1]
            indicators["ema_26"] = df["close"].ewm(span=26).mean().iloc[-1]
            
            # MACD
            macd_line = indicators["ema_12"] - indicators["ema_26"]
            indicators["macd"] = macd_line
            indicators["macd_signal"] = df["close"].ewm(span=9).mean().iloc[-1]
            indicators["macd_histogram"] = macd_line - indicators["macd_signal"]
            
            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators["rsi"] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Bollinger Bands
            sma = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            indicators["bb_upper"] = (sma + (std * 2)).iloc[-1]
            indicators["bb_middle"] = sma.iloc[-1]
            indicators["bb_lower"] = (sma - (std * 2)).iloc[-1]
            
            # ATR (Average True Range)
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators["atr"] = true_range.rolling(window=14).mean().iloc[-1]
            
            # Volume indicators
            indicators["volume_sma"] = df["volume"].rolling(window=20).mean().iloc[-1]
            indicators["volume_ratio"] = df["volume"].iloc[-1] / indicators["volume_sma"]
            
            # Volatility
            returns = df["close"].pct_change()
            indicators["volatility"] = returns.std() * np.sqrt(252)  # Annualized
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news and sentiment analysis"""
        # Placeholder - would integrate with news APIs
        return {
            "sentiment_score": 0.65,
            "sentiment_label": "positive",
            "news_volume": 15,
            "top_headlines": []
        }
    
    async def get_options_flow(self, symbol: str) -> Dict[str, Any]:
        """Get options flow data"""
        # Placeholder - would integrate with options data providers
        return {
            "put_call_ratio": 0.75,
            "unusual_activity": False,
            "largest_trades": []
        }
    
    async def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data"""
        # Placeholder - would integrate with fundamental data providers
        return {
            "pe_ratio": 25.5,
            "eps": 5.25,
            "market_cap": 1500000000,
            "dividend_yield": 0.015
        }
    
    def _can_use_provider(self, provider: DataProvider) -> bool:
        """Check if provider is available and has API key"""
        if provider == DataProvider.YAHOO:
            return True  # No API key needed
        
        return bool(self.api_keys.get(provider))
    
    async def _rate_limit(self, provider: DataProvider):
        """Apply rate limiting for provider"""
        if provider not in self.rate_limits:
            return
        
        limits = self.rate_limits[provider]
        now = datetime.now()
        
        # Clean old calls
        cutoff = now - timedelta(seconds=limits["period"])
        self.last_calls[provider] = [
            call for call in self.last_calls[provider]
            if call > cutoff
        ]
        
        # Check if at limit
        if len(self.last_calls[provider]) >= limits["calls"]:
            # Calculate wait time
            oldest_call = min(self.last_calls[provider])
            wait_time = (oldest_call + timedelta(seconds=limits["period"]) - now).total_seconds()
            
            if wait_time > 0:
                logger.debug(f"Rate limiting {provider.value}: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.last_calls[provider].append(now)
    
    async def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data"""
        # Check local cache first
        if key in self.local_cache:
            return self.local_cache[key]
        
        # Check distributed cache
        try:
            cached = await self.cache_manager.get(key)
            if cached:
                self.local_cache[key] = cached  # Update local cache
            return cached
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_result(self, key: str, data: Any, ttl: int = 300):
        """Cache result"""
        # Update local cache
        self.local_cache[key] = data
        
        # Update distributed cache
        try:
            await self.cache_manager.set(key, data, ttl=ttl)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        
        for provider in DataProvider:
            status[provider.value] = {
                "available": self._can_use_provider(provider),
                "has_key": bool(self.api_keys.get(provider)),
                "rate_limit": self.rate_limits.get(provider, {}),
                "recent_calls": len(self.last_calls.get(provider, []))
            }
        
        return status


# Singleton instance
unified_market_service = UnifiedMarketDataService()