"""
Polygon.io Market Data Provider
High-quality real-time and historical market data
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncIterator
import pandas as pd
import aiohttp
from urllib.parse import urljoin

from core.market_data.base import (
    IMarketDataProvider,
    ProviderConfig,
    MarketDataContext,
    MarketDataResponse,
    DataProviderType,
    DataQuality,
    ProviderStatus
)
from core.logging import get_logger

logger = get_logger(__name__)


class PolygonMarketDataProvider(IMarketDataProvider):
    """
    Polygon.io data provider implementation
    Premium market data with extensive coverage
    """
    
    def __init__(self, config: ProviderConfig = None):
        if not config:
            config = ProviderConfig(
                name="Polygon",
                type=DataProviderType.REAL_TIME,
                base_url="https://api.polygon.io",
                rate_limit=5,  # Free tier limit
                supported_markets=["stocks", "etf", "crypto", "forex", "options"],
                supported_data_types=["ohlcv", "trades", "quotes", "aggregates", "snapshots"],
                priority=2,
                cost_per_request=0.001  # Approximate cost
            )
        
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connection = None
        
    async def connect(self) -> bool:
        """Establish connection to Polygon.io"""
        try:
            if not self.config.api_key:
                logger.error("API key required for Polygon.io")
                self.status = ProviderStatus.ERROR
                return False
            
            # Create HTTP session
            self._session = aiohttp.ClientSession()
            
            # Test connection with reference data endpoint
            test_url = f"{self.config.base_url}/v3/reference/tickers"
            params = {"apiKey": self.config.api_key, "limit": 1}
            
            async with self._session.get(test_url, params=params) as response:
                if response.status == 200:
                    self.status = ProviderStatus.CONNECTED
                    logger.info("Connected to Polygon.io")
                    return True
                else:
                    self.status = ProviderStatus.ERROR
                    error_data = await response.json()
                    logger.error(f"Failed to connect to Polygon: {error_data}")
                    return False
                    
        except Exception as e:
            self.status = ProviderStatus.ERROR
            logger.error(f"Error connecting to Polygon: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Polygon.io"""
        try:
            if self._ws_connection:
                await self._ws_connection.close()
                self._ws_connection = None
            
            if self._session:
                await self._session.close()
                self._session = None
            
            self.status = ProviderStatus.DISCONNECTED
            logger.info("Disconnected from Polygon.io")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Polygon: {str(e)}")
            return False
    
    async def fetch_ohlcv(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch OHLCV aggregates from Polygon"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.connect()
            
            all_data = []
            
            for symbol in context.symbols:
                # Build aggregates endpoint
                multiplier, timespan = self._parse_timeframe(context.timeframe)
                
                url = f"{self.config.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}"
                
                params = {
                    "from": context.start_date.strftime("%Y-%m-%d") if context.start_date else "2023-01-01",
                    "to": context.end_date.strftime("%Y-%m-%d") if context.end_date else datetime.now().strftime("%Y-%m-%d"),
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 5000,
                    "apiKey": self.config.api_key
                }
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") == "OK" and data.get("results"):
                            df = pd.DataFrame(data["results"])
                            df['symbol'] = symbol
                            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                            df.rename(columns={
                                'o': 'open',
                                'h': 'high',
                                'l': 'low',
                                'c': 'close',
                                'v': 'volume',
                                'vw': 'vwap',
                                'n': 'transactions'
                            }, inplace=True)
                            all_data.append(df)
                    else:
                        logger.error(f"Failed to fetch data for {symbol}: {response.status}")
            
            # Combine all symbol data
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
            else:
                combined_df = pd.DataFrame()
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return MarketDataResponse(
                provider=self.config.name,
                timestamp=datetime.now(),
                data=combined_df,
                metadata={
                    "symbols": context.symbols,
                    "timeframe": context.timeframe,
                    "adjusted": True
                },
                quality=context.quality,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV from Polygon: {str(e)}")
            raise
    
    async def fetch_quotes(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch NBBO quotes from Polygon"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.connect()
            
            all_quotes = []
            
            for symbol in context.symbols:
                # Get latest quote
                url = f"{self.config.base_url}/v3/quotes/{symbol}"
                params = {
                    "apiKey": self.config.api_key,
                    "limit": 1,
                    "order": "desc"
                }
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") == "OK" and data.get("results"):
                            quote = data["results"][0]
                            quote_df = pd.DataFrame([{
                                'symbol': symbol,
                                'timestamp': pd.to_datetime(quote['participant_timestamp'], unit='ns'),
                                'bid_price': quote.get('bid_price'),
                                'bid_size': quote.get('bid_size'),
                                'ask_price': quote.get('ask_price'),
                                'ask_size': quote.get('ask_size'),
                                'bid_exchange': quote.get('bid_exchange'),
                                'ask_exchange': quote.get('ask_exchange'),
                                'conditions': quote.get('conditions', [])
                            }])
                            all_quotes.append(quote_df)
            
            if all_quotes:
                combined_df = pd.concat(all_quotes, ignore_index=True)
            else:
                combined_df = pd.DataFrame()
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return MarketDataResponse(
                provider=self.config.name,
                timestamp=datetime.now(),
                data=combined_df,
                metadata={
                    "symbols": context.symbols,
                    "quote_type": "nbbo",
                    "sip": True  # Polygon provides SIP data
                },
                quality=DataQuality.TICK,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Error fetching quotes from Polygon: {str(e)}")
            raise
    
    async def fetch_trades(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch trade data from Polygon"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.connect()
            
            all_trades = []
            
            for symbol in context.symbols:
                url = f"{self.config.base_url}/v3/trades/{symbol}"
                
                params = {
                    "apiKey": self.config.api_key,
                    "limit": 1000,
                    "order": "desc"
                }
                
                if context.start_date:
                    params["timestamp.gte"] = int(context.start_date.timestamp() * 1e9)
                if context.end_date:
                    params["timestamp.lte"] = int(context.end_date.timestamp() * 1e9)
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") == "OK" and data.get("results"):
                            trades_df = pd.DataFrame(data["results"])
                            trades_df['symbol'] = symbol
                            trades_df['timestamp'] = pd.to_datetime(trades_df['participant_timestamp'], unit='ns')
                            trades_df.rename(columns={
                                'price': 'price',
                                'size': 'size',
                                'exchange': 'exchange',
                                'conditions': 'conditions'
                            }, inplace=True)
                            all_trades.append(trades_df)
            
            if all_trades:
                combined_df = pd.concat(all_trades, ignore_index=True)
            else:
                combined_df = pd.DataFrame()
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return MarketDataResponse(
                provider=self.config.name,
                timestamp=datetime.now(),
                data=combined_df,
                metadata={
                    "symbols": context.symbols,
                    "trade_count": len(combined_df),
                    "sip": True
                },
                quality=DataQuality.TICK,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Error fetching trades from Polygon: {str(e)}")
            raise
    
    async def stream_data(self, context: MarketDataContext) -> AsyncIterator[MarketDataResponse]:
        """Stream real-time data from Polygon WebSocket"""
        try:
            ws_url = "wss://socket.polygon.io"
            
            # Determine feed based on asset class
            feed = "stocks" if "stocks" in context.metadata.get("asset_class", "stocks") else "crypto"
            ws_url = f"{ws_url}/{feed}"
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    # Authenticate
                    auth_msg = {
                        "action": "auth",
                        "params": self.config.api_key
                    }
                    await ws.send_json(auth_msg)
                    
                    # Wait for auth confirmation
                    auth_response = await ws.receive_json()
                    if auth_response[0]["status"] != "auth_success":
                        raise Exception("Authentication failed")
                    
                    # Subscribe to symbols
                    subscriptions = []
                    for symbol in context.symbols:
                        subscriptions.extend([
                            f"T.{symbol}",    # Trades
                            f"Q.{symbol}",    # Quotes
                            f"A.{symbol}",    # Aggregates (1 second)
                            f"AM.{symbol}"    # Aggregates (1 minute)
                        ])
                    
                    subscribe_msg = {
                        "action": "subscribe",
                        "params": ",".join(subscriptions)
                    }
                    await ws.send_json(subscribe_msg)
                    
                    # Stream data
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.json()
                            
                            for item in data:
                                df = self._parse_polygon_stream(item)
                                if not df.empty:
                                    yield MarketDataResponse(
                                        provider=self.config.name,
                                        timestamp=datetime.now(),
                                        data=df,
                                        metadata={
                                            "event_type": item.get("ev"),
                                            "symbol": item.get("sym")
                                        },
                                        quality=DataQuality.TICK,
                                        latency_ms=self._calculate_stream_latency(item)
                                    )
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                            
        except Exception as e:
            logger.error(f"Error streaming from Polygon: {str(e)}")
            raise
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from Polygon"""
        try:
            if not self._session:
                await self.connect()
            
            url = f"{self.config.base_url}/v3/reference/tickers"
            params = {
                "apiKey": self.config.api_key,
                "active": "true",
                "limit": 1000,
                "order": "asc",
                "sort": "ticker"
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK" and data.get("results"):
                        symbols = [ticker["ticker"] for ticker in data["results"]]
                        return symbols
                else:
                    logger.error(f"Failed to fetch tickers: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching supported symbols: {str(e)}")
            return []
    
    async def get_market_snapshot(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market snapshot for multiple symbols"""
        try:
            if not self._session:
                await self.connect()
            
            # Use grouped daily bars endpoint for efficiency
            url = f"{self.config.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
            params = {
                "apiKey": self.config.api_key,
                "tickers": ",".join(symbols[:100])  # Limit to 100 symbols
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("tickers", {})
                else:
                    logger.error(f"Failed to fetch snapshot: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching market snapshot: {str(e)}")
            return {}
    
    def _parse_timeframe(self, timeframe: str) -> tuple:
        """Parse timeframe to Polygon format (multiplier, timespan)"""
        mappings = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "1h": (1, "hour"),
            "4h": (4, "hour"),
            "1d": (1, "day"),
            "1w": (1, "week"),
            "1M": (1, "month")
        }
        return mappings.get(timeframe, (1, "day"))
    
    def _parse_polygon_stream(self, message: Dict[str, Any]) -> pd.DataFrame:
        """Parse Polygon WebSocket stream message"""
        event_type = message.get("ev")
        
        if event_type == "T":  # Trade
            return pd.DataFrame([{
                'symbol': message.get('sym'),
                'timestamp': pd.to_datetime(message.get('t'), unit='ms'),
                'price': message.get('p'),
                'size': message.get('s'),
                'exchange': message.get('x'),
                'conditions': message.get('c', []),
                'type': 'trade'
            }])
        elif event_type == "Q":  # Quote
            return pd.DataFrame([{
                'symbol': message.get('sym'),
                'timestamp': pd.to_datetime(message.get('t'), unit='ms'),
                'bid_price': message.get('bp'),
                'bid_size': message.get('bs'),
                'ask_price': message.get('ap'),
                'ask_size': message.get('as'),
                'bid_exchange': message.get('bx'),
                'ask_exchange': message.get('ax'),
                'type': 'quote'
            }])
        elif event_type in ["A", "AM"]:  # Aggregate
            return pd.DataFrame([{
                'symbol': message.get('sym'),
                'timestamp': pd.to_datetime(message.get('s'), unit='ms'),
                'open': message.get('o'),
                'high': message.get('h'),
                'low': message.get('l'),
                'close': message.get('c'),
                'volume': message.get('v'),
                'vwap': message.get('vw'),
                'type': 'aggregate'
            }])
        else:
            return pd.DataFrame()
    
    def _calculate_stream_latency(self, message: Dict[str, Any]) -> float:
        """Calculate streaming data latency"""
        try:
            # Message timestamp
            msg_time = message.get('t') or message.get('s')
            if msg_time:
                msg_datetime = datetime.fromtimestamp(msg_time / 1000)
                latency = (datetime.now() - msg_datetime).total_seconds() * 1000
                return max(0, latency)  # Ensure non-negative
        except:
            pass
        return 0