"""
Alpaca Market Data Provider
Real-time and historical data from Alpaca Markets
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


class AlpacaMarketDataProvider(IMarketDataProvider):
    """
    Alpaca Markets data provider implementation
    Supports stocks, ETFs, and crypto
    """
    
    def __init__(self, config: ProviderConfig = None):
        if not config:
            config = ProviderConfig(
                name="Alpaca",
                type=DataProviderType.REAL_TIME,
                base_url="https://data.alpaca.markets/v2",
                rate_limit=200,  # Alpaca's rate limit
                supported_markets=["stocks", "etf", "crypto"],
                supported_data_types=["ohlcv", "trades", "quotes", "bars"],
                priority=1
            )
        
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connection = None
        self._market_is_open = False
        
    async def connect(self) -> bool:
        """Establish connection to Alpaca"""
        try:
            if not self.config.api_key:
                logger.warning("No API key provided for Alpaca, using limited access")
            
            # Create HTTP session
            headers = {}
            if self.config.api_key and self.config.api_secret:
                headers = {
                    "APCA-API-KEY-ID": self.config.api_key,
                    "APCA-API-SECRET-KEY": self.config.api_secret
                }
            
            self._session = aiohttp.ClientSession(headers=headers)
            
            # Test connection
            test_url = urljoin(self.config.base_url, "/stocks/AAPL/bars/latest")
            async with self._session.get(test_url) as response:
                if response.status == 200:
                    self.status = ProviderStatus.CONNECTED
                    logger.info("Connected to Alpaca Markets")
                    
                    # Check market status
                    await self._check_market_status()
                    return True
                else:
                    self.status = ProviderStatus.ERROR
                    logger.error(f"Failed to connect to Alpaca: {response.status}")
                    return False
                    
        except Exception as e:
            self.status = ProviderStatus.ERROR
            logger.error(f"Error connecting to Alpaca: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Alpaca"""
        try:
            if self._ws_connection:
                await self._ws_connection.close()
                self._ws_connection = None
            
            if self._session:
                await self._session.close()
                self._session = None
            
            self.status = ProviderStatus.DISCONNECTED
            logger.info("Disconnected from Alpaca Markets")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca: {str(e)}")
            return False
    
    async def fetch_ohlcv(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch OHLCV data from Alpaca"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.connect()
            
            # Build request parameters
            params = self._build_bar_params(context)
            
            all_data = []
            
            for symbol in context.symbols:
                # Determine asset class
                asset_class = self._determine_asset_class(symbol)
                endpoint = f"/{asset_class}/{symbol}/bars"
                
                url = urljoin(self.config.base_url, endpoint)
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        bars = data.get("bars", [])
                        
                        # Convert to DataFrame
                        if bars:
                            df = pd.DataFrame(bars)
                            df['symbol'] = symbol
                            df['timestamp'] = pd.to_datetime(df['t'])
                            df.rename(columns={
                                'o': 'open',
                                'h': 'high',
                                'l': 'low',
                                'c': 'close',
                                'v': 'volume',
                                'vw': 'vwap',
                                'n': 'trade_count'
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
                    "market_open": self._market_is_open
                },
                quality=context.quality,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV from Alpaca: {str(e)}")
            raise
    
    async def fetch_quotes(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch quote data from Alpaca"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.connect()
            
            all_quotes = []
            
            for symbol in context.symbols:
                asset_class = self._determine_asset_class(symbol)
                endpoint = f"/{asset_class}/{symbol}/quotes/latest"
                
                url = urljoin(self.config.base_url, endpoint)
                
                async with self._session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote = data.get("quote", {})
                        
                        if quote:
                            quote_df = pd.DataFrame([{
                                'symbol': symbol,
                                'timestamp': pd.to_datetime(quote.get('t')),
                                'bid_price': quote.get('bp'),
                                'bid_size': quote.get('bs'),
                                'ask_price': quote.get('ap'),
                                'ask_size': quote.get('as'),
                                'bid_exchange': quote.get('bx'),
                                'ask_exchange': quote.get('ax')
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
                    "quote_type": "nbbo"  # National Best Bid and Offer
                },
                quality=DataQuality.TICK,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Error fetching quotes from Alpaca: {str(e)}")
            raise
    
    async def fetch_trades(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch trade data from Alpaca"""
        start_time = datetime.now()
        
        try:
            if not self._session:
                await self.connect()
            
            params = {
                "start": context.start_date.isoformat() if context.start_date else None,
                "end": context.end_date.isoformat() if context.end_date else None,
                "limit": 1000  # Max trades per request
            }
            
            all_trades = []
            
            for symbol in context.symbols:
                asset_class = self._determine_asset_class(symbol)
                endpoint = f"/{asset_class}/{symbol}/trades"
                
                url = urljoin(self.config.base_url, endpoint)
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        trades = data.get("trades", [])
                        
                        if trades:
                            trades_df = pd.DataFrame(trades)
                            trades_df['symbol'] = symbol
                            trades_df['timestamp'] = pd.to_datetime(trades_df['t'])
                            trades_df.rename(columns={
                                'p': 'price',
                                's': 'size',
                                'x': 'exchange',
                                'c': 'conditions',
                                'i': 'trade_id'
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
                    "trade_count": len(combined_df)
                },
                quality=DataQuality.TICK,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Error fetching trades from Alpaca: {str(e)}")
            raise
    
    async def stream_data(self, context: MarketDataContext) -> AsyncIterator[MarketDataResponse]:
        """Stream real-time data from Alpaca WebSocket"""
        try:
            # Initialize WebSocket connection
            ws_url = "wss://stream.data.alpaca.markets/v2/iex"
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    # Authenticate
                    auth_msg = {
                        "action": "auth",
                        "key": self.config.api_key,
                        "secret": self.config.api_secret
                    }
                    await ws.send_json(auth_msg)
                    
                    # Subscribe to symbols
                    subscribe_msg = {
                        "action": "subscribe",
                        "trades": context.symbols,
                        "quotes": context.symbols,
                        "bars": context.symbols
                    }
                    await ws.send_json(subscribe_msg)
                    
                    # Stream data
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.json()
                            
                            # Parse and yield data
                            df = self._parse_stream_message(data)
                            if not df.empty:
                                yield MarketDataResponse(
                                    provider=self.config.name,
                                    timestamp=datetime.now(),
                                    data=df,
                                    metadata={"stream_type": data.get("T")},
                                    quality=DataQuality.TICK,
                                    latency_ms=0
                                )
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                            
        except Exception as e:
            logger.error(f"Error streaming from Alpaca: {str(e)}")
            raise
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from Alpaca"""
        try:
            if not self._session:
                await self.connect()
            
            # Get tradable assets
            assets_url = "https://api.alpaca.markets/v2/assets"
            
            async with self._session.get(assets_url) as response:
                if response.status == 200:
                    assets = await response.json()
                    
                    # Filter for tradable assets
                    symbols = [
                        asset['symbol'] 
                        for asset in assets 
                        if asset['tradable'] and asset['status'] == 'active'
                    ]
                    
                    return symbols[:1000]  # Limit for performance
                else:
                    logger.error(f"Failed to fetch assets: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching supported symbols: {str(e)}")
            return []
    
    def _determine_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol"""
        # Simple heuristic - can be enhanced
        if symbol.endswith("USD") or symbol.endswith("USDT"):
            return "crypto"
        else:
            return "stocks"
    
    def _build_bar_params(self, context: MarketDataContext) -> Dict[str, Any]:
        """Build parameters for bar data request"""
        params = {
            "timeframe": self._convert_timeframe(context.timeframe),
            "limit": 1000
        }
        
        if context.start_date:
            params["start"] = context.start_date.isoformat()
        
        if context.end_date:
            params["end"] = context.end_date.isoformat()
        
        if context.include_extended:
            params["asof"] = "true"
            params["feed"] = "iex"  # Include extended hours
        
        return params
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpaca format"""
        conversions = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "1d": "1Day",
            "1w": "1Week",
            "1M": "1Month"
        }
        return conversions.get(timeframe, "1Day")
    
    def _parse_stream_message(self, message: Dict[str, Any]) -> pd.DataFrame:
        """Parse WebSocket stream message"""
        msg_type = message.get("T")
        
        if msg_type == "t":  # Trade
            return pd.DataFrame([{
                'symbol': message.get('S'),
                'timestamp': pd.to_datetime(message.get('t')),
                'price': message.get('p'),
                'size': message.get('s'),
                'exchange': message.get('x'),
                'type': 'trade'
            }])
        elif msg_type == "q":  # Quote
            return pd.DataFrame([{
                'symbol': message.get('S'),
                'timestamp': pd.to_datetime(message.get('t')),
                'bid_price': message.get('bp'),
                'bid_size': message.get('bs'),
                'ask_price': message.get('ap'),
                'ask_size': message.get('as'),
                'type': 'quote'
            }])
        elif msg_type == "b":  # Bar
            return pd.DataFrame([{
                'symbol': message.get('S'),
                'timestamp': pd.to_datetime(message.get('t')),
                'open': message.get('o'),
                'high': message.get('h'),
                'low': message.get('l'),
                'close': message.get('c'),
                'volume': message.get('v'),
                'type': 'bar'
            }])
        else:
            return pd.DataFrame()
    
    async def _check_market_status(self) -> None:
        """Check if market is open"""
        try:
            clock_url = "https://api.alpaca.markets/v2/clock"
            
            async with self._session.get(clock_url) as response:
                if response.status == 200:
                    clock = await response.json()
                    self._market_is_open = clock.get("is_open", False)
                    logger.info(f"Market is {'open' if self._market_is_open else 'closed'}")
                    
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")