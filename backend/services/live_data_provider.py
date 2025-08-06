"""
Live Data Provider Service
Integrates multiple real-time data sources
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from dataclasses import dataclass
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LiveQuote:
    """Live quote data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    bid: float
    ask: float
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime
    source: str


class LiveDataProvider:
    """Provides live market data from multiple sources"""
    
    def __init__(self):
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.twelve_data_key = os.getenv('TWELVEDATA_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        # Priority order for data sources
        self.sources = ['yfinance', 'finnhub', 'twelve_data', 'alpha_vantage']
        
    async def get_live_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get live quote from available sources"""
        
        # Try sources in priority order
        for source in self.sources:
            try:
                if source == 'yfinance':
                    quote = await self._get_yfinance_quote(symbol)
                elif source == 'finnhub' and self.finnhub_key:
                    quote = await self._get_finnhub_quote(symbol)
                elif source == 'twelve_data' and self.twelve_data_key:
                    quote = await self._get_twelve_data_quote(symbol)
                elif source == 'alpha_vantage' and self.alpha_vantage_key:
                    quote = await self._get_alpha_vantage_quote(symbol)
                else:
                    continue
                    
                if quote:
                    return quote
                    
            except Exception as e:
                logger.warning(f"Error getting quote from {source}: {e}")
                continue
                
        logger.error(f"Failed to get quote for {symbol} from all sources")
        return None
        
    async def _get_yfinance_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get quote from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', current_price)
            
            if current_price == 0:
                return None
                
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close else 0
            
            return LiveQuote(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=info.get('volume', 0),
                bid=info.get('bid', current_price),
                ask=info.get('ask', current_price),
                high=info.get('dayHigh', current_price),
                low=info.get('dayLow', current_price),
                open=info.get('open', previous_close),
                previous_close=previous_close,
                timestamp=datetime.utcnow(),
                source='yfinance'
            )
            
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return None
            
    async def _get_finnhub_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get quote from Finnhub"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.finnhub_key}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        current_price = data.get('c', 0)
                        previous_close = data.get('pc', current_price)
                        
                        if current_price == 0:
                            return None
                            
                        change = current_price - previous_close
                        change_percent = data.get('dp', 0)
                        
                        return LiveQuote(
                            symbol=symbol,
                            price=current_price,
                            change=change,
                            change_percent=change_percent,
                            volume=0,  # Finnhub doesn't provide volume in quote
                            bid=current_price,
                            ask=current_price,
                            high=data.get('h', current_price),
                            low=data.get('l', current_price),
                            open=data.get('o', previous_close),
                            previous_close=previous_close,
                            timestamp=datetime.fromtimestamp(data.get('t', datetime.utcnow().timestamp())),
                            source='finnhub'
                        )
                        
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            return None
            
    async def _get_twelve_data_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get quote from Twelve Data"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={self.twelve_data_key}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') == 'error':
                            return None
                            
                        current_price = float(data.get('close', 0))
                        previous_close = float(data.get('previous_close', current_price))
                        
                        if current_price == 0:
                            return None
                            
                        change = float(data.get('change', 0))
                        change_percent = float(data.get('percent_change', 0))
                        
                        return LiveQuote(
                            symbol=symbol,
                            price=current_price,
                            change=change,
                            change_percent=change_percent,
                            volume=int(data.get('volume', 0)),
                            bid=current_price,
                            ask=current_price,
                            high=float(data.get('high', current_price)),
                            low=float(data.get('low', current_price)),
                            open=float(data.get('open', previous_close)),
                            previous_close=previous_close,
                            timestamp=datetime.utcnow(),
                            source='twelve_data'
                        )
                        
        except Exception as e:
            logger.error(f"Twelve Data error for {symbol}: {e}")
            return None
            
    async def _get_alpha_vantage_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get quote from Alpha Vantage"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_vantage_key}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote_data = data.get('Global Quote', {})
                        
                        if not quote_data:
                            return None
                            
                        current_price = float(quote_data.get('05. price', 0))
                        previous_close = float(quote_data.get('08. previous close', current_price))
                        
                        if current_price == 0:
                            return None
                            
                        change = float(quote_data.get('09. change', 0))
                        change_percent_str = quote_data.get('10. change percent', '0%').rstrip('%')
                        change_percent = float(change_percent_str) if change_percent_str else 0
                        
                        return LiveQuote(
                            symbol=symbol,
                            price=current_price,
                            change=change,
                            change_percent=change_percent,
                            volume=int(quote_data.get('06. volume', 0)),
                            bid=current_price,
                            ask=current_price,
                            high=float(quote_data.get('03. high', current_price)),
                            low=float(quote_data.get('04. low', current_price)),
                            open=float(quote_data.get('02. open', previous_close)),
                            previous_close=previous_close,
                            timestamp=datetime.utcnow(),
                            source='alpha_vantage'
                        )
                        
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None
            
    async def stream_quotes(self, symbols: List[str], callback):
        """Stream live quotes for multiple symbols"""
        while True:
            try:
                tasks = [self.get_live_quote(symbol) for symbol in symbols]
                quotes = await asyncio.gather(*tasks)
                
                for quote in quotes:
                    if quote:
                        await callback(quote)
                        
                # Rate limiting - adjust based on your needs
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error streaming quotes: {e}")
                await asyncio.sleep(5)  # Wait longer on error


# Global instance
live_data_provider = LiveDataProvider()