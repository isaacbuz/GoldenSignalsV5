"""
YFinance market data service for fetching real-time and historical stock data
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class YFinanceService:
    """Service for fetching market data using yfinance"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._cache = {}
        self._cache_ttl = 60  # Cache TTL in seconds
        
    @lru_cache(maxsize=100)
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get a yfinance Ticker object (cached)"""
        return yf.Ticker(symbol)
    
    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price and basic info for a symbol"""
        try:
            ticker = self.get_ticker(symbol)
            info = ticker.info
            
            # Get current price (handle different field names)
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', 0)
            
            # Calculate change and change percentage
            change = current_price - previous_close if previous_close else 0
            change_percent = (change / previous_close * 100) if previous_close else 0
            
            return {
                "symbol": symbol,
                "price": current_price,
                "previousClose": previous_close,
                "change": change,
                "changePercent": change_percent,
                "volume": info.get('volume', 0),
                "marketCap": info.get('marketCap', 0),
                "dayHigh": info.get('dayHigh', 0),
                "dayLow": info.get('dayLow', 0),
                "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh', 0),
                "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow', 0),
                "name": info.get('longName', symbol),
                "exchange": info.get('exchange', 'N/A'),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Stock symbol
            period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        """
        try:
            ticker = self.get_ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Add technical indicators
            if not df.empty:
                df = self._add_technical_indicators(df)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Get intraday data for the current trading day"""
        try:
            ticker = self.get_ticker(symbol)
            df = ticker.history(period="1d", interval=interval)
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the dataframe"""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            df['RSI'] = self._calculate_rsi(df['Close'])
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
            bb_std_dev = df['Close'].rolling(window=bb_period).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * bb_std_dev)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * bb_std_dev)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_multiple_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get quotes for multiple symbols"""
        quotes = []
        for symbol in symbols:
            quote = self.get_current_price(symbol)
            quotes.append(quote)
        return quotes
    
    async def get_multiple_quotes_async(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get quotes for multiple symbols asynchronously"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.get_current_price, symbol)
            for symbol in symbols
        ]
        quotes = await asyncio.gather(*tasks)
        return quotes
    
    def get_options_chain(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get options chain data"""
        try:
            ticker = self.get_ticker(symbol)
            
            # Get expiration dates
            expirations = ticker.options
            
            if not expirations:
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
            
            # Get options for the nearest expiration
            opt = ticker.option_chain(expirations[0])
            
            return {
                "calls": opt.calls,
                "puts": opt.puts,
                "expiration": expirations[0]
            }
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary with major indices"""
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX",
            "^TNX": "10Y Treasury"
        }
        
        summary = {}
        for symbol, name in indices.items():
            data = self.get_current_price(symbol)
            summary[name] = data
        
        return summary
    
    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """Search for stock symbols"""
        try:
            # This is a simplified search - in production, you'd use a proper API
            # For now, we'll just validate if it's a valid ticker
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            
            if info.get('symbol'):
                return [{
                    "symbol": info.get('symbol', query.upper()),
                    "name": info.get('longName', 'N/A'),
                    "exchange": info.get('exchange', 'N/A'),
                    "type": info.get('quoteType', 'EQUITY')
                }]
            return []
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed company information"""
        try:
            ticker = self.get_ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "description": info.get('longBusinessSummary', 'N/A'),
                "website": info.get('website', 'N/A'),
                "employees": info.get('fullTimeEmployees', 0),
                "marketCap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "dividend_yield": info.get('dividendYield', 0),
                "beta": info.get('beta', 0),
                "earnings_date": info.get('earningsDate', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}


# Global service instance
yfinance_service = YFinanceService()