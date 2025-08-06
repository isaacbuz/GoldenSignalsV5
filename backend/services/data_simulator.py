"""
Market Data Simulator for Demo/Testing
Generates realistic market data when real APIs are unavailable
"""

import random
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np

class MarketDataSimulator:
    """Simulates realistic market data for stocks"""
    
    def __init__(self):
        # Base prices for popular stocks
        self.base_prices = {
            "AAPL": 175.50,
            "GOOGL": 140.25,
            "MSFT": 380.50,
            "AMZN": 180.75,
            "TSLA": 245.30,
            "META": 520.80,
            "NVDA": 875.50,
            "SPY": 470.25,
            "QQQ": 405.75,
            "DIA": 380.25,
            "IWM": 195.50,
            "VIX": 13.50,
            "JPM": 185.25,
            "V": 280.50,
            "JNJ": 155.75,
            "WMT": 185.25,
            "PG": 165.50,
            "MA": 470.25,
            "UNH": 520.75,
            "HD": 380.25,
        }
        
        # Track current prices
        self.current_prices = self.base_prices.copy()
        
        # Volatility settings
        self.volatilities = {
            "TSLA": 0.03,  # 3% volatility
            "NVDA": 0.025, # 2.5% volatility
            "VIX": 0.05,   # 5% volatility (VIX is more volatile)
            "default": 0.015  # 1.5% default volatility
        }
        
        # Market hours simulation
        self.market_open = True
        
        # Initialize price history for each symbol
        self.price_history = {symbol: [price] for symbol, price in self.base_prices.items()}
        
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate a realistic quote for a symbol"""
        
        if symbol not in self.current_prices:
            # Generate random price for unknown symbols
            self.current_prices[symbol] = random.uniform(10, 500)
            self.base_prices[symbol] = self.current_prices[symbol]
            self.price_history[symbol] = [self.current_prices[symbol]]
        
        # Get volatility for this symbol
        volatility = self.volatilities.get(symbol, self.volatilities["default"])
        
        # Generate price movement
        if self.market_open:
            # Use geometric Brownian motion for realistic price movement
            dt = 1/390  # 1 minute in trading day
            drift = 0.0001  # Slight upward drift
            random_shock = np.random.normal(0, 1)
            
            price_change = self.current_prices[symbol] * (
                drift * dt + volatility * np.sqrt(dt) * random_shock
            )
            
            new_price = self.current_prices[symbol] + price_change
            
            # Prevent negative prices
            new_price = max(new_price, 0.01)
            
            # Update current price
            self.current_prices[symbol] = new_price
            
            # Add to history (keep last 100 prices)
            self.price_history[symbol].append(new_price)
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol].pop(0)
        else:
            new_price = self.current_prices[symbol]
        
        # Calculate daily stats
        base_price = self.base_prices[symbol]
        change = new_price - base_price
        change_percent = (change / base_price) * 100
        
        # Generate volume (higher for ETFs)
        if symbol in ["SPY", "QQQ", "DIA", "IWM"]:
            volume = random.randint(50000000, 150000000)
        else:
            volume = random.randint(5000000, 30000000)
        
        # Calculate day high/low from history
        recent_prices = self.price_history[symbol][-20:] if len(self.price_history[symbol]) > 20 else self.price_history[symbol]
        day_high = max(recent_prices) * random.uniform(1.0, 1.002)
        day_low = min(recent_prices) * random.uniform(0.998, 1.0)
        
        # Generate quote
        quote = {
            "symbol": symbol,
            "price": round(new_price, 2),
            "change": round(change, 2),
            "changePercent": round(change_percent, 2),
            "volume": volume,
            "marketCap": int(new_price * random.uniform(1e9, 1e12)),  # Simplified market cap
            "dayHigh": round(day_high, 2),
            "dayLow": round(day_low, 2),
            "open": round(base_price * random.uniform(0.995, 1.005), 2),
            "previousClose": round(base_price, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bid": round(new_price - 0.01, 2),
            "ask": round(new_price + 0.01, 2),
            "bidSize": random.randint(100, 1000),
            "askSize": random.randint(100, 1000),
        }
        
        return quote
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_quote(symbol)
        return quotes
    
    def get_historical_data(self, symbol: str, period: str = "1d", interval: str = "5m"):
        """Generate historical data for charting"""
        # Determine number of data points based on period and interval
        data_points = {
            ("1d", "1m"): 390,
            ("1d", "5m"): 78,
            ("5d", "5m"): 390,
            ("1mo", "1h"): 180,
            ("1d", "15m"): 26,
            ("1d", "30m"): 13,
            ("1d", "1h"): 7,
        }.get((period, interval), 100)
        
        # Generate historical data
        historical = []
        current_price = self.base_prices.get(symbol, 100)
        volatility = self.volatilities.get(symbol, self.volatilities["default"])
        
        # Work backwards from current time
        current_time = time.time()
        interval_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
        }.get(interval, 300)
        
        for i in range(data_points, 0, -1):
            timestamp = current_time - (i * interval_seconds)
            
            # Generate OHLCV data
            open_price = current_price
            
            # Generate intrabar movement
            high_price = open_price * random.uniform(1.0, 1.0 + volatility/10)
            low_price = open_price * random.uniform(1.0 - volatility/10, 1.0)
            
            # Close price with trend
            trend = random.choice([-1, 1]) * random.uniform(0, volatility/5)
            close_price = open_price * (1 + trend)
            
            # Ensure logical OHLC relationship
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            volume = random.randint(100000, 5000000)
            
            historical.append({
                "time": int(timestamp),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume
            })
            
            # Update price for next candle
            current_price = close_price
        
        return historical
    
    def set_market_open(self, is_open: bool):
        """Set market open/closed state"""
        self.market_open = is_open
        
        if is_open:
            # Reset base prices at market open (new trading day)
            self.base_prices = self.current_prices.copy()
            self.price_history = {symbol: [price] for symbol, price in self.current_prices.items()}

# Global simulator instance
simulator = MarketDataSimulator()