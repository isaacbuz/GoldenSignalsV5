"""
Market data models for storing price data and indicators
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index, Boolean
from sqlalchemy.sql import func
from models.base import Base


class Stock(Base):
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    sector = Column(String)
    industry = Column(String)
    market_cap = Column(Float)
    exchange = Column(String)
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Stock {self.symbol}>"


class PriceData(Base):
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Additional metrics
    vwap = Column(Float)  # Volume Weighted Average Price
    trades_count = Column(Integer)
    
    # Create composite index for efficient queries
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<PriceData {self.symbol} @ {self.timestamp}>"


class TechnicalIndicators(Base):
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    
    # Moving averages
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Momentum indicators
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    
    # Volatility indicators
    bb_upper = Column(Float)  # Bollinger Band Upper
    bb_middle = Column(Float)  # Bollinger Band Middle
    bb_lower = Column(Float)  # Bollinger Band Lower
    atr = Column(Float)  # Average True Range
    
    # Volume indicators
    obv = Column(Float)  # On Balance Volume
    volume_sma = Column(Float)
    
    # Additional indicators stored as JSON
    additional_indicators = Column(JSON)
    
    __table_args__ = (
        Index('idx_indicators_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<TechnicalIndicators {self.symbol} @ {self.timestamp}>"


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Market indices
    sp500 = Column(Float)
    nasdaq = Column(Float)
    dow = Column(Float)
    vix = Column(Float)  # Volatility index
    
    # Market breadth
    advances = Column(Integer)
    declines = Column(Integer)
    unchanged = Column(Integer)
    
    # Volume metrics
    total_volume = Column(Float)
    
    # Sentiment indicators
    fear_greed_index = Column(Float)
    put_call_ratio = Column(Float)
    
    # Additional market data as JSON
    additional_data = Column(JSON)
    
    def __repr__(self):
        return f"<MarketSnapshot @ {self.timestamp}>"