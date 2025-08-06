"""
Trading models for signals, positions, and orders
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from models.base import Base
import enum


class SignalType(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class OrderStatus(enum.Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionStatus(enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


class TradingSignal(Base):
    __tablename__ = "trading_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Signal details
    signal_type = Column(Enum(SignalType), nullable=False)
    confidence = Column(Float, nullable=False)  # 0-1 confidence score
    
    # Price levels
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    
    # Source and reasoning
    source_agent = Column(String)  # Which agent generated this
    reasoning = Column(JSON)  # Detailed reasoning for the signal
    technical_indicators = Column(JSON)  # Snapshot of indicators
    
    # Performance tracking
    is_active = Column(Boolean, default=True)
    actual_entry = Column(Float)
    actual_exit = Column(Float)
    profit_loss = Column(Float)
    
    def __repr__(self):
        return f"<TradingSignal {self.symbol} {self.signal_type.value} @ {self.timestamp}>"


class Position(Base):
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    symbol = Column(String, index=True, nullable=False)
    
    # Position details
    status = Column(Enum(PositionStatus), default=PositionStatus.OPEN)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    trailing_stop_distance = Column(Float)
    
    # Timestamps
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # P&L tracking
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    
    # Signal reference
    signal_id = Column(Integer, ForeignKey("trading_signals.id"))
    
    # Relationships
    user = relationship("User", backref="positions")
    signal = relationship("TradingSignal", backref="positions")
    
    def __repr__(self):
        return f"<Position {self.symbol} qty={self.quantity} status={self.status.value}>"


class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    position_id = Column(Integer, ForeignKey("positions.id"))
    
    # Order details
    symbol = Column(String, index=True, nullable=False)
    order_type = Column(String, nullable=False)  # MARKET, LIMIT, STOP, etc.
    side = Column(String, nullable=False)  # BUY or SELL
    quantity = Column(Float, nullable=False)
    
    # Price information
    limit_price = Column(Float)
    stop_price = Column(Float)
    executed_price = Column(Float)
    
    # Status tracking
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING)
    filled_quantity = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    executed_at = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))
    
    # External references
    broker_order_id = Column(String)
    signal_id = Column(Integer, ForeignKey("trading_signals.id"))
    
    # Relationships
    user = relationship("User", backref="orders")
    position = relationship("Position", backref="orders")
    signal = relationship("TradingSignal", backref="orders")
    
    def __repr__(self):
        return f"<Order {self.symbol} {self.side} {self.quantity} @ {self.status.value}>"


class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, index=True)
    
    # Portfolio metrics
    total_value = Column(Float, default=0.0)
    cash_balance = Column(Float, default=10000.0)
    invested_value = Column(Float, default=0.0)
    
    # Performance metrics
    total_return = Column(Float, default=0.0)
    daily_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    
    # Risk metrics
    portfolio_beta = Column(Float)
    portfolio_volatility = Column(Float)
    value_at_risk = Column(Float)  # VaR
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Snapshot of holdings
    holdings_snapshot = Column(JSON)  # Current holdings as JSON
    
    # Relationships
    user = relationship("User", backref="portfolio", uselist=False)
    
    def __repr__(self):
        return f"<Portfolio user_id={self.user_id} value={self.total_value}>"