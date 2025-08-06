"""
Database models for GoldenSignalsAI V5
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, ForeignKey, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class AgentPerformance(Base):
    """Agent performance metrics storage"""
    __tablename__ = 'agent_performance'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Performance metrics
    accuracy = Column(Float)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    total_pnl = Column(Float)
    max_drawdown = Column(Float)
    avg_confidence = Column(Float)
    total_signals = Column(Integer)
    
    # Full metrics JSON
    metrics_json = Column(JSON)
    
    __table_args__ = (
        Index('idx_agent_performance_lookup', 'agent_id', 'timestamp'),
    )


class SignalHistory(Base):
    """Historical signals from agents"""
    __tablename__ = 'signal_history'
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(String(100), unique=True, nullable=False)
    agent_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Signal details
    signal_type = Column(String(20))  # buy, sell, hold
    confidence = Column(Float)
    expected_return = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Context
    market_conditions = Column(JSON)
    reasoning = Column(Text)
    
    # Outcome tracking
    trade_id = Column(String(100), ForeignKey('trades.trade_id'))
    outcome = Column(String(20))  # success, failure, partial, pending
    actual_return = Column(Float)
    
    trade = relationship("Trade", back_populates="signal")


class TradeOutcome(Base):
    """Trade outcomes for performance tracking"""
    __tablename__ = 'trade_outcomes'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(100), ForeignKey('trades.trade_id'), nullable=False)
    signal_id = Column(String(100), ForeignKey('signal_history.signal_id'))
    agent_id = Column(String(100), nullable=False, index=True)
    
    # Timing
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    holding_period_seconds = Column(Integer)
    
    # Prices
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    
    # Outcome
    pnl = Column(Float)
    pnl_percentage = Column(Float)
    outcome_type = Column(String(20))  # win, loss, breakeven
    
    # Risk metrics
    max_drawdown = Column(Float)
    risk_reward_ratio = Column(Float)
    
    trade = relationship("Trade", back_populates="outcome")
    signal = relationship("SignalHistory")


class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")
    trades = relationship("Trade", back_populates="user")


class APIKey(Base):
    """API keys for external services"""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    service = Column(String(50), nullable=False)  # alpaca, polygon, etc.
    key_name = Column(String(100))
    encrypted_key = Column(Text, nullable=False)
    encrypted_secret = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="api_keys")


class Portfolio(Base):
    """Portfolio tracking"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    initial_capital = Column(Float, nullable=False)
    current_value = Column(Float)
    cash_balance = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio")
    trades = relationship("Trade", back_populates="portfolio")


class Position(Base):
    """Current positions"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float, default=0)
    position_type = Column(String(10))  # long, short
    status = Column(String(20), default='open')  # open, closed
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    portfolio = relationship("Portfolio", back_populates="positions")
    trades = relationship("Trade", back_populates="position")
    
    __table_args__ = (
        Index('idx_position_lookup', 'portfolio_id', 'symbol', 'status'),
    )


class Trade(Base):
    """Trade execution records"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    position_id = Column(Integer, ForeignKey('positions.id'))
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    
    # Execution details
    order_type = Column(String(20))  # market, limit, stop
    status = Column(String(20), default='pending')  # pending, filled, cancelled
    executed_at = Column(DateTime)
    
    # Fees
    commission = Column(Float, default=0)
    slippage = Column(Float, default=0)
    
    # Metadata
    trade_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="trades")
    portfolio = relationship("Portfolio", back_populates="trades")
    position = relationship("Position", back_populates="trades")
    signal = relationship("SignalHistory", back_populates="trade")
    outcome = relationship("TradeOutcome", back_populates="trade", uselist=False)
    
    __table_args__ = (
        Index('idx_trade_lookup', 'user_id', 'symbol', 'executed_at'),
    )


class MarketData(Base):
    """Cached market data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 1h, 1d
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # OHLCV data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Additional data
    vwap = Column(Float)
    trade_count = Column(Integer)
    
    __table_args__ = (
        Index('idx_market_data_lookup', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_market_data_unique', 'symbol', 'timeframe', 'timestamp', unique=True),
    )


class Alert(Base):
    """System alerts and notifications"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    alert_type = Column(String(50), nullable=False)  # performance, risk, signal, system
    severity = Column(String(20), nullable=False)  # info, warning, error, critical
    source = Column(String(100))  # agent_id or system component
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    trade_metadata = Column(JSON)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_alert_lookup', 'alert_type', 'severity', 'created_at'),
    )


class BacktestResult(Base):
    """Backtest results storage"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    backtest_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(200))
    
    # Configuration
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    config = Column(JSON)  # Full backtest configuration
    
    # Results
    total_return = Column(Float)
    annual_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
    
    # Detailed results
    equity_curve = Column(JSON)  # Time series of portfolio value
    trade_log = Column(JSON)  # All trades executed
    metrics = Column(JSON)  # Complete metrics
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_backtest_lookup', 'created_at'),
    )