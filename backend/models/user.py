"""
User model for authentication and authorization
"""

from sqlalchemy import Column, String, Boolean, DateTime, Float, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from models.base import BaseModel


class User(BaseModel):
    """User model with trading account details"""
    
    __tablename__ = "users"

    # Basic user information
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    # Trading account details
    account_balance = Column(Float, default=10000.0)  # Starting balance
    total_profit_loss = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    risk_tolerance = Column(String(20), default='medium')  # low, medium, high
    preferred_timeframe = Column(String(20), default='1d')
    
    # Profile and preferences
    profile_image_url = Column(String(500))
    bio = Column(Text)
    timezone = Column(String(50), default='UTC')
    notification_preferences = Column(String(100), default='email,push')
    
    # API Keys (encrypted in production)
    api_key = Column(String(128), unique=True, index=True)
    api_secret = Column(String(128))
    
    # Authentication tracking
    last_login = Column(DateTime(timezone=True))
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime(timezone=True))
    
    # Relationships
    signals = relationship("Signal", back_populates="user", cascade="all, delete-orphan")
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"
    
    @property
    def is_locked(self):
        """Check if account is locked"""
        if not self.account_locked_until:
            return False
        return self.account_locked_until > func.now()
    
    @property
    def win_rate_percentage(self):
        """Calculate win rate as percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.successful_trades / self.total_trades) * 100
    
    @property
    def portfolio_return(self):
        """Calculate portfolio return percentage"""
        if self.account_balance <= 0:
            return 0.0
        return (self.total_profit_loss / 10000.0) * 100  # Against starting balance
    
    def update_trading_stats(self, trade_pnl: float, is_win: bool):
        """Update user trading statistics"""
        self.total_trades += 1
        self.total_profit_loss += trade_pnl
        self.account_balance += trade_pnl
        
        if is_win:
            self.successful_trades += 1
        
        # Recalculate win rate
        if self.total_trades > 0:
            self.win_rate = (self.successful_trades / self.total_trades) * 100
    
    def reset_failed_logins(self):
        """Reset failed login attempts"""
        self.failed_login_attempts = 0
        self.account_locked_until = None
    
    def increment_failed_login(self):
        """Increment failed login attempts"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:
            # Lock account for 15 minutes
            self.account_locked_until = func.now() + func.interval(15, 'minute')
    
    def to_dict(self):
        """Convert to dictionary with computed fields"""
        data = super().to_dict()
        
        # Add computed properties
        data["is_locked"] = self.is_locked
        data["win_rate_percentage"] = self.win_rate_percentage
        data["portfolio_return"] = self.portfolio_return
        
        # Remove sensitive fields
        data.pop("hashed_password", None)
        data.pop("api_secret", None)
        
        return data