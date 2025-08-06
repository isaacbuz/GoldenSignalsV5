"""
User Settings model for managing user preferences and configuration
"""

import json
from typing import Any, Dict

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, Float, JSON
from sqlalchemy.orm import relationship

from models.base import BaseModel


class UserSettings(BaseModel):
    """User settings and preferences model"""
    
    __tablename__ = "user_settings"
    
    # Display preferences
    theme = Column(String(20), default="dark", nullable=False)  # dark, light, auto
    language = Column(String(10), default="en", nullable=False)  # en, es, fr, etc.
    timezone = Column(String(50), default="UTC", nullable=False)
    date_format = Column(String(20), default="YYYY-MM-DD", nullable=False)
    time_format = Column(String(10), default="24h", nullable=False)  # 24h, 12h
    
    # Chart preferences
    default_chart_timeframe = Column(String(10), default="1d", nullable=False)
    chart_type = Column(String(20), default="candlestick", nullable=False)  # candlestick, line, area
    show_volume = Column(Boolean, default=True, nullable=False)
    show_indicators = Column(Boolean, default=True, nullable=False)
    chart_theme = Column(String(20), default="dark", nullable=False)
    
    # Technical indicator preferences
    rsi_period = Column(Integer, default=14, nullable=False)
    macd_fast = Column(Integer, default=12, nullable=False)
    macd_slow = Column(Integer, default=26, nullable=False)
    macd_signal = Column(Integer, default=9, nullable=False)
    bb_period = Column(Integer, default=20, nullable=False)
    bb_std = Column(Float, default=2.0, nullable=False)
    
    # Trading preferences
    default_position_size = Column(Float, default=0.1, nullable=False)  # 10% of portfolio
    risk_per_trade = Column(Float, default=0.02, nullable=False)  # 2% max risk per trade
    auto_execute_signals = Column(Boolean, default=False, nullable=False)
    require_confirmation = Column(Boolean, default=True, nullable=False)
    preferred_exchanges = Column(JSON, nullable=True)  # ["NYSE", "NASDAQ"]
    
    # Notification preferences
    email_notifications = Column(Boolean, default=True, nullable=False)
    push_notifications = Column(Boolean, default=True, nullable=False)
    sms_notifications = Column(Boolean, default=False, nullable=False)
    
    # Email notification types
    notify_new_signals = Column(Boolean, default=True, nullable=False)
    notify_signal_updates = Column(Boolean, default=True, nullable=False)
    notify_trade_executions = Column(Boolean, default=True, nullable=False)
    notify_portfolio_changes = Column(Boolean, default=False, nullable=False)
    notify_price_alerts = Column(Boolean, default=True, nullable=False)
    notify_system_updates = Column(Boolean, default=False, nullable=False)
    
    # Price alert settings
    price_alert_frequency = Column(String(20), default="immediate", nullable=False)  # immediate, hourly, daily
    price_alert_threshold = Column(Float, default=5.0, nullable=False)  # 5% price change
    
    # Dashboard preferences
    dashboard_layout = Column(JSON, nullable=True)  # Custom dashboard layout
    favorite_symbols = Column(JSON, nullable=True)  # ["AAPL", "TSLA", "GOOGL"]
    watchlist_symbols = Column(JSON, nullable=True)  # User's watchlist
    hidden_sections = Column(JSON, nullable=True)  # Hidden dashboard sections
    
    # Agent preferences
    preferred_agents = Column(JSON, nullable=True)  # Preferred AI agents
    agent_weights = Column(JSON, nullable=True)  # Custom agent weights
    min_signal_confidence = Column(Float, default=0.7, nullable=False)  # Minimum confidence threshold
    
    # API preferences
    api_rate_limit = Column(Integer, default=1000, nullable=False)  # Requests per hour
    api_access_level = Column(String(20), default="basic", nullable=False)  # basic, premium, pro
    
    # Privacy settings
    profile_public = Column(Boolean, default=False, nullable=False)
    share_portfolio_performance = Column(Boolean, default=False, nullable=False)
    allow_data_collection = Column(Boolean, default=True, nullable=False)
    marketing_emails = Column(Boolean, default=False, nullable=False)
    
    # Advanced settings
    custom_settings = Column(JSON, nullable=True)  # Flexible JSON for custom settings
    feature_flags = Column(JSON, nullable=True)  # Beta features enabled for user
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    user = relationship("User", back_populates="settings")
    
    def __repr__(self):
        return f"<UserSettings(user_id={self.user_id}, theme={self.theme})>"
    
    @property
    def notification_methods(self):
        """Get enabled notification methods"""
        methods = []
        if self.email_notifications:
            methods.append("email")
        if self.push_notifications:
            methods.append("push")
        if self.sms_notifications:
            methods.append("sms")
        return methods
    
    @property
    def enabled_notification_types(self):
        """Get enabled notification types"""
        types = []
        if self.notify_new_signals:
            types.append("new_signals")
        if self.notify_signal_updates:
            types.append("signal_updates")
        if self.notify_trade_executions:
            types.append("trade_executions")
        if self.notify_portfolio_changes:
            types.append("portfolio_changes")
        if self.notify_price_alerts:
            types.append("price_alerts")
        if self.notify_system_updates:
            types.append("system_updates")
        return types
    
    @property
    def technical_indicators_config(self):
        """Get technical indicators configuration"""
        return {
            "rsi": {"period": self.rsi_period},
            "macd": {
                "fast": self.macd_fast,
                "slow": self.macd_slow,
                "signal": self.macd_signal
            },
            "bollinger_bands": {
                "period": self.bb_period,
                "std_dev": self.bb_std
            }
        }
    
    def get_setting(self, key: str, default: Any = None):
        """Get a setting value by key"""
        # Check if it's a direct attribute
        if hasattr(self, key):
            return getattr(self, key)
        
        # Check custom settings
        if self.custom_settings and key in self.custom_settings:
            return self.custom_settings[key]
        
        return default
    
    def set_setting(self, key: str, value: Any):
        """Set a setting value by key"""
        # Check if it's a direct attribute
        if hasattr(self, key):
            setattr(self, key, value)
            return True
        
        # Store in custom settings
        if not self.custom_settings:
            self.custom_settings = {}
        
        self.custom_settings[key] = value
        return True
    
    def update_dashboard_layout(self, layout: Dict[str, Any]):
        """Update dashboard layout"""
        self.dashboard_layout = layout
    
    def add_favorite_symbol(self, symbol: str):
        """Add symbol to favorites"""
        if not self.favorite_symbols:
            self.favorite_symbols = []
        
        if symbol not in self.favorite_symbols:
            self.favorite_symbols.append(symbol)
    
    def remove_favorite_symbol(self, symbol: str):
        """Remove symbol from favorites"""
        if self.favorite_symbols and symbol in self.favorite_symbols:
            self.favorite_symbols.remove(symbol)
    
    def add_watchlist_symbol(self, symbol: str):
        """Add symbol to watchlist"""
        if not self.watchlist_symbols:
            self.watchlist_symbols = []
        
        if symbol not in self.watchlist_symbols:
            self.watchlist_symbols.append(symbol)
    
    def remove_watchlist_symbol(self, symbol: str):
        """Remove symbol from watchlist"""
        if self.watchlist_symbols and symbol in self.watchlist_symbols:
            self.watchlist_symbols.remove(symbol)
    
    def set_agent_weight(self, agent_name: str, weight: float):
        """Set custom weight for an agent"""
        if not self.agent_weights:
            self.agent_weights = {}
        
        self.agent_weights[agent_name] = max(0.0, min(1.0, weight))  # Clamp to 0-1
    
    def get_agent_weight(self, agent_name: str, default: float = 1.0):
        """Get custom weight for an agent"""
        if not self.agent_weights:
            return default
        
        return self.agent_weights.get(agent_name, default)
    
    def enable_feature(self, feature_name: str):
        """Enable a beta feature"""
        if not self.feature_flags:
            self.feature_flags = {}
        
        self.feature_flags[feature_name] = True
    
    def disable_feature(self, feature_name: str):
        """Disable a beta feature"""
        if self.feature_flags and feature_name in self.feature_flags:
            self.feature_flags[feature_name] = False
    
    def is_feature_enabled(self, feature_name: str):
        """Check if a beta feature is enabled"""
        if not self.feature_flags:
            return False
        
        return self.feature_flags.get(feature_name, False)
    
    def reset_to_defaults(self):
        """Reset settings to default values"""
        # Reset display preferences
        self.theme = "dark"
        self.language = "en"
        self.timezone = "UTC"
        self.date_format = "YYYY-MM-DD"
        self.time_format = "24h"
        
        # Reset chart preferences
        self.default_chart_timeframe = "1d"
        self.chart_type = "candlestick"
        self.show_volume = True
        self.show_indicators = True
        
        # Reset technical indicators
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2.0
        
        # Reset trading preferences
        self.default_position_size = 0.1
        self.risk_per_trade = 0.02
        self.auto_execute_signals = False
        self.require_confirmation = True
        
        # Reset notifications
        self.email_notifications = True
        self.push_notifications = True
        self.sms_notifications = False
        self.notify_new_signals = True
        self.notify_signal_updates = True
        self.notify_trade_executions = True
        
        # Clear custom settings
        self.custom_settings = None
        self.feature_flags = None
    
    @classmethod
    def get_or_create_for_user(cls, session, user_id: int):
        """Get existing settings or create default ones for user"""
        settings = session.query(cls).filter(cls.user_id == user_id).first()
        
        if not settings:
            settings = cls(user_id=user_id)
            session.add(settings)
            session.commit()
        
        return settings
    
    def to_dict(self):
        """Convert to dictionary"""
        data = super().to_dict()
        
        # Add computed properties
        data["notification_methods"] = self.notification_methods
        data["enabled_notification_types"] = self.enabled_notification_types
        data["technical_indicators_config"] = self.technical_indicators_config
        
        return data