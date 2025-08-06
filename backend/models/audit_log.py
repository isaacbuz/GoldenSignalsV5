"""
Audit Log model for tracking user actions and system events
"""

import enum
import json

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from models.base import BaseModel


class AuditEventType(enum.Enum):
    """Audit event types"""
    
    # Authentication events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_LOGIN_FAILED = "user_login_failed"
    USER_REGISTER = "user_register"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    
    # Account events
    ACCOUNT_CREATED = "account_created"
    ACCOUNT_UPDATED = "account_updated"
    ACCOUNT_DELETED = "account_deleted"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    ACCOUNT_VERIFIED = "account_verified"
    
    # Trading events
    SIGNAL_CREATED = "signal_created"
    SIGNAL_UPDATED = "signal_updated"
    SIGNAL_DELETED = "signal_deleted"
    SIGNAL_EXECUTED = "signal_executed"
    
    # Portfolio events
    PORTFOLIO_CREATED = "portfolio_created"
    PORTFOLIO_UPDATED = "portfolio_updated"
    PORTFOLIO_DELETED = "portfolio_deleted"
    TRADE_EXECUTED = "trade_executed"
    
    # API events
    API_KEY_CREATED = "api_key_created"
    API_KEY_USED = "api_key_used"
    API_KEY_REVOKED = "api_key_revoked"
    API_CALL_FAILED = "api_call_failed"
    
    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXPORT = "data_export"
    ADMIN_ACTION = "admin_action"
    
    # System events
    SYSTEM_ERROR = "system_error"
    SYSTEM_MAINTENANCE = "system_maintenance"
    CONFIGURATION_CHANGE = "configuration_change"


class AuditSeverity(enum.Enum):
    """Audit event severity levels"""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLog(BaseModel):
    """Audit log model for tracking user and system events"""
    
    __tablename__ = "audit_logs"
    
    # Event classification
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), default="info", nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)  # auth, trading, security, system
    
    # Event details
    description = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)  # Additional event data
    
    # Context information
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(255), nullable=True, index=True)
    request_id = Column(String(100), nullable=True, index=True)
    
    # Resource information
    resource_type = Column(String(50), nullable=True)  # user, signal, portfolio, etc.
    resource_id = Column(String(100), nullable=True, index=True)
    old_values = Column(JSON, nullable=True)  # Previous values for updates
    new_values = Column(JSON, nullable=True)  # New values for updates
    
    # Location and timing
    country = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    
    # System information
    service_name = Column(String(100), nullable=True)  # Which service generated the log
    service_version = Column(String(50), nullable=True)
    environment = Column(String(50), nullable=True)  # prod, staging, dev
    
    # Correlation
    correlation_id = Column(String(100), nullable=True, index=True)  # For tracing related events
    parent_event_id = Column(Integer, nullable=True)  # For event hierarchies
    
    # Risk assessment
    risk_score = Column(Integer, default=0, nullable=False)  # 0-100 risk score
    is_automated = Column(String(10), default="false", nullable=False)  # true/false as string
    requires_review = Column(String(10), default="false", nullable=False)  # true/false as string
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="audit_logs")
    
    # Additional indexes for common queries
    __table_args__ = (
        # Composite indexes for common query patterns
        {"extend_existing": True}
    )
    
    def __repr__(self):
        return f"<AuditLog(event_type={self.event_type}, user_id={self.user_id}, timestamp={self.timestamp})>"
    
    @property
    def is_security_event(self):
        """Check if this is a security-related event"""
        security_events = [
            AuditEventType.USER_LOGIN_FAILED.value,
            AuditEventType.SUSPICIOUS_ACTIVITY.value,
            AuditEventType.RATE_LIMIT_EXCEEDED.value,
            AuditEventType.UNAUTHORIZED_ACCESS.value,
            AuditEventType.ACCOUNT_LOCKED.value
        ]
        return self.event_type in security_events
    
    @property
    def is_high_risk(self):
        """Check if this is a high-risk event"""
        return self.risk_score >= 70
    
    @property
    def age_in_hours(self):
        """Calculate age of the audit log in hours"""
        from datetime import datetime
        return (datetime.utcnow() - self.timestamp).total_seconds() / 3600
    
    @property
    def formatted_details(self):
        """Get formatted details for display"""
        if not self.details:
            return {}
        
        if isinstance(self.details, str):
            try:
                return json.loads(self.details)
            except json.JSONDecodeError:
                return {"raw": self.details}
        
        return self.details
    
    @classmethod
    def log_event(
        cls,
        session,
        event_type: str,
        description: str,
        user_id: int = None,
        category: str = "general",
        severity: str = "info",
        details: dict = None,
        resource_type: str = None,
        resource_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        session_id: str = None,
        risk_score: int = 0
    ):
        """Create and save an audit log entry"""
        
        audit_log = cls(
            event_type=event_type,
            description=description,
            user_id=user_id,
            category=category,
            severity=severity,
            details=details,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            risk_score=risk_score,
            is_automated="false",
            requires_review="true" if risk_score >= 70 else "false"
        )
        
        session.add(audit_log)
        session.commit()
        return audit_log
    
    @classmethod
    def log_authentication_event(
        cls,
        session,
        event_type: str,
        user_id: int,
        success: bool,
        ip_address: str = None,
        user_agent: str = None,
        details: dict = None
    ):
        """Log authentication-related events"""
        
        severity = "info" if success else "warning"
        risk_score = 10 if not success else 0
        
        description = f"User authentication {'succeeded' if success else 'failed'}"
        
        return cls.log_event(
            session=session,
            event_type=event_type,
            description=description,
            user_id=user_id,
            category="authentication",
            severity=severity,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_score=risk_score
        )
    
    @classmethod
    def log_trading_event(
        cls,
        session,
        event_type: str,
        user_id: int,
        resource_id: str,
        description: str,
        details: dict = None
    ):
        """Log trading-related events"""
        
        return cls.log_event(
            session=session,
            event_type=event_type,
            description=description,
            user_id=user_id,
            category="trading",
            severity="info",
            resource_type="signal" if "signal" in event_type else "trade",
            resource_id=resource_id,
            details=details
        )
    
    @classmethod
    def log_security_event(
        cls,
        session,
        event_type: str,
        description: str,
        user_id: int = None,
        ip_address: str = None,
        details: dict = None,
        risk_score: int = 50
    ):
        """Log security-related events"""
        
        severity = "warning" if risk_score < 70 else "critical"
        
        return cls.log_event(
            session=session,
            event_type=event_type,
            description=description,
            user_id=user_id,
            category="security",
            severity=severity,
            details=details,
            ip_address=ip_address,
            risk_score=risk_score
        )
    
    @classmethod
    def get_user_activity(cls, session, user_id: int, hours: int = 24, limit: int = 100):
        """Get recent user activity"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return session.query(cls).filter(
            cls.user_id == user_id,
            cls.timestamp >= cutoff_time
        ).order_by(cls.timestamp.desc()).limit(limit).all()
    
    @classmethod
    def get_security_events(cls, session, hours: int = 24, limit: int = 100):
        """Get recent security events"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return session.query(cls).filter(
            cls.category == "security",
            cls.timestamp >= cutoff_time
        ).order_by(cls.timestamp.desc()).limit(limit).all()
    
    @classmethod
    def cleanup_old_logs(cls, session, days: int = 90):
        """Clean up old audit logs"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        deleted_count = session.query(cls).filter(
            cls.timestamp < cutoff_time,
            cls.severity.in_(["info"]),  # Only delete non-critical logs
            cls.requires_review == "false"
        ).delete()
        
        return deleted_count
    
    def to_dict(self):
        """Convert to dictionary"""
        data = super().to_dict()
        
        # Add computed properties
        data["is_security_event"] = self.is_security_event
        data["is_high_risk"] = self.is_high_risk
        data["age_in_hours"] = self.age_in_hours
        data["formatted_details"] = self.formatted_details
        
        return data