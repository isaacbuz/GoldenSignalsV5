"""
User Session model for tracking active user sessions
"""

import enum
from datetime import datetime, timedelta

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from models.base import BaseModel


class SessionStatus(enum.Enum):
    """Session status types"""
    
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INVALID = "invalid"


class UserSession(BaseModel):
    """User session tracking model"""
    
    __tablename__ = "user_sessions"
    
    # Session identification
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=True, index=True)
    
    # Session metadata
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    device_type = Column(String(50), nullable=True)  # mobile, desktop, tablet
    browser = Column(String(100), nullable=True)
    platform = Column(String(100), nullable=True)
    
    # Location information
    country = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    timezone = Column(String(50), nullable=True)
    
    # Session timing
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    login_at = Column(DateTime(timezone=True), default=func.now())
    logout_at = Column(DateTime(timezone=True), nullable=True)
    
    # Session status
    status = Column(String(20), default="active", nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    remember_me = Column(Boolean, default=False, nullable=False)
    
    # Security flags
    is_suspicious = Column(Boolean, default=False, nullable=False)
    login_method = Column(String(50), default="password", nullable=False)  # password, oauth, api_key
    two_factor_verified = Column(Boolean, default=False, nullable=False)
    
    # Session data
    session_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, status={self.status}, last_activity={self.last_activity})>"
    
    @property
    def is_expired(self):
        """Check if session is expired"""
        return self.expires_at < datetime.utcnow()
    
    @property
    def time_until_expiry(self):
        """Time until session expires"""
        if self.is_expired:
            return timedelta(0)
        return self.expires_at - datetime.utcnow()
    
    @property
    def duration(self):
        """Session duration"""
        end_time = self.logout_at or datetime.utcnow()
        return end_time - self.login_at if self.login_at else timedelta(0)
    
    @property
    def is_long_session(self):
        """Check if session is unusually long (>24 hours)"""
        return self.duration > timedelta(hours=24)
    
    def extend_session(self, hours: int = 24):
        """Extend session expiry"""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.last_activity = func.now()
    
    def revoke(self, reason: str = "manual"):
        """Revoke the session"""
        self.status = SessionStatus.REVOKED.value
        self.is_active = False
        self.logout_at = func.now()
        
        # Add reason to session data if exists
        import json
        data = json.loads(self.session_data) if self.session_data else {}
        data["revoke_reason"] = reason
        data["revoked_at"] = datetime.utcnow().isoformat()
        self.session_data = json.dumps(data)
    
    def mark_suspicious(self, reason: str = "unknown"):
        """Mark session as suspicious"""
        self.is_suspicious = True
        
        # Add suspicious activity to session data
        import json
        data = json.loads(self.session_data) if self.session_data else {}
        if "suspicious_activities" not in data:
            data["suspicious_activities"] = []
        
        data["suspicious_activities"].append({
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.session_data = json.dumps(data)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = func.now()
    
    @classmethod
    def cleanup_expired_sessions(cls, session):
        """Clean up expired sessions"""
        cutoff_time = datetime.utcnow()
        expired_sessions = session.query(cls).filter(
            cls.expires_at < cutoff_time,
            cls.status == SessionStatus.ACTIVE.value
        ).all()
        
        for sess in expired_sessions:
            sess.status = SessionStatus.EXPIRED.value
            sess.is_active = False
        
        return len(expired_sessions)
    
    @classmethod
    def get_active_sessions_count(cls, session, user_id: int):
        """Get count of active sessions for user"""
        return session.query(cls).filter(
            cls.user_id == user_id,
            cls.is_active == True,
            cls.expires_at > datetime.utcnow()
        ).count()
    
    def to_dict(self):
        """Convert to dictionary"""
        data = super().to_dict()
        
        # Add computed properties
        data["is_expired"] = self.is_expired
        data["time_until_expiry"] = str(self.time_until_expiry)
        data["duration"] = str(self.duration)
        data["is_long_session"] = self.is_long_session
        
        # Parse session data if exists
        if self.session_data:
            import json
            try:
                data["session_metadata"] = json.loads(self.session_data)
            except json.JSONDecodeError:
                data["session_metadata"] = {}
        
        # Remove sensitive fields
        data.pop("session_token", None)
        data.pop("refresh_token", None)
        
        return data