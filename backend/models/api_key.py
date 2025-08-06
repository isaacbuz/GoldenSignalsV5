"""
API Key model for managing user API access
"""

import enum
import secrets
import hashlib
from datetime import datetime, timedelta

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from models.base import BaseModel


class APIKeyStatus(enum.Enum):
    """API key status types"""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


class APIKeyScope(enum.Enum):
    """API key permission scopes"""
    
    READ = "read"  # Read-only access
    TRADE = "trade"  # Trading operations
    ADMIN = "admin"  # Administrative operations
    FULL = "full"  # Full access


class APIKey(BaseModel):
    """API key model for programmatic access"""
    
    __tablename__ = "api_keys"
    
    # Key identification
    name = Column(String(100), nullable=False)  # User-defined name
    key_id = Column(String(64), unique=True, nullable=False, index=True)  # Public key identifier
    key_hash = Column(String(128), nullable=False)  # Hashed secret key
    key_prefix = Column(String(16), nullable=False)  # First few chars for identification
    
    # Key permissions
    scopes = Column(JSON, nullable=False)  # List of allowed scopes
    rate_limit_per_hour = Column(Integer, default=1000, nullable=False)
    rate_limit_per_day = Column(Integer, default=10000, nullable=False)
    allowed_ips = Column(JSON, nullable=True)  # Whitelist of allowed IP addresses
    
    # Key status
    status = Column(String(20), default="active", nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Usage tracking
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_used_ip = Column(String(45), nullable=True)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)
    auto_renew = Column(Boolean, default=False, nullable=False)
    
    # Security
    created_by_ip = Column(String(45), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_reason = Column(Text, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # User-defined tags for organization
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey(name={self.name}, key_id={self.key_id}, status={self.status})>"
    
    @property
    def is_expired(self):
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return self.expires_at < datetime.utcnow()
    
    @property
    def is_usable(self):
        """Check if API key can be used"""
        return (
            self.is_active and 
            self.status == APIKeyStatus.ACTIVE.value and 
            not self.is_expired
        )
    
    @property
    def time_until_expiry(self):
        """Time until key expires"""
        if not self.expires_at:
            return None
        if self.is_expired:
            return timedelta(0)
        return self.expires_at - datetime.utcnow()
    
    @property
    def usage_percentage(self):
        """Current usage as percentage of daily limit"""
        if self.rate_limit_per_day <= 0:
            return 0
        return (self.usage_count / self.rate_limit_per_day) * 100
    
    @property
    def masked_key(self):
        """Return masked key for display"""
        return f"{self.key_prefix}{'*' * 40}"
    
    @classmethod
    def generate_key_pair(cls):
        """Generate a new API key pair"""
        # Generate random key
        secret_key = secrets.token_urlsafe(32)  # 256-bit key
        
        # Create key ID (first 8 chars + random suffix)
        key_id = f"gsk_{secrets.token_hex(8)}"
        
        # Hash the secret key
        key_hash = hashlib.sha256(secret_key.encode()).hexdigest()
        
        # Get prefix for identification
        key_prefix = secret_key[:8]
        
        return {
            "secret_key": secret_key,
            "key_id": key_id,
            "key_hash": key_hash,
            "key_prefix": key_prefix
        }
    
    @classmethod
    def create_api_key(
        cls,
        session,
        user_id: int,
        name: str,
        scopes: list,
        description: str = None,
        expires_in_days: int = None,
        rate_limit_per_hour: int = 1000,
        rate_limit_per_day: int = 10000,
        allowed_ips: list = None,
        created_by_ip: str = None
    ):
        """Create a new API key"""
        
        key_data = cls.generate_key_pair()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = cls(
            user_id=user_id,
            name=name,
            key_id=key_data["key_id"],
            key_hash=key_data["key_hash"],
            key_prefix=key_data["key_prefix"],
            scopes=scopes,
            description=description,
            expires_at=expires_at,
            rate_limit_per_hour=rate_limit_per_hour,
            rate_limit_per_day=rate_limit_per_day,
            allowed_ips=allowed_ips,
            created_by_ip=created_by_ip
        )
        
        session.add(api_key)
        session.commit()
        
        # Return both the model and the secret key (only time it's available)
        return api_key, key_data["secret_key"]
    
    def verify_key(self, secret_key: str):
        """Verify a secret key against this API key"""
        key_hash = hashlib.sha256(secret_key.encode()).hexdigest()
        return key_hash == self.key_hash
    
    def has_scope(self, required_scope: str):
        """Check if API key has required scope"""
        if not self.scopes:
            return False
        
        # Full scope grants all permissions
        if APIKeyScope.FULL.value in self.scopes:
            return True
        
        return required_scope in self.scopes
    
    def is_ip_allowed(self, ip_address: str):
        """Check if IP address is allowed"""
        if not self.allowed_ips:
            return True  # No IP restriction
        
        return ip_address in self.allowed_ips
    
    def record_usage(self, ip_address: str = None):
        """Record API key usage"""
        self.usage_count += 1
        self.last_used_at = func.now()
        if ip_address:
            self.last_used_ip = ip_address
    
    def revoke(self, reason: str = "Manual revocation"):
        """Revoke the API key"""
        self.status = APIKeyStatus.REVOKED.value
        self.is_active = False
        self.revoked_at = func.now()
        self.revoked_reason = reason
    
    def extend_expiry(self, days: int):
        """Extend API key expiry"""
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(days=days)
        else:
            self.expires_at += timedelta(days=days)
    
    def update_scopes(self, new_scopes: list):
        """Update API key scopes"""
        self.scopes = new_scopes
    
    def add_scope(self, scope: str):
        """Add a scope to the API key"""
        if not self.scopes:
            self.scopes = []
        
        if scope not in self.scopes:
            self.scopes.append(scope)
    
    def remove_scope(self, scope: str):
        """Remove a scope from the API key"""
        if self.scopes and scope in self.scopes:
            self.scopes.remove(scope)
    
    @classmethod
    def get_by_key_id(cls, session, key_id: str):
        """Get API key by key ID"""
        return session.query(cls).filter(cls.key_id == key_id).first()
    
    @classmethod
    def authenticate(cls, session, key_id: str, secret_key: str, ip_address: str = None):
        """Authenticate an API key"""
        api_key = cls.get_by_key_id(session, key_id)
        
        if not api_key:
            return None, "Invalid API key"
        
        if not api_key.is_usable:
            return None, "API key is not active or has expired"
        
        if not api_key.verify_key(secret_key):
            return None, "Invalid secret key"
        
        if ip_address and not api_key.is_ip_allowed(ip_address):
            return None, "IP address not allowed"
        
        # Record usage
        api_key.record_usage(ip_address)
        
        return api_key, None
    
    @classmethod
    def cleanup_expired_keys(cls, session):
        """Mark expired keys as expired"""
        cutoff_time = datetime.utcnow()
        
        expired_keys = session.query(cls).filter(
            cls.expires_at < cutoff_time,
            cls.status == APIKeyStatus.ACTIVE.value
        ).all()
        
        for key in expired_keys:
            key.status = APIKeyStatus.EXPIRED.value
            key.is_active = False
        
        return len(expired_keys)
    
    @classmethod
    def get_user_keys(cls, session, user_id: int, active_only: bool = False):
        """Get all API keys for a user"""
        query = session.query(cls).filter(cls.user_id == user_id)
        
        if active_only:
            query = query.filter(cls.is_active == True, cls.status == APIKeyStatus.ACTIVE.value)
        
        return query.order_by(cls.created_at.desc()).all()
    
    def to_dict(self):
        """Convert to dictionary (without sensitive data)"""
        data = super().to_dict()
        
        # Add computed properties
        data["is_expired"] = self.is_expired
        data["is_usable"] = self.is_usable
        data["usage_percentage"] = self.usage_percentage
        data["masked_key"] = self.masked_key
        
        if self.time_until_expiry:
            data["time_until_expiry"] = str(self.time_until_expiry)
        
        # Remove sensitive fields
        data.pop("key_hash", None)
        
        return data