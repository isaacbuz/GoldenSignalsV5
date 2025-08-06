"""
Application configuration and settings
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "GoldenSignalsAI"
    VERSION: str = "5.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-here-minimum-32-chars"
    
    # Database
    DATABASE_URL: str = "sqlite:///./goldensignals.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_POOL_RECYCLE: int = 300
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    TWELVEDATA_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    FMP_API_KEY: Optional[str] = None
    
    # Security
    JWT_SECRET_KEY: str = "your-jwt-secret-key-here"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # WebSocket
    WS_HOST: str = "localhost"
    WS_PORT: int = 8001
    WS_MAX_CONNECTIONS: int = 1000
    
    # AI Agent Settings
    MAX_AGENTS: int = 30
    AGENT_TIMEOUT: int = 30
    CONSENSUS_THRESHOLD: float = 0.6
    SIGNAL_GENERATION_INTERVAL: int = 30
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # Feature Flags
    FEATURE_AI_SIGNALS: bool = True
    FEATURE_LIVE_DATA: bool = True
    FEATURE_BACKTESTING: bool = True
    FEATURE_PORTFOLIO_TRACKING: bool = True
    FEATURE_RISK_MANAGEMENT: bool = True
    FEATURE_SOCIAL_TRADING: bool = False
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: int = 30
    CONNECTION_POOL_SIZE: int = 20
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Monitoring
    HEALTH_CHECK_ENDPOINT: str = "/health"
    READINESS_CHECK_ENDPOINT: str = "/ready"
    METRICS_ENDPOINT: str = "/metrics"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env


# Create global settings instance
settings = Settings()


# Validate critical settings on startup
def validate_settings():
    """Validate critical settings"""
    errors = []
    
    # Check for API keys if features are enabled
    if settings.FEATURE_AI_SIGNALS and not (settings.OPENAI_API_KEY or settings.ANTHROPIC_API_KEY):
        errors.append("AI signals feature enabled but no AI API key provided")
    
    if settings.FEATURE_LIVE_DATA and not any([
        settings.TWELVEDATA_API_KEY,
        settings.FINNHUB_API_KEY,
        settings.ALPHA_VANTAGE_API_KEY,
        settings.POLYGON_API_KEY,
        settings.FMP_API_KEY
    ]):
        errors.append("Live data feature enabled but no market data API key provided")
    
    # Check database URL
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL not configured")
    
    # Check secret keys
    if settings.SECRET_KEY == "your-secret-key-here-minimum-32-chars":
        errors.append("Using default SECRET_KEY - please set a secure key")
    
    if settings.JWT_SECRET_KEY == "your-jwt-secret-key-here":
        errors.append("Using default JWT_SECRET_KEY - please set a secure key")
    
    if errors and settings.ENVIRONMENT == "production":
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    elif errors:
        print(f"⚠️  Configuration warnings: {'; '.join(errors)}")


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings


# Export commonly used settings
__all__ = ["settings", "validate_settings", "get_settings"]