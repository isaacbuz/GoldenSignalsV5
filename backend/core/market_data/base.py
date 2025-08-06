"""
Base Market Data Provider Interface
Following V5 architecture patterns with context-aware, pluggable design
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, AsyncIterator
import asyncio
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)


class DataProviderType(Enum):
    """Types of market data providers"""
    REAL_TIME = "real_time"
    DELAYED = "delayed"
    HISTORICAL = "historical"
    SIMULATED = "simulated"
    AGGREGATED = "aggregated"


class DataQuality(Enum):
    """Data quality levels"""
    TICK = "tick"          # Raw tick data
    SECOND = "second"      # 1-second bars
    MINUTE = "minute"      # 1-minute bars
    HOURLY = "hourly"      # Hourly bars
    DAILY = "daily"        # Daily bars
    ADJUSTED = "adjusted"  # Corporate action adjusted


class ProviderStatus(Enum):
    """Provider connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


@dataclass
class MarketDataContext:
    """Context for market data requests - similar to AgentContext"""
    symbols: List[str]
    timeframe: str = "1d"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    data_types: List[str] = field(default_factory=lambda: ["ohlcv"])
    quality: DataQuality = DataQuality.MINUTE
    include_extended: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "data_types": self.data_types,
            "quality": self.quality.value,
            "include_extended": self.include_extended,
            "metadata": self.metadata
        }


@dataclass
class MarketDataResponse:
    """Standardized market data response"""
    provider: str
    timestamp: datetime
    data: pd.DataFrame
    metadata: Dict[str, Any]
    quality: DataQuality
    latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data.to_dict() if not self.data.empty else {},
            "metadata": self.metadata,
            "quality": self.quality.value,
            "latency_ms": self.latency_ms
        }


class ProviderConfig(BaseModel):
    """Configuration for market data providers"""
    name: str
    type: DataProviderType
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    retry_count: int = 3
    priority: int = 1  # Lower number = higher priority
    supported_markets: List[str] = Field(default_factory=lambda: ["stocks", "etf", "crypto"])
    supported_data_types: List[str] = Field(default_factory=lambda: ["ohlcv", "trades", "quotes"])
    cost_per_request: float = 0.0  # For cost optimization
    enabled: bool = True
    
    class Config:
        extra = "allow"


class ProviderHealth(BaseModel):
    """Health metrics for a provider"""
    provider_name: str
    status: ProviderStatus
    uptime_percentage: float
    avg_latency_ms: float
    error_rate: float
    requests_today: int
    rate_limit_remaining: int
    last_success: Optional[datetime]
    last_error: Optional[str]
    quality_score: float  # 0-1 score based on various factors


class IMarketDataProvider(ABC):
    """
    Abstract interface for all market data providers
    Following the same pattern as BaseAgent
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.status = ProviderStatus.DISCONNECTED
        self._health_metrics = ProviderHealth(
            provider_name=config.name,
            status=ProviderStatus.DISCONNECTED,
            uptime_percentage=100.0,
            avg_latency_ms=0.0,
            error_rate=0.0,
            requests_today=0,
            rate_limit_remaining=config.rate_limit,
            last_success=None,
            last_error=None,
            quality_score=1.0
        )
        self._request_history: List[float] = []
        self._error_history: List[str] = []
        self._rate_limiter = asyncio.Semaphore(config.rate_limit)
        
        logger.info(f"Initialized {config.name} market data provider")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data provider"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the data provider"""
        pass
    
    @abstractmethod
    async def fetch_ohlcv(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch OHLCV data"""
        pass
    
    @abstractmethod
    async def fetch_quotes(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch quote data (bid/ask)"""
        pass
    
    @abstractmethod
    async def fetch_trades(self, context: MarketDataContext) -> MarketDataResponse:
        """Fetch trade/tick data"""
        pass
    
    @abstractmethod
    async def stream_data(self, context: MarketDataContext) -> AsyncIterator[MarketDataResponse]:
        """Stream real-time data"""
        pass
    
    @abstractmethod
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        pass
    
    async def fetch_data(self, context: MarketDataContext) -> MarketDataResponse:
        """
        Generic fetch method that routes to appropriate specific method
        Similar to Agent.analyze pattern
        """
        start_time = datetime.now()
        
        try:
            # Apply rate limiting
            async with self._rate_limiter:
                # Route to appropriate method based on data types
                if "ohlcv" in context.data_types:
                    response = await self.fetch_ohlcv(context)
                elif "quotes" in context.data_types:
                    response = await self.fetch_quotes(context)
                elif "trades" in context.data_types:
                    response = await self.fetch_trades(context)
                else:
                    raise ValueError(f"Unsupported data types: {context.data_types}")
                
                # Update metrics
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self._update_health_metrics(success=True, latency=latency)
                
                return response
                
        except Exception as e:
            logger.error(f"Error fetching data from {self.config.name}: {str(e)}")
            self._update_health_metrics(success=False, error=str(e))
            raise
    
    def _update_health_metrics(self, success: bool, latency: float = 0, error: str = None):
        """Update provider health metrics"""
        self._health_metrics.requests_today += 1
        
        if success:
            self._health_metrics.last_success = datetime.now()
            self._request_history.append(latency)
            if len(self._request_history) > 100:
                self._request_history.pop(0)
            self._health_metrics.avg_latency_ms = sum(self._request_history) / len(self._request_history)
        else:
            self._health_metrics.last_error = error
            self._error_history.append(error)
            if len(self._error_history) > 100:
                self._error_history.pop(0)
        
        # Calculate error rate
        total_requests = self._health_metrics.requests_today
        error_count = len([e for e in self._error_history if e])
        self._health_metrics.error_rate = error_count / max(total_requests, 1)
        
        # Calculate quality score
        self._calculate_quality_score()
    
    def _calculate_quality_score(self):
        """Calculate overall quality score for the provider"""
        score = 1.0
        
        # Penalize for high error rate
        score -= self._health_metrics.error_rate * 0.5
        
        # Penalize for high latency
        if self._health_metrics.avg_latency_ms > 1000:
            score -= 0.2
        elif self._health_metrics.avg_latency_ms > 500:
            score -= 0.1
        
        # Penalize if disconnected
        if self.status != ProviderStatus.CONNECTED:
            score -= 0.3
        
        self._health_metrics.quality_score = max(0, min(1, score))
    
    def get_health(self) -> ProviderHealth:
        """Get current health metrics"""
        return self._health_metrics
    
    @asynccontextmanager
    async def session(self):
        """Context manager for provider session"""
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()


class MarketDataRegistry:
    """
    Registry pattern for managing multiple data providers
    Similar to service_registry in core.dependencies
    """
    
    def __init__(self):
        self._providers: Dict[str, IMarketDataProvider] = {}
        self._primary_provider: Optional[str] = None
        self._fallback_chain: List[str] = []
        
    def register(self, provider: IMarketDataProvider) -> None:
        """Register a new data provider"""
        self._providers[provider.config.name] = provider
        logger.info(f"Registered market data provider: {provider.config.name}")
        
        # Update fallback chain based on priority
        self._update_fallback_chain()
    
    def unregister(self, name: str) -> None:
        """Unregister a data provider"""
        if name in self._providers:
            del self._providers[name]
            self._update_fallback_chain()
            logger.info(f"Unregistered market data provider: {name}")
    
    def get_provider(self, name: str) -> Optional[IMarketDataProvider]:
        """Get a specific provider by name"""
        return self._providers.get(name)
    
    def get_primary_provider(self) -> Optional[IMarketDataProvider]:
        """Get the primary data provider"""
        if self._primary_provider:
            return self._providers.get(self._primary_provider)
        elif self._fallback_chain:
            return self._providers.get(self._fallback_chain[0])
        return None
    
    def set_primary_provider(self, name: str) -> None:
        """Set the primary data provider"""
        if name in self._providers:
            self._primary_provider = name
            logger.info(f"Set primary provider to: {name}")
    
    def _update_fallback_chain(self) -> None:
        """Update fallback chain based on provider priorities"""
        sorted_providers = sorted(
            self._providers.items(),
            key=lambda x: (x[1].config.priority, -x[1].get_health().quality_score)
        )
        self._fallback_chain = [name for name, _ in sorted_providers if _.config.enabled]
    
    async def fetch_with_fallback(self, context: MarketDataContext) -> Optional[MarketDataResponse]:
        """
        Fetch data with automatic fallback to other providers
        Implements circuit breaker pattern
        """
        errors = []
        
        for provider_name in self._fallback_chain:
            provider = self._providers.get(provider_name)
            if not provider or not provider.config.enabled:
                continue
            
            try:
                logger.debug(f"Attempting to fetch from {provider_name}")
                response = await provider.fetch_data(context)
                return response
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)}")
                logger.warning(f"Provider {provider_name} failed, trying next...")
                continue
        
        logger.error(f"All providers failed. Errors: {errors}")
        return None
    
    def get_all_providers(self) -> Dict[str, IMarketDataProvider]:
        """Get all registered providers"""
        return self._providers.copy()
    
    def get_health_report(self) -> Dict[str, ProviderHealth]:
        """Get health report for all providers"""
        return {
            name: provider.get_health() 
            for name, provider in self._providers.items()
        }


# Global registry instance
market_data_registry = MarketDataRegistry()