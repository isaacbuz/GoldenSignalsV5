"""
Dependency Injection and Service Registry
Centralized management of application services
"""

from typing import Any, Dict, Optional, Type, TypeVar, Callable
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from redis import asyncio as aioredis

from core.database import get_db
from core.config import settings
from core.logging import get_logger
from services.market_data_service import MarketDataService
from services.signal_service import SignalService
from services.websocket_manager import WebSocketManager
from agents.orchestrator import AgentOrchestrator
from rag.core import RAGEngine
from mcp.servers.market_data import MarketDataMCPServer
from mcp.client import MCPClient

logger = get_logger(__name__)

T = TypeVar('T')


class ServiceRegistry:
    """
    Central registry for all application services
    Implements singleton pattern for service instances
    """
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._initialized = False
        
    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for a service type"""
        self._factories[service_type] = factory
        logger.debug(f"Registered factory for {service_type.__name__}")
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register a singleton instance directly"""
        self._services[service_type] = instance
        logger.debug(f"Registered instance for {service_type.__name__}")
    
    def get(self, service_type: Type[T]) -> T:
        """Get or create a service instance"""
        if service_type not in self._services:
            if service_type in self._factories:
                instance = self._factories[service_type]()
                self._services[service_type] = instance
                logger.info(f"Created service instance: {service_type.__name__}")
            else:
                raise ValueError(f"No factory registered for {service_type.__name__}")
        
        return self._services[service_type]
    
    async def initialize_async_services(self) -> None:
        """Initialize all async services"""
        if self._initialized:
            return
            
        logger.info("Initializing async services...")
        
        # Initialize services that require async setup
        tasks = []
        
        # RAG Engine
        if RAGEngine in self._factories:
            rag_engine = self.get(RAGEngine)
            tasks.append(rag_engine.initialize())
        
        # Agent Orchestrator
        if AgentOrchestrator in self._factories:
            orchestrator = self.get(AgentOrchestrator)
            tasks.append(orchestrator.initialize_default_agents())
            tasks.append(orchestrator.start())
        
        # MCP Servers
        if MarketDataMCPServer in self._factories:
            mcp_server = self.get(MarketDataMCPServer)
            tasks.append(mcp_server.start())
        
        # Wait for all initializations
        if tasks:
            await asyncio.gather(*tasks)
        
        self._initialized = True
        logger.info("All async services initialized")
    
    async def shutdown_async_services(self) -> None:
        """Shutdown all async services"""
        logger.info("Shutting down async services...")
        
        tasks = []
        
        # Shutdown services in reverse order
        if AgentOrchestrator in self._services:
            orchestrator = self._services[AgentOrchestrator]
            tasks.append(orchestrator.shutdown())
        
        if MarketDataMCPServer in self._services:
            mcp_server = self._services[MarketDataMCPServer]
            tasks.append(mcp_server.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._services.clear()
        self._initialized = False
        logger.info("All async services shutdown")


# Global service registry
service_registry = ServiceRegistry()


# Service factory functions

def create_market_data_service() -> MarketDataService:
    """Factory for MarketDataService"""
    return MarketDataService()


def create_signal_service() -> SignalService:
    """Factory for SignalService"""
    return SignalService()


def create_websocket_manager() -> WebSocketManager:
    """Factory for WebSocketManager"""
    return WebSocketManager()


async def create_redis_client() -> aioredis.Redis:
    """Factory for Redis client"""
    client = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    await client.ping()
    logger.info("Redis client created")
    return client


def create_rag_engine() -> RAGEngine:
    """Factory for RAG Engine"""
    return RAGEngine()


def create_agent_orchestrator() -> AgentOrchestrator:
    """Factory for Agent Orchestrator"""
    signal_service = service_registry.get(SignalService)
    return AgentOrchestrator(signal_service=signal_service)


def create_market_data_mcp_server() -> MarketDataMCPServer:
    """Factory for Market Data MCP Server"""
    return MarketDataMCPServer()


# Register all factories
def setup_service_registry():
    """Setup service registry with all factories"""
    service_registry.register_factory(MarketDataService, create_market_data_service)
    service_registry.register_factory(SignalService, create_signal_service)
    service_registry.register_factory(WebSocketManager, create_websocket_manager)
    service_registry.register_factory(RAGEngine, create_rag_engine)
    service_registry.register_factory(AgentOrchestrator, create_agent_orchestrator)
    service_registry.register_factory(MarketDataMCPServer, create_market_data_mcp_server)
    
    logger.info("Service registry configured")


# FastAPI Dependencies

def get_market_data_service() -> MarketDataService:
    """Dependency for MarketDataService"""
    return service_registry.get(MarketDataService)


def get_signal_service() -> SignalService:
    """Dependency for SignalService"""
    return service_registry.get(SignalService)


def get_websocket_manager() -> WebSocketManager:
    """Dependency for WebSocketManager"""
    return service_registry.get(WebSocketManager)


def get_rag_engine() -> RAGEngine:
    """Dependency for RAG Engine"""
    return service_registry.get(RAGEngine)


def get_agent_orchestrator() -> AgentOrchestrator:
    """Dependency for Agent Orchestrator"""
    return service_registry.get(AgentOrchestrator)


async def get_redis() -> aioredis.Redis:
    """Dependency for Redis client"""
    # For Redis, we might want a new connection per request
    # or use a connection pool
    client = await create_redis_client()
    try:
        yield client
    finally:
        await client.close()


# Composite dependencies

class ServiceDependencies:
    """Container for commonly used service dependencies"""
    
    def __init__(
        self,
        db: AsyncSession = Depends(get_db),
        market_service: MarketDataService = Depends(get_market_data_service),
        signal_service: SignalService = Depends(get_signal_service),
        ws_manager: WebSocketManager = Depends(get_websocket_manager),
        rag_engine: RAGEngine = Depends(get_rag_engine),
        orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
    ):
        self.db = db
        self.market_service = market_service
        self.signal_service = signal_service
        self.ws_manager = ws_manager
        self.rag_engine = rag_engine
        self.orchestrator = orchestrator


# Health check dependencies

async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        async for db in get_db():
            result = await db.execute("SELECT 1")
            return {"database": "healthy", "connected": True}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"database": "unhealthy", "error": str(e)}


async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity"""
    try:
        async with get_redis() as redis:
            await redis.ping()
            return {"redis": "healthy", "connected": True}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"redis": "unhealthy", "error": str(e)}


async def check_services_health() -> Dict[str, Any]:
    """Check all services health"""
    health_status = {}
    
    # Check if services are registered
    for service_type in [MarketDataService, SignalService, WebSocketManager, AgentOrchestrator]:
        try:
            service = service_registry.get(service_type)
            health_status[service_type.__name__] = "healthy"
        except Exception as e:
            health_status[service_type.__name__] = f"unhealthy: {str(e)}"
    
    return health_status


# Configuration dependencies

@lru_cache()
def get_settings():
    """Get cached settings instance"""
    return settings


# Initialize service registry on module import
setup_service_registry()