"""
Base interfaces to prevent circular dependencies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class IMarketDataService(ABC):
    """Interface for market data services"""
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError("get_quote must be implemented")
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("get_historical_data must be implemented")

class IOrchestrator(ABC):
    """Interface for orchestrator services"""
    
    @abstractmethod
    async def analyze(self, symbol: str, **kwargs) -> Any:
        raise NotImplementedError("analyze must be implemented")
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        raise NotImplementedError("get_status must be implemented")

class IRAGService(ABC):
    """Interface for RAG services"""
    
    @abstractmethod
    async def query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError("query must be implemented")
    
    @abstractmethod
    async def ingest_documents(self, documents: List[Dict[str, Any]]) -> int:
        raise NotImplementedError("ingest_documents must be implemented")

class IWebSocketManager(ABC):
    """Interface for WebSocket management"""
    
    @abstractmethod
    async def broadcast_signal(self, data: Dict[str, Any]):
        raise NotImplementedError("broadcast_signal must be implemented")
    
    @abstractmethod
    async def broadcast_price_update(self, symbol: str, price: float, volume: int):
        raise NotImplementedError("broadcast_price_update must be implemented")
