"""
Pytest configuration and fixtures for GoldenSignalsAI testing
"""

import pytest
import asyncio
import sys
import os
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from core.database import Base, get_db
from core.config import settings
from models.user import User
from models.signal import Signal
from core.auth import get_password_hash
from fastapi.testclient import TestClient


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create a test database engine"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
    async_session_maker = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def client(db_session):
    """Create a test client with database override"""
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(db_session) -> User:
    """Create a test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword"),
        is_active=True,
        is_superuser=False
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_admin_user(db_session) -> User:
    """Create a test admin user"""
    user = User(
        username="adminuser",
        email="admin@example.com",
        hashed_password=get_password_hash("adminpassword"),
        is_active=True,
        is_superuser=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers for test user"""
    from core.auth import create_access_token
    
    access_token = create_access_token(
        data={"sub": test_user.username}
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def admin_auth_headers(test_admin_user):
    """Create authentication headers for admin user"""
    from core.auth import create_access_token
    
    access_token = create_access_token(
        data={"sub": test_admin_user.username}
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
async def test_signal(db_session, test_user) -> Signal:
    """Create a test signal"""
    signal = Signal(
        symbol="AAPL",
        action="BUY",
        confidence=0.85,
        price=150.00,
        source="TestAgent",
        user_id=test_user.id,
        metadata={
            "reasoning": ["Test reasoning"],
            "indicators": {"rsi": 45, "macd": 0.5}
        }
    )
    db_session.add(signal)
    await db_session.commit()
    await db_session.refresh(signal)
    return signal


@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 75000000,
        "change": 0.75,
        "change_percent": 0.5,
        "bid": 150.20,
        "ask": 150.30,
        "day_high": 151.00,
        "day_low": 149.00,
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def mock_historical_data():
    """Mock historical data for testing"""
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    return {
        "dates": [d.isoformat() for d in dates],
        "prices": [150 + i * 0.5 for i in range(30)],
        "volumes": [75000000 + i * 100000 for i in range(30)]
    }


@pytest.fixture
def mock_finnhub_client():
    """Mock Finnhub client"""
    client = MagicMock()
    client.quote.return_value = {
        'c': 150.25,  # Current price
        'h': 151.00,  # High
        'l': 149.00,  # Low
        'o': 149.50,  # Open
        'pc': 149.50, # Previous close
        't': 1234567890
    }
    return client


@pytest.fixture
def mock_yfinance():
    """Mock yfinance module"""
    mock = MagicMock()
    ticker = MagicMock()
    ticker.info = {
        'regularMarketPrice': 150.25,
        'regularMarketVolume': 75000000,
        'regularMarketDayHigh': 151.00,
        'regularMarketDayLow': 149.00
    }
    mock.Ticker.return_value = ticker
    return mock


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing"""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    from agents.base import BaseAgent, Signal, SignalAction, SignalStrength
    
    agent = AsyncMock(spec=BaseAgent)
    agent.agent_id = "test-agent-123"
    agent.config.name = "TestAgent"
    agent.config.enabled = True
    agent.config.weight = 1.0
    agent.config.timeout = 30
    
    # Mock signal
    signal = Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.85,
        strength=SignalStrength.STRONG,
        source="TestAgent",
        current_price=150.00,
        target_price=160.00,
        stop_loss=145.00,
        reasoning=["Test reasoning 1", "Test reasoning 2"],
        features={"test": "features"},
        indicators={"rsi": 45, "macd": 0.5}
    )
    
    agent.execute_with_monitoring.return_value = signal
    
    return agent


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    import aioredis
    from unittest.mock import create_autospec
    
    redis = create_autospec(aioredis.Redis, spec_set=True)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.ttl = AsyncMock(return_value=-1)
    redis.keys = AsyncMock(return_value=[])
    
    return redis


# Test data fixtures
@pytest.fixture
def sample_symbols():
    """Sample stock symbols for testing"""
    return ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]


@pytest.fixture
def sample_signals():
    """Sample signals for testing"""
    return [
        {
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 0.85,
            "price": 150.00
        },
        {
            "symbol": "GOOGL",
            "action": "SELL",
            "confidence": 0.75,
            "price": 2800.00
        },
        {
            "symbol": "MSFT",
            "action": "HOLD",
            "confidence": 0.60,
            "price": 380.00
        }
    ]


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# WebSocket testing helpers
@pytest.fixture
async def websocket_client():
    """Create a WebSocket test client"""
    from websockets.client import connect
    
    async def _create_client(uri="ws://localhost:8000/ws/signals"):
        client = await connect(uri)
        yield client
        await client.close()
    
    return _create_client