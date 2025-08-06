# GoldenSignalsAI Testing Guide

## Overview

This guide covers the comprehensive testing framework for GoldenSignalsAI, including unit tests, integration tests, performance tests, and best practices.

## Table of Contents
1. [Test Structure](#test-structure)
2. [Running Tests](#running-tests)
3. [Writing Tests](#writing-tests)
4. [Test Categories](#test-categories)
5. [Best Practices](#best-practices)
6. [CI/CD Integration](#cicd-integration)

---

## Test Structure

```
backend/
├── tests/
│   ├── conftest.py                    # Shared fixtures and configuration
│   ├── test_data_acquisition.py       # Data source tests
│   ├── test_agent_orchestration.py    # AI agent tests
│   ├── test_websocket_features.py     # WebSocket tests
│   ├── test_signal_generation_integration.py  # Integration tests
│   ├── test_auth.py                   # Authentication tests
│   ├── test_market_data.py            # Market data endpoint tests
│   └── test_trading_agents.py         # Trading agent tests
├── pytest.ini                         # Pytest configuration
├── .coveragerc                        # Coverage configuration
├── requirements-test.txt              # Test dependencies
└── run_tests.py                       # Test runner script
```

---

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test category
pytest -m unit
pytest -m integration
pytest -m websocket
```

### Using the Test Runner

```bash
# Check test requirements
python run_tests.py --check-only

# Run full test suite
python run_tests.py --type suite

# Run specific test file
python run_tests.py --file tests/test_market_data.py

# Run tests with specific markers
python run_tests.py --markers "agents and not slow"
```

### Test Categories

- `unit` - Fast, isolated unit tests
- `integration` - Integration tests requiring multiple components
- `websocket` - WebSocket functionality tests
- `agents` - AI agent tests
- `data` - Data acquisition tests
- `performance` - Performance benchmarks
- `slow` - Long-running tests
- `requires_redis` - Tests requiring Redis
- `requires_db` - Tests requiring database

---

## Writing Tests

### Basic Test Structure

```python
import pytest
from unittest.mock import patch, AsyncMock

class TestMarketDataService:
    """Test market data service functionality"""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing"""
        return MarketDataService()
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self, service, mock_market_data):
        """Test successful quote retrieval"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = mock_market_data
            
            quote = await service.get_quote("AAPL")
            
            assert quote is not None
            assert quote["symbol"] == "AAPL"
            assert quote["price"] > 0
```

### Testing Async Functions

```python
@pytest.mark.asyncio
async def test_websocket_connection(mock_websocket):
    """Test WebSocket connection handling"""
    manager = WebSocketManager()
    
    client_id = await manager.connect(mock_websocket)
    
    assert client_id is not None
    assert client_id in manager.clients
    mock_websocket.accept.assert_called_once()
```

### Testing with Database

```python
@pytest.mark.asyncio
async def test_signal_storage(db_session, test_user):
    """Test storing signals in database"""
    signal_service = SignalService()
    
    signal = await signal_service.create_signal(
        db=db_session,
        symbol="AAPL",
        action="BUY",
        confidence=0.85,
        price=150.00,
        source="TestAgent",
        user_id=test_user.id
    )
    
    assert signal.id is not None
    assert signal.symbol == "AAPL"
```

### Mocking External Services

```python
@pytest.mark.asyncio
async def test_finnhub_fallback(live_provider):
    """Test fallback to Finnhub when primary fails"""
    # Mock primary failure
    with patch('yfinance.Ticker', side_effect=Exception("Error")):
        # Mock Finnhub success
        with patch('finnhub.Client') as mock_client:
            mock_client.return_value.quote.return_value = {
                'c': 150.25,
                'h': 151.00,
                'l': 149.00
            }
            
            quote = await live_provider.get_live_quote("AAPL")
            
            assert quote is not None
            assert quote.source == "finnhub"
```

---

## Test Categories

### 1. Unit Tests

Focus on individual components in isolation.

```python
@pytest.mark.unit
class TestDataValidator:
    def test_validate_quote_valid(self):
        """Test quote validation with valid data"""
        quote = {
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 75000000
        }
        assert validate_quote(quote) is True
    
    def test_validate_quote_invalid_price(self):
        """Test quote validation with invalid price"""
        quote = {"symbol": "AAPL", "price": -10}
        assert validate_quote(quote) is False
```

### 2. Integration Tests

Test interaction between components.

```python
@pytest.mark.integration
class TestSignalGeneration:
    @pytest.mark.asyncio
    async def test_complete_signal_flow(self, pipeline_setup):
        """Test complete flow from data to signal"""
        # Setup market data
        market_data = create_test_market_data()
        
        # Generate signal
        signal = await pipeline_setup["orchestrator"].analyze_market(market_data)
        
        # Verify signal stored
        assert signal is not None
        stored = await pipeline_setup["signal_service"].get_recent_signals()
        assert len(stored) > 0
```

### 3. Performance Tests

Benchmark critical operations.

```python
@pytest.mark.performance
class TestPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, performance_timer):
        """Test handling 100 concurrent requests"""
        performance_timer.start()
        
        tasks = [get_quote("AAPL") for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        performance_timer.stop()
        
        assert performance_timer.elapsed() < 2.0  # Under 2 seconds
        assert all(r is not None for r in results)
```

### 4. WebSocket Tests

Test real-time functionality.

```python
@pytest.mark.websocket
class TestWebSocket:
    @pytest.mark.asyncio
    async def test_signal_broadcast(self, ws_manager):
        """Test broadcasting signals to subscribers"""
        # Connect clients
        client1 = await ws_manager.connect(mock_ws1)
        client2 = await ws_manager.connect(mock_ws2)
        
        # Subscribe to symbol
        await ws_manager.subscribe(client1, "AAPL")
        await ws_manager.subscribe(client2, "AAPL")
        
        # Broadcast signal
        await ws_manager.broadcast_signal(test_signal)
        
        # Verify both received
        assert mock_ws1.send_json.called
        assert mock_ws2.send_json.called
```

---

## Best Practices

### 1. Use Fixtures

```python
@pytest.fixture
async def market_service():
    """Reusable market service fixture"""
    service = MarketDataService()
    yield service
    await service.cleanup()
```

### 2. Mock External Dependencies

```python
@patch('external_api.get_data')
async def test_with_mock(mock_get_data):
    mock_get_data.return_value = {"status": "success"}
    # Test code here
```

### 3. Test Edge Cases

```python
@pytest.mark.parametrize("price,expected", [
    (150.25, True),   # Valid
    (0, False),       # Zero
    (-10, False),     # Negative
    (None, False),    # None
    ("abc", False),   # Invalid type
])
def test_price_validation(price, expected):
    assert validate_price(price) == expected
```

### 4. Use Descriptive Names

```python
# Good
async def test_signal_generation_with_multiple_agents_reaching_buy_consensus():
    pass

# Bad
async def test_sig_gen():
    pass
```

### 5. Test Error Conditions

```python
async def test_api_timeout_handling():
    """Test graceful handling of API timeouts"""
    with patch('aiohttp.ClientSession.get', side_effect=asyncio.TimeoutError):
        result = await fetch_data("AAPL")
        assert result is None  # Should handle gracefully
```

---

## CI/CD Integration

### GitHub Actions Configuration

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost/test
        REDIS_URL: redis://localhost:6379
      run: |
        python run_tests.py --type suite
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: tests
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

---

## Coverage Requirements

- **Overall Coverage**: 85%+
- **Critical Paths**: 95%+
- **New Code**: 90%+

### Viewing Coverage Reports

```bash
# Generate HTML report
pytest --cov=. --cov-report=html

# Open in browser
open htmlcov/index.html
```

### Coverage Configuration

```ini
# .coveragerc
[run]
omit = 
    */tests/*
    */migrations/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Add backend to Python path
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   ```

2. **Database Connection**
   ```bash
   # Use in-memory database for tests
   export DATABASE_URL="sqlite+aiosqlite:///:memory:"
   ```

3. **Async Test Warnings**
   ```python
   # Use pytest-asyncio
   @pytest.mark.asyncio
   async def test_async_function():
       pass
   ```

4. **Flaky Tests**
   ```python
   # Add retries for external services
   @pytest.mark.flaky(reruns=3)
   async def test_external_api():
       pass
   ```

---

## Next Steps

1. **Expand Test Coverage**
   - Add more edge case tests
   - Increase integration test coverage
   - Add load testing scenarios

2. **Improve Test Performance**
   - Parallelize test execution
   - Optimize fixture usage
   - Cache test data

3. **Enhanced Reporting**
   - Set up test dashboards
   - Track test metrics over time
   - Automated failure notifications

Remember: **Good tests are the foundation of reliable software!**