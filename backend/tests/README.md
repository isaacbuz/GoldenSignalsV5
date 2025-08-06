# GoldenSignalsAI V5 Test Suite

This directory contains all test files for the GoldenSignalsAI V5 backend, organized by functionality.

## 📁 Test Directory Structure

```
tests/
├── README.md                    # This file
├── conftest.py                 # Pytest configuration and fixtures
├── final_validation.py        # Final system validation tests
├── run_tests.py               # Test runner script (moved from root)
├── pytest.ini                # Pytest settings
│
├── agents/                    # Base agent framework tests
│   ├── __init__.py
│   ├── test_base_agent.py    # BaseAgent functionality tests
│   └── ...
│
├── agents_tests/             # Specific agent implementation tests
│   ├── __init__.py
│   ├── test_gamma_exposure.py
│   ├── test_liquidity_prediction.py
│   ├── test_market_regime.py
│   ├── test_options_flow.py
│   └── test_risk_management.py
│
├── api/                      # Existing API tests
│   ├── __init__.py
│   ├── test_agents_api.py
│   ├── test_auth_api.py
│   └── test_market_data_api.py
│
├── api_tests/               # Additional API tests
│   ├── __init__.py
│   ├── test_api_keys.py
│   └── test_backend.py
│
├── services/                # Base service tests
│   ├── __init__.py
│   └── test_data_validator.py
│
├── services_tests/          # Specific service tests
│   ├── __init__.py
│   ├── test_data_sources.py
│   ├── test_meta_orchestrator.py
│   ├── test_universal_data.py
│   └── quick_test_risk.py
│
├── ml_tests/               # Machine learning tests
│   ├── __init__.py
│   └── test_ml_training.py
│
├── websocket_tests/        # WebSocket functionality tests
│   ├── __init__.py
│   ├── test_websocket.py
│   └── test_websocket_orchestration.py
│
├── integration/           # Integration tests
│   ├── test_signal_generation_integration.py
│   └── test_websocket_integration.py
│
└── performance/          # Performance tests
    └── test_performance_benchmarks.py
```

## 🚀 Running Tests

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Categories
```bash
# Run only agent tests
pytest tests/agents_tests/ -v

# Run only API tests
pytest tests/api/ tests/api_tests/ -v

# Run only service tests
pytest tests/services/ tests/services_tests/ -v

# Run only ML tests  
pytest tests/ml_tests/ -v

# Run only WebSocket tests
pytest tests/websocket_tests/ -v
```

### Run Individual Test Files
```bash
# Run a specific test file
pytest tests/agents/test_base_agent.py -v

# Run with coverage
pytest tests/services/test_data_validator.py --cov=services.data_validator -v
```

### Run Tests by Markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run only performance tests
pytest -m performance
```

## 🧪 Test Categories

### **Agents Tests**
- **Base Framework** (`tests/agents/`): Core agent architecture tests
- **Specific Agents** (`tests/agents_tests/`): Individual agent implementation tests

### **API Tests** 
- **Core API** (`tests/api/`): Main API endpoint tests
- **Additional API** (`tests/api_tests/`): Authentication, keys, and backend tests

### **Services Tests**
- **Base Services** (`tests/services/`): Core service framework tests  
- **Specific Services** (`tests/services_tests/`): Data sources, orchestrator, and utility service tests

### **ML Tests**
- **Model Training** (`tests/ml_tests/`): Machine learning model and training pipeline tests

### **WebSocket Tests**
- **Real-time Features** (`tests/websocket_tests/`): WebSocket communication and orchestration tests

### **Integration Tests**
- **End-to-End** (`tests/integration/`): Cross-component integration tests

### **Performance Tests**
- **Benchmarks** (`tests/performance/`): Performance and load testing

## ⚙️ Configuration

### Pytest Configuration
The `pytest.ini` file contains test configuration including:
- Test markers (unit, integration, performance)
- Coverage settings
- Test discovery patterns

### Test Fixtures
The `conftest.py` file provides shared test fixtures:
- Database setup/teardown
- Mock services
- Sample data generators

## 📊 Coverage Reports

After running tests with coverage, reports are generated in:
- `htmlcov/` - HTML coverage report
- `coverage.json` - JSON coverage data
- Terminal output shows coverage summary

## 🔧 Writing Tests

When adding new tests:
1. Place in appropriate directory by functionality
2. Follow naming convention: `test_*.py`
3. Use appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`
4. Include docstrings describing test purpose
5. Use fixtures for common setup/teardown

## 📋 Test Quality Guidelines

- **Unit Tests**: Fast, isolated, test single functions/methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Measure and validate performance metrics
- **Use Mocks**: Mock external dependencies in unit tests
- **Clear Assertions**: Use descriptive assertion messages
- **Test Edge Cases**: Include boundary conditions and error cases

---

*Updated: 2025-08-06*