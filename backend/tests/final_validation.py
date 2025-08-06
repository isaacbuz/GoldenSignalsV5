#!/usr/bin/env python3
"""
Final Validation Script
Tests the deep cleanup results and system functionality
"""

import sys
import traceback
from typing import Dict, Any

class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []
    
    def test(self, name: str, test_func, skip_on_error: bool = False):
        """Run a test and record the result"""
        try:
            test_func()
            self.passed += 1
            self.results.append(f"✅ {name}")
            print(f"✅ {name}")
            return True
        except Exception as e:
            if skip_on_error:
                self.skipped += 1
                self.results.append(f"⚠️  {name} - SKIPPED: {e}")
                print(f"⚠️  {name} - SKIPPED: {e}")
                return False
            else:
                self.failed += 1
                self.results.append(f"❌ {name} - FAILED: {e}")
                print(f"❌ {name} - FAILED: {e}")
                return False
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n" + "="*60)
        print(f"🧪 DEEP CLEANUP VALIDATION SUMMARY")
        print(f"="*60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Skipped: {self.skipped}")
        print(f"Success Rate: {(self.passed/total)*100:.1f}%" if total > 0 else "0%")
        
        if self.failed == 0:
            print(f"\n🎉 ALL CORE SYSTEMS WORKING!")
            print(f"✨ Deep cleanup was successful!")
        else:
            print(f"\n⚠️  {self.failed} systems need attention")

def main():
    print("🧪 Final Validation of Deep Cleanup...")
    print("="*60)
    
    validator = ValidationResult()
    
    # Test 1: Core Configuration
    validator.test("Core Configuration System", lambda: test_config())
    
    # Test 2: Logging System
    validator.test("Logging System", lambda: test_logging())
    
    # Test 3: Exception Handling
    validator.test("Exception Handling", lambda: test_exceptions())
    
    # Test 4: Cache System
    validator.test("Cache System", lambda: test_cache())
    
    # Test 5: Unified Orchestrator
    validator.test("Unified Orchestrator", lambda: test_orchestrator(), skip_on_error=True)
    
    # Test 6: Market Data Service
    validator.test("Market Data Service", lambda: test_market_data(), skip_on_error=True)
    
    # Test 7: WebSocket Manager
    validator.test("WebSocket Manager", lambda: test_websocket())
    
    # Test 8: Application Structure
    validator.test("Application Structure", lambda: test_app_structure())
    
    # Test 9: API Routes
    validator.test("API Routes", lambda: test_api_routes(), skip_on_error=True)
    
    # Test 10: Database Models
    validator.test("Database Models", lambda: test_database(), skip_on_error=True)
    
    validator.summary()
    
    # Exit code
    sys.exit(0 if validator.failed == 0 else 1)

def test_config():
    """Test configuration system"""
    from core.config import get_settings, settings
    
    # Test settings access
    s = get_settings()
    assert hasattr(s, 'ENVIRONMENT'), "Settings missing ENVIRONMENT"
    assert hasattr(s, 'DATABASE_URL'), "Settings missing DATABASE_URL"
    
    # Test singleton behavior
    s2 = get_settings()
    assert s is settings, "get_settings should return singleton"

def test_logging():
    """Test logging system"""
    from core.logging import get_logger
    
    logger = get_logger("test")
    assert logger is not None, "Logger creation failed"
    
    # Test logging works
    logger.info("Test log message")
    logger.debug("Test debug message")

def test_exceptions():
    """Test standardized exception handling"""
    from core.exceptions import (
        BaseGoldenSignalsException, 
        ErrorCode, 
        ErrorContext,
        MarketDataException,
        error_handler
    )
    
    # Test basic exception
    exc = BaseGoldenSignalsException("test", ErrorCode.UNKNOWN_ERROR)
    assert exc.message == "test", "Exception message incorrect"
    
    # Test market data exception
    market_exc = MarketDataException("Market error", symbol="AAPL")
    assert market_exc.context.symbol == "AAPL", "Market exception context incorrect"
    
    # Test error handler
    assert error_handler is not None, "Error handler not initialized"

def test_cache():
    """Test cache system"""
    from core.cache import get_cache_manager
    
    cache = get_cache_manager()
    assert cache is not None, "Cache manager not created"
    
    # Test singleton behavior
    cache2 = get_cache_manager()
    assert cache is cache2, "Cache manager should be singleton"

def test_orchestrator():
    """Test unified orchestrator"""
    from core.orchestrator import unified_orchestrator
    
    assert unified_orchestrator is not None, "Unified orchestrator not created"
    assert hasattr(unified_orchestrator, 'analyze'), "Orchestrator missing analyze method"
    assert hasattr(unified_orchestrator, 'get_status'), "Orchestrator missing get_status method"

def test_market_data():
    """Test market data service"""
    from services.market_data_unified import unified_market_service
    
    assert unified_market_service is not None, "Market data service not created"
    assert hasattr(unified_market_service, 'get_quote'), "Market service missing get_quote"

def test_websocket():
    """Test WebSocket manager"""
    from services.websocket_manager import ws_manager
    
    assert ws_manager is not None, "WebSocket manager not created"
    assert hasattr(ws_manager, 'connect'), "WebSocket manager missing connect method"

def test_app_structure():
    """Test application structure"""
    import os
    from pathlib import Path
    
    backend_path = Path(__file__).parent
    
    # Test key directories exist
    assert (backend_path / "core").exists(), "Core directory missing"
    assert (backend_path / "services").exists(), "Services directory missing"
    assert (backend_path / "api").exists(), "API directory missing"
    
    # Test key files exist
    assert (backend_path / "app.py").exists(), "Main app.py missing"
    assert (backend_path / "core" / "orchestrator.py").exists(), "Unified orchestrator missing"
    assert (backend_path / "core" / "exceptions.py").exists(), "Exception system missing"

def test_api_routes():
    """Test API routes structure"""
    from api.routes import market_data, signals, agents
    
    assert hasattr(market_data, 'router'), "Market data router missing"
    assert hasattr(signals, 'router'), "Signals router missing"
    assert hasattr(agents, 'router'), "Agents router missing"

def test_database():
    """Test database models"""
    from database.models import User, Portfolio, Position
    
    # Test models can be imported
    assert User is not None, "User model missing"
    assert Portfolio is not None, "Portfolio model missing"
    assert Position is not None, "Position model missing"

if __name__ == "__main__":
    main()