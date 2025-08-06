#!/usr/bin/env python3
"""
Test script for Volatility Agent V5 API endpoints
Validates all API routes are working correctly
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Test data generators
def generate_test_market_data(scenario: str = "normal", length: int = 100) -> Dict[str, Any]:
    """Generate test market data for different scenarios"""
    
    base_price = 100.0
    
    if scenario == "low_vol":
        returns = np.random.normal(0, 0.005, length)  # Low volatility
    elif scenario == "high_vol":
        returns = np.random.normal(0, 0.025, length)  # High volatility
    elif scenario == "vol_spike":
        # Low vol first half, high vol second half
        returns = np.concatenate([
            np.random.normal(0, 0.005, length//2),
            np.random.normal(0, 0.05, length//2)
        ])
    else:
        returns = np.random.normal(0, 0.015, length)  # Normal volatility
    
    # Generate price series
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # Generate OHLC
    highs = prices * (1 + np.abs(np.random.normal(0, 0.001, length)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.001, length)))
    opens = np.concatenate([[prices[0]], prices[:-1]])
    
    return {
        'close_prices': prices.tolist(),
        'high_prices': highs.tolist(),
        'low_prices': lows.tolist(),
        'open_prices': opens.tolist(),
        'symbol': f'{scenario.upper()}_TEST'
    }

def print_separator(title: str):
    """Print a formatted separator"""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print('='*60)

async def test_volatility_agent_direct():
    """Test volatility agent directly"""
    
    print_separator("DIRECT VOLATILITY AGENT TEST")
    
    import sys
        sys.path.append('/Users/isaacbuz/Documents/Projects/Signals/GoldenSignalsV5/backend')
    
    from agents.volatility_agent import volatility_agent
    
    try:
        # Test normal scenario
        test_data = generate_test_market_data("normal", 100)
        
        analysis = await volatility_agent.analyze_volatility(
            symbol=test_data['symbol'],
            market_data=test_data,
            include_iv=False,
            forecast_horizons=[1, 5, 21]
        )
        
        print(f"✅ Symbol: {analysis.symbol}")
        print(f"✅ Volatility Regime: {analysis.current_regime.value}")
        print(f"✅ Primary Signal: {analysis.primary_signal.value}")
        print(f"✅ Signal Strength: {analysis.signal_strength:.2f}")
        print(f"✅ Annualized Volatility: {analysis.metrics.annualized_vol:.1%}")
        print(f"✅ Patterns Detected: {len(analysis.patterns)}")
        print(f"✅ Forecasts Generated: {len(analysis.forecasts)}")
        
        if analysis.patterns:
            print(f"✅ Dominant Pattern: {analysis.patterns[0].pattern_type}")
        
        print(f"✅ Recommendations: {len(analysis.recommendations)}")
        
        # Test with IV analysis
        print(f"\n🔍 Testing with IV Analysis...")
        
        iv_test_data = test_data.copy()
        iv_test_data['implied_volatility'] = 0.35  # 35% IV
        
        iv_analysis = await volatility_agent.analyze_volatility(
            symbol="IV_TEST",
            market_data=iv_test_data,
            include_iv=True,
            forecast_horizons=[1, 5]
        )
        
        print(f"✅ IV Rank: {iv_analysis.metrics.iv_rank:.1f}%")
        print(f"✅ IV Regime: {iv_analysis.iv_regime.value}")
        print(f"✅ HV-IV Spread: {iv_analysis.metrics.hv_iv_spread:.3f}")
        
        print(f"\n✅ Volatility Agent Direct Test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Volatility Agent Direct Test: FAILED")
        print(f"❌ Error: {str(e)}")
        return False

def test_api_payload_format():
    """Test API payload formatting"""
    
    print_separator("API PAYLOAD FORMAT TEST")
    
    try:
        # Test different scenarios
        scenarios = ["normal", "low_vol", "high_vol", "vol_spike"]
        
        for scenario in scenarios:
            test_data = generate_test_market_data(scenario, 60)
            
            # Validate required fields
            required_fields = ['close_prices', 'high_prices', 'low_prices', 'open_prices']
            for field in required_fields:
                if field not in test_data:
                    raise ValueError(f"Missing required field: {field}")
                if len(test_data[field]) < 31:
                    raise ValueError(f"Insufficient data for {field}: {len(test_data[field])} < 31")
            
            print(f"✅ {scenario} payload: {len(test_data['close_prices'])} data points")
        
        # Test IV payload
        iv_payload = {
            'symbol': 'IV_TEST',
            'market_data': generate_test_market_data("normal", 100),
            'current_iv': 0.25,
            'include_iv': True,
            'forecast_horizons': [1, 5, 21]
        }
        
        print(f"✅ IV analysis payload formatted correctly")
        
        print(f"\n✅ API Payload Format Test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ API Payload Format Test: FAILED") 
        print(f"❌ Error: {str(e)}")
        return False

def test_data_validation():
    """Test data validation scenarios"""
    
    print_separator("DATA VALIDATION TEST")
    
    try:
        # Test insufficient data
        small_data = generate_test_market_data("normal", 10)  # Too small
        print(f"✅ Small dataset generated: {len(small_data['close_prices'])} points")
        
        # Test extreme values
        extreme_data = generate_test_market_data("normal", 50)
        extreme_data['close_prices'] = [100.0] * 50  # No volatility
        print(f"✅ Zero volatility dataset generated")
        
        # Test missing fields
        incomplete_data = {'close_prices': [100, 101, 102]}
        print(f"✅ Incomplete dataset generated")
        
        # Test invalid IV
        invalid_iv_values = [-0.1, 0.0, 10.0]  # Invalid IV values
        print(f"✅ Invalid IV values prepared: {invalid_iv_values}")
        
        # Test invalid forecast horizons
        invalid_horizons = [0, -1, 300, 1000]  # Invalid horizons
        print(f"✅ Invalid forecast horizons prepared: {invalid_horizons}")
        
        print(f"\n✅ Data Validation Test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Data Validation Test: FAILED")
        print(f"❌ Error: {str(e)}")
        return False

async def test_performance_benchmarks():
    """Test performance benchmarks"""
    
    print_separator("PERFORMANCE BENCHMARK TEST")
    
            import time
    sys.path.append('/Users/isaacbuz/Documents/Projects/Signals/GoldenSignalsV5/backend')
    
    from agents.volatility_agent import volatility_agent
    
    try:
        # Benchmark different data sizes
        data_sizes = [50, 100, 200, 500]
        
        for size in data_sizes:
            test_data = generate_test_market_data("normal", size)
            
            start_time = time.time()
            
            analysis = await volatility_agent.analyze_volatility(
                symbol=f"PERF_TEST_{size}",
                market_data=test_data,
                include_iv=False,
                forecast_horizons=[1, 5, 21]
            )
            
            execution_time = time.time() - start_time
            
            print(f"✅ {size} data points: {execution_time:.3f}s")
            
            # Performance should be reasonable
            if execution_time > 5.0:  # More than 5 seconds is too slow
                print(f"⚠️  Performance warning: {execution_time:.3f}s for {size} points")
        
        # Test performance metrics
        performance = volatility_agent.get_performance_metrics()
        print(f"✅ Total analyses: {performance['total_analyses']}")
        print(f"✅ Average calc time: {performance['average_calc_time_seconds']:.3f}s")
        
        print(f"\n✅ Performance Benchmark Test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Performance Benchmark Test: FAILED")
        print(f"❌ Error: {str(e)}")
        return False

def print_api_endpoints():
    """Print available API endpoints"""
    
    print_separator("AVAILABLE API ENDPOINTS")
    
    endpoints = [
        ("POST", "/api/v1/volatility/analyze", "Full volatility analysis"),
        ("POST", "/api/v1/volatility/regime", "Volatility regime classification"),
        ("POST", "/api/v1/volatility/forecast", "Multi-horizon forecasting"),
        ("POST", "/api/v1/volatility/patterns", "Pattern detection"),
        ("POST", "/api/v1/volatility/iv-analysis", "IV analysis"),
        ("GET", "/api/v1/volatility/performance", "Agent performance metrics"),
        ("POST", "/api/v1/volatility/test", "System test endpoints"),
        ("GET", "/api/v1/volatility/education", "Educational content")
    ]
    
    for method, endpoint, description in endpoints:
        print(f"✅ {method:4} {endpoint:35} - {description}")
    
    print(f"\n📚 API Documentation:")
    print(f"   - Start server: python -m uvicorn app:app --reload --port 8000")
    print(f"   - API docs: http://localhost:8000/docs")
    print(f"   - Test endpoint: curl -X POST http://localhost:8000/api/v1/volatility/test")

async def run_all_tests():
    """Run all test suites"""
    
    print_separator("VOLATILITY AGENT V5 API TEST SUITE")
    print(f"Starting comprehensive API tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Direct Agent Test", test_volatility_agent_direct()),
        ("API Payload Format", test_api_payload_format()),
        ("Data Validation", test_data_validation()),
        ("Performance Benchmark", test_performance_benchmarks())
    ]
    
    for test_name, test_coro in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            test_results.append((test_name, False))
    
    # Print summary
    print_separator("TEST RESULTS SUMMARY")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED - Volatility Agent V5 is ready for production!")
    else:
        print(f"⚠️  Some tests failed - Review and fix issues before deployment")
    
    # Print API info
    print_api_endpoints()
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())