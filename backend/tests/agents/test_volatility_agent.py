"""
Comprehensive Test Suite for Volatility Agent V5
Tests all volatility analysis functionality including patterns, forecasting, and signals
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import the agent
from agents.volatility_agent import (
    volatility_agent, 
    VolatilityRegime, 
    VolatilitySignal, 
    IVRegime
)
from core.logging import get_logger

logger = get_logger(__name__)

class VolatilityAgentTester:
    """Comprehensive test suite for Volatility Agent"""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    def generate_test_data(self, scenario: str = "normal", length: int = 100) -> Dict[str, Any]:
        """Generate test market data for different volatility scenarios"""
        
        base_price = 100.0
        dates = pd.date_range(start='2024-01-01', periods=length, freq='D')
        
        if scenario == "low_vol":
            # Low volatility scenario
            returns = np.random.normal(0, 0.005, length)  # 0.5% daily vol ≈ 8% annual
            noise_factor = 0.3
        elif scenario == "high_vol":
            # High volatility scenario  
            returns = np.random.normal(0, 0.025, length)  # 2.5% daily vol ≈ 40% annual
            noise_factor = 1.5
        elif scenario == "vol_spike":
            # Volatility spike scenario (low then high)
            returns = np.concatenate([
                np.random.normal(0, 0.005, length//2),  # Low vol first half
                np.random.normal(0, 0.05, length//2)    # High vol second half
            ])
            noise_factor = 1.0
        elif scenario == "vol_squeeze":
            # Volatility squeeze (very low vol)
            returns = np.random.normal(0, 0.002, length)  # Very low vol
            noise_factor = 0.1
        elif scenario == "garch_clustering":
            # GARCH-like volatility clustering
            returns = []
            vol = 0.01  # Starting volatility
            for i in range(length):
                # GARCH-like process
                vol = 0.9 * vol + 0.05 * abs(returns[-1] if returns else 0) + 0.05 * np.random.normal(0, 0.01)
                ret = np.random.normal(0, max(0.001, vol))
                returns.append(ret)
            returns = np.array(returns)
            noise_factor = 1.0
        else:
            # Normal volatility scenario
            returns = np.random.normal(0, 0.015, length)  # Normal vol ≈ 24% annual
            noise_factor = 1.0
        
        # Generate price series
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])  # Remove first element
        
        # Generate OHLC data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.002 * noise_factor, length)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.002 * noise_factor, length)))
        opens = np.concatenate([[prices[0]], prices[:-1]])  # Opens are previous close
        
        # Optional: Add implied volatility data
        iv_base = np.std(returns) * np.sqrt(252) * 1.15  # IV premium to HV
        iv_data = iv_base + np.random.normal(0, iv_base * 0.1, length)
        
        return {
            'close_prices': prices.tolist(),
            'high_prices': highs.tolist(), 
            'low_prices': lows.tolist(),
            'open_prices': opens.tolist(),
            'implied_volatility': iv_data[-1],  # Current IV
            'historical_iv': iv_data.tolist(),   # Historical IV series
            'symbol': f'{scenario.upper()}_TEST',
            'scenario': scenario,
            'expected_regime': self._get_expected_regime(scenario)
        }
    
    def _get_expected_regime(self, scenario: str) -> VolatilityRegime:
        """Get expected volatility regime for scenario"""
        regime_map = {
            'low_vol': VolatilityRegime.LOW,
            'vol_squeeze': VolatilityRegime.EXTREMELY_LOW,
            'high_vol': VolatilityRegime.HIGH,
            'vol_spike': VolatilityRegime.EXTREMELY_HIGH,
            'normal': VolatilityRegime.NORMAL,
            'garch_clustering': VolatilityRegime.HIGH
        }
        return regime_map.get(scenario, VolatilityRegime.NORMAL)
    
    async def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic volatility analysis functionality"""
        logger.info("Testing basic volatility analysis...")
        
        test_data = self.generate_test_data("normal", 100)
        
        try:
            analysis = await volatility_agent.analyze_volatility(
                symbol=test_data['symbol'],
                market_data=test_data,
                include_iv=True,
                forecast_horizons=[1, 5, 21]
            )
            
            # Validate analysis structure
            assert hasattr(analysis, 'symbol')
            assert hasattr(analysis, 'metrics') 
            assert hasattr(analysis, 'current_regime')
            assert hasattr(analysis, 'patterns')
            assert hasattr(analysis, 'primary_signal')
            assert hasattr(analysis, 'forecasts')
            assert hasattr(analysis, 'recommendations')
            
            # Validate metrics
            assert analysis.metrics.daily_vol > 0
            assert analysis.metrics.annualized_vol > 0
            assert 0 <= analysis.metrics.vol_percentile <= 100
            
            # Validate forecasts
            assert len(analysis.forecasts) == 3
            for forecast in analysis.forecasts:
                assert forecast.horizon_days in [1, 5, 21]
                assert forecast.forecasted_vol > 0
                
            self.passed_tests += 1
            return {
                'test': 'basic_functionality',
                'status': 'PASSED',
                'analysis_summary': {
                    'regime': analysis.current_regime.value,
                    'signal': analysis.primary_signal.value,
                    'patterns_detected': len(analysis.patterns),
                    'forecasts_generated': len(analysis.forecasts)
                }
            }
            
        except Exception as e:
            return {
                'test': 'basic_functionality', 
                'status': 'FAILED',
                'error': str(e)
            }
        finally:
            self.total_tests += 1
    
    async def test_regime_classification(self) -> Dict[str, Any]:
        """Test volatility regime classification accuracy"""
        logger.info("Testing volatility regime classification...")
        
        scenarios = ['low_vol', 'high_vol', 'vol_squeeze', 'vol_spike']
        results = {}
        
        for scenario in scenarios:
            try:
                test_data = self.generate_test_data(scenario, 100)
                expected_regime = test_data['expected_regime']
                
                analysis = await volatility_agent.analyze_volatility(
                    symbol=test_data['symbol'],
                    market_data=test_data,
                    include_iv=False,
                    forecast_horizons=[1]
                )
                
                actual_regime = analysis.current_regime
                
                results[scenario] = {
                    'expected': expected_regime.value,
                    'actual': actual_regime.value,
                    'vol_percentile': analysis.metrics.vol_percentile,
                    'vol_ratio': analysis.metrics.vol_ratio,
                    'correct': self._regime_classification_correct(expected_regime, actual_regime)
                }
                
            except Exception as e:
                results[scenario] = {'error': str(e)}
        
        # Calculate accuracy
        correct_classifications = sum(1 for r in results.values() if r.get('correct', False))
        accuracy = correct_classifications / len(scenarios)
        
        if accuracy >= 0.75:  # 75% accuracy threshold
            self.passed_tests += 1
            status = 'PASSED'
        else:
            status = 'FAILED'
            
        self.total_tests += 1
        
        return {
            'test': 'regime_classification',
            'status': status,
            'accuracy': f"{accuracy:.2%}",
            'results': results
        }
    
    def _regime_classification_correct(self, expected: VolatilityRegime, actual: VolatilityRegime) -> bool:
        """Check if regime classification is reasonable"""
        # Allow some flexibility in classification
        low_regimes = [VolatilityRegime.EXTREMELY_LOW, VolatilityRegime.LOW]
        high_regimes = [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH]
        
        if expected in low_regimes and actual in low_regimes:
            return True
        elif expected in high_regimes and actual in high_regimes:
            return True
        elif expected == actual:
            return True
        else:
            return False
    
    async def test_pattern_detection(self) -> Dict[str, Any]:
        """Test volatility pattern detection"""
        logger.info("Testing volatility pattern detection...")
        
        scenarios = {
            'vol_squeeze': ['volatility_squeeze'],
            'vol_spike': ['volatility_spike'],
            'garch_clustering': ['volatility_clustering']
        }
        
        results = {}
        
        for scenario, expected_patterns in scenarios.items():
            try:
                test_data = self.generate_test_data(scenario, 100)
                
                analysis = await volatility_agent.analyze_volatility(
                    symbol=test_data['symbol'],
                    market_data=test_data,
                    include_iv=False,
                    forecast_horizons=[1]
                )
                
                detected_patterns = [p.pattern_type for p in analysis.patterns]
                
                # Check if any expected pattern was detected
                pattern_found = any(expected in detected_patterns for expected in expected_patterns)
                
                results[scenario] = {
                    'expected_patterns': expected_patterns,
                    'detected_patterns': detected_patterns,
                    'pattern_found': pattern_found,
                    'total_patterns': len(analysis.patterns)
                }
                
            except Exception as e:
                results[scenario] = {'error': str(e)}
        
        # Calculate success rate
        successful_detections = sum(1 for r in results.values() if r.get('pattern_found', False))
        success_rate = successful_detections / len(scenarios)
        
        if success_rate >= 0.66:  # 66% success threshold
            self.passed_tests += 1
            status = 'PASSED'
        else:
            status = 'FAILED'
            
        self.total_tests += 1
        
        return {
            'test': 'pattern_detection',
            'status': status,
            'success_rate': f"{success_rate:.2%}",
            'results': results
        }
    
    async def test_iv_analysis(self) -> Dict[str, Any]:
        """Test implied volatility analysis"""
        logger.info("Testing IV analysis...")
        
        try:
            test_data = self.generate_test_data("normal", 100)
            
            analysis = await volatility_agent.analyze_volatility(
                symbol=test_data['symbol'],
                market_data=test_data,
                include_iv=True,
                forecast_horizons=[1, 5]
            )
            
            # Validate IV metrics
            assert analysis.metrics.iv_rank is not None
            assert 0 <= analysis.metrics.iv_rank <= 100
            assert analysis.iv_regime is not None
            assert analysis.metrics.hv_iv_spread is not None
            
            # Test different IV scenarios
            high_iv_data = test_data.copy()
            high_iv_data['implied_volatility'] = 0.8  # Very high IV
            
            high_iv_analysis = await volatility_agent.analyze_volatility(
                symbol="HIGH_IV_TEST",
                market_data=high_iv_data,
                include_iv=True,
                forecast_horizons=[1]
            )
            
            # High IV should result in overvalued regime
            assert high_iv_analysis.iv_regime in [IVRegime.OVERVALUED, IVRegime.EXTREMELY_OVERVALUED]
            
            self.passed_tests += 1
            return {
                'test': 'iv_analysis',
                'status': 'PASSED',
                'normal_iv_rank': analysis.metrics.iv_rank,
                'high_iv_regime': high_iv_analysis.iv_regime.value,
                'hv_iv_spread': analysis.metrics.hv_iv_spread
            }
            
        except Exception as e:
            return {
                'test': 'iv_analysis',
                'status': 'FAILED', 
                'error': str(e)
            }
        finally:
            self.total_tests += 1
    
    async def test_forecasting(self) -> Dict[str, Any]:
        """Test volatility forecasting"""
        logger.info("Testing volatility forecasting...")
        
        try:
            test_data = self.generate_test_data("normal", 100)
            
            analysis = await volatility_agent.analyze_volatility(
                symbol=test_data['symbol'],
                market_data=test_data,
                include_iv=False,
                forecast_horizons=[1, 5, 10, 21, 63]
            )
            
            # Validate forecasts
            assert len(analysis.forecasts) == 5
            
            for forecast in analysis.forecasts:
                # Basic validation
                assert forecast.forecasted_vol > 0
                assert forecast.expected_move > 0
                assert len(forecast.confidence_interval) == 2
                assert forecast.confidence_interval[0] < forecast.confidence_interval[1]
                assert forecast.regime_forecast in VolatilityRegime
                
                # Reasonable forecast range
                current_vol = analysis.metrics.annualized_vol
                assert 0.5 * current_vol < forecast.forecasted_vol < 3.0 * current_vol
            
            # Test forecast consistency (longer horizons should have wider confidence intervals)
            short_forecast = analysis.forecasts[0]  # 1 day
            long_forecast = analysis.forecasts[-1]   # 63 days
            
            short_width = short_forecast.confidence_interval[1] - short_forecast.confidence_interval[0]
            long_width = long_forecast.confidence_interval[1] - long_forecast.confidence_interval[0]
            
            assert long_width >= short_width  # Longer horizons should have more uncertainty
            
            self.passed_tests += 1
            return {
                'test': 'forecasting',
                'status': 'PASSED',
                'forecast_count': len(analysis.forecasts),
                'forecast_range': f"{min(f.forecasted_vol for f in analysis.forecasts):.3f} - {max(f.forecasted_vol for f in analysis.forecasts):.3f}",
                'uncertainty_increase': long_width > short_width
            }
            
        except Exception as e:
            return {
                'test': 'forecasting',
                'status': 'FAILED',
                'error': str(e)
            }
        finally:
            self.total_tests += 1
    
    async def test_signal_generation(self) -> Dict[str, Any]:
        """Test trading signal generation"""
        logger.info("Testing signal generation...")
        
        scenarios = ['low_vol', 'high_vol', 'vol_squeeze', 'vol_spike']
        results = {}
        
        for scenario in scenarios:
            try:
                test_data = self.generate_test_data(scenario, 100)
                
                analysis = await volatility_agent.analyze_volatility(
                    symbol=test_data['symbol'],
                    market_data=test_data,
                    include_iv=False,
                    forecast_horizons=[1]
                )
                
                # Validate signal
                assert analysis.primary_signal in VolatilitySignal
                assert 0 <= analysis.signal_strength <= 1
                assert isinstance(analysis.recommendations, list)
                
                results[scenario] = {
                    'primary_signal': analysis.primary_signal.value,
                    'signal_strength': analysis.signal_strength,
                    'recommendation_count': len(analysis.recommendations)
                }
                
            except Exception as e:
                results[scenario] = {'error': str(e)}
        
        # Validate that different scenarios produce different signals (mostly)
        signals = [r.get('primary_signal') for r in results.values() if 'primary_signal' in r]
        unique_signals = len(set(signals))
        
        if unique_signals >= 2 and all('error' not in r for r in results.values()):
            self.passed_tests += 1
            status = 'PASSED'
        else:
            status = 'FAILED'
            
        self.total_tests += 1
        
        return {
            'test': 'signal_generation', 
            'status': status,
            'unique_signals': unique_signals,
            'results': results
        }
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test agent performance metrics"""
        logger.info("Testing agent performance...")
        
        try:
            # Generate multiple analyses to build performance history
            for i in range(5):
                test_data = self.generate_test_data("normal", 50)
                await volatility_agent.analyze_volatility(
                    symbol=f"PERF_TEST_{i}",
                    market_data=test_data,
                    include_iv=False,
                    forecast_horizons=[1]
                )
            
            # Get performance metrics
            performance = volatility_agent.get_performance_metrics()
            
            # Validate performance metrics
            assert performance['total_analyses'] >= 5
            assert performance['average_calc_time_seconds'] > 0
            assert isinstance(performance['regime_distribution'], dict)
            assert isinstance(performance['signal_distribution'], dict)
            
            self.passed_tests += 1
            return {
                'test': 'performance',
                'status': 'PASSED',
                'metrics': performance
            }
            
        except Exception as e:
            return {
                'test': 'performance',
                'status': 'FAILED',
                'error': str(e)
            }
        finally:
            self.total_tests += 1
    
    async def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling"""
        logger.info("Testing edge cases...")
        
        edge_cases = []
        
        # Test insufficient data
        try:
            small_data = self.generate_test_data("normal", 10)  # Too little data
            await volatility_agent.analyze_volatility("SMALL_TEST", small_data)
            edge_cases.append({'case': 'insufficient_data', 'status': 'FAILED', 'reason': 'Should have raised error'})
        except ValueError:
            edge_cases.append({'case': 'insufficient_data', 'status': 'PASSED'})
        except Exception as e:
            edge_cases.append({'case': 'insufficient_data', 'status': 'FAILED', 'error': str(e)})
        
        # Test extreme volatility values
        try:
            extreme_data = self.generate_test_data("normal", 100)
            extreme_data['close_prices'] = [100.0] * 100  # No volatility
            analysis = await volatility_agent.analyze_volatility("ZERO_VOL_TEST", extreme_data)
            edge_cases.append({'case': 'zero_volatility', 'status': 'PASSED', 'vol': analysis.metrics.annualized_vol})
        except Exception as e:
            edge_cases.append({'case': 'zero_volatility', 'status': 'FAILED', 'error': str(e)})
        
        # Test invalid forecast horizons
        try:
            normal_data = self.generate_test_data("normal", 100)
            await volatility_agent.analyze_volatility("INVALID_HORIZON_TEST", normal_data, 
                                                   forecast_horizons=[300])  # Too long
            edge_cases.append({'case': 'invalid_horizons', 'status': 'FAILED', 'reason': 'Should have been capped'})
        except Exception:
            edge_cases.append({'case': 'invalid_horizons', 'status': 'PASSED'})
        
        passed_edge_cases = sum(1 for case in edge_cases if case['status'] == 'PASSED')
        
        if passed_edge_cases >= 2:
            self.passed_tests += 1
            status = 'PASSED'
        else:
            status = 'FAILED'
            
        self.total_tests += 1
        
        return {
            'test': 'edge_cases',
            'status': status,
            'cases': edge_cases,
            'passed_cases': passed_edge_cases
        }

async def run_comprehensive_test():
    """Run comprehensive test suite for Volatility Agent"""
    logger.info("Starting Volatility Agent V5 Comprehensive Test Suite")
    print("=" * 80)
    print("VOLATILITY AGENT V5 - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tester = VolatilityAgentTester()
    all_results = []
    
    # Run all tests
    tests = [
        tester.test_basic_functionality,
        tester.test_regime_classification,
        tester.test_pattern_detection,
        tester.test_iv_analysis,
        tester.test_forecasting,
        tester.test_signal_generation,
        tester.test_performance,
        tester.test_edge_cases
    ]
    
    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__.replace('test_', '').replace('_', ' ').title()}...")
            result = await test_func()
            all_results.append(result)
            
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_emoji} {result['test'].replace('_', ' ').title()}: {result['status']}")
            
            if result['status'] == 'FAILED':
                print(f"   Error: {result.get('error', 'Test failed validation')}")
                
        except Exception as e:
            error_result = {
                'test': test_func.__name__.replace('test_', ''),
                'status': 'ERROR',
                'error': str(e)
            }
            all_results.append(error_result)
            print(f"❌ {test_func.__name__.replace('test_', '').replace('_', ' ').title()}: ERROR")
            print(f"   Error: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {tester.total_tests}")
    print(f"Passed: {tester.passed_tests}")
    print(f"Failed: {tester.total_tests - tester.passed_tests}")
    print(f"Success Rate: {(tester.passed_tests / max(1, tester.total_tests)):.1%}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    print("-" * 40)
    for result in all_results:
        print(f"\n{result['test'].replace('_', ' ').title()}: {result['status']}")
        
        # Print interesting details
        if 'accuracy' in result:
            print(f"  Accuracy: {result['accuracy']}")
        if 'success_rate' in result:
            print(f"  Success Rate: {result['success_rate']}")
        if 'forecast_count' in result:
            print(f"  Forecasts Generated: {result['forecast_count']}")
        if 'unique_signals' in result:
            print(f"  Unique Signals: {result['unique_signals']}")
        if result['status'] == 'FAILED' and 'error' in result:
            print(f"  Error: {result['error']}")
    
    print(f"\nVolatility Agent V5 Test Suite Complete!")
    
    # Return results for potential API usage
    return {
        'total_tests': tester.total_tests,
        'passed_tests': tester.passed_tests,
        'success_rate': (tester.passed_tests / max(1, tester.total_tests)),
        'detailed_results': all_results,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())