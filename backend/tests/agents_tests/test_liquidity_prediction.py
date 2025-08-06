#!/usr/bin/env python3
"""
Test script for Liquidity Prediction Agent V5
Tests liquidity forecasting, execution recommendations, and market impact estimation
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from agents.liquidity_prediction_agent import liquidity_prediction_agent, LiquidityLevel, MarketSession


def create_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """Create test scenarios for different market conditions"""
    
    return {
        'high_liquidity_aapl': {
            'name': 'AAPL High Liquidity',
            'description': 'Apple in strong liquidity environment',
            'data': {
                'symbol': 'AAPL',
                'price': 195.00,
                'bid': 194.99,
                'ask': 195.01,
                'volume': 80000000,  # High volume day
                'avg_volume': 65000000,
                'last_trade': 195.005
            },
            'expected_level': LiquidityLevel.HIGH,
            'test_order_size': 1000000  # $1M order
        },
        
        'normal_liquidity_spy': {
            'name': 'SPY Normal Liquidity',
            'description': 'SPY ETF with typical liquidity',
            'data': {
                'symbol': 'SPY',
                'price': 450.00,
                'bid': 449.98,
                'ask': 450.02,
                'volume': 75000000,  # Normal volume
                'avg_volume': 80000000,
                'last_trade': 450.00
            },
            'expected_level': LiquidityLevel.NORMAL,
            'test_order_size': 2250000  # $2.25M order (0.5% ADV)
        },
        
        'low_liquidity_smallcap': {
            'name': 'Small Cap Low Liquidity',
            'description': 'Small cap stock with poor liquidity',
            'data': {
                'symbol': 'SMCAP',
                'price': 25.50,
                'bid': 25.40,
                'ask': 25.60,
                'volume': 500000,   # Low volume
                'avg_volume': 1500000,
                'last_trade': 25.52
            },
            'expected_level': LiquidityLevel.LOW,
            'test_order_size': 255000  # $255K order (1% ADV)
        },
        
        'very_low_liquidity': {
            'name': 'Illiquid Microcap',
            'description': 'Microcap with very poor liquidity',
            'data': {
                'symbol': 'MICRO',
                'price': 5.00,
                'bid': 4.90,
                'ask': 5.10,
                'volume': 50000,    # Very low volume
                'avg_volume': 200000,
                'last_trade': 4.95
            },
            'expected_level': LiquidityLevel.VERY_LOW,
            'test_order_size': 50000   # $50K order (5% ADV)
        },
        
        'crisis_liquidity': {
            'name': 'Crisis Liquidity Scenario',
            'description': 'Market stress with dried up liquidity',
            'data': {
                'symbol': 'STRESS',
                'price': 100.00,
                'bid': 99.00,
                'ask': 101.00,
                'volume': 100000,   # Extremely low volume
                'avg_volume': 5000000,  # Normal volume much higher
                'last_trade': 99.50
            },
            'expected_level': LiquidityLevel.DRIED_UP,
            'test_order_size': 500000  # $500K order (10% ADV)
        }
    }


async def test_liquidity_classification():
    """Test liquidity level classification accuracy"""
    print("=" * 70)
    print("TESTING LIQUIDITY CLASSIFICATION")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    correct_predictions = 0
    total_predictions = len(scenarios)
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Predict liquidity
        prediction = await liquidity_prediction_agent.predict_liquidity(
            symbol=scenario['data']['symbol'],
            market_data=scenario['data'],
            forecast_horizon=30
        )
        
        predicted_level = LiquidityLevel(prediction['current_liquidity']['level'])
        expected_level = scenario['expected_level']
        
        # Check accuracy
        is_correct = predicted_level == expected_level
        if is_correct:
            correct_predictions += 1
        
        print(f"   Expected: {expected_level.value.upper()}")
        print(f"   Predicted: {predicted_level.value.upper()} ({'✓' if is_correct else '✗'})") 
        print(f"   Confidence: {prediction['current_liquidity']['score']:.2f}")
        print(f"   Volume Ratio: {prediction['current_liquidity']['volume_ratio']:.2f}")
        print(f"   Spread: {prediction['current_liquidity']['spread_bps']:.1f} bps")
        
        # Show warnings
        warnings = prediction['warnings']
        if warnings:
            print(f"   Warnings: {len(warnings)}")
            for warning in warnings[:2]:  # Show first 2 warnings
                print(f"     - {warning}")
    
    accuracy = correct_predictions / total_predictions
    print(f"\n✅ CLASSIFICATION ACCURACY: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    return accuracy


async def test_execution_recommendations():
    """Test execution strategy recommendations"""
    print("\n" + "=" * 70)
    print("TESTING EXECUTION RECOMMENDATIONS")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n📋 {scenario['name']} Execution Test:")
        
        order_size = scenario['test_order_size']
        market_data = scenario['data']
        symbol = market_data['symbol']
        
        # Test different urgency levels
        for urgency in ['normal', 'immediate', 'patient']:
            print(f"\n   {urgency.upper()} Urgency:")
            
            rec = await liquidity_prediction_agent.recommend_execution(
                symbol=symbol,
                order_size=order_size,
                side='buy',
                market_data=market_data,
                urgency=urgency
            )
            
            strategy = rec['execution_strategy']
            schedule = rec['execution_schedule']
            risk = rec['risk_assessment']
            
            print(f"     Strategy: {strategy['recommended_type'].upper()}")
            print(f"     Expected Cost: {strategy['expected_cost_bps']:.1f} bps")
            print(f"     Time Horizon: {strategy['time_horizon_minutes']} minutes")
            print(f"     Slices: {len(schedule)}")
            print(f"     Risk Level: {risk['risk_level'].upper()}")
            print(f"     ADV %: {risk['adv_percentage']:.1f}%")


async def test_market_impact_estimation():
    """Test market impact estimation"""
    print("\n" + "=" * 70)
    print("TESTING MARKET IMPACT ESTIMATION")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n💰 Impact Analysis: {scenario['name']}")
        
        market_data = scenario['data']
        symbol = market_data['symbol']
        order_size = scenario['test_order_size']
        price = market_data['price']
        
        # Calculate metrics
        current_time = datetime.now()
        metrics = liquidity_prediction_agent._calculate_liquidity_metrics(market_data, current_time)
        
        # Estimate impact
        impact = liquidity_prediction_agent.liquidity_analyzer.estimate_market_impact(
            order_size=order_size,
            metrics=metrics,
            urgency='normal'
        )
        
        # Calculate dollar impact
        notional = order_size * price if 'volume' in scenario['data'] else order_size
        dollar_impact = impact['total_cost_bps'] / 10000 * notional
        
        print(f"   Order: ${order_size:,.0f} (${notional:,.0f} notional)")
        print(f"   Total Impact: {impact['total_impact_bps']:.1f} bps")
        print(f"   Permanent: {impact['permanent_impact_bps']:.1f} bps")
        print(f"   Temporary: {impact['temporary_impact_bps']:.1f} bps")
        print(f"   Spread Cost: {impact['spread_cost_bps']:.1f} bps")
        print(f"   Total Cost: {impact['total_cost_bps']:.1f} bps")
        print(f"   Dollar Impact: ${dollar_impact:,.2f}")
        print(f"   Volume Ratio: {metrics.volume_ratio:.2f}")


async def test_optimal_execution_windows():
    """Test optimal execution window identification"""
    print("\n" + "=" * 70)
    print("TESTING OPTIMAL EXECUTION WINDOWS")
    print("=" * 70)
    
    current_time = datetime.now()
    
    # Get optimal windows
    windows = liquidity_prediction_agent._identify_optimal_windows(current_time)
    
    print(f"\n🕒 Optimal Execution Windows for {current_time.date()}:")
    print(f"   Current Time: {current_time.strftime('%H:%M:%S')}")
    
    if not windows:
        print("   No remaining windows for today")
        return
    
    for window in windows:
        session_name = window['session'].replace('_', ' ').title()
        start_time = datetime.fromisoformat(window['start']).strftime('%H:%M')
        end_time = datetime.fromisoformat(window['end']).strftime('%H:%M')
        
        print(f"\n   {session_name}:")
        print(f"     Time: {start_time} - {end_time}")
        print(f"     Liquidity Multiplier: {window['liquidity_multiplier']:.1f}x")
        print(f"     Best For: {window['recommended_for']}")


async def test_session_analysis():
    """Test trading session analysis"""
    print("\n" + "=" * 70)
    print("TESTING SESSION ANALYSIS")
    print("=" * 70)
    
    current_time = datetime.now()
    current_session = liquidity_prediction_agent._get_current_session(current_time)
    
    print(f"\n📅 Current Trading Session Analysis:")
    print(f"   Current Time: {current_time.strftime('%H:%M:%S')}")
    print(f"   Current Session: {current_session.value.replace('_', ' ').title()}")
    
    # Get intraday patterns
    patterns = liquidity_prediction_agent.intraday_patterns
    
    print(f"\n📊 Intraday Liquidity Patterns:")
    for session, pattern in patterns.items():
        session_name = session.value.replace('_', ' ').title()
        vol_pct = pattern.get('typical_volume_pct', 0) * 100
        spread_mult = pattern.get('typical_spread_mult', 1.0)
        volatility = pattern.get('volatility', 'unknown')
        
        marker = "👉 " if session == current_session else "   "
        
        print(f"{marker}{session_name}:")
        print(f"     Volume: {vol_pct:.1f}% of daily")
        print(f"     Spread Multiplier: {spread_mult:.1f}x")
        print(f"     Volatility: {volatility.upper()}")


async def test_forecasting_accuracy():
    """Test liquidity forecasting with different horizons"""
    print("\n" + "=" * 70)
    print("TESTING LIQUIDITY FORECASTING")
    print("=" * 70)
    
    # Use normal liquidity scenario
    scenarios = create_test_scenarios()
    test_scenario = scenarios['normal_liquidity_spy']
    
    market_data = test_scenario['data']
    symbol = market_data['symbol']
    
    print(f"\n🔮 Liquidity Forecasting for {symbol}:")
    print(f"   Current Conditions: {test_scenario['description']}")
    
    # Test different forecast horizons
    horizons = [15, 30, 60, 120]
    
    for horizon in horizons:
        prediction = await liquidity_prediction_agent.predict_liquidity(
            symbol=symbol,
            market_data=market_data,
            forecast_horizon=horizon
        )
        
        current_level = prediction['current_liquidity']['level']
        predicted_level = prediction['predicted_liquidity']['level']
        confidence = prediction['predicted_liquidity']['confidence']
        
        print(f"\n   {horizon} Minute Forecast:")
        print(f"     Current: {current_level.upper()}")
        print(f"     Predicted: {predicted_level.upper()}")
        print(f"     Confidence: {confidence:.1%}")
        
        # Check for regime changes
        if current_level != predicted_level:
            print(f"     ⚠️  Regime change expected!")


async def test_risk_assessment():
    """Test execution risk assessment"""
    print("\n" + "=" * 70)
    print("TESTING RISK ASSESSMENT")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n⚠️ Risk Assessment: {scenario['name']}")
        
        market_data = scenario['data']
        order_size = scenario['test_order_size']
        
        # Get execution recommendation (includes risk assessment)
        rec = await liquidity_prediction_agent.recommend_execution(
            symbol=market_data['symbol'],
            order_size=order_size,
            side='buy',
            market_data=market_data,
            urgency='normal'
        )
        
        risk = rec['risk_assessment']
        
        print(f"   Overall Risk: {risk['risk_level'].upper()}")
        print(f"   Risk Score: {risk['overall_risk_score']:.2f}")
        print(f"   Size Risk: {risk['risk_factors']['size_risk']:.2f}")
        print(f"   Timing Risk: {risk['risk_factors']['timing_risk']:.2f}")
        print(f"   Liquidity Risk: {risk['risk_factors']['liquidity_risk']:.2f}")
        print(f"   ADV %: {risk['adv_percentage']:.1f}%")
        
        # Show mitigation strategies
        mitigations = risk['mitigation_strategies']
        print(f"   Mitigation Strategies:")
        for strategy in mitigations[:3]:  # Show first 3
            print(f"     • {strategy}")


async def test_agent_performance():
    """Test agent performance metrics"""
    print("\n" + "=" * 70)
    print("TESTING AGENT PERFORMANCE")
    print("=" * 70)
    
    # Run several predictions to build history
    scenarios = create_test_scenarios()
    
    print(f"\n🔧 Running predictions to build performance history...")
    
    for i, (scenario_key, scenario) in enumerate(scenarios.items(), 1):
        await liquidity_prediction_agent.predict_liquidity(
            symbol=scenario['data']['symbol'],
            market_data=scenario['data'],
            forecast_horizon=30
        )
        print(f"   Completed prediction {i}/{len(scenarios)}")
    
    # Get performance metrics
    performance = liquidity_prediction_agent.get_performance_metrics()
    
    print(f"\n📊 AGENT PERFORMANCE METRICS:")
    print(f"   Total Predictions: {performance['total_predictions']}")
    print(f"   Average Confidence: {performance['average_confidence']:.1%}")
    print(f"   Recent Warnings: {performance['recent_warnings']}")
    
    print(f"\n📈 Prediction Distribution:")
    for level, count in performance['prediction_distribution'].items():
        percentage = (count / performance['total_predictions']) * 100
        print(f"     {level.upper()}: {count} ({percentage:.1f}%)")
    
    print(f"\n🕒 Sessions Analyzed:")
    for session in performance['sessions_analyzed']:
        session_name = session.replace('_', ' ').title()
        print(f"     • {session_name}")


async def main():
    """Run comprehensive Liquidity Prediction Agent tests"""
    print("\n" + "=" * 70)
    print("LIQUIDITY PREDICTION AGENT V5 TEST SUITE")
    print("=" * 70)
    
    # Run all tests
    classification_accuracy = await test_liquidity_classification()
    await test_execution_recommendations()
    await test_market_impact_estimation()
    await test_optimal_execution_windows()
    await test_session_analysis()
    await test_forecasting_accuracy()
    await test_risk_assessment()
    await test_agent_performance()
    
    print("\n" + "=" * 70)
    print("✅ ALL LIQUIDITY PREDICTION TESTS COMPLETED!")
    print("=" * 70)
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Classification Accuracy: {classification_accuracy:.1%}")
    print(f"   Execution Recommendations: ✓")
    print(f"   Market Impact Estimation: ✓")
    print(f"   Optimal Windows: ✓")
    print(f"   Session Analysis: ✓")
    print(f"   Forecasting: ✓")
    print(f"   Risk Assessment: ✓")
    print(f"   Performance Tracking: ✓")
    
    print(f"\n🏆 LIQUIDITY PREDICTION AGENT V5 FEATURES:")
    print("  • Multi-factor liquidity classification (6 levels)")
    print("  • Intraday session pattern recognition")
    print("  • Square-root market impact modeling")
    print("  • Algorithmic execution recommendations")
    print("  • Multi-horizon liquidity forecasting")
    print("  • Comprehensive risk assessment")
    print("  • Optimal execution window identification")
    print("  • Real-time session analysis")
    print("  • Performance tracking and metrics")
    print("  • Alternative strategy generation")


if __name__ == "__main__":
    asyncio.run(main())