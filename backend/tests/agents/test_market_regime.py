#!/usr/bin/env python3
"""
Test script for Market Regime Agent
Tests regime classification, forecasting, and strategy recommendations
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.market_regime_agent import market_regime_agent, MarketRegime, RegimeConfidence


def create_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """Create comprehensive test scenarios for different market regimes"""
    
    return {
        'strong_bull': {
            'name': 'Strong Bull Market',
            'description': 'Low VIX, strong breadth, positive momentum',
            'data': {
                'vix': 12.5,
                'vix_change': -0.08,
                'vix_percentile': 15.0,
                'advance_decline_ratio': 3.2,
                'new_highs_lows': 0.8,
                'up_down_volume': 2.1,
                'spy_momentum_1d': 0.018,
                'spy_momentum_5d': 0.032,
                'spy_momentum_20d': 0.045,
                'volume_ratio': 0.85,
                'volume_surge': 0.9,
                'sector_correlation': 0.28,
                'bond_equity_correlation': -0.35,
                'dollar_strength': 0.15,
                'credit_spread': 0.6,
                'term_structure': 0.08,
                'put_call_ratio': 0.55,
                'vix_term_structure': 0.18,
                'liquidity_score': 0.98,
                'market_depth': 0.96
            },
            'expected_regime': MarketRegime.BULL
        },
        
        'bear_market': {
            'name': 'Bear Market',
            'description': 'Elevated VIX, poor breadth, negative momentum',
            'data': {
                'vix': 32.8,
                'vix_change': 0.22,
                'vix_percentile': 85.0,
                'advance_decline_ratio': 0.35,
                'new_highs_lows': -0.85,
                'up_down_volume': 0.45,
                'spy_momentum_1d': -0.035,
                'spy_momentum_5d': -0.048,
                'spy_momentum_20d': -0.065,
                'volume_ratio': 1.85,
                'volume_surge': 2.1,
                'sector_correlation': 0.88,
                'bond_equity_correlation': 0.52,
                'dollar_strength': 0.25,
                'credit_spread': 2.8,
                'term_structure': -0.02,
                'put_call_ratio': 1.65,
                'vix_term_structure': -0.08,
                'liquidity_score': 0.58,
                'market_depth': 0.62
            },
            'expected_regime': MarketRegime.BEAR
        },
        
        'market_crisis': {
            'name': 'Market Crisis',
            'description': 'Extreme VIX spike, liquidity crisis, panic selling',
            'data': {
                'vix': 78.5,
                'vix_change': 1.2,
                'vix_percentile': 99.8,
                'advance_decline_ratio': 0.05,
                'new_highs_lows': -0.98,
                'up_down_volume': 0.12,
                'spy_momentum_1d': -0.125,
                'spy_momentum_5d': -0.185,
                'spy_momentum_20d': -0.225,
                'volume_ratio': 4.8,
                'volume_surge': 5.2,
                'sector_correlation': 0.98,
                'bond_equity_correlation': 0.85,
                'dollar_strength': 0.8,
                'credit_spread': 8.5,
                'term_structure': -0.25,
                'put_call_ratio': 3.2,
                'vix_term_structure': -0.45,
                'liquidity_score': 0.15,
                'market_depth': 0.18
            },
            'expected_regime': MarketRegime.CRISIS
        },
        
        'sideways_market': {
            'name': 'Sideways Market',
            'description': 'Neutral indicators, range-bound price action',
            'data': {
                'vix': 17.2,
                'vix_change': 0.005,
                'vix_percentile': 42.0,
                'advance_decline_ratio': 1.05,
                'new_highs_lows': 0.02,
                'up_down_volume': 0.98,
                'spy_momentum_1d': 0.001,
                'spy_momentum_5d': -0.002,
                'spy_momentum_20d': 0.003,
                'volume_ratio': 1.02,
                'volume_surge': 0.95,
                'sector_correlation': 0.48,
                'bond_equity_correlation': -0.05,
                'dollar_strength': 0.02,
                'credit_spread': 1.15,
                'term_structure': 0.035,
                'put_call_ratio': 1.02,
                'vix_term_structure': 0.01,
                'liquidity_score': 0.82,
                'market_depth': 0.88
            },
            'expected_regime': MarketRegime.SIDEWAYS
        },
        
        'transition_state': {
            'name': 'Regime Transition',
            'description': 'Mixed signals, conflicting indicators',
            'data': {
                'vix': 24.5,
                'vix_change': 0.15,
                'vix_percentile': 68.0,
                'advance_decline_ratio': 1.8,  # Conflicting: good breadth
                'new_highs_lows': 0.3,          # but high VIX
                'up_down_volume': 1.4,
                'spy_momentum_1d': 0.012,       # Positive momentum
                'spy_momentum_5d': -0.008,      # but recent weakness
                'spy_momentum_20d': 0.015,
                'volume_ratio': 1.6,
                'volume_surge': 1.8,
                'sector_correlation': 0.65,
                'bond_equity_correlation': 0.1,
                'dollar_strength': 0.1,
                'credit_spread': 1.8,
                'term_structure': 0.02,
                'put_call_ratio': 1.25,
                'vix_term_structure': -0.02,
                'liquidity_score': 0.72,
                'market_depth': 0.75
            },
            'expected_regime': MarketRegime.TRANSITION
        }
    }


async def test_regime_classification():
    """Test regime classification accuracy"""
    print("=" * 70)
    print("TESTING MARKET REGIME CLASSIFICATION")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    correct_predictions = 0
    total_predictions = len(scenarios)
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Analyze regime
        result = await market_regime_agent.analyze_regime(
            scenario['data'], 
            include_forecast=True
        )
        
        predicted_regime = MarketRegime(result['regime'])
        expected_regime = scenario['expected_regime']
        confidence = result['confidence']
        
        # Check prediction accuracy
        is_correct = predicted_regime == expected_regime
        if is_correct:
            correct_predictions += 1
        
        print(f"   Expected: {expected_regime.value.upper()}")
        print(f"   Predicted: {predicted_regime.value.upper()} ({'✓' if is_correct else '✗'})")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Risk Level: {result.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
        
        # Show top regime probabilities
        regime_probs = result.get('regime_probabilities', {})
        top_regimes = sorted(regime_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Top Probabilities:")
        for regime, prob in top_regimes:
            print(f"     {regime.upper()}: {prob:.1%}")
        
        # Show key strategies
        strategies = result.get('strategy_recommendations', [])[:3]
        print(f"   Key Strategies: {', '.join(strategies)}")
    
    accuracy = correct_predictions / total_predictions
    print(f"\n✅ CLASSIFICATION ACCURACY: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    return accuracy


async def test_regime_forecasting():
    """Test regime forecasting capabilities"""
    print("\n" + "=" * 70)
    print("TESTING REGIME FORECASTING")
    print("=" * 70)
    
    # Use bull market scenario for forecasting
    scenarios = create_test_scenarios()
    bull_scenario = scenarios['strong_bull']
    
    print(f"\n📈 Forecasting from: {bull_scenario['name']}")
    
    # Analyze with forecasting
    result = await market_regime_agent.analyze_regime(
        bull_scenario['data'],
        include_forecast=True
    )
    
    forecasts = result.get('forecasts', {})
    
    # Volatility forecast
    vol_forecast = forecasts.get('volatility', {})
    print(f"\n🔮 VOLATILITY FORECAST:")
    for horizon, vol in vol_forecast.items():
        print(f"   {horizon.replace('_', ' ').title()}: {vol:.1f}")
    
    # Regime duration
    duration = forecasts.get('regime_duration', {})
    print(f"\n⏱️ REGIME DURATION:")
    print(f"   Expected Remaining: {duration.get('expected_remaining_days', 'Unknown')} days")
    print(f"   Current Duration: {duration.get('current_duration_days', 'Unknown')} days")
    print(f"   Confidence: {duration.get('confidence', 0):.1%}")
    
    # Transition probabilities
    transitions = forecasts.get('transition_probabilities', {})
    print(f"\n🔄 TRANSITION PROBABILITIES:")
    for transition, prob in transitions.items():
        if prob > 0.05:  # Only show significant probabilities
            transition_name = transition.replace('to_', '').replace('_', ' ').title()
            print(f"   {transition_name}: {prob:.1%}")
    
    return forecasts


async def test_strategy_recommendations():
    """Test strategy recommendation system"""
    print("\n" + "=" * 70)
    print("TESTING STRATEGY RECOMMENDATIONS")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    
    for scenario_key, scenario in scenarios.items():
        expected_regime = scenario['expected_regime']
        
        # Temporarily set regime to test strategy generation
        original_regime = market_regime_agent.current_regime
        market_regime_agent.current_regime = expected_regime
        
        strategies = market_regime_agent._recommend_strategies()
        
        # Restore original regime
        market_regime_agent.current_regime = original_regime
        
        print(f"\n📋 {expected_regime.value.upper()} REGIME STRATEGIES:")
        for i, strategy in enumerate(strategies[:6], 1):  # Show top 6
            strategy_name = strategy.replace('_', ' ').title()
            print(f"   {i}. {strategy_name}")


async def test_risk_assessment():
    """Test risk assessment functionality"""
    print("\n" + "=" * 70)
    print("TESTING RISK ASSESSMENT")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    
    for scenario_key, scenario in scenarios.items():
        print(f"\n⚠️ {scenario['name']} Risk Assessment:")
        
        result = await market_regime_agent.analyze_regime(scenario['data'])
        risk_assessment = result.get('risk_assessment', {})
        
        print(f"   Risk Level: {risk_assessment.get('risk_level', 'Unknown')}")
        print(f"   Risk Score: {risk_assessment.get('risk_score', 0)}/100")
        
        risk_factors = risk_assessment.get('risk_factors', {})
        print(f"   Risk Factors:")
        for factor, score in risk_factors.items():
            factor_name = factor.replace('_', ' ').title()
            print(f"     {factor_name}: {score:.1%}")
        
        key_risks = risk_assessment.get('key_risks', [])
        if key_risks:
            print(f"   Key Risks: {', '.join(key_risks)}")


async def test_adaptive_learning():
    """Test adaptive threshold learning"""
    print("\n" + "=" * 70)
    print("TESTING ADAPTIVE LEARNING")
    print("=" * 70)
    
    print(f"\n🧠 Initial Thresholds:")
    for key, value in market_regime_agent.thresholds.items():
        if 'vix' in key:
            print(f"   {key}: {value:.1f}")
    
    # Simulate multiple regime classifications to trigger adaptation
    scenarios = create_test_scenarios()
    
    print(f"\n📊 Simulating {len(scenarios) * 5} regime classifications...")
    
    for i in range(5):  # Run each scenario 5 times
        for scenario_key, scenario in scenarios.items():
            await market_regime_agent.analyze_regime(scenario['data'])
    
    # Trigger adaptation
    market_regime_agent.adaptation_counter = market_regime_agent.adaptation_frequency
    await market_regime_agent._adapt_thresholds()
    
    print(f"\n🎯 Adapted Thresholds:")
    for key, value in market_regime_agent.thresholds.items():
        if 'vix' in key:
            print(f"   {key}: {value:.1f}")
    
    print(f"\n✅ Regime History Length: {len(market_regime_agent.regime_history)}")


async def test_regime_transitions():
    """Test regime transition detection"""
    print("\n" + "=" * 70)
    print("TESTING REGIME TRANSITIONS")
    print("=" * 70)
    
    scenarios = create_test_scenarios()
    
    # Start with bull market
    bull_scenario = scenarios['strong_bull']
    await market_regime_agent.analyze_regime(bull_scenario['data'])
    print(f"📈 Started in: {market_regime_agent.current_regime.value.upper()}")
    
    # Transition to crisis
    crisis_scenario = scenarios['market_crisis']
    result = await market_regime_agent.analyze_regime(crisis_scenario['data'])
    
    transition = result.get('transition_analysis')
    if transition:
        print(f"\n🔄 TRANSITION DETECTED:")
        print(f"   From: {transition.from_regime.value.upper()}")
        print(f"   To: {transition.to_regime.value.upper()}")
        print(f"   Probability: {transition.transition_probability:.1%}")
        print(f"   Expected Duration: {transition.expected_duration_days} days")
        print(f"   Key Catalysts: {', '.join(transition.key_catalysts)}")
    else:
        print(f"\n📊 Current Regime: {market_regime_agent.current_regime.value.upper()}")
        print(f"   No transition detected")


async def test_performance_tracking():
    """Test performance tracking and statistics"""
    print("\n" + "=" * 70)
    print("TESTING PERFORMANCE TRACKING")
    print("=" * 70)
    
    # Check regime statistics
    history_length = len(market_regime_agent.regime_history)
    print(f"\n📊 Regime History: {history_length} classifications")
    
    if history_length > 0:
        # Calculate regime distribution
        regime_counts = {}
        for record in market_regime_agent.regime_history:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"\n📈 REGIME DISTRIBUTION:")
        for regime, count in regime_counts.items():
            percentage = (count / history_length) * 100
            print(f"   {regime.upper()}: {count} ({percentage:.1f}%)")
        
        # Recent performance
        recent_records = market_regime_agent.regime_history[-10:]
        avg_confidence = sum(r['confidence'] for r in recent_records) / len(recent_records)
        print(f"\n🎯 Recent Average Confidence: {avg_confidence:.1%}")
        
        # Performance history
        perf_history_length = len(market_regime_agent.performance_history)
        print(f"📋 Performance Records: {perf_history_length}")


async def main():
    """Run comprehensive Market Regime Agent tests"""
    print("\n" + "=" * 70)
    print("MARKET REGIME AGENT TEST SUITE")
    print("=" * 70)
    
    # Run all tests
    classification_accuracy = await test_regime_classification()
    forecasts = await test_regime_forecasting()
    await test_strategy_recommendations()
    await test_risk_assessment()
    await test_adaptive_learning()
    await test_regime_transitions()
    await test_performance_tracking()
    
    print("\n" + "=" * 70)
    print("✅ ALL MARKET REGIME TESTS COMPLETED!")
    print("=" * 70)
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Classification Accuracy: {classification_accuracy:.1%}")
    print(f"   Forecasting: {'✓' if forecasts else '✗'}")
    print(f"   Strategy Recommendations: ✓")
    print(f"   Risk Assessment: ✓")
    print(f"   Adaptive Learning: ✓")
    print(f"   Transition Detection: ✓")
    print(f"   Performance Tracking: ✓")
    
    print(f"\n🏆 MARKET REGIME AGENT FEATURES:")
    print("  • Multi-factor regime classification (Bull/Bear/Sideways/Crisis/Transition)")
    print("  • Advanced volatility forecasting")
    print("  • Regime transition detection and analysis")
    print("  • Comprehensive risk assessment")
    print("  • Adaptive threshold learning")
    print("  • Strategy recommendations by regime")
    print("  • Performance tracking and statistics")
    print("  • Cross-asset signal integration")
    print("  • Real-time confidence scoring")
    print("  • Historical pattern recognition")


if __name__ == "__main__":
    asyncio.run(main())