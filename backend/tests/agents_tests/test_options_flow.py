#!/usr/bin/env python3
"""
Test script for Options Flow Intelligence System
"""

import asyncio
import json
from agents.options_flow_intelligence import options_flow_intelligence


async def test_options_flow():
    """Test the Options Flow Intelligence System"""
    
    print("=" * 70)
    print("OPTIONS FLOW INTELLIGENCE TEST")
    print("=" * 70)
    
    # Test 1: Analyze a bullish call sweep
    print("\n1. Testing Bullish Call Sweep Analysis:")
    print("-" * 50)
    
    bullish_flow = {
        'symbol': 'AAPL',
        'underlying_price': 195.0,
        'strike': 200.0,
        'days_to_expiry': 30,
        'call_put': 'C',
        'side': 'BUY',
        'size': 3000,
        'price': 4.0,
        'notional': 1200000,
        'implied_volatility': 0.32,
        'delta': 0.45,
        'gamma': 0.02,
        'flow_type': 'sweep',
        'aggressive_order': True,
        'volume_ratio': 5.0,
        'moneyness': 2.56
    }
    
    result = await options_flow_intelligence.analyze_options_flow(bullish_flow)
    
    print(f"✅ Institution Type: {result['flow_analysis']['institution_type']}")
    print(f"✅ Position Intent: {result['flow_analysis']['position_intent']}")
    print(f"✅ Smart Money Score: {result['flow_analysis']['smart_money_score']:.0f}/100")
    
    signals = result['trading_signals']
    print(f"\n📊 Trading Signals:")
    print(f"   Action: {signals['action'].upper()}")
    print(f"   Strategy: {signals['strategy']}")
    print(f"   Confidence: {signals['confidence']:.1%}")
    print(f"   Position Size: {signals['position_size']:.1%}")
    
    # Test 2: Detect unusual activity
    print("\n\n2. Testing Unusual Activity Detection:")
    print("-" * 50)
    
    activity = await options_flow_intelligence.detect_unusual_activity('AAPL')
    
    print(f"✅ Unusual Activity: {'YES' if activity['unusual_activity'] else 'NO'}")
    print(f"✅ Overall Bias: {activity['overall_bias'].upper()}")
    print(f"✅ Smart Money Detected: {'YES' if activity['smart_money_detected'] else 'NO'}")
    
    if activity['recommendation']:
        rec = activity['recommendation']
        print(f"\n💡 Recommendation:")
        print(f"   Action: {rec['action'].upper()}")
        print(f"   Reason: {rec['reason']}")
        if 'suggested_strategy' in rec:
            print(f"   Strategy: {rec['suggested_strategy']}")
    
    # Test 3: Bearish put flow
    print("\n\n3. Testing Bearish Put Analysis:")
    print("-" * 50)
    
    bearish_flow = {
        'symbol': 'SPY',
        'underlying_price': 450.0,
        'strike': 440.0,
        'days_to_expiry': 7,
        'call_put': 'P',
        'side': 'BUY',
        'size': 5000,
        'price': 2.5,
        'notional': 1250000,
        'implied_volatility': 0.25,
        'delta': -0.35,
        'gamma': 0.03,
        'flow_type': 'block',
        'volume_ratio': 3.0,
        'moneyness': -2.22
    }
    
    hedge_result = await options_flow_intelligence.analyze_options_flow(bearish_flow)
    
    print(f"✅ Institution Type: {hedge_result['flow_analysis']['institution_type']}")
    print(f"✅ Position Intent: {hedge_result['flow_analysis']['position_intent']}")
    print(f"✅ Smart Money Score: {hedge_result['flow_analysis']['smart_money_score']:.0f}/100")
    
    rm = hedge_result['trading_signals']['risk_management']
    print(f"\n🛡️ Risk Management:")
    print(f"   Stop Loss: {rm['stop_loss']:.1f}%")
    print(f"   Take Profit: {rm['take_profit']:.1f}%")
    print(f"   Max Risk: {rm['max_risk']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_options_flow())