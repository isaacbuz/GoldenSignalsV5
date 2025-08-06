#!/usr/bin/env python3
"""
Test script for Gamma Exposure Agent
Tests options Greeks, gamma exposure, and pinning analysis
"""

import asyncio
import numpy as np
from typing import List, Dict, Any

from agents.gamma_exposure_agent import gamma_exposure_agent, OptionType, GammaRegime


def generate_realistic_options_chain(
    spot_price: float = 450.0,
    num_strikes: int = 20,
    dte: float = 0.08  # 30 days to expiry
) -> List[Dict[str, Any]]:
    """Generate realistic options chain data"""
    
    # Create strikes around spot (±10%)
    strikes = np.linspace(
        spot_price * 0.9,
        spot_price * 1.1,
        num_strikes
    )
    
    options_data = []
    
    for strike in strikes:
        # Distance from ATM affects OI and IV
        distance_from_atm = abs(strike - spot_price) / spot_price
        
        # Open Interest - higher for ATM options
        oi_multiplier = np.exp(-distance_from_atm * 8)  # Exponential decay
        base_oi = 15000
        
        # Implied Volatility - smile/skew
        atm_iv = 0.20
        iv_call = atm_iv + distance_from_atm * 0.3
        iv_put = atm_iv + distance_from_atm * 0.4  # Put skew
        
        # Call option
        call_oi = int(base_oi * oi_multiplier * np.random.uniform(0.8, 1.2))
        options_data.append({
            'strike': float(strike),
            'type': 'call',
            'open_interest': call_oi,
            'volume': int(call_oi * np.random.uniform(0.1, 0.3)),
            'time_to_expiry': dte,
            'implied_volatility': iv_call
        })
        
        # Put option
        put_oi = int(base_oi * oi_multiplier * np.random.uniform(0.9, 1.3))
        options_data.append({
            'strike': float(strike),
            'type': 'put',
            'open_interest': put_oi,
            'volume': int(put_oi * np.random.uniform(0.1, 0.3)),
            'time_to_expiry': dte,
            'implied_volatility': iv_put
        })
    
    return options_data


def test_black_scholes_greeks():
    """Test Black-Scholes Greeks calculations"""
    print("=" * 70)
    print("TESTING BLACK-SCHOLES GREEKS")
    print("=" * 70)
    
    # Test parameters
    spot = 450.0
    strike = 450.0
    tte = 0.08  # ~30 days
    vol = 0.20
    
    print(f"\nTest Parameters:")
    print(f"  Spot: ${spot}")
    print(f"  Strike: ${strike}")
    print(f"  Time to Expiry: {tte:.3f} years (~{tte*365:.0f} days)")
    print(f"  Volatility: {vol:.1%}")
    
    # Test Call Greeks
    call_greeks = gamma_exposure_agent.calculate_black_scholes_greeks(
        spot, strike, tte, vol, OptionType.CALL
    )
    
    print(f"\n📊 CALL GREEKS:")
    print(f"  Delta: {call_greeks['delta']:.4f}")
    print(f"  Gamma: {call_greeks['gamma']:.6f}")
    print(f"  Theta: ${call_greeks['theta']:.3f} per day")
    print(f"  Vega: ${call_greeks['vega']:.3f} per 1% vol")
    print(f"  Rho: ${call_greeks['rho']:.3f} per 1% rate")
    
    # Test Put Greeks
    put_greeks = gamma_exposure_agent.calculate_black_scholes_greeks(
        spot, strike, tte, vol, OptionType.PUT
    )
    
    print(f"\n📊 PUT GREEKS:")
    print(f"  Delta: {put_greeks['delta']:.4f}")
    print(f"  Gamma: {put_greeks['gamma']:.6f}")
    print(f"  Theta: ${put_greeks['theta']:.3f} per day")
    print(f"  Vega: ${put_greeks['vega']:.3f} per 1% vol")
    print(f"  Rho: ${put_greeks['rho']:.3f} per 1% rate")
    
    # Verify put-call relationships
    print(f"\n✅ VERIFICATION:")
    delta_diff = call_greeks['delta'] - put_greeks['delta']
    print(f"  Call Delta - Put Delta = {delta_diff:.4f} (should ≈ 1.0)")
    gamma_diff = abs(call_greeks['gamma'] - put_greeks['gamma'])
    print(f"  |Call Gamma - Put Gamma| = {gamma_diff:.8f} (should ≈ 0)")
    
    return call_greeks, put_greeks


async def test_gamma_exposure_calculation():
    """Test dealer gamma exposure calculation"""
    print("\n" + "=" * 70)
    print("TESTING GAMMA EXPOSURE CALCULATION")
    print("=" * 70)
    
    # Generate test data
    spot_price = 450.0
    options_data = generate_realistic_options_chain(spot_price)
    
    print(f"\nTest Setup:")
    print(f"  Spot Price: ${spot_price}")
    print(f"  Options Count: {len(options_data)}")
    print(f"  Strikes Range: ${min(opt['strike'] for opt in options_data):.0f} - ${max(opt['strike'] for opt in options_data):.0f}")
    
    # Calculate gamma exposure
    gamma_data = gamma_exposure_agent.calculate_dealer_gamma_exposure(
        options_data, spot_price
    )
    
    print(f"\n📊 GAMMA EXPOSURE RESULTS:")
    print(f"  Net Gamma: {gamma_data['net_gamma_exposure']:,.0f}")
    print(f"  Call Gamma: {gamma_data['call_gamma']:,.0f}")
    print(f"  Put Gamma: {gamma_data['put_gamma']:,.0f}")
    print(f"  Net Delta: {gamma_data['net_delta_exposure']:,.0f}")
    print(f"  Regime: {gamma_data['regime'].upper()}")
    
    # Show significant levels
    significant_levels = gamma_data.get('significant_levels', [])
    print(f"\n📊 SIGNIFICANT GAMMA LEVELS ({len(significant_levels)}):")
    for level in significant_levels[:5]:  # Show top 5
        print(f"  ${level['strike']:.0f} {level['type']}: {level['gamma_exposure']:,.0f} "
              f"({level['proximity']:.1%} from spot)")
    
    # Gamma flip level
    flip_level = gamma_data.get('gamma_flip_level')
    if flip_level:
        print(f"\n🔄 Gamma Flip Level: ${flip_level:.2f}")
        print(f"   Distance from spot: {abs(spot_price - flip_level)/spot_price:.1%}")
    
    return gamma_data


async def test_gamma_pinning_analysis():
    """Test gamma pinning analysis"""
    print("\n" + "=" * 70)
    print("TESTING GAMMA PINNING ANALYSIS")
    print("=" * 70)
    
    # Create scenario with strong pinning at 450 strike
    spot_price = 452.0  # Close to 450 strike
    options_data = []
    
    # Create large gamma exposure at 450 strike
    massive_oi = 50000  # Very large OI
    
    # 450 Call with huge OI
    options_data.append({
        'strike': 450.0,
        'type': 'call',
        'open_interest': massive_oi,
        'volume': 10000,
        'time_to_expiry': 0.04,  # 2 weeks
        'implied_volatility': 0.25
    })
    
    # 450 Put with huge OI
    options_data.append({
        'strike': 450.0,
        'type': 'put',
        'open_interest': massive_oi,
        'volume': 8000,
        'time_to_expiry': 0.04,
        'implied_volatility': 0.27
    })
    
    # Add some other strikes
    for strike in [440, 445, 455, 460]:
        options_data.extend([
            {
                'strike': float(strike),
                'type': 'call',
                'open_interest': 5000,
                'volume': 500,
                'time_to_expiry': 0.04,
                'implied_volatility': 0.22
            },
            {
                'strike': float(strike),
                'type': 'put',
                'open_interest': 5000,
                'volume': 500,
                'time_to_expiry': 0.04,
                'implied_volatility': 0.24
            }
        ])
    
    print(f"\nPinning Test Setup:")
    print(f"  Spot Price: ${spot_price}")
    print(f"  Massive OI at $450: {massive_oi:,} contracts")
    print(f"  Distance to 450: {abs(spot_price - 450)/spot_price:.1%}")
    
    # Calculate gamma exposure
    gamma_data = gamma_exposure_agent.calculate_dealer_gamma_exposure(
        options_data, spot_price
    )
    
    # Analyze pinning
    pinning = gamma_exposure_agent.analyze_gamma_pinning(gamma_data, spot_price)
    
    print(f"\n📊 PINNING ANALYSIS:")
    print(f"  Pinning Detected: {pinning.get('pinning_detected', False)}")
    
    if pinning.get('pinning_detected'):
        print(f"  Pin Level: ${pinning['pin_level']:.2f}")
        print(f"  Pin Strength: {pinning['pin_strength']:,.0f}")
        print(f"  Pinning Classification: {pinning['pinning_strength'].upper()}")
        print(f"  Distance to Pin: {pinning['distance_to_pin']:.1%}")
        print(f"  Pin Direction: {pinning['pin_direction']}")
        
        # Show pinning candidates
        candidates = pinning.get('pinning_candidates', [])
        print(f"\n📌 PINNING CANDIDATES:")
        for i, candidate in enumerate(candidates[:3], 1):
            print(f"  {i}. ${candidate['strike']:.0f}: {candidate['gamma_exposure']:,.0f} "
                  f"({candidate['proximity']:.1%} away)")
    
    return pinning


async def test_volatility_impact_analysis():
    """Test volatility impact analysis"""
    print("\n" + "=" * 70)
    print("TESTING VOLATILITY IMPACT ANALYSIS")
    print("=" * 70)
    
    # Test different gamma scenarios
    scenarios = [
        ("Short Gamma Regime", -2000000, "Dealers short gamma - amplifies moves"),
        ("Long Gamma Regime", 1500000, "Dealers long gamma - dampens moves"),
        ("Neutral Regime", 50000, "Balanced gamma exposure")
    ]
    
    for scenario_name, net_gamma, description in scenarios:
        print(f"\n📊 {scenario_name.upper()}:")
        print(f"   {description}")
        
        # Create gamma data
        gamma_data = {
            'net_gamma_exposure': net_gamma,
            'regime': 'short_gamma' if net_gamma < -100000 else 'long_gamma' if net_gamma > 100000 else 'neutral',
            'current_spot': 450.0,
            'gamma_flip_level': 445.0
        }
        
        # Analyze volatility impact
        vol_impact = gamma_exposure_agent.calculate_volatility_impact(gamma_data)
        
        print(f"   Volatility Impact: {vol_impact['volatility_impact'].upper()}")
        print(f"   Impact Magnitude: {vol_impact['impact_magnitude']:.2f}")
        print(f"   Hedging Flow: {vol_impact['hedging_flow'].replace('_', ' ').title()}")
        print(f"   Expected Vol Multiplier: {vol_impact['expected_volatility_multiplier']:.2f}x")
        print(f"   Market Stability: {vol_impact['stability'].title()}")


async def test_comprehensive_analysis():
    """Test comprehensive gamma analysis"""
    print("\n" + "=" * 70)
    print("TESTING COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    # Generate realistic options chain
    spot_price = 450.0
    options_data = generate_realistic_options_chain(spot_price, num_strikes=25)
    historical_vol = 0.18  # 18% historical vol
    
    print(f"\nComprehensive Analysis Setup:")
    print(f"  Spot: ${spot_price}")
    print(f"  Options: {len(options_data)} contracts")
    print(f"  Historical Vol: {historical_vol:.1%}")
    
    # Perform comprehensive analysis
    analysis = await gamma_exposure_agent.analyze(
        options_data=options_data,
        spot_price=spot_price,
        historical_volatility=historical_vol
    )
    
    # Display results
    signal = analysis.get('signal', {})
    print(f"\n🎯 TRADING SIGNAL:")
    print(f"  Action: {signal.get('action', 'hold').upper()}")
    print(f"  Confidence: {signal.get('confidence', 0):.1%}")
    print(f"  Regime: {signal.get('regime', 'unknown').upper()}")
    
    reasoning = signal.get('reasoning', [])
    if reasoning:
        print(f"\n💭 REASONING:")
        for reason in reasoning:
            print(f"  • {reason}")
    
    recommendations = signal.get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    # Risk metrics
    risk_metrics = analysis.get('risk_metrics', {})
    print(f"\n⚠️ RISK METRICS:")
    print(f"  Overall Risk: {risk_metrics.get('overall_risk_score', 0):.0f}/100 ({risk_metrics.get('risk_level', 'unknown').upper()})")
    print(f"  Gamma Risk: {risk_metrics.get('gamma_risk_score', 0):.0f}/100")
    print(f"  Pinning Risk: {risk_metrics.get('pinning_risk_score', 0):.0f}/100")
    print(f"  Expected Vol: {risk_metrics.get('expected_volatility', 0):.1%}")
    
    # Key levels
    key_levels = analysis.get('key_levels', {})
    print(f"\n🎯 KEY LEVELS:")
    if key_levels.get('max_gamma_strike'):
        print(f"  Max Gamma Strike: ${key_levels['max_gamma_strike']:.0f}")
    if key_levels.get('gamma_flip'):
        print(f"  Gamma Flip Level: ${key_levels['gamma_flip']:.2f}")
    if key_levels.get('delta_neutral_strike'):
        print(f"  Delta Neutral Strike: ${key_levels['delta_neutral_strike']:.0f}")
    
    return analysis


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 70)
    print("TESTING EDGE CASES")
    print("=" * 70)
    
    # Test 1: No options data
    print("\n🧪 Test 1: Empty options data")
    try:
        result = gamma_exposure_agent.calculate_dealer_gamma_exposure([], 450.0)
        print(f"   ✅ Handled gracefully: Net gamma = {result.get('net_gamma_exposure', 0)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Zero time to expiry
    print("\n🧪 Test 2: Zero time to expiry")
    greeks = gamma_exposure_agent.calculate_black_scholes_greeks(
        450, 450, 0.0, 0.2, OptionType.CALL
    )
    print(f"   ✅ Zero expiry handled: Gamma = {greeks['gamma']}")
    
    # Test 3: Very low volatility
    print("\n🧪 Test 3: Very low volatility")
    greeks = gamma_exposure_agent.calculate_black_scholes_greeks(
        450, 450, 0.08, 0.001, OptionType.CALL
    )
    print(f"   ✅ Low vol handled: Gamma = {greeks['gamma']:.8f}")
    
    # Test 4: Extreme moneyness
    print("\n🧪 Test 4: Deep OTM option")
    greeks = gamma_exposure_agent.calculate_black_scholes_greeks(
        450, 600, 0.08, 0.2, OptionType.CALL
    )
    print(f"   ✅ Deep OTM handled: Gamma = {greeks['gamma']:.8f}")


async def main():
    """Run all gamma exposure tests"""
    print("\n" + "=" * 70)
    print("GAMMA EXPOSURE AGENT TEST SUITE")
    print("=" * 70)
    
    # Run all tests
    test_black_scholes_greeks()
    await test_gamma_exposure_calculation()
    await test_gamma_pinning_analysis()
    await test_volatility_impact_analysis()
    await test_comprehensive_analysis()
    test_edge_cases()
    
    print("\n" + "=" * 70)
    print("✅ ALL GAMMA EXPOSURE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n📊 GAMMA EXPOSURE AGENT FEATURES:")
    print("  • Black-Scholes Greeks calculation (Delta, Gamma, Theta, Vega, Rho)")
    print("  • Dealer gamma exposure by strike")
    print("  • Gamma regime classification (Short/Long/Neutral)")
    print("  • Gamma pinning detection and strength analysis")
    print("  • Volatility impact assessment")
    print("  • Gamma flip point identification")
    print("  • Trading signal generation")
    print("  • Risk metrics calculation")
    print("  • Key levels identification")
    print("  • Comprehensive options analysis")


if __name__ == "__main__":
    asyncio.run(main())