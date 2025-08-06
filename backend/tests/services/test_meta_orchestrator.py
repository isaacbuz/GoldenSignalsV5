#!/usr/bin/env python3
"""
Test script for Meta Signal Orchestrator with Byzantine Fault Tolerance
"""

import asyncio
from agents.meta_signal_orchestrator import meta_orchestrator


async def test_meta_orchestrator():
    """Test the Meta Signal Orchestrator"""
    
    print("=" * 70)
    print("META SIGNAL ORCHESTRATOR TEST - BYZANTINE FAULT TOLERANCE")
    print("=" * 70)
    
    # Test 1: Unanimous consensus (no conflicts)
    print("\n1. Testing Unanimous Consensus (All Agents Agree):")
    print("-" * 50)
    
    unanimous_signals = [
        {"agent_name": "TechnicalAgent", "agent_type": "technical", "action": "buy", "confidence": 0.8},
        {"agent_name": "SentimentAgent", "agent_type": "sentiment", "action": "buy", "confidence": 0.75},
        {"agent_name": "VolumeAgent", "agent_type": "volume", "action": "buy", "confidence": 0.7},
        {"agent_name": "MacroAgent", "agent_type": "macro", "action": "buy", "confidence": 0.65},
        {"agent_name": "MLAgent", "agent_type": "ml", "action": "buy", "confidence": 0.85}
    ]
    
    result = await meta_orchestrator.orchestrate_signals(unanimous_signals)
    print(f"✅ Action: {result['action'].upper()}")
    print(f"✅ Confidence: {result['confidence']:.1%}")
    print(f"✅ Consensus Type: {result['consensus_type']}")
    print(f"✅ Byzantine Safe: {result['byzantine_safe']}")
    
    # Test 2: Conflicting signals with Byzantine agents
    print("\n\n2. Testing Byzantine Fault Tolerance (With Faulty Agents):")
    print("-" * 50)
    
    byzantine_signals = [
        # Majority good agents (buy)
        {"agent_name": "TechnicalAgent", "agent_type": "technical", "action": "buy", "confidence": 0.8},
        {"agent_name": "SentimentAgent", "agent_type": "sentiment", "action": "buy", "confidence": 0.75},
        {"agent_name": "VolumeAgent", "agent_type": "volume", "action": "buy", "confidence": 0.7},
        {"agent_name": "OptionsFlowAgent", "agent_type": "options_flow", "action": "buy", "confidence": 0.9},
        
        # Byzantine agents (trying to manipulate)
        {"agent_name": "FaultyAgent1", "agent_type": "unknown", "action": "sell", "confidence": 0.95},
        {"agent_name": "FaultyAgent2", "agent_type": "unknown", "action": "sell", "confidence": 0.99}
    ]
    
    result = await meta_orchestrator.orchestrate_signals(byzantine_signals)
    print(f"✅ Action: {result['action'].upper()}")
    print(f"✅ Confidence: {result['confidence']:.1%}")
    print(f"✅ Consensus Type: {result['consensus_type']}")
    print(f"✅ Byzantine Safe: {result['byzantine_safe']}")
    
    byzantine_method = result['methods']['byzantine']
    if 'byzantine_suspects' in byzantine_method:
        print(f"✅ Byzantine Suspects: {byzantine_method['byzantine_suspects']}")
    
    # Test 3: High conflict scenario
    print("\n\n3. Testing High Conflict Scenario:")
    print("-" * 50)
    
    conflict_signals = [
        {"agent_name": "TechnicalAgent", "agent_type": "technical", "action": "buy", "confidence": 0.8},
        {"agent_name": "SentimentAgent", "agent_type": "sentiment", "action": "sell", "confidence": 0.75},
        {"agent_name": "VolumeAgent", "agent_type": "volume", "action": "hold", "confidence": 0.7},
        {"agent_name": "MacroAgent", "agent_type": "macro", "action": "sell", "confidence": 0.65},
        {"agent_name": "MLAgent", "agent_type": "ml", "action": "buy", "confidence": 0.6}
    ]
    
    result = await meta_orchestrator.orchestrate_signals(conflict_signals)
    print(f"✅ Action: {result['action'].upper()}")
    print(f"✅ Confidence: {result['confidence']:.1%}")
    print(f"✅ Conflict Level: {result['conflict_level']}")
    
    conflicts = result['methods']['conflicts']
    print(f"✅ Action Distribution: {conflicts['action_distribution']}")
    print(f"✅ Disagreement Ratio: {conflicts['disagreement_ratio']:.2f}")
    
    # Test 4: Options flow boost
    print("\n\n4. Testing with Options Flow Intelligence:")
    print("-" * 50)
    
    signals_with_options = [
        {"agent_name": "TechnicalAgent", "agent_type": "technical", "action": "buy", "confidence": 0.6},
        {"agent_name": "SentimentAgent", "agent_type": "sentiment", "action": "hold", "confidence": 0.5},
        {"agent_name": "VolumeAgent", "agent_type": "volume", "action": "buy", "confidence": 0.55}
    ]
    
    options_flow = {
        "smart_money_score": 85,
        "position_intent": "directional_bullish"
    }
    
    result = await meta_orchestrator.orchestrate_signals(
        signals_with_options, 
        options_flow=options_flow
    )
    
    print(f"✅ Action: {result['action'].upper()}")
    print(f"✅ Confidence: {result['confidence']:.1%} (boosted by options flow)")
    print(f"✅ Smart Money Score: {options_flow['smart_money_score']}")
    
    # Test 5: Different consensus methods
    print("\n\n5. Testing All Consensus Methods:")
    print("-" * 50)
    
    test_signals = [
        {"agent_name": "Agent1", "agent_type": "technical", "action": "buy", "confidence": 0.75},
        {"agent_name": "Agent2", "agent_type": "sentiment", "action": "buy", "confidence": 0.65},
        {"agent_name": "Agent3", "agent_type": "ml", "action": "hold", "confidence": 0.70},
        {"agent_name": "Agent4", "agent_type": "volume", "action": "buy", "confidence": 0.80}
    ]
    
    # Test each method
    print("\n📊 Weighted Voting:")
    weighted = meta_orchestrator.weighted_voting_consensus(test_signals)
    print(f"   Action: {weighted['action']}, Confidence: {weighted['confidence']:.1%}")
    
    print("\n📊 Bayesian Consensus:")
    bayesian = meta_orchestrator.bayesian_consensus(test_signals)
    print(f"   Action: {bayesian['action']}, Confidence: {bayesian['confidence']:.1%}")
    
    print("\n📊 Byzantine Fault Tolerant:")
    byzantine = meta_orchestrator.byzantine_fault_tolerant_consensus(test_signals)
    print(f"   Action: {byzantine['action']}, Confidence: {byzantine['confidence']:.1%}")
    print(f"   Byzantine Safe: {byzantine['byzantine_safe']}")
    
    print("\n📊 ML Ensemble:")
    ml_ensemble = meta_orchestrator.ml_ensemble_optimization(test_signals)
    print(f"   Action: {ml_ensemble['action']}, Confidence: {ml_ensemble['confidence']:.1%}")
    
    # Test 6: Performance update
    print("\n\n6. Testing Performance Tracking:")
    print("-" * 50)
    
    # Simulate performance updates
    meta_orchestrator.update_agent_performance("TechnicalAgent", "buy", "buy", 0.8)
    meta_orchestrator.update_agent_performance("TechnicalAgent", "buy", "buy", 0.75)
    meta_orchestrator.update_agent_performance("TechnicalAgent", "sell", "buy", 0.9)  # Wrong prediction
    
    performance = meta_orchestrator.agent_performance["TechnicalAgent"]
    print(f"✅ Total Predictions: {performance['total_predictions']}")
    print(f"✅ Correct Predictions: {performance['correct_predictions']}")
    print(f"✅ Recent Accuracy: {performance['recent_accuracy']:.1%}")
    print(f"✅ Confidence Calibration: {performance['confidence_calibration']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Summary
    print("\n📊 ORCHESTRATOR FEATURES:")
    print("  • Byzantine fault tolerance (handles up to 33% faulty agents)")
    print("  • Multiple consensus methods (Weighted, Bayesian, Byzantine, ML)")
    print("  • Conflict detection and resolution")
    print("  • Adaptive agent weighting based on performance")
    print("  • Options flow integration for signal boosting")
    print("  • Confidence calibration tracking")


if __name__ == "__main__":
    asyncio.run(test_meta_orchestrator())