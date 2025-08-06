"""
Test WebSocket Orchestration
Demonstrates real-time signal generation with agent activities
"""

import asyncio
import json
import websockets
from datetime import datetime
import sys


async def test_websocket_orchestration():
    """Test the orchestrated WebSocket functionality"""
    uri = "ws://localhost:8000/ws/signals"
    
    print("🚀 Testing WebSocket Orchestration")
    print("=" * 60)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Track messages
            message_counts = {
                "price_update": 0,
                "agent_update": 0,
                "signal_update": 0,
                "alert": 0,
                "other": 0
            }
            
            # 1. Subscribe to AAPL
            print("\n📊 Subscribing to AAPL...")
            await websocket.send(json.dumps({
                "type": "subscribe",
                "symbol": "AAPL"
            }))
            
            # 2. Trigger analysis
            print("🔍 Triggering analysis for AAPL...")
            await websocket.send(json.dumps({
                "type": "analyze",
                "symbol": "AAPL"
            }))
            
            # 3. Listen for updates
            print("\n📡 Listening for real-time updates...")
            print("-" * 60)
            
            start_time = datetime.now()
            timeout = 30  # 30 seconds timeout
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=1.0
                    )
                    
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    # Count message types
                    if msg_type in message_counts:
                        message_counts[msg_type] += 1
                    else:
                        message_counts["other"] += 1
                    
                    # Display message based on type
                    if msg_type == "agent_update":
                        agent_data = data.get("data", {})
                        print(f"🤖 Agent Update: {agent_data.get('agent')} - "
                              f"Signal: {agent_data.get('signal')} "
                              f"(Confidence: {agent_data.get('confidence', 0):.2f})")
                    
                    elif msg_type == "signal_update":
                        signal_data = data.get("data", {})
                        consensus = signal_data.get("agents_consensus", {})
                        print(f"\n🎯 SIGNAL GENERATED!")
                        print(f"   Symbol: {signal_data.get('symbol')}")
                        print(f"   Action: {signal_data.get('action')}")
                        print(f"   Confidence: {signal_data.get('confidence', 0):.2f}")
                        print(f"   Price: ${signal_data.get('price', 0):.2f}")
                        print(f"   Agents: {consensus.get('completed_agents')}/{consensus.get('total_agents')}")
                        print(f"   Consensus: {consensus.get('consensus_strength')}")
                        
                        # Show agent details
                        agent_details = consensus.get("agent_details", {})
                        if agent_details:
                            print("   Agent Breakdown:")
                            for agent_id, details in agent_details.items():
                                print(f"     - {details['name']}: {details['signal']} "
                                      f"(conf: {details['confidence']:.2f})")
                    
                    elif msg_type == "price_update":
                        price_data = data.get("data", {})
                        print(f"💰 Price Update: {price_data.get('symbol')} @ "
                              f"${price_data.get('price', 0):.2f}")
                    
                    elif msg_type == "alert":
                        print(f"ℹ️  Alert: {data.get('message')}")
                    
                    elif msg_type == "decision_update":
                        decision_data = data.get("data", {})
                        decision = decision_data.get("decision", {})
                        print(f"\n📋 Decision: {decision.get('action')}")
                        if decision.get('reason'):
                            print(f"   Reason: {decision.get('reason')}")
                    
                    else:
                        print(f"📨 {msg_type}: {json.dumps(data, indent=2)}")
                
                except asyncio.TimeoutError:
                    # No message received, send heartbeat
                    await websocket.send(json.dumps({"type": "heartbeat"}))
                    continue
            
            # 4. Get status
            print("\n📊 Getting orchestrator status...")
            await websocket.send(json.dumps({"type": "get_status"}))
            
            # Wait for status response
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(message)
                if data.get("type") == "status_update":
                    status = data.get("data", {})
                    print(f"   Active Agents: {status.get('active_agents', 0)}")
                    print(f"   Active Analyses: {status.get('active_analyses', 0)}")
                    print(f"   WebSocket Connections: {status.get('websocket_connections', 0)}")
            except:
                pass
            
            # 5. Summary
            print("\n" + "=" * 60)
            print("📊 MESSAGE SUMMARY")
            print("-" * 60)
            total_messages = sum(message_counts.values())
            print(f"Total Messages Received: {total_messages}")
            for msg_type, count in message_counts.items():
                if count > 0:
                    print(f"  - {msg_type}: {count}")
            
            print("\n✅ WebSocket Orchestration Test Complete!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True


async def test_symbol_websocket():
    """Test the symbol-specific WebSocket endpoint"""
    symbol = "NVDA"
    uri = f"ws://localhost:8000/ws/symbols/{symbol}"
    
    print("\n\n🎯 Testing Symbol-Specific WebSocket")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Symbol WebSocket")
            print("🔄 Auto-subscribed and analysis triggered")
            
            # Listen for updates
            print("\n📡 Listening for updates...")
            print("-" * 60)
            
            message_count = 0
            start_time = datetime.now()
            timeout = 20  # 20 seconds for symbol-specific
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                try:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=1.0
                    )
                    
                    data = json.loads(message)
                    message_count += 1
                    
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "signal_update":
                        signal_data = data.get("data", {})
                        print(f"\n🎯 Signal for {symbol}: {signal_data.get('action')} "
                              f"(conf: {signal_data.get('confidence', 0):.2f})")
                    elif msg_type == "agent_update":
                        agent_data = data.get("data", {})
                        print(f"🤖 {agent_data.get('agent')}: {agent_data.get('signal')}")
                    else:
                        print(f"📨 {msg_type}")
                    
                except asyncio.TimeoutError:
                    continue
            
            print(f"\n✅ Received {message_count} messages for {symbol}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True


async def main():
    """Run all WebSocket tests"""
    print("🌟 GoldenSignalsAI WebSocket Orchestration Test")
    print("=" * 60)
    print("This test will:")
    print("1. Connect to the orchestrated WebSocket")
    print("2. Subscribe to a symbol (AAPL)")
    print("3. Trigger real-time analysis")
    print("4. Display agent activities and signals")
    print("5. Test symbol-specific WebSocket")
    print("=" * 60)
    
    # Run tests
    success1 = await test_websocket_orchestration()
    success2 = await test_symbol_websocket()
    
    if success1 and success2:
        print("\n✅ All WebSocket tests passed!")
    else:
        print("\n⚠️  Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    print("\n⚠️  Make sure the backend is running on localhost:8000")
    print("Run with: cd backend && python app.py\n")
    
    asyncio.run(main())