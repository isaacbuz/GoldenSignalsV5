#!/usr/bin/env python3
"""
WebSocket Test Client for Live Data Verification
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

async def test_websocket():
    """Test WebSocket connection and subscribe to symbols"""
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri}")
            print("-" * 50)
            
            # Subscribe to symbols
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
            
            for symbol in symbols:
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "symbol": symbol
                })
                await websocket.send(subscribe_msg)
                print(f"📈 Subscribing to {symbol}...")
            
            print("-" * 50)
            print("Listening for price updates...")
            print("-" * 50)
            
            # Listen for messages
            message_count = 0
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "price_update":
                    quote = data.get("data", {})
                    symbol = quote.get("symbol", "")
                    price = quote.get("price", 0)
                    change = quote.get("change", 0)
                    change_pct = quote.get("changePercent", 0)
                    
                    # Color coding for price changes
                    if change >= 0:
                        color = "\033[92m"  # Green
                        arrow = "↑"
                    else:
                        color = "\033[91m"  # Red
                        arrow = "↓"
                    
                    reset = "\033[0m"
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {symbol:6s} ${price:8.2f} {color}{arrow} {change:+7.2f} ({change_pct:+6.2f}%){reset}")
                    
                    message_count += 1
                    if message_count >= 20:
                        print("-" * 50)
                        print(f"✅ Successfully received {message_count} price updates!")
                        print("Live data is flowing correctly.")
                        break
                        
                elif data.get("type") == "subscribed":
                    symbol = data.get("symbol", "")
                    print(f"✅ Confirmed subscription to {symbol}")
                    
    except websockets.exceptions.ConnectionRefused:
        print("❌ Could not connect to WebSocket server at localhost:8000")
        print("Make sure the backend is running: python live_backend.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 50)
    print("WebSocket Live Data Test")
    print("=" * 50)
    asyncio.run(test_websocket())