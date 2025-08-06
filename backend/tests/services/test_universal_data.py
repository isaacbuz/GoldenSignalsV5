#!/usr/bin/env python3
"""
Test script for Universal Market Data Service
"""

import asyncio
import time
from services.universal_market_data import universal_market_data, AssetClass


async def test_universal_data():
    """Test the Universal Market Data Service"""
    
    print("=" * 70)
    print("UNIVERSAL MARKET DATA SERVICE TEST")
    print("=" * 70)
    
    # Test 1: Get single price with failover
    print("\n1. Testing Price Fetch with Failover:")
    print("-" * 50)
    
    try:
        price_data = await universal_market_data.get_price("AAPL", AssetClass.EQUITY)
        print(f"✅ AAPL Price: ${price_data['price']:.2f}")
        print(f"✅ Data Source: {price_data['source']}")
        print(f"✅ Change: {price_data.get('change', 0):.2f} ({price_data.get('change_percent', 0):.2f}%)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Cache performance
    print("\n\n2. Testing Cache Performance:")
    print("-" * 50)
    
    # First call - will fetch from source
    start = time.time()
    await universal_market_data.get_price("MSFT")
    first_call_time = time.time() - start
    print(f"✅ First call (fetch): {first_call_time*1000:.2f}ms")
    
    # Second call - should be cached
    start = time.time()
    cached_data = await universal_market_data.get_price("MSFT")
    cached_call_time = time.time() - start
    print(f"✅ Cached call: {cached_call_time*1000:.2f}ms")
    
    if cached_call_time < first_call_time:
        improvement = first_call_time / cached_call_time
        print(f"✅ Speed improvement: {improvement:.1f}x faster")
    
    # Test 3: Multiple symbols
    print("\n\n3. Testing Multiple Symbols:")
    print("-" * 50)
    
    symbols = ["AAPL", "GOOGL", "TSLA", "SPY"]
    for symbol in symbols:
        try:
            data = await universal_market_data.get_price(symbol)
            print(f"✅ {symbol}: ${data['price']:.2f} (Source: {data['source']})")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    # Test 4: Market status
    print("\n\n4. Testing Market Status:")
    print("-" * 50)
    
    status = await universal_market_data.get_market_status()
    print(f"✅ US Market: {status['us_equity']}")
    print(f"✅ Europe Market: {status['europe_equity']}")
    print(f"✅ Asia Market: {status['asia_equity']}")
    print(f"✅ Forex: {status['forex']}")
    print(f"✅ Crypto: {status['crypto']}")
    
    # Test 5: Historical data
    print("\n\n5. Testing Historical Data:")
    print("-" * 50)
    
    try:
        historical = await universal_market_data.get_historical(
            "AAPL",
            "2024-01-01",
            "2024-01-10",
            "1d"
        )
        print(f"✅ Data points: {historical['data_points']}")
        if historical['data']:
            latest = historical['data'][-1]
            print(f"✅ Latest close: ${latest['close']:.2f}")
            if latest.get('sma_20'):
                print(f"✅ SMA 20: ${latest['sma_20']:.2f}")
            if latest.get('rsi'):
                print(f"✅ RSI: {latest['rsi']:.2f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 6: Rate limiting
    print("\n\n6. Testing Rate Limiter:")
    print("-" * 50)
    
    # Check rate limit status
    from services.universal_market_data import DataSource
    
    for source in [DataSource.YAHOO, DataSource.MOCK]:
        can_request = await universal_market_data.rate_limiter.check_limit(source)
        limit = universal_market_data.rate_limiter.SOURCE_LIMITS.get(source, 0)
        print(f"✅ {source.value}: {'Available' if can_request else 'Limited'} (Limit: {limit}/min)")
    
    # Test 7: Order book
    print("\n\n7. Testing Order Book:")
    print("-" * 50)
    
    try:
        orderbook = await universal_market_data.get_order_book("AAPL", depth=5)
        print(f"✅ Best Bid: ${orderbook['bids'][0]['price']:.2f} x {orderbook['bids'][0]['size']}")
        print(f"✅ Best Ask: ${orderbook['asks'][0]['price']:.2f} x {orderbook['asks'][0]['size']}")
        print(f"✅ Spread: ${orderbook['spread']:.2f}")
        print(f"✅ Mid Price: ${orderbook['mid_price']:.2f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
    
    # Summary
    print("\n📊 SERVICE FEATURES:")
    print("  • Multi-source data aggregation with failover")
    print("  • Intelligent caching (Redis + memory fallback)")
    print("  • Per-source rate limiting")
    print("  • Real-time WebSocket streaming")
    print("  • Historical data with indicators")
    print("  • Global market status tracking")
    print("  • Order book simulation")


if __name__ == "__main__":
    asyncio.run(test_universal_data())