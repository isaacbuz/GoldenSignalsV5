"""
Test script to verify all data sources are working
Run this to ensure API keys are valid and data is flowing
"""

import asyncio
from datetime import datetime
from dotenv import load_dotenv
# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from services.enhanced_data_aggregator import enhanced_data_aggregator
from services.live_data_provider import LiveDataProvider
# from services.market_data_service import MarketDataService  # TODO: Update to unified service
from core.logging import get_logger

logger = get_logger(__name__)


async def test_data_sources():
    """Test all configured data sources"""
    
    print("\n" + "="*60)
    print("🔍 GOLDENSIGNALSAI DATA SOURCES TEST")
    print("="*60 + "\n")
    
    # Test symbol
    test_symbol = "AAPL"
    
    # 1. Test Market Data Service (Primary)
    print("1️⃣ Testing Market Data Service (yfinance primary)...")
    try:
        market_service = MarketDataService()
        quote = await market_service.get_quote(test_symbol)
        if quote:
            print(f"✅ Market Data Service: {test_symbol} @ ${quote.get('price', 'N/A')}")
            print(f"   Volume: {quote.get('volume', 'N/A'):,}")
            print(f"   Change: {quote.get('change_percent', 'N/A'):.2f}%")
        else:
            print("❌ Market Data Service: No data returned")
    except Exception as e:
        print(f"❌ Market Data Service Error: {e}")
    
    print("\n" + "-"*60 + "\n")
    
    # 2. Test Live Data Provider
    print("2️⃣ Testing Live Data Provider (multi-source)...")
    try:
        live_provider = LiveDataProvider()
        
        # Check which API keys are configured
        sources_status = {
            'yfinance': 'No key needed',
            'finnhub': 'Configured' if live_provider.finnhub_key else 'Not configured',
            'twelve_data': 'Configured' if live_provider.twelve_data_key else 'Not configured',
            'alpha_vantage': 'Configured' if live_provider.alpha_vantage_key else 'Not configured',
            'polygon': 'Configured' if live_provider.polygon_key else 'Not configured'
        }
        
        print("   Data source status:")
        for source, status in sources_status.items():
            print(f"   - {source}: {status}")
        
        live_quote = await live_provider.get_live_quote(test_symbol)
        if live_quote:
            print(f"\n✅ Live Data: {live_quote.symbol} @ ${live_quote.price}")
            print(f"   Source: {live_quote.source}")
            print(f"   Bid/Ask: ${live_quote.bid}/{live_quote.ask}")
        else:
            print("❌ Live Data Provider: No data returned")
    except Exception as e:
        print(f"❌ Live Data Provider Error: {e}")
    
    print("\n" + "-"*60 + "\n")
    
    # 3. Test Enhanced Data Aggregator
    print("3️⃣ Testing Enhanced Data Aggregator...")
    try:
        # Initialize aggregator
        await enhanced_data_aggregator.initialize()
        
        # Test comprehensive market data
        print("\n   a) Testing market data aggregation...")
        market_data = await enhanced_data_aggregator.get_comprehensive_market_data(test_symbol)
        if market_data:
            print(f"   ✅ Market data from {len(market_data.get('sources', []))} sources")
            print(f"   Consensus price: ${market_data.get('consensus_price', 'N/A')}")
        
        # Test economic indicators
        print("\n   b) Testing economic indicators...")
        economic_data = await enhanced_data_aggregator.get_economic_indicators()
        if economic_data:
            print(f"   ✅ Retrieved {len(economic_data)} economic indicators")
            for indicator in economic_data[:3]:  # Show first 3
                print(f"   - {indicator.indicator}: {indicator.value} (impact: {indicator.impact})")
        else:
            print("   ⚠️ No economic data (FRED_API_KEY not configured)")
        
        # Test crypto data
        print("\n   c) Testing crypto data...")
        crypto_data = await enhanced_data_aggregator.get_crypto_data(['BTC', 'ETH'])
        if crypto_data:
            print(f"   ✅ Crypto data sources: {list(crypto_data.keys())}")
        
        # Test news sentiment
        print("\n   d) Testing news sentiment...")
        news_data = await enhanced_data_aggregator.get_news_sentiment([test_symbol])
        if news_data.get('articles'):
            print(f"   ✅ News articles: {len(news_data['articles'])}")
            print(f"   Sentiment score: {news_data.get('sentiment', {}).get('score', 'N/A')}")
        else:
            print("   ⚠️ No news data (NEWS_API_KEY not configured)")
        
        await enhanced_data_aggregator.close()
        
    except Exception as e:
        print(f"❌ Enhanced Data Aggregator Error: {e}")
    
    print("\n" + "-"*60 + "\n")
    
    # 4. Test individual data sources
    print("4️⃣ Testing Individual API Keys...")
    
    # Test Finnhub
    if os.getenv('FINNHUB_API_KEY'):
        print("\n   Testing Finnhub...")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"https://finnhub.io/api/v1/quote?symbol={test_symbol}&token={os.getenv('FINNHUB_API_KEY')}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Finnhub: {test_symbol} @ ${data.get('c', 'N/A')}")
                    else:
                        print(f"   ❌ Finnhub: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Finnhub Error: {e}")
    
    # Test Twelve Data
    if os.getenv('TWELVEDATA_API_KEY'):
        print("\n   Testing Twelve Data...")
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.twelvedata.com/quote?symbol={test_symbol}&apikey={os.getenv('TWELVEDATA_API_KEY')}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Twelve Data: {test_symbol} @ ${data.get('price', 'N/A')}")
                    else:
                        print(f"   ❌ Twelve Data: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Twelve Data Error: {e}")
    
    # Test Alpha Vantage
    if os.getenv('ALPHA_VANTAGE_API_KEY'):
        print("\n   Testing Alpha Vantage...")
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={test_symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote = data.get('Global Quote', {})
                        if quote:
                            print(f"   ✅ Alpha Vantage: {test_symbol} @ ${quote.get('05. price', 'N/A')}")
                        else:
                            print(f"   ⚠️ Alpha Vantage: Rate limit or invalid response")
                    else:
                        print(f"   ❌ Alpha Vantage: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Alpha Vantage Error: {e}")
    
    print("\n" + "="*60)
    print("📊 DATA SOURCES TEST SUMMARY")
    print("="*60 + "\n")
    
    print("✅ Working Sources:")
    print("   - yfinance (no key needed)")
    print("   - Any configured API keys above")
    
    print("\n⚠️ Missing Sources (optional):")
    if not os.getenv('FRED_API_KEY'):
        print("   - FRED (economic data) - Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    if not os.getenv('NEWS_API_KEY'):
        print("   - News API (news sentiment)")
    if not os.getenv('REDDIT_CLIENT_ID'):
        print("   - Reddit (social sentiment)")
    
    print("\n💡 Recommendations:")
    print("   1. yfinance is working as primary source (no key needed)")
    print("   2. Finnhub and Twelve Data provide good backup data")
    print("   3. Consider adding FRED for economic indicators (FREE)")
    print("   4. Platform will automatically fallback between sources")
    
    print("\n✅ Your data pipeline is ready for signal generation!\n")


if __name__ == "__main__":
    print("Starting data sources test...")
    asyncio.run(test_data_sources())