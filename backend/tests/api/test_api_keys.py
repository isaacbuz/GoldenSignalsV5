"""
Test API Keys and Services
"""

import asyncio
from dotenv import load_dotenv
import aiohttp
import json

# Load environment variables
load_dotenv()

async def test_polygon_api():
    """Test Polygon.io API"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return {"polygon": "No API key"}
    
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"polygon": "✅ Working", "data": data.get("status")}
                else:
                    return {"polygon": f"❌ Error: {response.status}"}
    except Exception as e:
        return {"polygon": f"❌ Error: {str(e)}"}

async def test_alpha_vantage_api():
    """Test Alpha Vantage API"""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        return {"alpha_vantage": "No API key"}
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Global Quote" in data:
                        return {"alpha_vantage": "✅ Working"}
                    elif "Note" in data:
                        return {"alpha_vantage": "⚠️ Rate limited"}
                    else:
                        return {"alpha_vantage": "❌ Invalid response"}
                else:
                    return {"alpha_vantage": f"❌ Error: {response.status}"}
    except Exception as e:
        return {"alpha_vantage": f"❌ Error: {str(e)}"}

async def test_finnhub_api():
    """Test Finnhub API"""
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        return {"finnhub": "No API key"}
    
    url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'c' in data:  # Current price
                        return {"finnhub": "✅ Working", "price": data['c']}
                    else:
                        return {"finnhub": "❌ Invalid response"}
                else:
                    return {"finnhub": f"❌ Error: {response.status}"}
    except Exception as e:
        return {"finnhub": f"❌ Error: {str(e)}"}

async def test_twelvedata_api():
    """Test Twelve Data API"""
    api_key = os.getenv('TWELVEDATA_API_KEY')
    if not api_key:
        return {"twelvedata": "No API key"}
    
    url = f"https://api.twelvedata.com/quote?symbol=AAPL&apikey={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'symbol' in data:
                        return {"twelvedata": "✅ Working", "price": data.get('close')}
                    else:
                        return {"twelvedata": "❌ Invalid response"}
                else:
                    return {"twelvedata": f"❌ Error: {response.status}"}
    except Exception as e:
        return {"twelvedata": f"❌ Error: {str(e)}"}

async def test_fmp_api():
    """Test Financial Modeling Prep API"""
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        return {"fmp": "No API key"}
    
    url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return {"fmp": "✅ Working", "price": data[0].get('price')}
                    else:
                        return {"fmp": "❌ Invalid response"}
                else:
                    return {"fmp": f"❌ Error: {response.status}"}
    except Exception as e:
        return {"fmp": f"❌ Error: {str(e)}"}

async def test_openai_api():
    """Test OpenAI API (if key provided)"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == "":
        return {"openai": "❌ No API key configured"}
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say 'API working'"}],
        "max_tokens": 10
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    return {"openai": "✅ Working"}
                elif response.status == 401:
                    return {"openai": "❌ Invalid API key"}
                else:
                    return {"openai": f"❌ Error: {response.status}"}
    except Exception as e:
        return {"openai": f"❌ Error: {str(e)}"}

async def test_anthropic_api():
    """Test Anthropic API (if key provided)"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == "":
        return {"anthropic": "❌ No API key configured"}
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Say 'API working'"}],
        "max_tokens": 10
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    return {"anthropic": "✅ Working"}
                elif response.status == 401:
                    return {"anthropic": "❌ Invalid API key"}
                else:
                    return {"anthropic": f"❌ Error: {response.status}"}
    except Exception as e:
        return {"anthropic": f"❌ Error: {str(e)}"}

async def main():
    """Run all API tests"""
    print("\n🔍 Testing API Keys Configuration...\n")
    print("=" * 50)
    
    # Run all tests in parallel
    results = await asyncio.gather(
        test_polygon_api(),
        test_alpha_vantage_api(),
        test_finnhub_api(),
        test_twelvedata_api(),
        test_fmp_api(),
        test_openai_api(),
        test_anthropic_api()
    )
    
    # Display results
    print("\n📊 Market Data APIs:")
    print("-" * 30)
    for result in results[:5]:  # Market data APIs
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    
    print("\n🤖 AI/LLM APIs:")
    print("-" * 30)
    for result in results[5:]:  # AI APIs
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # Summary
    working_apis = sum(1 for r in results[:5] for v in r.values() if "✅" in str(v))
    print(f"\n✨ Summary: {working_apis}/5 Market Data APIs working")
    
    if any("✅" in str(v) for r in results[5:] for v in r.values()):
        print("✅ At least one LLM API is configured")
    else:
        print("⚠️  No LLM APIs configured - AI features will be limited")

if __name__ == "__main__":
    asyncio.run(main())