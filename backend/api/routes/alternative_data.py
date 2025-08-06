"""
Alternative Data API Routes
Endpoints for accessing alternative data sources and analysis
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from services.alternative_data_service import (
    alternative_data_service,
    DataSourceType
)
from agents.news_sentiment_agent import news_sentiment_agent
from agents.social_sentiment_agent import social_sentiment_agent
from agents.weather_impact_agent import weather_impact_agent
from agents.commodity_data_agent import commodity_data_agent
from agents.alternative_data_master_agent import alternative_data_master_agent

router = APIRouter(prefix="/api/alternative", tags=["Alternative Data"])


class DataType(str, Enum):
    """Available data types"""
    NEWS = "news"
    SOCIAL = "social"
    WEATHER = "weather"
    COMMODITY = "commodity"
    ECONOMIC = "economic"
    ALL = "all"


@router.get("/data/comprehensive")
async def get_comprehensive_alternative_data(
    symbols: str = Query(default="SPY,QQQ,AAPL", description="Comma-separated symbols"),
    data_types: str = Query(default="all", description="Comma-separated data types or 'all'")
):
    """
    Get comprehensive alternative data from all sources
    
    Args:
        symbols: Comma-separated list of symbols
        data_types: Types of data to fetch (news, social, weather, commodity, economic, all)
    
    Returns:
        Comprehensive alternative data from multiple sources
    """
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        
        # Parse data types
        if data_types == "all":
            data_type_list = None
        else:
            data_type_map = {
                'news': DataSourceType.NEWS,
                'social': DataSourceType.SOCIAL,
                'weather': DataSourceType.WEATHER,
                'commodity': DataSourceType.COMMODITY,
                'economic': DataSourceType.ECONOMIC
            }
            data_type_list = [
                data_type_map[dt.strip()]
                for dt in data_types.split(',')
                if dt.strip() in data_type_map
            ]
        
        # Fetch comprehensive data
        data = await alternative_data_service.get_comprehensive_alternative_data(
            symbols=symbol_list,
            data_types=data_type_list
        )
        
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news/sentiment")
async def get_news_sentiment(
    symbols: str = Query(default="SPY", description="Comma-separated symbols")
):
    """
    Get news sentiment analysis
    
    Args:
        symbols: Comma-separated list of symbols
    
    Returns:
        News articles with sentiment analysis
    """
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        
        data = await alternative_data_service.get_comprehensive_alternative_data(
            symbols=symbol_list,
            data_types=[DataSourceType.NEWS]
        )
        
        return {
            "status": "success",
            "articles": data.get('news', []),
            "analysis": data.get('analysis', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/social/sentiment")
async def get_social_sentiment(
    symbols: str = Query(default="SPY", description="Comma-separated symbols")
):
    """
    Get social media sentiment analysis
    
    Args:
        symbols: Comma-separated list of symbols
    
    Returns:
        Social media posts with sentiment analysis
    """
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        
        data = await alternative_data_service.get_comprehensive_alternative_data(
            symbols=symbol_list,
            data_types=[DataSourceType.SOCIAL]
        )
        
        return {
            "status": "success",
            "posts": data.get('social', []),
            "analysis": data.get('analysis', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather/impact")
async def get_weather_impact():
    """
    Get weather data and commodity impact analysis
    
    Returns:
        Weather conditions affecting commodities and markets
    """
    try:
        data = await alternative_data_service.get_comprehensive_alternative_data(
            data_types=[DataSourceType.WEATHER]
        )
        
        weather_data = data.get('weather', {})
        
        # Calculate impacts
        impacts = []
        for location, weather in weather_data.items():
            if weather.impact_score > 0.3:
                impacts.append({
                    'location': location,
                    'conditions': weather.conditions,
                    'temperature': weather.temperature,
                    'impact_score': weather.impact_score,
                    'affected_commodities': weather.affected_commodities
                })
        
        return {
            "status": "success",
            "weather_data": weather_data,
            "high_impact_areas": impacts,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/commodity/data")
async def get_commodity_data():
    """
    Get commodity market data
    
    Returns:
        Current commodity prices and trends
    """
    try:
        data = await alternative_data_service.get_comprehensive_alternative_data(
            data_types=[DataSourceType.COMMODITY]
        )
        
        commodities = data.get('commodity', {})
        
        # Format response
        formatted = []
        for name, commodity in commodities.items():
            formatted.append({
                'name': name,
                'price': commodity.price,
                'volume': commodity.volume,
                'exchange': commodity.exchange,
                'contract': commodity.contract,
                'sentiment': commodity.sentiment,
                'timestamp': commodity.timestamp.isoformat()
            })
        
        return {
            "status": "success",
            "commodities": formatted,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crypto/data")
async def get_crypto_data(
    symbols: str = Query(default="bitcoin,ethereum", description="Comma-separated crypto IDs")
):
    """
    Get cryptocurrency data
    
    Args:
        symbols: Comma-separated list of cryptocurrency IDs
    
    Returns:
        Cryptocurrency prices and market data
    """
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        
        data = await alternative_data_service.get_crypto_data(symbols=symbol_list)
        
        return {
            "status": "success",
            "crypto_data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}/news")
async def analyze_news_sentiment(symbol: str):
    """
    Get trading signal based on news sentiment
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Trading signal from news sentiment analysis
    """
    try:
        market_data = {"symbol": symbol, "price": 100.0}  # Placeholder price
        signal = await news_sentiment_agent.analyze(market_data)
        
        if signal:
            return {
                "status": "success",
                "signal": {
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                    "reasoning": signal.reasoning,
                    "features": signal.features
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "no_signal",
                "message": "Insufficient news data for analysis",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}/social")
async def analyze_social_sentiment(symbol: str):
    """
    Get trading signal based on social sentiment
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Trading signal from social sentiment analysis
    """
    try:
        market_data = {"symbol": symbol, "price": 100.0}
        signal = await social_sentiment_agent.analyze(market_data)
        
        if signal:
            return {
                "status": "success",
                "signal": {
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                    "reasoning": signal.reasoning,
                    "features": signal.features
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "no_signal",
                "message": "Insufficient social data for analysis",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}/weather")
async def analyze_weather_impact(symbol: str):
    """
    Get trading signal based on weather impact
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Trading signal from weather impact analysis
    """
    try:
        market_data = {"symbol": symbol, "price": 100.0}
        signal = await weather_impact_agent.analyze(market_data)
        
        if signal:
            return {
                "status": "success",
                "signal": {
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                    "reasoning": signal.reasoning,
                    "features": signal.features
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "no_signal",
                "message": "Symbol not weather-sensitive or no weather impact",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}/commodity")
async def analyze_commodity_impact(symbol: str):
    """
    Get trading signal based on commodity correlations
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Trading signal from commodity analysis
    """
    try:
        market_data = {"symbol": symbol, "price": 100.0}
        signal = await commodity_data_agent.analyze(market_data)
        
        if signal:
            return {
                "status": "success",
                "signal": {
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                    "reasoning": signal.reasoning,
                    "features": signal.features
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "no_signal",
                "message": "Symbol not commodity-sensitive or no commodity data",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}/master")
async def analyze_all_alternative_data(symbol: str):
    """
    Get comprehensive trading signal from all alternative data sources
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Master trading signal combining all alternative data sources
    """
    try:
        market_data = {"symbol": symbol, "price": 100.0}
        signal = await alternative_data_master_agent.analyze(market_data)
        
        if signal:
            return {
                "status": "success",
                "signal": {
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                    "reasoning": signal.reasoning,
                    "features": signal.features,
                    "market_conditions": signal.market_conditions
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "no_signal",
                "message": "Insufficient alternative data for comprehensive analysis",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending")
async def get_trending_topics():
    """
    Get trending topics across all alternative data sources
    
    Returns:
        Trending stocks, commodities, and topics
    """
    try:
        # Get crypto trending
        crypto_data = await alternative_data_service.get_crypto_data()
        
        # Get news and social for popular symbols
        popular_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
        alt_data = await alternative_data_service.get_comprehensive_alternative_data(
            symbols=popular_symbols,
            data_types=[DataSourceType.NEWS, DataSourceType.SOCIAL]
        )
        
        # Extract trending information
        trending = {
            'crypto': crypto_data.get('trending', []),
            'stocks': [],
            'topics': [],
            'commodities': []
        }
        
        # Extract mentioned tickers from news
        if 'news' in alt_data:
            ticker_counts = {}
            for article in alt_data['news'][:50]:
                for ticker in article.tickers:
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            
            trending['stocks'] = [
                {'symbol': ticker, 'mentions': count}
                for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        
        return {
            "status": "success",
            "trending": trending,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_alternative_data_summary():
    """
    Get summary of all alternative data sources status
    
    Returns:
        Status and availability of each data source
    """
    try:
        # Check each data source
        sources_status = {
            'news': {
                'available': alternative_data_service.api_keys.get('newsapi') is not None or
                           alternative_data_service.api_keys.get('marketaux') is not None,
                'providers': []
            },
            'social': {
                'available': alternative_data_service.api_keys.get('stocktwits') is not None or
                            alternative_data_service.api_keys.get('reddit') is not None,
                'providers': []
            },
            'weather': {
                'available': alternative_data_service.api_keys.get('openweathermap') is not None,
                'providers': ['openweathermap'] if alternative_data_service.api_keys.get('openweathermap') else []
            },
            'commodity': {
                'available': alternative_data_service.api_keys.get('quandl') is not None or
                            alternative_data_service.api_keys.get('eia') is not None,
                'providers': []
            },
            'crypto': {
                'available': True,  # CoinGecko doesn't require API key
                'providers': ['coingecko']
            }
        }
        
        # Check providers
        if alternative_data_service.api_keys.get('newsapi'):
            sources_status['news']['providers'].append('newsapi')
        if alternative_data_service.api_keys.get('marketaux'):
            sources_status['news']['providers'].append('marketaux')
        if alternative_data_service.api_keys.get('finnhub'):
            sources_status['news']['providers'].append('finnhub')
            
        if alternative_data_service.api_keys.get('stocktwits'):
            sources_status['social']['providers'].append('stocktwits')
        if alternative_data_service.api_keys.get('reddit'):
            sources_status['social']['providers'].append('reddit')
            
        if alternative_data_service.api_keys.get('quandl'):
            sources_status['commodity']['providers'].append('quandl')
        if alternative_data_service.api_keys.get('eia'):
            sources_status['commodity']['providers'].append('eia')
        
        # Count total available
        total_sources = sum(1 for s in sources_status.values() if s['available'])
        
        return {
            "status": "success",
            "sources": sources_status,
            "total_available": total_sources,
            "total_possible": len(sources_status),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))