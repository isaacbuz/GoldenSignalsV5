"""
Alternative Data Service
Integrates multiple alternative data sources for enhanced market insights
"""

import asyncio
import aiohttp
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of alternative data sources"""
    NEWS = "news"
    SOCIAL = "social"
    WEATHER = "weather"
    SATELLITE = "satellite"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    WEB_TRAFFIC = "web_traffic"
    SEARCH_TRENDS = "search_trends"
    ECONOMIC = "economic"
    COMMODITY = "commodity"
    ENERGY = "energy"
    AGRICULTURE = "agriculture"


@dataclass
class AlternativeDataPoint:
    """Standardized alternative data point"""
    source: str
    data_type: DataSourceType
    timestamp: datetime
    value: Any
    metadata: Dict[str, Any]
    relevance_score: float  # 0-1 relevance to markets
    affected_sectors: List[str]
    sentiment: Optional[float] = None  # -1 to 1


@dataclass
class NewsArticle:
    """News article with sentiment"""
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str
    sentiment: float  # -1 (negative) to 1 (positive)
    relevance: float  # 0-1
    tickers: List[str]
    categories: List[str]


@dataclass
class SocialPost:
    """Social media post with sentiment"""
    platform: str  # twitter, reddit, stocktwits
    author: str
    content: str
    timestamp: datetime
    sentiment: float
    engagement: int  # likes, retweets, etc.
    tickers: List[str]
    influence_score: float  # 0-1


@dataclass
class WeatherData:
    """Weather data affecting markets"""
    location: str
    temperature: float
    precipitation: float
    wind_speed: float
    conditions: str
    timestamp: datetime
    affected_commodities: List[str]
    impact_score: float  # -1 to 1


@dataclass
class CommodityData:
    """Commodity market data"""
    commodity: str
    price: float
    volume: float
    timestamp: datetime
    exchange: str
    contract: str  # futures contract
    sentiment: float


class AlternativeDataService:
    """Service for fetching and processing alternative data"""
    
    def __init__(self):
        # API keys for various services
        self.api_keys = {
            # News APIs
            'newsapi': os.getenv('NEWSAPI_KEY'),
            'marketaux': os.getenv('MARKETAUX_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'alphavantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            
            # Social APIs
            'twitter': os.getenv('TWITTER_BEARER_TOKEN'),
            'reddit': os.getenv('REDDIT_CLIENT_ID'),
            'reddit_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'stocktwits': os.getenv('STOCKTWITS_TOKEN'),
            
            # Weather & Environmental
            'openweathermap': os.getenv('OPENWEATHER_API_KEY'),
            'noaa': os.getenv('NOAA_API_KEY'),
            
            # Commodity & Energy
            'eia': os.getenv('EIA_API_KEY'),  # Energy Information Administration
            'usda': os.getenv('USDA_API_KEY'),  # Agriculture data
            
            # Alternative Data
            'quandl': os.getenv('QUANDL_API_KEY'),
            'google_trends': os.getenv('GOOGLE_TRENDS_KEY'),
            
            # Crypto
            'coingecko': None,  # No key required
            'etherscan': os.getenv('ETHERSCAN_API_KEY'),
            
            # Global Economic
            'worldbank': None,  # No key required
            'oecd': os.getenv('OECD_API_KEY'),
            'ecb': None,  # No key required
        }
        
        # API endpoints
        self.endpoints = {
            'newsapi': 'https://newsapi.org/v2',
            'marketaux': 'https://api.marketaux.com/v1',
            'finnhub': 'https://finnhub.io/api/v1',
            'alphavantage': 'https://www.alphavantage.co/query',
            'stocktwits': 'https://api.stocktwits.com/api/2',
            'reddit': 'https://oauth.reddit.com',
            'twitter': 'https://api.twitter.com/2',
            'openweathermap': 'https://api.openweathermap.org/data/2.5',
            'coingecko': 'https://api.coingecko.com/api/v3',
            'etherscan': 'https://api.etherscan.io/api',
            'worldbank': 'https://api.worldbank.org/v2',
            'eia': 'https://api.eia.gov',
            'usda': 'https://quickstats.nass.usda.gov/api',
            'quandl': 'https://www.quandl.com/api/v3',
            'ecb': 'https://data-api.ecb.europa.eu/service/data',
            'oecd': 'https://stats.oecd.org/sdmx-json/data',
        }
        
        # Cache for API responses
        self.cache = {}
        self.cache_expiry = {}
        
        # Sector mappings for different data types
        self.sector_impacts = {
            'weather': {
                'extreme_heat': ['energy', 'utilities', 'agriculture'],
                'extreme_cold': ['energy', 'utilities', 'retail'],
                'drought': ['agriculture', 'water_utilities', 'food'],
                'flooding': ['insurance', 'real_estate', 'agriculture'],
                'hurricanes': ['insurance', 'energy', 'retail']
            },
            'commodity': {
                'oil': ['energy', 'transportation', 'chemicals'],
                'gold': ['mining', 'jewelry', 'technology'],
                'wheat': ['food', 'agriculture', 'consumer_staples'],
                'copper': ['mining', 'construction', 'technology'],
                'natural_gas': ['utilities', 'chemicals', 'energy']
            },
            'social_sentiment': {
                'bullish': ['technology', 'growth_stocks', 'crypto'],
                'bearish': ['utilities', 'consumer_staples', 'bonds'],
                'meme_stocks': ['retail_favorites', 'heavily_shorted']
            }
        }
    
    async def get_comprehensive_alternative_data(
        self,
        symbols: List[str] = None,
        data_types: List[DataSourceType] = None
    ) -> Dict[str, Any]:
        """Fetch comprehensive alternative data from all sources"""
        
        if not symbols:
            symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
        
        if not data_types:
            data_types = [DataSourceType.NEWS, DataSourceType.SOCIAL, 
                         DataSourceType.WEATHER, DataSourceType.COMMODITY]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Fetch different data types in parallel
            if DataSourceType.NEWS in data_types:
                tasks.append(self._fetch_news_sentiment(session, symbols))
            
            if DataSourceType.SOCIAL in data_types:
                tasks.append(self._fetch_social_sentiment(session, symbols))
            
            if DataSourceType.WEATHER in data_types:
                tasks.append(self._fetch_weather_data(session))
            
            if DataSourceType.COMMODITY in data_types:
                tasks.append(self._fetch_commodity_data(session))
            
            if DataSourceType.ECONOMIC in data_types:
                tasks.append(self._fetch_global_economic_data(session))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            data = {
                'timestamp': datetime.utcnow(),
                'symbols': symbols,
                'news': results[0] if len(results) > 0 and not isinstance(results[0], Exception) else [],
                'social': results[1] if len(results) > 1 and not isinstance(results[1], Exception) else [],
                'weather': results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {},
                'commodity': results[3] if len(results) > 3 and not isinstance(results[3], Exception) else {},
                'economic': results[4] if len(results) > 4 and not isinstance(results[4], Exception) else {},
                'analysis': self._analyze_alternative_data(results)
            }
            
            return data
    
    async def _fetch_news_sentiment(
        self,
        session: aiohttp.ClientSession,
        symbols: List[str]
    ) -> List[NewsArticle]:
        """Fetch news with sentiment analysis"""
        
        articles = []
        
        # Marketaux API (best for financial news)
        if self.api_keys.get('marketaux'):
            try:
                params = {
                    'api_token': self.api_keys['marketaux'],
                    'symbols': ','.join(symbols),
                    'filter_entities': 'true',
                    'language': 'en',
                    'limit': 50
                }
                
                url = f"{self.endpoints['marketaux']}/news/all"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for article in data.get('data', []):
                            articles.append(NewsArticle(
                                title=article['title'],
                                source=article['source'],
                                url=article['url'],
                                published_at=datetime.fromisoformat(article['published_at']),
                                summary=article.get('description', ''),
                                sentiment=self._calculate_sentiment(article.get('sentiment', 0)),
                                relevance=article.get('relevance_score', 0.5),
                                tickers=article.get('symbols', []),
                                categories=article.get('topics', [])
                            ))
            except Exception as e:
                logger.error(f"Marketaux API error: {e}")
        
        # Finnhub News (backup)
        if self.api_keys.get('finnhub') and len(articles) < 10:
            try:
                for symbol in symbols[:3]:  # Limit to avoid rate limits
                    params = {
                        'symbol': symbol,
                        'token': self.api_keys['finnhub']
                    }
                    
                    url = f"{self.endpoints['finnhub']}/news"
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            for item in data[:5]:  # Limit per symbol
                                articles.append(NewsArticle(
                                    title=item['headline'],
                                    source=item['source'],
                                    url=item['url'],
                                    published_at=datetime.fromtimestamp(item['datetime']),
                                    summary=item['summary'],
                                    sentiment=0.0,  # Finnhub doesn't provide sentiment
                                    relevance=0.7,
                                    tickers=[symbol],
                                    categories=[item.get('category', 'general')]
                                ))
            except Exception as e:
                logger.error(f"Finnhub news error: {e}")
        
        # Alpha Vantage News Sentiment
        if self.api_keys.get('alphavantage'):
            try:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ','.join(symbols[:5]),
                    'apikey': self.api_keys['alphavantage']
                }
                
                url = self.endpoints['alphavantage']
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for item in data.get('feed', [])[:10]:
                            sentiment_scores = item.get('overall_sentiment_score', 0)
                            articles.append(NewsArticle(
                                title=item['title'],
                                source=item['source'],
                                url=item['url'],
                                published_at=datetime.strptime(item['time_published'], '%Y%m%dT%H%M%S'),
                                summary=item.get('summary', ''),
                                sentiment=float(sentiment_scores),
                                relevance=item.get('relevance_score', 0.5),
                                tickers=[t['ticker'] for t in item.get('ticker_sentiment', [])],
                                categories=item.get('topics', [])
                            ))
            except Exception as e:
                logger.error(f"Alpha Vantage news error: {e}")
        
        return articles
    
    async def _fetch_social_sentiment(
        self,
        session: aiohttp.ClientSession,
        symbols: List[str]
    ) -> List[SocialPost]:
        """Fetch social media sentiment"""
        
        posts = []
        
        # StockTwits
        if self.api_keys.get('stocktwits'):
            try:
                for symbol in symbols[:5]:
                    url = f"{self.endpoints['stocktwits']}/streams/symbol/{symbol}.json"
                    headers = {'Authorization': f"Bearer {self.api_keys['stocktwits']}"}
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            for message in data.get('messages', [])[:10]:
                                sentiment_val = 0.5
                                if message.get('entities', {}).get('sentiment'):
                                    sentiment_val = 1.0 if message['entities']['sentiment']['basic'] == 'Bullish' else -1.0
                                
                                posts.append(SocialPost(
                                    platform='stocktwits',
                                    author=message['user']['username'],
                                    content=message['body'],
                                    timestamp=datetime.fromisoformat(message['created_at']),
                                    sentiment=sentiment_val,
                                    engagement=message.get('likes', {}).get('total', 0),
                                    tickers=[symbol],
                                    influence_score=min(message['user']['followers'] / 10000, 1.0)
                                ))
            except Exception as e:
                logger.error(f"StockTwits error: {e}")
        
        # Reddit (r/wallstreetbets, r/stocks)
        if self.api_keys.get('reddit') and self.api_keys.get('reddit_secret'):
            try:
                # Get Reddit access token
                auth = aiohttp.BasicAuth(self.api_keys['reddit'], self.api_keys['reddit_secret'])
                data = {'grant_type': 'client_credentials'}
                headers = {'User-Agent': 'GoldenSignalsAI/1.0'}
                
                async with session.post('https://www.reddit.com/api/v1/access_token',
                                       auth=auth, data=data, headers=headers) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        access_token = token_data['access_token']
                        
                        # Fetch posts from relevant subreddits
                        headers = {
                            'Authorization': f"Bearer {access_token}",
                            'User-Agent': 'GoldenSignalsAI/1.0'
                        }
                        
                        for subreddit in ['wallstreetbets', 'stocks', 'investing']:
                            url = f"{self.endpoints['reddit']}/r/{subreddit}/hot.json?limit=10"
                            
                            async with session.get(url, headers=headers) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    for post in data['data']['children']:
                                        post_data = post['data']
                                        
                                        # Simple sentiment based on upvote ratio
                                        sentiment = (post_data['upvote_ratio'] - 0.5) * 2
                                        
                                        # Extract tickers from title/text
                                        tickers = self._extract_tickers(post_data['title'])
                                        
                                        if tickers:
                                            posts.append(SocialPost(
                                                platform='reddit',
                                                author=post_data['author'],
                                                content=post_data['title'],
                                                timestamp=datetime.fromtimestamp(post_data['created_utc']),
                                                sentiment=sentiment,
                                                engagement=post_data['score'],
                                                tickers=tickers,
                                                influence_score=min(post_data['score'] / 1000, 1.0)
                                            ))
            except Exception as e:
                logger.error(f"Reddit error: {e}")
        
        return posts
    
    async def _fetch_weather_data(
        self,
        session: aiohttp.ClientSession
    ) -> Dict[str, WeatherData]:
        """Fetch weather data for key commodity regions"""
        
        weather_data = {}
        
        if not self.api_keys.get('openweathermap'):
            return weather_data
        
        # Key locations affecting commodities
        locations = {
            'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'commodities': ['wheat', 'corn', 'soybeans']},
            'Houston': {'lat': 29.7604, 'lon': -95.3698, 'commodities': ['oil', 'natural_gas', 'chemicals']},
            'London': {'lat': 51.5074, 'lon': -0.1278, 'commodities': ['metals', 'oil']},
            'Singapore': {'lat': 1.3521, 'lon': 103.8198, 'commodities': ['oil', 'shipping']},
            'Sydney': {'lat': -33.8688, 'lon': 151.2093, 'commodities': ['coal', 'iron_ore']},
        }
        
        try:
            for city, info in locations.items():
                params = {
                    'lat': info['lat'],
                    'lon': info['lon'],
                    'appid': self.api_keys['openweathermap'],
                    'units': 'metric'
                }
                
                url = f"{self.endpoints['openweathermap']}/weather"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Calculate impact score based on extreme conditions
                        impact = 0.0
                        temp = data['main']['temp']
                        wind_speed = data['wind']['speed']
                        
                        if temp > 35:  # Extreme heat
                            impact = 0.5
                        elif temp < -10:  # Extreme cold
                            impact = 0.5
                        
                        if wind_speed > 15:  # High winds
                            impact += 0.3
                        
                        if 'rain' in data:
                            impact += 0.2
                        
                        weather_data[city] = WeatherData(
                            location=city,
                            temperature=temp,
                            precipitation=data.get('rain', {}).get('1h', 0),
                            wind_speed=wind_speed,
                            conditions=data['weather'][0]['main'],
                            timestamp=datetime.fromtimestamp(data['dt']),
                            affected_commodities=info['commodities'],
                            impact_score=min(impact, 1.0)
                        )
        except Exception as e:
            logger.error(f"Weather API error: {e}")
        
        return weather_data
    
    async def _fetch_commodity_data(
        self,
        session: aiohttp.ClientSession
    ) -> Dict[str, CommodityData]:
        """Fetch commodity market data"""
        
        commodities = {}
        
        # Quandl for commodity data
        if self.api_keys.get('quandl'):
            try:
                # Key commodities
                commodity_codes = {
                    'oil': 'CHRIS/CME_CL1',  # WTI Crude Oil
                    'gold': 'CHRIS/CME_GC1',  # Gold
                    'silver': 'CHRIS/CME_SI1',  # Silver
                    'wheat': 'CHRIS/CME_W1',  # Wheat
                    'corn': 'CHRIS/CME_C1',  # Corn
                    'natural_gas': 'CHRIS/CME_NG1',  # Natural Gas
                }
                
                for name, code in commodity_codes.items():
                    params = {
                        'api_key': self.api_keys['quandl'],
                        'limit': 1
                    }
                    
                    url = f"{self.endpoints['quandl']}/datasets/{code}.json"
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            dataset = data.get('dataset', {})
                            if dataset.get('data'):
                                latest = dataset['data'][0]
                                # Quandl data structure: Date, Open, High, Low, Last, Change, Settle, Volume
                                commodities[name] = CommodityData(
                                    commodity=name,
                                    price=latest[6] if len(latest) > 6 else latest[4],  # Settle or Last
                                    volume=latest[7] if len(latest) > 7 else 0,
                                    timestamp=datetime.strptime(latest[0], '%Y-%m-%d'),
                                    exchange='CME',
                                    contract=code.split('/')[1],
                                    sentiment=0.0  # Calculate based on price change
                                )
            except Exception as e:
                logger.error(f"Quandl error: {e}")
        
        # EIA for energy data
        if self.api_keys.get('eia'):
            try:
                # Weekly petroleum status
                params = {
                    'api_key': self.api_keys['eia'],
                    'series_id': 'PET.WCESTUS1.W'  # US crude oil stocks
                }
                
                url = f"{self.endpoints['eia']}/series/"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('series'):
                            series = data['series'][0]
                            latest = series['data'][0]
                            
                            # Add to commodities as inventory data
                            commodities['oil_inventory'] = CommodityData(
                                commodity='oil_inventory',
                                price=latest[1],  # Inventory level
                                volume=0,
                                timestamp=datetime.strptime(latest[0], '%Y%m%d'),
                                exchange='EIA',
                                contract='WCESTUS1',
                                sentiment=0.0
                            )
            except Exception as e:
                logger.error(f"EIA error: {e}")
        
        return commodities
    
    async def _fetch_global_economic_data(
        self,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Fetch global economic indicators"""
        
        economic_data = {}
        
        # World Bank API (no key required)
        try:
            # GDP growth for major economies
            countries = 'US;CN;JP;DE;GB;FR'  # USA, China, Japan, Germany, UK, France
            indicator = 'NY.GDP.MKTP.KD.ZG'  # GDP growth
            
            url = f"{self.endpoints['worldbank']}/country/{countries}/indicator/{indicator}"
            params = {
                'format': 'json',
                'per_page': 10,
                'date': '2020:2024'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 1:
                        for entry in data[1]:
                            if entry['value'] is not None:
                                country = entry['country']['value']
                                if country not in economic_data:
                                    economic_data[country] = {}
                                economic_data[country]['gdp_growth'] = {
                                    'value': entry['value'],
                                    'year': entry['date']
                                }
        except Exception as e:
            logger.error(f"World Bank error: {e}")
        
        # ECB data (European Central Bank)
        try:
            # EUR exchange rates
            url = f"{self.endpoints['ecb']}/EXR/D.USD.EUR.SP00.A"
            params = {'format': 'json'}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Parse ECB's complex JSON structure
                    if 'dataSets' in data and data['dataSets']:
                        observations = data['dataSets'][0].get('series', {}).get('0:0:0:0:0', {}).get('observations', {})
                        if observations:
                            latest_key = max(observations.keys())
                            economic_data['EUR_USD'] = {
                                'rate': observations[latest_key][0],
                                'timestamp': datetime.utcnow()
                            }
        except Exception as e:
            logger.error(f"ECB error: {e}")
        
        return economic_data
    
    async def get_crypto_data(
        self,
        symbols: List[str] = None
    ) -> Dict[str, Any]:
        """Fetch cryptocurrency data from CoinGecko (no API key required)"""
        
        if not symbols:
            symbols = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']
        
        crypto_data = {}
        
        async with aiohttp.ClientSession() as session:
            try:
                # CoinGecko API
                params = {
                    'ids': ','.join(symbols),
                    'vs_currencies': 'usd',
                    'include_market_cap': 'true',
                    'include_24hr_vol': 'true',
                    'include_24hr_change': 'true'
                }
                
                url = f"{self.endpoints['coingecko']}/simple/price"
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for coin_id, coin_data in data.items():
                            crypto_data[coin_id] = {
                                'price': coin_data['usd'],
                                'market_cap': coin_data.get('usd_market_cap', 0),
                                'volume_24h': coin_data.get('usd_24h_vol', 0),
                                'change_24h': coin_data.get('usd_24h_change', 0),
                                'timestamp': datetime.utcnow()
                            }
                
                # Get trending coins
                url = f"{self.endpoints['coingecko']}/search/trending"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        crypto_data['trending'] = [
                            coin['item']['id'] for coin in data.get('coins', [])[:5]
                        ]
                        
            except Exception as e:
                logger.error(f"CoinGecko error: {e}")
        
        return crypto_data
    
    def _calculate_sentiment(self, value: Any) -> float:
        """Calculate normalized sentiment score"""
        
        if isinstance(value, (int, float)):
            # Normalize to -1 to 1 range
            return max(-1, min(1, float(value)))
        elif isinstance(value, str):
            # Simple keyword-based sentiment
            positive_words = ['bullish', 'buy', 'long', 'growth', 'profit', 'gain']
            negative_words = ['bearish', 'sell', 'short', 'loss', 'decline', 'crash']
            
            text_lower = value.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return 0.5
            elif neg_count > pos_count:
                return -0.5
            else:
                return 0.0
        
        return 0.0
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        
        import re
        
        # Look for $TICKER or common ticker patterns
        pattern = r'\$([A-Z]{1,5})\b|(?:^|\s)([A-Z]{2,5})(?:\s|$)'
        matches = re.findall(pattern, text)
        
        tickers = []
        for match in matches:
            ticker = match[0] if match[0] else match[1]
            if ticker and len(ticker) <= 5 and ticker.isupper():
                # Filter out common words that might match pattern
                if ticker not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'NEW', 'OLD']:
                    tickers.append(ticker)
        
        return list(set(tickers))
    
    def _analyze_alternative_data(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze all alternative data sources for insights"""
        
        analysis = {
            'market_sentiment': 'neutral',
            'risk_factors': [],
            'opportunities': [],
            'key_themes': [],
            'sentiment_score': 0.0
        }
        
        # Aggregate sentiment from news and social
        total_sentiment = 0.0
        sentiment_count = 0
        
        # News sentiment
        if results and len(results) > 0 and isinstance(results[0], list):
            for article in results[0]:
                if hasattr(article, 'sentiment'):
                    total_sentiment += article.sentiment
                    sentiment_count += 1
        
        # Social sentiment
        if len(results) > 1 and isinstance(results[1], list):
            for post in results[1]:
                if hasattr(post, 'sentiment'):
                    total_sentiment += post.sentiment * post.influence_score
                    sentiment_count += 1
        
        if sentiment_count > 0:
            avg_sentiment = total_sentiment / sentiment_count
            analysis['sentiment_score'] = avg_sentiment
            
            if avg_sentiment > 0.3:
                analysis['market_sentiment'] = 'bullish'
            elif avg_sentiment < -0.3:
                analysis['market_sentiment'] = 'bearish'
            else:
                analysis['market_sentiment'] = 'neutral'
        
        # Weather impacts
        if len(results) > 2 and isinstance(results[2], dict):
            for location, weather in results[2].items():
                if hasattr(weather, 'impact_score') and weather.impact_score > 0.5:
                    analysis['risk_factors'].append(
                        f"Extreme weather in {location} affecting {', '.join(weather.affected_commodities)}"
                    )
        
        # Commodity trends
        if len(results) > 3 and isinstance(results[3], dict):
            for name, commodity in results[3].items():
                if hasattr(commodity, 'price'):
                    # Add commodity insights
                    pass
        
        return analysis
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        
        if key in self.cache and key in self.cache_expiry:
            return datetime.utcnow() < self.cache_expiry[key]
        return False
    
    def _cache_data(self, key: str, data: Any, minutes: int = 5):
        """Cache data with expiration"""
        
        self.cache[key] = data
        self.cache_expiry[key] = datetime.utcnow() + timedelta(minutes=minutes)


# Create global instance
alternative_data_service = AlternativeDataService()