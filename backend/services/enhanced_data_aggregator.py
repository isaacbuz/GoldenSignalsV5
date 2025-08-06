"""
Enhanced Data Aggregator Service
Integrates multiple premium data sources for comprehensive market analysis
"""

import asyncio
import aiohttp
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""
    # Market Data
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    FINANCIAL_MODELING_PREP = "fmp"
    
    # Economic Data
    FRED = "fred"
    US_TREASURY = "treasury"
    DATA_GOV = "data_gov"
    
    # Cryptocurrency
    COINGECKO = "coingecko"
    BINANCE = "binance"
    COINBASE = "coinbase"
    
    # News & Sentiment
    NEWSAPI = "newsapi"
    CRYPTO_NEWS = "crypto_news"
    
    # Alternative Data
    WEATHER = "openweathermap"
    AVIATION = "aviationstack"
    
    # Forex
    FIXER = "fixer"
    EXCHANGE_RATES = "exchangerates"


@dataclass
class MarketDataPoint:
    """Unified market data structure"""
    symbol: str
    timestamp: datetime
    source: str
    data_type: str
    value: float
    metadata: Dict[str, Any]


@dataclass
class EconomicIndicator:
    """Economic indicator data"""
    indicator: str
    value: float
    previous_value: float
    timestamp: datetime
    source: str
    impact: str  # high, medium, low
    forecast: Optional[float] = None


@dataclass
class AlternativeData:
    """Alternative data for market analysis"""
    data_type: str  # weather, transportation, social, etc.
    location: Optional[str]
    value: Any
    timestamp: datetime
    relevance_score: float  # 0-1 how relevant to markets
    affected_sectors: List[str]


class EnhancedDataAggregator:
    """Aggregates data from multiple premium sources"""
    
    def __init__(self):
        # API Keys
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'iex_cloud': os.getenv('IEX_CLOUD_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'polygon': os.getenv('POLYGON_API_KEY'),
            'fmp': os.getenv('FMP_API_KEY'),
            'fred': os.getenv('FRED_API_KEY'),
            'newsapi': os.getenv('NEWS_API_KEY'),
            'openweathermap': os.getenv('OPENWEATHER_API_KEY'),
            'coingecko': os.getenv('COINGECKO_API_KEY'),
            'aviationstack': os.getenv('AVIATIONSTACK_API_KEY'),
            'fixer': os.getenv('FIXER_API_KEY'),
        }
        
        # Cache configuration
        self.cache = {}
        self.cache_ttl = {
            'market_data': timedelta(minutes=1),
            'economic_data': timedelta(hours=1),
            'weather_data': timedelta(hours=3),
            'news_data': timedelta(minutes=15),
        }
        
        # Rate limiting
        self.rate_limits = {
            'alpha_vantage': {'calls': 5, 'period': 60},  # 5 calls/minute
            'iex_cloud': {'calls': 100, 'period': 1},     # 100 calls/second
            'finnhub': {'calls': 60, 'period': 60},       # 60 calls/minute
            'fred': {'calls': 120, 'period': 60},         # 120 calls/minute
        }
        
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize the aggregator"""
        self.session = aiohttp.ClientSession()
        logger.info("Enhanced Data Aggregator initialized")
        
    async def close(self):
        """Close the aggregator"""
        if self.session:
            await self.session.close()
            
    async def get_comprehensive_market_data(
        self, 
        symbol: str,
        include_sources: List[DataSource] = None
    ) -> Dict[str, Any]:
        """Get comprehensive market data from multiple sources"""
        
        if not include_sources:
            include_sources = [
                DataSource.ALPHA_VANTAGE,
                DataSource.IEX_CLOUD,
                DataSource.FINNHUB,
                DataSource.POLYGON,
            ]
            
        tasks = []
        
        # Gather data from all sources
        for source in include_sources:
            if source == DataSource.ALPHA_VANTAGE and self.api_keys.get('alpha_vantage'):
                tasks.append(self._get_alpha_vantage_data(symbol))
            elif source == DataSource.IEX_CLOUD and self.api_keys.get('iex_cloud'):
                tasks.append(self._get_iex_cloud_data(symbol))
            elif source == DataSource.FINNHUB and self.api_keys.get('finnhub'):
                tasks.append(self._get_finnhub_data(symbol))
            elif source == DataSource.POLYGON and self.api_keys.get('polygon'):
                tasks.append(self._get_polygon_data(symbol))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        aggregated_data = self._aggregate_market_data(results)
        
        # Add technical indicators
        if aggregated_data.get('historical_data'):
            aggregated_data['technical_indicators'] = self._calculate_advanced_indicators(
                aggregated_data['historical_data']
            )
            
        return aggregated_data
        
    async def get_economic_indicators(
        self,
        indicators: List[str] = None
    ) -> List[EconomicIndicator]:
        """Get economic indicators from FRED and other sources"""
        
        if not indicators:
            indicators = [
                'DGS10',      # 10-Year Treasury Rate
                'DFF',        # Federal Funds Rate
                'UNRATE',     # Unemployment Rate
                'CPIAUCSL',   # Consumer Price Index
                'GDPC1',      # Real GDP
                'DXY',        # US Dollar Index
                'DCOILWTICO', # Crude Oil Prices
                'GOLDAMGBD228NLBM', # Gold Price
            ]
            
        economic_data = []
        
        # FRED API
        if self.api_keys.get('fred'):
            for indicator in indicators:
                try:
                    data = await self._get_fred_data(indicator)
                    if data:
                        economic_data.append(data)
                except Exception as e:
                    logger.error(f"Error fetching {indicator}: {e}")
                    
        # US Treasury Data
        treasury_data = await self._get_treasury_data()
        if treasury_data:
            economic_data.extend(treasury_data)
            
        return economic_data
        
    async def get_alternative_data(
        self,
        data_types: List[str] = None
    ) -> List[AlternativeData]:
        """Get alternative data sources"""
        
        if not data_types:
            data_types = ['weather', 'transportation', 'social']
            
        alt_data = []
        
        # Weather data (affects commodities, energy, agriculture)
        if 'weather' in data_types and self.api_keys.get('openweathermap'):
            weather_data = await self._get_weather_data()
            if weather_data:
                alt_data.extend(weather_data)
                
        # Aviation data (economic indicator)
        if 'transportation' in data_types and self.api_keys.get('aviationstack'):
            aviation_data = await self._get_aviation_data()
            if aviation_data:
                alt_data.extend(aviation_data)
                
        return alt_data
        
    async def get_crypto_data(
        self,
        symbols: List[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive cryptocurrency data"""
        
        if not symbols:
            symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP']
            
        crypto_data = {}
        
        # CoinGecko (free tier)
        coingecko_data = await self._get_coingecko_data(symbols)
        if coingecko_data:
            crypto_data['coingecko'] = coingecko_data
            
        # Binance real-time data
        binance_data = await self._get_binance_data(symbols)
        if binance_data:
            crypto_data['binance'] = binance_data
            
        # On-chain metrics
        onchain_data = await self._get_onchain_metrics(symbols)
        if onchain_data:
            crypto_data['onchain'] = onchain_data
            
        return crypto_data
        
    async def get_forex_data(
        self,
        pairs: List[str] = None
    ) -> Dict[str, Any]:
        """Get foreign exchange data"""
        
        if not pairs:
            pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF']
            
        forex_data = {}
        
        if self.api_keys.get('fixer'):
            fixer_data = await self._get_fixer_data(pairs)
            if fixer_data:
                forex_data['rates'] = fixer_data
                
        return forex_data
        
    async def get_news_sentiment(
        self,
        queries: List[str],
        sources: List[str] = None
    ) -> Dict[str, Any]:
        """Get news and sentiment data"""
        
        news_data = {}
        
        if self.api_keys.get('newsapi'):
            news = await self._get_newsapi_data(queries, sources)
            if news:
                news_data['articles'] = news
                news_data['sentiment'] = self._analyze_news_sentiment(news)
                
        return news_data
        
    # Private methods for each data source
    
    async def _get_alpha_vantage_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage"""
        try:
            # Real-time quote
            quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_keys['alpha_vantage']}"
            
            async with self.session.get(quote_url) as response:
                if response.status == 200:
                    data = await response.json()
                    quote = data.get('Global Quote', {})
                    
                    return {
                        'source': 'alpha_vantage',
                        'symbol': symbol,
                        'price': float(quote.get('05. price', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': quote.get('10. change percent', '0%'),
                        'volume': int(quote.get('06. volume', 0)),
                        'timestamp': datetime.utcnow()
                    }
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return None
            
    async def _get_iex_cloud_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from IEX Cloud"""
        try:
            base_url = "https://cloud.iexapis.com/stable"
            token = self.api_keys['iex_cloud']
            
            # Get quote and stats
            quote_url = f"{base_url}/stock/{symbol}/quote?token={token}"
            stats_url = f"{base_url}/stock/{symbol}/stats?token={token}"
            
            async with self.session.get(quote_url) as response:
                if response.status == 200:
                    quote = await response.json()
                    
                    # Get additional stats
                    async with self.session.get(stats_url) as stats_response:
                        stats = await stats_response.json() if stats_response.status == 200 else {}
                    
                    return {
                        'source': 'iex_cloud',
                        'symbol': symbol,
                        'price': quote.get('latestPrice'),
                        'change': quote.get('change'),
                        'change_percent': quote.get('changePercent'),
                        'volume': quote.get('volume'),
                        'market_cap': quote.get('marketCap'),
                        'pe_ratio': quote.get('peRatio'),
                        '52_week_high': quote.get('week52High'),
                        '52_week_low': quote.get('week52Low'),
                        'stats': stats,
                        'timestamp': datetime.utcnow()
                    }
        except Exception as e:
            logger.error(f"IEX Cloud error: {e}")
            return None
            
    async def _get_finnhub_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Finnhub"""
        try:
            # Quote
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_keys['finnhub']}"
            
            # Metrics
            metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={self.api_keys['finnhub']}"
            
            async with self.session.get(quote_url) as response:
                if response.status == 200:
                    quote = await response.json()
                    
                    # Get metrics
                    async with self.session.get(metrics_url) as metrics_response:
                        metrics = await metrics_response.json() if metrics_response.status == 200 else {}
                    
                    return {
                        'source': 'finnhub',
                        'symbol': symbol,
                        'price': quote.get('c'),
                        'change': quote.get('d'),
                        'change_percent': quote.get('dp'),
                        'high': quote.get('h'),
                        'low': quote.get('l'),
                        'open': quote.get('o'),
                        'previous_close': quote.get('pc'),
                        'metrics': metrics.get('metric', {}),
                        'timestamp': datetime.fromtimestamp(quote.get('t', datetime.utcnow().timestamp()))
                    }
        except Exception as e:
            logger.error(f"Finnhub error: {e}")
            return None
            
    async def _get_polygon_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Polygon"""
        try:
            # Previous close
            prev_close_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={self.api_keys['polygon']}"
            
            # Real-time aggregates
            date = datetime.utcnow().strftime('%Y-%m-%d')
            aggs_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}?apiKey={self.api_keys['polygon']}"
            
            async with self.session.get(prev_close_url) as response:
                if response.status == 200:
                    prev_data = await response.json()
                    results = prev_data.get('results', [{}])[0]
                    
                    return {
                        'source': 'polygon',
                        'symbol': symbol,
                        'price': results.get('c'),
                        'volume': results.get('v'),
                        'weighted_avg': results.get('vw'),
                        'open': results.get('o'),
                        'close': results.get('c'),
                        'high': results.get('h'),
                        'low': results.get('l'),
                        'timestamp': datetime.fromtimestamp(results.get('t', 0) / 1000)
                    }
        except Exception as e:
            logger.error(f"Polygon error: {e}")
            return None
            
    async def _get_fred_data(self, series_id: str) -> Optional[EconomicIndicator]:
        """Fetch data from FRED API"""
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.api_keys['fred']}&file_type=json&limit=2&sort_order=desc"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = data.get('observations', [])
                    
                    if len(observations) >= 2:
                        latest = observations[0]
                        previous = observations[1]
                        
                        # Determine impact based on indicator
                        impact_map = {
                            'DFF': 'high',        # Fed Funds Rate
                            'DGS10': 'high',      # 10-Year Treasury
                            'UNRATE': 'high',     # Unemployment
                            'CPIAUCSL': 'high',   # CPI
                            'GDPC1': 'high',      # GDP
                            'DXY': 'medium',      # Dollar Index
                        }
                        
                        return EconomicIndicator(
                            indicator=series_id,
                            value=float(latest['value']),
                            previous_value=float(previous['value']),
                            timestamp=datetime.strptime(latest['date'], '%Y-%m-%d'),
                            source='FRED',
                            impact=impact_map.get(series_id, 'medium')
                        )
        except Exception as e:
            logger.error(f"FRED API error: {e}")
            return None
            
    async def _get_treasury_data(self) -> List[EconomicIndicator]:
        """Fetch data from US Treasury"""
        try:
            # Treasury yield curve rates
            url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/avg_interest_rates?filter=record_date:gte:2024-01-01&sort=-record_date&page[size]=10"
            
            indicators = []
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    records = data.get('data', [])
                    
                    for record in records[:2]:  # Latest 2 records
                        indicators.append(EconomicIndicator(
                            indicator='Treasury_Avg_Interest_Rate',
                            value=float(record.get('avg_interest_rate_amt', 0)),
                            previous_value=0,  # Would need to track
                            timestamp=datetime.strptime(record['record_date'], '%Y-%m-%d'),
                            source='US_Treasury',
                            impact='high'
                        ))
                        
            return indicators
        except Exception as e:
            logger.error(f"Treasury API error: {e}")
            return []
            
    async def _get_weather_data(self) -> List[AlternativeData]:
        """Get weather data affecting commodities"""
        try:
            # Key agricultural and energy regions
            locations = [
                {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298, 'affects': ['corn', 'wheat', 'soybeans']},
                {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698, 'affects': ['oil', 'natural_gas', 'energy']},
                {'name': 'London', 'lat': 51.5074, 'lon': -0.1278, 'affects': ['brent', 'european_markets']},
            ]
            
            weather_data = []
            
            for location in locations:
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={location['lat']}&lon={location['lon']}&appid={self.api_keys['openweathermap']}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Analyze weather impact
                        temp = data['main']['temp'] - 273.15  # Convert to Celsius
                        conditions = data['weather'][0]['main']
                        
                        # Calculate relevance based on extreme conditions
                        relevance = 0.3  # Base relevance
                        if temp < -10 or temp > 40:  # Extreme temperatures
                            relevance = 0.8
                        elif conditions in ['Storm', 'Hurricane', 'Tornado']:
                            relevance = 0.9
                        elif conditions in ['Rain', 'Snow']:
                            relevance = 0.5
                            
                        weather_data.append(AlternativeData(
                            data_type='weather',
                            location=location['name'],
                            value={
                                'temperature': temp,
                                'conditions': conditions,
                                'humidity': data['main']['humidity'],
                                'wind_speed': data['wind']['speed']
                            },
                            timestamp=datetime.utcnow(),
                            relevance_score=relevance,
                            affected_sectors=location['affects']
                        ))
                        
            return weather_data
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return []
            
    async def _get_coingecko_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get cryptocurrency data from CoinGecko"""
        try:
            # Map symbols to CoinGecko IDs
            symbol_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'SOL': 'solana',
                'XRP': 'ripple'
            }
            
            ids = ','.join([symbol_map.get(s, s.lower()) for s in symbols])
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"CoinGecko error: {e}")
            return {}
            
    async def _get_binance_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time crypto data from Binance"""
        try:
            data = {}
            
            for symbol in symbols:
                ticker = f"{symbol}USDT"
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={ticker}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        ticker_data = await response.json()
                        data[symbol] = {
                            'price': float(ticker_data['lastPrice']),
                            'volume_24h': float(ticker_data['volume']),
                            'change_24h': float(ticker_data['priceChangePercent'])
                        }
                        
            return data
        except Exception as e:
            logger.error(f"Binance error: {e}")
            return {}
            
    async def _get_onchain_metrics(self, symbols: List[str]) -> Dict[str, Any]:
        """Get on-chain metrics for cryptocurrencies"""
        # This would integrate with blockchain explorers
        # For now, return placeholder
        return {
            'BTC': {
                'hash_rate': 'placeholder',
                'difficulty': 'placeholder',
                'active_addresses': 'placeholder'
            }
        }
        
    async def _get_newsapi_data(
        self, 
        queries: List[str], 
        sources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get news from NewsAPI"""
        try:
            articles = []
            
            for query in queries:
                url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_keys['newsapi']}&language=en&sortBy=publishedAt"
                
                if sources:
                    url += f"&sources={','.join(sources)}"
                    
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles.extend(data.get('articles', []))
                        
            return articles
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []
            
    async def _get_fixer_data(self, pairs: List[str]) -> Dict[str, float]:
        """Get forex data from Fixer"""
        try:
            symbols = ','.join([p.replace('/', '') for p in pairs])
            url = f"https://api.fixer.io/latest?access_key={self.api_keys['fixer']}&symbols={symbols}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('rates', {})
        except Exception as e:
            logger.error(f"Fixer error: {e}")
            return {}
            
    async def _get_aviation_data(self) -> List[AlternativeData]:
        """Get aviation data as economic indicator"""
        try:
            # Get flight statistics for major airports
            url = f"http://api.aviationstack.com/v1/flights?access_key={self.api_keys['aviationstack']}&limit=100"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    flights = data.get('data', [])
                    
                    # Calculate flight volume metrics
                    total_flights = len(flights)
                    delayed_flights = sum(1 for f in flights if f.get('flight_status') == 'delayed')
                    
                    return [AlternativeData(
                        data_type='aviation',
                        location='global',
                        value={
                            'total_flights': total_flights,
                            'delayed_flights': delayed_flights,
                            'delay_rate': delayed_flights / total_flights if total_flights > 0 else 0
                        },
                        timestamp=datetime.utcnow(),
                        relevance_score=0.4,  # Moderate relevance to economy
                        affected_sectors=['airlines', 'travel', 'hospitality']
                    )]
        except Exception as e:
            logger.error(f"Aviation API error: {e}")
            return []
            
    def _aggregate_market_data(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data from multiple sources"""
        valid_results = [r for r in results if r and not isinstance(r, Exception)]
        
        if not valid_results:
            return {}
            
        # Aggregate by taking weighted average based on source reliability
        weights = {
            'iex_cloud': 1.0,
            'finnhub': 0.9,
            'alpha_vantage': 0.8,
            'polygon': 0.85,
        }
        
        aggregated = {
            'sources': [r['source'] for r in valid_results],
            'timestamp': datetime.utcnow(),
            'consensus_price': 0,
            'consensus_change': 0,
            'raw_data': valid_results
        }
        
        # Calculate weighted averages
        total_weight = 0
        weighted_price = 0
        weighted_change = 0
        
        for result in valid_results:
            weight = weights.get(result['source'], 0.5)
            if result.get('price'):
                weighted_price += result['price'] * weight
                total_weight += weight
            if result.get('change'):
                weighted_change += result['change'] * weight
                
        if total_weight > 0:
            aggregated['consensus_price'] = weighted_price / total_weight
            aggregated['consensus_change'] = weighted_change / total_weight
            
        return aggregated
        
    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced technical indicators"""
        indicators = {}
        
        # Volatility metrics
        returns = df['close'].pct_change()
        indicators['volatility'] = returns.std() * np.sqrt(252)  # Annualized
        
        # Advanced momentum
        indicators['rsi_divergence'] = self._detect_rsi_divergence(df)
        
        # Market microstructure
        indicators['bid_ask_spread'] = self._estimate_spread(df)
        
        # Institutional activity
        indicators['large_trades'] = self._detect_large_trades(df)
        
        return indicators
        
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> bool:
        """Detect RSI divergence"""
        # Simplified implementation
        return False
        
    def _estimate_spread(self, df: pd.DataFrame) -> float:
        """Estimate bid-ask spread from price data"""
        # Roll's estimator
        price_changes = df['close'].diff()
        return 2 * np.sqrt(-np.cov(price_changes[1:], price_changes[:-1])[0, 1])
        
    def _detect_large_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional trading activity"""
        volume_mean = df['volume'].rolling(20).mean()
        large_volume = df['volume'] > volume_mean * 2
        
        return {
            'count': large_volume.sum(),
            'percentage': large_volume.mean() * 100
        }
        
    def _analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from news articles"""
        # Simplified sentiment analysis
        positive_words = ['bullish', 'growth', 'surge', 'rally', 'gain']
        negative_words = ['bearish', 'crash', 'plunge', 'loss', 'decline']
        
        total_positive = 0
        total_negative = 0
        
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            positive_count = sum(word in text for word in positive_words)
            negative_count = sum(word in text for word in negative_words)
            
            total_positive += positive_count
            total_negative += negative_count
            
        sentiment_score = (total_positive - total_negative) / max(total_positive + total_negative, 1)
        
        return {
            'score': sentiment_score,
            'positive_mentions': total_positive,
            'negative_mentions': total_negative,
            'article_count': len(articles)
        }


# Global instance
enhanced_data_aggregator = EnhancedDataAggregator()