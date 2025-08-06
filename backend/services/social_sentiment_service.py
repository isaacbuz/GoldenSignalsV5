"""
Enhanced Social Sentiment Analysis Service
Real-time social media sentiment analysis for trading signals
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from collections import defaultdict

# External APIs
import aiohttp
import praw
from newsapi import NewsApiClient
import numpy as np

# NLP Libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Internal imports
from core.config import get_settings
from core.logging import get_logger
from core.cache import get_cache_manager
from services.rag_service import rag_service

logger = get_logger(__name__)
settings = get_settings()

# Download NLTK data if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SentimentSource(Enum):
    """Available sentiment data sources"""
    REDDIT = "reddit"
    TWITTER = "twitter"
    NEWS = "news"
    STOCKTWITS = "stocktwits"
    DISCORD = "discord"
    TELEGRAM = "telegram"


class SentimentLevel(Enum):
    """Sentiment classification levels"""
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2


@dataclass
class SentimentData:
    """Individual sentiment data point"""
    source: SentimentSource
    text: str
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: datetime
    author: Optional[str] = None
    engagement: int = 0  # likes, upvotes, etc.
    reach: int = 0  # followers, subscribers
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted sentiment score based on engagement and reach"""
        weight = 1.0
        if self.engagement > 0:
            weight += min(self.engagement / 1000, 2.0)  # Cap at 3x
        if self.reach > 0:
            weight += min(self.reach / 10000, 1.0)  # Cap at 2x
        return self.score * weight * self.confidence


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment analysis results"""
    symbol: str
    overall_score: float  # -1 to 1
    overall_level: SentimentLevel
    confidence: float
    volume: int  # Number of mentions
    sources_breakdown: Dict[SentimentSource, Dict[str, Any]]
    trending_topics: List[str]
    key_influencers: List[Dict[str, Any]]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "symbol": self.symbol,
            "overall_score": self.overall_score,
            "overall_level": self.overall_level.name,
            "confidence": self.confidence,
            "volume": self.volume,
            "sources_breakdown": {
                source.value: data 
                for source, data in self.sources_breakdown.items()
            },
            "trending_topics": self.trending_topics,
            "key_influencers": self.key_influencers,
            "timestamp": self.timestamp.isoformat()
        }


class SocialSentimentService:
    """
    Comprehensive social sentiment analysis service
    Aggregates sentiment from multiple social media sources
    """
    
    def __init__(self):
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # API clients
        self.reddit_client = None
        self.news_client = None
        
        # Initialize API clients
        self._initialize_clients()
        
        # Cache manager
        self.cache_manager = get_cache_manager()
        
        # Sentiment history for trend analysis
        self.sentiment_history = defaultdict(list)
        
        # Rate limiting
        self.rate_limits = {
            SentimentSource.REDDIT: {"calls": 60, "period": 60},
            SentimentSource.NEWS: {"calls": 100, "period": 86400},
            SentimentSource.STOCKTWITS: {"calls": 200, "period": 3600}
        }
        
        self.last_calls = {source: [] for source in SentimentSource}
        
        # Metrics
        self.metrics = {
            "analyses_performed": 0,
            "sources_queried": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        logger.info("Social Sentiment Service initialized")
    
    def _initialize_clients(self):
        """Initialize API clients for social media sources"""
        # Reddit
        if all([settings.REDDIT_CLIENT_ID, settings.REDDIT_CLIENT_SECRET]):
            try:
                self.reddit_client = praw.Reddit(
                    client_id=settings.REDDIT_CLIENT_ID,
                    client_secret=settings.REDDIT_CLIENT_SECRET,
                    user_agent="GoldenSignalsAI/1.0"
                )
                logger.info("Reddit client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {e}")
        
        # News API
        if settings.NEWS_API_KEY:
            try:
                self.news_client = NewsApiClient(api_key=settings.NEWS_API_KEY)
                logger.info("News API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize News API client: {e}")
    
    async def analyze_symbol(
        self,
        symbol: str,
        sources: Optional[List[SentimentSource]] = None,
        lookback_hours: int = 24
    ) -> AggregatedSentiment:
        """
        Analyze sentiment for a symbol across multiple sources
        
        Args:
            symbol: Stock symbol to analyze
            sources: List of sources to query (None = all available)
            lookback_hours: Hours of historical data to analyze
            
        Returns:
            AggregatedSentiment object
        """
        start_time = datetime.now()
        
        # Check cache
        cache_key = f"sentiment:{symbol}:{lookback_hours}"
        cached = await self.cache_manager.get(cache_key)
        if cached:
            self.metrics["cache_hits"] += 1
            return AggregatedSentiment(**cached)
        
        try:
            # Determine sources to query
            if sources is None:
                sources = self._get_available_sources()
            
            # Collect sentiment data from all sources
            all_sentiment_data = []
            sources_breakdown = {}
            
            # Create tasks for parallel execution
            tasks = []
            for source in sources:
                if source == SentimentSource.REDDIT and self.reddit_client:
                    tasks.append(self._analyze_reddit(symbol, lookback_hours))
                elif source == SentimentSource.NEWS and self.news_client:
                    tasks.append(self._analyze_news(symbol, lookback_hours))
                elif source == SentimentSource.STOCKTWITS:
                    tasks.append(self._analyze_stocktwits(symbol, lookback_hours))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for source, result in zip(sources, results):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing {source.value}: {result}")
                    self.metrics["errors"] += 1
                else:
                    sentiment_data, breakdown = result
                    all_sentiment_data.extend(sentiment_data)
                    sources_breakdown[source] = breakdown
            
            # Aggregate sentiment
            aggregated = self._aggregate_sentiment(
                symbol,
                all_sentiment_data,
                sources_breakdown
            )
            
            # Store in RAG for future reference
            await self._store_in_rag(aggregated)
            
            # Cache result
            await self.cache_manager.set(
                cache_key,
                aggregated.to_dict(),
                ttl=300  # 5 minutes
            )
            
            # Update metrics
            self.metrics["analyses_performed"] += 1
            self.metrics["sources_queried"] += len(sources)
            
            # Store in history
            self.sentiment_history[symbol].append(aggregated)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            self.metrics["errors"] += 1
            
            # Return neutral sentiment on error
            return AggregatedSentiment(
                symbol=symbol,
                overall_score=0.0,
                overall_level=SentimentLevel.NEUTRAL,
                confidence=0.0,
                volume=0,
                sources_breakdown={},
                trending_topics=[],
                key_influencers=[],
                timestamp=datetime.now()
            )
    
    async def _analyze_reddit(
        self,
        symbol: str,
        lookback_hours: int
    ) -> Tuple[List[SentimentData], Dict[str, Any]]:
        """Analyze Reddit sentiment"""
        sentiment_data = []
        
        try:
            # Search relevant subreddits
            subreddits = ["wallstreetbets", "stocks", "investing", "StockMarket"]
            
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            total_posts = 0
            total_comments = 0
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Search for symbol mentions
                for submission in subreddit.search(
                    f"${symbol} OR {symbol}",
                    time_filter="day",
                    limit=25
                ):
                    if datetime.fromtimestamp(submission.created_utc) < cutoff_time:
                        continue
                    
                    total_posts += 1
                    
                    # Analyze post title and body
                    text = f"{submission.title} {submission.selftext}"
                    sentiment = self._analyze_text(text)
                    
                    sentiment_data.append(SentimentData(
                        source=SentimentSource.REDDIT,
                        text=text[:500],  # Truncate for storage
                        score=sentiment["score"],
                        confidence=sentiment["confidence"],
                        timestamp=datetime.fromtimestamp(submission.created_utc),
                        author=str(submission.author) if submission.author else None,
                        engagement=submission.score,
                        reach=subreddit.subscribers,
                        metadata={
                            "subreddit": subreddit_name,
                            "type": "post",
                            "url": submission.url
                        }
                    ))
                    
                    # Analyze top comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:10]:
                        total_comments += 1
                        
                        sentiment = self._analyze_text(comment.body)
                        
                        sentiment_data.append(SentimentData(
                            source=SentimentSource.REDDIT,
                            text=comment.body[:500],
                            score=sentiment["score"],
                            confidence=sentiment["confidence"],
                            timestamp=datetime.fromtimestamp(comment.created_utc),
                            author=str(comment.author) if comment.author else None,
                            engagement=comment.score,
                            reach=subreddit.subscribers,
                            metadata={
                                "subreddit": subreddit_name,
                                "type": "comment"
                            }
                        ))
            
            # Calculate breakdown
            breakdown = {
                "posts_analyzed": total_posts,
                "comments_analyzed": total_comments,
                "average_score": sum(s.score for s in sentiment_data) / len(sentiment_data) if sentiment_data else 0,
                "top_subreddits": subreddits
            }
            
            return sentiment_data, breakdown
            
        except Exception as e:
            logger.error(f"Reddit analysis failed: {e}")
            return [], {}
    
    async def _analyze_news(
        self,
        symbol: str,
        lookback_hours: int
    ) -> Tuple[List[SentimentData], Dict[str, Any]]:
        """Analyze news sentiment"""
        sentiment_data = []
        
        try:
            # Get news articles
            from_date = (datetime.now() - timedelta(hours=lookback_hours)).date()
            
            # Search for company news
            articles = self.news_client.get_everything(
                q=symbol,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=50
            )
            
            for article in articles.get('articles', []):
                # Combine title and description for sentiment
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                sentiment = self._analyze_text(text)
                
                sentiment_data.append(SentimentData(
                    source=SentimentSource.NEWS,
                    text=text[:500],
                    score=sentiment["score"],
                    confidence=sentiment["confidence"],
                    timestamp=datetime.fromisoformat(
                        article['publishedAt'].replace('Z', '+00:00')
                    ),
                    author=article.get('author'),
                    engagement=0,  # News doesn't have direct engagement
                    reach=0,  # Could estimate based on source
                    metadata={
                        "source": article.get('source', {}).get('name'),
                        "url": article.get('url')
                    }
                ))
            
            # Calculate breakdown
            breakdown = {
                "articles_analyzed": len(sentiment_data),
                "average_score": sum(s.score for s in sentiment_data) / len(sentiment_data) if sentiment_data else 0,
                "sources": list(set(
                    s.metadata.get("source") for s in sentiment_data
                    if s.metadata.get("source")
                ))
            }
            
            return sentiment_data, breakdown
            
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return [], {}
    
    async def _analyze_stocktwits(
        self,
        symbol: str,
        lookback_hours: int
    ) -> Tuple[List[SentimentData], Dict[str, Any]]:
        """Analyze StockTwits sentiment"""
        sentiment_data = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
                        
                        for message in data.get('messages', []):
                            # Parse timestamp
                            timestamp = datetime.fromisoformat(
                                message['created_at'].replace('Z', '+00:00')
                            )
                            
                            if timestamp < cutoff_time:
                                continue
                            
                            # Analyze message body
                            text = message.get('body', '')
                            sentiment = self._analyze_text(text)
                            
                            # StockTwits provides its own sentiment
                            st_sentiment = message.get('entities', {}).get('sentiment', {})
                            if st_sentiment:
                                # Adjust score based on StockTwits sentiment
                                if st_sentiment.get('basic') == 'Bullish':
                                    sentiment["score"] = max(sentiment["score"], 0.5)
                                elif st_sentiment.get('basic') == 'Bearish':
                                    sentiment["score"] = min(sentiment["score"], -0.5)
                            
                            sentiment_data.append(SentimentData(
                                source=SentimentSource.STOCKTWITS,
                                text=text[:500],
                                score=sentiment["score"],
                                confidence=sentiment["confidence"],
                                timestamp=timestamp,
                                author=message.get('user', {}).get('username'),
                                engagement=message.get('likes', {}).get('total', 0),
                                reach=message.get('user', {}).get('followers', 0),
                                metadata={
                                    "message_id": message.get('id'),
                                    "stocktwits_sentiment": st_sentiment.get('basic')
                                }
                            ))
            
            # Calculate breakdown
            breakdown = {
                "messages_analyzed": len(sentiment_data),
                "average_score": sum(s.score for s in sentiment_data) / len(sentiment_data) if sentiment_data else 0,
                "bullish_count": sum(1 for s in sentiment_data if s.score > 0.2),
                "bearish_count": sum(1 for s in sentiment_data if s.score < -0.2)
            }
            
            return sentiment_data, breakdown
            
        except Exception as e:
            logger.error(f"StockTwits analysis failed: {e}")
            return [], {}
    
    def _analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with score (-1 to 1) and confidence (0 to 1)
        """
        try:
            # Clean text
            text = self._clean_text(text)
            
            if not text:
                return {"score": 0.0, "confidence": 0.0}
            
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            vader_sentiment = vader_scores['compound']
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Combine sentiments (weighted average)
            combined_score = (vader_sentiment * 0.6 + textblob_sentiment * 0.4)
            
            # Calculate confidence based on agreement
            agreement = 1 - abs(vader_sentiment - textblob_sentiment)
            confidence = min(agreement * 1.5, 1.0)  # Scale up but cap at 1
            
            # Adjust for financial keywords
            score_adjustment = self._financial_keyword_adjustment(text)
            final_score = max(-1, min(1, combined_score + score_adjustment))
            
            return {
                "score": final_score,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {"score": 0.0, "confidence": 0.0}
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove stock symbols (keep for context but not sentiment)
        text = re.sub(r'\$[A-Z]+', '', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s\.\!\?\,\-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _financial_keyword_adjustment(self, text: str) -> float:
        """Adjust sentiment based on financial keywords"""
        text_lower = text.lower()
        
        bullish_keywords = [
            'moon', 'rocket', 'squeeze', 'breakout', 'bullish', 'calls',
            'buy', 'long', 'diamond hands', 'hold', 'accumulate',
            'oversold', 'undervalued', 'upgrade', 'beat', 'strong'
        ]
        
        bearish_keywords = [
            'crash', 'dump', 'puts', 'bearish', 'sell', 'short',
            'overvalued', 'overbought', 'downgrade', 'miss', 'weak',
            'bankruptcy', 'delisted', 'fraud', 'investigation'
        ]
        
        adjustment = 0.0
        
        for keyword in bullish_keywords:
            if keyword in text_lower:
                adjustment += 0.1
        
        for keyword in bearish_keywords:
            if keyword in text_lower:
                adjustment -= 0.1
        
        return max(-0.3, min(0.3, adjustment))  # Cap adjustment
    
    def _aggregate_sentiment(
        self,
        symbol: str,
        sentiment_data: List[SentimentData],
        sources_breakdown: Dict[SentimentSource, Dict[str, Any]]
    ) -> AggregatedSentiment:
        """Aggregate sentiment data into final analysis"""
        if not sentiment_data:
            return AggregatedSentiment(
                symbol=symbol,
                overall_score=0.0,
                overall_level=SentimentLevel.NEUTRAL,
                confidence=0.0,
                volume=0,
                sources_breakdown=sources_breakdown,
                trending_topics=[],
                key_influencers=[],
                timestamp=datetime.now()
            )
        
        # Calculate weighted average sentiment
        total_weight = 0
        weighted_sum = 0
        
        for data in sentiment_data:
            weight = data.confidence * (1 + min(data.engagement / 100, 5))
            weighted_sum += data.score * weight
            total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine sentiment level
        if overall_score <= -0.6:
            level = SentimentLevel.VERY_BEARISH
        elif overall_score <= -0.2:
            level = SentimentLevel.BEARISH
        elif overall_score <= 0.2:
            level = SentimentLevel.NEUTRAL
        elif overall_score <= 0.6:
            level = SentimentLevel.BULLISH
        else:
            level = SentimentLevel.VERY_BULLISH
        
        # Calculate confidence
        confidence = sum(d.confidence for d in sentiment_data) / len(sentiment_data)
        
        # Extract trending topics (simplified)
        trending_topics = self._extract_trending_topics(sentiment_data)
        
        # Identify key influencers
        key_influencers = self._identify_influencers(sentiment_data)
        
        return AggregatedSentiment(
            symbol=symbol,
            overall_score=overall_score,
            overall_level=level,
            confidence=confidence,
            volume=len(sentiment_data),
            sources_breakdown=sources_breakdown,
            trending_topics=trending_topics,
            key_influencers=key_influencers,
            timestamp=datetime.now()
        )
    
    def _extract_trending_topics(self, sentiment_data: List[SentimentData]) -> List[str]:
        """Extract trending topics from sentiment data"""
        # Simple keyword extraction (could be enhanced with NLP)
        topics = defaultdict(int)
        
        keywords = [
            'earnings', 'revenue', 'guidance', 'merger', 'acquisition',
            'FDA', 'approval', 'lawsuit', 'SEC', 'investigation',
            'product', 'launch', 'partnership', 'dividend', 'buyback'
        ]
        
        for data in sentiment_data:
            text_lower = data.text.lower()
            for keyword in keywords:
                if keyword in text_lower:
                    topics[keyword] += 1
        
        # Return top 5 topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:5]]
    
    def _identify_influencers(self, sentiment_data: List[SentimentData]) -> List[Dict[str, Any]]:
        """Identify key influencers from sentiment data"""
        influencer_scores = defaultdict(lambda: {"score": 0, "posts": 0, "reach": 0})
        
        for data in sentiment_data:
            if data.author:
                influencer_scores[data.author]["score"] += abs(data.weighted_score)
                influencer_scores[data.author]["posts"] += 1
                influencer_scores[data.author]["reach"] = max(
                    influencer_scores[data.author]["reach"],
                    data.reach
                )
        
        # Calculate influence score
        influencers = []
        for author, stats in influencer_scores.items():
            influence = stats["score"] * (1 + min(stats["reach"] / 10000, 5))
            influencers.append({
                "author": author,
                "influence_score": influence,
                "posts": stats["posts"],
                "reach": stats["reach"]
            })
        
        # Return top 10 influencers
        influencers.sort(key=lambda x: x["influence_score"], reverse=True)
        return influencers[:10]
    
    async def _store_in_rag(self, sentiment: AggregatedSentiment):
        """Store sentiment analysis in RAG for future reference"""
        try:
            document = {
                "id": f"sentiment_{sentiment.symbol}_{sentiment.timestamp.timestamp()}",
                "content": f"""Social Sentiment Analysis for {sentiment.symbol}:
                Overall Score: {sentiment.overall_score:.2f}
                Sentiment Level: {sentiment.overall_level.name}
                Confidence: {sentiment.confidence:.2f}
                Volume: {sentiment.volume} mentions
                Trending Topics: {', '.join(sentiment.trending_topics)}
                Top Influencers: {', '.join([i['author'] for i in sentiment.key_influencers[:3]])}
                """,
                "symbol": sentiment.symbol,
                "document_type": "sentiment_analysis",
                "source": "social_media",
                "metadata": sentiment.to_dict()
            }
            
            await rag_service.ingest_documents([document])
            
        except Exception as e:
            logger.error(f"Failed to store sentiment in RAG: {e}")
    
    def _get_available_sources(self) -> List[SentimentSource]:
        """Get list of available sentiment sources"""
        sources = []
        
        if self.reddit_client:
            sources.append(SentimentSource.REDDIT)
        
        if self.news_client:
            sources.append(SentimentSource.NEWS)
        
        # StockTwits is always available (public API)
        sources.append(SentimentSource.STOCKTWITS)
        
        return sources
    
    async def get_sentiment_trend(
        self,
        symbol: str,
        period_hours: int = 168  # 1 week
    ) -> Dict[str, Any]:
        """Get sentiment trend over time"""
        try:
            # Get historical sentiment from cache/history
            history = self.sentiment_history.get(symbol, [])
            
            cutoff = datetime.now() - timedelta(hours=period_hours)
            recent_history = [
                s for s in history
                if s.timestamp >= cutoff
            ]
            
            if not recent_history:
                # No historical data, return current sentiment
                current = await self.analyze_symbol(symbol)
                return {
                    "symbol": symbol,
                    "current": current.to_dict(),
                    "trend": "unknown",
                    "history": []
                }
            
            # Calculate trend
            scores = [s.overall_score for s in recent_history]
            
            # Simple linear regression for trend
            if len(scores) >= 2:
                x = list(range(len(scores)))
                slope = np.polyfit(x, scores, 1)[0]
                
                if slope > 0.01:
                    trend = "improving"
                elif slope < -0.01:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            return {
                "symbol": symbol,
                "current": recent_history[-1].to_dict(),
                "trend": trend,
                "history": [s.to_dict() for s in recent_history],
                "slope": slope if len(scores) >= 2 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get sentiment trend: {e}")
            return {
                "symbol": symbol,
                "trend": "error",
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            **self.metrics,
            "available_sources": [s.value for s in self._get_available_sources()],
            "history_symbols": list(self.sentiment_history.keys()),
            "total_data_points": sum(
                len(history) for history in self.sentiment_history.values()
            )
        }


# Singleton instance
social_sentiment_service = SocialSentimentService()