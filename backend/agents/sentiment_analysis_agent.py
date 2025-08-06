"""
Sentiment Analysis Agent V5
Comprehensive multi-source sentiment analysis combining news, social media, and market sentiment
Enhanced from archive with V5 architecture patterns
"""

import asyncio
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import uuid
import re

import numpy as np
import pandas as pd
from textblob import TextBlob

from core.logging import get_logger
from services.social_sentiment_analyzer import social_sentiment_analyzer, SentimentScore

logger = get_logger(__name__)


class SentimentSignal(Enum):
    """Sentiment-based trading signals"""
    EXTREME_BULLISH = "extreme_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    EXTREME_BEARISH = "extreme_bearish"


class SentimentSource(Enum):
    """Sentiment data sources"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    REDDIT = "reddit"
    TWITTER = "twitter"
    STOCKTWITS = "stocktwits"
    ANALYST = "analyst"
    OPTIONS_FLOW = "options_flow"
    INSIDER = "insider"


@dataclass
class SourceSentiment:
    """Sentiment from a specific source"""
    source: SentimentSource
    sentiment_score: float  # -1 to 1
    confidence: float
    volume: int  # Number of mentions/posts
    velocity: float  # Rate of change
    top_keywords: List[str]
    sample_texts: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source.value,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'volume': self.volume,
            'velocity': self.velocity,
            'top_keywords': self.top_keywords,
            'sample_texts': self.sample_texts[:3],  # Limit samples
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SentimentMetrics:
    """Aggregated sentiment metrics"""
    overall_sentiment: float
    sentiment_momentum: float  # Rate of sentiment change
    sentiment_dispersion: float  # Agreement across sources
    bullish_percentage: float
    bearish_percentage: float
    volume_weighted_sentiment: float
    smart_money_sentiment: float  # From options/insider
    retail_sentiment: float  # From social media
    fear_greed_index: float  # 0-100 scale
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentimentAnalysis:
    """Complete sentiment analysis result"""
    symbol: str
    sources: List[SourceSentiment]
    metrics: SentimentMetrics
    signal: SentimentSignal
    signal_strength: float
    trending_topics: List[str]
    sentiment_drivers: List[str]  # Key factors driving sentiment
    contrarian_indicators: List[str]  # Extreme sentiment warnings
    recommendations: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'sources': [s.to_dict() for s in self.sources],
            'metrics': self.metrics.to_dict(),
            'signal': self.signal.value,
            'signal_strength': self.signal_strength,
            'trending_topics': self.trending_topics,
            'sentiment_drivers': self.sentiment_drivers,
            'contrarian_indicators': self.contrarian_indicators,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class SentimentAnalysisAgent:
    """
    V5 Sentiment Analysis Agent
    Multi-source sentiment aggregation and analysis
    """
    
    def __init__(self):
        """Initialize the sentiment analysis agent"""
        # Configuration
        self.lookback_hours = 24
        self.min_volume_threshold = 10  # Minimum mentions for relevance
        self.sentiment_decay_hours = 6  # How fast sentiment impact decays
        
        # Source weights for aggregation
        self.source_weights = {
            SentimentSource.NEWS: 0.25,
            SentimentSource.SOCIAL_MEDIA: 0.15,
            SentimentSource.REDDIT: 0.15,
            SentimentSource.TWITTER: 0.15,
            SentimentSource.STOCKTWITS: 0.10,
            SentimentSource.ANALYST: 0.10,
            SentimentSource.OPTIONS_FLOW: 0.05,
            SentimentSource.INSIDER: 0.05
        }
        
        # Sentiment thresholds
        self.extreme_bullish_threshold = 0.7
        self.bullish_threshold = 0.3
        self.bearish_threshold = -0.3
        self.extreme_bearish_threshold = -0.7
        
        # Contrarian thresholds
        self.extreme_sentiment_threshold = 0.8
        self.high_volume_threshold = 1000
        
        # Cache
        self.sentiment_cache = {}
        self.analysis_history = deque(maxlen=100)
        
        # Performance tracking
        self.api_call_times = {}
        
        # Initialize FinBERT for advanced NLP (optional)
        self.finbert_model = None
        self._init_finbert()
        
        logger.info("Sentiment Analysis Agent V5 initialized")
    
    def _init_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        try:
            from transformers import pipeline
            self.finbert_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT model not available: {str(e)}")
            self.finbert_model = None
    
    async def analyze(self, symbol: str, market_data: Optional[Dict[str, Any]] = None) -> SentimentAnalysis:
        """
        Perform comprehensive sentiment analysis
        
        Args:
            symbol: Trading symbol
            market_data: Optional market data for context
        
        Returns:
            Complete sentiment analysis
        """
        try:
            start_time = datetime.now()
            
            # Gather sentiment from all sources
            source_sentiments = await self._gather_all_sentiments(symbol, market_data)
            
            # Calculate aggregated metrics
            metrics = self._calculate_sentiment_metrics(source_sentiments)
            
            # Identify trending topics and drivers
            trending_topics = self._extract_trending_topics(source_sentiments)
            sentiment_drivers = self._identify_sentiment_drivers(source_sentiments, metrics)
            
            # Check for contrarian indicators
            contrarian_indicators = self._check_contrarian_indicators(metrics, source_sentiments)
            
            # Generate trading signal
            signal, signal_strength = self._generate_signal(metrics, contrarian_indicators)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                signal, metrics, sentiment_drivers, contrarian_indicators
            )
            
            # Create analysis result
            analysis = SentimentAnalysis(
                symbol=symbol,
                sources=source_sentiments,
                metrics=metrics,
                signal=signal,
                signal_strength=signal_strength,
                trending_topics=trending_topics,
                sentiment_drivers=sentiment_drivers,
                contrarian_indicators=contrarian_indicators,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.analysis_history.append(analysis)
            
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds()
            if 'sentiment_analysis' not in self.api_call_times:
                self.api_call_times['sentiment_analysis'] = []
            self.api_call_times['sentiment_analysis'].append(calc_time)
            
            logger.info(f"Sentiment analysis for {symbol}: {signal.value} (score: {metrics.overall_sentiment:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {str(e)}")
            raise
    
    async def _gather_all_sentiments(self, symbol: str, market_data: Optional[Dict[str, Any]]) -> List[SourceSentiment]:
        """Gather sentiment from all available sources"""
        source_sentiments = []
        
        # Gather sentiments concurrently
        tasks = [
            self._get_news_sentiment(symbol),
            self._get_social_media_sentiment(symbol),
            self._get_reddit_sentiment(symbol),
            self._get_twitter_sentiment(symbol),
            self._get_stocktwits_sentiment(symbol),
            self._get_analyst_sentiment(symbol),
            self._get_options_sentiment(symbol, market_data),
            self._get_insider_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Source sentiment gathering failed: {str(result)}")
            elif result:
                source_sentiments.append(result)
        
        return source_sentiments
    
    async def _get_news_sentiment(self, symbol: str) -> Optional[SourceSentiment]:
        """Get sentiment from news sources"""
        try:
            # Simulate news sentiment (would integrate with news API)
            sample_headlines = [
                f"{symbol} beats earnings expectations",
                f"Analysts upgrade {symbol} to buy",
                f"{symbol} announces new product launch"
            ]
            
            sentiments = []
            for headline in sample_headlines:
                if self.finbert_model:
                    result = self.finbert_model(headline)[0]
                    score = 1.0 if result['label'] == 'positive' else -1.0 if result['label'] == 'negative' else 0.0
                else:
                    blob = TextBlob(headline)
                    score = blob.sentiment.polarity
                sentiments.append(score)
            
            avg_sentiment = np.mean(sentiments)
            
            return SourceSentiment(
                source=SentimentSource.NEWS,
                sentiment_score=avg_sentiment,
                confidence=0.7,
                volume=len(sample_headlines),
                velocity=0.1,  # Placeholder
                top_keywords=[symbol, "earnings", "upgrade"],
                sample_texts=sample_headlines[:3],
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"News sentiment failed: {str(e)}")
            return None
    
    async def _get_social_media_sentiment(self, symbol: str) -> Optional[SourceSentiment]:
        """Get aggregated social media sentiment"""
        try:
            # Use the social sentiment analyzer service
            sentiment_data = await social_sentiment_analyzer.analyze_symbol_sentiment(
                symbol=symbol,
                time_window_hours=self.lookback_hours
            )
            
            if not sentiment_data:
                return None
            
            return SourceSentiment(
                source=SentimentSource.SOCIAL_MEDIA,
                sentiment_score=sentiment_data.get('weighted_sentiment', 0),
                confidence=sentiment_data.get('confidence', 0.5),
                volume=sentiment_data.get('total_posts_analyzed', 0),
                velocity=sentiment_data.get('velocity', 0),
                top_keywords=sentiment_data.get('trending_keywords', [])[:5],
                sample_texts=sentiment_data.get('sample_posts', [])[:3],
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Social media sentiment failed: {str(e)}")
            return None
    
    async def _get_reddit_sentiment(self, symbol: str) -> Optional[SourceSentiment]:
        """Get Reddit sentiment (WSB, investing subreddits)"""
        try:
            # Placeholder for Reddit API integration
            # Would search r/wallstreetbets, r/stocks, r/investing
            
            sample_posts = [
                f"${symbol} to the moon! 🚀",
                f"DD on {symbol}: Strong fundamentals",
                f"Buying the dip on {symbol}"
            ]
            
            sentiments = []
            for post in sample_posts:
                blob = TextBlob(post)
                sentiments.append(blob.sentiment.polarity)
            
            return SourceSentiment(
                source=SentimentSource.REDDIT,
                sentiment_score=np.mean(sentiments),
                confidence=0.6,
                volume=len(sample_posts) * 10,  # Simulated volume
                velocity=0.2,
                top_keywords=[symbol, "moon", "DD", "fundamentals"],
                sample_texts=sample_posts,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Reddit sentiment failed: {str(e)}")
            return None
    
    async def _get_twitter_sentiment(self, symbol: str) -> Optional[SourceSentiment]:
        """Get Twitter/X sentiment"""
        try:
            # Placeholder for Twitter API integration
            sample_tweets = [
                f"${symbol} looking strong today",
                f"Bullish on ${symbol} long term",
                f"${symbol} chart setting up nicely"
            ]
            
            sentiments = []
            for tweet in sample_tweets:
                blob = TextBlob(tweet)
                sentiments.append(blob.sentiment.polarity)
            
            return SourceSentiment(
                source=SentimentSource.TWITTER,
                sentiment_score=np.mean(sentiments),
                confidence=0.65,
                volume=len(sample_tweets) * 20,
                velocity=0.3,
                top_keywords=[f"${symbol}", "bullish", "strong"],
                sample_texts=sample_tweets,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Twitter sentiment failed: {str(e)}")
            return None
    
    async def _get_stocktwits_sentiment(self, symbol: str) -> Optional[SourceSentiment]:
        """Get StockTwits sentiment"""
        try:
            # Placeholder for StockTwits API integration
            sample_messages = [
                f"${symbol} breakout incoming",
                f"Loading up on ${symbol} here",
                f"${symbol} momentum play"
            ]
            
            sentiments = []
            for message in sample_messages:
                blob = TextBlob(message)
                sentiments.append(blob.sentiment.polarity)
            
            return SourceSentiment(
                source=SentimentSource.STOCKTWITS,
                sentiment_score=np.mean(sentiments),
                confidence=0.6,
                volume=len(sample_messages) * 15,
                velocity=0.25,
                top_keywords=[f"${symbol}", "breakout", "momentum"],
                sample_texts=sample_messages,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"StockTwits sentiment failed: {str(e)}")
            return None
    
    async def _get_analyst_sentiment(self, symbol: str) -> Optional[SourceSentiment]:
        """Get analyst ratings sentiment"""
        try:
            # Placeholder for analyst data
            # Would integrate with financial data providers
            
            ratings = {
                'buy': 5,
                'hold': 2,
                'sell': 1
            }
            
            # Calculate sentiment from ratings distribution
            total_ratings = sum(ratings.values())
            weighted_score = (ratings['buy'] * 1.0 + ratings['hold'] * 0.0 + ratings['sell'] * -1.0) / total_ratings
            
            return SourceSentiment(
                source=SentimentSource.ANALYST,
                sentiment_score=weighted_score,
                confidence=0.8,
                volume=total_ratings,
                velocity=0.0,  # Analysts change slowly
                top_keywords=["buy", "upgrade", "target"],
                sample_texts=[f"{ratings['buy']} buy, {ratings['hold']} hold, {ratings['sell']} sell ratings"],
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Analyst sentiment failed: {str(e)}")
            return None
    
    async def _get_options_sentiment(self, symbol: str, market_data: Optional[Dict[str, Any]]) -> Optional[SourceSentiment]:
        """Get sentiment from options flow"""
        try:
            # Placeholder for options data
            # Would integrate with options data provider
            
            put_call_ratio = 0.8  # Example: more calls than puts
            
            # Convert P/C ratio to sentiment
            # Lower P/C ratio = bullish, Higher = bearish
            sentiment = (1.0 - put_call_ratio) / 0.5  # Normalize around 1.0
            sentiment = max(-1, min(1, sentiment))  # Clamp to [-1, 1]
            
            return SourceSentiment(
                source=SentimentSource.OPTIONS_FLOW,
                sentiment_score=sentiment,
                confidence=0.75,
                volume=1000,  # Example volume
                velocity=0.1,
                top_keywords=["calls", "puts", "flow"],
                sample_texts=[f"Put/Call ratio: {put_call_ratio:.2f}"],
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Options sentiment failed: {str(e)}")
            return None
    
    async def _get_insider_sentiment(self, symbol: str) -> Optional[SourceSentiment]:
        """Get insider trading sentiment"""
        try:
            # Placeholder for insider data
            # Would integrate with SEC filings API
            
            # Example: more buys than sells
            insider_buys = 3
            insider_sells = 1
            
            if insider_buys + insider_sells > 0:
                sentiment = (insider_buys - insider_sells) / (insider_buys + insider_sells)
            else:
                sentiment = 0.0
            
            return SourceSentiment(
                source=SentimentSource.INSIDER,
                sentiment_score=sentiment,
                confidence=0.9,  # High confidence in insider signals
                volume=insider_buys + insider_sells,
                velocity=0.0,
                top_keywords=["insider", "buying", "SEC"],
                sample_texts=[f"{insider_buys} insider buys, {insider_sells} sells"],
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Insider sentiment failed: {str(e)}")
            return None
    
    def _calculate_sentiment_metrics(self, sources: List[SourceSentiment]) -> SentimentMetrics:
        """Calculate aggregated sentiment metrics"""
        
        if not sources:
            return SentimentMetrics(
                overall_sentiment=0.0,
                sentiment_momentum=0.0,
                sentiment_dispersion=0.0,
                bullish_percentage=0.0,
                bearish_percentage=0.0,
                volume_weighted_sentiment=0.0,
                smart_money_sentiment=0.0,
                retail_sentiment=0.0,
                fear_greed_index=50.0
            )
        
        # Calculate weighted overall sentiment
        weighted_sum = 0.0
        weight_total = 0.0
        
        for source in sources:
            weight = self.source_weights.get(source.source, 0.1) * source.confidence
            weighted_sum += source.sentiment_score * weight
            weight_total += weight
        
        overall_sentiment = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        # Calculate sentiment momentum (velocity-weighted)
        momentum = np.mean([s.velocity * s.sentiment_score for s in sources])
        
        # Calculate dispersion (agreement across sources)
        sentiments = [s.sentiment_score for s in sources]
        dispersion = 1.0 - np.std(sentiments) if len(sentiments) > 1 else 1.0
        
        # Calculate bullish/bearish percentages
        bullish_count = sum(1 for s in sources if s.sentiment_score > 0.1)
        bearish_count = sum(1 for s in sources if s.sentiment_score < -0.1)
        total_count = len(sources)
        
        bullish_pct = bullish_count / total_count if total_count > 0 else 0.0
        bearish_pct = bearish_count / total_count if total_count > 0 else 0.0
        
        # Volume-weighted sentiment
        total_volume = sum(s.volume for s in sources)
        if total_volume > 0:
            volume_weighted = sum(s.sentiment_score * s.volume for s in sources) / total_volume
        else:
            volume_weighted = overall_sentiment
        
        # Smart money sentiment (options, insider, analyst)
        smart_sources = [s for s in sources if s.source in [
            SentimentSource.OPTIONS_FLOW, SentimentSource.INSIDER, SentimentSource.ANALYST
        ]]
        smart_money = np.mean([s.sentiment_score for s in smart_sources]) if smart_sources else overall_sentiment
        
        # Retail sentiment (social media)
        retail_sources = [s for s in sources if s.source in [
            SentimentSource.REDDIT, SentimentSource.TWITTER, SentimentSource.STOCKTWITS, SentimentSource.SOCIAL_MEDIA
        ]]
        retail = np.mean([s.sentiment_score for s in retail_sources]) if retail_sources else overall_sentiment
        
        # Fear/Greed Index (0-100 scale)
        fear_greed = 50 + (overall_sentiment * 50)  # Convert -1,1 to 0,100
        fear_greed = max(0, min(100, fear_greed))
        
        return SentimentMetrics(
            overall_sentiment=overall_sentiment,
            sentiment_momentum=momentum,
            sentiment_dispersion=dispersion,
            bullish_percentage=bullish_pct,
            bearish_percentage=bearish_pct,
            volume_weighted_sentiment=volume_weighted,
            smart_money_sentiment=smart_money,
            retail_sentiment=retail,
            fear_greed_index=fear_greed
        )
    
    def _extract_trending_topics(self, sources: List[SourceSentiment]) -> List[str]:
        """Extract trending topics from all sources"""
        all_keywords = []
        
        for source in sources:
            all_keywords.extend(source.top_keywords)
        
        # Count frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword = keyword.lower()
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top trending topics
        return [k for k, _ in sorted_keywords[:10]]
    
    def _identify_sentiment_drivers(self, sources: List[SourceSentiment], metrics: SentimentMetrics) -> List[str]:
        """Identify key factors driving sentiment"""
        drivers = []
        
        # Check for strong source signals
        for source in sources:
            if abs(source.sentiment_score) > 0.5 and source.confidence > 0.7:
                if source.sentiment_score > 0:
                    drivers.append(f"Strong bullish {source.source.value} sentiment")
                else:
                    drivers.append(f"Strong bearish {source.source.value} sentiment")
        
        # Check for momentum
        if abs(metrics.sentiment_momentum) > 0.2:
            direction = "increasing" if metrics.sentiment_momentum > 0 else "decreasing"
            drivers.append(f"Sentiment momentum {direction}")
        
        # Check for smart money divergence
        if abs(metrics.smart_money_sentiment - metrics.retail_sentiment) > 0.3:
            if metrics.smart_money_sentiment > metrics.retail_sentiment:
                drivers.append("Smart money more bullish than retail")
            else:
                drivers.append("Retail more bullish than smart money")
        
        # Check for high agreement
        if metrics.sentiment_dispersion > 0.8:
            drivers.append("High consensus across sources")
        elif metrics.sentiment_dispersion < 0.3:
            drivers.append("Divergent opinions across sources")
        
        # Check for extreme sentiment
        if metrics.fear_greed_index > 80:
            drivers.append("Extreme greed in market")
        elif metrics.fear_greed_index < 20:
            drivers.append("Extreme fear in market")
        
        return drivers[:5]  # Top 5 drivers
    
    def _check_contrarian_indicators(self, metrics: SentimentMetrics, sources: List[SourceSentiment]) -> List[str]:
        """Check for contrarian indicators (extreme sentiment)"""
        indicators = []
        
        # Extreme overall sentiment
        if abs(metrics.overall_sentiment) > self.extreme_sentiment_threshold:
            if metrics.overall_sentiment > 0:
                indicators.append("Extreme bullish sentiment - potential top")
            else:
                indicators.append("Extreme bearish sentiment - potential bottom")
        
        # Extreme fear/greed
        if metrics.fear_greed_index > 90:
            indicators.append("Extreme greed - consider taking profits")
        elif metrics.fear_greed_index < 10:
            indicators.append("Extreme fear - potential buying opportunity")
        
        # One-sided sentiment
        if metrics.bullish_percentage > 0.9:
            indicators.append("Overcrowded long - reversal risk")
        elif metrics.bearish_percentage > 0.9:
            indicators.append("Overcrowded short - squeeze risk")
        
        # High retail vs smart money divergence
        if metrics.retail_sentiment > 0.7 and metrics.smart_money_sentiment < 0.3:
            indicators.append("Retail euphoria while smart money cautious")
        elif metrics.retail_sentiment < -0.7 and metrics.smart_money_sentiment > -0.3:
            indicators.append("Retail panic while smart money accumulating")
        
        # Check for high volume at extremes
        total_volume = sum(s.volume for s in sources)
        if total_volume > self.high_volume_threshold:
            if abs(metrics.overall_sentiment) > 0.7:
                indicators.append("High volume at sentiment extreme")
        
        return indicators
    
    def _generate_signal(self, metrics: SentimentMetrics, contrarian_indicators: List[str]) -> Tuple[SentimentSignal, float]:
        """Generate trading signal from sentiment analysis"""
        
        sentiment = metrics.overall_sentiment
        
        # Apply contrarian logic if extreme sentiment
        if contrarian_indicators:
            # Fade extreme sentiment
            if "potential top" in " ".join(contrarian_indicators):
                sentiment *= 0.5  # Reduce bullish signal
            elif "potential bottom" in " ".join(contrarian_indicators):
                sentiment *= 0.5  # Reduce bearish signal
        
        # Determine signal based on sentiment
        if sentiment >= self.extreme_bullish_threshold:
            signal = SentimentSignal.EXTREME_BULLISH
            strength = min(1.0, abs(sentiment))
        elif sentiment >= self.bullish_threshold:
            signal = SentimentSignal.BULLISH
            strength = (sentiment - self.bullish_threshold) / (self.extreme_bullish_threshold - self.bullish_threshold)
        elif sentiment <= self.extreme_bearish_threshold:
            signal = SentimentSignal.EXTREME_BEARISH
            strength = min(1.0, abs(sentiment))
        elif sentiment <= self.bearish_threshold:
            signal = SentimentSignal.BEARISH
            strength = (self.bearish_threshold - sentiment) / (self.bearish_threshold - self.extreme_bearish_threshold)
        else:
            signal = SentimentSignal.NEUTRAL
            strength = 1.0 - abs(sentiment) / self.bullish_threshold
        
        # Adjust strength based on dispersion (higher agreement = stronger signal)
        strength *= (0.5 + 0.5 * metrics.sentiment_dispersion)
        
        # Adjust for smart money alignment
        if signal in [SentimentSignal.BULLISH, SentimentSignal.EXTREME_BULLISH]:
            if metrics.smart_money_sentiment > 0:
                strength *= 1.2
        elif signal in [SentimentSignal.BEARISH, SentimentSignal.EXTREME_BEARISH]:
            if metrics.smart_money_sentiment < 0:
                strength *= 1.2
        
        strength = min(1.0, strength)
        
        return signal, strength
    
    def _generate_recommendations(self, signal: SentimentSignal, metrics: SentimentMetrics,
                                 drivers: List[str], contrarian_indicators: List[str]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Signal-based recommendations
        if signal == SentimentSignal.EXTREME_BULLISH:
            if contrarian_indicators:
                recommendations.append("Extreme bullish sentiment - consider taking partial profits")
            else:
                recommendations.append("Strong bullish sentiment - maintain or add to positions")
        elif signal == SentimentSignal.BULLISH:
            recommendations.append("Bullish sentiment - look for entry on pullbacks")
        elif signal == SentimentSignal.EXTREME_BEARISH:
            if contrarian_indicators:
                recommendations.append("Extreme bearish sentiment - potential contrarian buy")
            else:
                recommendations.append("Strong bearish sentiment - avoid or reduce exposure")
        elif signal == SentimentSignal.BEARISH:
            recommendations.append("Bearish sentiment - consider defensive positioning")
        else:
            recommendations.append("Neutral sentiment - wait for clearer signals")
        
        # Smart money recommendations
        if abs(metrics.smart_money_sentiment) > 0.5:
            if metrics.smart_money_sentiment > 0:
                recommendations.append("Smart money bullish - follow institutional positioning")
            else:
                recommendations.append("Smart money bearish - exercise caution")
        
        # Momentum recommendations
        if abs(metrics.sentiment_momentum) > 0.3:
            if metrics.sentiment_momentum > 0:
                recommendations.append("Sentiment improving - momentum favors longs")
            else:
                recommendations.append("Sentiment deteriorating - momentum favors shorts")
        
        # Fear/Greed recommendations
        if metrics.fear_greed_index > 75:
            recommendations.append(f"Fear/Greed at {metrics.fear_greed_index:.0f} - market greedy, be cautious")
        elif metrics.fear_greed_index < 25:
            recommendations.append(f"Fear/Greed at {metrics.fear_greed_index:.0f} - market fearful, consider buying")
        
        # Add contrarian warnings
        for indicator in contrarian_indicators[:2]:
            recommendations.append(f"Warning: {indicator}")
        
        # Add key drivers
        for driver in drivers[:2]:
            recommendations.append(f"Driver: {driver}")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_analyses = len(self.analysis_history)
        
        if total_analyses == 0:
            return {
                'total_analyses': 0,
                'average_api_time': 0.0,
                'signal_distribution': {},
                'source_coverage': {}
            }
        
        # Calculate metrics
        signal_counts = {}
        source_counts = {}
        
        for analysis in self.analysis_history:
            # Signal distribution
            signal = analysis.signal.value
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            # Source coverage
            for source in analysis.sources:
                source_name = source.source.value
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
        
        # Average API time
        avg_api_time = 0.0
        if 'sentiment_analysis' in self.api_call_times:
            times = self.api_call_times['sentiment_analysis']
            avg_api_time = sum(times) / len(times) if times else 0.0
        
        return {
            'total_analyses': total_analyses,
            'average_api_time_seconds': avg_api_time,
            'signal_distribution': signal_counts,
            'source_coverage': source_counts,
            'sources_available': len(SentimentSource),
            'finbert_available': self.finbert_model is not None,
            'cache_size': len(self.sentiment_cache)
        }


# Create global instance
sentiment_analysis_agent = SentimentAnalysisAgent()