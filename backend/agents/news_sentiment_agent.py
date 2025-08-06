"""
News Sentiment Agent
Analyzes news sentiment from multiple sources to generate trading signals
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from agents.base import BaseAgent, AgentContext, AgentCapability, AgentConfig, Signal, SignalAction, SignalStrength
from services.alternative_data_service import alternative_data_service, DataSourceType, NewsArticle

logger = logging.getLogger(__name__)


class NewsSentimentAgent(BaseAgent):
    """Agent that analyzes news sentiment for trading decisions"""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="NewsSentiment",
                confidence_threshold=0.65
            )
        super().__init__(config)
        
        # Sentiment thresholds and impacts
        self.sentiment_thresholds = {
            'very_bullish': 0.7,
            'bullish': 0.3,
            'neutral_positive': 0.1,
            'neutral_negative': -0.1,
            'bearish': -0.3,
            'very_bearish': -0.7
        }
        
        # Source credibility weights
        self.source_weights = {
            'bloomberg': 1.0,
            'reuters': 0.95,
            'wsj': 0.95,
            'ft': 0.9,
            'cnbc': 0.85,
            'marketwatch': 0.8,
            'seekingalpha': 0.75,
            'benzinga': 0.7,
            'yahoo': 0.65,
            'default': 0.5
        }
        
        # Category impacts on sectors
        self.category_impacts = {
            'earnings': {
                'impact': 'high',
                'decay_hours': 48,
                'sectors': ['all']
            },
            'merger': {
                'impact': 'very_high',
                'decay_hours': 72,
                'sectors': ['specific']
            },
            'regulation': {
                'impact': 'high',
                'decay_hours': 168,  # 1 week
                'sectors': ['affected']
            },
            'macro': {
                'impact': 'medium',
                'decay_hours': 96,
                'sectors': ['all']
            },
            'product': {
                'impact': 'medium',
                'decay_hours': 24,
                'sectors': ['specific']
            },
            'analyst': {
                'impact': 'medium',
                'decay_hours': 48,
                'sectors': ['specific']
            },
            'scandal': {
                'impact': 'very_high',
                'decay_hours': 168,
                'sectors': ['specific']
            }
        }
        
        # Sentiment momentum tracking
        self.sentiment_history = defaultdict(list)
        self.max_history_hours = 72
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze news sentiment to generate trading signals"""
        
        try:
            symbol = market_data.get('symbol', 'SPY')
            
            # Fetch news data
            news_data = await alternative_data_service.get_comprehensive_alternative_data(
                symbols=[symbol],
                data_types=[DataSourceType.NEWS]
            )
            
            if not news_data or not news_data.get('news'):
                logger.warning(f"No news data available for {symbol}")
                return None
            
            articles = news_data['news']
            
            # Analyze news sentiment
            analysis = self._analyze_news_sentiment(articles, symbol)
            
            # Calculate momentum
            momentum = self._calculate_sentiment_momentum(symbol, analysis['weighted_sentiment'])
            
            # Identify key themes and events
            themes = self._extract_key_themes(articles)
            
            # Assess news velocity and importance
            velocity = self._calculate_news_velocity(articles)
            importance = self._calculate_news_importance(articles, symbol)
            
            # Generate signal
            signal_data = self._generate_signal_from_news(
                analysis, momentum, velocity, importance, themes
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                analysis, momentum, velocity, len(articles)
            )
            
            # Determine strength
            if confidence > 0.8 and abs(analysis['weighted_sentiment']) > 0.5:
                strength = SignalStrength.STRONG
            elif confidence > 0.6:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Map action
            action_map = {
                'strong_buy': SignalAction.BUY,
                'buy': SignalAction.BUY,
                'hold': SignalAction.HOLD,
                'sell': SignalAction.SELL,
                'strong_sell': SignalAction.SELL
            }
            
            action = action_map.get(signal_data['action'], SignalAction.HOLD)
            current_price = market_data.get('price', 0.0)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                analysis, momentum, velocity, themes, signal_data
            )
            
            # Identify risks
            risks = self._identify_risks(analysis, momentum, themes)
            
            return Signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                strength=strength,
                source=self.config.name,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    'sentiment_score': analysis['weighted_sentiment'],
                    'sentiment_distribution': analysis['sentiment_distribution'],
                    'momentum': momentum,
                    'news_velocity': velocity,
                    'news_importance': importance,
                    'key_themes': themes[:5],
                    'top_articles': self._format_top_articles(articles[:3]),
                    'risks': risks
                },
                market_conditions={
                    'news_count': len(articles),
                    'unique_sources': analysis['unique_sources'],
                    'avg_relevance': analysis['avg_relevance'],
                    'time_span_hours': analysis['time_span_hours']
                }
            )
            
        except Exception as e:
            logger.error(f"News sentiment analysis error: {e}")
            return None
    
    def _analyze_news_sentiment(
        self,
        articles: List[NewsArticle],
        symbol: str
    ) -> Dict[str, Any]:
        """Analyze sentiment from news articles"""
        
        if not articles:
            return {
                'weighted_sentiment': 0.0,
                'raw_sentiment': 0.0,
                'sentiment_distribution': {},
                'unique_sources': 0,
                'avg_relevance': 0.0,
                'time_span_hours': 0
            }
        
        # Calculate weighted sentiment
        total_weighted_sentiment = 0.0
        total_weight = 0.0
        sentiment_counts = defaultdict(int)
        sources = set()
        
        # Time range
        latest_time = max(a.published_at for a in articles)
        earliest_time = min(a.published_at for a in articles)
        time_span_hours = (latest_time - earliest_time).total_seconds() / 3600
        
        for article in articles:
            # Get source weight
            source_lower = article.source.lower()
            source_weight = self.source_weights.get('default', 0.5)
            for source_name, weight in self.source_weights.items():
                if source_name in source_lower:
                    source_weight = weight
                    break
            
            # Time decay (more recent = higher weight)
            hours_old = (datetime.utcnow() - article.published_at).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - (hours_old / 72))  # Decay over 72 hours
            
            # Relevance weight
            relevance_weight = article.relevance
            
            # Check if symbol is in tickers
            symbol_weight = 1.5 if symbol in article.tickers else 0.8
            
            # Combined weight
            weight = source_weight * time_weight * relevance_weight * symbol_weight
            
            # Add to weighted sentiment
            total_weighted_sentiment += article.sentiment * weight
            total_weight += weight
            
            # Track sentiment distribution
            if article.sentiment > self.sentiment_thresholds['bullish']:
                sentiment_counts['bullish'] += 1
            elif article.sentiment < self.sentiment_thresholds['bearish']:
                sentiment_counts['bearish'] += 1
            else:
                sentiment_counts['neutral'] += 1
            
            sources.add(article.source)
        
        # Calculate averages
        weighted_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0.0
        raw_sentiment = np.mean([a.sentiment for a in articles])
        avg_relevance = np.mean([a.relevance for a in articles])
        
        # Sentiment distribution
        total_articles = len(articles)
        sentiment_distribution = {
            k: v / total_articles for k, v in sentiment_counts.items()
        }
        
        return {
            'weighted_sentiment': weighted_sentiment,
            'raw_sentiment': raw_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'unique_sources': len(sources),
            'avg_relevance': avg_relevance,
            'time_span_hours': time_span_hours
        }
    
    def _calculate_sentiment_momentum(
        self,
        symbol: str,
        current_sentiment: float
    ) -> float:
        """Calculate sentiment momentum (rate of change)"""
        
        # Add current sentiment to history
        self.sentiment_history[symbol].append({
            'timestamp': datetime.utcnow(),
            'sentiment': current_sentiment
        })
        
        # Clean old history
        cutoff_time = datetime.utcnow() - timedelta(hours=self.max_history_hours)
        self.sentiment_history[symbol] = [
            h for h in self.sentiment_history[symbol]
            if h['timestamp'] > cutoff_time
        ]
        
        history = self.sentiment_history[symbol]
        
        if len(history) < 3:
            return 0.0
        
        # Calculate momentum over different timeframes
        momentum_1h = 0.0
        momentum_6h = 0.0
        momentum_24h = 0.0
        
        now = datetime.utcnow()
        
        # 1-hour momentum
        one_hour_ago = now - timedelta(hours=1)
        recent_sentiments = [h['sentiment'] for h in history if h['timestamp'] > one_hour_ago]
        if len(recent_sentiments) > 1:
            momentum_1h = recent_sentiments[-1] - recent_sentiments[0]
        
        # 6-hour momentum
        six_hours_ago = now - timedelta(hours=6)
        medium_sentiments = [h['sentiment'] for h in history if h['timestamp'] > six_hours_ago]
        if len(medium_sentiments) > 1:
            momentum_6h = (medium_sentiments[-1] - medium_sentiments[0]) / 6
        
        # 24-hour momentum
        day_ago = now - timedelta(hours=24)
        long_sentiments = [h['sentiment'] for h in history if h['timestamp'] > day_ago]
        if len(long_sentiments) > 1:
            momentum_24h = (long_sentiments[-1] - long_sentiments[0]) / 24
        
        # Weighted average momentum
        weighted_momentum = (
            momentum_1h * 0.5 +
            momentum_6h * 0.3 +
            momentum_24h * 0.2
        )
        
        return weighted_momentum
    
    def _extract_key_themes(self, articles: List[NewsArticle]) -> List[str]:
        """Extract key themes from news articles"""
        
        theme_counts = defaultdict(int)
        
        # Keywords for different themes
        theme_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'guidance'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover', 'deal'],
            'product': ['launch', 'product', 'release', 'announcement', 'innovation'],
            'regulation': ['regulation', 'sec', 'fda', 'compliance', 'investigation'],
            'macro': ['fed', 'inflation', 'gdp', 'unemployment', 'economy'],
            'analyst': ['upgrade', 'downgrade', 'rating', 'target', 'analyst'],
            'scandal': ['scandal', 'lawsuit', 'fraud', 'violation', 'investigation'],
            'technology': ['ai', 'technology', 'innovation', 'breakthrough', 'patent'],
            'competition': ['competitor', 'market share', 'rivalry', 'competition'],
            'expansion': ['expansion', 'growth', 'international', 'new market']
        }
        
        for article in articles:
            text_lower = (article.title + ' ' + article.summary).lower()
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    # Weight by relevance and recency
                    hours_old = (datetime.utcnow() - article.published_at).total_seconds() / 3600
                    recency_weight = max(0.1, 1.0 - (hours_old / 48))
                    theme_counts[theme] += article.relevance * recency_weight
        
        # Sort themes by count
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [theme for theme, _ in sorted_themes]
    
    def _calculate_news_velocity(self, articles: List[NewsArticle]) -> float:
        """Calculate news velocity (rate of news flow)"""
        
        if len(articles) < 2:
            return 0.0
        
        # Group articles by hour
        hourly_counts = defaultdict(int)
        
        for article in articles:
            hour_key = article.published_at.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
        
        if not hourly_counts:
            return 0.0
        
        # Calculate average articles per hour
        avg_per_hour = len(articles) / len(hourly_counts)
        
        # Normalize (assume 5 articles/hour is high velocity)
        velocity = min(avg_per_hour / 5, 1.0)
        
        # Check for acceleration (recent hours have more articles)
        recent_hours = sorted(hourly_counts.keys())[-3:]
        older_hours = sorted(hourly_counts.keys())[:-3]
        
        if recent_hours and older_hours:
            recent_avg = np.mean([hourly_counts[h] for h in recent_hours])
            older_avg = np.mean([hourly_counts[h] for h in older_hours])
            
            if recent_avg > older_avg * 1.5:
                velocity *= 1.3  # Acceleration bonus
        
        return min(velocity, 1.0)
    
    def _calculate_news_importance(
        self,
        articles: List[NewsArticle],
        symbol: str
    ) -> float:
        """Calculate importance of news"""
        
        if not articles:
            return 0.0
        
        importance_score = 0.0
        
        for article in articles:
            article_importance = 0.0
            
            # Direct mention of symbol
            if symbol in article.tickers:
                article_importance += 0.3
            
            # Check for important categories
            text_lower = (article.title + ' ' + article.summary).lower()
            
            high_importance_keywords = [
                'breaking', 'urgent', 'exclusive', 'confirmed',
                'sec', 'fda', 'doj', 'federal',
                'bankruptcy', 'fraud', 'investigation',
                'merger', 'acquisition', 'buyout'
            ]
            
            for keyword in high_importance_keywords:
                if keyword in text_lower:
                    article_importance += 0.2
                    break
            
            # Source credibility
            source_lower = article.source.lower()
            if any(s in source_lower for s in ['bloomberg', 'reuters', 'wsj']):
                article_importance += 0.2
            
            # Relevance score
            article_importance += article.relevance * 0.3
            
            importance_score += min(article_importance, 1.0)
        
        # Average importance
        avg_importance = importance_score / len(articles)
        
        return min(avg_importance * 1.5, 1.0)  # Scale up slightly
    
    def _generate_signal_from_news(
        self,
        analysis: Dict,
        momentum: float,
        velocity: float,
        importance: float,
        themes: List[str]
    ) -> Dict[str, Any]:
        """Generate trading signal from news analysis"""
        
        sentiment = analysis['weighted_sentiment']
        
        # Base signal on sentiment
        if sentiment > self.sentiment_thresholds['very_bullish']:
            base_action = 'strong_buy'
        elif sentiment > self.sentiment_thresholds['bullish']:
            base_action = 'buy'
        elif sentiment < self.sentiment_thresholds['very_bearish']:
            base_action = 'strong_sell'
        elif sentiment < self.sentiment_thresholds['bearish']:
            base_action = 'sell'
        else:
            base_action = 'hold'
        
        # Adjust for momentum
        if momentum > 0.1 and sentiment > 0:
            # Positive momentum reinforces bullish sentiment
            if base_action == 'buy':
                base_action = 'strong_buy'
        elif momentum < -0.1 and sentiment < 0:
            # Negative momentum reinforces bearish sentiment
            if base_action == 'sell':
                base_action = 'strong_sell'
        elif momentum * sentiment < 0:
            # Momentum contradicts sentiment - be cautious
            if base_action in ['strong_buy', 'strong_sell']:
                base_action = base_action.replace('strong_', '')
        
        # Adjust for velocity and importance
        if velocity > 0.7 and importance > 0.7:
            # High velocity + high importance = stronger signal
            pass  # Keep strong signals
        elif velocity < 0.3 or importance < 0.3:
            # Low velocity or importance = weaker signal
            if base_action == 'strong_buy':
                base_action = 'buy'
            elif base_action == 'strong_sell':
                base_action = 'sell'
        
        # Check for specific high-impact themes
        if themes:
            if 'scandal' in themes[:2] or 'investigation' in themes[:2]:
                # Negative themes
                if base_action in ['buy', 'strong_buy']:
                    base_action = 'hold'
                elif base_action == 'hold':
                    base_action = 'sell'
            elif 'merger' in themes[:2] or 'buyout' in themes[:2]:
                # Usually positive
                if base_action == 'hold':
                    base_action = 'buy'
        
        return {
            'action': base_action,
            'sentiment': sentiment,
            'momentum': momentum
        }
    
    def _calculate_confidence(
        self,
        analysis: Dict,
        momentum: float,
        velocity: float,
        article_count: int
    ) -> float:
        """Calculate confidence in news signal"""
        
        base_confidence = 0.5
        
        # Factor 1: Number of articles (more = better)
        article_confidence = min(article_count / 20, 1.0) * 0.2
        
        # Factor 2: Source diversity
        source_confidence = min(analysis['unique_sources'] / 5, 1.0) * 0.2
        
        # Factor 3: Sentiment consensus
        distribution = analysis['sentiment_distribution']
        if distribution:
            max_sentiment = max(distribution.values())
            consensus_confidence = max_sentiment * 0.2
        else:
            consensus_confidence = 0.0
        
        # Factor 4: Momentum alignment
        sentiment = analysis['weighted_sentiment']
        if sentiment * momentum > 0:  # Same direction
            momentum_confidence = 0.15
        else:
            momentum_confidence = 0.0
        
        # Factor 5: News velocity (moderate is best)
        if 0.3 < velocity < 0.7:
            velocity_confidence = 0.1
        else:
            velocity_confidence = 0.05
        
        # Factor 6: Relevance
        relevance_confidence = analysis['avg_relevance'] * 0.15
        
        confidence = (
            base_confidence +
            article_confidence +
            source_confidence +
            consensus_confidence +
            momentum_confidence +
            velocity_confidence +
            relevance_confidence
        )
        
        return min(confidence, 0.95)
    
    def _generate_reasoning(
        self,
        analysis: Dict,
        momentum: float,
        velocity: float,
        themes: List[str],
        signal_data: Dict
    ) -> List[str]:
        """Generate reasoning for the signal"""
        
        reasoning = []
        
        # Sentiment summary
        sentiment = analysis['weighted_sentiment']
        if abs(sentiment) > 0.5:
            direction = "strongly bullish" if sentiment > 0 else "strongly bearish"
        elif abs(sentiment) > 0.2:
            direction = "bullish" if sentiment > 0 else "bearish"
        else:
            direction = "neutral"
        
        reasoning.append(f"News sentiment is {direction} ({sentiment:.2f})")
        
        # Momentum
        if abs(momentum) > 0.1:
            momentum_dir = "improving" if momentum > 0 else "deteriorating"
            reasoning.append(f"Sentiment momentum is {momentum_dir}")
        
        # Key themes
        if themes:
            reasoning.append(f"Key themes: {', '.join(themes[:3])}")
        
        # News velocity
        if velocity > 0.7:
            reasoning.append("High news velocity indicates significant event")
        elif velocity < 0.3:
            reasoning.append("Low news activity")
        
        # Source consensus
        distribution = analysis['sentiment_distribution']
        if distribution:
            dominant = max(distribution, key=distribution.get)
            if distribution[dominant] > 0.6:
                reasoning.append(f"Strong {dominant} consensus across sources")
        
        return reasoning[:5]
    
    def _identify_risks(
        self,
        analysis: Dict,
        momentum: float,
        themes: List[str]
    ) -> List[str]:
        """Identify risks from news analysis"""
        
        risks = []
        
        # Sentiment momentum divergence
        if analysis['weighted_sentiment'] * momentum < 0:
            risks.append("Sentiment momentum divergence detected")
        
        # Low source diversity
        if analysis['unique_sources'] < 3:
            risks.append("Limited news source diversity")
        
        # Negative themes
        negative_themes = ['scandal', 'investigation', 'lawsuit', 'fraud', 'bankruptcy']
        for theme in themes[:5]:
            if theme in negative_themes:
                risks.append(f"Negative theme detected: {theme}")
                break
        
        # Old news
        if analysis['time_span_hours'] > 48:
            risks.append("News may be stale (>48 hours)")
        
        # Mixed sentiment
        distribution = analysis['sentiment_distribution']
        if distribution and distribution.get('neutral', 0) > 0.5:
            risks.append("Mixed sentiment signals")
        
        return risks[:4]
    
    def _format_top_articles(self, articles: List[NewsArticle]) -> List[Dict]:
        """Format top articles for display"""
        
        formatted = []
        
        for article in articles:
            formatted.append({
                'title': article.title[:100],
                'source': article.source,
                'sentiment': round(article.sentiment, 2),
                'time': article.published_at.isoformat(),
                'relevance': round(article.relevance, 2)
            })
        
        return formatted
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for news sentiment analysis
        
        Returns:
            List of data type strings
        """
        return [
            'news',  # Primary requirement - news articles with sentiment
            'symbol',  # Stock symbol
            'price'  # Current price for context
        ]


# Create global instance
news_sentiment_agent = NewsSentimentAgent()