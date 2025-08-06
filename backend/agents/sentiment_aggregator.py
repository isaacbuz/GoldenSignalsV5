"""
Sentiment Aggregator Agent
Combines sentiment analysis from all social media sources
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from agents.base import BaseAgent, AgentContext, AgentCapability, AgentConfig
from services.social_sentiment_analyzer import social_sentiment_analyzer

logger = logging.getLogger(__name__)


class SentimentAggregatorAgent(BaseAgent):
    """Agent that aggregates sentiment from multiple social media platforms"""
    
    def __init__(self):
        config = AgentConfig(
            name="SentimentAggregator",
            capabilities=[
                AgentCapability.SENTIMENT_ANALYSIS,
                AgentCapability.SOCIAL_MEDIA_MONITORING
            ],
            confidence_threshold=0.6
        )
        super().__init__(config)
        
        # Platform weights based on reliability and relevance
        self.platform_weights = {
            'twitter': 0.35,    # High impact, real-time
            'reddit': 0.30,     # Deep discussions, WSB influence
            'stocktwits': 0.20, # Trading-focused
            'discord': 0.10,    # Insider communities
            'news': 0.05        # Professional sentiment
        }
        
        # Sentiment impact on trading signals
        self.sentiment_impact = {
            'STRONG_BUY': 0.3,
            'BUY': 0.15,
            'NEUTRAL': 0.0,
            'SELL': -0.15,
            'STRONG_SELL': -0.3
        }
        
    def get_required_data_types(self) -> List[str]:
        """Returns list of required data types for this agent"""
        return ['social_media', 'news', 'sentiment']
    
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze sentiment across all platforms"""
        
        symbol = context.symbol
        timeframe = context.timeframe
        
        try:
            # Determine time window based on timeframe
            time_window_hours = self._get_time_window(timeframe)
            
            # Get sentiment from all platforms
            sentiment_data = await social_sentiment_analyzer.analyze_symbol_sentiment(
                symbol=symbol,
                time_window_hours=time_window_hours
            )
            
            # Process and enhance sentiment data
            enhanced_sentiment = self._enhance_sentiment_analysis(sentiment_data)
            
            # Generate trading signal based on sentiment
            signal = self._generate_sentiment_signal(enhanced_sentiment)
            
            # Calculate confidence
            confidence = self._calculate_confidence(enhanced_sentiment)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'sentiment_score': enhanced_sentiment['weighted_score'],
                'sentiment_trend': enhanced_sentiment['trend'],
                'platform_consensus': enhanced_sentiment['consensus'],
                'risk_factors': enhanced_sentiment['risk_factors'],
                'recommendation': enhanced_sentiment['recommendation'],
                'metadata': {
                    'posts_analyzed': sentiment_data['total_posts_analyzed'],
                    'platforms': list(sentiment_data['platform_breakdown'].keys()),
                    'anomalies': sentiment_data['anomalies'],
                    'top_topics': enhanced_sentiment['trending_topics']
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e)
            }
            
    def _get_time_window(self, timeframe: str) -> int:
        """Get appropriate time window for sentiment analysis"""
        
        timeframe_windows = {
            '1min': 1,
            '5min': 2,
            '15min': 4,
            '30min': 6,
            '1h': 12,
            '4h': 24,
            '1d': 72,
            '1w': 168
        }
        
        return timeframe_windows.get(timeframe, 24)
        
    def _enhance_sentiment_analysis(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance raw sentiment data with additional analysis"""
        
        # Calculate weighted sentiment score
        weighted_score = 0.0
        total_weight = 0.0
        
        platform_scores = []
        for platform, data in sentiment_data['platform_breakdown'].items():
            if platform in self.platform_weights:
                weight = self.platform_weights[platform]
                score = data.average_sentiment
                weighted_score += score * weight
                total_weight += weight
                platform_scores.append(score)
                
        if total_weight > 0:
            weighted_score /= total_weight
            
        # Calculate sentiment trend (momentum)
        # This would need historical data in production
        sentiment_trend = 'stable'
        if weighted_score > 0.3:
            sentiment_trend = 'improving'
        elif weighted_score < -0.3:
            sentiment_trend = 'deteriorating'
            
        # Platform consensus
        consensus = 'mixed'
        if all(score > 0 for score in platform_scores):
            consensus = 'bullish'
        elif all(score < 0 for score in platform_scores):
            consensus = 'bearish'
        elif np.std(platform_scores) < 0.2:
            consensus = 'neutral'
            
        # Risk factors
        risk_factors = []
        
        # Check for anomalies
        for anomaly in sentiment_data['anomalies']:
            if anomaly['type'] == 'coordinated_activity':
                risk_factors.append('Potential manipulation detected')
            elif anomaly['type'] == 'volume_spike':
                risk_factors.append('Unusual activity spike')
            elif anomaly['type'] == 'influencer_activity':
                risk_factors.append('Influencer involvement')
                
        # Check platform disagreement
        if np.std(platform_scores) > 0.5:
            risk_factors.append('High disagreement between platforms')
            
        # Extract trending topics
        all_topics = []
        for platform_data in sentiment_data['platform_breakdown'].values():
            all_topics.extend(platform_data.trending_topics[:5])
            
        # Count topic frequency
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
        trending_topics = sorted(
            topic_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            weighted_score, 
            consensus, 
            risk_factors
        )
        
        return {
            'weighted_score': weighted_score,
            'trend': sentiment_trend,
            'consensus': consensus,
            'risk_factors': risk_factors,
            'trending_topics': [topic for topic, _ in trending_topics],
            'recommendation': recommendation
        }
        
    def _generate_sentiment_signal(self, enhanced_sentiment: Dict[str, Any]) -> str:
        """Generate trading signal from sentiment"""
        
        score = enhanced_sentiment['weighted_score']
        consensus = enhanced_sentiment['consensus']
        risk_factors = enhanced_sentiment['risk_factors']
        
        # Base signal on score
        if score > 0.5:
            signal = 'STRONG_BUY'
        elif score > 0.2:
            signal = 'BUY'
        elif score < -0.5:
            signal = 'STRONG_SELL'
        elif score < -0.2:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
            
        # Adjust for consensus
        if consensus == 'mixed' and signal != 'NEUTRAL':
            # Downgrade signal if no consensus
            if signal == 'STRONG_BUY':
                signal = 'BUY'
            elif signal == 'STRONG_SELL':
                signal = 'SELL'
            elif signal in ['BUY', 'SELL']:
                signal = 'NEUTRAL'
                
        # Adjust for risk
        if len(risk_factors) > 2:
            # Too risky, move toward neutral
            if signal in ['STRONG_BUY', 'STRONG_SELL']:
                signal = signal.replace('STRONG_', '')
            elif signal in ['BUY', 'SELL']:
                signal = 'NEUTRAL'
                
        return signal
        
    def _calculate_confidence(self, enhanced_sentiment: Dict[str, Any]) -> float:
        """Calculate confidence in sentiment signal"""
        
        # Base confidence on score magnitude
        base_confidence = min(abs(enhanced_sentiment['weighted_score']), 1.0)
        
        # Adjust for consensus
        consensus_multiplier = {
            'bullish': 1.2,
            'bearish': 1.2,
            'neutral': 1.0,
            'mixed': 0.7
        }
        
        confidence = base_confidence * consensus_multiplier.get(
            enhanced_sentiment['consensus'], 1.0
        )
        
        # Reduce for risk factors
        risk_penalty = len(enhanced_sentiment['risk_factors']) * 0.1
        confidence = max(0, confidence - risk_penalty)
        
        # Cap at threshold
        return min(confidence, 0.95)
        
    def _generate_recommendation(
        self, 
        score: float, 
        consensus: str, 
        risk_factors: List[str]
    ) -> str:
        """Generate human-readable recommendation"""
        
        if score > 0.5 and consensus == 'bullish' and len(risk_factors) < 2:
            return "Strong positive sentiment across platforms. Consider bullish positions."
        elif score > 0.2 and len(risk_factors) < 3:
            return "Moderately positive sentiment. Potential buying opportunity."
        elif score < -0.5 and consensus == 'bearish' and len(risk_factors) < 2:
            return "Strong negative sentiment detected. Consider defensive positions."
        elif score < -0.2 and len(risk_factors) < 3:
            return "Moderately negative sentiment. Exercise caution."
        elif len(risk_factors) >= 3:
            return "High risk detected due to unusual activity. Wait for clarity."
        else:
            return "Mixed or neutral sentiment. No clear directional bias."
            
    async def get_detailed_sentiment(
        self, 
        symbol: str,
        include_posts: bool = False
    ) -> Dict[str, Any]:
        """Get detailed sentiment analysis with individual posts"""
        
        sentiment_data = await social_sentiment_analyzer.analyze_symbol_sentiment(
            symbol=symbol,
            time_window_hours=24
        )
        
        detailed_report = {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'overall_sentiment': sentiment_data['overall_sentiment'],
            'platforms': {}
        }
        
        # Add platform details
        for platform, data in sentiment_data['platform_breakdown'].items():
            platform_detail = {
                'posts_analyzed': data.posts_analyzed,
                'average_sentiment': data.average_sentiment,
                'sentiment_distribution': data.sentiment_distribution,
                'influencer_sentiment': data.influencer_sentiment,
                'trending_topics': data.trending_topics
            }
            
            if include_posts:
                platform_detail['top_bullish_posts'] = [
                    {
                        'content': post.content[:200],
                        'author': post.author,
                        'sentiment': post.sentiment_score,
                        'engagement': post.engagement,
                        'url': post.url
                    }
                    for post in data.top_bullish_posts[:3]
                ]
                
                platform_detail['top_bearish_posts'] = [
                    {
                        'content': post.content[:200],
                        'author': post.author,
                        'sentiment': post.sentiment_score,
                        'engagement': post.engagement,
                        'url': post.url
                    }
                    for post in data.top_bearish_posts[:3]
                ]
                
            detailed_report['platforms'][platform] = platform_detail
            
        return detailed_report


# Create global instance
sentiment_aggregator_agent = SentimentAggregatorAgent()