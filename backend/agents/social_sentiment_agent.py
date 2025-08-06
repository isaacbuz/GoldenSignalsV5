"""
Social Sentiment Agent
Analyzes social media sentiment from Twitter, Reddit, StockTwits to generate trading signals
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import re

from agents.base import BaseAgent, AgentContext, AgentCapability, AgentConfig, Signal, SignalAction, SignalStrength
from services.alternative_data_service import alternative_data_service, DataSourceType, SocialPost

logger = logging.getLogger(__name__)


class SocialSentimentAgent(BaseAgent):
    """Agent that analyzes social media sentiment for trading decisions"""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="SocialSentiment",
                confidence_threshold=0.60
            )
        super().__init__(config)
        
        # Platform-specific weights
        self.platform_weights = {
            'stocktwits': 1.0,  # Most trading-focused
            'twitter': 0.8,
            'reddit': 0.9,  # WSB effect
            'discord': 0.7,
            'telegram': 0.6
        }
        
        # Meme stock detection patterns
        self.meme_patterns = {
            'rocket_emojis': ['🚀', '🌙', '🌕', '💎', '🙌'],
            'phrases': [
                'to the moon', 'diamond hands', 'paper hands',
                'yolo', 'tendies', 'apes together', 'hodl',
                'short squeeze', 'gamma squeeze', 'moass'
            ],
            'high_risk_indicators': [
                'all in', '100%', 'mortgage', 'loan',
                'life savings', 'margin', 'leverage'
            ]
        }
        
        # Influencer tracking
        self.influencer_thresholds = {
            'micro': 1000,      # 1K+ followers
            'mid': 10000,       # 10K+ followers
            'macro': 100000,    # 100K+ followers
            'mega': 1000000     # 1M+ followers
        }
        
        # Sentiment aggregation windows
        self.time_windows = {
            'immediate': 1,  # 1 hour
            'short': 6,      # 6 hours
            'medium': 24,    # 24 hours
            'long': 72       # 72 hours
        }
        
        # Viral detection thresholds
        self.viral_thresholds = {
            'engagement_rate': 0.1,  # 10% engagement
            'share_velocity': 100,   # shares per hour
            'mention_spike': 5       # 5x normal mentions
        }
        
        # Historical baseline for comparison
        self.baseline_activity = defaultdict(lambda: {
            'avg_mentions': 0,
            'avg_sentiment': 0,
            'avg_engagement': 0
        })
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze social media sentiment to generate trading signals"""
        
        try:
            symbol = market_data.get('symbol', 'SPY')
            
            # Fetch social data
            social_data = await alternative_data_service.get_comprehensive_alternative_data(
                symbols=[symbol],
                data_types=[DataSourceType.SOCIAL]
            )
            
            if not social_data or not social_data.get('social'):
                logger.warning(f"No social data available for {symbol}")
                return None
            
            posts = social_data['social']
            
            # Analyze social sentiment
            analysis = self._analyze_social_sentiment(posts, symbol)
            
            # Detect meme stock activity
            meme_score = self._detect_meme_activity(posts)
            
            # Analyze influencer impact
            influencer_impact = self._analyze_influencer_impact(posts)
            
            # Calculate viral metrics
            viral_metrics = self._calculate_viral_metrics(posts, symbol)
            
            # Detect sentiment shifts
            sentiment_shift = self._detect_sentiment_shift(posts, symbol)
            
            # Analyze crowd behavior
            crowd_behavior = self._analyze_crowd_behavior(posts)
            
            # Generate signal
            signal_data = self._generate_signal_from_social(
                analysis, meme_score, influencer_impact, 
                viral_metrics, sentiment_shift, crowd_behavior
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                analysis, viral_metrics, crowd_behavior, len(posts)
            )
            
            # Adjust for meme stock risk
            if meme_score > 0.7:
                confidence *= 0.8  # Reduce confidence for meme stocks
            
            # Determine strength
            if confidence > 0.75 and abs(analysis['weighted_sentiment']) > 0.6:
                strength = SignalStrength.STRONG
            elif confidence > 0.55:
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
                analysis, meme_score, viral_metrics, 
                sentiment_shift, signal_data
            )
            
            # Identify risks
            risks = self._identify_risks(
                analysis, meme_score, crowd_behavior, viral_metrics
            )
            
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
                    'platform_breakdown': analysis['platform_sentiment'],
                    'meme_score': meme_score,
                    'influencer_impact': influencer_impact,
                    'viral_metrics': viral_metrics,
                    'sentiment_shift': sentiment_shift,
                    'crowd_behavior': crowd_behavior,
                    'top_posts': self._format_top_posts(posts[:5]),
                    'risks': risks
                },
                market_conditions={
                    'post_count': len(posts),
                    'unique_authors': analysis['unique_authors'],
                    'total_engagement': analysis['total_engagement'],
                    'platforms_active': len(analysis['platform_sentiment'])
                }
            )
            
        except Exception as e:
            logger.error(f"Social sentiment analysis error: {e}")
            return None
    
    def _analyze_social_sentiment(
        self,
        posts: List[SocialPost],
        symbol: str
    ) -> Dict[str, Any]:
        """Analyze sentiment from social media posts"""
        
        if not posts:
            return {
                'weighted_sentiment': 0.0,
                'raw_sentiment': 0.0,
                'platform_sentiment': {},
                'unique_authors': 0,
                'total_engagement': 0
            }
        
        # Calculate weighted sentiment
        total_weighted_sentiment = 0.0
        total_weight = 0.0
        platform_sentiments = defaultdict(list)
        authors = set()
        total_engagement = 0
        
        for post in posts:
            # Platform weight
            platform_weight = self.platform_weights.get(post.platform, 0.5)
            
            # Time decay
            hours_old = (datetime.utcnow() - post.timestamp).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - (hours_old / 48))
            
            # Influence weight
            influence_weight = post.influence_score
            
            # Engagement weight (normalized)
            engagement_weight = min(post.engagement / 1000, 1.0)
            
            # Check if symbol mentioned
            symbol_mentioned = symbol in post.tickers
            symbol_weight = 1.5 if symbol_mentioned else 0.7
            
            # Combined weight
            weight = (
                platform_weight * time_weight * 
                influence_weight * (1 + engagement_weight) * symbol_weight
            )
            
            # Add to weighted sentiment
            total_weighted_sentiment += post.sentiment * weight
            total_weight += weight
            
            # Track platform-specific sentiment
            platform_sentiments[post.platform].append(post.sentiment)
            
            # Track unique authors
            authors.add(post.author)
            
            # Total engagement
            total_engagement += post.engagement
        
        # Calculate averages
        weighted_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0.0
        raw_sentiment = np.mean([p.sentiment for p in posts])
        
        # Platform breakdown
        platform_sentiment = {
            platform: np.mean(sentiments)
            for platform, sentiments in platform_sentiments.items()
        }
        
        return {
            'weighted_sentiment': weighted_sentiment,
            'raw_sentiment': raw_sentiment,
            'platform_sentiment': platform_sentiment,
            'unique_authors': len(authors),
            'total_engagement': total_engagement
        }
    
    def _detect_meme_activity(self, posts: List[SocialPost]) -> float:
        """Detect meme stock activity patterns"""
        
        if not posts:
            return 0.0
        
        meme_score = 0.0
        meme_indicators = 0
        
        for post in posts:
            content_lower = post.content.lower()
            
            # Check for rocket emojis
            emoji_count = sum(
                content_lower.count(emoji) 
                for emoji in self.meme_patterns['rocket_emojis']
            )
            if emoji_count > 0:
                meme_indicators += min(emoji_count / 3, 1.0)
            
            # Check for meme phrases
            phrase_count = sum(
                1 for phrase in self.meme_patterns['phrases']
                if phrase in content_lower
            )
            if phrase_count > 0:
                meme_indicators += min(phrase_count / 2, 1.0)
            
            # Check for high-risk indicators
            risk_count = sum(
                1 for indicator in self.meme_patterns['high_risk_indicators']
                if indicator in content_lower
            )
            if risk_count > 0:
                meme_indicators += risk_count * 0.5
            
            # Check for ALL CAPS enthusiasm
            if len(post.content) > 10:
                caps_ratio = sum(1 for c in post.content if c.isupper()) / len(post.content)
                if caps_ratio > 0.5:
                    meme_indicators += 0.3
            
            # Platform bonus for Reddit
            if post.platform == 'reddit' and any(
                sub in post.author.lower() 
                for sub in ['wsb', 'wallstreet', 'yolo']
            ):
                meme_indicators += 0.5
        
        # Normalize meme score
        meme_score = min(meme_indicators / len(posts), 1.0)
        
        # Check for sudden spike in activity (meme stock characteristic)
        if len(posts) > 50:  # High volume
            meme_score = min(meme_score * 1.3, 1.0)
        
        return meme_score
    
    def _analyze_influencer_impact(self, posts: List[SocialPost]) -> Dict[str, Any]:
        """Analyze impact of influencers on sentiment"""
        
        influencer_posts = {
            'mega': [],
            'macro': [],
            'mid': [],
            'micro': []
        }
        
        for post in posts:
            # Estimate followers from influence score
            estimated_followers = post.influence_score * 1000000
            
            if estimated_followers >= self.influencer_thresholds['mega']:
                influencer_posts['mega'].append(post)
            elif estimated_followers >= self.influencer_thresholds['macro']:
                influencer_posts['macro'].append(post)
            elif estimated_followers >= self.influencer_thresholds['mid']:
                influencer_posts['mid'].append(post)
            elif estimated_followers >= self.influencer_thresholds['micro']:
                influencer_posts['micro'].append(post)
        
        # Calculate weighted sentiment by influencer tier
        impact = {
            'mega_sentiment': np.mean([p.sentiment for p in influencer_posts['mega']]) if influencer_posts['mega'] else 0,
            'macro_sentiment': np.mean([p.sentiment for p in influencer_posts['macro']]) if influencer_posts['macro'] else 0,
            'mid_sentiment': np.mean([p.sentiment for p in influencer_posts['mid']]) if influencer_posts['mid'] else 0,
            'micro_sentiment': np.mean([p.sentiment for p in influencer_posts['micro']]) if influencer_posts['micro'] else 0,
            'mega_count': len(influencer_posts['mega']),
            'total_influencer_posts': sum(len(posts) for posts in influencer_posts.values())
        }
        
        # Calculate overall influencer impact
        weighted_impact = (
            impact['mega_sentiment'] * 1.0 +
            impact['macro_sentiment'] * 0.7 +
            impact['mid_sentiment'] * 0.4 +
            impact['micro_sentiment'] * 0.2
        ) / 2.3  # Normalize
        
        impact['overall_impact'] = weighted_impact
        
        return impact
    
    def _calculate_viral_metrics(
        self,
        posts: List[SocialPost],
        symbol: str
    ) -> Dict[str, Any]:
        """Calculate viral metrics for social posts"""
        
        if not posts:
            return {
                'is_viral': False,
                'viral_score': 0.0,
                'engagement_rate': 0.0,
                'share_velocity': 0.0,
                'mention_spike': 0.0
            }
        
        # Calculate engagement rate
        total_engagement = sum(p.engagement for p in posts)
        avg_engagement = total_engagement / len(posts)
        
        # Estimate reach (simplified)
        total_reach = sum(p.influence_score * 10000 for p in posts)
        engagement_rate = total_engagement / total_reach if total_reach > 0 else 0
        
        # Calculate share velocity (posts per hour)
        time_span = (
            max(p.timestamp for p in posts) - 
            min(p.timestamp for p in posts)
        ).total_seconds() / 3600
        
        share_velocity = len(posts) / time_span if time_span > 0 else len(posts)
        
        # Compare to baseline
        baseline = self.baseline_activity[symbol]
        if baseline['avg_mentions'] > 0:
            mention_spike = len(posts) / baseline['avg_mentions']
        else:
            mention_spike = 1.0
        
        # Update baseline (moving average)
        alpha = 0.1  # Smoothing factor
        baseline['avg_mentions'] = alpha * len(posts) + (1 - alpha) * baseline['avg_mentions']
        baseline['avg_engagement'] = alpha * avg_engagement + (1 - alpha) * baseline['avg_engagement']
        
        # Determine if viral
        is_viral = (
            engagement_rate > self.viral_thresholds['engagement_rate'] or
            share_velocity > self.viral_thresholds['share_velocity'] or
            mention_spike > self.viral_thresholds['mention_spike']
        )
        
        # Calculate viral score
        viral_score = min(
            (engagement_rate / self.viral_thresholds['engagement_rate']) * 0.3 +
            (share_velocity / self.viral_thresholds['share_velocity']) * 0.3 +
            (mention_spike / self.viral_thresholds['mention_spike']) * 0.4,
            1.0
        )
        
        return {
            'is_viral': is_viral,
            'viral_score': viral_score,
            'engagement_rate': engagement_rate,
            'share_velocity': share_velocity,
            'mention_spike': mention_spike
        }
    
    def _detect_sentiment_shift(
        self,
        posts: List[SocialPost],
        symbol: str
    ) -> Dict[str, Any]:
        """Detect sudden sentiment shifts"""
        
        if len(posts) < 10:
            return {
                'shift_detected': False,
                'shift_magnitude': 0.0,
                'shift_direction': 'neutral'
            }
        
        # Sort posts by time
        sorted_posts = sorted(posts, key=lambda p: p.timestamp)
        
        # Split into time windows
        mid_point = len(sorted_posts) // 2
        earlier_posts = sorted_posts[:mid_point]
        recent_posts = sorted_posts[mid_point:]
        
        # Calculate sentiment for each period
        earlier_sentiment = np.mean([p.sentiment for p in earlier_posts])
        recent_sentiment = np.mean([p.sentiment for p in recent_posts])
        
        # Calculate shift
        shift_magnitude = abs(recent_sentiment - earlier_sentiment)
        shift_direction = 'bullish' if recent_sentiment > earlier_sentiment else 'bearish'
        
        # Significant shift threshold
        shift_detected = shift_magnitude > 0.3
        
        return {
            'shift_detected': shift_detected,
            'shift_magnitude': shift_magnitude,
            'shift_direction': shift_direction,
            'earlier_sentiment': earlier_sentiment,
            'recent_sentiment': recent_sentiment
        }
    
    def _analyze_crowd_behavior(self, posts: List[SocialPost]) -> Dict[str, Any]:
        """Analyze crowd psychology patterns"""
        
        if not posts:
            return {
                'herding_score': 0.0,
                'contrarian_score': 0.0,
                'fomo_score': 0.0,
                'panic_score': 0.0
            }
        
        # Herding behavior (everyone saying the same thing)
        sentiments = [p.sentiment for p in posts]
        sentiment_std = np.std(sentiments)
        herding_score = max(0, 1 - sentiment_std * 2)  # Low std = high herding
        
        # Contrarian indicators
        extreme_bullish = sum(1 for s in sentiments if s > 0.7) / len(sentiments)
        extreme_bearish = sum(1 for s in sentiments if s < -0.7) / len(sentiments)
        contrarian_score = max(extreme_bullish, extreme_bearish)
        
        # FOMO indicators
        fomo_keywords = ['missing out', 'too late', 'already up', 'wish i bought', 'fomo']
        fomo_count = sum(
            1 for p in posts 
            if any(keyword in p.content.lower() for keyword in fomo_keywords)
        )
        fomo_score = min(fomo_count / len(posts), 1.0)
        
        # Panic indicators
        panic_keywords = ['crash', 'dump', 'sell everything', 'get out', 'blood bath', 'panic']
        panic_count = sum(
            1 for p in posts
            if any(keyword in p.content.lower() for keyword in panic_keywords)
        )
        panic_score = min(panic_count / len(posts), 1.0)
        
        return {
            'herding_score': herding_score,
            'contrarian_score': contrarian_score,
            'fomo_score': fomo_score,
            'panic_score': panic_score
        }
    
    def _generate_signal_from_social(
        self,
        analysis: Dict,
        meme_score: float,
        influencer_impact: Dict,
        viral_metrics: Dict,
        sentiment_shift: Dict,
        crowd_behavior: Dict
    ) -> Dict[str, Any]:
        """Generate trading signal from social analysis"""
        
        sentiment = analysis['weighted_sentiment']
        
        # Base signal on sentiment
        if sentiment > 0.6:
            base_action = 'strong_buy'
        elif sentiment > 0.2:
            base_action = 'buy'
        elif sentiment < -0.6:
            base_action = 'strong_sell'
        elif sentiment < -0.2:
            base_action = 'sell'
        else:
            base_action = 'hold'
        
        # Adjust for meme activity (be cautious)
        if meme_score > 0.7:
            if base_action == 'strong_buy':
                base_action = 'buy'  # Reduce strength
            elif base_action == 'buy':
                base_action = 'hold'  # Be more cautious
        
        # Adjust for viral metrics
        if viral_metrics['is_viral']:
            if viral_metrics['viral_score'] > 0.7:
                # Very viral - potential for reversal
                if base_action in ['strong_buy', 'strong_sell']:
                    base_action = base_action.replace('strong_', '')
        
        # Adjust for sentiment shift
        if sentiment_shift['shift_detected']:
            if sentiment_shift['shift_direction'] == 'bullish' and base_action in ['hold', 'buy']:
                base_action = 'buy' if base_action == 'hold' else 'strong_buy'
            elif sentiment_shift['shift_direction'] == 'bearish' and base_action in ['hold', 'sell']:
                base_action = 'sell' if base_action == 'hold' else 'strong_sell'
        
        # Adjust for crowd behavior
        if crowd_behavior['panic_score'] > 0.5:
            # Panic selling might be opportunity
            if base_action == 'sell':
                base_action = 'hold'
        elif crowd_behavior['fomo_score'] > 0.5:
            # FOMO buying might be top
            if base_action == 'buy':
                base_action = 'hold'
        
        # Contrarian adjustments
        if crowd_behavior['contrarian_score'] > 0.7:
            # Extreme sentiment often reverses
            if base_action == 'strong_buy':
                base_action = 'hold'
            elif base_action == 'strong_sell':
                base_action = 'hold'
        
        return {
            'action': base_action,
            'sentiment': sentiment,
            'meme_adjusted': meme_score > 0.5
        }
    
    def _calculate_confidence(
        self,
        analysis: Dict,
        viral_metrics: Dict,
        crowd_behavior: Dict,
        post_count: int
    ) -> float:
        """Calculate confidence in social signal"""
        
        base_confidence = 0.4  # Lower base for social data
        
        # Factor 1: Post volume
        volume_confidence = min(post_count / 50, 1.0) * 0.15
        
        # Factor 2: Author diversity
        author_confidence = min(analysis['unique_authors'] / 20, 1.0) * 0.15
        
        # Factor 3: Platform consensus
        platform_sentiments = list(analysis['platform_sentiment'].values())
        if platform_sentiments:
            sentiment_std = np.std(platform_sentiments)
            consensus_confidence = max(0, (1 - sentiment_std) * 0.15)
        else:
            consensus_confidence = 0.0
        
        # Factor 4: Engagement level
        engagement_confidence = min(analysis['total_engagement'] / 10000, 1.0) * 0.1
        
        # Factor 5: Not too viral (moderate is best)
        if 0.2 < viral_metrics['viral_score'] < 0.6:
            viral_confidence = 0.1
        else:
            viral_confidence = 0.05
        
        # Factor 6: Low herding behavior
        herding_penalty = crowd_behavior['herding_score'] * 0.1
        
        confidence = (
            base_confidence +
            volume_confidence +
            author_confidence +
            consensus_confidence +
            engagement_confidence +
            viral_confidence -
            herding_penalty
        )
        
        return min(max(confidence, 0.1), 0.85)  # Cap at 85% for social data
    
    def _generate_reasoning(
        self,
        analysis: Dict,
        meme_score: float,
        viral_metrics: Dict,
        sentiment_shift: Dict,
        signal_data: Dict
    ) -> List[str]:
        """Generate reasoning for the signal"""
        
        reasoning = []
        
        # Overall sentiment
        sentiment = analysis['weighted_sentiment']
        if abs(sentiment) > 0.5:
            direction = "strongly bullish" if sentiment > 0 else "strongly bearish"
        else:
            direction = "mixed"
        
        reasoning.append(f"Social sentiment is {direction} ({sentiment:.2f})")
        
        # Platform breakdown
        platform_sentiments = analysis['platform_sentiment']
        if platform_sentiments:
            dominant_platform = max(platform_sentiments, key=lambda k: abs(platform_sentiments[k]))
            reasoning.append(f"Strongest signal from {dominant_platform}")
        
        # Meme activity
        if meme_score > 0.5:
            reasoning.append(f"High meme stock activity detected ({meme_score:.2f})")
        
        # Viral status
        if viral_metrics['is_viral']:
            reasoning.append(f"Content going viral (score: {viral_metrics['viral_score']:.2f})")
        
        # Sentiment shift
        if sentiment_shift['shift_detected']:
            reasoning.append(f"Sentiment shifting {sentiment_shift['shift_direction']}")
        
        return reasoning[:5]
    
    def _identify_risks(
        self,
        analysis: Dict,
        meme_score: float,
        crowd_behavior: Dict,
        viral_metrics: Dict
    ) -> List[str]:
        """Identify risks from social analysis"""
        
        risks = []
        
        # Meme stock risk
        if meme_score > 0.6:
            risks.append("High meme stock activity - increased volatility")
        
        # Herding behavior
        if crowd_behavior['herding_score'] > 0.7:
            risks.append("Strong herding behavior detected")
        
        # Panic or FOMO
        if crowd_behavior['panic_score'] > 0.5:
            risks.append("Panic selling detected")
        elif crowd_behavior['fomo_score'] > 0.5:
            risks.append("FOMO buying detected")
        
        # Too viral
        if viral_metrics['viral_score'] > 0.8:
            risks.append("Extremely viral - potential pump and dump")
        
        # Low author diversity
        if analysis['unique_authors'] < 10:
            risks.append("Low author diversity - possible manipulation")
        
        return risks[:4]
    
    def _format_top_posts(self, posts: List[SocialPost]) -> List[Dict]:
        """Format top posts for display"""
        
        formatted = []
        
        for post in posts:
            formatted.append({
                'platform': post.platform,
                'author': post.author[:20],
                'content': post.content[:100],
                'sentiment': round(post.sentiment, 2),
                'engagement': post.engagement,
                'time': post.timestamp.isoformat()
            })
        
        return formatted
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for social sentiment analysis
        
        Returns:
            List of data type strings
        """
        return [
            'social',  # Primary requirement - social media posts
            'symbol',  # Stock symbol
            'price'   # Current price for context
        ]


# Create global instance
social_sentiment_agent = SocialSentimentAgent()