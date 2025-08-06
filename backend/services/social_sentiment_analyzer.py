"""
Social Media Sentiment Analyzer
Comprehensive sentiment analysis from multiple social media platforms
"""

import asyncio
import aiohttp
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from textblob import TextBlob
import praw  # Reddit API
import tweepy  # X/Twitter API
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    """Sentiment classification"""
    VERY_BULLISH = 1.0
    BULLISH = 0.5
    NEUTRAL = 0.0
    BEARISH = -0.5
    VERY_BEARISH = -1.0


@dataclass
class SocialPost:
    """Individual social media post"""
    platform: str
    content: str
    author: str
    timestamp: datetime
    engagement: int  # likes + shares + comments
    sentiment_score: float
    confidence: float
    url: Optional[str] = None
    ticker_mentions: List[str] = None


@dataclass
class PlatformSentiment:
    """Sentiment data for a specific platform"""
    platform: str
    posts_analyzed: int
    average_sentiment: float
    sentiment_distribution: Dict[str, int]
    top_bullish_posts: List[SocialPost]
    top_bearish_posts: List[SocialPost]
    trending_topics: List[str]
    influencer_sentiment: float
    volume_change: float  # % change in mentions


class SocialMediaPlatform:
    """Base class for social media platforms"""
    
    def __init__(self, name: str):
        self.name = name
        self.rate_limit_remaining = 100
        self.last_request_time = datetime.utcnow()
        
    async def search_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search for posts mentioning the query"""
        raise NotImplementedError
        
    async def get_trending_topics(self) -> List[str]:
        """Get trending topics related to finance/stocks"""
        raise NotImplementedError


class TwitterPlatform(SocialMediaPlatform):
    """X/Twitter integration"""
    
    def __init__(self):
        super().__init__("Twitter/X")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.client = None
        if self.bearer_token:
            self.client = tweepy.Client(bearer_token=self.bearer_token)
            
    async def search_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search Twitter for posts"""
        if not self.client:
            return []
            
        try:
            # Search recent tweets
            tweets = self.client.search_recent_tweets(
                query=f"{query} -is:retweet lang:en",
                max_results=min(limit, 100),
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'entities']
            )
            
            posts = []
            if tweets.data:
                for tweet in tweets.data:
                    posts.append({
                        'content': tweet.text,
                        'author_id': tweet.author_id,
                        'timestamp': tweet.created_at,
                        'engagement': sum([
                            tweet.public_metrics.get('like_count', 0),
                            tweet.public_metrics.get('retweet_count', 0),
                            tweet.public_metrics.get('reply_count', 0)
                        ]),
                        'url': f"https://twitter.com/i/status/{tweet.id}"
                    })
            return posts
        except Exception as e:
            logger.error(f"Twitter search error: {e}")
            return []


class RedditPlatform(SocialMediaPlatform):
    """Reddit integration"""
    
    def __init__(self):
        super().__init__("Reddit")
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit = None
        
        if self.client_id and self.client_secret:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent="GoldenSignalsAI/1.0"
            )
            
    async def search_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search Reddit for posts and comments"""
        if not self.reddit:
            return []
            
        posts = []
        try:
            # Search in relevant subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'options', 'StockMarket']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search posts
                for submission in subreddit.search(query, limit=limit//len(subreddits)):
                    posts.append({
                        'content': f"{submission.title} {submission.selftext}",
                        'author': str(submission.author) if submission.author else 'deleted',
                        'timestamp': datetime.fromtimestamp(submission.created_utc),
                        'engagement': submission.score + submission.num_comments,
                        'url': f"https://reddit.com{submission.permalink}"
                    })
                    
                    # Also get top comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:5]:
                        if hasattr(comment, 'body'):
                            posts.append({
                                'content': comment.body,
                                'author': str(comment.author) if comment.author else 'deleted',
                                'timestamp': datetime.fromtimestamp(comment.created_utc),
                                'engagement': comment.score,
                                'url': f"https://reddit.com{comment.permalink}"
                            })
                            
            return posts
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
            return []


class DiscordPlatform(SocialMediaPlatform):
    """Discord integration (via webhooks and public channels)"""
    
    def __init__(self):
        super().__init__("Discord")
        self.webhook_urls = os.getenv("DISCORD_WEBHOOKS", "").split(",")
        
    async def search_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Monitor Discord channels for stock mentions"""
        # Note: This would require Discord bot implementation
        # For now, return empty - would need proper Discord bot setup
        return []


class SocialSentimentAnalyzer:
    """Main sentiment analyzer combining all platforms"""
    
    def __init__(self):
        # Initialize platforms
        self.platforms = {
            'twitter': TwitterPlatform(),
            'reddit': RedditPlatform(),
            'discord': DiscordPlatform(),
        }
        
        # Initialize sentiment models
        self.finbert_model = None
        self.sentiment_pipeline = None
        self._init_models()
        
        # Sentiment keywords
        self.bullish_keywords = [
            'bullish', 'buy', 'long', 'calls', 'moon', 'rocket', 'squeeze',
            'breakout', 'upgrade', 'strong', 'growth', 'gains', 'rally',
            'pump', 'diamond hands', 'hodl', 'to the moon', 'tendies',
            'green', 'soaring', 'skyrocket', 'explosive', 'accumulate'
        ]
        
        self.bearish_keywords = [
            'bearish', 'sell', 'short', 'puts', 'crash', 'dump', 'tank',
            'breakdown', 'downgrade', 'weak', 'decline', 'losses', 'plunge',
            'overvalued', 'bubble', 'correction', 'resist', 'red', 'falling',
            'collapse', 'bankruptcy', 'delisted', 'worthless'
        ]
        
        # Influencer list (users with high following/impact)
        self.influencers = {
            'twitter': ['elonmusk', 'jimcramer', 'stoolpresidente', 'chamath'],
            'reddit': ['DeepFuckingValue', 'SIR_JACK_A_LOT'],
        }
        
    def _init_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Try to load FinBERT for financial sentiment
            self.finbert_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU
            )
        except Exception as e:
            logger.warning(f"Could not load FinBERT model: {e}")
            
        # Fallback to general sentiment
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {e}")
            
    async def analyze_symbol_sentiment(
        self, 
        symbol: str,
        platforms: List[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze sentiment for a specific symbol across platforms"""
        
        if platforms is None:
            platforms = list(self.platforms.keys())
            
        # Collect posts from all platforms
        all_posts = []
        platform_results = {}
        
        for platform_name in platforms:
            if platform_name in self.platforms:
                platform = self.platforms[platform_name]
                
                # Search for symbol mentions
                queries = [
                    f"${symbol}",
                    f"#{symbol}",
                    symbol,
                    f"{symbol} stock",
                    f"{symbol} calls",
                    f"{symbol} puts"
                ]
                
                platform_posts = []
                for query in queries:
                    posts = await platform.search_posts(query, limit=50)
                    platform_posts.extend(posts)
                    
                # Analyze posts
                analyzed_posts = []
                for post in platform_posts:
                    # Skip old posts
                    if isinstance(post.get('timestamp'), datetime):
                        if datetime.utcnow() - post['timestamp'] > timedelta(hours=time_window_hours):
                            continue
                            
                    sentiment = self._analyze_post_sentiment(post['content'])
                    
                    social_post = SocialPost(
                        platform=platform_name,
                        content=post['content'],
                        author=post.get('author', 'unknown'),
                        timestamp=post.get('timestamp', datetime.utcnow()),
                        engagement=post.get('engagement', 0),
                        sentiment_score=sentiment['score'],
                        confidence=sentiment['confidence'],
                        url=post.get('url'),
                        ticker_mentions=[symbol]
                    )
                    
                    analyzed_posts.append(social_post)
                    all_posts.append(social_post)
                    
                # Calculate platform metrics
                if analyzed_posts:
                    platform_sentiment = self._calculate_platform_sentiment(
                        platform_name, analyzed_posts
                    )
                    platform_results[platform_name] = platform_sentiment
                    
        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(all_posts)
        
        # Detect unusual activity
        anomalies = self._detect_sentiment_anomalies(all_posts)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'overall_sentiment': overall_sentiment,
            'platform_breakdown': platform_results,
            'total_posts_analyzed': len(all_posts),
            'anomalies': anomalies,
            'recommendation': self._generate_recommendation(overall_sentiment, anomalies)
        }
        
    def _analyze_post_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of a single post"""
        
        # Clean content
        content = re.sub(r'http\S+', '', content)  # Remove URLs
        content = re.sub(r'[^a-zA-Z0-9\s$#]', ' ', content)  # Keep only alphanumeric
        
        # Keyword-based sentiment
        content_lower = content.lower()
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in content_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in content_lower)
        
        keyword_sentiment = 0.0
        if bullish_count > bearish_count:
            keyword_sentiment = min(bullish_count / 10, 1.0)
        elif bearish_count > bullish_count:
            keyword_sentiment = -min(bearish_count / 10, 1.0)
            
        # Model-based sentiment
        model_sentiment = 0.0
        confidence = 0.5
        
        if self.finbert_model and len(content.split()) > 3:
            try:
                result = self.finbert_model(content[:512])[0]  # Truncate for model
                
                if result['label'] == 'positive':
                    model_sentiment = result['score']
                elif result['label'] == 'negative':
                    model_sentiment = -result['score']
                    
                confidence = result['score']
            except:
                pass
                
        elif self.sentiment_pipeline and len(content.split()) > 3:
            try:
                blob = TextBlob(content)
                model_sentiment = blob.sentiment.polarity
                confidence = abs(model_sentiment)
            except:
                pass
                
        # Combine keyword and model sentiment
        final_sentiment = (keyword_sentiment * 0.4 + model_sentiment * 0.6)
        
        return {
            'score': final_sentiment,
            'confidence': confidence
        }
        
    def _calculate_platform_sentiment(
        self, 
        platform: str, 
        posts: List[SocialPost]
    ) -> PlatformSentiment:
        """Calculate aggregate sentiment for a platform"""
        
        if not posts:
            return None
            
        # Calculate average sentiment weighted by engagement
        total_weight = sum(max(p.engagement, 1) for p in posts)
        weighted_sentiment = sum(
            p.sentiment_score * max(p.engagement, 1) for p in posts
        ) / total_weight if total_weight > 0 else 0
        
        # Sentiment distribution
        distribution = {
            'very_bullish': sum(1 for p in posts if p.sentiment_score > 0.7),
            'bullish': sum(1 for p in posts if 0.2 < p.sentiment_score <= 0.7),
            'neutral': sum(1 for p in posts if -0.2 <= p.sentiment_score <= 0.2),
            'bearish': sum(1 for p in posts if -0.7 <= p.sentiment_score < -0.2),
            'very_bearish': sum(1 for p in posts if p.sentiment_score < -0.7)
        }
        
        # Top posts
        sorted_posts = sorted(posts, key=lambda p: p.engagement, reverse=True)
        top_bullish = [p for p in sorted_posts if p.sentiment_score > 0.2][:5]
        top_bearish = [p for p in sorted_posts if p.sentiment_score < -0.2][:5]
        
        # Influencer sentiment
        influencer_posts = [
            p for p in posts 
            if p.author.lower() in self.influencers.get(platform, [])
        ]
        influencer_sentiment = np.mean([p.sentiment_score for p in influencer_posts]) if influencer_posts else weighted_sentiment
        
        # Extract trending topics
        all_text = ' '.join([p.content.lower() for p in posts])
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in ['stock', 'calls', 'puts']:
                word_freq[word] = word_freq.get(word, 0) + 1
        trending = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return PlatformSentiment(
            platform=platform,
            posts_analyzed=len(posts),
            average_sentiment=weighted_sentiment,
            sentiment_distribution=distribution,
            top_bullish_posts=top_bullish,
            top_bearish_posts=top_bearish,
            trending_topics=[word for word, _ in trending],
            influencer_sentiment=influencer_sentiment,
            volume_change=0.0  # Would need historical data to calculate
        )
        
    def _calculate_overall_sentiment(self, posts: List[SocialPost]) -> Dict[str, Any]:
        """Calculate overall sentiment metrics"""
        
        if not posts:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'signal': 'NEUTRAL'
            }
            
        # Weight by engagement and recency
        now = datetime.utcnow()
        weighted_scores = []
        weights = []
        
        for post in posts:
            # Recency weight (exponential decay)
            hours_old = (now - post.timestamp).total_seconds() / 3600
            recency_weight = np.exp(-hours_old / 24)  # Half-life of 24 hours
            
            # Engagement weight (logarithmic)
            engagement_weight = np.log1p(post.engagement)
            
            # Combined weight
            weight = recency_weight * engagement_weight * post.confidence
            
            weighted_scores.append(post.sentiment_score * weight)
            weights.append(weight)
            
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = 0.0
            
        # Determine signal strength
        if overall_score > 0.5:
            signal = 'STRONG_BUY'
        elif overall_score > 0.2:
            signal = 'BUY'
        elif overall_score < -0.5:
            signal = 'STRONG_SELL'
        elif overall_score < -0.2:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
            
        # Calculate confidence based on volume and agreement
        volume_confidence = min(len(posts) / 100, 1.0)  # More posts = more confidence
        
        # Agreement (low variance = high agreement)
        if len(posts) > 1:
            sentiment_variance = np.var([p.sentiment_score for p in posts])
            agreement_confidence = max(0, 1 - sentiment_variance)
        else:
            agreement_confidence = 0.5
            
        overall_confidence = (volume_confidence + agreement_confidence) / 2
        
        return {
            'score': overall_score,
            'confidence': overall_confidence,
            'signal': signal,
            'bullish_ratio': sum(1 for p in posts if p.sentiment_score > 0.2) / len(posts),
            'bearish_ratio': sum(1 for p in posts if p.sentiment_score < -0.2) / len(posts)
        }
        
    def _detect_sentiment_anomalies(self, posts: List[SocialPost]) -> List[Dict[str, Any]]:
        """Detect unusual sentiment patterns"""
        
        anomalies = []
        
        if not posts:
            return anomalies
            
        # Sudden spike in volume
        posts_by_hour = {}
        for post in posts:
            hour = post.timestamp.replace(minute=0, second=0, microsecond=0)
            posts_by_hour[hour] = posts_by_hour.get(hour, 0) + 1
            
        if len(posts_by_hour) > 2:
            volumes = list(posts_by_hour.values())
            avg_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            for hour, volume in posts_by_hour.items():
                if volume > avg_volume + 2 * std_volume:
                    anomalies.append({
                        'type': 'volume_spike',
                        'timestamp': hour.isoformat(),
                        'severity': 'high',
                        'details': f'Unusual volume: {volume} posts (avg: {avg_volume:.1f})'
                    })
                    
        # Coordinated activity (many similar posts)
        content_hashes = {}
        for post in posts:
            # Simple hash of content
            content_hash = hash(post.content.lower().strip())
            if content_hash in content_hashes:
                content_hashes[content_hash].append(post)
            else:
                content_hashes[content_hash] = [post]
                
        for content_hash, similar_posts in content_hashes.items():
            if len(similar_posts) > 5:
                anomalies.append({
                    'type': 'coordinated_activity',
                    'timestamp': datetime.utcnow().isoformat(),
                    'severity': 'medium',
                    'details': f'Found {len(similar_posts)} similar posts'
                })
                
        # Influencer activity
        influencer_posts = [
            p for p in posts 
            if any(p.author.lower() in self.influencers.get(p.platform, []))
        ]
        
        if influencer_posts:
            anomalies.append({
                'type': 'influencer_activity',
                'timestamp': datetime.utcnow().isoformat(),
                'severity': 'medium',
                'details': f'{len(influencer_posts)} posts from influencers'
            })
            
        return anomalies
        
    def _generate_recommendation(
        self, 
        overall_sentiment: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate trading recommendation based on sentiment"""
        
        # Base recommendation on sentiment
        sentiment_score = overall_sentiment['score']
        confidence = overall_sentiment['confidence']
        
        # Adjust for anomalies
        if any(a['type'] == 'coordinated_activity' for a in anomalies):
            confidence *= 0.7  # Reduce confidence if manipulation suspected
            
        if any(a['type'] == 'volume_spike' for a in anomalies):
            confidence *= 1.2  # Increase confidence if high interest
            
        # Generate recommendation
        if sentiment_score > 0.5 and confidence > 0.7:
            action = 'STRONG_BUY'
            rationale = 'Strong positive sentiment with high confidence'
        elif sentiment_score > 0.2 and confidence > 0.5:
            action = 'BUY'
            rationale = 'Positive sentiment detected'
        elif sentiment_score < -0.5 and confidence > 0.7:
            action = 'STRONG_SELL'
            rationale = 'Strong negative sentiment with high confidence'
        elif sentiment_score < -0.2 and confidence > 0.5:
            action = 'SELL'
            rationale = 'Negative sentiment detected'
        else:
            action = 'HOLD'
            rationale = 'Mixed or neutral sentiment'
            
        return {
            'action': action,
            'confidence': min(confidence, 1.0),
            'rationale': rationale,
            'risk_level': 'high' if confidence < 0.5 else 'medium' if confidence < 0.8 else 'low'
        }


# Global instance
social_sentiment_analyzer = SocialSentimentAnalyzer()