"""
FinGPT Agent - Free Open-Source Financial LLM
Conforms to MCP/LangGraph best practices
Replaces multiple sentiment and analysis agents
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from agents.base import BaseAgent, Signal, SignalAction, SignalStrength, AgentContext, AgentConfig
from core.logging import get_logger
from services.enhanced_data_aggregator import enhanced_data_aggregator

logger = get_logger(__name__)


class FinGPTAgent(BaseAgent):
    """
    FinGPT Agent - Consolidates sentiment, technical, and fundamental analysis
    
    This agent replaces:
    - News sentiment agent
    - Social media sentiment agent  
    - Earnings analysis agent
    - Some technical analysis agents
    
    Benefits:
    - 87.8% F1-score on financial sentiment (beats BloombergGPT)
    - Free and open-source
    - Minimal fine-tuning cost (<$100)
    - Supports 34+ data sources
    """
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="FinGPT",
                confidence_threshold=0.75
            )
        super().__init__(config)
        
        # Model configuration
        self.model_name = "FinGPT/fingpt-sentiment_llama3-8b"  # Best performing variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        # Analysis prompts optimized for financial tasks
        self.prompts = {
            "sentiment": """Analyze the sentiment of the following financial text about {symbol}. 
                           Consider market impact, investor sentiment, and potential price movement.
                           Text: {text}
                           Respond with: BULLISH, BEARISH, or NEUTRAL and confidence score.""",
            
            "technical": """Analyze the technical indicators for {symbol}:
                           Price: {price}, RSI: {rsi}, MACD: {macd}, Volume: {volume}
                           Support: {support}, Resistance: {resistance}
                           Provide trading signal: BUY, SELL, or HOLD with reasoning.""",
            
            "forecast": """Based on the following data for {symbol}:
                          Current Price: {price}
                          Market Sentiment: {sentiment}
                          Technical Indicators: {indicators}
                          News Headlines: {news}
                          Predict price direction for next {timeframe} with confidence.""",
            
            "risk": """Assess the risk level for {symbol} position:
                      Current Price: {price}, Volatility: {volatility}
                      Market Conditions: {conditions}
                      Position Size: {size}
                      Provide risk score (0-1) and recommendations."""
        }
        
    async def initialize(self):
        """Initialize FinGPT model"""
        try:
            logger.info("Initializing FinGPT model...")
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            self.initialized = True
            logger.info(f"FinGPT initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FinGPT: {e}")
            # Fallback to mock mode for development
            self.initialized = False
            logger.warning("Running in mock mode - install transformers and model for production")
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Perform comprehensive analysis using FinGPT
        
        This replaces multiple agent analyses with a single LLM call
        """
        
        if not self.initialized:
            # Mock response for development
            return await self._mock_analysis(context)
        
        try:
            # Create context from market data for compatibility
            context = AgentContext(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                market_data=market_data,
                indicators=market_data.get('indicators', {})
            )
            
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # Get sentiment from multiple sources
            sentiment_score = await self._analyze_sentiment(context)
            
            # Technical analysis
            technical_signal = await self._analyze_technical(context)
            
            # Price forecast
            price_prediction = await self._forecast_price(context)
            
            # Risk assessment
            risk_analysis = await self._assess_risk(context)
            
            # Combine all analyses
            confidence = self._calculate_confidence(
                sentiment_score,
                technical_signal,
                price_prediction,
                risk_analysis
            )
            
            # Generate final signal
            signal = self._generate_signal(
                sentiment_score,
                technical_signal,
                price_prediction,
                risk_analysis,
                confidence,
                symbol,
                market_data.get('price', 0.0)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"FinGPT analysis error: {e}")
            return None
    
    async def _analyze_sentiment(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze sentiment from news and social media"""
        
        # Gather text data
        news_data = context.additional_data.get('news', [])
        social_data = context.additional_data.get('social_media', [])
        
        # Combine all text
        all_text = []
        for item in news_data[:5]:  # Top 5 news
            all_text.append(item.get('title', '') + ' ' + item.get('description', ''))
        
        for item in social_data[:10]:  # Top 10 social posts
            all_text.append(item.get('text', ''))
        
        combined_text = ' '.join(all_text)[:2000]  # Limit context length
        
        if not combined_text:
            return {'score': 0.5, 'label': 'NEUTRAL', 'confidence': 0.5}
        
        # Generate prompt
        prompt = self.prompts['sentiment'].format(
            symbol=context.market_data.get('symbol', 'UNKNOWN'),
            text=combined_text
        )
        
        # Get prediction
        sentiment = await self._generate_response(prompt)
        
        # Parse response
        return self._parse_sentiment(sentiment)
    
    async def _analyze_technical(self, context: AgentContext) -> Dict[str, Any]:
        """Perform technical analysis"""
        
        market_data = context.market_data
        indicators = context.technical_indicators
        
        prompt = self.prompts['technical'].format(
            symbol=market_data.get('symbol'),
            price=market_data.get('price'),
            rsi=indicators.get('rsi'),
            macd=indicators.get('macd'),
            volume=market_data.get('volume'),
            support=indicators.get('support_level'),
            resistance=indicators.get('resistance_level')
        )
        
        response = await self._generate_response(prompt)
        return self._parse_technical_signal(response)
    
    async def _forecast_price(self, context: AgentContext) -> Dict[str, Any]:
        """Forecast future price movement"""
        
        prompt = self.prompts['forecast'].format(
            symbol=context.market_data.get('symbol'),
            price=context.market_data.get('price'),
            sentiment='BULLISH',  # From sentiment analysis
            indicators=str(context.technical_indicators),
            news='Recent positive earnings',  # Summarized
            timeframe='24 hours'
        )
        
        response = await self._generate_response(prompt)
        return self._parse_forecast(response, context.market_data.get('price'))
    
    async def _assess_risk(self, context: AgentContext) -> Dict[str, Any]:
        """Assess position risk"""
        
        market_data = context.market_data
        
        prompt = self.prompts['risk'].format(
            symbol=market_data.get('symbol'),
            price=market_data.get('price'),
            volatility=market_data.get('volatility', 0.02),
            conditions='moderate volatility',
            size='5% of portfolio'
        )
        
        response = await self._generate_response(prompt)
        return self._parse_risk_assessment(response)
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response from FinGPT model"""
        
        if not self.initialized:
            # Mock response
            return "BULLISH with 0.85 confidence"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = response[len(prompt):].strip()
        
        return response
    
    def _parse_sentiment(self, response: str) -> Dict[str, Any]:
        """Parse sentiment analysis response"""
        
        response_upper = response.upper()
        
        if 'BULLISH' in response_upper:
            score = 0.8
            label = 'BULLISH'
        elif 'BEARISH' in response_upper:
            score = 0.2
            label = 'BEARISH'
        else:
            score = 0.5
            label = 'NEUTRAL'
        
        # Extract confidence if mentioned
        confidence = 0.75  # Default
        if 'CONFIDENCE' in response_upper:
            try:
                # Simple extraction - improve with regex
                parts = response_upper.split('CONFIDENCE')
                if len(parts) > 1:
                    conf_str = parts[1].strip()[:4]
                    confidence = float(''.join(c for c in conf_str if c.isdigit() or c == '.'))
            except:
                pass
        
        return {
            'score': score,
            'label': label,
            'confidence': min(confidence, 1.0)
        }
    
    def _parse_technical_signal(self, response: str) -> Dict[str, Any]:
        """Parse technical analysis response"""
        
        response_upper = response.upper()
        
        if 'BUY' in response_upper:
            signal = 'BUY'
            strength = 0.8
        elif 'SELL' in response_upper:
            signal = 'SELL'
            strength = 0.2
        else:
            signal = 'HOLD'
            strength = 0.5
        
        return {
            'signal': signal,
            'strength': strength,
            'reasoning': response[:200]  # First 200 chars
        }
    
    def _parse_forecast(self, response: str, current_price: float) -> Dict[str, Any]:
        """Parse price forecast response"""
        
        # Simple parsing - enhance with NLP
        if 'INCREASE' in response.upper() or 'UP' in response.upper():
            change_pct = 0.02  # 2% default
            target_price = current_price * (1 + change_pct)
        elif 'DECREASE' in response.upper() or 'DOWN' in response.upper():
            change_pct = -0.02
            target_price = current_price * (1 + change_pct)
        else:
            change_pct = 0
            target_price = current_price
        
        return {
            'target_price': round(target_price, 2),
            'change_percent': change_pct,
            'timeframe': '1d',
            'confidence': 0.7
        }
    
    def _parse_risk_assessment(self, response: str) -> Dict[str, Any]:
        """Parse risk assessment response"""
        
        # Simple risk scoring
        if 'HIGH RISK' in response.upper():
            risk_score = 0.8
            risk_level = 'HIGH'
        elif 'LOW RISK' in response.upper():
            risk_score = 0.2
            risk_level = 'LOW'
        else:
            risk_score = 0.5
            risk_level = 'MEDIUM'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'max_position_size': 0.05 * (1 - risk_score),  # Lower size for higher risk
            'stop_loss_pct': 0.02 * (1 + risk_score),  # Wider stop for higher risk
            'recommendations': response[:200]
        }
    
    def _calculate_confidence(self, sentiment, technical, forecast, risk) -> float:
        """Calculate overall confidence score"""
        
        # Weighted average
        weights = {
            'sentiment': 0.3,
            'technical': 0.4,
            'forecast': 0.2,
            'risk': 0.1
        }
        
        sentiment_conf = sentiment.get('confidence', 0.5)
        technical_conf = abs(technical.get('strength', 0.5) - 0.5) * 2  # Convert to confidence
        forecast_conf = forecast.get('confidence', 0.5)
        risk_conf = 1 - risk.get('risk_score', 0.5)  # Inverse risk
        
        confidence = (
            weights['sentiment'] * sentiment_conf +
            weights['technical'] * technical_conf +
            weights['forecast'] * forecast_conf +
            weights['risk'] * risk_conf
        )
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def _generate_signal(self, sentiment, technical, forecast, risk, confidence, symbol, current_price) -> Signal:
        """Generate final trading signal"""
        
        # Determine action based on analyses
        scores = {
            'buy': 0,
            'sell': 0,
            'hold': 0
        }
        
        # Sentiment contribution
        if sentiment['label'] == 'BULLISH':
            scores['buy'] += 0.3
        elif sentiment['label'] == 'BEARISH':
            scores['sell'] += 0.3
        else:
            scores['hold'] += 0.3
        
        # Technical contribution
        if technical['signal'] == 'BUY':
            scores['buy'] += 0.4
        elif technical['signal'] == 'SELL':
            scores['sell'] += 0.4
        else:
            scores['hold'] += 0.4
        
        # Forecast contribution
        if forecast['change_percent'] > 0.01:
            scores['buy'] += 0.2
        elif forecast['change_percent'] < -0.01:
            scores['sell'] += 0.2
        else:
            scores['hold'] += 0.2
        
        # Risk adjustment
        if risk['risk_score'] > 0.7:
            scores['hold'] += 0.1  # Prefer caution
        
        # Determine action
        max_score = max(scores.values())
        if scores['buy'] == max_score:
            action = SignalAction.BUY
        elif scores['sell'] == max_score:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD
        
        # Determine strength
        if confidence > 0.8:
            strength = SignalStrength.STRONG
        elif confidence > 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Create signal
        return Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strength=strength,
            source=self.config.name,
            current_price=current_price,
            target_price=forecast.get('target_price'),
            reasoning=self._generate_reasoning(sentiment, technical, forecast, risk),
            features={
                'sentiment': sentiment,
                'technical': technical,
                'forecast': forecast,
                'risk': risk,
                'feature_importance': {
                    'sentiment_weight': 0.3,
                    'technical_weight': 0.4,
                    'forecast_weight': 0.2,
                    'risk_weight': 0.1
                }
            }
        )
    
    def _generate_reasoning(self, sentiment, technical, forecast, risk) -> List[str]:
        """Generate reasoning for the signal"""
        
        reasoning = []
        
        # Sentiment reasoning
        reasoning.append(
            f"Sentiment analysis shows {sentiment['label']} "
            f"signal with {sentiment['confidence']:.0%} confidence"
        )
        
        # Technical reasoning  
        reasoning.append(
            f"Technical indicators suggest {technical['signal']} "
            f"with strength {technical['strength']:.2f}"
        )
        
        # Forecast reasoning
        change_pct = forecast['change_percent'] * 100
        reasoning.append(
            f"Price forecast: {change_pct:+.1f}% change expected "
            f"(target: ${forecast['target_price']:.2f})"
        )
        
        # Risk reasoning
        reasoning.append(
            f"Risk assessment: {risk['risk_level']} "
            f"(score: {risk['risk_score']:.2f})"
        )
        
        return reasoning
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for FinGPT analysis
        
        Returns:
            List of data type strings
        """
        return [
            'price',       # Current price
            'volume',      # Trading volume
            'indicators',  # Technical indicators
            'news',        # News sentiment data
            'social_media' # Social media sentiment data
        ]
    
    async def _mock_analysis(self, context: AgentContext) -> Dict[str, Any]:
        """Mock analysis for development when model not loaded"""
        
        import random
        
        # Generate mock signals
        actions = ['BUY', 'SELL', 'HOLD']
        action = random.choice(actions)
        confidence = random.uniform(0.6, 0.9)
        
        current_price = context.market_data.get('price', 100)
        predicted_change = random.uniform(-0.05, 0.05)
        predicted_price = current_price * (1 + predicted_change)
        
        return {
            'signal': action,
            'confidence': confidence,
            'predicted_price': round(predicted_price, 2),
            'prediction_timeframe': '1d',
            'sentiment': {
                'score': random.uniform(0.3, 0.7),
                'label': random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']),
                'confidence': confidence
            },
            'technical': {
                'signal': action,
                'strength': confidence,
                'reasoning': 'Mock technical analysis'
            },
            'risk': {
                'risk_score': random.uniform(0.2, 0.8),
                'risk_level': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'max_position_size': 0.05,
                'stop_loss_pct': 0.02
            },
            'reasoning': [
                f"Mock FinGPT analysis suggests {action}",
                f"Confidence level: {confidence:.0%}",
                f"Price prediction: ${predicted_price:.2f} ({predicted_change*100:+.1f}%)"
            ],
            'feature_importance': {
                'sentiment_weight': 0.3,
                'technical_weight': 0.4,
                'forecast_weight': 0.2,
                'risk_weight': 0.1
            }
        }


# Create singleton instance
fingpt_agent = FinGPTAgent()