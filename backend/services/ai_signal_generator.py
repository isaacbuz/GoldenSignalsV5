"""
AI Signal Generator Service
Generates trading signals using multiple AI agents and technical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging

from services.market_data_service import market_data_service
from agents.orchestrator import AgentOrchestrator
from agents.base import SignalAction, SignalStrength
from agents.sentiment_aggregator import sentiment_aggregator_agent

logger = logging.getLogger(__name__)


class AISignalGenerator:
    """Generates AI-powered trading signals"""
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.min_confidence = 0.65  # Minimum confidence for signal generation
        
    async def generate_signal(
        self, 
        symbol: str, 
        timeframe: str = "15min"
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal for a symbol"""
        
        try:
            # Get historical data
            period_map = {
                "1min": "1d",
                "5min": "5d", 
                "15min": "1mo",
                "30min": "1mo",
                "1h": "3mo",
                "1d": "6mo"
            }
            
            period = period_map.get(timeframe, "1mo")
            df = await market_data_service.get_historical_data(symbol, period, timeframe)
            
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
                
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Detect patterns
            patterns = self._detect_patterns(df)
            
            # Get market regime
            regime = self._analyze_market_regime(df, indicators)
            
            # Get sentiment analysis
            sentiment_result = None
            try:
                from agents.base import AgentContext
                context = AgentContext(
                    symbol=symbol,
                    timeframe=timeframe,
                    market_data={'dataframe': df}
                )
                sentiment_result = await sentiment_aggregator_agent.analyze(context)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
            
            # Generate signal based on multiple factors
            signal = self._generate_signal_decision(
                df, indicators, patterns, regime, sentiment_result
            )
            
            if signal and signal['confidence'] >= self.min_confidence:
                # Calculate entry, targets, and stop loss
                current_price = float(df['Close'].iloc[-1])
                
                signal_data = {
                    "id": f"sig_{symbol}_{datetime.utcnow().timestamp()}",
                    "symbol": symbol,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": signal['action'],
                    "strength": signal['strength'],
                    "confidence": signal['confidence'],
                    "entry": current_price,
                    "pattern": patterns[0] if patterns else None,
                    "indicators": indicators,
                    "regime": regime
                }
                
                # Calculate targets and stop loss based on signal type
                if signal['action'] in ['BUY', 'STRONG_BUY']:
                    signal_data['targets'] = [
                        current_price * 1.01,  # 1% target
                        current_price * 1.02,  # 2% target
                        current_price * 1.03   # 3% target
                    ]
                    signal_data['stopLoss'] = current_price * 0.98  # 2% stop
                else:  # SELL signals
                    signal_data['targets'] = [
                        current_price * 0.99,  # 1% target
                        current_price * 0.98,  # 2% target
                        current_price * 0.97   # 3% target
                    ]
                    signal_data['stopLoss'] = current_price * 1.02  # 2% stop
                    
                return signal_data
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
            
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        
        # Moving averages
        sma20 = df['Close'].rolling(20).mean()
        sma50 = df['Close'].rolling(50).mean()
        
        # Volume analysis
        volume_sma = df['Volume'].rolling(20).mean()
        volume_ratio = df['Volume'].iloc[-1] / volume_sma.iloc[-1]
        
        return {
            'rsi': float(rsi.iloc[-1]),
            'macd': float(macd.iloc[-1]),
            'macd_signal': float(macd_signal.iloc[-1]),
            'sma20': float(sma20.iloc[-1]),
            'sma50': float(sma50.iloc[-1]) if len(df) >= 50 else float(sma20.iloc[-1]),
            'volume_ratio': float(volume_ratio),
            'price_vs_sma20': float((df['Close'].iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1]),
            'trend_strength': float(abs(sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]) if len(df) >= 50 else 0
        }
        
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect chart patterns"""
        patterns = []
        
        # Simple pattern detection
        closes = df['Close'].values[-20:]
        highs = df['High'].values[-20:]
        lows = df['Low'].values[-20:]
        
        # Bullish patterns
        if self._is_double_bottom(lows):
            patterns.append("Double Bottom")
        if self._is_ascending_triangle(highs, lows):
            patterns.append("Ascending Triangle")
            
        # Bearish patterns  
        if self._is_double_top(highs):
            patterns.append("Double Top")
        if self._is_descending_triangle(highs, lows):
            patterns.append("Descending Triangle")
            
        # Continuation patterns
        if self._is_flag_pattern(closes):
            patterns.append("Flag Pattern")
            
        return patterns
        
    def _is_double_bottom(self, lows: np.ndarray) -> bool:
        """Check for double bottom pattern"""
        if len(lows) < 10:
            return False
            
        # Find two local minima
        min1_idx = np.argmin(lows[:10])
        min2_idx = np.argmin(lows[10:]) + 10
        
        # Check if they're approximately equal
        if abs(lows[min1_idx] - lows[min2_idx]) / lows[min1_idx] < 0.02:
            # Check for peak between them
            peak = np.max(lows[min1_idx:min2_idx])
            if peak > lows[min1_idx] * 1.03:
                return True
        return False
        
    def _is_double_top(self, highs: np.ndarray) -> bool:
        """Check for double top pattern"""
        if len(highs) < 10:
            return False
            
        # Find two local maxima
        max1_idx = np.argmax(highs[:10])
        max2_idx = np.argmax(highs[10:]) + 10
        
        # Check if they're approximately equal
        if abs(highs[max1_idx] - highs[max2_idx]) / highs[max1_idx] < 0.02:
            # Check for valley between them
            valley = np.min(highs[max1_idx:max2_idx])
            if valley < highs[max1_idx] * 0.97:
                return True
        return False
        
    def _is_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Check for ascending triangle pattern"""
        if len(highs) < 5:
            return False
            
        # Check if highs are relatively flat
        high_std = np.std(highs[-5:])
        if high_std / np.mean(highs[-5:]) > 0.01:
            return False
            
        # Check if lows are rising
        low_slope = np.polyfit(range(len(lows[-5:])), lows[-5:], 1)[0]
        return low_slope > 0
        
    def _is_descending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Check for descending triangle pattern"""
        if len(lows) < 5:
            return False
            
        # Check if lows are relatively flat
        low_std = np.std(lows[-5:])
        if low_std / np.mean(lows[-5:]) > 0.01:
            return False
            
        # Check if highs are falling
        high_slope = np.polyfit(range(len(highs[-5:])), highs[-5:], 1)[0]
        return high_slope < 0
        
    def _is_flag_pattern(self, closes: np.ndarray) -> bool:
        """Check for flag pattern"""
        if len(closes) < 10:
            return False
            
        # Check for strong move followed by consolidation
        initial_move = abs(closes[5] - closes[0]) / closes[0]
        consolidation_range = (np.max(closes[5:]) - np.min(closes[5:])) / np.mean(closes[5:])
        
        return initial_move > 0.03 and consolidation_range < 0.02
        
    def _analyze_market_regime(
        self, 
        df: pd.DataFrame, 
        indicators: Dict[str, float]
    ) -> str:
        """Analyze current market regime"""
        
        # Trend analysis
        if indicators['sma20'] > indicators['sma50'] * 1.01:
            trend = "uptrend"
        elif indicators['sma20'] < indicators['sma50'] * 0.99:
            trend = "downtrend"
        else:
            trend = "sideways"
            
        # Volatility analysis
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        if volatility < 0.15:
            vol_regime = "low_volatility"
        elif volatility > 0.30:
            vol_regime = "high_volatility"
        else:
            vol_regime = "normal_volatility"
            
        # Volume analysis
        if indicators['volume_ratio'] > 1.5:
            volume_regime = "high_volume"
        elif indicators['volume_ratio'] < 0.5:
            volume_regime = "low_volume"
        else:
            volume_regime = "normal_volume"
            
        return f"{trend}_{vol_regime}_{volume_regime}"
        
    def _generate_signal_decision(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, float],
        patterns: List[str],
        regime: str,
        sentiment_result: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate final signal decision"""
        
        score = 0
        factors = []
        
        # RSI signals
        if indicators['rsi'] < 30:
            score += 2
            factors.append("RSI oversold")
        elif indicators['rsi'] > 70:
            score -= 2
            factors.append("RSI overbought")
            
        # MACD signals
        if indicators['macd'] > indicators['macd_signal']:
            score += 1
            factors.append("MACD bullish")
        else:
            score -= 1
            factors.append("MACD bearish")
            
        # Moving average signals
        if indicators['price_vs_sma20'] > 0:
            score += 1
            factors.append("Price above SMA20")
        else:
            score -= 1
            factors.append("Price below SMA20")
            
        # Pattern signals
        if "Double Bottom" in patterns or "Ascending Triangle" in patterns:
            score += 2
            factors.append(f"Bullish pattern: {patterns[0]}")
        elif "Double Top" in patterns or "Descending Triangle" in patterns:
            score -= 2
            factors.append(f"Bearish pattern: {patterns[0]}")
            
        # Volume confirmation
        if indicators['volume_ratio'] > 1.2:
            score = int(score * 1.2)  # Amplify signal with volume
            factors.append("High volume confirmation")
            
        # Sentiment signals
        if sentiment_result and sentiment_result.get('signal'):
            sentiment_signal = sentiment_result['signal']
            sentiment_confidence = sentiment_result.get('confidence', 0)
            
            if sentiment_signal in ['STRONG_BUY', 'BUY']:
                sentiment_boost = 2 if sentiment_signal == 'STRONG_BUY' else 1
                score += sentiment_boost * sentiment_confidence
                factors.append(f"Positive social sentiment ({sentiment_signal})")
            elif sentiment_signal in ['STRONG_SELL', 'SELL']:
                sentiment_penalty = 2 if sentiment_signal == 'STRONG_SELL' else 1
                score -= sentiment_penalty * sentiment_confidence
                factors.append(f"Negative social sentiment ({sentiment_signal})")
                
            # Add sentiment metadata
            if sentiment_result.get('metadata'):
                factors.append(f"Sentiment from {sentiment_result['metadata'].get('posts_analyzed', 0)} posts")
            
        # Generate signal based on score
        if score >= 3:
            action = "STRONG_BUY"
            strength = "STRONG"
        elif score >= 2:
            action = "BUY"
            strength = "MODERATE"
        elif score <= -3:
            action = "STRONG_SELL"
            strength = "STRONG"
        elif score <= -2:
            action = "SELL"
            strength = "MODERATE"
        else:
            action = "HOLD"
            strength = "WEAK"
            
        # Calculate confidence
        confidence = min(0.95, 0.5 + (abs(score) * 0.1))
        
        return {
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'score': score,
            'factors': factors
        }


# Global instance
ai_signal_generator = AISignalGenerator()