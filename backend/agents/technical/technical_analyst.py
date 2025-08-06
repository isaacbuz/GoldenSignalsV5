"""
Technical Analysis Agent
Analyzes price patterns, trends, and technical indicators
"""

from typing import Any, Dict, List, Optional
import numpy as np

from agents.base import BaseAgent, AgentConfig, Signal, SignalAction, SignalStrength
from core.logging import get_logger

logger = get_logger(__name__)


class TechnicalAnalystAgent(BaseAgent):
    """
    Agent that performs technical analysis on market data
    
    Analyzes:
    - Moving averages and crossovers
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Volume patterns
    - Support/resistance levels
    - Trend strength
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if not config:
            config = AgentConfig(
                name="TechnicalAnalyst",
                confidence_threshold=0.65,
                weight=0.8
            )
        super().__init__(config)
        
        # Technical analysis parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.trend_strength_threshold = 0.6
    
    def get_required_data_types(self) -> List[str]:
        """Technical analysis requires price, volume, and indicators"""
        return ['price', 'volume', 'indicators', 'ohlcv']
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Analyze market data for technical signals
        
        Args:
            market_data: Dictionary containing price, volume, indicators, etc.
            
        Returns:
            Trading signal or None
        """
        try:
            symbol = market_data['symbol']
            price = market_data['price']
            volume = market_data.get('volume', 0)
            indicators = market_data.get('indicators', {})
            
            # Extract technical indicators
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            sma_20 = indicators.get('sma_20', price)
            sma_50 = indicators.get('sma_50', price)
            sma_200 = indicators.get('sma_200', price)
            ema_12 = indicators.get('ema_12', price)
            ema_26 = indicators.get('ema_26', price)
            volume_avg = indicators.get('volume_avg_20', volume)
            bb_upper = indicators.get('bb_upper', price * 1.02)
            bb_lower = indicators.get('bb_lower', price * 0.98)
            
            # Calculate additional metrics
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            price_position = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # Collect signals and reasoning
            signals = []
            reasoning = []
            confidence_factors = []
            
            # 1. RSI Analysis
            if rsi <= self.rsi_oversold:
                signals.append('buy')
                reasoning.append(f"RSI oversold at {rsi:.1f}")
                confidence_factors.append(0.7 + (self.rsi_oversold - rsi) / 100)
            elif rsi >= self.rsi_overbought:
                signals.append('sell')
                reasoning.append(f"RSI overbought at {rsi:.1f}")
                confidence_factors.append(0.7 + (rsi - self.rsi_overbought) / 100)
            
            # 2. MACD Analysis
            if macd > macd_signal and macd > 0:
                signals.append('buy')
                reasoning.append("MACD bullish crossover above zero")
                confidence_factors.append(0.75)
            elif macd < macd_signal and macd < 0:
                signals.append('sell')
                reasoning.append("MACD bearish crossover below zero")
                confidence_factors.append(0.75)
            
            # 3. Moving Average Analysis
            if price > sma_20 > sma_50 > sma_200:
                signals.append('buy')
                reasoning.append("Price above all major moving averages (bullish alignment)")
                confidence_factors.append(0.8)
            elif price < sma_20 < sma_50 < sma_200:
                signals.append('sell')
                reasoning.append("Price below all major moving averages (bearish alignment)")
                confidence_factors.append(0.8)
            
            # 4. Golden/Death Cross
            if sma_50 > sma_200 * 1.01 and price > sma_50:
                signals.append('buy')
                reasoning.append("Golden cross pattern with price confirmation")
                confidence_factors.append(0.85)
            elif sma_50 < sma_200 * 0.99 and price < sma_50:
                signals.append('sell')
                reasoning.append("Death cross pattern with price confirmation")
                confidence_factors.append(0.85)
            
            # 5. Volume Analysis
            if volume_ratio > self.volume_spike_threshold:
                if price > sma_20:
                    signals.append('buy')
                    reasoning.append(f"Volume spike ({volume_ratio:.1f}x avg) with upward price movement")
                    confidence_factors.append(0.7)
                elif price < sma_20:
                    signals.append('sell')
                    reasoning.append(f"Volume spike ({volume_ratio:.1f}x avg) with downward price movement")
                    confidence_factors.append(0.7)
            
            # 6. Bollinger Bands
            if price <= bb_lower * 1.01:
                signals.append('buy')
                reasoning.append("Price at lower Bollinger Band (oversold)")
                confidence_factors.append(0.65)
            elif price >= bb_upper * 0.99:
                signals.append('sell')
                reasoning.append("Price at upper Bollinger Band (overbought)")
                confidence_factors.append(0.65)
            
            # Determine consensus signal
            if not signals:
                return None
            
            buy_count = signals.count('buy')
            sell_count = signals.count('sell')
            
            if buy_count == 0 and sell_count == 0:
                return None
            
            # Calculate final signal
            if buy_count > sell_count:
                action = SignalAction.BUY
                signal_confidence = sum(c for i, c in enumerate(confidence_factors) if signals[i] == 'buy') / buy_count
                strength_ratio = buy_count / len(signals)
            else:
                action = SignalAction.SELL
                signal_confidence = sum(c for i, c in enumerate(confidence_factors) if signals[i] == 'sell') / sell_count
                strength_ratio = sell_count / len(signals)
            
            # Determine signal strength
            if strength_ratio >= 0.7:
                strength = SignalStrength.STRONG
            elif strength_ratio >= 0.5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Adjust confidence based on consensus
            final_confidence = signal_confidence * (0.5 + 0.5 * strength_ratio)
            
            # Calculate target and stop loss
            if action == SignalAction.BUY:
                # Use ATR or fixed percentage for targets
                atr = indicators.get('atr', price * 0.02)
                target_price = price * (1 + 2 * atr / price)  # 2 ATR target
                stop_loss = price * (1 - atr / price)  # 1 ATR stop
            else:
                atr = indicators.get('atr', price * 0.02)
                target_price = price * (1 - 2 * atr / price)  # 2 ATR target
                stop_loss = price * (1 + atr / price)  # 1 ATR stop
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                action=action,
                confidence=min(final_confidence, 0.95),
                strength=strength,
                source=self.config.name,
                current_price=price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning,
                features={
                    "buy_signals": buy_count,
                    "sell_signals": sell_count,
                    "total_signals": len(signals),
                    "volume_ratio": volume_ratio,
                    "price_position": price_position
                },
                indicators={
                    "rsi": rsi,
                    "macd": macd,
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "volume_ratio": volume_ratio
                },
                market_conditions={
                    "trend": "bullish" if price > sma_50 else "bearish",
                    "volatility": "high" if (bb_upper - bb_lower) / price > 0.04 else "normal",
                    "volume_trend": "increasing" if volume_ratio > 1.2 else "normal"
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}", exc_info=True)
            return None
    
    async def initialize(self) -> None:
        """Initialize technical analysis agent"""
        logger.info(f"Technical Analyst Agent initialized with RSI thresholds: {self.rsi_oversold}/{self.rsi_overbought}")
        
        # Could load historical performance data here
        # Could warm up any caches or models