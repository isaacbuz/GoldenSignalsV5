"""
Technical Analysis Agent V5
Comprehensive technical analysis combining multiple indicators and pattern recognition
Enhanced from archive with V5 architecture patterns
"""

import asyncio
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
from scipy import stats

from core.logging import get_logger

logger = get_logger(__name__)


class TechnicalSignal(Enum):
    """Technical analysis signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TrendDirection(Enum):
    """Market trend directions"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class PatternType(Enum):
    """Technical pattern types"""
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


@dataclass
class TechnicalIndicators:
    """Container for all technical indicators"""
    # Moving Averages
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    
    # Momentum Indicators
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    stochastic_k: float
    stochastic_d: float
    
    # Volatility Indicators
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_width: float
    atr: float
    
    # Volume Indicators
    obv: float
    volume_sma: float
    volume_ratio: float
    
    # Trend Indicators
    adx: float
    plus_di: float
    minus_di: float
    
    # Other
    vwap: float
    pivot_point: float
    support_1: float
    resistance_1: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternDetection:
    """Detected technical pattern"""
    pattern_type: PatternType
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'description': self.description
        }


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis result"""
    symbol: str
    indicators: TechnicalIndicators
    trend: TrendDirection
    signal: TechnicalSignal
    signal_strength: float
    patterns: List[PatternDetection]
    support_levels: List[float]
    resistance_levels: List[float]
    recommendations: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'indicators': self.indicators.to_dict(),
            'trend': self.trend.value,
            'signal': self.signal.value,
            'signal_strength': self.signal_strength,
            'patterns': [p.to_dict() for p in self.patterns],
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class TechnicalAnalysisAgent:
    """
    V5 Technical Analysis Agent
    Comprehensive technical analysis with multiple indicators and pattern recognition
    """
    
    def __init__(self):
        """Initialize the technical analysis agent"""
        # Configuration parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        self.bb_period = 20
        self.bb_std = 2
        
        self.stoch_period = 14
        self.stoch_smooth = 3
        
        self.adx_period = 14
        self.atr_period = 14
        
        # Signal thresholds
        self.strong_signal_threshold = 0.8
        self.signal_threshold = 0.6
        
        # Pattern detection parameters
        self.pattern_lookback = 50
        self.min_pattern_confidence = 0.6
        
        # Cache for calculations
        self.calculation_cache = {}
        self.analysis_history = deque(maxlen=100)
        
        # Performance tracking
        self.calculation_times = {}
        
        logger.info("Technical Analysis Agent V5 initialized")
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> TechnicalAnalysis:
        """
        Perform comprehensive technical analysis
        
        Args:
            symbol: Trading symbol
            market_data: OHLCV market data
        
        Returns:
            Complete technical analysis
        """
        try:
            start_time = datetime.now()
            
            # Extract and validate data
            df = self._prepare_dataframe(market_data)
            
            if len(df) < 200:
                raise ValueError(f"Insufficient data: need at least 200 data points, got {len(df)}")
            
            # Calculate all indicators
            indicators = self._calculate_all_indicators(df)
            
            # Determine trend
            trend = self._determine_trend(df, indicators)
            
            # Detect patterns
            patterns = self._detect_patterns(df, indicators)
            
            # Calculate support and resistance levels
            support_levels, resistance_levels = self._calculate_support_resistance(df)
            
            # Generate trading signal
            signal, signal_strength = self._generate_signal(indicators, trend, patterns)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                indicators, trend, patterns, signal
            )
            
            # Create analysis result
            analysis = TechnicalAnalysis(
                symbol=symbol,
                indicators=indicators,
                trend=trend,
                signal=signal,
                signal_strength=signal_strength,
                patterns=patterns,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.analysis_history.append(analysis)
            
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds()
            if 'technical_analysis' not in self.calculation_times:
                self.calculation_times['technical_analysis'] = []
            self.calculation_times['technical_analysis'].append(calc_time)
            
            logger.info(f"Technical analysis for {symbol}: {trend.value} trend, {signal.value} signal")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
            raise
    
    def _prepare_dataframe(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare pandas DataFrame from market data"""
        
        df = pd.DataFrame({
            'open': market_data.get('open_prices', market_data.get('close_prices', [])),
            'high': market_data.get('high_prices', market_data.get('close_prices', [])),
            'low': market_data.get('low_prices', market_data.get('close_prices', [])),
            'close': market_data['close_prices'],
            'volume': market_data.get('volume', [0] * len(market_data['close_prices']))
        })
        
        # Ensure volume is not zero
        df['volume'] = df['volume'].replace(0, df['volume'].mean())
        
        return df
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        
        # Moving Averages
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
        sma_200 = df['close'].rolling(window=200).mean().iloc[-1]
        ema_12 = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
        
        # RSI
        rsi = self._calculate_rsi(df['close'])
        
        # MACD
        macd, macd_signal, macd_histogram = self._calculate_macd(df['close'])
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(df)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger_bands(df['close'])
        
        # ATR
        atr = self._calculate_atr(df)
        
        # ADX
        adx, plus_di, minus_di = self._calculate_adx(df)
        
        # Volume indicators
        obv = self._calculate_obv(df)
        volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / volume_sma if volume_sma > 0 else 1.0
        
        # VWAP
        vwap = self._calculate_vwap(df)
        
        # Pivot Points
        pivot_point, support_1, resistance_1 = self._calculate_pivot_points(df)
        
        return TechnicalIndicators(
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            stochastic_k=stoch_k,
            stochastic_d=stoch_d,
            bollinger_upper=bb_upper,
            bollinger_middle=bb_middle,
            bollinger_lower=bb_lower,
            bollinger_width=bb_width,
            atr=atr,
            obv=obv,
            volume_sma=volume_sma,
            volume_ratio=volume_ratio,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            vwap=vwap,
            pivot_point=pivot_point,
            support_1=support_1,
            resistance_1=resistance_1
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        """Calculate Relative Strength Index"""
        if period is None:
            period = self.rsi_period
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD indicators"""
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return (
            macd_line.iloc[-1],
            signal_line.iloc[-1],
            histogram.iloc[-1]
        )
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=self.stoch_period).min()
        high_max = df['high'].rolling(window=self.stoch_period).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=self.stoch_smooth).mean()
        
        return k_percent.iloc[-1], d_percent.iloc[-1]
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[float, float, float, float]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        width = (upper - lower) / middle
        
        return (
            upper.iloc[-1],
            middle.iloc[-1],
            lower.iloc[-1],
            width.iloc[-1]
        )
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr.iloc[-1]
    
    def _calculate_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate ADX and Directional Indicators"""
        # Calculate directional movement
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When both are positive, keep the larger
        mask = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[mask & (plus_dm < minus_dm)] = 0
        minus_dm[mask & (plus_dm >= minus_dm)] = 0
        
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return (
            adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0,
            plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0,
            minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0
        )
    
    def _calculate_obv(self, df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv.iloc[-1]
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.iloc[-1]
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate Pivot Points"""
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        support_1 = (2 * pivot) - high
        resistance_1 = (2 * pivot) - low
        
        return pivot, support_1, resistance_1
    
    def _determine_trend(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> TrendDirection:
        """Determine overall market trend"""
        
        current_price = df['close'].iloc[-1]
        
        # Price vs moving averages
        ma_score = 0
        if current_price > indicators.sma_20:
            ma_score += 1
        if current_price > indicators.sma_50:
            ma_score += 1
        if current_price > indicators.sma_200:
            ma_score += 1
        if indicators.sma_20 > indicators.sma_50:
            ma_score += 1
        if indicators.sma_50 > indicators.sma_200:
            ma_score += 1
        
        # ADX trend strength
        trend_strength = indicators.adx
        
        # Directional movement
        if indicators.plus_di > indicators.minus_di:
            direction_score = 1
        else:
            direction_score = -1
        
        # MACD trend
        if indicators.macd > indicators.macd_signal:
            macd_score = 1
        else:
            macd_score = -1
        
        # Combine scores
        total_score = (ma_score - 2.5) * direction_score + macd_score
        
        if total_score >= 3 and trend_strength > 25:
            return TrendDirection.STRONG_UPTREND
        elif total_score >= 1:
            return TrendDirection.UPTREND
        elif total_score <= -3 and trend_strength > 25:
            return TrendDirection.STRONG_DOWNTREND
        elif total_score <= -1:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS
    
    def _detect_patterns(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> List[PatternDetection]:
        """Detect technical patterns"""
        patterns = []
        
        current_price = df['close'].iloc[-1]
        
        # Golden Cross / Death Cross
        if len(df) >= 50:
            sma_50_prev = df['close'].rolling(window=50).mean().iloc[-2]
            sma_200_prev = df['close'].rolling(window=200).mean().iloc[-2]
            
            if indicators.sma_50 > indicators.sma_200 and sma_50_prev <= sma_200_prev:
                patterns.append(PatternDetection(
                    pattern_type=PatternType.GOLDEN_CROSS,
                    confidence=0.85,
                    entry_price=current_price,
                    target_price=current_price * 1.10,
                    stop_loss=current_price * 0.95,
                    description="50 SMA crossed above 200 SMA - Bullish signal"
                ))
            elif indicators.sma_50 < indicators.sma_200 and sma_50_prev >= sma_200_prev:
                patterns.append(PatternDetection(
                    pattern_type=PatternType.DEATH_CROSS,
                    confidence=0.85,
                    entry_price=current_price,
                    target_price=current_price * 0.90,
                    stop_loss=current_price * 1.05,
                    description="50 SMA crossed below 200 SMA - Bearish signal"
                ))
        
        # RSI Divergence
        if len(df) >= 20:
            price_trend = np.polyfit(range(20), df['close'].tail(20).values, 1)[0]
            rsi_series = df['close'].rolling(window=self.rsi_period).apply(
                lambda x: self._calculate_rsi(pd.Series(x)) if len(x) == self.rsi_period else 50
            )
            rsi_trend = np.polyfit(range(20), rsi_series.tail(20).values, 1)[0]
            
            if price_trend > 0 and rsi_trend < 0 and indicators.rsi > self.rsi_overbought:
                patterns.append(PatternDetection(
                    pattern_type=PatternType.BEARISH_DIVERGENCE,
                    confidence=0.7,
                    entry_price=current_price,
                    target_price=current_price * 0.95,
                    stop_loss=current_price * 1.03,
                    description="Price rising but RSI falling - Bearish divergence"
                ))
            elif price_trend < 0 and rsi_trend > 0 and indicators.rsi < self.rsi_oversold:
                patterns.append(PatternDetection(
                    pattern_type=PatternType.BULLISH_DIVERGENCE,
                    confidence=0.7,
                    entry_price=current_price,
                    target_price=current_price * 1.05,
                    stop_loss=current_price * 0.97,
                    description="Price falling but RSI rising - Bullish divergence"
                ))
        
        # Bollinger Band Breakout
        if current_price > indicators.bollinger_upper:
            patterns.append(PatternDetection(
                pattern_type=PatternType.BREAKOUT,
                confidence=0.65,
                entry_price=current_price,
                target_price=current_price * 1.03,
                stop_loss=indicators.bollinger_middle,
                description="Price broke above upper Bollinger Band"
            ))
        elif current_price < indicators.bollinger_lower:
            patterns.append(PatternDetection(
                pattern_type=PatternType.BREAKDOWN,
                confidence=0.65,
                entry_price=current_price,
                target_price=current_price * 0.97,
                stop_loss=indicators.bollinger_middle,
                description="Price broke below lower Bollinger Band"
            ))
        
        # Double Top/Bottom detection (simplified)
        if len(df) >= 50:
            highs = df['high'].tail(50)
            lows = df['low'].tail(50)
            
            # Find peaks and troughs
            peaks = self._find_peaks(highs.values)
            troughs = self._find_troughs(lows.values)
            
            # Double Top
            if len(peaks) >= 2:
                if abs(highs.iloc[peaks[-1]] - highs.iloc[peaks[-2]]) / highs.iloc[peaks[-1]] < 0.03:
                    patterns.append(PatternDetection(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=0.6,
                        entry_price=current_price,
                        target_price=current_price * 0.93,
                        stop_loss=highs.iloc[peaks[-1]] * 1.02,
                        description="Double top pattern detected - Bearish reversal"
                    ))
            
            # Double Bottom
            if len(troughs) >= 2:
                if abs(lows.iloc[troughs[-1]] - lows.iloc[troughs[-2]]) / lows.iloc[troughs[-1]] < 0.03:
                    patterns.append(PatternDetection(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=0.6,
                        entry_price=current_price,
                        target_price=current_price * 1.07,
                        stop_loss=lows.iloc[troughs[-1]] * 0.98,
                        description="Double bottom pattern detected - Bullish reversal"
                    ))
        
        return patterns
    
    def _find_peaks(self, data: np.ndarray, prominence: float = 0.02) -> List[int]:
        """Find peaks in price data"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data, prominence=data.mean() * prominence)
        return peaks.tolist()
    
    def _find_troughs(self, data: np.ndarray, prominence: float = 0.02) -> List[int]:
        """Find troughs in price data"""
        from scipy.signal import find_peaks
        inverted = -data
        troughs, _ = find_peaks(inverted, prominence=abs(inverted.mean()) * prominence)
        return troughs.tolist()
    
    def _calculate_support_resistance(self, df: pd.DataFrame, num_levels: int = 3) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        
        # Use recent highs and lows
        recent_data = df.tail(100)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        closes = recent_data['close'].values
        
        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)
        
        # Get resistance levels from peaks
        resistance_levels = []
        if peaks:
            peak_prices = [highs[p] for p in peaks]
            resistance_levels = sorted(peak_prices, reverse=True)[:num_levels]
        
        # Get support levels from troughs
        support_levels = []
        if troughs:
            trough_prices = [lows[t] for t in troughs]
            support_levels = sorted(trough_prices)[:num_levels]
        
        # Add pivot-based levels if needed
        if len(resistance_levels) < num_levels:
            pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
            resistance_levels.append(pivot * 1.02)
        
        if len(support_levels) < num_levels:
            pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
            support_levels.append(pivot * 0.98)
        
        return support_levels, resistance_levels
    
    def _generate_signal(self, indicators: TechnicalIndicators, trend: TrendDirection, 
                        patterns: List[PatternDetection]) -> Tuple[TechnicalSignal, float]:
        """Generate trading signal based on all analysis"""
        
        signal_scores = {
            TechnicalSignal.STRONG_BUY: 0.0,
            TechnicalSignal.BUY: 0.0,
            TechnicalSignal.NEUTRAL: 0.5,
            TechnicalSignal.SELL: 0.0,
            TechnicalSignal.STRONG_SELL: 0.0
        }
        
        # RSI signals
        if indicators.rsi < self.rsi_oversold:
            signal_scores[TechnicalSignal.BUY] += 0.3
            if indicators.rsi < 20:
                signal_scores[TechnicalSignal.STRONG_BUY] += 0.2
        elif indicators.rsi > self.rsi_overbought:
            signal_scores[TechnicalSignal.SELL] += 0.3
            if indicators.rsi > 80:
                signal_scores[TechnicalSignal.STRONG_SELL] += 0.2
        
        # MACD signals
        if indicators.macd_histogram > 0:
            if indicators.macd > indicators.macd_signal:
                signal_scores[TechnicalSignal.BUY] += 0.25
            else:
                signal_scores[TechnicalSignal.NEUTRAL] += 0.1
        else:
            if indicators.macd < indicators.macd_signal:
                signal_scores[TechnicalSignal.SELL] += 0.25
            else:
                signal_scores[TechnicalSignal.NEUTRAL] += 0.1
        
        # Stochastic signals
        if indicators.stochastic_k < 20:
            signal_scores[TechnicalSignal.BUY] += 0.15
        elif indicators.stochastic_k > 80:
            signal_scores[TechnicalSignal.SELL] += 0.15
        
        # Trend-based adjustments
        if trend == TrendDirection.STRONG_UPTREND:
            signal_scores[TechnicalSignal.BUY] += 0.2
            signal_scores[TechnicalSignal.STRONG_BUY] += 0.1
        elif trend == TrendDirection.UPTREND:
            signal_scores[TechnicalSignal.BUY] += 0.15
        elif trend == TrendDirection.STRONG_DOWNTREND:
            signal_scores[TechnicalSignal.SELL] += 0.2
            signal_scores[TechnicalSignal.STRONG_SELL] += 0.1
        elif trend == TrendDirection.DOWNTREND:
            signal_scores[TechnicalSignal.SELL] += 0.15
        
        # Pattern-based adjustments
        for pattern in patterns:
            if pattern.confidence > self.min_pattern_confidence:
                if pattern.pattern_type in [PatternType.GOLDEN_CROSS, PatternType.DOUBLE_BOTTOM, 
                                           PatternType.BULLISH_DIVERGENCE, PatternType.INVERSE_HEAD_SHOULDERS]:
                    signal_scores[TechnicalSignal.BUY] += pattern.confidence * 0.2
                elif pattern.pattern_type in [PatternType.DEATH_CROSS, PatternType.DOUBLE_TOP,
                                             PatternType.BEARISH_DIVERGENCE, PatternType.HEAD_SHOULDERS]:
                    signal_scores[TechnicalSignal.SELL] += pattern.confidence * 0.2
        
        # Volume confirmation
        if indicators.volume_ratio > 1.5:
            # High volume confirms the signal
            max_signal = max(signal_scores.keys(), key=lambda k: signal_scores[k])
            if max_signal != TechnicalSignal.NEUTRAL:
                signal_scores[max_signal] *= 1.2
        
        # ADX trend strength adjustment
        if indicators.adx > 25:
            # Strong trend
            if indicators.plus_di > indicators.minus_di:
                signal_scores[TechnicalSignal.BUY] *= 1.1
            else:
                signal_scores[TechnicalSignal.SELL] *= 1.1
        elif indicators.adx < 20:
            # Weak trend, increase neutral
            signal_scores[TechnicalSignal.NEUTRAL] *= 1.2
        
        # Find dominant signal
        primary_signal = max(signal_scores.keys(), key=lambda k: signal_scores[k])
        signal_strength = min(1.0, signal_scores[primary_signal])
        
        # Upgrade to strong signal if confidence is high
        if signal_strength > self.strong_signal_threshold:
            if primary_signal == TechnicalSignal.BUY:
                primary_signal = TechnicalSignal.STRONG_BUY
            elif primary_signal == TechnicalSignal.SELL:
                primary_signal = TechnicalSignal.STRONG_SELL
        
        return primary_signal, signal_strength
    
    def _generate_recommendations(self, indicators: TechnicalIndicators, trend: TrendDirection,
                                 patterns: List[PatternDetection], signal: TechnicalSignal) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Signal-based recommendations
        if signal == TechnicalSignal.STRONG_BUY:
            recommendations.append("Strong buy signal - Consider entering long position")
            recommendations.append(f"Entry near support at {indicators.support_1:.2f}")
        elif signal == TechnicalSignal.BUY:
            recommendations.append("Buy signal - Look for entry on pullback")
            recommendations.append(f"Target resistance at {indicators.resistance_1:.2f}")
        elif signal == TechnicalSignal.STRONG_SELL:
            recommendations.append("Strong sell signal - Consider exiting long positions")
            recommendations.append(f"Entry near resistance at {indicators.resistance_1:.2f}")
        elif signal == TechnicalSignal.SELL:
            recommendations.append("Sell signal - Consider reducing position size")
            recommendations.append(f"Watch support at {indicators.support_1:.2f}")
        else:
            recommendations.append("Neutral signal - Wait for clearer direction")
        
        # RSI recommendations
        if indicators.rsi < self.rsi_oversold:
            recommendations.append(f"RSI oversold at {indicators.rsi:.1f} - Potential bounce")
        elif indicators.rsi > self.rsi_overbought:
            recommendations.append(f"RSI overbought at {indicators.rsi:.1f} - Potential pullback")
        
        # MACD recommendations
        if indicators.macd_histogram > 0 and abs(indicators.macd_histogram) > abs(indicators.macd) * 0.1:
            recommendations.append("MACD histogram expanding - Momentum increasing")
        elif indicators.macd_histogram < 0 and abs(indicators.macd_histogram) > abs(indicators.macd) * 0.1:
            recommendations.append("MACD histogram contracting - Momentum decreasing")
        
        # Bollinger Band recommendations
        if indicators.bollinger_width < 0.05:
            recommendations.append("Bollinger squeeze - Expect volatility expansion")
        elif indicators.bollinger_width > 0.15:
            recommendations.append("Wide Bollinger Bands - High volatility environment")
        
        # Pattern recommendations
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern.confidence > self.min_pattern_confidence:
                recommendations.append(f"{pattern.description}")
        
        # Volume recommendations
        if indicators.volume_ratio > 2.0:
            recommendations.append("High volume confirmation - Strong conviction")
        elif indicators.volume_ratio < 0.5:
            recommendations.append("Low volume warning - Weak conviction")
        
        # ADX trend strength
        if indicators.adx > 40:
            recommendations.append("Very strong trend - Trade with trend direction")
        elif indicators.adx < 20:
            recommendations.append("Weak trend - Consider range-trading strategies")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_analyses = len(self.analysis_history)
        
        if total_analyses == 0:
            return {
                'total_analyses': 0,
                'average_calc_time': 0.0,
                'signal_distribution': {},
                'trend_distribution': {}
            }
        
        # Calculate performance metrics
        signal_counts = {}
        trend_counts = {}
        
        for analysis in self.analysis_history:
            signal = analysis.signal.value
            trend = analysis.trend.value
            
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        # Average calculation time
        avg_calc_time = 0.0
        if 'technical_analysis' in self.calculation_times:
            times = self.calculation_times['technical_analysis']
            avg_calc_time = sum(times) / len(times) if times else 0.0
        
        return {
            'total_analyses': total_analyses,
            'average_calc_time_seconds': avg_calc_time,
            'signal_distribution': signal_counts,
            'trend_distribution': trend_counts,
            'supported_indicators': 25,
            'pattern_types': len(PatternType),
            'cache_size': len(self.calculation_cache)
        }


# Create global instance
technical_analysis_agent = TechnicalAnalysisAgent()