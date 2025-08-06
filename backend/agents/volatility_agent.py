"""
Volatility Agent V5
Comprehensive volatility analysis combining realized vol, IV rank, skew analysis, and regime detection
Enhanced from archive with V5 architecture integration
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
from scipy.stats import norm

from core.logging import get_logger

logger = get_logger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    EXTREMELY_LOW = "extremely_low"     # <10th percentile
    LOW = "low"                        # 10th-25th percentile  
    NORMAL = "normal"                  # 25th-75th percentile
    HIGH = "high"                      # 75th-90th percentile
    EXTREMELY_HIGH = "extremely_high"  # >90th percentile


class VolatilitySignal(Enum):
    """Volatility-based trading signals"""
    LONG_VOLATILITY = "long_volatility"           # Buy vol (options, VIX calls)
    SHORT_VOLATILITY = "short_volatility"         # Sell vol (covered calls, VIX puts)
    VOLATILITY_BREAKOUT = "volatility_breakout"   # Prepare for vol expansion
    MEAN_REVERSION = "mean_reversion"             # Vol likely to revert
    VOLATILITY_MOMENTUM = "volatility_momentum"   # Vol trend continuation
    NEUTRAL = "neutral"                           # No clear vol signal


class IVRegime(Enum):
    """Implied Volatility regime classifications"""
    UNDERVALUED = "undervalued"    # IV rank < 25
    FAIRLY_VALUED = "fairly_valued" # IV rank 25-75
    OVERVALUED = "overvalued"      # IV rank > 75
    EXTREMELY_OVERVALUED = "extremely_overvalued"  # IV rank > 90


@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics"""
    # Realized volatility
    daily_vol: float
    annualized_vol: float
    ewma_vol: float
    yang_zhang_vol: float
    vol_of_vol: float
    
    # ATR metrics
    atr: float
    atr_percent: float
    atr_percentile: float
    
    # Regime metrics
    short_term_vol: float
    long_term_vol: float
    vol_ratio: float
    vol_percentile: float
    
    # Skew metrics
    skewness: float
    kurtosis: float
    upside_vol: float
    downside_vol: float
    vol_skew_ratio: float
    
    # IV metrics (if available)
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    hv_iv_spread: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VolatilityPattern:
    """Detected volatility pattern"""
    pattern_type: str
    signal: str
    direction: str
    strength: float
    confidence: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VolatilityForecast:
    """Volatility forecast"""
    horizon_days: int
    forecasted_vol: float
    confidence_interval: Tuple[float, float]
    forecast_method: str
    regime_forecast: VolatilityRegime
    expected_move: float  # Expected price move based on vol
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'horizon_days': self.horizon_days,
            'forecasted_vol': self.forecasted_vol,
            'confidence_interval': list(self.confidence_interval),
            'forecast_method': self.forecast_method,
            'regime_forecast': self.regime_forecast.value,
            'expected_move': self.expected_move
        }


@dataclass
class VolatilityAnalysis:
    """Complete volatility analysis result"""
    symbol: str
    metrics: VolatilityMetrics
    current_regime: VolatilityRegime
    iv_regime: Optional[IVRegime]
    patterns: List[VolatilityPattern]
    primary_signal: VolatilitySignal
    signal_strength: float
    forecasts: List[VolatilityForecast]
    recommendations: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'metrics': self.metrics.to_dict(),
            'current_regime': self.current_regime.value,
            'iv_regime': self.iv_regime.value if self.iv_regime else None,
            'patterns': [p.to_dict() for p in self.patterns],
            'primary_signal': self.primary_signal.value,
            'signal_strength': self.signal_strength,
            'forecasts': [f.to_dict() for f in self.forecasts],
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class VolatilityAgent:
    """
    V5 Volatility Agent
    Comprehensive volatility analysis and forecasting system
    """
    
    def __init__(self):
        """Initialize the volatility agent"""
        # Configuration parameters
        self.lookback_period = 30
        self.short_vol_period = 10
        self.long_vol_period = 60
        self.vol_percentile_period = 252  # 1 year
        self.iv_rank_period = 252
        
        # Thresholds
        self.vol_spike_threshold = 2.0        # 2x average vol
        self.vol_contraction_threshold = 0.5   # 0.5x average vol
        self.skew_threshold = 0.15            # Significant skew threshold
        self.high_iv_rank_threshold = 80.0
        self.low_iv_rank_threshold = 20.0
        self.extreme_iv_threshold = 95.0
        self.hv_iv_divergence_threshold = 0.15
        
        # Historical data cache
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.iv_cache: Dict[str, List[float]] = {}
        self.analysis_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.calculation_times: Dict[str, List[float]] = {}
        
        logger.info("Volatility Agent V5 initialized")
    
    async def analyze_volatility(self, symbol: str, market_data: Dict[str, Any],
                               include_iv: bool = False, forecast_horizons: List[int] = None) -> VolatilityAnalysis:
        """
        Comprehensive volatility analysis
        
        Args:
            symbol: Trading symbol
            market_data: OHLC price data and optional IV data
            include_iv: Whether to include IV analysis
            forecast_horizons: Days to forecast (default: [1, 5, 21])
        
        Returns:
            Complete volatility analysis
        """
        try:
            start_time = datetime.now()
            
            if forecast_horizons is None:
                forecast_horizons = [1, 5, 21]
            
            # Extract price data
            prices = pd.Series(market_data.get('close_prices', []))
            highs = pd.Series(market_data.get('high_prices', prices))
            lows = pd.Series(market_data.get('low_prices', prices))
            opens = pd.Series(market_data.get('open_prices', prices))
            
            if len(prices) < self.lookback_period + 1:
                raise ValueError(f"Insufficient data: need at least {self.lookback_period + 1} data points")
            
            # Calculate comprehensive volatility metrics
            metrics = self._calculate_comprehensive_metrics(prices, highs, lows, opens, include_iv, market_data)
            
            # Determine volatility regime
            current_regime = self._determine_volatility_regime(metrics)
            
            # IV regime analysis (if available)
            iv_regime = None
            if include_iv and metrics.iv_rank is not None:
                iv_regime = self._determine_iv_regime(metrics.iv_rank)
            
            # Pattern detection
            patterns = self._detect_volatility_patterns(prices, metrics)
            
            # Signal generation
            primary_signal, signal_strength = self._generate_primary_signal(
                current_regime, iv_regime, patterns, metrics
            )
            
            # Volatility forecasting
            forecasts = []
            for horizon in forecast_horizons:
                forecast = self._forecast_volatility(prices, metrics, horizon)
                forecasts.append(forecast)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                current_regime, iv_regime, patterns, primary_signal, metrics
            )
            
            # Create analysis result
            analysis = VolatilityAnalysis(
                symbol=symbol,
                metrics=metrics,
                current_regime=current_regime,
                iv_regime=iv_regime,
                patterns=patterns,
                primary_signal=primary_signal,
                signal_strength=signal_strength,
                forecasts=forecasts,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.analysis_history.append(analysis)
            
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds()
            if 'volatility_analysis' not in self.calculation_times:
                self.calculation_times['volatility_analysis'] = []
            self.calculation_times['volatility_analysis'].append(calc_time)
            
            logger.info(f"Volatility analysis for {symbol}: {current_regime.value}, signal: {primary_signal.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Volatility analysis failed for {symbol}: {str(e)}")
            raise
    
    def _calculate_comprehensive_metrics(self, prices: pd.Series, highs: pd.Series, 
                                       lows: pd.Series, opens: pd.Series,
                                       include_iv: bool, market_data: Dict[str, Any]) -> VolatilityMetrics:
        """Calculate all volatility metrics"""
        
        # Realized volatility calculations
        returns = prices.pct_change().dropna()
        recent_returns = returns.tail(self.lookback_period)
        
        # Basic volatility metrics
        daily_vol = recent_returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        ewma_vol = returns.ewm(span=self.lookback_period).std().iloc[-1] * np.sqrt(252)
        
        # Yang-Zhang volatility (uses OHLC)
        yang_zhang_vol = self._calculate_yang_zhang_volatility(opens, highs, lows, prices)
        
        # Vol of vol
        vol_series = returns.rolling(window=10).std()
        vol_of_vol = vol_series.std() if len(vol_series.dropna()) > 5 else daily_vol * 0.1
        
        # ATR calculations
        atr_metrics = self._calculate_atr(highs, lows, prices)
        
        # Short vs long term volatility
        short_returns = returns.tail(self.short_vol_period)
        long_returns = returns.tail(self.long_vol_period)
        
        short_term_vol = short_returns.std() * np.sqrt(252)
        long_term_vol = long_returns.std() * np.sqrt(252)
        vol_ratio = short_term_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Volatility percentile
        vol_series_full = returns.rolling(window=self.lookback_period).std() * np.sqrt(252)
        if len(vol_series_full.dropna()) >= self.vol_percentile_period:
            vol_percentile = stats.percentileofscore(
                vol_series_full.dropna().tail(self.vol_percentile_period), 
                annualized_vol
            )
        else:
            vol_percentile = stats.percentileofscore(vol_series_full.dropna(), annualized_vol)
        
        # Skew and kurtosis
        skewness = stats.skew(recent_returns)
        kurtosis = stats.kurtosis(recent_returns)
        
        # Upside vs downside volatility
        positive_returns = recent_returns[recent_returns > 0]
        negative_returns = recent_returns[recent_returns < 0]
        
        upside_vol = positive_returns.std() if len(positive_returns) > 1 else 0
        downside_vol = negative_returns.std() if len(negative_returns) > 1 else 0
        vol_skew_ratio = downside_vol / upside_vol if upside_vol > 0 else float('inf')
        
        # IV metrics (if available)
        iv_rank = None
        iv_percentile = None
        hv_iv_spread = None
        
        if include_iv:
            current_iv = market_data.get('implied_volatility')
            if current_iv is not None:
                iv_data = self._calculate_iv_metrics(current_iv, market_data, annualized_vol)
                iv_rank = iv_data.get('iv_rank')
                iv_percentile = iv_data.get('iv_percentile')
                hv_iv_spread = iv_data.get('hv_iv_spread')
        
        return VolatilityMetrics(
            daily_vol=daily_vol,
            annualized_vol=annualized_vol,
            ewma_vol=ewma_vol,
            yang_zhang_vol=yang_zhang_vol,
            vol_of_vol=vol_of_vol,
            atr=atr_metrics['atr'],
            atr_percent=atr_metrics['atr_percent'],
            atr_percentile=atr_metrics['atr_percentile'],
            short_term_vol=short_term_vol,
            long_term_vol=long_term_vol,
            vol_ratio=vol_ratio,
            vol_percentile=vol_percentile,
            skewness=skewness,
            kurtosis=kurtosis,
            upside_vol=upside_vol,
            downside_vol=downside_vol,
            vol_skew_ratio=vol_skew_ratio,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            hv_iv_spread=hv_iv_spread
        )
    
    def _calculate_yang_zhang_volatility(self, opens: pd.Series, highs: pd.Series, 
                                       lows: pd.Series, closes: pd.Series) -> float:
        """Calculate Yang-Zhang volatility estimator"""
        try:
            if len(closes) < 2:
                return 0.0
            
            # Overnight returns (close to open)
            overnight = np.log(opens / closes.shift(1)).dropna()
            
            # Opening gap (open to close)
            rs = np.log(highs / closes) * np.log(highs / opens) + np.log(lows / closes) * np.log(lows / opens)
            
            # Yang-Zhang estimator
            k = 0.34 / (1.34 + (len(closes) + 1) / (len(closes) - 1))
            
            overnight_var = overnight.var()
            rs_var = rs.mean()
            close_var = np.log(closes / opens).var()
            
            yz_var = overnight_var + k * close_var + (1 - k) * rs_var
            
            return np.sqrt(yz_var * 252) if yz_var > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Yang-Zhang volatility calculation failed: {str(e)}")
            # Fallback to simple volatility
            returns = closes.pct_change().dropna()
            return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
    
    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> Dict[str, float]:
        """Calculate ATR and related metrics"""
        try:
            if len(highs) < period + 1:
                return {'atr': 0.0, 'atr_percent': 0.0, 'atr_percentile': 50.0}
            
            # True Range calculation
            tr1 = highs - lows
            tr2 = abs(highs - closes.shift(1))
            tr3 = abs(lows - closes.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR
            atr = true_range.rolling(window=period).mean()
            current_atr = atr.iloc[-1]
            
            # ATR as percentage of price
            current_price = closes.iloc[-1]
            atr_percent = (current_atr / current_price) * 100
            
            # ATR percentile
            atr_percentile = stats.percentileofscore(atr.dropna(), current_atr)
            
            return {
                'atr': current_atr,
                'atr_percent': atr_percent,
                'atr_percentile': atr_percentile
            }
            
        except Exception as e:
            logger.warning(f"ATR calculation failed: {str(e)}")
            return {'atr': 0.0, 'atr_percent': 0.0, 'atr_percentile': 50.0}
    
    def _calculate_iv_metrics(self, current_iv: float, market_data: Dict[str, Any], current_hv: float) -> Dict[str, float]:
        """Calculate IV rank and related metrics"""
        try:
            # Get historical IV data (simplified - in production would use real IV history)
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # Generate synthetic IV history for demonstration
            historical_iv = market_data.get('historical_iv', [])
            if not historical_iv:
                # Fallback: generate based on HV with typical IV premium
                base_iv = current_hv * 1.2  # Typical IV premium
                noise = np.random.normal(0, base_iv * 0.1, self.iv_rank_period)
                historical_iv = [max(0.05, base_iv + n) for n in noise]
            
            # Calculate IV rank
            iv_dataset = historical_iv[-self.iv_rank_period:] + [current_iv]
            iv_array = np.array(iv_dataset)
            
            iv_percentile = stats.percentileofscore(iv_array, current_iv)
            
            min_iv = np.min(iv_array)
            max_iv = np.max(iv_array)
            
            if max_iv != min_iv:
                iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
            else:
                iv_rank = 50.0
            
            # HV-IV relationship
            hv_iv_spread = current_iv - current_hv
            
            return {
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'hv_iv_spread': hv_iv_spread,
                'min_iv': min_iv,
                'max_iv': max_iv,
                'mean_iv': np.mean(iv_array)
            }
            
        except Exception as e:
            logger.warning(f"IV metrics calculation failed: {str(e)}")
            return {'iv_rank': 50.0, 'iv_percentile': 50.0, 'hv_iv_spread': 0.0}
    
    def _determine_volatility_regime(self, metrics: VolatilityMetrics) -> VolatilityRegime:
        """Determine current volatility regime"""
        vol_percentile = metrics.vol_percentile
        
        if vol_percentile >= 90:
            return VolatilityRegime.EXTREMELY_HIGH
        elif vol_percentile >= 75:
            return VolatilityRegime.HIGH
        elif vol_percentile >= 25:
            return VolatilityRegime.NORMAL
        elif vol_percentile >= 10:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.EXTREMELY_LOW
    
    def _determine_iv_regime(self, iv_rank: float) -> IVRegime:
        """Determine IV regime based on IV rank"""
        if iv_rank >= 90:
            return IVRegime.EXTREMELY_OVERVALUED
        elif iv_rank >= 75:
            return IVRegime.OVERVALUED
        elif iv_rank >= 25:
            return IVRegime.FAIRLY_VALUED
        else:
            return IVRegime.UNDERVALUED
    
    def _detect_volatility_patterns(self, prices: pd.Series, metrics: VolatilityMetrics) -> List[VolatilityPattern]:
        """Detect specific volatility patterns"""
        patterns = []
        
        try:
            # Pattern 1: Volatility Squeeze
            if (metrics.vol_percentile < 20 and 
                metrics.vol_ratio < 0.8 and 
                metrics.atr_percentile < 25):
                
                patterns.append(VolatilityPattern(
                    pattern_type='volatility_squeeze',
                    signal='prepare_for_breakout',
                    direction='neutral',
                    strength=(20 - metrics.vol_percentile) / 20,
                    confidence=0.8,
                    description='Low volatility squeeze - breakout expected'
                ))
            
            # Pattern 2: Volatility Spike
            elif (metrics.vol_percentile > 80 and 
                  metrics.vol_ratio > 1.5 and 
                  metrics.atr_percentile > 75):
                
                patterns.append(VolatilityPattern(
                    pattern_type='volatility_spike',
                    signal='mean_reversion_opportunity',
                    direction='sell_volatility',
                    strength=(metrics.vol_percentile - 80) / 20,
                    confidence=0.75,
                    description='High volatility spike - mean reversion expected'
                ))
            
            # Pattern 3: Volatility Expansion
            elif metrics.vol_ratio > 1.3:
                patterns.append(VolatilityPattern(
                    pattern_type='volatility_expansion',
                    signal='momentum_continuation',
                    direction='buy_volatility',
                    strength=min((metrics.vol_ratio - 1.0), 1.0),
                    confidence=0.65,
                    description='Volatility expanding - momentum continuation'
                ))
            
            # Pattern 4: Volatility Compression
            elif metrics.vol_ratio < 0.7:
                patterns.append(VolatilityPattern(
                    pattern_type='volatility_compression',
                    signal='range_trading',
                    direction='neutral',
                    strength=min((1.0 - metrics.vol_ratio), 1.0),
                    confidence=0.6,
                    description='Volatility compressing - range-bound expected'
                ))
            
            # Pattern 5: Negative Skew Pattern
            if metrics.skewness < -0.5:
                patterns.append(VolatilityPattern(
                    pattern_type='negative_skew',
                    signal='tail_risk',
                    direction='buy_protection',
                    strength=min(abs(metrics.skewness), 1.0),
                    confidence=0.7,
                    description='Negative skew - tail risk present'
                ))
            
            # Pattern 6: Volatility Clustering (GARCH effect)
            if metrics.vol_of_vol > metrics.daily_vol * 0.15:
                patterns.append(VolatilityPattern(
                    pattern_type='volatility_clustering',
                    signal='persistence',
                    direction='trend_follow',
                    strength=0.6,
                    confidence=0.5,
                    description='Volatility clustering - high vol persistence'
                ))
            
            # Pattern 7: IV-HV Divergence
            if metrics.hv_iv_spread is not None and abs(metrics.hv_iv_spread) > self.hv_iv_divergence_threshold:
                if metrics.hv_iv_spread > self.hv_iv_divergence_threshold:
                    patterns.append(VolatilityPattern(
                        pattern_type='iv_premium',
                        signal='iv_overvalued',
                        direction='sell_options',
                        strength=min(metrics.hv_iv_spread / 0.3, 1.0),
                        confidence=0.7,
                        description='IV premium to HV - sell options'
                    ))
                else:
                    patterns.append(VolatilityPattern(
                        pattern_type='iv_discount',
                        signal='iv_undervalued', 
                        direction='buy_options',
                        strength=min(abs(metrics.hv_iv_spread) / 0.3, 1.0),
                        confidence=0.7,
                        description='IV discount to HV - buy options'
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
            return []
    
    def _generate_primary_signal(self, vol_regime: VolatilityRegime, iv_regime: Optional[IVRegime],
                                patterns: List[VolatilityPattern], metrics: VolatilityMetrics) -> Tuple[VolatilitySignal, float]:
        """Generate primary volatility trading signal"""
        
        signal_scores = {
            VolatilitySignal.LONG_VOLATILITY: 0.0,
            VolatilitySignal.SHORT_VOLATILITY: 0.0,
            VolatilitySignal.VOLATILITY_BREAKOUT: 0.0,
            VolatilitySignal.MEAN_REVERSION: 0.0,
            VolatilitySignal.VOLATILITY_MOMENTUM: 0.0,
            VolatilitySignal.NEUTRAL: 0.5
        }
        
        # Regime-based signals
        if vol_regime == VolatilityRegime.EXTREMELY_LOW:
            signal_scores[VolatilitySignal.LONG_VOLATILITY] += 0.4
            signal_scores[VolatilitySignal.VOLATILITY_BREAKOUT] += 0.3
        elif vol_regime == VolatilityRegime.LOW:
            signal_scores[VolatilitySignal.LONG_VOLATILITY] += 0.2
            signal_scores[VolatilitySignal.VOLATILITY_BREAKOUT] += 0.2
        elif vol_regime == VolatilityRegime.EXTREMELY_HIGH:
            signal_scores[VolatilitySignal.SHORT_VOLATILITY] += 0.4
            signal_scores[VolatilitySignal.MEAN_REVERSION] += 0.3
        elif vol_regime == VolatilityRegime.HIGH:
            signal_scores[VolatilitySignal.SHORT_VOLATILITY] += 0.2
            signal_scores[VolatilitySignal.MEAN_REVERSION] += 0.2
        
        # IV regime signals
        if iv_regime == IVRegime.EXTREMELY_OVERVALUED:
            signal_scores[VolatilitySignal.SHORT_VOLATILITY] += 0.3
        elif iv_regime == IVRegime.OVERVALUED:
            signal_scores[VolatilitySignal.SHORT_VOLATILITY] += 0.2
        elif iv_regime == IVRegime.UNDERVALUED:
            signal_scores[VolatilitySignal.LONG_VOLATILITY] += 0.2
        
        # Pattern-based signals
        for pattern in patterns:
            if pattern.pattern_type == 'volatility_squeeze':
                signal_scores[VolatilitySignal.VOLATILITY_BREAKOUT] += pattern.strength * 0.3
            elif pattern.pattern_type == 'volatility_spike':
                signal_scores[VolatilitySignal.MEAN_REVERSION] += pattern.strength * 0.3
            elif pattern.pattern_type == 'volatility_expansion':
                signal_scores[VolatilitySignal.VOLATILITY_MOMENTUM] += pattern.strength * 0.2
            elif pattern.pattern_type in ['iv_premium']:
                signal_scores[VolatilitySignal.SHORT_VOLATILITY] += pattern.strength * 0.2
            elif pattern.pattern_type in ['iv_discount']:
                signal_scores[VolatilitySignal.LONG_VOLATILITY] += pattern.strength * 0.2
        
        # Volatility ratio adjustments
        if metrics.vol_ratio > 1.5:
            signal_scores[VolatilitySignal.VOLATILITY_MOMENTUM] += 0.1
        elif metrics.vol_ratio < 0.7:
            signal_scores[VolatilitySignal.MEAN_REVERSION] += 0.1
        
        # Find dominant signal
        primary_signal = max(signal_scores.keys(), key=lambda k: signal_scores[k])
        signal_strength = signal_scores[primary_signal]
        
        # Normalize strength to 0-1
        signal_strength = min(1.0, signal_strength)
        
        return primary_signal, signal_strength
    
    def _forecast_volatility(self, prices: pd.Series, metrics: VolatilityMetrics, horizon_days: int) -> VolatilityForecast:
        """Forecast volatility for given horizon"""
        try:
            current_vol = metrics.annualized_vol
            
            # Simple mean reversion forecast
            long_term_mean = metrics.long_term_vol
            reversion_speed = 0.1  # Daily mean reversion speed
            
            # Forecast assuming mean reversion
            forecast_vol = current_vol * np.exp(-reversion_speed * horizon_days) + \
                          long_term_mean * (1 - np.exp(-reversion_speed * horizon_days))
            
            # Add uncertainty based on vol of vol
            vol_uncertainty = metrics.vol_of_vol * np.sqrt(horizon_days / 252)
            
            confidence_interval = (
                max(0.05, forecast_vol - 1.96 * vol_uncertainty),
                forecast_vol + 1.96 * vol_uncertainty
            )
            
            # Forecast regime
            if forecast_vol > long_term_mean * 1.2:
                regime_forecast = VolatilityRegime.HIGH
            elif forecast_vol < long_term_mean * 0.8:
                regime_forecast = VolatilityRegime.LOW
            else:
                regime_forecast = VolatilityRegime.NORMAL
            
            # Expected move (1 standard deviation)
            current_price = prices.iloc[-1]
            expected_move = current_price * forecast_vol / np.sqrt(252 / horizon_days)
            
            return VolatilityForecast(
                horizon_days=horizon_days,
                forecasted_vol=forecast_vol,
                confidence_interval=confidence_interval,
                forecast_method='mean_reversion',
                regime_forecast=regime_forecast,
                expected_move=expected_move
            )
            
        except Exception as e:
            logger.warning(f"Volatility forecast failed for {horizon_days}d: {str(e)}")
            return VolatilityForecast(
                horizon_days=horizon_days,
                forecasted_vol=metrics.annualized_vol,
                confidence_interval=(metrics.annualized_vol * 0.8, metrics.annualized_vol * 1.2),
                forecast_method='fallback',
                regime_forecast=VolatilityRegime.NORMAL,
                expected_move=prices.iloc[-1] * 0.02
            )
    
    def _generate_recommendations(self, vol_regime: VolatilityRegime, iv_regime: Optional[IVRegime],
                                 patterns: List[VolatilityPattern], primary_signal: VolatilitySignal,
                                 metrics: VolatilityMetrics) -> List[str]:
        """Generate trading recommendations based on volatility analysis"""
        recommendations = []
        
        # Regime-based recommendations
        if vol_regime == VolatilityRegime.EXTREMELY_LOW:
            recommendations.append("Consider buying straddles/strangles - volatility likely to increase")
            recommendations.append("Avoid selling premium - volatility at historical lows")
        elif vol_regime == VolatilityRegime.EXTREMELY_HIGH:
            recommendations.append("Consider selling premium - volatility likely to decrease")
            recommendations.append("Be cautious with long volatility positions")
        
        # IV-specific recommendations
        if iv_regime == IVRegime.EXTREMELY_OVERVALUED:
            recommendations.append("IV extremely high - consider selling options strategies")
            recommendations.append("Look for covered call or cash-secured put opportunities")
        elif iv_regime == IVRegime.UNDERVALUED:
            recommendations.append("IV relatively low - consider buying options")
            recommendations.append("Good time for protective puts or speculative calls")
        
        # Pattern-specific recommendations
        for pattern in patterns:
            if pattern.pattern_type == 'volatility_squeeze' and pattern.strength > 0.6:
                recommendations.append("Volatility squeeze detected - prepare for directional move")
            elif pattern.pattern_type == 'volatility_spike' and pattern.strength > 0.7:
                recommendations.append("Volatility spike - consider selling premium strategies")
            elif pattern.pattern_type == 'iv_premium' and pattern.strength > 0.5:
                recommendations.append("IV premium detected - sell options, buy underlying")
            elif pattern.pattern_type == 'negative_skew':
                recommendations.append("Negative skew present - consider tail risk hedging")
        
        # Risk management recommendations
        if metrics.vol_of_vol > metrics.daily_vol * 0.2:
            recommendations.append("High vol-of-vol - use dynamic hedging strategies")
        
        if abs(metrics.skewness) > 1.0:
            recommendations.append("Extreme skew detected - be aware of tail risks")
        
        if metrics.kurtosis > 4:
            recommendations.append("Fat tails detected - size positions conservatively")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_analyses = len(self.analysis_history)
        
        if total_analyses == 0:
            return {
                'total_analyses': 0,
                'average_calc_time': 0.0,
                'regime_distribution': {},
                'signal_distribution': {}
            }
        
        # Calculate performance metrics
        regime_counts = {}
        signal_counts = {}
        
        for analysis in self.analysis_history:
            regime = analysis.current_regime.value
            signal = analysis.primary_signal.value
            
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        # Average calculation time
        avg_calc_time = 0.0
        if 'volatility_analysis' in self.calculation_times:
            times = self.calculation_times['volatility_analysis']
            avg_calc_time = sum(times) / len(times) if times else 0.0
        
        return {
            'total_analyses': total_analyses,
            'average_calc_time_seconds': avg_calc_time,
            'regime_distribution': regime_counts,
            'signal_distribution': signal_counts,
            'cache_size': len(self.price_cache),
            'supported_regimes': [r.value for r in VolatilityRegime],
            'supported_signals': [s.value for s in VolatilitySignal]
        }


# Create global instance
volatility_agent = VolatilityAgent()