"""
Market Regime Classification Agent - Enhanced for GoldenSignalsAI V5
Continuously classifies market regime (Bull/Bear/Sideways/Crisis) for adaptive strategy selection
Migrated from archive with production enhancements and V5 integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd
from scipy import stats

from core.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    TRANSITION = "transition"  # New: regime in transition
    UNKNOWN = "unknown"


class RegimeConfidence(Enum):
    """Confidence levels for regime classification"""
    VERY_HIGH = "very_high"    # > 80%
    HIGH = "high"              # 60-80%
    MODERATE = "moderate"      # 40-60%
    LOW = "low"               # 20-40%
    VERY_LOW = "very_low"     # < 20%


@dataclass
class RegimeIndicators:
    """Container for regime indicators with enhanced metrics"""
    # Volatility indicators
    vix_level: float = 20.0
    vix_change: float = 0.0
    vix_percentile: float = 50.0
    
    # Market breadth
    advance_decline_ratio: float = 1.0
    new_highs_lows: float = 0.0
    up_down_volume: float = 1.0
    
    # Momentum indicators
    spy_momentum_1d: float = 0.0
    spy_momentum_5d: float = 0.0
    spy_momentum_20d: float = 0.0
    
    # Volume analysis
    volume_ratio: float = 1.0
    volume_surge: float = 1.0
    
    # Cross-asset signals
    sector_correlation: float = 0.5
    bond_equity_correlation: float = 0.0
    dollar_strength: float = 0.0
    
    # Credit and risk
    credit_spread: float = 1.0
    term_structure: float = 0.0  # Yield curve slope
    
    # Options flow
    put_call_ratio: float = 1.0
    vix_term_structure: float = 0.0
    
    # Market structure
    liquidity_score: float = 1.0
    market_depth: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)
    
    def normalize(self) -> Dict[str, float]:
        """Normalize indicators for ML processing"""
        normalized = {}
        for key, value in self.to_dict().items():
            # Simple min-max normalization based on typical ranges
            if 'vix' in key:
                normalized[key] = min(max(value / 100.0, 0), 1)
            elif 'ratio' in key:
                normalized[key] = min(max(value / 5.0, 0), 1)
            elif 'momentum' in key:
                normalized[key] = min(max((value + 0.1) / 0.2, 0), 1)
            else:
                normalized[key] = min(max(value, 0), 1)
        return normalized


@dataclass
class RegimeTransition:
    """Information about regime transitions"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_probability: float
    expected_duration_days: int
    confidence: float
    key_catalysts: List[str]


class MarketRegimeAgent:
    """
    Enhanced Market Regime Classification Agent for GoldenSignalsAI V5
    
    Features:
    - Multi-factor regime classification
    - Adaptive threshold learning
    - Regime transition analysis
    - Strategy recommendations
    - Risk assessment integration
    - Real-time volatility forecasting
    """
    
    def __init__(
        self,
        name: str = "MarketRegimeAgent",
        lookback_days: int = 252,
        adaptation_frequency: int = 50,
        min_confidence_threshold: float = 0.6
    ):
        """
        Initialize Market Regime Agent
        
        Args:
            name: Agent name
            lookback_days: Days of history for regime analysis
            adaptation_frequency: How often to adapt thresholds
            min_confidence_threshold: Minimum confidence for regime changes
        """
        self.name = name
        self.lookback_days = lookback_days
        self.adaptation_frequency = adaptation_frequency
        self.min_confidence_threshold = min_confidence_threshold
        
        # Current state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.confidence_level = RegimeConfidence.VERY_LOW
        
        # Historical tracking
        self.regime_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.indicators = RegimeIndicators()
        
        # Adaptive thresholds
        self.thresholds = self._initialize_thresholds()
        self.adaptation_counter = 0
        
        # Regime statistics
        self.regime_stats = {
            regime: {
                'count': 0,
                'avg_duration': 0,
                'success_rate': 0.0,
                'volatility': 0.0
            }
            for regime in MarketRegime
        }
        
        logger.info(f"Initialized {name} with {lookback_days} day lookback")
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize adaptive thresholds"""
        return {
            # VIX thresholds
            'vix_crisis': 35.0,
            'vix_high': 25.0,
            'vix_normal': 20.0,
            'vix_low': 15.0,
            
            # Breadth thresholds
            'breadth_strong_bull': 2.5,
            'breadth_bull': 1.5,
            'breadth_bear': 0.67,
            'breadth_strong_bear': 0.4,
            
            # Momentum thresholds
            'momentum_strong_bull': 0.03,
            'momentum_bull': 0.015,
            'momentum_bear': -0.015,
            'momentum_strong_bear': -0.03,
            
            # Correlation thresholds
            'correlation_crisis': 0.9,
            'correlation_high': 0.7,
            'correlation_normal': 0.5,
            
            # Volume thresholds
            'volume_spike': 2.0,
            'volume_elevated': 1.5,
            'volume_normal': 1.0,
            
            # Credit spread thresholds
            'credit_crisis': 3.0,
            'credit_elevated': 1.5,
            'credit_normal': 1.0
        }
    
    async def analyze_regime(
        self,
        market_data: Dict[str, Any],
        include_forecast: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive regime analysis
        
        Args:
            market_data: Market indicators and data
            include_forecast: Whether to include forecasting
        
        Returns:
            Complete regime analysis
        """
        try:
            # Update indicators
            self._update_indicators(market_data)
            
            # Calculate regime probabilities
            regime_scores = self._calculate_regime_probabilities()
            
            # Determine regime and confidence
            new_regime, confidence = self._determine_regime(regime_scores)
            
            # Check for regime transitions
            transition_info = self._analyze_transition(new_regime)
            
            # Update state
            self._update_regime_state(new_regime, confidence, regime_scores)
            
            # Generate analysis result
            result = {
                'regime': self.current_regime.value,
                'confidence': self.regime_confidence,
                'confidence_level': self.confidence_level.value,
                'regime_probabilities': {r.value: s for r, s in regime_scores.items()},
                'indicators': self.indicators.to_dict(),
                'transition_analysis': transition_info,
                'risk_assessment': self._assess_risk(),
                'strategy_recommendations': self._recommend_strategies(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add forecasting if requested
            if include_forecast:
                result['forecasts'] = {
                    'volatility': self._forecast_volatility(),
                    'regime_duration': self._estimate_regime_duration(),
                    'transition_probabilities': self._calculate_transition_probabilities()
                }
            
            # Performance tracking
            self._track_performance(result)
            
            # Adaptive learning
            if self.adaptation_counter % self.adaptation_frequency == 0:
                await self._adapt_thresholds()
            self.adaptation_counter += 1
            
            logger.info(
                f"Regime: {self.current_regime.value.upper()} "
                f"({self.regime_confidence:.1%} confidence)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Regime analysis failed: {str(e)}")
            return {
                'regime': MarketRegime.UNKNOWN.value,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_indicators(self, market_data: Dict[str, Any]):
        """Update regime indicators from market data"""
        # VIX indicators
        self.indicators.vix_level = market_data.get('vix', 20.0)
        self.indicators.vix_change = market_data.get('vix_change', 0.0)
        self.indicators.vix_percentile = market_data.get('vix_percentile', 50.0)
        
        # Market breadth
        self.indicators.advance_decline_ratio = market_data.get('advance_decline_ratio', 1.0)
        self.indicators.new_highs_lows = market_data.get('new_highs_lows', 0.0)
        self.indicators.up_down_volume = market_data.get('up_down_volume', 1.0)
        
        # Momentum
        self.indicators.spy_momentum_1d = market_data.get('spy_momentum_1d', 0.0)
        self.indicators.spy_momentum_5d = market_data.get('spy_momentum_5d', 0.0)
        self.indicators.spy_momentum_20d = market_data.get('spy_momentum_20d', 0.0)
        
        # Volume
        self.indicators.volume_ratio = market_data.get('volume_ratio', 1.0)
        self.indicators.volume_surge = market_data.get('volume_surge', 1.0)
        
        # Cross-asset
        self.indicators.sector_correlation = market_data.get('sector_correlation', 0.5)
        self.indicators.bond_equity_correlation = market_data.get('bond_equity_correlation', 0.0)
        self.indicators.dollar_strength = market_data.get('dollar_strength', 0.0)
        
        # Credit and risk
        self.indicators.credit_spread = market_data.get('credit_spread', 1.0)
        self.indicators.term_structure = market_data.get('term_structure', 0.0)
        
        # Options
        self.indicators.put_call_ratio = market_data.get('put_call_ratio', 1.0)
        self.indicators.vix_term_structure = market_data.get('vix_term_structure', 0.0)
        
        # Market structure
        self.indicators.liquidity_score = market_data.get('liquidity_score', 1.0)
        self.indicators.market_depth = market_data.get('market_depth', 1.0)
    
    def _calculate_regime_probabilities(self) -> Dict[MarketRegime, float]:
        """Calculate probability for each regime using enhanced scoring"""
        scores = {
            MarketRegime.BULL: self._score_bull_regime(),
            MarketRegime.BEAR: self._score_bear_regime(),
            MarketRegime.SIDEWAYS: self._score_sideways_regime(),
            MarketRegime.CRISIS: self._score_crisis_regime(),
            MarketRegime.TRANSITION: self._score_transition_regime()
        }
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _score_bull_regime(self) -> float:
        """Enhanced bull regime scoring"""
        score = 0.0
        
        # VIX conditions (25% weight)
        if self.indicators.vix_level < self.thresholds['vix_low']:
            score += 0.15
        elif self.indicators.vix_level < self.thresholds['vix_normal']:
            score += 0.10
        
        if self.indicators.vix_change < -0.05:  # VIX declining
            score += 0.05
        
        if self.indicators.vix_percentile < 30:  # Low volatility regime
            score += 0.05
        
        # Market breadth (25% weight)
        if self.indicators.advance_decline_ratio > self.thresholds['breadth_strong_bull']:
            score += 0.15
        elif self.indicators.advance_decline_ratio > self.thresholds['breadth_bull']:
            score += 0.10
        
        if self.indicators.new_highs_lows > 0.5:
            score += 0.05
        
        if self.indicators.up_down_volume > 1.5:
            score += 0.05
        
        # Momentum (20% weight)
        if self.indicators.spy_momentum_20d > self.thresholds['momentum_strong_bull']:
            score += 0.10
        elif self.indicators.spy_momentum_20d > self.thresholds['momentum_bull']:
            score += 0.05
        
        if self.indicators.spy_momentum_5d > 0.01:
            score += 0.05
        
        # Cross-asset signals (15% weight)
        if self.indicators.sector_correlation < self.thresholds['correlation_normal']:
            score += 0.08  # Healthy diversification
        
        if self.indicators.bond_equity_correlation < -0.3:
            score += 0.04  # Risk-on regime
        
        if self.indicators.credit_spread < self.thresholds['credit_normal']:
            score += 0.03
        
        # Volume characteristics (10% weight)
        if 0.8 <= self.indicators.volume_ratio <= 1.3:  # Orderly volume
            score += 0.05
        
        if self.indicators.liquidity_score > 0.8:
            score += 0.05
        
        # Options signals (5% weight)
        if self.indicators.put_call_ratio < 0.8:  # Low fear
            score += 0.03
        
        if self.indicators.vix_term_structure > 0:  # Contango
            score += 0.02
        
        return score
    
    def _score_bear_regime(self) -> float:
        """Enhanced bear regime scoring"""
        score = 0.0
        
        # VIX conditions (25% weight)
        if self.thresholds['vix_normal'] <= self.indicators.vix_level < self.thresholds['vix_crisis']:
            score += 0.10
        
        if self.indicators.vix_change > 0.05:  # VIX rising
            score += 0.08
        
        if self.indicators.vix_percentile > 70:
            score += 0.07
        
        # Market breadth (25% weight)
        if self.indicators.advance_decline_ratio < self.thresholds['breadth_strong_bear']:
            score += 0.15
        elif self.indicators.advance_decline_ratio < self.thresholds['breadth_bear']:
            score += 0.10
        
        if self.indicators.new_highs_lows < -0.5:
            score += 0.05
        
        if self.indicators.up_down_volume < 0.7:
            score += 0.05
        
        # Momentum (20% weight)
        if self.indicators.spy_momentum_20d < self.thresholds['momentum_strong_bear']:
            score += 0.10
        elif self.indicators.spy_momentum_20d < self.thresholds['momentum_bear']:
            score += 0.05
        
        if self.indicators.spy_momentum_5d < -0.01:
            score += 0.05
        
        # Cross-asset signals (15% weight)
        if self.indicators.sector_correlation > self.thresholds['correlation_high']:
            score += 0.08  # Risk-off correlation
        
        if self.indicators.bond_equity_correlation > 0.3:
            score += 0.04  # Flight to quality
        
        if self.indicators.credit_spread > self.thresholds['credit_elevated']:
            score += 0.03
        
        # Volume characteristics (10% weight)
        if self.indicators.volume_ratio > self.thresholds['volume_elevated']:
            score += 0.05  # Panic selling
        
        if self.indicators.liquidity_score < 0.6:
            score += 0.05
        
        # Options signals (5% weight)
        if self.indicators.put_call_ratio > 1.2:  # High fear
            score += 0.03
        
        if self.indicators.vix_term_structure < -0.1:  # Backwardation
            score += 0.02
        
        return score
    
    def _score_sideways_regime(self) -> float:
        """Enhanced sideways regime scoring"""
        score = 0.0
        
        # VIX in normal range (20% weight)
        vix_mid = (self.thresholds['vix_low'] + self.thresholds['vix_normal']) / 2
        if abs(self.indicators.vix_level - vix_mid) < 3:
            score += 0.15
        
        if abs(self.indicators.vix_change) < 0.02:  # Stable VIX
            score += 0.05
        
        # Balanced breadth (25% weight)
        if 0.8 <= self.indicators.advance_decline_ratio <= 1.2:
            score += 0.15
        
        if abs(self.indicators.new_highs_lows) < 0.2:
            score += 0.05
        
        if 0.9 <= self.indicators.up_down_volume <= 1.1:
            score += 0.05
        
        # Low momentum (25% weight)
        if abs(self.indicators.spy_momentum_20d) < 0.005:
            score += 0.15
        
        if abs(self.indicators.spy_momentum_5d) < 0.005:
            score += 0.10
        
        # Moderate correlations (15% weight)
        if 0.4 <= self.indicators.sector_correlation <= 0.6:
            score += 0.08
        
        if abs(self.indicators.bond_equity_correlation) < 0.2:
            score += 0.04
        
        if 0.8 <= self.indicators.credit_spread <= 1.2:
            score += 0.03
        
        # Normal volume (10% weight)
        if 0.8 <= self.indicators.volume_ratio <= 1.2:
            score += 0.05
        
        if self.indicators.liquidity_score > 0.7:
            score += 0.05
        
        # Neutral options sentiment (5% weight)
        if 0.9 <= self.indicators.put_call_ratio <= 1.1:
            score += 0.03
        
        if abs(self.indicators.vix_term_structure) < 0.05:
            score += 0.02
        
        return score
    
    def _score_crisis_regime(self) -> float:
        """Enhanced crisis regime scoring"""
        score = 0.0
        
        # Extreme VIX (30% weight)
        if self.indicators.vix_level > self.thresholds['vix_crisis']:
            score += 0.20
        
        if self.indicators.vix_change > 0.2:  # VIX spiking
            score += 0.10
        
        # Extreme market conditions (30% weight)
        if self.indicators.advance_decline_ratio < 0.2:
            score += 0.15
        
        if self.indicators.new_highs_lows < -0.8:
            score += 0.08
        
        if self.indicators.spy_momentum_5d < -0.05:
            score += 0.07
        
        # High correlations (20% weight)
        if self.indicators.sector_correlation > self.thresholds['correlation_crisis']:
            score += 0.15
        
        if self.indicators.bond_equity_correlation > 0.5:
            score += 0.05
        
        # Volume spikes and liquidity (15% weight)
        if self.indicators.volume_ratio > self.thresholds['volume_spike']:
            score += 0.08
        
        if self.indicators.liquidity_score < 0.4:
            score += 0.07
        
        # Credit stress (5% weight)
        if self.indicators.credit_spread > self.thresholds['credit_crisis']:
            score += 0.05
        
        return score
    
    def _score_transition_regime(self) -> float:
        """Score regime transition state"""
        if len(self.regime_history) < 5:
            return 0.0
        
        score = 0.0
        
        # Check for recent regime instability
        recent_regimes = [h['regime'] for h in self.regime_history[-5:]]
        unique_regimes = len(set(recent_regimes))
        
        if unique_regimes > 2:
            score += 0.3  # Multiple regime changes
        
        # Check for conflicting indicators
        vix_score = self._score_crisis_regime() + self._score_bear_regime()
        momentum_score = self._score_bull_regime()
        
        if vix_score > 0.3 and momentum_score > 0.3:
            score += 0.2  # Conflicting signals
        
        # Check confidence levels
        if len(self.regime_history) > 0:
            recent_confidence = np.mean([h['confidence'] for h in self.regime_history[-3:]])
            if recent_confidence < 0.6:
                score += 0.2
        
        return score
    
    def _determine_regime(
        self,
        regime_scores: Dict[MarketRegime, float]
    ) -> Tuple[MarketRegime, float]:
        """Determine regime with enhanced logic"""
        if not regime_scores:
            return MarketRegime.UNKNOWN, 0.0
        
        # Get best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        # Apply regime persistence logic
        if (self.regime_history and 
            len(self.regime_history) > 0 and 
            confidence < self.min_confidence_threshold):
            
            # Stay with current regime if confidence is low
            last_regime = MarketRegime(self.regime_history[-1]['regime'])
            if last_regime != MarketRegime.UNKNOWN:
                return last_regime, regime_scores.get(last_regime, 0.5)
        
        return best_regime, confidence
    
    def _analyze_transition(self, new_regime: MarketRegime) -> Optional[RegimeTransition]:
        """Analyze potential regime transition"""
        if not self.regime_history or len(self.regime_history) == 0:
            return None
        
        current_regime = MarketRegime(self.regime_history[-1]['regime'])
        
        if new_regime != current_regime:
            # Potential transition detected
            return RegimeTransition(
                from_regime=current_regime,
                to_regime=new_regime,
                transition_probability=self.regime_confidence,
                expected_duration_days=self._estimate_regime_duration_days(new_regime),
                confidence=self.regime_confidence,
                key_catalysts=self._identify_transition_catalysts(current_regime, new_regime)
            )
        
        return None
    
    def _update_regime_state(
        self,
        new_regime: MarketRegime,
        confidence: float,
        regime_scores: Dict[MarketRegime, float]
    ):
        """Update regime state and history"""
        self.current_regime = new_regime
        self.regime_confidence = confidence
        
        # Update confidence level
        if confidence > 0.8:
            self.confidence_level = RegimeConfidence.VERY_HIGH
        elif confidence > 0.6:
            self.confidence_level = RegimeConfidence.HIGH
        elif confidence > 0.4:
            self.confidence_level = RegimeConfidence.MODERATE
        elif confidence > 0.2:
            self.confidence_level = RegimeConfidence.LOW
        else:
            self.confidence_level = RegimeConfidence.VERY_LOW
        
        # Add to history
        regime_record = {
            'timestamp': datetime.now().isoformat(),
            'regime': new_regime.value,
            'confidence': confidence,
            'confidence_level': self.confidence_level.value,
            'indicators': self.indicators.to_dict(),
            'regime_scores': {r.value: s for r, s in regime_scores.items()}
        }
        
        self.regime_history.append(regime_record)
        
        # Limit history size
        if len(self.regime_history) > self.lookback_days:
            self.regime_history = self.regime_history[-self.lookback_days:]
    
    def _assess_risk(self) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        if self.current_regime == MarketRegime.CRISIS:
            risk_level = "EXTREME"
            risk_score = 95
        elif self.current_regime == MarketRegime.BEAR and self.indicators.vix_level > 30:
            risk_level = "HIGH"
            risk_score = 80
        elif self.current_regime == MarketRegime.BEAR:
            risk_level = "ELEVATED"
            risk_score = 65
        elif self.current_regime == MarketRegime.TRANSITION:
            risk_level = "ELEVATED"
            risk_score = 60
        elif self.current_regime == MarketRegime.SIDEWAYS:
            risk_level = "MODERATE"
            risk_score = 45
        else:  # BULL
            risk_level = "LOW"
            risk_score = 25
        
        # Adjust based on volatility
        if self.indicators.vix_level > 25:
            risk_score += 10
        elif self.indicators.vix_level < 15:
            risk_score -= 5
        
        # Adjust based on liquidity
        if self.indicators.liquidity_score < 0.5:
            risk_score += 15
        
        risk_score = max(0, min(100, risk_score))
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'key_risks': self._identify_key_risks(),
            'risk_factors': {
                'volatility_risk': min(self.indicators.vix_level / 50.0, 1.0),
                'liquidity_risk': 1.0 - self.indicators.liquidity_score,
                'correlation_risk': self.indicators.sector_correlation,
                'momentum_risk': abs(self.indicators.spy_momentum_5d) * 10
            }
        }
    
    def _recommend_strategies(self) -> List[str]:
        """Enhanced strategy recommendations"""
        strategies = []
        
        if self.current_regime == MarketRegime.BULL:
            strategies.extend([
                "momentum_long",
                "buy_breakouts",
                "sell_volatility",
                "growth_stocks",
                "call_spreads",
                "covered_calls"
            ])
            
            if self.indicators.vix_level < 15:
                strategies.append("short_volatility")
                
        elif self.current_regime == MarketRegime.BEAR:
            strategies.extend([
                "defensive_positions",
                "buy_volatility",
                "short_momentum",
                "quality_stocks",
                "put_spreads",
                "protective_puts"
            ])
            
            if self.indicators.credit_spread > 2.0:
                strategies.append("credit_hedges")
                
        elif self.current_regime == MarketRegime.SIDEWAYS:
            strategies.extend([
                "mean_reversion",
                "range_trading",
                "iron_condors",
                "pairs_trading",
                "theta_strategies",
                "straddle_shorts"
            ])
            
        elif self.current_regime == MarketRegime.CRISIS:
            strategies.extend([
                "risk_off",
                "tail_hedges",
                "cash_preservation",
                "safe_havens",
                "long_volatility",
                "crisis_alpha"
            ])
            
        elif self.current_regime == MarketRegime.TRANSITION:
            strategies.extend([
                "wait_and_see",
                "reduced_position_size",
                "hedge_portfolio",
                "volatility_trading",
                "regime_momentum"
            ])
        
        return strategies
    
    def _forecast_volatility(self) -> Dict[str, float]:
        """Enhanced volatility forecasting"""
        base_vol = self.indicators.vix_level
        
        # Regime-based adjustments
        if self.current_regime == MarketRegime.CRISIS:
            multipliers = {'1_day': 1.2, '1_week': 1.1, '1_month': 0.8}
        elif self.current_regime == MarketRegime.BEAR:
            multipliers = {'1_day': 1.1, '1_week': 1.05, '1_month': 0.9}
        elif self.current_regime == MarketRegime.BULL:
            multipliers = {'1_day': 0.95, '1_week': 0.9, '1_month': 0.85}
        elif self.current_regime == MarketRegime.TRANSITION:
            multipliers = {'1_day': 1.15, '1_week': 1.1, '1_month': 1.0}
        else:  # SIDEWAYS
            multipliers = {'1_day': 1.0, '1_week': 0.98, '1_month': 0.95}
        
        return {
            horizon: base_vol * mult
            for horizon, mult in multipliers.items()
        }
    
    def _estimate_regime_duration(self) -> Dict[str, Any]:
        """Estimate regime duration with enhanced logic"""
        typical_durations = {
            MarketRegime.BULL: 180,
            MarketRegime.BEAR: 90,
            MarketRegime.SIDEWAYS: 60,
            MarketRegime.CRISIS: 21,
            MarketRegime.TRANSITION: 14
        }
        
        base_duration = typical_durations.get(self.current_regime, 60)
        current_duration = self._get_current_regime_duration()
        
        # Adjust based on confidence
        if self.regime_confidence > 0.8:
            expected_remaining = int(base_duration * 1.2 - current_duration)
        elif self.regime_confidence < 0.4:
            expected_remaining = int(base_duration * 0.6 - current_duration)
        else:
            expected_remaining = base_duration - current_duration
        
        expected_remaining = max(1, expected_remaining)
        
        return {
            'expected_remaining_days': expected_remaining,
            'total_expected_days': base_duration,
            'current_duration_days': current_duration,
            'confidence': self.regime_confidence
        }
    
    def _calculate_transition_probabilities(self) -> Dict[str, float]:
        """Calculate regime transition probabilities"""
        # Enhanced transition matrix
        base_transitions = {
            MarketRegime.BULL: {
                'to_bear': 0.12, 'to_sideways': 0.25, 'to_crisis': 0.03, 
                'to_transition': 0.10, 'stay': 0.50
            },
            MarketRegime.BEAR: {
                'to_bull': 0.15, 'to_sideways': 0.30, 'to_crisis': 0.15, 
                'to_transition': 0.15, 'stay': 0.25
            },
            MarketRegime.SIDEWAYS: {
                'to_bull': 0.30, 'to_bear': 0.25, 'to_crisis': 0.05, 
                'to_transition': 0.15, 'stay': 0.25
            },
            MarketRegime.CRISIS: {
                'to_bull': 0.05, 'to_bear': 0.40, 'to_sideways': 0.30, 
                'to_transition': 0.15, 'stay': 0.10
            },
            MarketRegime.TRANSITION: {
                'to_bull': 0.25, 'to_bear': 0.25, 'to_sideways': 0.25, 
                'to_crisis': 0.10, 'stay': 0.15
            }
        }
        
        return base_transitions.get(self.current_regime, {})
    
    def _estimate_regime_duration_days(self, regime: MarketRegime) -> int:
        """Estimate duration for a specific regime"""
        durations = {
            MarketRegime.BULL: 180,
            MarketRegime.BEAR: 90,
            MarketRegime.SIDEWAYS: 60,
            MarketRegime.CRISIS: 21,
            MarketRegime.TRANSITION: 14
        }
        return durations.get(regime, 60)
    
    def _identify_transition_catalysts(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime
    ) -> List[str]:
        """Identify key catalysts for regime transition"""
        catalysts = []
        
        if to_regime == MarketRegime.CRISIS:
            catalysts.extend(['VIX spike', 'Liquidity crisis', 'Correlation surge'])
        elif to_regime == MarketRegime.BEAR:
            catalysts.extend(['Breadth deterioration', 'Momentum breakdown'])
        elif to_regime == MarketRegime.BULL:
            catalysts.extend(['VIX compression', 'Breadth expansion', 'Momentum pickup'])
        elif to_regime == MarketRegime.SIDEWAYS:
            catalysts.extend(['Momentum neutralization', 'VIX normalization'])
        
        return catalysts
    
    def _identify_key_risks(self) -> List[str]:
        """Identify current key market risks"""
        risks = []
        
        if self.indicators.vix_level > 30:
            risks.append('High volatility')
        if self.indicators.sector_correlation > 0.8:
            risks.append('High correlation')
        if self.indicators.liquidity_score < 0.5:
            risks.append('Liquidity concerns')
        if self.indicators.credit_spread > 2.0:
            risks.append('Credit stress')
        if abs(self.indicators.spy_momentum_5d) > 0.03:
            risks.append('Momentum extremes')
        
        return risks
    
    def _get_current_regime_duration(self) -> int:
        """Get duration of current regime"""
        if not self.regime_history:
            return 0
        
        count = 0
        for record in reversed(self.regime_history):
            if record['regime'] == self.current_regime.value:
                count += 1
            else:
                break
        
        return count
    
    def _track_performance(self, result: Dict[str, Any]):
        """Track performance for adaptive learning"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'regime': result['regime'],
            'confidence': result['confidence'],
            'indicators': result['indicators']
        })
        
        # Limit performance history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    async def _adapt_thresholds(self):
        """Adaptive threshold learning"""
        if len(self.regime_history) < 100:
            return
        
        logger.info("Adapting regime thresholds based on performance")
        
        try:
            # Analyze recent VIX levels by regime
            regime_vix = {}
            for record in self.regime_history[-100:]:
                regime = record['regime']
                vix = record['indicators']['vix_level']
                
                if regime not in regime_vix:
                    regime_vix[regime] = []
                regime_vix[regime].append(vix)
            
            # Update VIX thresholds
            if 'bull' in regime_vix and len(regime_vix['bull']) > 10:
                bull_vix = np.percentile(regime_vix['bull'], 75)
                self.thresholds['vix_low'] = max(10, bull_vix * 0.8)
                self.thresholds['vix_normal'] = bull_vix * 1.2
            
            if 'bear' in regime_vix and len(regime_vix['bear']) > 10:
                bear_vix = np.percentile(regime_vix['bear'], 25)
                self.thresholds['vix_high'] = bear_vix * 0.9
            
            if 'crisis' in regime_vix and len(regime_vix['crisis']) > 5:
                crisis_vix = np.percentile(regime_vix['crisis'], 25)
                self.thresholds['vix_crisis'] = max(30, crisis_vix * 0.9)
            
            logger.info(f"Updated thresholds: {self.thresholds}")
            
        except Exception as e:
            logger.error(f"Threshold adaptation failed: {e}")


# Global instance
market_regime_agent = MarketRegimeAgent()