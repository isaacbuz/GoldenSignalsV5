"""
Gamma Exposure Agent - Advanced Options Greeks Analysis
Analyzes gamma positioning, dealer exposure, and market pinning effects
Migrated from archive with production enhancements
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.logging import get_logger

logger = get_logger(__name__)


class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"


class GammaRegime(Enum):
    """Market gamma regime classification"""
    SHORT_GAMMA = "short_gamma"  # Dealers short gamma - volatility amplifying
    LONG_GAMMA = "long_gamma"    # Dealers long gamma - volatility dampening
    NEUTRAL = "neutral"           # Balanced gamma exposure


class PinningStrength(Enum):
    """Gamma pinning strength classification"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class GammaLevel:
    """Significant gamma level data"""
    strike: float
    gamma_exposure: float
    option_type: OptionType
    proximity: float
    open_interest: int
    
    @property
    def is_significant(self) -> bool:
        """Check if gamma level is significant"""
        return abs(self.gamma_exposure) > 100000


class GammaExposureAgent:
    """
    Advanced agent for analyzing gamma exposure and its market impact.
    
    Key capabilities:
    - Dealer gamma exposure calculation
    - Gamma pinning detection
    - Volatility regime identification
    - Black-Scholes Greeks calculation
    - Gamma flip point detection
    """
    
    def __init__(
        self,
        name: str = "GammaExposure",
        gamma_threshold: float = 100000,  # Gamma exposure threshold
        pin_proximity_threshold: float = 0.02,  # 2% proximity to strike
        large_gamma_multiplier: float = 2.0,
        days_to_expiry_weight: float = 0.5,
        min_open_interest: int = 100,
        risk_free_rate: float = 0.05
    ):
        """
        Initialize Gamma Exposure agent.
        
        Args:
            name: Agent name
            gamma_threshold: Threshold for significant gamma exposure
            pin_proximity_threshold: Price proximity to strike for pinning
            large_gamma_multiplier: Multiplier for large gamma positions
            days_to_expiry_weight: Weight for time decay in gamma calculation
            min_open_interest: Minimum open interest to consider
            risk_free_rate: Risk-free rate for Black-Scholes calculations
        """
        self.name = name
        self.agent_type = "options"
        self.gamma_threshold = gamma_threshold
        self.pin_proximity_threshold = pin_proximity_threshold
        self.large_gamma_multiplier = large_gamma_multiplier
        self.days_to_expiry_weight = days_to_expiry_weight
        self.min_open_interest = min_open_interest
        self.risk_free_rate = risk_free_rate
        
        logger.info(f"Initialized {name} agent with gamma threshold: {gamma_threshold:,.0f}")
    
    def calculate_black_scholes_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: OptionType = OptionType.CALL
    ) -> Dict[str, float]:
        """
        Calculate all Black-Scholes Greeks for an option.
        
        Returns:
            Dictionary containing delta, gamma, theta, vega, rho
        """
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return {
                    'delta': 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            
            # Calculate d1 and d2
            d1 = (np.log(spot / strike) + (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
                volatility * np.sqrt(time_to_expiry)
            )
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
            
            # Delta
            if option_type == OptionType.CALL:
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Theta
            theta_common = -(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))
            if option_type == OptionType.CALL:
                theta = theta_common - self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                theta = theta_common + self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            
            # Vega (same for calls and puts)
            vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry)
            
            # Rho
            if option_type == OptionType.CALL:
                rho = strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                rho = -strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            
            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta / 365),  # Convert to daily theta
                'vega': float(vega / 100),     # Convert to per 1% vol move
                'rho': float(rho / 100)        # Convert to per 1% rate move
            }
            
        except Exception as e:
            logger.error(f"Greeks calculation failed: {str(e)}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
    
    def calculate_dealer_gamma_exposure(
        self,
        options_data: List[Dict[str, Any]],
        spot_price: float
    ) -> Dict[str, Any]:
        """
        Calculate net dealer gamma exposure across all strikes.
        
        Dealers are typically short options (opposite of customer flow).
        Negative gamma means dealers amplify moves (destabilizing).
        Positive gamma means dealers dampen moves (stabilizing).
        """
        try:
            total_call_gamma = 0.0
            total_put_gamma = 0.0
            total_call_delta = 0.0
            total_put_delta = 0.0
            gamma_by_strike = {}
            delta_by_strike = {}
            significant_levels = []
            
            for option in options_data:
                strike = option.get('strike', 0)
                option_type_str = option.get('type', '').lower()
                open_interest = option.get('open_interest', 0)
                volume = option.get('volume', 0)
                time_to_expiry = option.get('time_to_expiry', 0.0)
                implied_vol = option.get('implied_volatility', 0.2)
                
                if open_interest < self.min_open_interest or time_to_expiry <= 0:
                    continue
                
                # Determine option type
                option_type = OptionType.CALL if option_type_str == 'call' else OptionType.PUT
                
                # Calculate Greeks
                greeks = self.calculate_black_scholes_greeks(
                    spot_price, strike, time_to_expiry, implied_vol, option_type
                )
                
                # Dealer position multiplier (dealers are short options)
                dealer_position_multiplier = -1.0
                
                # Weight by open interest and time decay
                time_weight = 1.0 + self.days_to_expiry_weight * (1.0 / max(time_to_expiry, 0.01))
                
                # Position Greeks (multiply by 100 for contract multiplier)
                position_gamma = greeks['gamma'] * open_interest * 100 * dealer_position_multiplier * time_weight
                position_delta = greeks['delta'] * open_interest * 100 * dealer_position_multiplier
                
                if option_type == OptionType.CALL:
                    total_call_gamma += position_gamma
                    total_call_delta += position_delta
                else:
                    total_put_gamma += position_gamma
                    total_put_delta += position_delta
                
                # Track by strike
                if strike not in gamma_by_strike:
                    gamma_by_strike[strike] = 0.0
                    delta_by_strike[strike] = 0.0
                gamma_by_strike[strike] += position_gamma
                delta_by_strike[strike] += position_delta
                
                # Check for significant gamma levels
                if abs(position_gamma) > self.gamma_threshold:
                    level = GammaLevel(
                        strike=strike,
                        gamma_exposure=position_gamma,
                        option_type=option_type,
                        proximity=abs(spot_price - strike) / spot_price,
                        open_interest=open_interest
                    )
                    significant_levels.append(level)
            
            # Net exposures
            net_gamma = total_call_gamma + total_put_gamma
            net_delta = total_call_delta + total_put_delta
            
            # Find gamma flip point
            gamma_flip_level = self.find_gamma_flip_point(gamma_by_strike, spot_price)
            
            # Determine gamma regime
            if abs(net_gamma) < self.gamma_threshold / 10:
                regime = GammaRegime.NEUTRAL
            elif net_gamma < 0:
                regime = GammaRegime.SHORT_GAMMA
            else:
                regime = GammaRegime.LONG_GAMMA
            
            return {
                'net_gamma_exposure': net_gamma,
                'net_delta_exposure': net_delta,
                'call_gamma': total_call_gamma,
                'put_gamma': total_put_gamma,
                'call_delta': total_call_delta,
                'put_delta': total_put_delta,
                'gamma_by_strike': gamma_by_strike,
                'delta_by_strike': delta_by_strike,
                'significant_levels': [
                    {
                        'strike': level.strike,
                        'gamma_exposure': level.gamma_exposure,
                        'type': level.option_type.value,
                        'proximity': level.proximity,
                        'open_interest': level.open_interest
                    }
                    for level in significant_levels
                ],
                'gamma_flip_level': gamma_flip_level,
                'current_spot': spot_price,
                'regime': regime.value
            }
            
        except Exception as e:
            logger.error(f"Dealer gamma exposure calculation failed: {str(e)}")
            return {
                'net_gamma_exposure': 0.0,
                'regime': GammaRegime.NEUTRAL.value,
                'error': str(e)
            }
    
    def find_gamma_flip_point(
        self,
        gamma_by_strike: Dict[float, float],
        spot_price: float
    ) -> Optional[float]:
        """
        Find the price level where net gamma exposure flips sign.
        This is a critical level where market dynamics change.
        """
        try:
            if not gamma_by_strike:
                return None
            
            strikes = sorted(gamma_by_strike.keys())
            
            # Calculate cumulative gamma
            cumulative_gamma = 0.0
            flip_points = []
            
            prev_gamma = 0.0
            for strike in strikes:
                cumulative_gamma += gamma_by_strike[strike]
                
                # Check for sign change
                if prev_gamma * cumulative_gamma < 0:  # Sign changed
                    flip_points.append(strike)
                
                prev_gamma = cumulative_gamma
            
            # Find flip point closest to current spot
            if flip_points:
                closest_flip = min(flip_points, key=lambda x: abs(x - spot_price))
                return float(closest_flip)
            
            return None
            
        except Exception as e:
            logger.error(f"Gamma flip point calculation failed: {str(e)}")
            return None
    
    def analyze_gamma_pinning(
        self,
        gamma_data: Dict[str, Any],
        spot_price: float
    ) -> Dict[str, Any]:
        """
        Analyze risk of price pinning due to gamma exposure.
        
        Gamma pinning occurs when large gamma exposure at certain strikes
        creates a gravitational effect that pulls prices toward those levels.
        """
        try:
            significant_levels = gamma_data.get('significant_levels', [])
            
            if not significant_levels:
                return {
                    'pinning_detected': False,
                    'pinning_strength': PinningStrength.NONE.value
                }
            
            # Find strikes with high gamma near current price
            pinning_candidates = []
            
            for level in significant_levels:
                strike = level['strike']
                proximity = level['proximity']
                gamma_exposure = level['gamma_exposure']
                
                if proximity <= self.pin_proximity_threshold:
                    # Calculate pinning strength
                    pinning_strength = abs(gamma_exposure) / (proximity + 0.001)
                    
                    pinning_candidates.append({
                        'strike': strike,
                        'gamma_exposure': gamma_exposure,
                        'proximity': proximity,
                        'pinning_strength': pinning_strength,
                        'direction': 'attractive' if gamma_exposure < 0 else 'repulsive'
                    })
            
            if not pinning_candidates:
                return {
                    'pinning_detected': False,
                    'pinning_strength': PinningStrength.NONE.value
                }
            
            # Sort by pinning strength
            pinning_candidates.sort(key=lambda x: x['pinning_strength'], reverse=True)
            strongest_pin = pinning_candidates[0]
            
            # Classify pinning strength
            pin_strength_value = strongest_pin['pinning_strength']
            if pin_strength_value > 10000000:
                strength = PinningStrength.VERY_STRONG
            elif pin_strength_value > 5000000:
                strength = PinningStrength.STRONG
            elif pin_strength_value > 1000000:
                strength = PinningStrength.MODERATE
            else:
                strength = PinningStrength.WEAK
            
            return {
                'pinning_detected': True,
                'pin_level': strongest_pin['strike'],
                'pin_strength': pin_strength_value,
                'pinning_strength': strength.value,
                'pinning_candidates': pinning_candidates[:3],  # Top 3
                'distance_to_pin': abs(spot_price - strongest_pin['strike']) / spot_price,
                'pin_direction': strongest_pin['direction']
            }
            
        except Exception as e:
            logger.error(f"Gamma pinning analysis failed: {str(e)}")
            return {
                'pinning_detected': False,
                'pinning_strength': PinningStrength.NONE.value,
                'error': str(e)
            }
    
    def calculate_volatility_impact(
        self,
        gamma_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate how gamma exposure affects realized volatility.
        
        Short gamma (negative): Dealers hedge by buying high/selling low = amplifies moves
        Long gamma (positive): Dealers hedge by selling high/buying low = dampens moves
        """
        try:
            net_gamma = gamma_data.get('net_gamma_exposure', 0)
            gamma_flip_level = gamma_data.get('gamma_flip_level')
            current_spot = gamma_data.get('current_spot', 0)
            regime = gamma_data.get('regime', GammaRegime.NEUTRAL.value)
            
            # Calculate impact magnitude based on gamma size
            gamma_magnitude = abs(net_gamma)
            
            if regime == GammaRegime.SHORT_GAMMA.value:
                vol_impact = 'amplifying'
                # Short gamma amplifies moves - up to 2x volatility
                impact_magnitude = min(gamma_magnitude / 1000000, 2.0)
                hedging_flow = 'momentum_following'  # Buy rallies, sell dips
                
            elif regime == GammaRegime.LONG_GAMMA.value:
                vol_impact = 'dampening'
                # Long gamma dampens moves - up to 50% volatility reduction
                impact_magnitude = min(gamma_magnitude / 2000000, 0.5)
                hedging_flow = 'mean_reverting'  # Sell rallies, buy dips
                
            else:
                vol_impact = 'neutral'
                impact_magnitude = 0.0
                hedging_flow = 'balanced'
            
            # Distance from gamma flip affects stability
            stability = 'stable'
            if gamma_flip_level and current_spot:
                distance_from_flip = abs(current_spot - gamma_flip_level) / current_spot
                if distance_from_flip < 0.02:  # Within 2% of flip
                    stability = 'unstable'
                elif distance_from_flip < 0.05:  # Within 5% of flip
                    stability = 'transitioning'
            
            return {
                'volatility_impact': vol_impact,
                'impact_magnitude': impact_magnitude,
                'hedging_flow': hedging_flow,
                'regime': regime,
                'stability': stability,
                'gamma_flip_level': gamma_flip_level,
                'expected_volatility_multiplier': 1.0 + (impact_magnitude if vol_impact == 'amplifying' else -impact_magnitude * 0.5)
            }
            
        except Exception as e:
            logger.error(f"Volatility impact calculation failed: {str(e)}")
            return {
                'volatility_impact': 'neutral',
                'impact_magnitude': 0.0,
                'error': str(e)
            }
    
    async def analyze(
        self,
        options_data: List[Dict[str, Any]],
        spot_price: float,
        historical_volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive gamma exposure analysis.
        
        Args:
            options_data: List of options with strike, type, OI, volume, etc.
            spot_price: Current spot price
            historical_volatility: Historical volatility for comparison
        
        Returns:
            Complete gamma analysis with signals and recommendations
        """
        try:
            # Calculate dealer gamma exposure
            gamma_data = self.calculate_dealer_gamma_exposure(options_data, spot_price)
            
            # Analyze pinning risk
            pinning_analysis = self.analyze_gamma_pinning(gamma_data, spot_price)
            
            # Calculate volatility impact
            vol_impact = self.calculate_volatility_impact(gamma_data)
            
            # Generate trading signals
            signal = self.generate_signal(gamma_data, pinning_analysis, vol_impact)
            
            # Calculate key levels
            key_levels = self.identify_key_levels(gamma_data)
            
            # Risk metrics
            risk_metrics = self.calculate_risk_metrics(
                gamma_data,
                pinning_analysis,
                vol_impact,
                historical_volatility
            )
            
            return {
                'signal': signal,
                'gamma_exposure': gamma_data,
                'pinning_analysis': pinning_analysis,
                'volatility_impact': vol_impact,
                'key_levels': key_levels,
                'risk_metrics': risk_metrics,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Gamma analysis failed: {str(e)}")
            return {
                'signal': {
                    'action': 'hold',
                    'confidence': 0.0,
                    'reasoning': [f'Analysis error: {str(e)}']
                },
                'error': str(e)
            }
    
    def generate_signal(
        self,
        gamma_data: Dict[str, Any],
        pinning_analysis: Dict[str, Any],
        vol_impact: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signal based on gamma analysis."""
        try:
            action = "hold"
            confidence = 0.0
            reasoning = []
            recommendations = []
            
            regime = gamma_data.get('regime', GammaRegime.NEUTRAL.value)
            net_gamma = gamma_data.get('net_gamma_exposure', 0)
            
            # Regime-based signals
            if regime == GammaRegime.SHORT_GAMMA.value:
                action = "buy_volatility"
                confidence += 0.4
                reasoning.append(f"Dealers short gamma ({net_gamma:,.0f}) - expect volatility expansion")
                recommendations.append("Consider long straddles/strangles")
                recommendations.append("Use wider stops due to amplified moves")
                
            elif regime == GammaRegime.LONG_GAMMA.value:
                action = "sell_volatility"
                confidence += 0.4
                reasoning.append(f"Dealers long gamma ({net_gamma:,.0f}) - expect volatility compression")
                recommendations.append("Consider short straddles/strangles")
                recommendations.append("Tighter stops appropriate due to dampened moves")
            
            # Pinning signals
            if pinning_analysis.get('pinning_detected'):
                pin_strength = pinning_analysis.get('pinning_strength', PinningStrength.NONE.value)
                pin_level = pinning_analysis.get('pin_level')
                
                if pin_strength in [PinningStrength.STRONG.value, PinningStrength.VERY_STRONG.value]:
                    confidence += 0.3
                    reasoning.append(f"Strong gamma pinning at {pin_level:.2f}")
                    recommendations.append(f"Expect price gravitation toward {pin_level:.2f}")
                    
                    if action == "hold":
                        action = "range_trade"
            
            # Volatility impact
            if vol_impact.get('stability') == 'unstable':
                confidence *= 0.7  # Reduce confidence in unstable conditions
                reasoning.append("Near gamma flip - regime may change")
                recommendations.append("Monitor for regime transition")
            
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'recommendations': recommendations,
                'regime': regime
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': [str(e)]
            }
    
    def identify_key_levels(self, gamma_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify key price levels based on gamma exposure."""
        try:
            gamma_by_strike = gamma_data.get('gamma_by_strike', {})
            delta_by_strike = gamma_data.get('delta_by_strike', {})
            current_spot = gamma_data.get('current_spot', 0)
            
            if not gamma_by_strike:
                return {}
            
            # Sort strikes by absolute gamma exposure
            sorted_strikes = sorted(
                gamma_by_strike.keys(),
                key=lambda x: abs(gamma_by_strike[x]),
                reverse=True
            )
            
            # Key levels
            key_levels = {
                'max_gamma_strike': sorted_strikes[0] if sorted_strikes else None,
                'gamma_flip': gamma_data.get('gamma_flip_level'),
                'high_gamma_strikes': sorted_strikes[:5] if len(sorted_strikes) >= 5 else sorted_strikes,
                'current_spot': current_spot
            }
            
            # Add delta-neutral strike (where net delta is closest to zero)
            if delta_by_strike:
                delta_neutral = min(delta_by_strike.keys(), key=lambda x: abs(delta_by_strike[x]))
                key_levels['delta_neutral_strike'] = delta_neutral
            
            return key_levels
            
        except Exception as e:
            logger.error(f"Key levels identification failed: {str(e)}")
            return {}
    
    def calculate_risk_metrics(
        self,
        gamma_data: Dict[str, Any],
        pinning_analysis: Dict[str, Any],
        vol_impact: Dict[str, Any],
        historical_volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate risk metrics based on gamma analysis."""
        try:
            net_gamma = gamma_data.get('net_gamma_exposure', 0)
            net_delta = gamma_data.get('net_delta_exposure', 0)
            
            # Gamma risk score (0-100)
            gamma_risk = min(abs(net_gamma) / 10000000 * 100, 100)
            
            # Pinning risk score (0-100)
            if pinning_analysis.get('pinning_detected'):
                pin_strength = pinning_analysis.get('pin_strength', 0)
                pinning_risk = min(pin_strength / 10000000 * 100, 100)
            else:
                pinning_risk = 0
            
            # Volatility risk
            expected_vol_mult = vol_impact.get('expected_volatility_multiplier', 1.0)
            if historical_volatility:
                expected_volatility = historical_volatility * expected_vol_mult
            else:
                expected_volatility = 0.2 * expected_vol_mult  # Default 20% vol
            
            # Overall risk score
            overall_risk = (gamma_risk * 0.5 + pinning_risk * 0.3 + expected_volatility * 100 * 0.2)
            
            return {
                'gamma_risk_score': gamma_risk,
                'pinning_risk_score': pinning_risk,
                'expected_volatility': expected_volatility,
                'overall_risk_score': overall_risk,
                'delta_exposure': net_delta,
                'gamma_exposure': net_gamma,
                'risk_level': 'high' if overall_risk > 70 else 'moderate' if overall_risk > 30 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}")
            return {
                'overall_risk_score': 50,
                'risk_level': 'unknown'
            }


# Create global instance
gamma_exposure_agent = GammaExposureAgent()