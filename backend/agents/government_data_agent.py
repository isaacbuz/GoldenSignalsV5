"""
Government Data Agent
Analyzes official government economic data to generate trading signals
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from agents.base import BaseAgent, AgentContext, AgentCapability, AgentConfig, Signal, SignalAction, SignalStrength
from services.government_data_service import government_data_service, GovernmentDataSource

logger = logging.getLogger(__name__)


class GovernmentDataAgent(BaseAgent):
    """Agent that analyzes government economic data for trading signals"""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="GovernmentData",
                confidence_threshold=0.75
            )
        super().__init__(config)
        
        # Economic indicator thresholds and market impacts
        self.indicator_thresholds = {
            'fed_funds_rate': {
                'critical_high': 5.5,
                'high': 4.0,
                'neutral': 2.5,
                'low': 1.0,
                'impact': {
                    'equity': -0.7,  # Higher rates negative for equities
                    'bonds': -0.9,   # Negative for bond prices
                    'dollar': 0.8,   # Positive for dollar
                    'gold': -0.5     # Negative for gold
                }
            },
            'unemployment_rate': {
                'critical_high': 6.0,
                'high': 5.0,
                'neutral': 4.0,
                'low': 3.5,
                'impact': {
                    'equity': -0.8,  # High unemployment bad for stocks
                    'consumer': -0.9,
                    'financials': -0.6
                }
            },
            'cpi': {
                'critical_high': 5.0,
                'high': 3.5,
                'target': 2.0,
                'low': 1.0,
                'impact': {
                    'bonds': -0.8,
                    'commodities': 0.7,
                    'tech': -0.6,
                    'energy': 0.8
                }
            },
            'gdp_growth': {
                'strong': 3.0,
                'moderate': 2.0,
                'weak': 1.0,
                'recession': 0.0,
                'impact': {
                    'equity': 0.9,
                    'cyclicals': 0.8,
                    'defensives': -0.3
                }
            },
            'yield_curve_10_2': {
                'steep': 2.0,
                'normal': 1.0,
                'flat': 0.25,
                'inverted': 0.0,
                'impact': {
                    'equity': -0.9,  # Inverted = recession signal
                    'financials': -0.8,
                    'utilities': 0.5
                }
            },
            'vix': {
                'extreme_fear': 35,
                'high_volatility': 25,
                'normal': 20,
                'low_volatility': 15,
                'complacency': 12,
                'impact': {
                    'equity': -0.7,
                    'options': 0.9,
                    'bonds': 0.4
                }
            }
        }
        
        # Sector sensitivities to economic data
        self.sector_sensitivities = {
            'technology': {
                'interest_rates': -0.8,  # High sensitivity to rates
                'gdp_growth': 0.7,
                'inflation': -0.5
            },
            'financials': {
                'interest_rates': 0.6,   # Banks benefit from higher rates
                'yield_curve': 0.9,      # Steep curve good for banks
                'unemployment': -0.6
            },
            'energy': {
                'inflation': 0.8,
                'gdp_growth': 0.6,
                'dollar_index': -0.7     # Inverse correlation
            },
            'utilities': {
                'interest_rates': -0.7,  # Rate sensitive
                'inflation': -0.4,
                'recession_risk': 0.6    # Defensive
            },
            'consumer_discretionary': {
                'unemployment': -0.9,
                'gdp_growth': 0.8,
                'consumer_sentiment': 0.9
            },
            'real_estate': {
                'interest_rates': -0.9,  # Very rate sensitive
                'inflation': 0.5,        # Inflation hedge
                'gdp_growth': 0.6
            },
            'materials': {
                'inflation': 0.7,
                'gdp_growth': 0.7,
                'dollar_index': -0.6
            },
            'healthcare': {
                'recession_risk': 0.4,   # Defensive
                'demographics': 0.6,
                'policy_changes': -0.5
            }
        }
        
        # Market regime patterns
        self.regime_patterns = {
            'goldilocks': {  # Low inflation, moderate growth
                'indicators': {
                    'gdp_growth': (2.0, 3.5),
                    'cpi': (1.5, 2.5),
                    'unemployment': (3.5, 4.5)
                },
                'favored_sectors': ['technology', 'consumer_discretionary', 'financials'],
                'market_bias': 'bullish'
            },
            'stagflation': {  # High inflation, low growth
                'indicators': {
                    'gdp_growth': (0.0, 1.5),
                    'cpi': (4.0, float('inf')),
                    'unemployment': (4.5, float('inf'))
                },
                'favored_sectors': ['energy', 'materials', 'utilities'],
                'market_bias': 'bearish'
            },
            'recession': {
                'indicators': {
                    'gdp_growth': (float('-inf'), 0.5),
                    'unemployment': (5.0, float('inf')),
                    'yield_curve_10_2': (float('-inf'), 0.0)
                },
                'favored_sectors': ['utilities', 'healthcare', 'consumer_staples'],
                'market_bias': 'very_bearish'
            },
            'overheating': {  # High growth, high inflation
                'indicators': {
                    'gdp_growth': (3.5, float('inf')),
                    'cpi': (3.5, float('inf')),
                    'unemployment': (0.0, 3.5)
                },
                'favored_sectors': ['energy', 'materials', 'industrials'],
                'market_bias': 'volatile'
            }
        }
        
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze government economic data to generate trading signals"""
        
        try:
            # Fetch comprehensive government data
            gov_data = await government_data_service.get_comprehensive_economic_data()
            
            if not gov_data:
                logger.warning("No government data available")
                return None
            
            symbol = market_data.get('symbol', 'SPY')  # Default to SPY for market signals
            
            # Extract key data points
            fred_indicators = gov_data.get('fred_indicators', {})
            treasury_yields = gov_data.get('treasury_yields', {})
            labor_stats = gov_data.get('labor_statistics', {})
            analysis = gov_data.get('analysis', {})
            
            # Determine market regime
            regime = self._identify_market_regime(fred_indicators, analysis)
            
            # Calculate indicator scores
            indicator_scores = self._calculate_indicator_scores(
                fred_indicators, 
                treasury_yields, 
                labor_stats
            )
            
            # Determine sector impacts
            sector_impacts = self._calculate_sector_impacts(
                indicator_scores, 
                regime
            )
            
            # Generate trading signal
            signal_data = self._generate_signal(
                indicator_scores,
                sector_impacts,
                regime,
                symbol,
                analysis
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                indicator_scores,
                regime,
                gov_data
            )
            
            # Determine signal strength
            if confidence > 0.85:
                strength = SignalStrength.STRONG
            elif confidence > 0.70:
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
                regime,
                indicator_scores,
                analysis,
                signal_data
            )
            
            # Identify risks
            risks = self._identify_risks(fred_indicators, analysis, regime)
            
            return Signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                strength=strength,
                source=self.config.name,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    'market_regime': regime,
                    'economic_indicators': self._format_indicators(fred_indicators),
                    'sector_recommendations': sector_impacts[:5],  # Top 5 sectors
                    'market_implications': gov_data.get('market_implications', {}),
                    'yield_curve_analysis': analysis.get('yield_curve', {}),
                    'inflation_outlook': analysis.get('inflation_outlook', {}),
                    'risks': risks
                },
                market_conditions={
                    'data_sources': ['FRED', 'Treasury', 'BLS'],
                    'indicators_analyzed': len(indicator_scores),
                    'regime_confidence': signal_data.get('regime_confidence', 0.5)
                }
            )
            
        except Exception as e:
            logger.error(f"Government data analysis error: {e}")
            return None
    
    def _identify_market_regime(
        self, 
        indicators: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Identify the current market regime based on government data"""
        
        # Get key indicator values
        gdp_growth = indicators.get('gdp_growth_rate', {}).get('value')
        cpi = indicators.get('cpi', {}).get('value')
        unemployment = indicators.get('unemployment_rate', {}).get('value')
        yield_curve = analysis.get('yield_curve', {}).get('spread_10_2')
        
        # Check each regime pattern
        regime_scores = {}
        
        for regime_name, pattern in self.regime_patterns.items():
            score = 0
            count = 0
            
            for indicator, (min_val, max_val) in pattern['indicators'].items():
                value = None
                
                if indicator == 'gdp_growth':
                    value = gdp_growth
                elif indicator == 'cpi':
                    value = cpi
                elif indicator == 'unemployment':
                    value = unemployment
                elif indicator == 'yield_curve_10_2':
                    value = yield_curve
                
                if value is not None:
                    count += 1
                    if min_val <= value <= max_val:
                        score += 1
            
            if count > 0:
                regime_scores[regime_name] = score / count
        
        # Return regime with highest score
        if regime_scores:
            best_regime = max(regime_scores, key=regime_scores.get)
            if regime_scores[best_regime] > 0.5:
                return best_regime
        
        # Use analysis regime if available
        economic_regime = analysis.get('economic_regime')
        if economic_regime and economic_regime != 'uncertain':
            return economic_regime
        
        return 'mixed'
    
    def _calculate_indicator_scores(
        self,
        fred_indicators: Dict,
        treasury_yields: Dict,
        labor_stats: Dict
    ) -> Dict[str, float]:
        """Calculate normalized scores for each indicator"""
        
        scores = {}
        
        # Process FRED indicators
        for indicator_name, data in fred_indicators.items():
            if indicator_name in self.indicator_thresholds:
                value = data.get('value')
                if value is not None:
                    thresholds = self.indicator_thresholds[indicator_name]
                    
                    # Calculate normalized score (-1 to 1)
                    if 'critical_high' in thresholds:
                        if value >= thresholds['critical_high']:
                            score = -1.0
                        elif value >= thresholds.get('high', thresholds['critical_high']):
                            score = -0.5
                        elif value >= thresholds.get('neutral', thresholds.get('high')):
                            score = 0.0
                        elif value >= thresholds.get('low', 0):
                            score = 0.5
                        else:
                            score = 1.0
                    else:
                        # For indicators like GDP where higher is better
                        if 'strong' in thresholds:
                            if value >= thresholds['strong']:
                                score = 1.0
                            elif value >= thresholds['moderate']:
                                score = 0.5
                            elif value >= thresholds['weak']:
                                score = 0.0
                            elif value >= thresholds.get('recession', float('-inf')):
                                score = -0.5
                            else:
                                score = -1.0
                        else:
                            score = 0.0
                    
                    scores[indicator_name] = score
        
        return scores
    
    def _calculate_sector_impacts(
        self,
        indicator_scores: Dict[str, float],
        regime: str
    ) -> List[Dict[str, Any]]:
        """Calculate impact scores for different sectors"""
        
        sector_scores = {}
        
        # Calculate base scores from indicators
        for sector, sensitivities in self.sector_sensitivities.items():
            score = 0
            weight_sum = 0
            
            # Map indicators to sensitivity factors
            indicator_mapping = {
                'fed_funds_rate': 'interest_rates',
                'gdp_growth_rate': 'gdp_growth',
                'cpi': 'inflation',
                'unemployment_rate': 'unemployment',
                'yield_curve_10_2': 'yield_curve',
                'dollar_index': 'dollar_index'
            }
            
            for indicator, indicator_score in indicator_scores.items():
                sensitivity_key = indicator_mapping.get(indicator)
                if sensitivity_key in sensitivities:
                    sensitivity = sensitivities[sensitivity_key]
                    
                    # Adjust score based on sensitivity and indicator score
                    impact = indicator_score * sensitivity
                    score += impact
                    weight_sum += abs(sensitivity)
            
            if weight_sum > 0:
                sector_scores[sector] = score / weight_sum
        
        # Apply regime adjustments
        if regime in self.regime_patterns:
            favored_sectors = self.regime_patterns[regime]['favored_sectors']
            for sector in favored_sectors:
                if sector in sector_scores:
                    sector_scores[sector] += 0.3  # Boost favored sectors
        
        # Convert to sorted list
        sector_impacts = []
        for sector, score in sector_scores.items():
            if score > 0.3:
                recommendation = 'strong_buy'
            elif score > 0.1:
                recommendation = 'buy'
            elif score < -0.3:
                recommendation = 'strong_sell'
            elif score < -0.1:
                recommendation = 'sell'
            else:
                recommendation = 'neutral'
            
            sector_impacts.append({
                'sector': sector,
                'score': score,
                'recommendation': recommendation
            })
        
        return sorted(sector_impacts, key=lambda x: abs(x['score']), reverse=True)
    
    def _generate_signal(
        self,
        indicator_scores: Dict[str, float],
        sector_impacts: List[Dict],
        regime: str,
        symbol: str,
        analysis: Dict
    ) -> Dict[str, Any]:
        """Generate trading signal based on analysis"""
        
        # Calculate overall market score
        market_score = np.mean(list(indicator_scores.values())) if indicator_scores else 0
        
        # Adjust for regime
        regime_bias = self.regime_patterns.get(regime, {}).get('market_bias', 'neutral')
        regime_multiplier = {
            'very_bearish': 0.5,
            'bearish': 0.7,
            'volatile': 0.8,
            'neutral': 1.0,
            'bullish': 1.2
        }.get(regime_bias, 1.0)
        
        adjusted_score = market_score * regime_multiplier
        
        # Consider market implications
        implications = analysis.get('market_implications', {})
        equity_outlook = implications.get('equity_outlook', 'neutral')
        
        outlook_adjustment = {
            'very_bullish': 0.3,
            'bullish': 0.15,
            'neutral': 0,
            'bearish': -0.15,
            'very_bearish': -0.3
        }.get(equity_outlook, 0)
        
        final_score = adjusted_score + outlook_adjustment
        
        # Determine action
        if final_score > 0.4:
            action = 'strong_buy'
        elif final_score > 0.15:
            action = 'buy'
        elif final_score < -0.4:
            action = 'strong_sell'
        elif final_score < -0.15:
            action = 'sell'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'score': final_score,
            'regime_confidence': abs(market_score)
        }
    
    def _calculate_confidence(
        self,
        indicator_scores: Dict[str, float],
        regime: str,
        gov_data: Dict
    ) -> float:
        """Calculate confidence in the signal"""
        
        base_confidence = 0.6
        
        # Factor 1: Data completeness
        expected_indicators = 7
        actual_indicators = len(indicator_scores)
        completeness_score = min(actual_indicators / expected_indicators, 1.0)
        
        # Factor 2: Indicator agreement
        if indicator_scores:
            scores = list(indicator_scores.values())
            if all(s > 0 for s in scores) or all(s < 0 for s in scores):
                agreement_score = 1.0
            else:
                # Calculate variance in signals
                variance = np.var(scores)
                agreement_score = max(0, 1 - variance)
        else:
            agreement_score = 0.5
        
        # Factor 3: Regime clarity
        regime_clarity = {
            'goldilocks': 0.9,
            'recession': 0.85,
            'stagflation': 0.8,
            'overheating': 0.75,
            'mixed': 0.5
        }.get(regime, 0.6)
        
        # Calculate final confidence
        confidence = (
            base_confidence * 0.2 +
            completeness_score * 0.3 +
            agreement_score * 0.3 +
            regime_clarity * 0.2
        )
        
        return min(confidence, 0.95)
    
    def _generate_reasoning(
        self,
        regime: str,
        indicator_scores: Dict,
        analysis: Dict,
        signal_data: Dict
    ) -> List[str]:
        """Generate reasoning for the signal"""
        
        reasoning = [f"Market regime: {regime}"]
        
        # Add key indicator insights
        for indicator, score in indicator_scores.items():
            if abs(score) > 0.5:
                direction = "bullish" if score > 0 else "bearish"
                indicator_name = indicator.replace('_', ' ').title()
                reasoning.append(f"{indicator_name} is {direction} (score: {score:.2f})")
        
        # Add yield curve analysis
        yield_curve = analysis.get('yield_curve', {})
        if yield_curve.get('shape'):
            reasoning.append(f"Yield curve is {yield_curve['shape']}: {yield_curve.get('implication', 'N/A')}")
        
        # Add inflation outlook
        inflation = analysis.get('inflation_outlook', {})
        if inflation.get('outlook'):
            reasoning.append(f"Inflation outlook: {inflation['outlook']}")
        
        # Add signal strength
        if signal_data['score'] > 0.3:
            reasoning.append("Strong bullish signals from government data")
        elif signal_data['score'] < -0.3:
            reasoning.append("Strong bearish signals from government data")
        
        return reasoning[:5]  # Limit to 5 key points
    
    def _identify_risks(
        self,
        indicators: Dict,
        analysis: Dict,
        regime: str
    ) -> List[str]:
        """Identify key risks from government data"""
        
        risks = []
        
        # Check yield curve inversion
        yield_curve = analysis.get('yield_curve', {})
        if yield_curve.get('shape') == 'inverted':
            risks.append("Inverted yield curve signals recession risk")
        
        # Check inflation levels
        cpi = indicators.get('cpi', {}).get('value')
        if cpi and cpi > 4:
            risks.append(f"High inflation at {cpi:.1f}% may trigger aggressive Fed action")
        
        # Check unemployment trend
        unemployment = indicators.get('unemployment_rate', {})
        if unemployment.get('change_percent', 0) > 10:
            risks.append("Rising unemployment threatening economic growth")
        
        # Check VIX levels
        vix = indicators.get('vix', {}).get('value')
        if vix and vix > 30:
            risks.append(f"High market volatility (VIX: {vix:.1f})")
        
        # Regime-specific risks
        regime_risks = {
            'recession': "Recessionary conditions may worsen",
            'stagflation': "Stagflation limiting policy options",
            'overheating': "Economy overheating, correction likely"
        }
        
        if regime in regime_risks:
            risks.append(regime_risks[regime])
        
        return risks[:4]  # Return top 4 risks
    
    def _format_indicators(self, indicators: Dict) -> List[Dict]:
        """Format indicators for display"""
        
        formatted = []
        
        priority_indicators = [
            'fed_funds_rate', 'unemployment_rate', 'cpi', 
            '10_year_treasury', 'gdp_growth_rate', 'vix'
        ]
        
        for name in priority_indicators:
            if name in indicators:
                data = indicators[name]
                formatted.append({
                    'name': name.replace('_', ' ').title(),
                    'value': data.get('value'),
                    'change': data.get('change_percent'),
                    'date': data.get('date')
                })
        
        return formatted
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for government data analysis
        
        Returns:
            List of data type strings
        """
        return [
            'government_data',  # Primary requirement - government economic data
            'symbol',  # Basic symbol requirement
            'price'  # Current price for context
        ]


# Create global instance
government_data_agent = GovernmentDataAgent()