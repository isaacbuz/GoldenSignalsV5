"""
Economic Indicator Agent
Analyzes macroeconomic data to inform trading decisions
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from agents.base import BaseAgent, AgentContext, AgentCapability
from services.enhanced_data_aggregator import enhanced_data_aggregator, EconomicIndicator

logger = logging.getLogger(__name__)


class EconomicIndicatorAgent(BaseAgent):
    """Agent that analyzes economic indicators and their market impact"""
    
    def __init__(self):
        super().__init__(
            name="EconomicIndicator",
            capabilities=[
                AgentCapability.FUNDAMENTAL_ANALYSIS,
                AgentCapability.RISK_ASSESSMENT
            ],
            confidence_threshold=0.7
        )
        
        # Economic indicator impacts on different sectors
        self.indicator_impacts = {
            'DFF': {  # Federal Funds Rate
                'sectors': {
                    'financials': 0.8,    # Banks benefit from higher rates
                    'real_estate': -0.9,  # REITs hurt by higher rates
                    'utilities': -0.7,    # Dividend stocks less attractive
                    'technology': -0.6,   # Growth stocks hurt by higher rates
                    'consumer_discretionary': -0.5
                },
                'threshold': 0.25  # 25 basis points
            },
            'DGS10': {  # 10-Year Treasury Yield
                'sectors': {
                    'financials': 0.6,
                    'real_estate': -0.8,
                    'utilities': -0.7,
                    'technology': -0.5,
                    'materials': 0.3
                },
                'threshold': 0.1
            },
            'UNRATE': {  # Unemployment Rate
                'sectors': {
                    'consumer_discretionary': -0.8,
                    'consumer_staples': 0.3,  # Defensive
                    'financials': -0.5,
                    'real_estate': -0.4
                },
                'threshold': 0.2
            },
            'CPIAUCSL': {  # Consumer Price Index (Inflation)
                'sectors': {
                    'materials': 0.7,      # Commodity plays
                    'energy': 0.8,
                    'financials': 0.4,
                    'consumer_staples': 0.5,
                    'technology': -0.6,
                    'utilities': -0.4
                },
                'threshold': 0.3
            },
            'GDPC1': {  # Real GDP
                'sectors': {
                    'technology': 0.8,
                    'consumer_discretionary': 0.7,
                    'industrials': 0.6,
                    'financials': 0.5,
                    'utilities': 0.2
                },
                'threshold': 0.5
            },
            'DXY': {  # US Dollar Index
                'sectors': {
                    'materials': -0.7,     # Inverse correlation
                    'energy': -0.6,
                    'technology': 0.4,     # Large cap tech benefits
                    'industrials': -0.3,
                    'consumer_staples': 0.3
                },
                'threshold': 1.0
            },
            'DCOILWTICO': {  # Crude Oil
                'sectors': {
                    'energy': 0.9,
                    'materials': 0.4,
                    'industrials': -0.3,
                    'airlines': -0.8,
                    'consumer_discretionary': -0.4
                },
                'threshold': 5.0
            },
            'GOLDAMGBD228NLBM': {  # Gold Price
                'sectors': {
                    'materials': 0.8,      # Gold miners
                    'financials': -0.3,
                    'technology': -0.2,
                    'utilities': 0.3,      # Safe haven correlation
                    'real_estate': 0.2
                },
                'threshold': 20.0
            }
        }
        
        # Market regime thresholds
        self.regime_thresholds = {
            'recession': {
                'gdp_growth': -0.5,
                'unemployment_change': 1.0,
                'yield_curve': -0.1  # Inverted
            },
            'expansion': {
                'gdp_growth': 2.0,
                'unemployment_change': -0.5,
                'inflation': 2.0
            },
            'stagflation': {
                'gdp_growth': 0.5,
                'inflation': 4.0
            }
        }
        
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze economic indicators and their market impact"""
        
        try:
            # Get economic indicators
            indicators = await enhanced_data_aggregator.get_economic_indicators()
            
            if not indicators:
                logger.warning("No economic indicators available")
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'error': 'No economic data available'
                }
                
            # Analyze each indicator
            indicator_analysis = self._analyze_indicators(indicators)
            
            # Determine market regime
            regime = self._determine_market_regime(indicator_analysis)
            
            # Get sector recommendations
            sector_recommendations = self._get_sector_recommendations(
                indicator_analysis, 
                regime
            )
            
            # Generate overall market signal
            market_signal = self._generate_market_signal(
                indicator_analysis,
                regime,
                context.symbol
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(indicator_analysis, regime)
            
            return {
                'signal': market_signal['action'],
                'confidence': confidence,
                'market_regime': regime,
                'economic_outlook': market_signal['outlook'],
                'sector_recommendations': sector_recommendations,
                'key_indicators': self._get_key_indicators(indicator_analysis),
                'risks': self._identify_risks(indicator_analysis, regime),
                'metadata': {
                    'indicators_analyzed': len(indicators),
                    'data_quality': self._assess_data_quality(indicators)
                }
            }
            
        except Exception as e:
            logger.error(f"Economic analysis error: {e}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e)
            }
            
    def _analyze_indicators(
        self, 
        indicators: List[EconomicIndicator]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze individual economic indicators"""
        
        analysis = {}
        
        for indicator in indicators:
            # Calculate change
            if indicator.previous_value != 0:
                change = indicator.value - indicator.previous_value
                change_percent = (change / abs(indicator.previous_value)) * 100
            else:
                change = indicator.value
                change_percent = 0
                
            # Determine trend
            if abs(change_percent) < 1:
                trend = 'stable'
            elif change > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
                
            # Check if significant change
            threshold = self.indicator_impacts.get(
                indicator.indicator, {}
            ).get('threshold', 1.0)
            
            is_significant = abs(change_percent) > threshold
            
            # Analyze impact
            impact_analysis = self._analyze_indicator_impact(
                indicator.indicator,
                change_percent,
                is_significant
            )
            
            analysis[indicator.indicator] = {
                'value': indicator.value,
                'previous': indicator.previous_value,
                'change': change,
                'change_percent': change_percent,
                'trend': trend,
                'is_significant': is_significant,
                'impact': impact_analysis,
                'timestamp': indicator.timestamp
            }
            
        return analysis
        
    def _analyze_indicator_impact(
        self,
        indicator_name: str,
        change_percent: float,
        is_significant: bool
    ) -> Dict[str, Any]:
        """Analyze the market impact of an indicator change"""
        
        if not is_significant:
            return {
                'market_impact': 'neutral',
                'strength': 0.0,
                'affected_sectors': []
            }
            
        impacts = self.indicator_impacts.get(indicator_name, {})
        sector_impacts = impacts.get('sectors', {})
        
        # Calculate impact direction and strength
        if indicator_name in ['UNRATE']:  # Negative indicators
            impact_multiplier = -1
        else:
            impact_multiplier = 1
            
        impact_strength = min(abs(change_percent) / 10, 1.0) * impact_multiplier
        
        # Determine affected sectors
        affected_sectors = []
        for sector, sensitivity in sector_impacts.items():
            sector_impact = sensitivity * impact_strength
            if abs(sector_impact) > 0.3:
                affected_sectors.append({
                    'sector': sector,
                    'impact': sector_impact,
                    'recommendation': 'bullish' if sector_impact > 0 else 'bearish'
                })
                
        return {
            'market_impact': 'positive' if impact_strength > 0 else 'negative',
            'strength': abs(impact_strength),
            'affected_sectors': sorted(
                affected_sectors, 
                key=lambda x: abs(x['impact']), 
                reverse=True
            )
        }
        
    def _determine_market_regime(
        self, 
        indicator_analysis: Dict[str, Dict[str, Any]]
    ) -> str:
        """Determine current market regime based on indicators"""
        
        # Extract key metrics
        gdp_data = indicator_analysis.get('GDPC1', {})
        unemployment_data = indicator_analysis.get('UNRATE', {})
        inflation_data = indicator_analysis.get('CPIAUCSL', {})
        yield_data = indicator_analysis.get('DGS10', {})
        fed_funds_data = indicator_analysis.get('DFF', {})
        
        # Calculate yield curve (10Y - Fed Funds)
        if yield_data and fed_funds_data:
            yield_curve = yield_data.get('value', 0) - fed_funds_data.get('value', 0)
        else:
            yield_curve = 0
            
        # Check recession indicators
        recession_score = 0
        if gdp_data.get('change_percent', 0) < -0.5:
            recession_score += 1
        if unemployment_data.get('trend') == 'increasing' and unemployment_data.get('change_percent', 0) > 0.5:
            recession_score += 1
        if yield_curve < 0:  # Inverted yield curve
            recession_score += 1
            
        # Check expansion indicators
        expansion_score = 0
        if gdp_data.get('change_percent', 0) > 2.0:
            expansion_score += 1
        if unemployment_data.get('trend') == 'decreasing':
            expansion_score += 1
        if inflation_data.get('value', 0) > 1.5 and inflation_data.get('value', 0) < 3.0:
            expansion_score += 1
            
        # Check stagflation
        stagflation_score = 0
        if gdp_data.get('change_percent', 0) < 1.0 and inflation_data.get('value', 0) > 4.0:
            stagflation_score += 2
            
        # Determine regime
        if recession_score >= 2:
            return 'recession'
        elif stagflation_score >= 2:
            return 'stagflation'
        elif expansion_score >= 2:
            return 'expansion'
        elif yield_curve < 0:
            return 'late_cycle'
        else:
            return 'mid_cycle'
            
    def _get_sector_recommendations(
        self,
        indicator_analysis: Dict[str, Dict[str, Any]],
        regime: str
    ) -> List[Dict[str, Any]]:
        """Get sector recommendations based on economic conditions"""
        
        sector_scores = {}
        
        # Aggregate impacts from all indicators
        for indicator_name, analysis in indicator_analysis.items():
            if analysis['impact']['affected_sectors']:
                for sector_impact in analysis['impact']['affected_sectors']:
                    sector = sector_impact['sector']
                    impact = sector_impact['impact']
                    
                    if sector not in sector_scores:
                        sector_scores[sector] = []
                    sector_scores[sector].append(impact)
                    
        # Calculate average scores and add regime adjustments
        recommendations = []
        
        regime_adjustments = {
            'recession': {
                'consumer_staples': 0.3,
                'utilities': 0.3,
                'healthcare': 0.2,
                'technology': -0.2,
                'consumer_discretionary': -0.4
            },
            'expansion': {
                'technology': 0.3,
                'consumer_discretionary': 0.3,
                'financials': 0.2,
                'industrials': 0.2,
                'utilities': -0.2
            },
            'stagflation': {
                'energy': 0.4,
                'materials': 0.3,
                'consumer_staples': 0.2,
                'technology': -0.3,
                'consumer_discretionary': -0.3
            }
        }
        
        for sector, impacts in sector_scores.items():
            avg_impact = np.mean(impacts)
            
            # Apply regime adjustment
            regime_adj = regime_adjustments.get(regime, {}).get(sector, 0)
            final_score = avg_impact + regime_adj
            
            # Generate recommendation
            if final_score > 0.5:
                recommendation = 'STRONG_BUY'
            elif final_score > 0.2:
                recommendation = 'BUY'
            elif final_score < -0.5:
                recommendation = 'STRONG_SELL'
            elif final_score < -0.2:
                recommendation = 'SELL'
            else:
                recommendation = 'NEUTRAL'
                
            recommendations.append({
                'sector': sector,
                'score': final_score,
                'recommendation': recommendation,
                'rationale': self._get_sector_rationale(sector, regime, final_score)
            })
            
        return sorted(recommendations, key=lambda x: abs(x['score']), reverse=True)
        
    def _generate_market_signal(
        self,
        indicator_analysis: Dict[str, Dict[str, Any]],
        regime: str,
        symbol: str
    ) -> Dict[str, Any]:
        """Generate overall market signal"""
        
        # Calculate aggregate market score
        positive_factors = 0
        negative_factors = 0
        
        for indicator_name, analysis in indicator_analysis.items():
            if analysis['is_significant']:
                if analysis['impact']['market_impact'] == 'positive':
                    positive_factors += analysis['impact']['strength']
                else:
                    negative_factors += analysis['impact']['strength']
                    
        net_score = positive_factors - negative_factors
        
        # Adjust for regime
        regime_multipliers = {
            'recession': 0.7,      # More cautious
            'stagflation': 0.6,
            'late_cycle': 0.8,
            'mid_cycle': 1.0,
            'expansion': 1.2       # More aggressive
        }
        
        adjusted_score = net_score * regime_multipliers.get(regime, 1.0)
        
        # Generate signal
        if adjusted_score > 0.5:
            action = 'STRONG_BUY'
            outlook = 'Very positive economic conditions'
        elif adjusted_score > 0.2:
            action = 'BUY'
            outlook = 'Positive economic conditions'
        elif adjusted_score < -0.5:
            action = 'STRONG_SELL'
            outlook = 'Very negative economic conditions'
        elif adjusted_score < -0.2:
            action = 'SELL'
            outlook = 'Negative economic conditions'
        else:
            action = 'NEUTRAL'
            outlook = 'Mixed economic signals'
            
        return {
            'action': action,
            'outlook': outlook,
            'score': adjusted_score
        }
        
    def _calculate_confidence(
        self,
        indicator_analysis: Dict[str, Dict[str, Any]],
        regime: str
    ) -> float:
        """Calculate confidence in economic signal"""
        
        # Base confidence on data quality and consistency
        base_confidence = 0.5
        
        # Check data freshness
        fresh_data_count = sum(
            1 for analysis in indicator_analysis.values()
            if (datetime.utcnow() - analysis['timestamp']).days < 7
        )
        
        freshness_score = fresh_data_count / len(indicator_analysis)
        
        # Check signal consistency
        trends = [a['trend'] for a in indicator_analysis.values()]
        if all(t == trends[0] for t in trends):
            consistency_score = 1.0
        else:
            consistency_score = 0.5
            
        # Regime clarity
        regime_confidence = {
            'expansion': 0.9,
            'recession': 0.9,
            'mid_cycle': 0.7,
            'late_cycle': 0.6,
            'stagflation': 0.5
        }
        
        regime_score = regime_confidence.get(regime, 0.5)
        
        # Calculate final confidence
        confidence = (
            base_confidence * 0.2 +
            freshness_score * 0.3 +
            consistency_score * 0.3 +
            regime_score * 0.2
        )
        
        return min(confidence, 0.95)
        
    def _get_key_indicators(
        self,
        indicator_analysis: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract key indicators for display"""
        
        key_indicators = []
        
        # Priority indicators
        priority_order = ['DFF', 'UNRATE', 'CPIAUCSL', 'DGS10', 'GDPC1']
        
        for indicator_name in priority_order:
            if indicator_name in indicator_analysis:
                analysis = indicator_analysis[indicator_name]
                
                # Format display name
                display_names = {
                    'DFF': 'Fed Funds Rate',
                    'UNRATE': 'Unemployment',
                    'CPIAUCSL': 'Inflation (CPI)',
                    'DGS10': '10-Year Treasury',
                    'GDPC1': 'GDP Growth'
                }
                
                key_indicators.append({
                    'name': display_names.get(indicator_name, indicator_name),
                    'value': analysis['value'],
                    'change': analysis['change_percent'],
                    'trend': analysis['trend'],
                    'impact': analysis['impact']['market_impact']
                })
                
        return key_indicators
        
    def _identify_risks(
        self,
        indicator_analysis: Dict[str, Dict[str, Any]],
        regime: str
    ) -> List[str]:
        """Identify economic risks"""
        
        risks = []
        
        # Check specific risk conditions
        if 'DGS10' in indicator_analysis and 'DFF' in indicator_analysis:
            yield_curve = (
                indicator_analysis['DGS10']['value'] - 
                indicator_analysis['DFF']['value']
            )
            if yield_curve < 0:
                risks.append("Inverted yield curve signals recession risk")
                
        if 'CPIAUCSL' in indicator_analysis:
            inflation = indicator_analysis['CPIAUCSL']['value']
            if inflation > 4.0:
                risks.append("High inflation may trigger aggressive Fed action")
            elif inflation < 1.0:
                risks.append("Deflationary pressures present")
                
        if 'UNRATE' in indicator_analysis:
            if indicator_analysis['UNRATE']['trend'] == 'increasing':
                risks.append("Rising unemployment weakening consumer demand")
                
        # Regime-specific risks
        regime_risks = {
            'recession': "Recessionary conditions may persist",
            'stagflation': "Stagflation limiting policy options",
            'late_cycle': "Late cycle dynamics increase volatility"
        }
        
        if regime in regime_risks:
            risks.append(regime_risks[regime])
            
        return risks
        
    def _assess_data_quality(
        self,
        indicators: List[EconomicIndicator]
    ) -> str:
        """Assess quality of economic data"""
        
        if not indicators:
            return 'poor'
            
        # Check data freshness
        latest_data = max(ind.timestamp for ind in indicators)
        days_old = (datetime.utcnow() - latest_data).days
        
        if days_old < 7:
            return 'excellent'
        elif days_old < 14:
            return 'good'
        elif days_old < 30:
            return 'fair'
        else:
            return 'poor'
            
    def _get_sector_rationale(
        self,
        sector: str,
        regime: str,
        score: float
    ) -> str:
        """Generate rationale for sector recommendation"""
        
        regime_rationales = {
            'recession': {
                'consumer_staples': "Defensive sector performs well in downturns",
                'utilities': "Stable dividends attractive in uncertain times",
                'technology': "Growth stocks vulnerable to economic weakness"
            },
            'expansion': {
                'technology': "Growth acceleration benefits tech leaders",
                'consumer_discretionary': "Strong consumer spending environment",
                'financials': "Rising rates boost bank margins"
            },
            'stagflation': {
                'energy': "Energy benefits from inflation pressures",
                'materials': "Commodities hedge against inflation",
                'technology': "Valuations pressured by rising rates"
            }
        }
        
        base_rationale = regime_rationales.get(regime, {}).get(sector, "")
        
        if score > 0:
            return f"{base_rationale} Economic conditions favor this sector."
        else:
            return f"{base_rationale} Economic headwinds for this sector."


# Create global instance
economic_indicator_agent = EconomicIndicatorAgent()