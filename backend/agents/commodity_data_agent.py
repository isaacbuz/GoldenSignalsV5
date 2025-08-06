"""
Commodity Data Agent
Analyzes commodity markets and their impact on equities and sectors
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from agents.base import BaseAgent, AgentContext, AgentCapability, AgentConfig, Signal, SignalAction, SignalStrength
from services.alternative_data_service import alternative_data_service, DataSourceType, CommodityData

logger = logging.getLogger(__name__)


class CommodityDataAgent(BaseAgent):
    """Agent that analyzes commodity data for trading decisions"""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="CommodityData",
                confidence_threshold=0.65
            )
        super().__init__(config)
        
        # Commodity-equity correlations
        self.commodity_equity_map = {
            'oil': {
                'positive': ['XOM', 'CVX', 'COP', 'SLB', 'HAL', 'OXY', 'EOG', 'PXD'],
                'negative': ['AAL', 'DAL', 'UAL', 'LUV', 'FDX', 'UPS'],
                'etfs': ['XLE', 'OIH', 'XOP'],
                'sensitivity': 0.7
            },
            'gold': {
                'positive': ['GLD', 'NEM', 'GOLD', 'AEM', 'KGC', 'AU', 'HL'],
                'negative': ['TQQQ', 'SPXL'],  # Risk-on assets
                'etfs': ['GLD', 'GDX', 'GDXJ'],
                'sensitivity': 0.6
            },
            'silver': {
                'positive': ['SLV', 'HL', 'PAAS', 'CDE', 'FSM'],
                'negative': [],
                'etfs': ['SLV', 'SILJ'],
                'sensitivity': 0.5
            },
            'copper': {
                'positive': ['FCX', 'SCCO', 'TECK', 'RIO', 'BHP'],
                'negative': [],
                'etfs': ['COPX', 'CPER'],
                'sensitivity': 0.6,
                'economic_indicator': True  # Dr. Copper
            },
            'wheat': {
                'positive': ['ADM', 'BG', 'AGRO'],
                'negative': ['TSN', 'HRL'],  # Food processors (input costs)
                'etfs': ['WEAT', 'DBA'],
                'sensitivity': 0.4
            },
            'corn': {
                'positive': ['ADM', 'BG', 'MON'],
                'negative': ['TSN', 'PPC'],  # Livestock (feed costs)
                'etfs': ['CORN', 'DBA'],
                'sensitivity': 0.4
            },
            'natural_gas': {
                'positive': ['CHK', 'RRC', 'AR', 'COG', 'EQT'],
                'negative': ['CF', 'MOS'],  # Fertilizer (input cost)
                'etfs': ['UNG', 'FCG'],
                'sensitivity': 0.8
            }
        }
        
        # Sector impacts from commodities
        self.sector_impacts = {
            'energy': {
                'commodities': ['oil', 'natural_gas'],
                'correlation': 0.8
            },
            'materials': {
                'commodities': ['gold', 'silver', 'copper', 'iron_ore'],
                'correlation': 0.7
            },
            'industrials': {
                'commodities': ['copper', 'aluminum', 'steel'],
                'correlation': 0.5
            },
            'consumer_staples': {
                'commodities': ['wheat', 'corn', 'sugar', 'coffee'],
                'correlation': -0.4  # Input costs
            },
            'utilities': {
                'commodities': ['natural_gas', 'coal'],
                'correlation': -0.3  # Input costs
            },
            'technology': {
                'commodities': ['rare_earths', 'lithium', 'cobalt'],
                'correlation': -0.2  # Input costs
            }
        }
        
        # Commodity cycle indicators
        self.cycle_indicators = {
            'supercycle': {
                'duration_years': (10, 20),
                'drivers': ['emerging_market_growth', 'infrastructure', 'monetary_policy'],
                'current_phase': None
            },
            'seasonal': {
                'agricultural': {
                    'planting': [3, 4, 5],
                    'growing': [6, 7, 8],
                    'harvest': [9, 10, 11]
                },
                'energy': {
                    'driving_season': [5, 6, 7, 8],
                    'heating_season': [11, 12, 1, 2]
                }
            },
            'inventory_cycle': {
                'low_inventory': 'bullish',
                'high_inventory': 'bearish',
                'normal_range': (0.3, 0.7)  # Percentile
            }
        }
        
        # Technical patterns for commodities
        self.commodity_patterns = {
            'backwardation': 'bullish',  # Spot > Futures
            'contango': 'bearish',       # Futures > Spot
            'curve_flattening': 'trend_change',
            'curve_steepening': 'trend_acceleration'
        }
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze commodity data to generate trading signals"""
        
        try:
            symbol = market_data.get('symbol', 'SPY')
            
            # Fetch commodity data
            commodity_response = await alternative_data_service.get_comprehensive_alternative_data(
                symbols=[symbol],
                data_types=[DataSourceType.COMMODITY]
            )
            
            if not commodity_response or not commodity_response.get('commodity'):
                logger.warning("No commodity data available")
                return None
            
            commodity_data = commodity_response['commodity']
            
            # Analyze commodity trends
            commodity_analysis = self._analyze_commodity_trends(commodity_data)
            
            # Calculate correlations with symbol
            correlations = self._calculate_symbol_correlations(symbol, commodity_data)
            
            # Detect commodity cycles
            cycle_phase = self._detect_cycle_phase(commodity_data)
            
            # Analyze supply/demand dynamics
            supply_demand = self._analyze_supply_demand(commodity_data)
            
            # Check for regime changes
            regime_change = self._detect_regime_change(commodity_data)
            
            # Generate signal
            signal_data = self._generate_signal_from_commodities(
                commodity_analysis,
                correlations,
                cycle_phase,
                supply_demand,
                regime_change,
                symbol
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                commodity_analysis,
                correlations,
                len(commodity_data)
            )
            
            # Determine strength
            if confidence > 0.75 and abs(signal_data['impact_score']) > 0.6:
                strength = SignalStrength.STRONG
            elif confidence > 0.60:
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
                commodity_analysis,
                correlations,
                cycle_phase,
                signal_data
            )
            
            # Identify risks
            risks = self._identify_risks(
                commodity_analysis,
                regime_change,
                supply_demand
            )
            
            return Signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                strength=strength,
                source=self.config.name,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    'commodity_trends': commodity_analysis,
                    'symbol_correlations': correlations,
                    'cycle_phase': cycle_phase,
                    'supply_demand': supply_demand,
                    'regime_change': regime_change,
                    'key_commodities': self._format_key_commodities(commodity_data),
                    'sector_impacts': self._calculate_sector_impacts(commodity_analysis),
                    'risks': risks
                },
                market_conditions={
                    'commodities_tracked': len(commodity_data),
                    'avg_volatility': commodity_analysis.get('avg_volatility', 0),
                    'regime_stability': not regime_change.get('detected', False)
                }
            )
            
        except Exception as e:
            logger.error(f"Commodity data analysis error: {e}")
            return None
    
    def _analyze_commodity_trends(
        self,
        commodity_data: Dict[str, CommodityData]
    ) -> Dict[str, Any]:
        """Analyze trends across commodities"""
        
        if not commodity_data:
            return {
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_momentum': 0,
                'avg_volatility': 0,
                'trending_commodities': []
            }
        
        bullish_count = 0
        bearish_count = 0
        momentums = []
        trending = []
        
        for name, data in commodity_data.items():
            # Simple momentum (would be better with historical data)
            momentum = data.sentiment  # Using sentiment as proxy
            momentums.append(momentum)
            
            if momentum > 0.2:
                bullish_count += 1
                trending.append({
                    'commodity': name,
                    'direction': 'bullish',
                    'strength': momentum
                })
            elif momentum < -0.2:
                bearish_count += 1
                trending.append({
                    'commodity': name,
                    'direction': 'bearish',
                    'strength': abs(momentum)
                })
        
        # Calculate volatility (simplified)
        avg_volatility = 0.2  # Default assumption
        
        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'avg_momentum': np.mean(momentums) if momentums else 0,
            'avg_volatility': avg_volatility,
            'trending_commodities': sorted(trending, key=lambda x: x['strength'], reverse=True)[:5]
        }
    
    def _calculate_symbol_correlations(
        self,
        symbol: str,
        commodity_data: Dict[str, CommodityData]
    ) -> Dict[str, Any]:
        """Calculate correlations between symbol and commodities"""
        
        correlations = {}
        direct_impact = None
        
        # Check each commodity's relationship with the symbol
        for commodity_name, data in commodity_data.items():
            if commodity_name in self.commodity_equity_map:
                mapping = self.commodity_equity_map[commodity_name]
                
                # Check if symbol is directly affected
                if symbol in mapping['positive']:
                    correlations[commodity_name] = {
                        'correlation': mapping['sensitivity'],
                        'relationship': 'positive'
                    }
                    if not direct_impact or mapping['sensitivity'] > direct_impact['sensitivity']:
                        direct_impact = {
                            'commodity': commodity_name,
                            'sensitivity': mapping['sensitivity'],
                            'relationship': 'positive'
                        }
                elif symbol in mapping['negative']:
                    correlations[commodity_name] = {
                        'correlation': -mapping['sensitivity'],
                        'relationship': 'negative'
                    }
                    if not direct_impact or mapping['sensitivity'] > direct_impact['sensitivity']:
                        direct_impact = {
                            'commodity': commodity_name,
                            'sensitivity': mapping['sensitivity'],
                            'relationship': 'negative'
                        }
                elif symbol in mapping.get('etfs', []):
                    correlations[commodity_name] = {
                        'correlation': mapping['sensitivity'] * 1.2,  # ETFs more sensitive
                        'relationship': 'direct'
                    }
                    direct_impact = {
                        'commodity': commodity_name,
                        'sensitivity': mapping['sensitivity'] * 1.2,
                        'relationship': 'direct'
                    }
        
        return {
            'direct_correlations': correlations,
            'primary_impact': direct_impact,
            'is_commodity_sensitive': direct_impact is not None
        }
    
    def _detect_cycle_phase(
        self,
        commodity_data: Dict[str, CommodityData]
    ) -> Dict[str, Any]:
        """Detect current phase in commodity cycles"""
        
        current_month = datetime.now().month
        
        # Seasonal cycle detection
        seasonal_phase = None
        if current_month in self.cycle_indicators['seasonal']['agricultural']['planting']:
            seasonal_phase = 'planting_season'
        elif current_month in self.cycle_indicators['seasonal']['agricultural']['harvest']:
            seasonal_phase = 'harvest_season'
        elif current_month in self.cycle_indicators['seasonal']['energy']['driving_season']:
            seasonal_phase = 'driving_season'
        elif current_month in self.cycle_indicators['seasonal']['energy']['heating_season']:
            seasonal_phase = 'heating_season'
        
        # Supercycle detection (simplified)
        avg_price_level = np.mean([d.price for d in commodity_data.values()]) if commodity_data else 0
        
        # Determine supercycle phase based on price levels and trends
        if avg_price_level > 0:  # Placeholder logic
            supercycle_phase = 'expansion'
        else:
            supercycle_phase = 'contraction'
        
        return {
            'seasonal_phase': seasonal_phase,
            'supercycle_phase': supercycle_phase,
            'implications': self._get_cycle_implications(seasonal_phase, supercycle_phase)
        }
    
    def _analyze_supply_demand(
        self,
        commodity_data: Dict[str, CommodityData]
    ) -> Dict[str, Any]:
        """Analyze supply and demand dynamics"""
        
        # Check for oil inventory data
        oil_inventory = commodity_data.get('oil_inventory')
        
        supply_demand = {
            'oil_inventory_level': None,
            'inventory_signal': 'neutral',
            'supply_disruptions': [],
            'demand_indicators': []
        }
        
        if oil_inventory:
            # Simplified inventory analysis
            inventory_level = oil_inventory.price  # Using price field for inventory
            
            # Compare to historical average (simplified)
            historical_avg = 450000  # Thousand barrels (example)
            
            if inventory_level < historical_avg * 0.9:
                supply_demand['inventory_signal'] = 'bullish'
                supply_demand['oil_inventory_level'] = 'low'
            elif inventory_level > historical_avg * 1.1:
                supply_demand['inventory_signal'] = 'bearish'
                supply_demand['oil_inventory_level'] = 'high'
            else:
                supply_demand['oil_inventory_level'] = 'normal'
        
        # Check for supply disruptions (would need news data integration)
        # Placeholder for now
        
        # Demand indicators
        if 'copper' in commodity_data:
            # Copper as economic indicator
            copper_price = commodity_data['copper'].price
            if copper_price > 0:  # Placeholder
                supply_demand['demand_indicators'].append('Strong industrial demand (copper up)')
        
        return supply_demand
    
    def _detect_regime_change(
        self,
        commodity_data: Dict[str, CommodityData]
    ) -> Dict[str, Any]:
        """Detect regime changes in commodity markets"""
        
        regime_change = {
            'detected': False,
            'type': None,
            'commodities_affected': []
        }
        
        # Check for broad-based moves
        if commodity_data:
            prices = [d.price for d in commodity_data.values()]
            sentiments = [d.sentiment for d in commodity_data.values()]
            
            # All commodities moving same direction
            if all(s > 0 for s in sentiments):
                regime_change['detected'] = True
                regime_change['type'] = 'risk_on'
                regime_change['commodities_affected'] = list(commodity_data.keys())
            elif all(s < 0 for s in sentiments):
                regime_change['detected'] = True
                regime_change['type'] = 'risk_off'
                regime_change['commodities_affected'] = list(commodity_data.keys())
            
            # Check for inflation regime
            if 'gold' in commodity_data and 'oil' in commodity_data:
                if commodity_data['gold'].sentiment > 0.5 and commodity_data['oil'].sentiment > 0.5:
                    regime_change['detected'] = True
                    regime_change['type'] = 'inflationary'
                    regime_change['commodities_affected'] = ['gold', 'oil']
        
        return regime_change
    
    def _generate_signal_from_commodities(
        self,
        commodity_analysis: Dict,
        correlations: Dict,
        cycle_phase: Dict,
        supply_demand: Dict,
        regime_change: Dict,
        symbol: str
    ) -> Dict[str, Any]:
        """Generate trading signal from commodity analysis"""
        
        # Check if symbol is commodity-sensitive
        if not correlations['is_commodity_sensitive']:
            return {
                'action': 'hold',
                'impact_score': 0,
                'reason': 'Symbol not commodity-sensitive'
            }
        
        # Get primary commodity impact
        primary_impact = correlations['primary_impact']
        commodity_name = primary_impact['commodity']
        
        # Calculate impact score
        impact_score = 0
        
        # Base score from commodity trend
        for trend in commodity_analysis['trending_commodities']:
            if trend['commodity'] == commodity_name:
                if primary_impact['relationship'] == 'positive':
                    impact_score = trend['strength'] if trend['direction'] == 'bullish' else -trend['strength']
                else:  # negative relationship
                    impact_score = -trend['strength'] if trend['direction'] == 'bullish' else trend['strength']
                break
        
        # Adjust for cycle phase
        if cycle_phase['seasonal_phase']:
            if cycle_phase['seasonal_phase'] == 'driving_season' and commodity_name == 'oil':
                impact_score *= 1.2  # Amplify oil impact during driving season
            elif cycle_phase['seasonal_phase'] == 'heating_season' and commodity_name == 'natural_gas':
                impact_score *= 1.3
        
        # Adjust for supply/demand
        if supply_demand['inventory_signal'] != 'neutral':
            if commodity_name in ['oil', 'natural_gas']:
                if supply_demand['inventory_signal'] == 'bullish':
                    impact_score += 0.2
                else:
                    impact_score -= 0.2
        
        # Adjust for regime change
        if regime_change['detected']:
            if regime_change['type'] == 'inflationary':
                if symbol in ['GLD', 'SLV', 'XLE']:  # Inflation beneficiaries
                    impact_score += 0.3
                else:
                    impact_score -= 0.2
        
        # Determine action
        if impact_score > 0.5:
            action = 'strong_buy'
        elif impact_score > 0.2:
            action = 'buy'
        elif impact_score < -0.5:
            action = 'strong_sell'
        elif impact_score < -0.2:
            action = 'sell'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'impact_score': impact_score
        }
    
    def _calculate_confidence(
        self,
        commodity_analysis: Dict,
        correlations: Dict,
        commodity_count: int
    ) -> float:
        """Calculate confidence in commodity signal"""
        
        base_confidence = 0.5
        
        # Factor 1: Data coverage
        coverage_confidence = min(commodity_count / 7, 1.0) * 0.2
        
        # Factor 2: Direct correlation exists
        if correlations['is_commodity_sensitive']:
            correlation_confidence = 0.2
        else:
            correlation_confidence = 0.0
        
        # Factor 3: Trend clarity
        if commodity_analysis['trending_commodities']:
            trend_confidence = 0.15
        else:
            trend_confidence = 0.05
        
        # Factor 4: Market consensus (bullish vs bearish)
        total_trending = commodity_analysis['bullish_count'] + commodity_analysis['bearish_count']
        if total_trending > 0:
            consensus = max(commodity_analysis['bullish_count'], commodity_analysis['bearish_count']) / total_trending
            consensus_confidence = consensus * 0.15
        else:
            consensus_confidence = 0.0
        
        confidence = (
            base_confidence +
            coverage_confidence +
            correlation_confidence +
            trend_confidence +
            consensus_confidence
        )
        
        return min(confidence, 0.90)
    
    def _generate_reasoning(
        self,
        commodity_analysis: Dict,
        correlations: Dict,
        cycle_phase: Dict,
        signal_data: Dict
    ) -> List[str]:
        """Generate reasoning for the signal"""
        
        reasoning = []
        
        # Primary commodity impact
        if correlations['primary_impact']:
            impact = correlations['primary_impact']
            reasoning.append(
                f"{impact['commodity'].title()} {impact['relationship']} correlation "
                f"(sensitivity: {impact['sensitivity']:.2f})"
            )
        
        # Trending commodities
        if commodity_analysis['trending_commodities']:
            top_trend = commodity_analysis['trending_commodities'][0]
            reasoning.append(
                f"{top_trend['commodity'].title()} trending {top_trend['direction']}"
            )
        
        # Cycle phase
        if cycle_phase['seasonal_phase']:
            reasoning.append(f"Currently in {cycle_phase['seasonal_phase'].replace('_', ' ')}")
        
        # Overall momentum
        momentum = commodity_analysis['avg_momentum']
        if abs(momentum) > 0.2:
            direction = "bullish" if momentum > 0 else "bearish"
            reasoning.append(f"Commodity complex {direction} (momentum: {momentum:.2f})")
        
        return reasoning[:5]
    
    def _identify_risks(
        self,
        commodity_analysis: Dict,
        regime_change: Dict,
        supply_demand: Dict
    ) -> List[str]:
        """Identify commodity-related risks"""
        
        risks = []
        
        # Regime change risk
        if regime_change['detected']:
            risks.append(f"Commodity regime change: {regime_change['type']}")
        
        # Volatility risk
        if commodity_analysis['avg_volatility'] > 0.3:
            risks.append("High commodity volatility")
        
        # Inventory concerns
        if supply_demand['oil_inventory_level'] == 'low':
            risks.append("Low oil inventories may spike prices")
        elif supply_demand['oil_inventory_level'] == 'high':
            risks.append("High oil inventories may pressure prices")
        
        # Mixed signals
        if commodity_analysis['bullish_count'] > 0 and commodity_analysis['bearish_count'] > 0:
            risks.append("Mixed commodity signals")
        
        return risks[:4]
    
    def _get_cycle_implications(
        self,
        seasonal_phase: str,
        supercycle_phase: str
    ) -> List[str]:
        """Get implications of current cycle phases"""
        
        implications = []
        
        if seasonal_phase == 'planting_season':
            implications.append("Agricultural volatility expected")
        elif seasonal_phase == 'harvest_season':
            implications.append("Agricultural supply increasing")
        elif seasonal_phase == 'driving_season':
            implications.append("Oil demand typically strong")
        elif seasonal_phase == 'heating_season':
            implications.append("Natural gas demand elevated")
        
        if supercycle_phase == 'expansion':
            implications.append("Broad commodity strength likely")
        elif supercycle_phase == 'contraction':
            implications.append("Commodity weakness expected")
        
        return implications
    
    def _format_key_commodities(
        self,
        commodity_data: Dict[str, CommodityData]
    ) -> List[Dict]:
        """Format key commodities for display"""
        
        formatted = []
        
        for name, data in list(commodity_data.items())[:5]:
            formatted.append({
                'name': name,
                'price': round(data.price, 2),
                'exchange': data.exchange,
                'sentiment': round(data.sentiment, 2)
            })
        
        return formatted
    
    def _calculate_sector_impacts(
        self,
        commodity_analysis: Dict
    ) -> List[Dict]:
        """Calculate sector impacts from commodity movements"""
        
        sector_impacts = []
        
        for sector, config in self.sector_impacts.items():
            impact_score = 0
            
            # Calculate based on relevant commodity trends
            for trend in commodity_analysis['trending_commodities']:
                if trend['commodity'] in config['commodities']:
                    if trend['direction'] == 'bullish':
                        impact_score += trend['strength'] * config['correlation']
                    else:
                        impact_score -= trend['strength'] * config['correlation']
            
            if abs(impact_score) > 0.1:
                sector_impacts.append({
                    'sector': sector,
                    'impact': impact_score,
                    'direction': 'positive' if impact_score > 0 else 'negative'
                })
        
        return sorted(sector_impacts, key=lambda x: abs(x['impact']), reverse=True)
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for commodity analysis
        
        Returns:
            List of data type strings
        """
        return [
            'commodity',  # Primary requirement - commodity data
            'symbol',     # Stock symbol
            'price'       # Current price for context
        ]


# Create global instance
commodity_data_agent = CommodityDataAgent()