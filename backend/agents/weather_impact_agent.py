"""
Weather Impact Agent
Analyzes weather data to predict commodity and sector impacts
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from agents.base import BaseAgent, AgentContext, AgentCapability, AgentConfig, Signal, SignalAction, SignalStrength
from services.alternative_data_service import alternative_data_service, DataSourceType, WeatherData

logger = logging.getLogger(__name__)


class WeatherImpactAgent(BaseAgent):
    """Agent that analyzes weather patterns for commodity and sector trading"""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="WeatherImpact",
                confidence_threshold=0.70
            )
        super().__init__(config)
        
        # Weather impact on commodities
        self.commodity_impacts = {
            'extreme_heat': {
                'wheat': -0.8,      # Drought impact
                'corn': -0.7,
                'soybeans': -0.6,
                'natural_gas': 0.9,  # Cooling demand
                'electricity': 0.8
            },
            'extreme_cold': {
                'natural_gas': 0.9,  # Heating demand
                'heating_oil': 0.8,
                'electricity': 0.7,
                'orange_juice': -0.8,  # Frost damage
                'coffee': -0.6
            },
            'drought': {
                'wheat': -0.9,
                'corn': -0.9,
                'soybeans': -0.8,
                'cattle': -0.6,      # Feed costs
                'water_utilities': 0.5
            },
            'flooding': {
                'wheat': -0.7,
                'corn': -0.7,
                'rice': -0.8,
                'insurance': -0.5,
                'construction': 0.3   # Rebuilding
            },
            'hurricanes': {
                'oil': 0.8,          # Supply disruption
                'natural_gas': 0.7,
                'insurance': -0.9,
                'home_builders': -0.6,
                'utilities': -0.7
            }
        }
        
        # Sector impacts from weather
        self.sector_impacts = {
            'agriculture': {
                'temperature_optimal': (15, 25),  # Celsius
                'precipitation_optimal': (50, 150),  # mm/month
                'companies': ['DE', 'MON', 'ADM', 'BG']
            },
            'energy': {
                'temperature_extremes': True,  # Both hot and cold impact
                'wind_impact': True,
                'companies': ['XOM', 'CVX', 'COP', 'SLB', 'HAL']
            },
            'utilities': {
                'temperature_extremes': True,
                'storm_impact': True,
                'companies': ['NEE', 'DUK', 'SO', 'D', 'AEP']
            },
            'insurance': {
                'catastrophe_sensitive': True,
                'companies': ['AIG', 'TRV', 'ALL', 'PRU', 'MET']
            },
            'retail': {
                'seasonal_sensitivity': True,
                'companies': ['WMT', 'TGT', 'COST', 'HD', 'LOW']
            },
            'transportation': {
                'weather_disruption': True,
                'companies': ['UPS', 'FDX', 'DAL', 'UAL', 'LUV']
            }
        }
        
        # Regional weights for global markets
        self.regional_weights = {
            'Chicago': 0.3,      # Agricultural hub
            'Houston': 0.25,     # Energy hub
            'London': 0.15,      # Financial/commodity hub
            'Singapore': 0.15,   # Asian trade hub
            'Sydney': 0.15       # Mining/resources hub
        }
        
        # Historical patterns
        self.seasonal_patterns = {
            'winter': {
                'months': [12, 1, 2],
                'typical_impacts': ['heating_demand', 'snow_disruption']
            },
            'spring': {
                'months': [3, 4, 5],
                'typical_impacts': ['planting_season', 'flood_risk']
            },
            'summer': {
                'months': [6, 7, 8],
                'typical_impacts': ['drought_risk', 'cooling_demand', 'hurricane_season']
            },
            'fall': {
                'months': [9, 10, 11],
                'typical_impacts': ['harvest_season', 'hurricane_season']
            }
        }
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze weather data to generate trading signals"""
        
        try:
            symbol = market_data.get('symbol', 'SPY')
            
            # Fetch weather data
            weather_response = await alternative_data_service.get_comprehensive_alternative_data(
                symbols=[symbol],
                data_types=[DataSourceType.WEATHER]
            )
            
            if not weather_response or not weather_response.get('weather'):
                logger.warning("No weather data available")
                return None
            
            weather_data = weather_response['weather']
            
            # Analyze weather conditions
            weather_analysis = self._analyze_weather_conditions(weather_data)
            
            # Determine affected commodities
            commodity_impacts = self._calculate_commodity_impacts(weather_analysis)
            
            # Determine affected sectors
            sector_impacts = self._calculate_sector_impacts(weather_analysis, symbol)
            
            # Check for extreme events
            extreme_events = self._detect_extreme_events(weather_data)
            
            # Calculate seasonal factors
            seasonal_factors = self._calculate_seasonal_factors()
            
            # Generate signal
            signal_data = self._generate_signal_from_weather(
                weather_analysis,
                commodity_impacts,
                sector_impacts,
                extreme_events,
                seasonal_factors,
                symbol
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                weather_analysis,
                extreme_events,
                len(weather_data)
            )
            
            # Determine strength
            if confidence > 0.75 and extreme_events:
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
                weather_analysis,
                commodity_impacts,
                extreme_events,
                signal_data
            )
            
            # Identify risks
            risks = self._identify_risks(weather_analysis, extreme_events)
            
            return Signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                strength=strength,
                source=self.config.name,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    'weather_conditions': weather_analysis,
                    'commodity_impacts': commodity_impacts[:5],
                    'sector_impacts': sector_impacts[:5],
                    'extreme_events': extreme_events,
                    'seasonal_factors': seasonal_factors,
                    'regional_data': self._format_regional_data(weather_data),
                    'risks': risks
                },
                market_conditions={
                    'locations_monitored': len(weather_data),
                    'avg_impact_score': weather_analysis['avg_impact'],
                    'extreme_event_detected': len(extreme_events) > 0
                }
            )
            
        except Exception as e:
            logger.error(f"Weather impact analysis error: {e}")
            return None
    
    def _analyze_weather_conditions(
        self,
        weather_data: Dict[str, WeatherData]
    ) -> Dict[str, Any]:
        """Analyze weather conditions across regions"""
        
        if not weather_data:
            return {
                'avg_temperature': 20,
                'avg_precipitation': 50,
                'avg_wind_speed': 10,
                'avg_impact': 0,
                'conditions_summary': {}
            }
        
        temperatures = []
        precipitations = []
        wind_speeds = []
        impacts = []
        conditions_summary = {}
        
        for location, data in weather_data.items():
            weight = self.regional_weights.get(location, 0.1)
            
            temperatures.append(data.temperature * weight)
            precipitations.append(data.precipitation * weight)
            wind_speeds.append(data.wind_speed * weight)
            impacts.append(data.impact_score * weight)
            
            # Categorize conditions
            if data.temperature > 35:
                condition = 'extreme_heat'
            elif data.temperature < -10:
                condition = 'extreme_cold'
            elif data.precipitation > 100:
                condition = 'heavy_rain'
            elif data.precipitation < 10 and data.temperature > 25:
                condition = 'drought'
            else:
                condition = 'normal'
            
            conditions_summary[location] = {
                'condition': condition,
                'impact': data.impact_score,
                'affected_commodities': data.affected_commodities
            }
        
        return {
            'avg_temperature': sum(temperatures),
            'avg_precipitation': sum(precipitations),
            'avg_wind_speed': sum(wind_speeds),
            'avg_impact': sum(impacts),
            'conditions_summary': conditions_summary
        }
    
    def _calculate_commodity_impacts(
        self,
        weather_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate impact on commodities from weather"""
        
        commodity_scores = {}
        
        # Check each location's conditions
        for location, conditions in weather_analysis['conditions_summary'].items():
            condition = conditions['condition']
            
            if condition in self.commodity_impacts:
                impacts = self.commodity_impacts[condition]
                
                for commodity, impact in impacts.items():
                    if commodity not in commodity_scores:
                        commodity_scores[commodity] = 0
                    
                    # Weight by regional importance and condition impact
                    regional_weight = self.regional_weights.get(location, 0.1)
                    commodity_scores[commodity] += impact * regional_weight * conditions['impact']
        
        # Sort by impact magnitude
        sorted_impacts = []
        for commodity, score in commodity_scores.items():
            if abs(score) > 0.1:
                sorted_impacts.append({
                    'commodity': commodity,
                    'impact_score': score,
                    'direction': 'bullish' if score > 0 else 'bearish',
                    'magnitude': abs(score)
                })
        
        return sorted(sorted_impacts, key=lambda x: x['magnitude'], reverse=True)
    
    def _calculate_sector_impacts(
        self,
        weather_analysis: Dict[str, Any],
        symbol: str
    ) -> List[Dict[str, Any]]:
        """Calculate impact on sectors from weather"""
        
        sector_scores = {}
        
        avg_temp = weather_analysis['avg_temperature']
        avg_precip = weather_analysis['avg_precipitation']
        avg_wind = weather_analysis['avg_wind_speed']
        
        for sector, config in self.sector_impacts.items():
            score = 0
            
            # Temperature impacts
            if 'temperature_optimal' in config:
                opt_min, opt_max = config['temperature_optimal']
                if opt_min <= avg_temp <= opt_max:
                    score += 0.3
                else:
                    deviation = min(abs(avg_temp - opt_min), abs(avg_temp - opt_max))
                    score -= deviation / 20
            
            if config.get('temperature_extremes'):
                if avg_temp > 30 or avg_temp < 5:
                    score += 0.4  # Benefits from extreme temps
            
            # Precipitation impacts
            if 'precipitation_optimal' in config:
                opt_min, opt_max = config['precipitation_optimal']
                if opt_min <= avg_precip <= opt_max:
                    score += 0.2
                else:
                    score -= 0.3
            
            # Wind impacts
            if config.get('wind_impact'):
                if avg_wind > 15:
                    score += 0.2  # Wind energy benefit
            
            # Storm impacts
            if config.get('storm_impact') or config.get('catastrophe_sensitive'):
                if weather_analysis['avg_impact'] > 0.5:
                    score -= 0.5
            
            # Check if symbol is in sector
            if symbol in config.get('companies', []):
                score *= 1.5  # Direct impact
            
            if abs(score) > 0.1:
                sector_scores[sector] = score
        
        # Convert to sorted list
        sorted_impacts = []
        for sector, score in sector_scores.items():
            sorted_impacts.append({
                'sector': sector,
                'impact_score': score,
                'direction': 'positive' if score > 0 else 'negative',
                'companies': self.sector_impacts[sector].get('companies', [])
            })
        
        return sorted(sorted_impacts, key=lambda x: abs(x['impact_score']), reverse=True)
    
    def _detect_extreme_events(
        self,
        weather_data: Dict[str, WeatherData]
    ) -> List[Dict[str, Any]]:
        """Detect extreme weather events"""
        
        extreme_events = []
        
        for location, data in weather_data.items():
            # Extreme heat
            if data.temperature > 40:
                extreme_events.append({
                    'type': 'extreme_heat',
                    'location': location,
                    'severity': 'high',
                    'affected_commodities': ['wheat', 'corn', 'electricity']
                })
            
            # Extreme cold
            elif data.temperature < -15:
                extreme_events.append({
                    'type': 'extreme_cold',
                    'location': location,
                    'severity': 'high',
                    'affected_commodities': ['natural_gas', 'heating_oil']
                })
            
            # Heavy precipitation (potential flooding)
            if data.precipitation > 200:
                extreme_events.append({
                    'type': 'flooding',
                    'location': location,
                    'severity': 'medium',
                    'affected_commodities': ['crops', 'insurance']
                })
            
            # Drought conditions
            elif data.precipitation < 5 and data.temperature > 30:
                extreme_events.append({
                    'type': 'drought',
                    'location': location,
                    'severity': 'high',
                    'affected_commodities': ['wheat', 'corn', 'soybeans']
                })
            
            # High winds (potential storm)
            if data.wind_speed > 25:
                extreme_events.append({
                    'type': 'storm',
                    'location': location,
                    'severity': 'medium',
                    'affected_commodities': ['oil', 'insurance']
                })
        
        return extreme_events
    
    def _calculate_seasonal_factors(self) -> Dict[str, Any]:
        """Calculate seasonal weather factors"""
        
        current_month = datetime.now().month
        current_season = None
        
        for season, config in self.seasonal_patterns.items():
            if current_month in config['months']:
                current_season = season
                break
        
        if not current_season:
            current_season = 'unknown'
        
        seasonal_factors = {
            'current_season': current_season,
            'typical_impacts': self.seasonal_patterns.get(current_season, {}).get('typical_impacts', []),
            'seasonal_trades': []
        }
        
        # Suggest seasonal trades
        if current_season == 'winter':
            seasonal_factors['seasonal_trades'] = [
                {'commodity': 'natural_gas', 'direction': 'long'},
                {'commodity': 'heating_oil', 'direction': 'long'}
            ]
        elif current_season == 'summer':
            seasonal_factors['seasonal_trades'] = [
                {'commodity': 'electricity', 'direction': 'long'},
                {'sector': 'utilities', 'direction': 'long'}
            ]
        elif current_season == 'spring':
            seasonal_factors['seasonal_trades'] = [
                {'commodity': 'corn', 'direction': 'watch'},
                {'commodity': 'soybeans', 'direction': 'watch'}
            ]
        
        return seasonal_factors
    
    def _generate_signal_from_weather(
        self,
        weather_analysis: Dict,
        commodity_impacts: List[Dict],
        sector_impacts: List[Dict],
        extreme_events: List[Dict],
        seasonal_factors: Dict,
        symbol: str
    ) -> Dict[str, Any]:
        """Generate trading signal from weather analysis"""
        
        # Check if symbol is weather-sensitive
        weather_sensitive = False
        affected_score = 0
        
        # Check sectors
        for impact in sector_impacts:
            if symbol in impact.get('companies', []):
                weather_sensitive = True
                affected_score = impact['impact_score']
                break
        
        # Default action
        if not weather_sensitive:
            return {
                'action': 'hold',
                'reason': 'Symbol not significantly weather-sensitive'
            }
        
        # Base signal on impact
        if affected_score > 0.5:
            action = 'strong_buy'
        elif affected_score > 0.2:
            action = 'buy'
        elif affected_score < -0.5:
            action = 'strong_sell'
        elif affected_score < -0.2:
            action = 'sell'
        else:
            action = 'hold'
        
        # Adjust for extreme events
        if extreme_events:
            severity_score = sum(
                1 if e['severity'] == 'high' else 0.5
                for e in extreme_events
            )
            
            if severity_score > 1.5:
                # Multiple severe events
                if action in ['buy', 'strong_buy']:
                    action = 'hold'  # Be cautious
                elif action == 'hold':
                    action = 'sell'
        
        # Seasonal adjustments
        if seasonal_factors['current_season'] == 'winter':
            # Energy stocks benefit in winter
            if symbol in ['XOM', 'CVX', 'COP']:
                if action == 'hold':
                    action = 'buy'
        elif seasonal_factors['current_season'] == 'summer':
            # Utilities benefit in summer
            if symbol in ['NEE', 'DUK', 'SO']:
                if action == 'hold':
                    action = 'buy'
        
        return {
            'action': action,
            'impact_score': affected_score
        }
    
    def _calculate_confidence(
        self,
        weather_analysis: Dict,
        extreme_events: List[Dict],
        location_count: int
    ) -> float:
        """Calculate confidence in weather signal"""
        
        base_confidence = 0.5
        
        # Factor 1: Data coverage
        coverage_confidence = min(location_count / 5, 1.0) * 0.2
        
        # Factor 2: Impact magnitude
        impact_confidence = min(weather_analysis['avg_impact'], 1.0) * 0.2
        
        # Factor 3: Extreme events
        if extreme_events:
            extreme_confidence = 0.2
        else:
            extreme_confidence = 0.1
        
        # Factor 4: Condition consistency
        conditions = [c['condition'] for c in weather_analysis['conditions_summary'].values()]
        if conditions:
            unique_conditions = len(set(conditions))
            consistency_confidence = (1 - unique_conditions / len(conditions)) * 0.1
        else:
            consistency_confidence = 0
        
        confidence = (
            base_confidence +
            coverage_confidence +
            impact_confidence +
            extreme_confidence +
            consistency_confidence
        )
        
        return min(confidence, 0.90)
    
    def _generate_reasoning(
        self,
        weather_analysis: Dict,
        commodity_impacts: List[Dict],
        extreme_events: List[Dict],
        signal_data: Dict
    ) -> List[str]:
        """Generate reasoning for the signal"""
        
        reasoning = []
        
        # Weather summary
        avg_temp = weather_analysis['avg_temperature']
        if avg_temp > 30:
            reasoning.append(f"High temperatures ({avg_temp:.1f}°C) affecting energy demand")
        elif avg_temp < 5:
            reasoning.append(f"Low temperatures ({avg_temp:.1f}°C) increasing heating demand")
        
        # Top commodity impacts
        if commodity_impacts:
            top_commodity = commodity_impacts[0]
            reasoning.append(
                f"{top_commodity['commodity'].title()} {top_commodity['direction']} "
                f"(impact: {top_commodity['impact_score']:.2f})"
            )
        
        # Extreme events
        if extreme_events:
            event_types = list(set(e['type'] for e in extreme_events))
            reasoning.append(f"Extreme weather: {', '.join(event_types)}")
        
        # Overall impact
        if weather_analysis['avg_impact'] > 0.5:
            reasoning.append("Significant weather impact detected across regions")
        
        return reasoning[:5]
    
    def _identify_risks(
        self,
        weather_analysis: Dict,
        extreme_events: List[Dict]
    ) -> List[str]:
        """Identify weather-related risks"""
        
        risks = []
        
        # Extreme event risks
        if extreme_events:
            for event in extreme_events[:2]:
                risks.append(f"{event['type'].replace('_', ' ').title()} in {event['location']}")
        
        # High impact score
        if weather_analysis['avg_impact'] > 0.7:
            risks.append("Severe weather conditions may disrupt operations")
        
        # Mixed conditions
        conditions = set(c['condition'] for c in weather_analysis['conditions_summary'].values())
        if len(conditions) > 3:
            risks.append("Diverse weather patterns creating uncertainty")
        
        return risks[:4]
    
    def _format_regional_data(
        self,
        weather_data: Dict[str, WeatherData]
    ) -> List[Dict]:
        """Format regional weather data for display"""
        
        formatted = []
        
        for location, data in weather_data.items():
            formatted.append({
                'location': location,
                'temperature': round(data.temperature, 1),
                'precipitation': round(data.precipitation, 1),
                'conditions': data.conditions,
                'impact': round(data.impact_score, 2)
            })
        
        return formatted
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for weather impact analysis
        
        Returns:
            List of data type strings
        """
        return [
            'weather',  # Primary requirement - weather data
            'symbol',   # Stock symbol
            'price'     # Current price for context
        ]


# Create global instance
weather_impact_agent = WeatherImpactAgent()