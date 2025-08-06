"""
Alternative Data Master Agent
Orchestrates all alternative data sources to generate comprehensive trading signals
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from agents.base import BaseAgent, AgentContext, AgentCapability, AgentConfig, Signal, SignalAction, SignalStrength
from agents.news_sentiment_agent import news_sentiment_agent
from agents.social_sentiment_agent import social_sentiment_agent
from agents.weather_impact_agent import weather_impact_agent
from agents.commodity_data_agent import commodity_data_agent
from services.alternative_data_service import alternative_data_service, DataSourceType

logger = logging.getLogger(__name__)


class AlternativeDataMasterAgent(BaseAgent):
    """Master agent that orchestrates all alternative data sources"""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="AlternativeDataMaster",
                confidence_threshold=0.65
            )
        super().__init__(config)
        
        # Sub-agents with their weights
        self.sub_agents = {
            'news': {
                'agent': news_sentiment_agent,
                'weight': 0.30,
                'enabled': True
            },
            'social': {
                'agent': social_sentiment_agent,
                'weight': 0.25,
                'enabled': True
            },
            'weather': {
                'agent': weather_impact_agent,
                'weight': 0.20,
                'enabled': True
            },
            'commodity': {
                'agent': commodity_data_agent,
                'weight': 0.25,
                'enabled': True
            }
        }
        
        # Cross-signal validation thresholds
        self.validation_thresholds = {
            'agreement_threshold': 0.6,  # 60% of agents must agree
            'conflict_penalty': 0.2,     # Penalty for conflicting signals
            'strong_consensus_bonus': 0.15  # Bonus for unanimous agreement
        }
        
        # Signal aggregation methods
        self.aggregation_methods = {
            'weighted_average': self._weighted_average_aggregation,
            'majority_vote': self._majority_vote_aggregation,
            'confidence_weighted': self._confidence_weighted_aggregation,
            'adaptive': self._adaptive_aggregation
        }
        
        # Market condition adjustments
        self.market_conditions = {
            'high_volatility': {
                'news_weight_adjustment': 0.1,
                'social_weight_adjustment': -0.1
            },
            'trending': {
                'commodity_weight_adjustment': 0.1,
                'news_weight_adjustment': -0.05
            },
            'range_bound': {
                'social_weight_adjustment': 0.1,
                'commodity_weight_adjustment': -0.1
            }
        }
        
        # Historical performance tracking
        self.agent_performance = defaultdict(lambda: {
            'correct_signals': 0,
            'total_signals': 0,
            'avg_confidence': 0,
            'recent_accuracy': []
        })
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Orchestrate all alternative data sources to generate a unified signal"""
        
        try:
            symbol = market_data.get('symbol', 'SPY')
            
            # Collect signals from all sub-agents in parallel
            sub_signals = await self._collect_sub_signals(market_data)
            
            if not sub_signals:
                logger.warning("No alternative data signals available")
                return None
            
            # Analyze signal agreement
            agreement_analysis = self._analyze_signal_agreement(sub_signals)
            
            # Detect anomalies across data sources
            anomalies = self._detect_cross_source_anomalies(sub_signals)
            
            # Determine market condition
            market_condition = self._determine_market_condition(sub_signals, market_data)
            
            # Adjust weights based on performance and conditions
            adjusted_weights = self._adjust_agent_weights(market_condition)
            
            # Aggregate signals using adaptive method
            aggregated_signal = self._adaptive_aggregation(
                sub_signals,
                adjusted_weights,
                agreement_analysis
            )
            
            # Generate comprehensive analysis
            comprehensive_analysis = self._generate_comprehensive_analysis(
                sub_signals,
                agreement_analysis,
                anomalies,
                market_condition
            )
            
            # Calculate final confidence
            final_confidence = self._calculate_final_confidence(
                sub_signals,
                agreement_analysis,
                anomalies
            )
            
            # Determine signal strength
            strength = self._determine_signal_strength(
                final_confidence,
                agreement_analysis,
                aggregated_signal
            )
            
            # Map action
            action = self._map_action(aggregated_signal['action'])
            current_price = market_data.get('price', 0.0)
            
            # Generate reasoning
            reasoning = self._generate_comprehensive_reasoning(
                sub_signals,
                agreement_analysis,
                comprehensive_analysis,
                aggregated_signal
            )
            
            # Identify risks
            risks = self._identify_comprehensive_risks(
                sub_signals,
                anomalies,
                agreement_analysis
            )
            
            return Signal(
                symbol=symbol,
                action=action,
                confidence=final_confidence,
                strength=strength,
                source=self.config.name,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    'sub_signals': self._format_sub_signals(sub_signals),
                    'agreement_analysis': agreement_analysis,
                    'anomalies': anomalies,
                    'market_condition': market_condition,
                    'comprehensive_analysis': comprehensive_analysis,
                    'data_sources_used': list(sub_signals.keys()),
                    'risks': risks
                },
                market_conditions={
                    'signals_collected': len(sub_signals),
                    'agreement_score': agreement_analysis['agreement_score'],
                    'anomaly_detected': len(anomalies) > 0,
                    'market_regime': market_condition
                }
            )
            
        except Exception as e:
            logger.error(f"Alternative data master analysis error: {e}")
            return None
    
    async def _collect_sub_signals(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Signal]:
        """Collect signals from all enabled sub-agents"""
        
        tasks = {}
        
        for name, config in self.sub_agents.items():
            if config['enabled']:
                tasks[name] = config['agent'].analyze(market_data)
        
        # Run all analyses in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Map results back to agent names
        sub_signals = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Signal):
                sub_signals[name] = result
            elif isinstance(result, Exception):
                logger.error(f"Error from {name} agent: {result}")
        
        return sub_signals
    
    def _analyze_signal_agreement(
        self,
        sub_signals: Dict[str, Signal]
    ) -> Dict[str, Any]:
        """Analyze agreement between different data sources"""
        
        if not sub_signals:
            return {
                'agreement_score': 0,
                'consensus_action': 'hold',
                'dissenting_sources': [],
                'confidence_spread': 0
            }
        
        # Count actions
        action_counts = defaultdict(int)
        confidences = []
        
        for name, signal in sub_signals.items():
            action_counts[signal.action.value] += 1
            confidences.append(signal.confidence)
        
        # Find consensus action
        total_signals = len(sub_signals)
        consensus_action = max(action_counts, key=action_counts.get)
        agreement_count = action_counts[consensus_action]
        
        # Calculate agreement score
        agreement_score = agreement_count / total_signals
        
        # Find dissenting sources
        dissenting_sources = [
            name for name, signal in sub_signals.items()
            if signal.action.value != consensus_action
        ]
        
        # Calculate confidence spread
        confidence_spread = max(confidences) - min(confidences) if confidences else 0
        
        return {
            'agreement_score': agreement_score,
            'consensus_action': consensus_action,
            'dissenting_sources': dissenting_sources,
            'confidence_spread': confidence_spread,
            'action_distribution': dict(action_counts)
        }
    
    def _detect_cross_source_anomalies(
        self,
        sub_signals: Dict[str, Signal]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies across different data sources"""
        
        anomalies = []
        
        # Check for conflicting strong signals
        strong_signals = {
            name: signal for name, signal in sub_signals.items()
            if signal.strength == SignalStrength.STRONG
        }
        
        if len(strong_signals) > 1:
            actions = [s.action.value for s in strong_signals.values()]
            if 'buy' in actions and 'sell' in actions:
                anomalies.append({
                    'type': 'conflicting_strong_signals',
                    'sources': list(strong_signals.keys()),
                    'severity': 'high'
                })
        
        # Check for unusual confidence patterns
        if sub_signals:
            confidences = [s.confidence for s in sub_signals.values()]
            avg_confidence = np.mean(confidences)
            
            for name, signal in sub_signals.items():
                if abs(signal.confidence - avg_confidence) > 0.3:
                    anomalies.append({
                        'type': 'confidence_outlier',
                        'source': name,
                        'confidence': signal.confidence,
                        'average': avg_confidence,
                        'severity': 'medium'
                    })
        
        # Check for data source specific anomalies
        if 'news' in sub_signals and 'social' in sub_signals:
            news_signal = sub_signals['news']
            social_signal = sub_signals['social']
            
            # News and social usually correlate
            if news_signal.action != social_signal.action:
                sentiment_divergence = abs(
                    news_signal.features.get('sentiment_score', 0) -
                    social_signal.features.get('sentiment_score', 0)
                )
                
                if sentiment_divergence > 0.5:
                    anomalies.append({
                        'type': 'sentiment_divergence',
                        'sources': ['news', 'social'],
                        'divergence': sentiment_divergence,
                        'severity': 'medium'
                    })
        
        return anomalies
    
    def _determine_market_condition(
        self,
        sub_signals: Dict[str, Signal],
        market_data: Dict[str, Any]
    ) -> str:
        """Determine current market condition"""
        
        # Default condition
        condition = 'normal'
        
        # Check volatility from signals
        if 'commodity' in sub_signals:
            commodity_signal = sub_signals['commodity']
            if commodity_signal.features.get('commodity_trends', {}).get('avg_volatility', 0) > 0.3:
                condition = 'high_volatility'
        
        # Check trending from multiple sources
        trending_indicators = 0
        
        if 'news' in sub_signals:
            if abs(sub_signals['news'].features.get('momentum', 0)) > 0.2:
                trending_indicators += 1
        
        if 'social' in sub_signals:
            if sub_signals['social'].features.get('viral_metrics', {}).get('is_viral', False):
                trending_indicators += 1
        
        if trending_indicators >= 2:
            condition = 'trending'
        
        # Check for range-bound conditions
        if all(s.action == SignalAction.HOLD for s in sub_signals.values()):
            condition = 'range_bound'
        
        return condition
    
    def _adjust_agent_weights(
        self,
        market_condition: str
    ) -> Dict[str, float]:
        """Adjust agent weights based on market conditions and performance"""
        
        adjusted_weights = {}
        
        for name, config in self.sub_agents.items():
            base_weight = config['weight']
            
            # Apply market condition adjustments
            if market_condition in self.market_conditions:
                adjustment_key = f"{name}_weight_adjustment"
                adjustment = self.market_conditions[market_condition].get(adjustment_key, 0)
                base_weight += adjustment
            
            # Apply performance-based adjustments
            performance = self.agent_performance[name]
            if performance['total_signals'] > 10:
                accuracy = performance['correct_signals'] / performance['total_signals']
                if accuracy > 0.6:
                    base_weight *= 1.1
                elif accuracy < 0.4:
                    base_weight *= 0.9
            
            adjusted_weights[name] = max(0.1, min(0.4, base_weight))  # Keep weights reasonable
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _weighted_average_aggregation(
        self,
        sub_signals: Dict[str, Signal],
        weights: Dict[str, float],
        agreement_analysis: Dict
    ) -> Dict[str, Any]:
        """Aggregate signals using weighted average"""
        
        if not sub_signals:
            return {'action': 'hold', 'score': 0}
        
        action_scores = {
            'buy': 1.0,
            'hold': 0.0,
            'sell': -1.0
        }
        
        weighted_score = 0
        total_weight = 0
        
        for name, signal in sub_signals.items():
            weight = weights.get(name, 0.25)
            score = action_scores.get(signal.action.value, 0)
            weighted_score += score * weight * signal.confidence
            total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine action from score
        if final_score > 0.3:
            action = 'buy'
        elif final_score < -0.3:
            action = 'sell'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'score': final_score
        }
    
    def _majority_vote_aggregation(
        self,
        sub_signals: Dict[str, Signal],
        weights: Dict[str, float],
        agreement_analysis: Dict
    ) -> Dict[str, Any]:
        """Aggregate signals using majority vote"""
        
        return {
            'action': agreement_analysis['consensus_action'],
            'score': agreement_analysis['agreement_score']
        }
    
    def _confidence_weighted_aggregation(
        self,
        sub_signals: Dict[str, Signal],
        weights: Dict[str, float],
        agreement_analysis: Dict
    ) -> Dict[str, Any]:
        """Aggregate signals weighted by confidence"""
        
        action_confidence = defaultdict(float)
        
        for signal in sub_signals.values():
            action_confidence[signal.action.value] += signal.confidence
        
        # Find action with highest total confidence
        best_action = max(action_confidence, key=action_confidence.get)
        total_confidence = sum(action_confidence.values())
        
        return {
            'action': best_action,
            'score': action_confidence[best_action] / total_confidence if total_confidence > 0 else 0
        }
    
    def _adaptive_aggregation(
        self,
        sub_signals: Dict[str, Signal],
        weights: Dict[str, float],
        agreement_analysis: Dict
    ) -> Dict[str, Any]:
        """Adaptively choose best aggregation method"""
        
        # High agreement: use majority vote
        if agreement_analysis['agreement_score'] > 0.75:
            return self._majority_vote_aggregation(sub_signals, weights, agreement_analysis)
        
        # Low confidence spread: use weighted average
        elif agreement_analysis['confidence_spread'] < 0.2:
            return self._weighted_average_aggregation(sub_signals, weights, agreement_analysis)
        
        # Default: confidence weighted
        else:
            return self._confidence_weighted_aggregation(sub_signals, weights, agreement_analysis)
    
    def _generate_comprehensive_analysis(
        self,
        sub_signals: Dict[str, Signal],
        agreement_analysis: Dict,
        anomalies: List[Dict],
        market_condition: str
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis from all data sources"""
        
        analysis = {
            'primary_drivers': [],
            'sentiment_summary': {},
            'key_risks': [],
            'opportunities': [],
            'data_quality': 'good'
        }
        
        # Extract primary drivers
        for name, signal in sub_signals.items():
            if signal.confidence > 0.7:
                analysis['primary_drivers'].append({
                    'source': name,
                    'action': signal.action.value,
                    'confidence': signal.confidence
                })
        
        # Sentiment summary
        if 'news' in sub_signals:
            analysis['sentiment_summary']['news'] = sub_signals['news'].features.get('sentiment_score', 0)
        if 'social' in sub_signals:
            analysis['sentiment_summary']['social'] = sub_signals['social'].features.get('sentiment_score', 0)
        
        # Key risks from all sources
        for signal in sub_signals.values():
            if 'risks' in signal.features:
                analysis['key_risks'].extend(signal.features['risks'][:2])
        
        # Data quality assessment
        if len(anomalies) > 2:
            analysis['data_quality'] = 'poor'
        elif len(anomalies) > 0:
            analysis['data_quality'] = 'fair'
        
        return analysis
    
    def _calculate_final_confidence(
        self,
        sub_signals: Dict[str, Signal],
        agreement_analysis: Dict,
        anomalies: List[Dict]
    ) -> float:
        """Calculate final confidence for the master signal"""
        
        # Start with average confidence
        if sub_signals:
            base_confidence = np.mean([s.confidence for s in sub_signals.values()])
        else:
            base_confidence = 0.5
        
        # Adjust for agreement
        agreement_adjustment = (agreement_analysis['agreement_score'] - 0.5) * 0.2
        
        # Penalty for anomalies
        anomaly_penalty = len(anomalies) * 0.05
        
        # Penalty for confidence spread
        spread_penalty = agreement_analysis['confidence_spread'] * 0.1
        
        final_confidence = base_confidence + agreement_adjustment - anomaly_penalty - spread_penalty
        
        return max(0.1, min(0.95, final_confidence))
    
    def _determine_signal_strength(
        self,
        confidence: float,
        agreement_analysis: Dict,
        aggregated_signal: Dict
    ) -> SignalStrength:
        """Determine strength of the aggregated signal"""
        
        if confidence > 0.75 and agreement_analysis['agreement_score'] > 0.75:
            return SignalStrength.STRONG
        elif confidence > 0.60 or agreement_analysis['agreement_score'] > 0.60:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _map_action(self, action_str: str) -> SignalAction:
        """Map action string to SignalAction enum"""
        
        action_map = {
            'buy': SignalAction.BUY,
            'sell': SignalAction.SELL,
            'hold': SignalAction.HOLD
        }
        
        return action_map.get(action_str, SignalAction.HOLD)
    
    def _generate_comprehensive_reasoning(
        self,
        sub_signals: Dict[str, Signal],
        agreement_analysis: Dict,
        comprehensive_analysis: Dict,
        aggregated_signal: Dict
    ) -> List[str]:
        """Generate comprehensive reasoning from all sources"""
        
        reasoning = []
        
        # Overall signal
        reasoning.append(
            f"Alternative data consensus: {aggregated_signal['action']} "
            f"(score: {aggregated_signal['score']:.2f})"
        )
        
        # Agreement level
        if agreement_analysis['agreement_score'] > 0.75:
            reasoning.append("Strong agreement across data sources")
        elif agreement_analysis['agreement_score'] < 0.5:
            reasoning.append("Mixed signals from different sources")
        
        # Primary drivers
        if comprehensive_analysis['primary_drivers']:
            driver = comprehensive_analysis['primary_drivers'][0]
            reasoning.append(f"Primary driver: {driver['source']} ({driver['confidence']:.2f})")
        
        # Sentiment summary
        sentiment = comprehensive_analysis['sentiment_summary']
        if sentiment:
            avg_sentiment = np.mean(list(sentiment.values()))
            if abs(avg_sentiment) > 0.3:
                direction = "bullish" if avg_sentiment > 0 else "bearish"
                reasoning.append(f"Overall sentiment {direction}")
        
        # Add specific insights from top signals
        for name, signal in list(sub_signals.items())[:2]:
            if signal.reasoning:
                reasoning.append(f"{name.title()}: {signal.reasoning[0]}")
        
        return reasoning[:5]
    
    def _identify_comprehensive_risks(
        self,
        sub_signals: Dict[str, Signal],
        anomalies: List[Dict],
        agreement_analysis: Dict
    ) -> List[str]:
        """Identify comprehensive risks from all sources"""
        
        risks = []
        
        # Disagreement risk
        if agreement_analysis['agreement_score'] < 0.5:
            risks.append("Low agreement between data sources")
        
        # Anomaly risks
        for anomaly in anomalies[:2]:
            if anomaly['severity'] == 'high':
                risks.append(f"High severity: {anomaly['type'].replace('_', ' ')}")
        
        # Collect risks from sub-signals
        risk_counts = defaultdict(int)
        for signal in sub_signals.values():
            if 'risks' in signal.features:
                for risk in signal.features['risks']:
                    risk_counts[risk] += 1
        
        # Add most common risks
        if risk_counts:
            common_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
            for risk, count in common_risks[:2]:
                if count > 1:
                    risks.append(f"Multiple sources: {risk}")
        
        return risks[:4]
    
    def _format_sub_signals(
        self,
        sub_signals: Dict[str, Signal]
    ) -> Dict[str, Dict]:
        """Format sub-signals for display"""
        
        formatted = {}
        
        for name, signal in sub_signals.items():
            formatted[name] = {
                'action': signal.action.value,
                'confidence': round(signal.confidence, 2),
                'strength': signal.strength.value
            }
        
        return formatted
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types
        
        Returns:
            List of data type strings
        """
        return [
            'alternative_data',  # All alternative data sources
            'symbol',
            'price'
        ]


# Create global instance
alternative_data_master_agent = AlternativeDataMasterAgent()