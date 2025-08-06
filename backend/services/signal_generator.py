"""
Signal Generator Service
Combines multiple AI models and agents to generate high-confidence trading signals
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from core.logging import get_logger
from services.ml.lstm_predictor import lstm_predictor
from agents.orchestrator import AgentOrchestrator
from services.signal_service import SignalService
from models.signal import Signal, SignalAction, SignalStatus, RiskLevel
from core.database import get_db

logger = get_logger(__name__)


class SignalGenerator:
    """
    Master signal generator that combines:
    - LSTM predictions
    - FinGPT analysis
    - Technical indicators
    - Economic indicators
    - Risk assessment
    """
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.signal_service = SignalService()
        self.is_initialized = False
        
        # Signal generation configuration
        self.config = {
            'min_confidence': 0.65,  # Minimum confidence to generate signal
            'consensus_threshold': 0.7,  # Minimum consensus among models
            'risk_limits': {
                'low': 0.02,  # 2% risk
                'medium': 0.05,  # 5% risk
                'high': 0.10  # 10% risk
            }
        }
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Signal Generator")
            
            # Initialize orchestrator with agents
            await self.orchestrator.initialize_default_agents()
            
            # Start orchestrator
            await self.orchestrator.start()
            
            self.is_initialized = True
            logger.info("Signal Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Signal Generator: {e}")
            raise
    
    async def generate_signals(self, symbols: List[str]) -> List[Signal]:
        """
        Generate signals for multiple symbols
        
        Args:
            symbols: List of stock symbols to analyze
            
        Returns:
            List of generated signals
        """
        
        if not self.is_initialized:
            await self.initialize()
        
        # Generate signals concurrently
        tasks = [self.generate_signal(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and None results
        signals = []
        for result in results:
            if isinstance(result, Signal):
                signals.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error generating signal: {result}")
        
        return signals
    
    async def generate_signal(self, symbol: str) -> Optional[Signal]:
        """
        Generate a single signal for a symbol
        
        This is the main entry point that combines all analyses
        """
        
        try:
            logger.info(f"Generating signal for {symbol}")
            
            # Get market data
            market_data = await self._get_market_data(symbol)
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            # Run parallel analyses
            analyses = await asyncio.gather(
                self._get_lstm_prediction(symbol),
                self._get_agent_analysis(market_data),
                self._get_risk_assessment(symbol, market_data),
                return_exceptions=True
            )
            
            lstm_result = analyses[0] if not isinstance(analyses[0], Exception) else None
            agent_result = analyses[1] if not isinstance(analyses[1], Exception) else None
            risk_result = analyses[2] if not isinstance(analyses[2], Exception) else None
            
            # Combine analyses
            combined_analysis = self._combine_analyses(
                lstm_result,
                agent_result,
                risk_result,
                market_data
            )
            
            # Check if we should generate a signal
            if not self._should_generate_signal(combined_analysis):
                logger.info(f"Signal criteria not met for {symbol}")
                return None
            
            # Create signal
            signal = await self._create_signal(symbol, combined_analysis, market_data)
            
            # Store signal in database
            async for db in get_db():
                stored_signal = await self.signal_service.create_signal(
                    db=db,
                    symbol=signal.symbol,
                    action=signal.action.value,
                    confidence=signal.confidence,
                    source="AI_Ensemble",
                    reasoning=signal.reasoning,
                    metadata={
                        'predicted_price': signal.predicted_price,
                        'model_accuracy': signal.model_accuracy,
                        'feature_importance': signal.feature_importance
                    }
                )
                logger.info(f"Signal generated and stored: {stored_signal.id}")
                break
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol"""
        
        # This would normally call your market data service
        # For now, returning mock data
        return {
            'symbol': symbol,
            'price': 150.00,
            'volume': 1000000,
            'change': 2.50,
            'change_percent': 1.69,
            'high': 152.00,
            'low': 148.00,
            'open': 149.00,
            'timestamp': datetime.now()
        }
    
    async def _get_lstm_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get LSTM prediction"""
        
        try:
            prediction = await lstm_predictor.predict(symbol)
            return prediction
        except Exception as e:
            logger.error(f"LSTM prediction error for {symbol}: {e}")
            return None
    
    async def _get_agent_analysis(self, market_data: Dict[str, Any]) -> Optional[Any]:
        """Get multi-agent analysis"""
        
        try:
            # Run orchestrator analysis
            signal = await self.orchestrator.analyze_market(market_data)
            return signal
        except Exception as e:
            logger.error(f"Agent analysis error: {e}")
            return None
    
    async def _get_risk_assessment(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for the position"""
        
        # Calculate basic risk metrics
        volatility = 0.02  # 2% daily volatility (would calculate from historical data)
        
        # Determine risk level based on volatility
        if volatility < 0.015:
            risk_level = 'low'
        elif volatility < 0.025:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        # Calculate position sizing
        risk_limit = self.config['risk_limits'][risk_level]
        position_size = risk_limit / volatility  # Simple position sizing
        
        return {
            'risk_level': risk_level,
            'volatility': volatility,
            'position_size': min(position_size, 0.10),  # Max 10% position
            'stop_loss': market_data['price'] * (1 - 2 * volatility),
            'take_profit': market_data['price'] * (1 + 3 * volatility)
        }
    
    def _combine_analyses(
        self,
        lstm_result: Optional[Dict[str, Any]],
        agent_result: Optional[Any],
        risk_result: Optional[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all analyses into a unified view"""
        
        combined = {
            'signals': [],
            'confidences': [],
            'predicted_prices': [],
            'reasoning': []
        }
        
        # Add LSTM results
        if lstm_result and 'signal' in lstm_result:
            combined['signals'].append(lstm_result['signal'])
            combined['confidences'].append(lstm_result.get('confidence', 0.5))
            combined['predicted_prices'].append(lstm_result.get('predicted_price'))
            combined['reasoning'].append(f"LSTM predicts {lstm_result['signal']} with {lstm_result.get('confidence', 0):.0%} confidence")
        
        # Add agent results
        if agent_result:
            action_map = {
                SignalAction.BUY: 'BUY',
                SignalAction.SELL: 'SELL',
                SignalAction.HOLD: 'HOLD'
            }
            combined['signals'].append(action_map.get(agent_result.action, 'HOLD'))
            combined['confidences'].append(agent_result.confidence)
            combined['reasoning'].extend(agent_result.reasoning[:3])  # Top 3 reasons
        
        # Add risk assessment
        if risk_result:
            combined['risk'] = risk_result
        
        # Calculate consensus
        if combined['signals']:
            # Count votes
            buy_votes = combined['signals'].count('BUY')
            sell_votes = combined['signals'].count('SELL')
            hold_votes = combined['signals'].count('HOLD')
            
            total_votes = len(combined['signals'])
            
            if buy_votes > sell_votes and buy_votes > hold_votes:
                combined['consensus_signal'] = 'BUY'
                combined['consensus_strength'] = buy_votes / total_votes
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                combined['consensus_signal'] = 'SELL'
                combined['consensus_strength'] = sell_votes / total_votes
            else:
                combined['consensus_signal'] = 'HOLD'
                combined['consensus_strength'] = hold_votes / total_votes
            
            # Average confidence
            combined['average_confidence'] = np.mean(combined['confidences'])
            
            # Average predicted price
            valid_prices = [p for p in combined['predicted_prices'] if p]
            if valid_prices:
                combined['predicted_price'] = np.mean(valid_prices)
        
        return combined
    
    def _should_generate_signal(self, analysis: Dict[str, Any]) -> bool:
        """Determine if we should generate a signal based on analysis"""
        
        # Check minimum confidence
        avg_confidence = analysis.get('average_confidence', 0)
        if avg_confidence < self.config['min_confidence']:
            return False
        
        # Check consensus strength
        consensus_strength = analysis.get('consensus_strength', 0)
        if consensus_strength < self.config['consensus_threshold']:
            return False
        
        # Don't generate HOLD signals
        if analysis.get('consensus_signal') == 'HOLD':
            return False
        
        return True
    
    async def _create_signal(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Signal:
        """Create a Signal object from analysis"""
        
        # Map signal to action
        action_map = {
            'BUY': SignalAction.BUY,
            'SELL': SignalAction.SELL,
            'HOLD': SignalAction.HOLD
        }
        
        action = action_map.get(analysis['consensus_signal'], SignalAction.HOLD)
        
        # Get risk info
        risk_info = analysis.get('risk', {})
        risk_level_map = {
            'low': RiskLevel.LOW,
            'medium': RiskLevel.MEDIUM,
            'high': RiskLevel.HIGH
        }
        risk_level = risk_level_map.get(risk_info.get('risk_level', 'medium'), RiskLevel.MEDIUM)
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=analysis['average_confidence'],
            price=market_data['price'],
            target_price=analysis.get('predicted_price'),
            stop_loss=risk_info.get('stop_loss'),
            take_profit=risk_info.get('take_profit'),
            risk_level=risk_level,
            timeframe='1d',
            reasoning=' | '.join(analysis['reasoning'][:5]),  # Top 5 reasons
            consensus_strength=analysis['consensus_strength'],
            agents_consensus={
                'lstm': analysis['signals'][0] if analysis['signals'] else None,
                'agents': analysis['signals'][1] if len(analysis['signals']) > 1 else None
            },
            signal_source='AI_Ensemble',
            market_regime='normal',  # Would determine from market conditions
            volatility_score=risk_info.get('volatility', 0.02),
            status=SignalStatus.ACTIVE
        )
        
        # Add AI-specific fields (these would be added to the model)
        signal.predicted_price = analysis.get('predicted_price')
        signal.prediction_timeframe = '1d'
        signal.model_version = 'v1.0'
        signal.model_accuracy = 0.92  # Target accuracy
        signal.feature_importance = {
            'lstm_weight': 0.4,
            'fingpt_weight': 0.3,
            'technical_weight': 0.2,
            'risk_weight': 0.1
        }
        
        return signal


# Create singleton instance
signal_generator = SignalGenerator()