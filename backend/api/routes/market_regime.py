"""
Market Regime API Routes
Provides endpoints for market regime classification and analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

from core.logging import get_logger
from agents.market_regime_agent import market_regime_agent, MarketRegime

logger = get_logger(__name__)
router = APIRouter(tags=["Market Regime"])


@router.post("/regime/analyze")
async def analyze_market_regime(
    market_data: Dict[str, Any],
    include_forecast: bool = True
) -> Dict[str, Any]:
    """
    Analyze current market regime based on provided market data.
    
    Args:
        market_data: Market indicators and data
        include_forecast: Whether to include forecasting analysis
    
    Returns:
        Complete regime analysis with classification, confidence, and recommendations
    """
    try:
        logger.info("Analyzing market regime")
        
        result = await market_regime_agent.analyze_regime(
            market_data=market_data,
            include_forecast=include_forecast
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Regime analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/current")
async def get_current_regime() -> Dict[str, Any]:
    """
    Get the current market regime classification.
    
    Returns:
        Current regime information
    """
    try:
        return {
            'regime': market_regime_agent.current_regime.value,
            'confidence': market_regime_agent.regime_confidence,
            'confidence_level': market_regime_agent.confidence_level.value,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get current regime: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/history")
async def get_regime_history(
    days: int = 30,
    include_indicators: bool = False
) -> Dict[str, Any]:
    """
    Get historical regime classifications.
    
    Args:
        days: Number of days of history to return
        include_indicators: Whether to include indicator data
    
    Returns:
        Historical regime data
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        history = []
        for record in market_regime_agent.regime_history:
            record_date = datetime.fromisoformat(record['timestamp'])
            if record_date >= cutoff_date:
                if not include_indicators:
                    # Exclude indicators to reduce payload size
                    filtered_record = {k: v for k, v in record.items() if k != 'indicators'}
                    history.append(filtered_record)
                else:
                    history.append(record)
        
        return {
            'history': history,
            'total_records': len(history),
            'date_range': {
                'from': cutoff_date.isoformat(),
                'to': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get regime history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/statistics")
async def get_regime_statistics() -> Dict[str, Any]:
    """
    Get regime classification statistics and performance metrics.
    
    Returns:
        Regime statistics and analytics
    """
    try:
        if not market_regime_agent.regime_history:
            return {
                'message': 'No regime history available',
                'statistics': {}
            }
        
        # Calculate regime frequency
        regime_counts = {}
        total_records = len(market_regime_agent.regime_history)
        
        for record in market_regime_agent.regime_history:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        regime_percentages = {
            regime: (count / total_records) * 100
            for regime, count in regime_counts.items()
        }
        
        # Calculate average confidence by regime
        regime_confidence = {}
        for regime in regime_counts.keys():
            confidences = [
                record['confidence'] 
                for record in market_regime_agent.regime_history 
                if record['regime'] == regime
            ]
            regime_confidence[regime] = sum(confidences) / len(confidences) if confidences else 0
        
        # Recent performance
        recent_records = market_regime_agent.regime_history[-30:] if len(market_regime_agent.regime_history) >= 30 else market_regime_agent.regime_history
        recent_avg_confidence = sum(r['confidence'] for r in recent_records) / len(recent_records)
        
        # Regime transitions
        transitions = []
        for i in range(1, len(market_regime_agent.regime_history)):
            prev_regime = market_regime_agent.regime_history[i-1]['regime']
            curr_regime = market_regime_agent.regime_history[i]['regime']
            if prev_regime != curr_regime:
                transitions.append({
                    'from': prev_regime,
                    'to': curr_regime,
                    'timestamp': market_regime_agent.regime_history[i]['timestamp']
                })
        
        return {
            'total_classifications': total_records,
            'regime_distribution': regime_percentages,
            'regime_counts': regime_counts,
            'average_confidence_by_regime': regime_confidence,
            'recent_average_confidence': recent_avg_confidence,
            'total_transitions': len(transitions),
            'recent_transitions': transitions[-10:] if len(transitions) > 10 else transitions,
            'current_regime': market_regime_agent.current_regime.value,
            'current_confidence': market_regime_agent.regime_confidence
        }
        
    except Exception as e:
        logger.error(f"Failed to get regime statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regime/forecast")
async def forecast_regime(
    market_data: Dict[str, Any],
    forecast_days: int = 30
) -> Dict[str, Any]:
    """
    Forecast regime evolution over specified time horizon.
    
    Args:
        market_data: Current market data
        forecast_days: Number of days to forecast
    
    Returns:
        Regime forecast and transition probabilities
    """
    try:
        # Analyze current regime
        current_analysis = await market_regime_agent.analyze_regime(market_data, include_forecast=True)
        
        # Generate forecast
        forecasts = current_analysis.get('forecasts', {})
        
        # Enhanced forecast with scenario analysis
        scenarios = {
            'base_case': {
                'probability': 0.6,
                'regime': current_analysis['regime'],
                'description': f"Continue in {current_analysis['regime']} regime"
            },
            'bull_scenario': {
                'probability': 0.15,
                'regime': MarketRegime.BULL.value,
                'description': "Transition to bull market"
            },
            'bear_scenario': {
                'probability': 0.15,
                'regime': MarketRegime.BEAR.value,
                'description': "Transition to bear market"
            },
            'crisis_scenario': {
                'probability': 0.05,
                'regime': MarketRegime.CRISIS.value,
                'description': "Market crisis scenario"
            },
            'sideways_scenario': {
                'probability': 0.05,
                'regime': MarketRegime.SIDEWAYS.value,
                'description': "Range-bound market"
            }
        }
        
        return {
            'current_regime': current_analysis['regime'],
            'current_confidence': current_analysis['confidence'],
            'forecast_horizon_days': forecast_days,
            'volatility_forecast': forecasts.get('volatility', {}),
            'regime_duration': forecasts.get('regime_duration', {}),
            'transition_probabilities': forecasts.get('transition_probabilities', {}),
            'scenarios': scenarios,
            'key_risks': current_analysis.get('risk_assessment', {}).get('key_risks', []),
            'recommended_strategies': current_analysis.get('strategy_recommendations', []),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Regime forecasting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/strategies")
async def get_regime_strategies(
    regime: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recommended trading strategies for current or specified regime.
    
    Args:
        regime: Specific regime to get strategies for (optional)
    
    Returns:
        Strategy recommendations
    """
    try:
        if regime:
            # Validate regime
            try:
                target_regime = MarketRegime(regime.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid regime: {regime}")
        else:
            target_regime = market_regime_agent.current_regime
        
        # Get strategies for the regime
        # Temporarily set the regime to get strategies
        original_regime = market_regime_agent.current_regime
        market_regime_agent.current_regime = target_regime
        
        strategies = market_regime_agent._recommend_strategies()
        
        # Restore original regime
        market_regime_agent.current_regime = original_regime
        
        # Enhanced strategy descriptions
        strategy_descriptions = {
            'momentum_long': 'Buy trending stocks and ETFs',
            'buy_breakouts': 'Enter positions on technical breakouts',
            'sell_volatility': 'Short volatility through options',
            'growth_stocks': 'Focus on growth over value',
            'defensive_positions': 'Hold defensive sectors and quality stocks',
            'buy_volatility': 'Long volatility through VIX products',
            'short_momentum': 'Fade momentum and buy weakness',
            'quality_stocks': 'Focus on high-quality, profitable companies',
            'mean_reversion': 'Buy oversold, sell overbought',
            'range_trading': 'Trade within established ranges',
            'iron_condors': 'Sell options strangles in low volatility',
            'pairs_trading': 'Long/short related securities',
            'risk_off': 'Reduce risk and preserve capital',
            'tail_hedges': 'Buy protection against extreme moves',
            'cash_preservation': 'Hold cash and short-term instruments',
            'safe_havens': 'Buy gold, bonds, and defensive assets',
            'theta_strategies': 'Sell time premium in options',
            'volatility_trading': 'Trade volatility products actively'
        }
        
        detailed_strategies = [
            {
                'strategy': strategy,
                'description': strategy_descriptions.get(strategy, 'Advanced trading strategy'),
                'risk_level': 'High' if 'short' in strategy or 'momentum' in strategy else 'Medium' if 'volatility' in strategy else 'Low'
            }
            for strategy in strategies
        ]
        
        return {
            'regime': target_regime.value,
            'total_strategies': len(strategies),
            'strategies': detailed_strategies,
            'risk_assessment': {
                'overall_risk': 'High' if target_regime in [MarketRegime.CRISIS, MarketRegime.BEAR] else 'Medium' if target_regime == MarketRegime.TRANSITION else 'Low',
                'recommended_position_size': '25%' if target_regime == MarketRegime.CRISIS else '50%' if target_regime == MarketRegime.BEAR else '75%' if target_regime == MarketRegime.TRANSITION else '100%'
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get regime strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regime/risk-assessment")
async def assess_regime_risk(
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess market risk based on current regime and conditions.
    
    Args:
        market_data: Current market indicators
    
    Returns:
        Comprehensive risk assessment
    """
    try:
        # Analyze regime first
        regime_analysis = await market_regime_agent.analyze_regime(market_data, include_forecast=False)
        
        risk_assessment = regime_analysis.get('risk_assessment', {})
        
        # Enhanced risk metrics
        risk_factors = {
            'regime_risk': {
                'level': risk_assessment.get('risk_level', 'UNKNOWN'),
                'score': risk_assessment.get('risk_score', 50),
                'description': f"Risk associated with {regime_analysis['regime']} regime"
            },
            'volatility_risk': {
                'level': 'High' if market_data.get('vix', 20) > 30 else 'Medium' if market_data.get('vix', 20) > 20 else 'Low',
                'score': min(market_data.get('vix', 20) * 2, 100),
                'description': 'Market volatility risk'
            },
            'liquidity_risk': {
                'level': 'High' if market_data.get('liquidity_score', 1.0) < 0.5 else 'Medium' if market_data.get('liquidity_score', 1.0) < 0.8 else 'Low',
                'score': (1.0 - market_data.get('liquidity_score', 1.0)) * 100,
                'description': 'Market liquidity risk'
            },
            'correlation_risk': {
                'level': 'High' if market_data.get('sector_correlation', 0.5) > 0.8 else 'Medium' if market_data.get('sector_correlation', 0.5) > 0.6 else 'Low',
                'score': market_data.get('sector_correlation', 0.5) * 100,
                'description': 'Cross-asset correlation risk'
            }
        }
        
        # Overall risk score (weighted average)
        weights = {'regime_risk': 0.4, 'volatility_risk': 0.3, 'liquidity_risk': 0.2, 'correlation_risk': 0.1}
        overall_score = sum(risk_factors[factor]['score'] * weights[factor] for factor in risk_factors)
        
        if overall_score > 80:
            overall_level = 'EXTREME'
        elif overall_score > 60:
            overall_level = 'HIGH'
        elif overall_score > 40:
            overall_level = 'ELEVATED'
        else:
            overall_level = 'MODERATE'
        
        return {
            'overall_risk': {
                'level': overall_level,
                'score': round(overall_score, 1)
            },
            'risk_factors': risk_factors,
            'key_risks': risk_assessment.get('key_risks', []),
            'regime_context': {
                'current_regime': regime_analysis['regime'],
                'confidence': regime_analysis['confidence'],
                'regime_risk_level': risk_assessment.get('risk_level', 'UNKNOWN')
            },
            'recommendations': {
                'position_sizing': '25%' if overall_level == 'EXTREME' else '50%' if overall_level == 'HIGH' else '75%' if overall_level == 'ELEVATED' else '100%',
                'hedging': 'Required' if overall_level in ['EXTREME', 'HIGH'] else 'Recommended' if overall_level == 'ELEVATED' else 'Optional',
                'strategies': market_regime_agent._recommend_strategies()[:3]
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/test-data")
async def get_test_market_data(
    scenario: str = "normal"
) -> Dict[str, Any]:
    """
    Generate test market data for different scenarios.
    
    Args:
        scenario: Test scenario (normal, bull, bear, crisis, sideways)
    
    Returns:
        Test market data for the specified scenario
    """
    try:
        scenarios = {
            'normal': {
                'vix': 18.5,
                'vix_change': 0.02,
                'vix_percentile': 45.0,
                'advance_decline_ratio': 1.2,
                'new_highs_lows': 0.1,
                'up_down_volume': 1.1,
                'spy_momentum_1d': 0.005,
                'spy_momentum_5d': 0.008,
                'spy_momentum_20d': 0.012,
                'volume_ratio': 1.0,
                'volume_surge': 1.0,
                'sector_correlation': 0.45,
                'bond_equity_correlation': -0.1,
                'dollar_strength': 0.0,
                'credit_spread': 1.0,
                'term_structure': 0.02,
                'put_call_ratio': 0.95,
                'vix_term_structure': 0.05,
                'liquidity_score': 0.85,
                'market_depth': 0.9
            },
            'bull': {
                'vix': 12.8,
                'vix_change': -0.05,
                'vix_percentile': 20.0,
                'advance_decline_ratio': 2.8,
                'new_highs_lows': 0.6,
                'up_down_volume': 1.8,
                'spy_momentum_1d': 0.015,
                'spy_momentum_5d': 0.025,
                'spy_momentum_20d': 0.035,
                'volume_ratio': 0.9,
                'volume_surge': 0.8,
                'sector_correlation': 0.35,
                'bond_equity_correlation': -0.3,
                'dollar_strength': 0.1,
                'credit_spread': 0.7,
                'term_structure': 0.05,
                'put_call_ratio': 0.65,
                'vix_term_structure': 0.15,
                'liquidity_score': 0.95,
                'market_depth': 0.95
            },
            'bear': {
                'vix': 28.5,
                'vix_change': 0.15,
                'vix_percentile': 75.0,
                'advance_decline_ratio': 0.4,
                'new_highs_lows': -0.7,
                'up_down_volume': 0.6,
                'spy_momentum_1d': -0.02,
                'spy_momentum_5d': -0.025,
                'spy_momentum_20d': -0.03,
                'volume_ratio': 1.6,
                'volume_surge': 1.8,
                'sector_correlation': 0.8,
                'bond_equity_correlation': 0.4,
                'dollar_strength': 0.2,
                'credit_spread': 2.1,
                'term_structure': -0.01,
                'put_call_ratio': 1.4,
                'vix_term_structure': -0.05,
                'liquidity_score': 0.65,
                'market_depth': 0.7
            },
            'crisis': {
                'vix': 65.0,
                'vix_change': 0.8,
                'vix_percentile': 99.0,
                'advance_decline_ratio': 0.1,
                'new_highs_lows': -0.95,
                'up_down_volume': 0.2,
                'spy_momentum_1d': -0.08,
                'spy_momentum_5d': -0.12,
                'spy_momentum_20d': -0.15,
                'volume_ratio': 3.5,
                'volume_surge': 4.0,
                'sector_correlation': 0.95,
                'bond_equity_correlation': 0.7,
                'dollar_strength': 0.5,
                'credit_spread': 5.0,
                'term_structure': -0.1,
                'put_call_ratio': 2.5,
                'vix_term_structure': -0.3,
                'liquidity_score': 0.2,
                'market_depth': 0.3
            },
            'sideways': {
                'vix': 16.2,
                'vix_change': 0.01,
                'vix_percentile': 35.0,
                'advance_decline_ratio': 1.0,
                'new_highs_lows': 0.0,
                'up_down_volume': 1.0,
                'spy_momentum_1d': 0.002,
                'spy_momentum_5d': -0.001,
                'spy_momentum_20d': 0.003,
                'volume_ratio': 0.95,
                'volume_surge': 0.9,
                'sector_correlation': 0.5,
                'bond_equity_correlation': 0.0,
                'dollar_strength': 0.0,
                'credit_spread': 1.1,
                'term_structure': 0.03,
                'put_call_ratio': 1.0,
                'vix_term_structure': 0.02,
                'liquidity_score': 0.8,
                'market_depth': 0.85
            }
        }
        
        if scenario not in scenarios:
            raise HTTPException(status_code=400, detail=f"Invalid scenario: {scenario}")
        
        return {
            'scenario': scenario,
            'market_data': scenarios[scenario],
            'description': f"Test market data for {scenario} market conditions",
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate test data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))