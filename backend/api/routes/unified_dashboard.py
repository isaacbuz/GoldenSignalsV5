"""
Unified Trading Dashboard API Routes
Integrates all agents and provides comprehensive market analysis
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json

from core.logging import get_logger

# Import all agents
from agents.market_regime_agent import market_regime_agent
from agents.smart_execution_agent import smart_execution_agent
from agents.liquidity_prediction_agent import liquidity_prediction_agent
from agents.risk_management_agent import risk_management_agent
from agents.volatility_agent import volatility_agent
from agents.technical_analysis_agent import technical_analysis_agent
from agents.sentiment_analysis_agent import sentiment_analysis_agent
from agents.arbitrage_detection_agent import arbitrage_detection_agent

logger = get_logger(__name__)
router = APIRouter(tags=["Unified Dashboard"])


@router.post("/dashboard/analyze/{symbol}")
async def comprehensive_analysis(
    symbol: str,
    include_realtime: bool = True,
    include_predictions: bool = True,
    include_arbitrage: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive analysis combining all agents for a single symbol
    
    Args:
        symbol: Trading symbol to analyze
        include_realtime: Include real-time market data
        include_predictions: Include predictive analytics
        include_arbitrage: Include arbitrage opportunities
    
    Returns:
        Complete analysis from all agents
    """
    try:
        logger.info(f"Running comprehensive analysis for {symbol}")
        
        # Prepare market data (would be fetched from data provider)
        market_data = await _fetch_market_data(symbol)
        
        # Run all analyses in parallel
        tasks = []
        
        # Core analyses
        tasks.append(market_regime_agent.analyze(symbol, market_data))
        tasks.append(technical_analysis_agent.analyze(symbol, market_data))
        tasks.append(volatility_agent.analyze_volatility(symbol, market_data, include_iv=True))
        tasks.append(sentiment_analysis_agent.analyze(symbol, market_data))
        tasks.append(liquidity_prediction_agent.predict_liquidity(symbol, market_data))
        
        # Risk assessment
        if market_data.get('portfolio'):
            tasks.append(risk_management_agent.analyze_risk(symbol, market_data))
        
        # Arbitrage detection (optional due to computational cost)
        if include_arbitrage:
            tasks.append(arbitrage_detection_agent.analyze(market_data))
        
        # Execute all analyses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        analysis_results = {}
        analysis_names = [
            'market_regime', 'technical_analysis', 'volatility', 
            'sentiment', 'liquidity', 'risk', 'arbitrage'
        ]
        
        for name, result in zip(analysis_names[:len(results)], results):
            if isinstance(result, Exception):
                logger.warning(f"{name} analysis failed: {str(result)}")
                analysis_results[name] = {'error': str(result)}
            else:
                analysis_results[name] = result.to_dict() if hasattr(result, 'to_dict') else result
        
        # Generate unified signal
        unified_signal = _generate_unified_signal(analysis_results)
        
        # Generate execution plan
        execution_plan = None
        if unified_signal['action'] != 'HOLD':
            execution_plan = await smart_execution_agent.create_execution_plan(
                symbol=symbol,
                side='buy' if 'BUY' in unified_signal['action'] else 'sell',
                quantity=unified_signal.get('suggested_quantity', 100),
                market_data=market_data
            )
            if execution_plan:
                analysis_results['execution_plan'] = execution_plan.to_dict()
        
        # Compile final response
        response = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analyses': analysis_results,
            'unified_signal': unified_signal,
            'market_data': {
                'price': market_data.get('close_prices', [])[-1] if market_data.get('close_prices') else None,
                'volume': market_data.get('volume', [])[-1] if market_data.get('volume') else None,
                'change_percent': _calculate_change_percent(market_data)
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/signals")
async def get_active_signals(
    min_confidence: float = 0.6,
    signal_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get all active trading signals across all agents
    
    Args:
        min_confidence: Minimum confidence threshold
        signal_types: Filter by specific signal types
    
    Returns:
        Active signals from all agents
    """
    try:
        signals = []
        
        # Collect signals from each agent's history
        agents = [
            ('market_regime', market_regime_agent),
            ('technical', technical_analysis_agent),
            ('volatility', volatility_agent),
            ('sentiment', sentiment_analysis_agent),
            ('arbitrage', arbitrage_detection_agent)
        ]
        
        for agent_name, agent in agents:
            if hasattr(agent, 'analysis_history'):
                for analysis in list(agent.analysis_history)[-10:]:  # Last 10 analyses
                    if hasattr(analysis, 'signal_strength'):
                        if analysis.signal_strength >= min_confidence:
                            signal = {
                                'source': agent_name,
                                'symbol': getattr(analysis, 'symbol', 'UNKNOWN'),
                                'signal': getattr(analysis, 'signal', 'UNKNOWN').value if hasattr(getattr(analysis, 'signal', None), 'value') else str(getattr(analysis, 'signal', 'UNKNOWN')),
                                'strength': analysis.signal_strength,
                                'timestamp': getattr(analysis, 'timestamp', datetime.now()).isoformat(),
                                'metadata': {}
                            }
                            
                            # Add agent-specific metadata
                            if agent_name == 'volatility':
                                signal['metadata']['regime'] = getattr(analysis, 'current_regime', 'UNKNOWN').value if hasattr(getattr(analysis, 'current_regime', None), 'value') else 'UNKNOWN'
                            elif agent_name == 'sentiment':
                                signal['metadata']['fear_greed'] = getattr(analysis.metrics, 'fear_greed_index', 50)
                            
                            signals.append(signal)
        
        # Sort by strength and timestamp
        signals.sort(key=lambda x: (x['strength'], x['timestamp']), reverse=True)
        
        return {
            'total_signals': len(signals),
            'signals': signals[:50],  # Top 50 signals
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/portfolio/risk")
async def get_portfolio_risk(
    positions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get comprehensive risk analysis for portfolio
    
    Args:
        positions: List of current positions
    
    Returns:
        Portfolio risk metrics and recommendations
    """
    try:
        if not positions:
            return {
                'risk_score': 0,
                'var_95': 0,
                'cvar_95': 0,
                'recommendations': ['No positions to analyze']
            }
        
        # Prepare portfolio data
        portfolio_data = {
            'positions': positions,
            'total_value': sum(p.get('value', 0) for p in positions)
        }
        
        # Get market data for all symbols
        symbols = [p['symbol'] for p in positions]
        market_data_tasks = [_fetch_market_data(symbol) for symbol in symbols]
        market_data_list = await asyncio.gather(*market_data_tasks)
        
        # Run risk analysis for each position
        risk_analyses = []
        for position, market_data in zip(positions, market_data_list):
            market_data['portfolio'] = portfolio_data
            risk_analysis = await risk_management_agent.analyze_risk(
                position['symbol'], 
                market_data
            )
            risk_analyses.append(risk_analysis)
        
        # Aggregate portfolio risk
        portfolio_risk = _aggregate_portfolio_risk(risk_analyses, positions)
        
        return portfolio_risk
        
    except Exception as e:
        logger.error(f"Portfolio risk analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/market/overview")
async def get_market_overview() -> Dict[str, Any]:
    """
    Get overall market overview and conditions
    
    Returns:
        Market overview including regime, sentiment, and key metrics
    """
    try:
        # Analyze major indices
        indices = ['SPY', 'QQQ', 'DIA', 'IWM']
        
        overview_data = {
            'indices': {},
            'market_regime': None,
            'overall_sentiment': None,
            'volatility_regime': None,
            'key_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Analyze each index
        for index in indices:
            market_data = await _fetch_market_data(index)
            
            # Get regime for SPY (market proxy)
            if index == 'SPY':
                regime_analysis = await market_regime_agent.analyze(index, market_data)
                overview_data['market_regime'] = regime_analysis.current_regime.value
                
                vol_analysis = await volatility_agent.analyze_volatility(index, market_data)
                overview_data['volatility_regime'] = vol_analysis.current_regime.value
                
                sentiment_analysis = await sentiment_analysis_agent.analyze(index)
                overview_data['overall_sentiment'] = {
                    'signal': sentiment_analysis.signal.value,
                    'fear_greed_index': sentiment_analysis.metrics.fear_greed_index
                }
            
            # Store index data
            overview_data['indices'][index] = {
                'price': market_data.get('close_prices', [])[-1] if market_data.get('close_prices') else None,
                'change_percent': _calculate_change_percent(market_data)
            }
        
        # Key market metrics
        overview_data['key_metrics'] = {
            'vix': await _get_vix_level(),
            'dollar_index': await _get_dollar_index(),
            'bond_yields': await _get_bond_yields(),
            'commodity_prices': await _get_commodity_prices()
        }
        
        return overview_data
        
    except Exception as e:
        logger.error(f"Market overview failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/dashboard/ws/{client_id}")
async def dashboard_websocket(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time dashboard updates
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await websocket.accept()
    logger.info(f"Dashboard WebSocket connected: {client_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            'type': 'connection',
            'status': 'connected',
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Start background task for periodic updates
        update_task = asyncio.create_task(
            _send_periodic_updates(websocket, client_id)
        )
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'subscribe':
                # Handle subscription to specific symbols
                symbols = data.get('symbols', [])
                await _handle_subscription(websocket, client_id, symbols)
            
            elif data.get('type') == 'unsubscribe':
                # Handle unsubscription
                symbols = data.get('symbols', [])
                await _handle_unsubscription(websocket, client_id, symbols)
            
            elif data.get('type') == 'ping':
                # Respond to ping
                await websocket.send_json({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        logger.info(f"Dashboard WebSocket disconnected: {client_id}")
        update_task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}")
        await websocket.close()


@router.get("/dashboard/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics for all agents
    
    Returns:
        Performance metrics for each agent
    """
    try:
        performance = {}
        
        # Collect performance from each agent
        agents = {
            'market_regime': market_regime_agent,
            'smart_execution': smart_execution_agent,
            'liquidity_prediction': liquidity_prediction_agent,
            'risk_management': risk_management_agent,
            'volatility': volatility_agent,
            'technical_analysis': technical_analysis_agent,
            'sentiment_analysis': sentiment_analysis_agent,
            'arbitrage_detection': arbitrage_detection_agent
        }
        
        for name, agent in agents.items():
            if hasattr(agent, 'get_performance_metrics'):
                performance[name] = agent.get_performance_metrics()
            else:
                performance[name] = {'status': 'no_metrics_available'}
        
        # Add system-wide metrics
        performance['system'] = {
            'total_analyses': sum(
                p.get('total_analyses', 0) 
                for p in performance.values() 
                if isinstance(p, dict)
            ),
            'active_agents': len(agents),
            'timestamp': datetime.now().isoformat()
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

async def _fetch_market_data(symbol: str) -> Dict[str, Any]:
    """Fetch market data for a symbol"""
    # This would integrate with actual data providers
    # For now, return mock data
    import numpy as np
    
    days = 100
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    return {
        'symbol': symbol,
        'close_prices': prices.tolist(),
        'high_prices': (prices * 1.01).tolist(),
        'low_prices': (prices * 0.99).tolist(),
        'open_prices': np.roll(prices, 1).tolist(),
        'volume': (np.random.uniform(1000000, 5000000, days)).tolist(),
        'exchange_prices': {
            'NYSE': {'price': prices[-1], 'volume': 1000000},
            'NASDAQ': {'price': prices[-1] * 1.001, 'volume': 800000}
        }
    }


def _generate_unified_signal(analyses: Dict[str, Any]) -> Dict[str, Any]:
    """Generate unified trading signal from all analyses"""
    
    signals = []
    weights = {
        'market_regime': 0.15,
        'technical_analysis': 0.20,
        'volatility': 0.15,
        'sentiment': 0.20,
        'liquidity': 0.10,
        'risk': 0.10,
        'arbitrage': 0.10
    }
    
    # Collect signals from each analysis
    for agent_name, analysis in analyses.items():
        if 'error' in analysis:
            continue
        
        weight = weights.get(agent_name, 0.1)
        
        # Extract signal and confidence
        signal = None
        confidence = 0.0
        
        if agent_name == 'technical_analysis' and 'signal' in analysis:
            signal_map = {
                'strong_buy': 1.0, 'buy': 0.5, 'neutral': 0.0,
                'sell': -0.5, 'strong_sell': -1.0
            }
            signal = signal_map.get(analysis['signal'], 0.0)
            confidence = analysis.get('signal_strength', 0.5)
        
        elif agent_name == 'sentiment' and 'signal' in analysis:
            signal_map = {
                'extreme_bullish': 1.0, 'bullish': 0.5, 'neutral': 0.0,
                'bearish': -0.5, 'extreme_bearish': -1.0
            }
            signal = signal_map.get(analysis['signal'], 0.0)
            confidence = analysis.get('signal_strength', 0.5)
        
        elif agent_name == 'volatility' and 'primary_signal' in analysis:
            signal_map = {
                'long_volatility': 0.3, 'short_volatility': -0.3,
                'volatility_breakout': 0.5, 'mean_reversion': 0.0,
                'neutral': 0.0
            }
            signal = signal_map.get(analysis['primary_signal'], 0.0)
            confidence = analysis.get('signal_strength', 0.5)
        
        if signal is not None:
            signals.append({
                'value': signal,
                'weight': weight,
                'confidence': confidence
            })
    
    # Calculate weighted signal
    if not signals:
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'contributors': []
        }
    
    total_weight = sum(s['weight'] * s['confidence'] for s in signals)
    weighted_signal = sum(s['value'] * s['weight'] * s['confidence'] for s in signals) / total_weight if total_weight > 0 else 0
    
    # Determine action
    if weighted_signal > 0.6:
        action = 'STRONG_BUY'
    elif weighted_signal > 0.2:
        action = 'BUY'
    elif weighted_signal < -0.6:
        action = 'STRONG_SELL'
    elif weighted_signal < -0.2:
        action = 'SELL'
    else:
        action = 'HOLD'
    
    # Calculate overall confidence
    avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
    
    return {
        'action': action,
        'signal_value': weighted_signal,
        'confidence': avg_confidence,
        'contributors': [
            {'agent': agent_name, 'contribution': s['value'] * s['weight'] * s['confidence']}
            for agent_name, s in zip(analyses.keys(), signals)
        ],
        'suggested_quantity': _calculate_position_size(weighted_signal, avg_confidence)
    }


def _calculate_change_percent(market_data: Dict[str, Any]) -> float:
    """Calculate percentage change"""
    prices = market_data.get('close_prices', [])
    if len(prices) >= 2:
        return ((prices[-1] - prices[-2]) / prices[-2]) * 100
    return 0.0


def _calculate_position_size(signal_value: float, confidence: float) -> int:
    """Calculate suggested position size based on signal and confidence"""
    base_size = 100
    size_multiplier = abs(signal_value) * confidence
    return int(base_size * (1 + size_multiplier))


def _aggregate_portfolio_risk(risk_analyses: List[Any], positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate individual position risks into portfolio risk"""
    
    if not risk_analyses:
        return {'error': 'No risk analyses available'}
    
    # Calculate weighted risk metrics
    total_value = sum(p.get('value', 0) for p in positions)
    
    weighted_var = 0
    weighted_risk_score = 0
    all_risks = []
    
    for analysis, position in zip(risk_analyses, positions):
        weight = position.get('value', 0) / total_value if total_value > 0 else 0
        
        if hasattr(analysis, 'metrics'):
            weighted_var += analysis.metrics.var_95 * weight
            weighted_risk_score += analysis.risk_score * weight
            all_risks.extend(analysis.risk_factors)
    
    # Compile recommendations
    recommendations = []
    if weighted_risk_score > 0.7:
        recommendations.append("High portfolio risk - consider reducing position sizes")
    if len(set(p['symbol'][:3] for p in positions)) < 3:
        recommendations.append("Low diversification - consider adding positions in different sectors")
    
    return {
        'portfolio_risk_score': weighted_risk_score,
        'portfolio_var_95': weighted_var,
        'unique_risks': list(set(all_risks))[:10],
        'recommendations': recommendations,
        'position_count': len(positions),
        'total_value': total_value
    }


async def _get_vix_level() -> float:
    """Get current VIX level"""
    # Would fetch from data provider
    return 18.5  # Mock value


async def _get_dollar_index() -> float:
    """Get current dollar index"""
    # Would fetch from data provider
    return 102.3  # Mock value


async def _get_bond_yields() -> Dict[str, float]:
    """Get current bond yields"""
    # Would fetch from data provider
    return {
        '2Y': 4.85,
        '10Y': 4.25,
        '30Y': 4.40
    }


async def _get_commodity_prices() -> Dict[str, float]:
    """Get current commodity prices"""
    # Would fetch from data provider
    return {
        'gold': 2050.0,
        'oil': 78.5,
        'silver': 23.4
    }


async def _send_periodic_updates(websocket: WebSocket, client_id: str):
    """Send periodic updates to WebSocket client"""
    try:
        while True:
            # Send market update every 5 seconds
            await asyncio.sleep(5)
            
            # Get latest signals
            signals = await get_active_signals(min_confidence=0.7)
            
            await websocket.send_json({
                'type': 'update',
                'data': {
                    'signals': signals['signals'][:5],  # Top 5 signals
                    'timestamp': datetime.now().isoformat()
                }
            })
    except Exception as e:
        logger.error(f"Error sending periodic updates to {client_id}: {str(e)}")


async def _handle_subscription(websocket: WebSocket, client_id: str, symbols: List[str]):
    """Handle symbol subscription"""
    logger.info(f"Client {client_id} subscribed to: {symbols}")
    
    # Send initial data for subscribed symbols
    for symbol in symbols:
        market_data = await _fetch_market_data(symbol)
        await websocket.send_json({
            'type': 'symbol_data',
            'symbol': symbol,
            'data': {
                'price': market_data['close_prices'][-1],
                'change_percent': _calculate_change_percent(market_data)
            }
        })


async def _handle_unsubscription(websocket: WebSocket, client_id: str, symbols: List[str]):
    """Handle symbol unsubscription"""
    logger.info(f"Client {client_id} unsubscribed from: {symbols}")
    
    await websocket.send_json({
        'type': 'unsubscribed',
        'symbols': symbols,
        'timestamp': datetime.now().isoformat()
    })