"""
Liquidity Prediction API Routes
Provides endpoints for liquidity analysis and execution recommendations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from core.logging import get_logger
from agents.liquidity_prediction_agent import liquidity_prediction_agent, LiquidityLevel, MarketSession

logger = get_logger(__name__)
router = APIRouter(tags=["Liquidity Prediction"])


@router.post("/liquidity/predict")
async def predict_liquidity(
    symbol: str,
    market_data: Dict[str, Any],
    forecast_horizon: int = 30
) -> Dict[str, Any]:
    """
    Predict future liquidity conditions for a symbol.
    
    Args:
        symbol: Trading symbol
        market_data: Current market data (price, volume, bid, ask, etc.)
        forecast_horizon: Minutes to forecast ahead
    
    Returns:
        Comprehensive liquidity prediction and analysis
    """
    try:
        logger.info(f"Predicting liquidity for {symbol} with {forecast_horizon}min horizon")
        
        prediction = await liquidity_prediction_agent.predict_liquidity(
            symbol=symbol,
            market_data=market_data,
            forecast_horizon=forecast_horizon
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Liquidity prediction failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/liquidity/execution-recommendation")
async def get_execution_recommendation(
    symbol: str,
    order_size: float,
    side: str,
    market_data: Dict[str, Any],
    urgency: str = "normal",
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get optimal execution strategy recommendation.
    
    Args:
        symbol: Trading symbol
        order_size: Order size (shares or notional value)
        side: 'buy' or 'sell'
        market_data: Current market data
        urgency: 'immediate', 'normal', 'patient', or 'opportunistic'
        constraints: Optional execution constraints
    
    Returns:
        Comprehensive execution recommendation
    """
    try:
        if side.lower() not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
        
        if order_size <= 0:
            raise HTTPException(status_code=400, detail="Order size must be positive")
        
        if urgency not in ['immediate', 'normal', 'patient', 'opportunistic']:
            raise HTTPException(status_code=400, detail="Invalid urgency level")
        
        logger.info(f"Generating execution recommendation: {symbol} {side} {order_size}")
        
        recommendation = await liquidity_prediction_agent.recommend_execution(
            symbol=symbol,
            order_size=order_size,
            side=side.lower(),
            market_data=market_data,
            urgency=urgency,
            constraints=constraints
        )
        
        return recommendation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execution recommendation failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liquidity/optimal-windows")
async def get_optimal_execution_windows(
    date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get optimal execution windows for the trading day.
    
    Args:
        date: Date in YYYY-MM-DD format (optional, defaults to today)
    
    Returns:
        Optimal execution windows with recommendations
    """
    try:
        # Use current time or parse provided date
        if date:
            try:
                target_date = datetime.fromisoformat(f"{date}T09:30:00")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            target_date = datetime.now()
        
        # Get optimal windows using the agent's internal method
        windows = liquidity_prediction_agent._identify_optimal_windows(target_date)
        
        # Add session descriptions
        session_descriptions = {
            'pre_market': 'Limited liquidity, wider spreads, institutional activity',
            'open_auction': 'High volume, price discovery, good for large orders',
            'morning': 'Best overall liquidity, institutional activity peak',
            'lunch': 'Reduced liquidity, avoid if possible',
            'afternoon': 'Moderate liquidity, retail activity increase',
            'close_auction': 'High volume, benchmarking orders, volatility',
            'after_hours': 'Very limited liquidity, wide spreads'
        }
        
        # Enhance windows with additional info
        for window in windows:
            session = window['session']
            window['description'] = session_descriptions.get(session, 'Standard trading session')
            
            # Add risk assessment
            if session in ['open_auction', 'close_auction']:
                window['risk_level'] = 'medium'
                window['volatility'] = 'high'
            elif session == 'lunch':
                window['risk_level'] = 'high'
                window['volatility'] = 'low'
            else:
                window['risk_level'] = 'low'
                window['volatility'] = 'medium'
        
        return {
            'date': target_date.date().isoformat(),
            'optimal_windows': windows,
            'total_windows': len(windows),
            'recommendations': {
                'best_for_large_orders': ['open_auction', 'morning'],
                'avoid_for_sensitive_orders': ['lunch', 'close_auction'],
                'best_overall_liquidity': ['morning', 'afternoon']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimal windows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/liquidity/impact-estimate")
async def estimate_market_impact(
    symbol: str,
    order_size: float,
    market_data: Dict[str, Any],
    urgency: str = "normal"
) -> Dict[str, Any]:
    """
    Estimate market impact for an order.
    
    Args:
        symbol: Trading symbol
        order_size: Order size (shares or notional value)
        market_data: Current market data
        urgency: Execution urgency level
    
    Returns:
        Detailed market impact estimate
    """
    try:
        if order_size <= 0:
            raise HTTPException(status_code=400, detail="Order size must be positive")
        
        # Calculate liquidity metrics
        current_time = datetime.now()
        metrics = liquidity_prediction_agent._calculate_liquidity_metrics(market_data, current_time)
        
        # Estimate impact using the liquidity analyzer
        impact_est = liquidity_prediction_agent.liquidity_analyzer.estimate_market_impact(
            order_size, metrics, urgency
        )
        
        # Additional analysis
        adv_percentage = order_size / metrics.average_volume * 100
        mid_price = market_data.get('price', (market_data.get('bid', 100) + market_data.get('ask', 100.1)) / 2)
        
        # Risk categorization
        if adv_percentage > 10:
            size_category = "very_large"
            risk_level = "extreme"
        elif adv_percentage > 5:
            size_category = "large"
            risk_level = "high"
        elif adv_percentage > 1:
            size_category = "medium"
            risk_level = "medium"
        else:
            size_category = "small"
            risk_level = "low"
        
        return {
            'symbol': symbol,
            'order_size': order_size,
            'market_conditions': {
                'volume_ratio': metrics.volume_ratio,
                'spread_bps': metrics.bid_ask_spread * 10000,
                'session': metrics.session.value,
                'depth_imbalance': metrics.depth_imbalance
            },
            'impact_estimate': {
                'total_impact_bps': impact_est['total_impact_bps'],
                'permanent_impact_bps': impact_est['permanent_impact_bps'],
                'temporary_impact_bps': impact_est['temporary_impact_bps'],
                'spread_cost_bps': impact_est['spread_cost_bps'],
                'total_cost_bps': impact_est['total_cost_bps']
            },
            'order_analysis': {
                'adv_percentage': adv_percentage,
                'size_category': size_category,
                'risk_level': risk_level,
                'estimated_cost_usd': impact_est['total_cost_bps'] / 10000 * order_size * mid_price,
                'urgency_multiplier': {
                    'immediate': 1.5,
                    'normal': 1.0,
                    'patient': 0.7,
                    'opportunistic': 0.5
                }.get(urgency, 1.0)
            },
            'recommendations': [
                f"Order represents {adv_percentage:.1f}% of average daily volume",
                f"Expected total cost: {impact_est['total_cost_bps']:.1f} basis points",
                f"Risk level: {risk_level.upper()}",
                "Consider using algorithmic execution for orders >5% ADV" if adv_percentage > 5 else "Standard execution acceptable"
            ],
            'timestamp': current_time.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market impact estimation failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liquidity/session-analysis")
async def get_session_analysis() -> Dict[str, Any]:
    """
    Get analysis of current trading session and intraday patterns.
    
    Returns:
        Current session analysis and typical patterns
    """
    try:
        current_time = datetime.now()
        current_session = liquidity_prediction_agent._get_current_session(current_time)
        
        # Get intraday patterns
        patterns = liquidity_prediction_agent.intraday_patterns
        current_pattern = patterns.get(current_session, {})
        
        # Session timing
        session_times = {
            'pre_market': {'start': '04:00', 'end': '09:30'},
            'open_auction': {'start': '09:30', 'end': '10:00'},
            'morning': {'start': '10:00', 'end': '12:00'},
            'lunch': {'start': '12:00', 'end': '14:00'},
            'afternoon': {'start': '14:00', 'end': '15:30'},
            'close_auction': {'start': '15:30', 'end': '16:00'},
            'after_hours': {'start': '16:00', 'end': '20:00'}
        }
        
        # Calculate minutes until next session
        session_order = ['pre_market', 'open_auction', 'morning', 'lunch', 'afternoon', 'close_auction', 'after_hours']
        current_index = session_order.index(current_session.value)
        
        if current_index < len(session_order) - 1:
            next_session = session_order[current_index + 1]
            next_session_enum = getattr(MarketSession, next_session.upper())
        else:
            next_session = 'pre_market'  # Next day
            next_session_enum = MarketSession.PRE_MARKET
        
        return {
            'current_session': {
                'name': current_session.value,
                'display_name': current_session.value.replace('_', ' ').title(),
                'typical_volume_pct': current_pattern.get('typical_volume_pct', 0) * 100,
                'typical_spread_mult': current_pattern.get('typical_spread_mult', 1.0),
                'volatility': current_pattern.get('volatility', 'unknown'),
                'times': session_times.get(current_session.value, {})
            },
            'next_session': {
                'name': next_session,
                'display_name': next_session.replace('_', ' ').title()
            },
            'intraday_patterns': {
                session.value: {
                    'volume_pct': pattern.get('typical_volume_pct', 0) * 100,
                    'spread_multiplier': pattern.get('typical_spread_mult', 1.0),
                    'volatility': pattern.get('volatility', 'unknown'),
                    'recommendation': liquidity_prediction_agent._get_window_recommendation(session)
                }
                for session, pattern in patterns.items()
            },
            'trading_insights': [
                "Morning session (10:00-12:00) typically has best liquidity",
                "Lunch period (12:00-14:00) has reduced liquidity - avoid large orders",
                "Opening and closing auctions have high volatility but good volume",
                "After-hours trading has very limited liquidity and wide spreads"
            ],
            'timestamp': current_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Session analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liquidity/performance")
async def get_agent_performance() -> Dict[str, Any]:
    """
    Get liquidity prediction agent performance metrics.
    
    Returns:
        Agent performance and accuracy statistics
    """
    try:
        performance = liquidity_prediction_agent.get_performance_metrics()
        
        # Add additional context
        performance['agent_status'] = {
            'initialized': True,
            'prediction_history_size': len(liquidity_prediction_agent.prediction_history),
            'max_history_size': liquidity_prediction_agent.prediction_history.maxlen,
            'supported_urgency_levels': ['immediate', 'normal', 'patient', 'opportunistic'],
            'supported_order_types': ['market', 'limit', 'iceberg', 'vwap', 'twap', 'moc', 'loc']
        }
        
        # Add model capabilities
        performance['capabilities'] = {
            'liquidity_classification': ['very_high', 'high', 'normal', 'low', 'very_low', 'dried_up'],
            'market_sessions': ['pre_market', 'open_auction', 'morning', 'lunch', 'afternoon', 'close_auction', 'after_hours'],
            'impact_estimation': 'Square-root model with liquidity adjustments',
            'execution_scheduling': 'Algorithmic slice optimization',
            'risk_assessment': 'Multi-factor risk scoring'
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get agent performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/liquidity/test")
async def run_liquidity_test(
    scenario: str = "normal"
) -> Dict[str, Any]:
    """
    Run liquidity prediction test with predefined scenarios.
    
    Args:
        scenario: Test scenario ('normal', 'high_volume', 'low_liquidity', 'crisis')
    
    Returns:
        Test results with predictions and recommendations
    """
    try:
        # Predefined test scenarios
        scenarios = {
            'normal': {
                'symbol': 'AAPL',
                'price': 195.00,
                'bid': 194.98,
                'ask': 195.02,
                'volume': 50000000,
                'avg_volume': 65000000
            },
            'high_volume': {
                'symbol': 'TSLA',
                'price': 312.20,
                'bid': 312.10,
                'ask': 312.30,
                'volume': 100000000,
                'avg_volume': 75000000
            },
            'low_liquidity': {
                'symbol': 'XYZ',
                'price': 50.05,
                'bid': 50.00,
                'ask': 50.10,
                'volume': 500000,
                'avg_volume': 2000000
            },
            'crisis': {
                'symbol': 'SPY',
                'price': 450.00,
                'bid': 449.50,
                'ask': 450.50,
                'volume': 200000000,
                'avg_volume': 80000000
            }
        }
        
        if scenario not in scenarios:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid scenario. Choose from: {list(scenarios.keys())}"
            )
        
        test_data = scenarios[scenario]
        symbol = test_data['symbol']
        
        logger.info(f"Running liquidity test for scenario: {scenario}")
        
        # Run prediction
        prediction = await liquidity_prediction_agent.predict_liquidity(
            symbol=symbol,
            market_data=test_data,
            forecast_horizon=30
        )
        
        # Run execution recommendation
        test_order_size = test_data['avg_volume'] * 0.02  # 2% of ADV
        execution_rec = await liquidity_prediction_agent.recommend_execution(
            symbol=symbol,
            order_size=test_order_size,
            side='buy',
            market_data=test_data,
            urgency='normal'
        )
        
        # Impact estimation
        impact_est = await estimate_market_impact(
            symbol=symbol,
            order_size=test_order_size,
            market_data=test_data,
            urgency='normal'
        )
        
        return {
            'scenario': scenario,
            'test_data': test_data,
            'liquidity_prediction': prediction,
            'execution_recommendation': execution_rec,
            'impact_estimation': impact_est,
            'test_summary': {
                'current_liquidity': prediction['current_liquidity']['level'],
                'predicted_liquidity': prediction['predicted_liquidity']['level'],
                'recommended_strategy': execution_rec['execution_strategy']['recommended_type'],
                'expected_cost_bps': execution_rec['execution_strategy']['expected_cost_bps'],
                'risk_level': execution_rec['risk_assessment']['risk_level'],
                'warnings_count': len(prediction['warnings'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liquidity test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))