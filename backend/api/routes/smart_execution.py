"""
Smart Execution API Routes
Professional order execution with multiple strategies
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import uuid

from core.logging import get_logger
from agents.smart_execution_agent import (
    smart_execution_agent, ExecutionOrder, ExecutionStrategy, 
    OrderType, Venue, MarketConditions
)

logger = get_logger(__name__)
router = APIRouter(tags=["Smart Execution"])


@router.post("/execution/order")
async def create_execution_order(
    symbol: str,
    side: str,  # 'buy' or 'sell'
    quantity: int,
    strategy: str = "twap",
    order_type: str = "limit",
    limit_price: Optional[float] = None,
    time_constraint: Optional[int] = None,
    urgency: str = "normal",
    participate_rate: float = 0.1,
    venue_preferences: List[str] = [],
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Create and execute a smart order
    
    Args:
        symbol: Trading symbol
        side: 'buy' or 'sell'
        quantity: Order quantity
        strategy: Execution strategy (twap, vwap, pov, is, iceberg, etc.)
        order_type: Order type (limit, market, etc.)
        limit_price: Limit price (optional)
        time_constraint: Time constraint in minutes
        urgency: Urgency level (low, normal, high, immediate)
        participate_rate: Max participation rate (0.0-1.0)
        venue_preferences: Preferred venues
    """
    try:
        # Validate inputs
        if side.lower() not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
        
        if quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        # Convert string enums
        try:
            exec_strategy = ExecutionStrategy(strategy.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
        
        try:
            exec_order_type = OrderType(order_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid order type: {order_type}")
        
        # Convert venue preferences
        venues = []
        for venue_str in venue_preferences:
            try:
                venues.append(Venue(venue_str.lower()))
            except ValueError:
                logger.warning(f"Invalid venue: {venue_str}")
        
        # Create execution order
        order = ExecutionOrder(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side.lower(),
            total_quantity=quantity,
            strategy=exec_strategy,
            order_type=exec_order_type,
            limit_price=limit_price,
            time_constraint=time_constraint,
            urgency=urgency,
            participate_rate=participate_rate,
            venue_preferences=venues
        )
        
        # Mock market conditions (in production, fetch real data)
        market = MarketConditions(
            symbol=symbol.upper(),
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.2,
            bid_size=1000,
            ask_size=1200,
            last_price=100.1,
            volume=50000,
            mid=100.1,
            spread_bps=20,
            avg_volume=100000,
            volatility=0.25,
            liquidity_score=0.8,
            momentum=0.1,
            market_impact=5.0
        )
        
        # Execute in background if requested
        if background_tasks:
            background_tasks.add_task(
                execute_order_background,
                order,
                market
            )
            
            return {
                'order_id': order.order_id,
                'status': 'submitted',
                'message': 'Order submitted for execution',
                'estimated_completion': f"{time_constraint or 60} minutes"
            }
        else:
            # Execute immediately
            result = await smart_execution_agent.execute_order(order, market)
            return {
                'order_id': order.order_id,
                'status': 'completed',
                'execution_result': result.to_dict()
            }
        
    except Exception as e:
        logger.error(f"Order creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_order_background(order: ExecutionOrder, market: MarketConditions):
    """Execute order in background"""
    try:
        result = await smart_execution_agent.execute_order(order, market)
        logger.info(f"Background execution completed: {order.order_id}")
    except Exception as e:
        logger.error(f"Background execution failed: {str(e)}")


@router.get("/execution/strategies")
async def get_execution_strategies() -> Dict[str, Any]:
    """Get available execution strategies"""
    
    strategies = {
        'twap': {
            'name': 'Time-Weighted Average Price',
            'description': 'Spread order evenly over time',
            'best_for': 'Large orders with time flexibility',
            'parameters': ['time_constraint']
        },
        'vwap': {
            'name': 'Volume-Weighted Average Price',
            'description': 'Execute based on historical volume patterns',
            'best_for': 'Orders seeking market VWAP',
            'parameters': ['time_constraint', 'participate_rate']
        },
        'pov': {
            'name': 'Percentage of Volume',
            'description': 'Execute as percentage of market volume',
            'best_for': 'Consistent market participation',
            'parameters': ['participate_rate']
        },
        'is': {
            'name': 'Implementation Shortfall',
            'description': 'Optimize trade-off between market impact and timing risk',
            'best_for': 'Sophisticated execution optimization',
            'parameters': ['time_constraint', 'urgency']
        },
        'iceberg': {
            'name': 'Iceberg Strategy',
            'description': 'Hide order size using small visible slices',
            'best_for': 'Large orders requiring stealth',
            'parameters': ['max_display_size', 'iceberg_refresh_rate']
        },
        'sniper': {
            'name': 'Sniper Strategy',
            'description': 'Opportunistic fast execution',
            'best_for': 'Immediate execution needs',
            'parameters': ['urgency']
        },
        'adaptive': {
            'name': 'Adaptive Strategy',
            'description': 'AI-driven strategy selection based on market conditions',
            'best_for': 'Dynamic market conditions',
            'parameters': ['adaptive_parameters']
        },
        'dark_pool': {
            'name': 'Dark Pool Strategy',
            'description': 'Execute primarily in dark pools',
            'best_for': 'Large institutional orders',
            'parameters': ['dark_pool_preference']
        },
        'smart_routing': {
            'name': 'Smart Order Routing',
            'description': 'Intelligent multi-venue execution',
            'best_for': 'Best execution across venues',
            'parameters': ['venue_preferences']
        }
    }
    
    return {
        'strategies': strategies,
        'total': len(strategies)
    }


@router.get("/execution/venues")
async def get_execution_venues() -> Dict[str, Any]:
    """Get available execution venues"""
    
    venues = {
        'primary': {
            'name': 'Primary Exchange',
            'description': 'NYSE, NASDAQ primary listings',
            'latency_ms': 1.2,
            'liquidity': 'High',
            'cost': 'Standard'
        },
        'dark_pool': {
            'name': 'Dark Pools',
            'description': 'Hidden liquidity pools',
            'latency_ms': 2.8,
            'liquidity': 'Variable',
            'cost': 'Low impact'
        },
        'ecn': {
            'name': 'Electronic Communication Networks',
            'description': 'ECNs like ARCA, BATS',
            'latency_ms': 0.8,
            'liquidity': 'Good',
            'cost': 'Competitive'
        },
        'ats': {
            'name': 'Alternative Trading Systems',
            'description': 'Alternative trading venues',
            'latency_ms': 1.5,
            'liquidity': 'Moderate',
            'cost': 'Variable'
        }
    }
    
    return {
        'venues': venues,
        'total': len(venues)
    }


@router.get("/execution/order/{order_id}")
async def get_order_status(order_id: str) -> Dict[str, Any]:
    """Get order execution status"""
    
    try:
        # Check active orders
        if order_id in smart_execution_agent.active_orders:
            order = smart_execution_agent.active_orders[order_id]
            active_slices = [
                slice_data for slice_id, slice_data in smart_execution_agent.active_slices.items()
                if slice_data.parent_order_id == order_id
            ]
            
            return {
                'order_id': order_id,
                'status': 'working',
                'order_details': order.to_dict(),
                'active_slices': len(active_slices),
                'slice_details': [
                    {
                        'slice_id': s.slice_id,
                        'status': s.status.value,
                        'quantity': s.quantity,
                        'filled': s.filled_quantity,
                        'venue': s.venue.value
                    }
                    for s in active_slices
                ]
            }
        
        # Check execution history
        for result in smart_execution_agent.execution_history:
            if result.order_id == order_id:
                return {
                    'order_id': order_id,
                    'status': 'completed',
                    'execution_result': result.to_dict()
                }
        
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        
    except Exception as e:
        logger.error(f"Failed to get order status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/performance")
async def get_execution_performance() -> Dict[str, Any]:
    """Get execution agent performance metrics"""
    
    try:
        performance = smart_execution_agent.get_performance_summary()
        
        return {
            'performance_summary': performance,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execution/market-impact")
async def estimate_market_impact(
    symbol: str,
    quantity: int,
    side: str,
    urgency: str = "normal",
    participate_rate: float = 0.1
) -> Dict[str, Any]:
    """Estimate market impact for an order"""
    
    try:
        # Create temporary order for impact estimation
        order = ExecutionOrder(
            order_id="impact_estimate",
            symbol=symbol.upper(),
            side=side.lower(),
            total_quantity=quantity,
            strategy=ExecutionStrategy.TWAP,  # Default for estimation
            urgency=urgency,
            participate_rate=participate_rate
        )
        
        # Mock market conditions
        market = MarketConditions(
            symbol=symbol.upper(),
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.2,
            bid_size=1000,
            ask_size=1200,
            last_price=100.1,
            volume=50000,
            mid=100.1,
            spread_bps=20,
            avg_volume=100000,
            volatility=0.25,
            liquidity_score=0.8
        )
        
        # Estimate impact
        impact_estimate = smart_execution_agent.impact_model.estimate_market_impact(
            order, market
        )
        
        return {
            'symbol': symbol.upper(),
            'quantity': quantity,
            'side': side,
            'impact_estimate_bps': impact_estimate,
            'estimated_cost_usd': (impact_estimate['total_impact_bps'] / 10000) * (quantity * market.mid),
            'market_conditions': {
                'spread_bps': market.spread_bps,
                'liquidity_score': market.liquidity_score,
                'volatility': market.volatility
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market impact estimation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/test-order")
async def create_test_order() -> Dict[str, Any]:
    """Create a test execution order for demonstration"""
    
    try:
        # Create test order
        order = ExecutionOrder(
            order_id=str(uuid.uuid4()),
            symbol="TSLA",
            side="buy",
            total_quantity=5000,
            strategy=ExecutionStrategy.TWAP,
            order_type=OrderType.LIMIT,
            time_constraint=30,  # 30 minutes
            urgency="normal",
            participate_rate=0.15
        )
        
        # Mock market conditions for TSLA
        market = MarketConditions(
            symbol="TSLA",
            timestamp=datetime.now(),
            bid=312.10,
            ask=312.30,
            bid_size=800,
            ask_size=1200,
            last_price=312.21,
            volume=25000,
            mid=312.20,
            spread_bps=6.4,  # 20 cents / 312.20 * 10000
            avg_volume=75000,
            volatility=0.35,  # Tesla's typical volatility
            liquidity_score=0.85,
            momentum=0.15  # Positive momentum
        )
        
        # Execute the test order
        result = await smart_execution_agent.execute_order(order, market)
        
        return {
            'test_order_id': order.order_id,
            'status': 'completed',
            'execution_summary': {
                'symbol': result.symbol,
                'strategy': result.strategy.value,
                'total_quantity': result.total_quantity,
                'filled_quantity': result.filled_quantity,
                'fill_rate': f"{result.fill_rate:.1%}",
                'average_price': f"${result.average_price:.2f}",
                'slippage_bps': f"{result.slippage_bps:.1f} bps",
                'execution_score': f"{result.overall_score:.1f}/100",
                'venue_breakdown': {
                    venue.value: data['quantity'] 
                    for venue, data in result.venue_breakdown.items()
                }
            },
            'full_result': result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Test order creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))