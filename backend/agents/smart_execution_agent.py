"""
Smart Execution Agent - Enhanced for GoldenSignalsAI V5
Intelligent order execution with market impact minimization and optimal routing
Migrated from archive with production enhancements and real-time capabilities
"""

import asyncio
import heapq
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from core.logging import get_logger

logger = get_logger(__name__)


class ExecutionStrategy(Enum):
    """Advanced execution strategies"""
    TWAP = "twap"              # Time-weighted average price
    VWAP = "vwap"              # Volume-weighted average price
    POV = "pov"                # Percentage of volume
    IS = "is"                  # Implementation shortfall
    ICEBERG = "iceberg"        # Hidden order strategy
    SNIPER = "sniper"          # Opportunistic execution
    ADAPTIVE = "adaptive"      # ML-driven adaptive
    LIQUIDITY_SEEKING = "liquidity_seeking"  # Hunt for liquidity
    DARK_POOL = "dark_pool"    # Dark pool execution
    SMART_ROUTING = "smart_routing"  # Multi-venue routing
    STEALTH = "stealth"        # Minimal market impact
    AGGRESSIVE = "aggressive"   # Fast execution priority


class OrderType(Enum):
    """Order types supported"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    PEG = "peg"                # Pegged to market
    HIDDEN = "hidden"          # Hidden/reserve orders
    MIDPOINT = "midpoint"      # Midpoint execution
    IOC = "ioc"               # Immediate or cancel
    FOK = "fok"               # Fill or kill
    TWAP_SLICE = "twap_slice"  # TWAP slice order


class Venue(Enum):
    """Execution venues"""
    PRIMARY = "primary"        # Primary exchange (NYSE, NASDAQ)
    DARK_POOL = "dark_pool"   # Dark pools (CrossFinder, Liquidnet)
    ECN = "ecn"               # Electronic communication networks
    ATS = "ats"               # Alternative trading systems
    DIRECT = "direct"         # Direct market access
    RETAIL = "retail"         # Retail market makers
    INSTITUTIONAL = "institutional"  # Institutional networks


class ExecutionStatus(Enum):
    """Order status tracking"""
    PENDING = "pending"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class MarketConditions:
    """Real-time market microstructure data"""
    symbol: str
    timestamp: datetime
    
    # Level 1 data
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    
    # Derived metrics
    mid: float = 0.0
    spread_bps: float = 0.0
    avg_volume: int = 0
    volatility: float = 0.0
    liquidity_score: float = 0.0  # 0-1 scale
    momentum: float = 0.0         # -1 to 1
    market_impact: float = 0.0    # Expected impact in bps
    
    # Market structure
    venue_liquidity: Dict[Venue, float] = field(default_factory=dict)
    order_book_depth: int = 0
    imbalance_ratio: float = 0.0  # (bid_size - ask_size) / (bid_size + ask_size)
    
    def __post_init__(self):
        if self.mid == 0.0:
            self.mid = (self.bid + self.ask) / 2
        if self.spread_bps == 0.0:
            self.spread_bps = ((self.ask - self.bid) / self.mid) * 10000


@dataclass
class ExecutionOrder:
    """Smart execution order with advanced parameters"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: int
    strategy: ExecutionStrategy
    
    # Pricing
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    price_improvement_bps: float = 0.0
    max_slippage_bps: float = 50.0
    
    # Timing
    time_constraint: Optional[int] = None  # Minutes
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    urgency: str = "normal"  # low, normal, high, immediate
    
    # Execution parameters
    min_fill_size: int = 100
    max_display_size: Optional[int] = None
    participate_rate: float = 0.1  # Max % of volume
    iceberg_refresh_rate: float = 0.8  # Refresh when 80% filled
    
    # Venue routing
    venue_preferences: List[Venue] = field(default_factory=list)
    exclude_venues: List[Venue] = field(default_factory=list)
    dark_pool_preference: float = 0.3  # 30% preference for dark pools
    
    # Risk controls
    max_position_pct: float = 0.05  # 5% of daily volume
    cancel_on_news: bool = True
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_buy(self) -> bool:
        return self.side.lower() == 'buy'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionSlice:
    """Individual execution slice with advanced features"""
    slice_id: str
    parent_order_id: str
    quantity: int
    order_type: OrderType
    venue: Venue
    
    # Pricing
    limit_price: Optional[float] = None
    expected_price: float = 0.0
    price_tolerance_bps: float = 10.0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    expire_time: Optional[datetime] = None
    working_duration: timedelta = timedelta(minutes=5)
    
    # Display logic
    min_fill: int = 100
    max_show: Optional[int] = None
    reserve_quantity: int = 0
    
    # Execution priority and logic
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    
    # Status tracking
    status: ExecutionStatus = ExecutionStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    
    # Performance metrics
    creation_timestamp: datetime = field(default_factory=datetime.now)
    first_fill_timestamp: Optional[datetime] = None
    completion_timestamp: Optional[datetime] = None
    
    def __lt__(self, other):
        return self.priority < other.priority
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    
    @property
    def is_complete(self) -> bool:
        return self.status in [ExecutionStatus.FILLED, ExecutionStatus.CANCELLED, 
                              ExecutionStatus.REJECTED, ExecutionStatus.EXPIRED]


@dataclass
class ExecutionResult:
    """Comprehensive execution result analysis"""
    order_id: str
    symbol: str
    strategy: ExecutionStrategy
    
    # Execution summary
    total_quantity: int
    filled_quantity: int
    remaining_quantity: int
    fill_rate: float
    
    # Pricing analysis
    average_price: float
    vwap: float
    market_vwap: float
    arrival_price: float
    slippage_bps: float
    market_impact_bps: float
    price_improvement_bps: float
    
    # Timing analysis
    execution_time_seconds: float
    first_fill_latency_ms: float
    completion_latency_ms: float
    time_to_complete_pct: float
    
    # Venue analysis
    venue_breakdown: Dict[Venue, Dict[str, Any]]
    dark_pool_fill_rate: float
    lit_market_fill_rate: float
    
    # Cost analysis
    commission_cost: float
    market_impact_cost: float
    opportunity_cost: float
    total_execution_cost: float
    
    # Performance metrics
    implementation_shortfall: float
    arrival_cost: float
    timing_cost: float
    market_risk_cost: float
    
    # Execution quality scores
    speed_score: float      # 0-100
    cost_score: float       # 0-100
    stealth_score: float    # 0-100
    overall_score: float    # 0-100
    
    # Detailed analytics
    slice_performance: List[Dict[str, Any]]
    market_conditions_during: List[MarketConditions]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketImpactModel:
    """Advanced market impact modeling"""
    
    def __init__(self):
        self.permanent_impact_factor = 0.1
        self.temporary_impact_half_life = 300  # 5 minutes
        self.participation_impact_exp = 0.6
        self.volatility_scaling = 1.5
        self.liquidity_scaling = 0.8
        
        # Historical calibration data (would be loaded from database)
        self.symbol_parameters = {}
    
    def estimate_market_impact(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        venue: Venue = Venue.PRIMARY
    ) -> Dict[str, float]:
        """Estimate comprehensive market impact"""
        
        # Size impact (Almgren-Chriss model)
        size_ratio = order.total_quantity / max(market.avg_volume, 1)
        size_impact = 10 * np.sqrt(size_ratio * 100)  # bps
        
        # Participation rate impact (non-linear)
        participation_impact = 50 * (order.participate_rate ** self.participation_impact_exp)
        
        # Volatility scaling
        vol_norm = market.volatility / 0.20  # Normalize to 20% annualized
        volatility_impact = size_impact * vol_norm * self.volatility_scaling
        
        # Liquidity impact
        liquidity_penalty = (1 - market.liquidity_score) ** 2 * 20
        
        # Venue impact
        venue_multipliers = {
            Venue.PRIMARY: 1.0,
            Venue.DARK_POOL: 0.3,
            Venue.ECN: 0.8,
            Venue.ATS: 0.6,
            Venue.RETAIL: 1.2
        }
        venue_mult = venue_multipliers.get(venue, 1.0)
        
        # Urgency impact
        urgency_multipliers = {
            'low': 0.7,
            'normal': 1.0,
            'high': 1.4,
            'immediate': 2.2
        }
        urgency_mult = urgency_multipliers.get(order.urgency, 1.0)
        
        # Calculate components
        temporary_impact = (size_impact + participation_impact + 
                          volatility_impact + liquidity_penalty) * venue_mult * urgency_mult
        permanent_impact = temporary_impact * self.permanent_impact_factor
        
        return {
            'temporary_impact_bps': temporary_impact,
            'permanent_impact_bps': permanent_impact,
            'total_impact_bps': temporary_impact + permanent_impact,
            'size_component': size_impact,
            'participation_component': participation_impact,
            'volatility_component': volatility_impact,
            'liquidity_component': liquidity_penalty,
            'venue_multiplier': venue_mult,
            'urgency_multiplier': urgency_mult
        }
    
    def estimate_timing_risk(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        execution_horizon_minutes: int
    ) -> float:
        """Estimate timing risk cost in bps"""
        
        # Volatility-based timing cost
        time_factor = np.sqrt(execution_horizon_minutes / (24 * 60))  # Scale to daily vol
        timing_cost = market.volatility * 100 * time_factor * 0.5  # 50% of vol risk
        
        # Momentum adjustment
        if abs(market.momentum) > 0.5:
            momentum_penalty = abs(market.momentum) * 20  # Additional cost for trending
            timing_cost += momentum_penalty
        
        return timing_cost


class VenueRouter:
    """Smart order routing across venues"""
    
    def __init__(self):
        self.venue_scores = {}
        self.routing_history = deque(maxlen=1000)
        self.venue_latencies = {
            Venue.PRIMARY: 1.2,      # ms
            Venue.DARK_POOL: 2.8,
            Venue.ECN: 0.8,
            Venue.ATS: 1.5,
            Venue.DIRECT: 0.5
        }
    
    def score_venues(
        self,
        order: ExecutionOrder,
        market: MarketConditions
    ) -> Dict[Venue, float]:
        """Score venues for order routing"""
        
        scores = {}
        
        for venue in Venue:
            score = 0.0
            
            # Liquidity score
            venue_liquidity = market.venue_liquidity.get(venue, 0.5)
            score += venue_liquidity * 30
            
            # Latency score (inverse - lower is better)
            latency = self.venue_latencies.get(venue, 2.0)
            score += (3.0 - latency) * 10
            
            # Cost score (based on expected fees and impact)
            if venue == Venue.DARK_POOL:
                score += 25  # Dark pools typically have lower impact
            elif venue == Venue.PRIMARY:
                score += 15  # Reliable but higher impact
            elif venue == Venue.ECN:
                score += 20  # Good balance
            
            # Size appropriateness
            if order.total_quantity > 10000 and venue == Venue.DARK_POOL:
                score += 15  # Large orders benefit from dark pools
            elif order.total_quantity < 1000 and venue == Venue.RETAIL:
                score += 10  # Small orders good for retail
            
            # Urgency adjustment
            if order.urgency == 'immediate' and venue in [Venue.PRIMARY, Venue.ECN]:
                score += 20
            elif order.urgency == 'low' and venue == Venue.DARK_POOL:
                score += 15
            
            # Venue preferences and exclusions
            if venue in order.venue_preferences:
                score += 25
            if venue in order.exclude_venues:
                score = 0
            
            scores[venue] = score
        
        return scores
    
    def route_order(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        slice_size: int
    ) -> List[Tuple[Venue, int]]:
        """Route order across multiple venues"""
        
        venue_scores = self.score_venues(order, market)
        
        # Sort venues by score
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate quantity across top venues
        allocations = []
        remaining = slice_size
        
        # Primary allocation to top venue
        if sorted_venues:
            primary_venue, primary_score = sorted_venues[0]
            primary_allocation = min(remaining, int(slice_size * 0.6))
            if primary_allocation > 0:
                allocations.append((primary_venue, primary_allocation))
                remaining -= primary_allocation
        
        # Secondary allocations
        for venue, score in sorted_venues[1:]:
            if remaining <= 0:
                break
            
            if score > 40:  # Only use high-scoring venues
                allocation = min(remaining, int(slice_size * 0.3))
                if allocation > 0:
                    allocations.append((venue, allocation))
                    remaining -= allocation
        
        # Put any remainder in primary venue
        if remaining > 0 and allocations:
            allocations[0] = (allocations[0][0], allocations[0][1] + remaining)
        
        return allocations


class SmartExecutionAgent:
    """
    Advanced Smart Execution Agent for GoldenSignalsAI V5
    
    Features:
    - Multiple execution strategies (TWAP, VWAP, POV, IS, etc.)
    - Intelligent venue routing
    - Real-time market impact modeling
    - Adaptive execution parameters
    - Comprehensive performance analytics
    - Risk controls and monitoring
    """
    
    def __init__(
        self,
        name: str = "SmartExecutionAgent",
        commission_rate: float = 0.001,  # 10 bps
        max_position_size: int = 1000000,
        enable_dark_pools: bool = True
    ):
        """
        Initialize Smart Execution Agent
        
        Args:
            name: Agent name
            commission_rate: Commission rate (as decimal)
            max_position_size: Maximum position size
            enable_dark_pools: Whether to use dark pools
        """
        self.name = name
        self.commission_rate = commission_rate
        self.max_position_size = max_position_size
        self.enable_dark_pools = enable_dark_pools
        
        # Core components
        self.impact_model = MarketImpactModel()
        self.venue_router = VenueRouter()
        
        # State tracking
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.active_slices: Dict[str, ExecutionSlice] = {}
        self.execution_queue = []  # Priority queue for slices
        self.execution_history: List[ExecutionResult] = []
        
        # Performance metrics
        self.total_executed_value = 0.0
        self.total_execution_cost = 0.0
        self.average_fill_rate = 0.0
        self.average_slippage = 0.0
        
        # Strategy handlers
        self.strategy_handlers = {
            ExecutionStrategy.TWAP: self._execute_twap,
            ExecutionStrategy.VWAP: self._execute_vwap,
            ExecutionStrategy.POV: self._execute_pov,
            ExecutionStrategy.IS: self._execute_implementation_shortfall,
            ExecutionStrategy.ICEBERG: self._execute_iceberg,
            ExecutionStrategy.SNIPER: self._execute_sniper,
            ExecutionStrategy.ADAPTIVE: self._execute_adaptive,
            ExecutionStrategy.LIQUIDITY_SEEKING: self._execute_liquidity_seeking,
            ExecutionStrategy.DARK_POOL: self._execute_dark_pool,
            ExecutionStrategy.SMART_ROUTING: self._execute_smart_routing
        }
        
        logger.info(f"Initialized {name} with {len(self.strategy_handlers)} execution strategies")
    
    async def execute_order(
        self,
        order: ExecutionOrder,
        market_conditions: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """
        Execute order using specified strategy
        
        Args:
            order: Order to execute
            market_conditions: Current market conditions
            callback: Optional callback for updates
        
        Returns:
            Execution result with comprehensive analytics
        """
        try:
            logger.info(f"Executing {order.strategy.value} order: {order.symbol} {order.side} {order.total_quantity}")
            
            # Validate order
            self._validate_order(order, market_conditions)
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Get strategy handler
            handler = self.strategy_handlers.get(order.strategy)
            if not handler:
                raise ValueError(f"Unsupported execution strategy: {order.strategy}")
            
            # Execute using strategy
            start_time = datetime.now()
            result = await handler(order, market_conditions, callback)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update result with timing
            result.execution_time_seconds = execution_time
            
            # Calculate comprehensive metrics
            self._calculate_execution_metrics(result, order, market_conditions)
            
            # Store in history
            self.execution_history.append(result)
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Remove from active orders
            self.active_orders.pop(order.order_id, None)
            
            logger.info(
                f"Execution complete: {result.fill_rate:.1%} filled, "
                f"{result.slippage_bps:.1f} bps slippage, "
                f"score: {result.overall_score:.1f}/100"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed for order {order.order_id}: {str(e)}")
            # Create failed result
            return self._create_failed_result(order, str(e))
    
    def _validate_order(self, order: ExecutionOrder, market: MarketConditions):
        """Validate order parameters"""
        if order.total_quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if order.total_quantity > self.max_position_size:
            raise ValueError(f"Order size {order.total_quantity} exceeds max {self.max_position_size}")
        
        if order.participate_rate > 1.0:
            raise ValueError("Participation rate cannot exceed 100%")
        
        if order.limit_price and order.limit_price <= 0:
            raise ValueError("Limit price must be positive")
        
        # Market-specific validations
        if market.spread_bps > 500:  # 5% spread
            logger.warning(f"Wide spread detected: {market.spread_bps:.1f} bps")
        
        if market.liquidity_score < 0.3:
            logger.warning(f"Low liquidity detected: {market.liquidity_score:.2f}")
    
    async def _execute_twap(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Time-Weighted Average Price strategy"""
        
        # Calculate time slicing
        execution_horizon = order.time_constraint or 60  # Default 1 hour
        slice_duration = max(1, execution_horizon // 20)  # 20 slices max
        num_slices = max(1, execution_horizon // slice_duration)
        slice_size = order.total_quantity // num_slices
        
        logger.info(f"TWAP: {num_slices} slices of {slice_size} over {execution_horizon} minutes")
        
        # Create execution slices
        slices = []
        remaining_qty = order.total_quantity
        
        for i in range(num_slices):
            current_slice_size = min(slice_size, remaining_qty)
            if current_slice_size <= 0:
                break
            
            # Route slice across venues
            venue_allocations = self.venue_router.route_order(order, market, current_slice_size)
            
            for venue, allocation in venue_allocations:
                slice_obj = ExecutionSlice(
                    slice_id=f"{order.order_id}_twap_{i}_{venue.value}",
                    parent_order_id=order.order_id,
                    quantity=allocation,
                    order_type=OrderType.LIMIT,
                    venue=venue,
                    limit_price=self._calculate_limit_price(order, market, venue),
                    start_time=datetime.now() + timedelta(minutes=i * slice_duration),
                    expire_time=datetime.now() + timedelta(minutes=(i + 1) * slice_duration),
                    min_fill=min(allocation, order.min_fill_size),
                    priority=i
                )
                slices.append(slice_obj)
            
            remaining_qty -= current_slice_size
        
        # Execute slices sequentially with timing
        return await self._execute_slices(slices, order, market, callback)
    
    async def _execute_vwap(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Volume-Weighted Average Price strategy"""
        
        # Use historical volume profile (simplified)
        volume_profile = self._get_volume_profile(market.symbol)
        
        # Calculate volume-based slicing
        total_expected_volume = sum(volume_profile)
        participation_volume = total_expected_volume * order.participate_rate
        
        if participation_volume < order.total_quantity:
            logger.warning(f"Order size {order.total_quantity} exceeds participation volume {participation_volume}")
        
        # Create slices based on volume profile
        slices = []
        remaining_qty = order.total_quantity
        
        for i, expected_volume in enumerate(volume_profile):
            if remaining_qty <= 0:
                break
            
            # Size slice based on expected volume
            volume_ratio = expected_volume / total_expected_volume
            slice_size = min(int(order.total_quantity * volume_ratio), remaining_qty)
            
            if slice_size > 0:
                slice_obj = ExecutionSlice(
                    slice_id=f"{order.order_id}_vwap_{i}",
                    parent_order_id=order.order_id,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    venue=Venue.PRIMARY,  # Simplified - would use smart routing
                    limit_price=self._calculate_limit_price(order, market, Venue.PRIMARY),
                    start_time=datetime.now() + timedelta(minutes=i * 15),  # 15-min intervals
                    expire_time=datetime.now() + timedelta(minutes=(i + 1) * 15),
                    priority=i
                )
                slices.append(slice_obj)
                remaining_qty -= slice_size
        
        return await self._execute_slices(slices, order, market, callback)
    
    async def _execute_pov(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Percentage of Volume strategy"""
        
        # Monitor market volume and adjust participation
        target_participation = order.participate_rate
        slice_interval = 5  # 5-minute intervals
        
        slices = []
        remaining_qty = order.total_quantity
        slice_num = 0
        
        while remaining_qty > 0:
            # Estimate volume for next interval
            expected_interval_volume = market.avg_volume * (slice_interval / (24 * 60))
            target_slice_size = int(expected_interval_volume * target_participation)
            actual_slice_size = min(target_slice_size, remaining_qty)
            
            if actual_slice_size > 0:
                slice_obj = ExecutionSlice(
                    slice_id=f"{order.order_id}_pov_{slice_num}",
                    parent_order_id=order.order_id,
                    quantity=actual_slice_size,
                    order_type=OrderType.LIMIT,
                    venue=Venue.PRIMARY,
                    limit_price=self._calculate_limit_price(order, market, Venue.PRIMARY),
                    start_time=datetime.now() + timedelta(minutes=slice_num * slice_interval),
                    expire_time=datetime.now() + timedelta(minutes=(slice_num + 1) * slice_interval),
                    priority=slice_num
                )
                slices.append(slice_obj)
                remaining_qty -= actual_slice_size
            
            slice_num += 1
            
            # Safety check - max 24 hours
            if slice_num > 288:  # 24 hours / 5 minutes
                break
        
        return await self._execute_slices(slices, order, market, callback)
    
    async def _execute_implementation_shortfall(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Implementation Shortfall strategy (Almgren-Chriss)"""
        
        # Calculate optimal execution trajectory
        impact_params = self.impact_model.estimate_market_impact(order, market)
        timing_risk = self.impact_model.estimate_timing_risk(
            order, market, order.time_constraint or 60
        )
        
        # Optimize trade-off between market impact and timing risk
        optimal_rate = self._calculate_optimal_trading_rate(
            order, market, impact_params, timing_risk
        )
        
        # Create execution schedule
        horizon_minutes = order.time_constraint or 60
        num_slices = min(20, horizon_minutes // 3)  # Max 20 slices, min 3 minutes each
        
        slices = []
        remaining_qty = order.total_quantity
        
        for i in range(num_slices):
            # Exponential decay in slice sizes (front-loaded)
            decay_factor = np.exp(-optimal_rate * i / num_slices)
            slice_ratio = decay_factor / sum(np.exp(-optimal_rate * j / num_slices) for j in range(num_slices))
            slice_size = min(int(order.total_quantity * slice_ratio), remaining_qty)
            
            if slice_size > 0:
                slice_obj = ExecutionSlice(
                    slice_id=f"{order.order_id}_is_{i}",
                    parent_order_id=order.order_id,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    venue=Venue.PRIMARY,
                    limit_price=self._calculate_limit_price(order, market, Venue.PRIMARY),
                    start_time=datetime.now() + timedelta(minutes=i * (horizon_minutes // num_slices)),
                    priority=i
                )
                slices.append(slice_obj)
                remaining_qty -= slice_size
        
        return await self._execute_slices(slices, order, market, callback)
    
    async def _execute_iceberg(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Iceberg strategy with hidden quantity"""
        
        # Iceberg parameters
        display_size = order.max_display_size or min(1000, order.total_quantity // 10)
        refresh_threshold = int(display_size * order.iceberg_refresh_rate)
        
        slices = []
        remaining_qty = order.total_quantity
        slice_num = 0
        
        while remaining_qty > 0:
            current_display = min(display_size, remaining_qty)
            reserve_qty = remaining_qty - current_display
            
            slice_obj = ExecutionSlice(
                slice_id=f"{order.order_id}_iceberg_{slice_num}",
                parent_order_id=order.order_id,
                quantity=current_display,
                order_type=OrderType.HIDDEN,
                venue=Venue.DARK_POOL if self.enable_dark_pools else Venue.PRIMARY,
                limit_price=self._calculate_limit_price(order, market, Venue.DARK_POOL),
                max_show=current_display,
                reserve_quantity=reserve_qty,
                priority=slice_num
            )
            slices.append(slice_obj)
            
            # Next slice will be triggered when current slice is mostly filled
            remaining_qty -= current_display
            slice_num += 1
            
            # Safety limit
            if slice_num > 100:
                break
        
        return await self._execute_slices(slices, order, market, callback)
    
    async def _execute_sniper(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Sniper strategy - opportunistic execution"""
        
        # Look for liquidity opportunities
        # This is a simplified version - production would use real-time order book analysis
        
        slices = []
        
        # Single aggressive slice for immediate execution
        slice_obj = ExecutionSlice(
            slice_id=f"{order.order_id}_sniper_0",
            parent_order_id=order.order_id,
            quantity=order.total_quantity,
            order_type=OrderType.IOC,  # Immediate or Cancel
            venue=Venue.ECN,  # Use ECN for fast execution
            limit_price=market.ask if order.is_buy else market.bid,  # Cross the spread
            start_time=datetime.now(),
            expire_time=datetime.now() + timedelta(seconds=30),  # Very short expiry
            priority=0
        )
        slices.append(slice_obj)
        
        return await self._execute_slices(slices, order, market, callback)
    
    async def _execute_adaptive(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Adaptive strategy with ML-driven adjustments"""
        
        # Analyze market conditions and adapt strategy
        if market.volatility > 0.3:  # High volatility
            # Use more aggressive execution
            return await self._execute_sniper(order, market, callback)
        elif market.liquidity_score < 0.4:  # Low liquidity
            # Use patient TWAP
            return await self._execute_twap(order, market, callback)
        elif order.total_quantity > market.avg_volume * 0.1:  # Large order
            # Use iceberg to hide size
            return await self._execute_iceberg(order, market, callback)
        else:
            # Use VWAP for normal conditions
            return await self._execute_vwap(order, market, callback)
    
    async def _execute_liquidity_seeking(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute Liquidity Seeking strategy"""
        
        # Hunt for liquidity across venues
        venue_scores = self.venue_router.score_venues(order, market)
        
        slices = []
        remaining_qty = order.total_quantity
        
        # Create slices for top-scoring venues
        for venue, score in sorted(venue_scores.items(), key=lambda x: x[1], reverse=True):
            if score < 30 or remaining_qty <= 0:  # Only use good venues
                continue
            
            # Allocate based on venue score
            allocation_pct = min(0.4, score / 100)  # Max 40% per venue
            slice_size = min(int(order.total_quantity * allocation_pct), remaining_qty)
            
            if slice_size > 0:
                slice_obj = ExecutionSlice(
                    slice_id=f"{order.order_id}_liq_{venue.value}",
                    parent_order_id=order.order_id,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    venue=venue,
                    limit_price=self._calculate_limit_price(order, market, venue),
                    priority=int(100 - score)  # Higher score = higher priority
                )
                slices.append(slice_obj)
                remaining_qty -= slice_size
        
        return await self._execute_slices(slices, order, market, callback)
    
    async def _execute_dark_pool(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute primarily in dark pools"""
        
        if not self.enable_dark_pools:
            logger.warning("Dark pools disabled, falling back to lit markets")
            return await self._execute_twap(order, market, callback)
        
        # Single large dark pool order
        slice_obj = ExecutionSlice(
            slice_id=f"{order.order_id}_dark_0",
            parent_order_id=order.order_id,
            quantity=order.total_quantity,
            order_type=OrderType.HIDDEN,
            venue=Venue.DARK_POOL,
            limit_price=market.mid,  # Mid-point execution
            start_time=datetime.now(),
            expire_time=datetime.now() + timedelta(hours=4),  # Long expiry
            priority=0
        )
        
        return await self._execute_slices([slice_obj], order, market, callback)
    
    async def _execute_smart_routing(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None
    ) -> ExecutionResult:
        """Execute with intelligent multi-venue routing"""
        
        # Route across multiple venues simultaneously
        venue_allocations = self.venue_router.route_order(order, market, order.total_quantity)
        
        slices = []
        for i, (venue, allocation) in enumerate(venue_allocations):
            slice_obj = ExecutionSlice(
                slice_id=f"{order.order_id}_route_{venue.value}",
                parent_order_id=order.order_id,
                quantity=allocation,
                order_type=OrderType.LIMIT,
                venue=venue,
                limit_price=self._calculate_limit_price(order, market, venue),
                start_time=datetime.now(),
                priority=i
            )
            slices.append(slice_obj)
        
        return await self._execute_slices(slices, order, market, callback, parallel=True)
    
    async def _execute_slices(
        self,
        slices: List[ExecutionSlice],
        order: ExecutionOrder,
        market: MarketConditions,
        callback: Optional[Callable] = None,
        parallel: bool = False
    ) -> ExecutionResult:
        """Execute list of slices"""
        
        # Add slices to active tracking
        for slice_obj in slices:
            self.active_slices[slice_obj.slice_id] = slice_obj
        
        # Execute slices
        filled_slices = []
        
        if parallel:
            # Execute all slices simultaneously
            tasks = [self._execute_single_slice(slice_obj, market) for slice_obj in slices]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    filled_slices.append(result)
                else:
                    logger.error(f"Slice execution failed: {result}")
        else:
            # Execute slices sequentially with timing
            for slice_obj in slices:
                # Wait until slice start time
                if slice_obj.start_time > datetime.now():
                    wait_seconds = (slice_obj.start_time - datetime.now()).total_seconds()
                    await asyncio.sleep(min(wait_seconds, 300))  # Max 5 minute wait
                
                try:
                    result = await self._execute_single_slice(slice_obj, market)
                    filled_slices.append(result)
                    
                    # Callback for progress updates
                    if callback:
                        await callback(result)
                        
                except Exception as e:
                    logger.error(f"Slice execution failed: {e}")
        
        # Create comprehensive result
        return self._create_execution_result(order, filled_slices, market)
    
    async def _execute_single_slice(
        self,
        slice_obj: ExecutionSlice,
        market: MarketConditions
    ) -> ExecutionSlice:
        """Simulate execution of a single slice"""
        
        # This is a simulation - in production, this would interface with brokers/exchanges
        slice_obj.status = ExecutionStatus.WORKING
        
        # Simulate market dynamics and fill probability
        fill_probability = self._calculate_fill_probability(slice_obj, market)
        
        # Simulate partial or full fill
        if np.random.random() < fill_probability:
            if slice_obj.order_type == OrderType.IOC and np.random.random() < 0.8:
                # IOC orders have higher fill rate but might be partial
                fill_ratio = np.random.uniform(0.7, 1.0)
            else:
                fill_ratio = np.random.uniform(0.3, 1.0)
            
            filled_qty = int(slice_obj.quantity * fill_ratio)
            fill_price = self._simulate_fill_price(slice_obj, market)
            
            # Update slice
            slice_obj.filled_quantity = filled_qty
            slice_obj.average_fill_price = fill_price
            slice_obj.status = ExecutionStatus.FILLED if filled_qty == slice_obj.quantity else ExecutionStatus.PARTIALLY_FILLED
            slice_obj.first_fill_timestamp = datetime.now()
            
            if slice_obj.status == ExecutionStatus.FILLED:
                slice_obj.completion_timestamp = datetime.now()
        else:
            # No fill
            slice_obj.status = ExecutionStatus.CANCELLED
        
        # Remove from active tracking
        self.active_slices.pop(slice_obj.slice_id, None)
        
        return slice_obj
    
    def _calculate_fill_probability(self, slice_obj: ExecutionSlice, market: MarketConditions) -> float:
        """Calculate probability of slice getting filled"""
        
        base_probability = 0.7
        
        # Venue adjustments
        if slice_obj.venue == Venue.DARK_POOL:
            base_probability *= 0.6  # Lower fill rate in dark pools
        elif slice_obj.venue == Venue.PRIMARY:
            base_probability *= 1.0
        elif slice_obj.venue == Venue.ECN:
            base_probability *= 0.9
        
        # Order type adjustments
        if slice_obj.order_type == OrderType.MARKET:
            base_probability = 0.98
        elif slice_obj.order_type == OrderType.IOC:
            base_probability *= 0.8
        elif slice_obj.order_type == OrderType.HIDDEN:
            base_probability *= 0.5
        
        # Market conditions adjustments
        if market.volatility > 0.3:
            base_probability *= 0.8  # Lower fill rate in volatile markets
        if market.liquidity_score < 0.4:
            base_probability *= 0.7  # Lower fill rate in illiquid markets
        
        return min(1.0, max(0.1, base_probability))
    
    def _simulate_fill_price(self, slice_obj: ExecutionSlice, market: MarketConditions) -> float:
        """Simulate fill price based on order type and market conditions"""
        
        if slice_obj.order_type == OrderType.MARKET:
            # Market orders get current ask/bid with some slippage
            base_price = market.ask if slice_obj.quantity > 0 else market.bid
            slippage = np.random.normal(0, market.spread_bps / 20000)  # Small random slippage
            return base_price * (1 + slippage)
        
        elif slice_obj.limit_price:
            # Limit orders get filled at or better than limit price
            if slice_obj.quantity > 0:  # Buy order
                return min(slice_obj.limit_price, market.ask)
            else:  # Sell order
                return max(slice_obj.limit_price, market.bid)
        
        else:
            # Default to midpoint with small random variation
            noise = np.random.normal(0, market.spread_bps / 40000)
            return market.mid * (1 + noise)
    
    def _calculate_limit_price(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        venue: Venue
    ) -> float:
        """Calculate appropriate limit price for venue"""
        
        if order.limit_price:
            return order.limit_price
        
        base_price = market.mid
        
        # Venue-specific pricing
        if venue == Venue.DARK_POOL:
            return base_price  # Mid-point execution
        elif venue == Venue.PRIMARY:
            # Slightly inside the spread
            improvement = market.spread_bps * 0.3 / 10000
            if order.is_buy:
                return base_price - (base_price * improvement)
            else:
                return base_price + (base_price * improvement)
        else:
            # Standard limit price at mid
            return base_price
    
    def _get_volume_profile(self, symbol: str) -> List[float]:
        """Get historical volume profile (simplified)"""
        # In production, this would query historical data
        # Return normalized hourly volume profile
        return [
            0.02, 0.03, 0.04, 0.06, 0.08, 0.12,  # 6 AM - 12 PM
            0.15, 0.18, 0.16, 0.14, 0.12, 0.10   # 12 PM - 6 PM
        ]
    
    def _calculate_optimal_trading_rate(
        self,
        order: ExecutionOrder,
        market: MarketConditions,
        impact_params: Dict[str, float],
        timing_risk: float
    ) -> float:
        """Calculate optimal trading rate for IS strategy"""
        
        # Simplified Almgren-Chriss optimization
        market_impact = impact_params['total_impact_bps']
        
        # Trade-off parameter (risk aversion)
        risk_aversion = 1e-6
        
        # Optimal rate balances market impact vs. timing risk
        optimal_rate = np.sqrt(2 * risk_aversion * timing_risk) / market_impact
        
        return np.clip(optimal_rate, 0.1, 5.0)  # Reasonable bounds
    
    def _create_execution_result(
        self,
        order: ExecutionOrder,
        filled_slices: List[ExecutionSlice],
        market: MarketConditions
    ) -> ExecutionResult:
        """Create comprehensive execution result"""
        
        # Calculate basic metrics
        total_filled = sum(slice_obj.filled_quantity for slice_obj in filled_slices)
        total_value = sum(slice_obj.filled_quantity * slice_obj.average_fill_price 
                         for slice_obj in filled_slices if slice_obj.filled_quantity > 0)
        
        avg_price = total_value / total_filled if total_filled > 0 else 0.0
        fill_rate = total_filled / order.total_quantity
        
        # Calculate slippage
        benchmark_price = market.mid
        if order.is_buy:
            slippage_bps = ((avg_price - benchmark_price) / benchmark_price) * 10000
        else:
            slippage_bps = ((benchmark_price - avg_price) / benchmark_price) * 10000
        
        # Venue breakdown
        venue_breakdown = {}
        for slice_obj in filled_slices:
            if slice_obj.filled_quantity > 0:
                venue = slice_obj.venue
                if venue not in venue_breakdown:
                    venue_breakdown[venue] = {
                        'quantity': 0,
                        'value': 0.0,
                        'avg_price': 0.0,
                        'slices': 0
                    }
                
                venue_breakdown[venue]['quantity'] += slice_obj.filled_quantity
                venue_breakdown[venue]['value'] += slice_obj.filled_quantity * slice_obj.average_fill_price
                venue_breakdown[venue]['slices'] += 1
        
        # Calculate average prices per venue
        for venue_data in venue_breakdown.values():
            if venue_data['quantity'] > 0:
                venue_data['avg_price'] = venue_data['value'] / venue_data['quantity']
        
        # Calculate costs
        commission_cost = total_value * self.commission_rate
        market_impact_cost = abs(slippage_bps) * total_value / 10000
        
        # Performance scores (0-100)
        speed_score = min(100, (fill_rate * 100))
        cost_score = max(0, 100 - abs(slippage_bps) / 2)  # Penalize high slippage
        stealth_score = 100 if any(s.venue == Venue.DARK_POOL for s in filled_slices) else 50
        overall_score = (speed_score * 0.4 + cost_score * 0.4 + stealth_score * 0.2)
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            strategy=order.strategy,
            total_quantity=order.total_quantity,
            filled_quantity=total_filled,
            remaining_quantity=order.total_quantity - total_filled,
            fill_rate=fill_rate,
            average_price=avg_price,
            vwap=avg_price,  # Simplified
            market_vwap=benchmark_price,
            arrival_price=benchmark_price,
            slippage_bps=slippage_bps,
            market_impact_bps=slippage_bps,
            price_improvement_bps=0.0,
            execution_time_seconds=0.0,  # Will be set by caller
            first_fill_latency_ms=50.0,
            completion_latency_ms=200.0,
            time_to_complete_pct=100.0,
            venue_breakdown=venue_breakdown,
            dark_pool_fill_rate=sum(v['quantity'] for k, v in venue_breakdown.items() 
                                   if k == Venue.DARK_POOL) / total_filled if total_filled > 0 else 0.0,
            lit_market_fill_rate=1.0 - (sum(v['quantity'] for k, v in venue_breakdown.items() 
                                           if k == Venue.DARK_POOL) / total_filled if total_filled > 0 else 0.0),
            commission_cost=commission_cost,
            market_impact_cost=market_impact_cost,
            opportunity_cost=0.0,
            total_execution_cost=commission_cost + market_impact_cost,
            implementation_shortfall=slippage_bps,
            arrival_cost=slippage_bps,
            timing_cost=0.0,
            market_risk_cost=0.0,
            speed_score=speed_score,
            cost_score=cost_score,
            stealth_score=stealth_score,
            overall_score=overall_score,
            slice_performance=[asdict(s) for s in filled_slices],
            market_conditions_during=[market]
        )
    
    def _calculate_execution_metrics(
        self,
        result: ExecutionResult,
        order: ExecutionOrder,
        market: MarketConditions
    ):
        """Calculate comprehensive execution metrics"""
        
        # Implementation Shortfall components
        arrival_cost = result.slippage_bps
        timing_cost = 0.0  # Would calculate based on price movement during execution
        market_risk_cost = market.volatility * 100 * 0.1  # Simplified
        
        result.arrival_cost = arrival_cost
        result.timing_cost = timing_cost
        result.market_risk_cost = market_risk_cost
        result.implementation_shortfall = arrival_cost + timing_cost + market_risk_cost
        
        # Opportunity cost for unfilled quantity
        unfilled_ratio = 1.0 - result.fill_rate
        result.opportunity_cost = unfilled_ratio * market.volatility * 100 * 0.5
        
        # Update total cost
        result.total_execution_cost += result.opportunity_cost
    
    def _create_failed_result(self, order: ExecutionOrder, error: str) -> ExecutionResult:
        """Create result for failed execution"""
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            strategy=order.strategy,
            total_quantity=order.total_quantity,
            filled_quantity=0,
            remaining_quantity=order.total_quantity,
            fill_rate=0.0,
            average_price=0.0,
            vwap=0.0,
            market_vwap=0.0,
            arrival_price=0.0,
            slippage_bps=0.0,
            market_impact_bps=0.0,
            price_improvement_bps=0.0,
            execution_time_seconds=0.0,
            first_fill_latency_ms=0.0,
            completion_latency_ms=0.0,
            time_to_complete_pct=0.0,
            venue_breakdown={},
            dark_pool_fill_rate=0.0,
            lit_market_fill_rate=0.0,
            commission_cost=0.0,
            market_impact_cost=0.0,
            opportunity_cost=0.0,
            total_execution_cost=0.0,
            implementation_shortfall=0.0,
            arrival_cost=0.0,
            timing_cost=0.0,
            market_risk_cost=0.0,
            speed_score=0.0,
            cost_score=0.0,
            stealth_score=0.0,
            overall_score=0.0,
            slice_performance=[],
            market_conditions_during=[]
        )
    
    def _update_performance_metrics(self, result: ExecutionResult):
        """Update agent performance metrics"""
        
        self.total_executed_value += result.filled_quantity * result.average_price
        self.total_execution_cost += result.total_execution_cost
        
        # Update running averages
        n = len(self.execution_history)
        if n > 0:
            self.average_fill_rate = ((self.average_fill_rate * (n - 1)) + result.fill_rate) / n
            self.average_slippage = ((self.average_slippage * (n - 1)) + result.slippage_bps) / n
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary"""
        
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        results = self.execution_history
        
        return {
            'total_executions': len(results),
            'total_executed_value': self.total_executed_value,
            'total_execution_cost': self.total_execution_cost,
            'average_fill_rate': self.average_fill_rate,
            'average_slippage_bps': self.average_slippage,
            'average_overall_score': np.mean([r.overall_score for r in results]),
            'strategy_breakdown': {
                strategy.value: len([r for r in results if r.strategy == strategy])
                for strategy in ExecutionStrategy
            },
            'venue_usage': {
                venue.value: sum(
                    sum(vb.get(venue, {}).get('quantity', 0) for vb in [r.venue_breakdown])
                    for r in results
                )
                for venue in Venue
            }
        }


# Global instance
smart_execution_agent = SmartExecutionAgent()