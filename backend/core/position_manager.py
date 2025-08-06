"""
Position Management System
Handles all position tracking, P&L calculation, and portfolio management
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from decimal import Decimal
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from database.models import Position, Portfolio, Trade, User
from database.connection import get_db
from core.events.bus import event_bus, EventTypes
from core.logging import get_logger
from core.market_data.aggregator import market_data_aggregator

logger = get_logger(__name__)


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionStatus(Enum):
    """Position status"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class PositionInfo:
    """Complete position information"""
    position_id: str
    portfolio_id: int
    symbol: str
    position_type: PositionType
    quantity: float
    avg_entry_price: float
    current_price: float
    
    # P&L
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    pnl_percentage: float
    
    # Risk metrics
    position_value: float
    exposure: float
    margin_required: float
    leverage: float
    
    # Timing
    opened_at: datetime
    last_updated: datetime
    holding_period: timedelta
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    max_position_size: Optional[float] = None
    
    # Performance
    max_profit: float = 0
    max_loss: float = 0
    volatility: float = 0
    sharpe_ratio: float = 0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    source_signal_id: Optional[str] = None
    source_agent_id: Optional[str] = None


class PositionRequest(BaseModel):
    """Request to open a position"""
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_percent: Optional[float] = Field(None, ge=0, le=100)
    
    # Position limits
    max_position_size: Optional[float] = None
    max_loss_amount: Optional[float] = None
    
    # Metadata
    signal_id: Optional[str] = None
    agent_id: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class PositionUpdate(BaseModel):
    """Position update request"""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_percent: Optional[float] = None
    add_quantity: Optional[float] = None
    reduce_quantity: Optional[float] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class PortfolioSummary(BaseModel):
    """Portfolio summary"""
    portfolio_id: int
    name: str
    total_value: float
    cash_balance: float
    positions_value: float
    
    # P&L
    total_unrealized_pnl: float
    total_realized_pnl: float
    daily_pnl: float
    total_return_pct: float
    
    # Risk metrics
    total_exposure: float
    margin_used: float
    margin_available: float
    leverage: float
    
    # Position stats
    open_positions: int
    winning_positions: int
    losing_positions: int
    
    # Performance
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    
    # Timing
    created_at: datetime
    last_updated: datetime


class PositionManager:
    """
    Centralized position management system
    """
    
    def __init__(self):
        self.positions_cache: Dict[str, PositionInfo] = {}
        self.portfolios_cache: Dict[int, PortfolioSummary] = {}
        self._price_update_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
        
        # Risk parameters
        self.max_position_size_pct = 0.1  # Max 10% per position
        self.max_leverage = 3.0
        self.margin_requirement = 0.25  # 25% margin
        
        # Performance tracking
        self.position_history: List[PositionInfo] = []
        self.trade_history: List[Trade] = []
        
        logger.info("Position Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize position manager"""
        # Load existing positions from database
        await self._load_positions_from_db()
        
        # Subscribe to events
        await event_bus.subscribe(EventTypes.MARKET_DATA_RECEIVED, self._on_market_data)
        await event_bus.subscribe(EventTypes.TRADE_EXECUTED, self._on_trade_executed)
        
        # Start price monitoring
        await self.start_monitoring()
        
        logger.info("Position Manager initialized with {} positions", len(self.positions_cache))
    
    async def open_position(
        self,
        portfolio_id: int,
        request: PositionRequest
    ) -> PositionInfo:
        """
        Open a new position
        
        Args:
            portfolio_id: Portfolio ID
            request: Position request details
            
        Returns:
            Created position info
        """
        try:
            # Validate request
            await self._validate_position_request(portfolio_id, request)
            
            # Get current market price
            market_data = await market_data_aggregator.get_latest_price(request.symbol)
            current_price = market_data.get("price", 0)
            
            if not current_price:
                raise ValueError(f"Unable to get price for {request.symbol}")
            
            # Determine execution price based on order type
            execution_price = self._calculate_execution_price(
                request.order_type,
                request.side,
                current_price,
                request.limit_price,
                request.stop_price
            )
            
            # Check risk limits
            await self._check_risk_limits(
                portfolio_id,
                request.symbol,
                request.quantity,
                execution_price
            )
            
            # Create position in database
            position_id = str(uuid.uuid4())
            
            async with get_db() as session:
                # Create position record
                position = Position(
                    portfolio_id=portfolio_id,
                    symbol=request.symbol,
                    quantity=request.quantity if request.side == OrderSide.BUY else -request.quantity,
                    avg_entry_price=execution_price,
                    current_price=current_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    position_type=PositionType.LONG.value if request.side == OrderSide.BUY else PositionType.SHORT.value,
                    status=PositionStatus.OPEN.value,
                    opened_at=datetime.now()
                )
                
                session.add(position)
                
                # Create trade record
                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    user_id=1,  # TODO: Get from context
                    portfolio_id=portfolio_id,
                    position_id=position.id,
                    symbol=request.symbol,
                    side=request.side.value,
                    quantity=request.quantity,
                    price=execution_price,
                    total_value=request.quantity * execution_price,
                    order_type=request.order_type.value,
                    status="filled",
                    executed_at=datetime.now(),
                    metadata={
                        "signal_id": request.signal_id,
                        "agent_id": request.agent_id,
                        "notes": request.notes,
                        "tags": request.tags
                    }
                )
                
                session.add(trade)
                
                # Update portfolio cash
                portfolio = await session.get(Portfolio, portfolio_id)
                if portfolio:
                    trade_cost = request.quantity * execution_price
                    if request.side == OrderSide.BUY:
                        portfolio.cash_balance -= trade_cost
                    else:
                        portfolio.cash_balance += trade_cost
                
                await session.commit()
                
                # Refresh to get ID
                await session.refresh(position)
                position_id = str(position.id)
            
            # Create position info
            position_info = PositionInfo(
                position_id=position_id,
                portfolio_id=portfolio_id,
                symbol=request.symbol,
                position_type=PositionType.LONG if request.side == OrderSide.BUY else PositionType.SHORT,
                quantity=request.quantity,
                avg_entry_price=execution_price,
                current_price=current_price,
                unrealized_pnl=0,
                realized_pnl=0,
                total_pnl=0,
                pnl_percentage=0,
                position_value=request.quantity * current_price,
                exposure=request.quantity * current_price,
                margin_required=request.quantity * current_price * self.margin_requirement,
                leverage=1.0,
                opened_at=datetime.now(),
                last_updated=datetime.now(),
                holding_period=timedelta(0),
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                trailing_stop_distance=request.trailing_stop_percent,
                max_position_size=request.max_position_size,
                tags=request.tags,
                notes=request.notes or "",
                source_signal_id=request.signal_id,
                source_agent_id=request.agent_id
            )
            
            # Cache position
            self.positions_cache[position_id] = position_info
            
            # Publish event
            await event_bus.publish(
                EventTypes.POSITION_OPENED,
                data={
                    "position_id": position_id,
                    "portfolio_id": portfolio_id,
                    "symbol": request.symbol,
                    "side": request.side.value,
                    "quantity": request.quantity,
                    "price": execution_price,
                    "signal_id": request.signal_id,
                    "agent_id": request.agent_id
                }
            )
            
            logger.info(
                f"Opened {request.side.value} position for {request.symbol}: "
                f"{request.quantity} @ {execution_price}"
            )
            
            return position_info
            
        except Exception as e:
            logger.error(f"Failed to open position: {str(e)}")
            raise
    
    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close a position (fully or partially)
        
        Args:
            position_id: Position ID to close
            quantity: Quantity to close (None for full close)
            order_type: Order type for closing
            limit_price: Limit price if applicable
            
        Returns:
            Close execution details
        """
        try:
            position_info = self.positions_cache.get(position_id)
            if not position_info:
                raise ValueError(f"Position {position_id} not found")
            
            # Determine close quantity
            close_quantity = quantity or position_info.quantity
            if close_quantity > position_info.quantity:
                raise ValueError(f"Cannot close more than position quantity: {position_info.quantity}")
            
            # Get current market price
            market_data = await market_data_aggregator.get_latest_price(position_info.symbol)
            current_price = market_data.get("price", 0)
            
            if not current_price:
                raise ValueError(f"Unable to get price for {position_info.symbol}")
            
            # Calculate execution price
            close_side = OrderSide.SELL if position_info.position_type == PositionType.LONG else OrderSide.BUY
            execution_price = self._calculate_execution_price(
                order_type,
                close_side,
                current_price,
                limit_price,
                None
            )
            
            # Calculate P&L
            if position_info.position_type == PositionType.LONG:
                pnl = (execution_price - position_info.avg_entry_price) * close_quantity
            else:
                pnl = (position_info.avg_entry_price - execution_price) * close_quantity
            
            async with get_db() as session:
                # Get position from DB
                position = await session.get(Position, int(position_id))
                if not position:
                    raise ValueError(f"Position {position_id} not found in database")
                
                # Update position
                if close_quantity >= position_info.quantity:
                    # Full close
                    position.status = PositionStatus.CLOSED.value
                    position.closed_at = datetime.now()
                    position.quantity = 0
                else:
                    # Partial close
                    position.status = PositionStatus.PARTIALLY_CLOSED.value
                    position.quantity -= close_quantity
                
                position.realized_pnl = (position.realized_pnl or 0) + pnl
                position.current_price = execution_price
                
                # Create trade record for close
                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    user_id=1,  # TODO: Get from context
                    portfolio_id=position_info.portfolio_id,
                    position_id=position.id,
                    symbol=position_info.symbol,
                    side=close_side.value,
                    quantity=close_quantity,
                    price=execution_price,
                    total_value=close_quantity * execution_price,
                    order_type=order_type.value,
                    status="filled",
                    executed_at=datetime.now(),
                    metadata={
                        "close_reason": "manual",
                        "pnl": pnl
                    }
                )
                
                session.add(trade)
                
                # Update portfolio cash
                portfolio = await session.get(Portfolio, position_info.portfolio_id)
                if portfolio:
                    trade_value = close_quantity * execution_price
                    if close_side == OrderSide.SELL:
                        portfolio.cash_balance += trade_value
                    else:
                        portfolio.cash_balance -= trade_value
                
                await session.commit()
            
            # Update cache
            if close_quantity >= position_info.quantity:
                # Remove from cache if fully closed
                del self.positions_cache[position_id]
            else:
                # Update cached position
                position_info.quantity -= close_quantity
                position_info.realized_pnl += pnl
                position_info.total_pnl = position_info.realized_pnl + position_info.unrealized_pnl
                position_info.last_updated = datetime.now()
            
            # Publish event
            await event_bus.publish(
                EventTypes.POSITION_CLOSED,
                data={
                    "position_id": position_id,
                    "portfolio_id": position_info.portfolio_id,
                    "symbol": position_info.symbol,
                    "quantity_closed": close_quantity,
                    "price": execution_price,
                    "pnl": pnl,
                    "remaining_quantity": position_info.quantity if close_quantity < position_info.quantity else 0,
                    "signal_id": position_info.source_signal_id,
                    "agent_id": position_info.source_agent_id
                }
            )
            
            logger.info(
                f"Closed position {position_id}: {close_quantity} @ {execution_price}, "
                f"P&L: ${pnl:.2f}"
            )
            
            return {
                "position_id": position_id,
                "quantity_closed": close_quantity,
                "execution_price": execution_price,
                "pnl": pnl,
                "remaining_quantity": position_info.quantity if close_quantity < position_info.quantity else 0,
                "fully_closed": close_quantity >= position_info.quantity
            }
            
        except Exception as e:
            logger.error(f"Failed to close position: {str(e)}")
            raise
    
    async def update_position(
        self,
        position_id: str,
        update: PositionUpdate
    ) -> PositionInfo:
        """
        Update position parameters
        
        Args:
            position_id: Position ID
            update: Update parameters
            
        Returns:
            Updated position info
        """
        try:
            position_info = self.positions_cache.get(position_id)
            if not position_info:
                raise ValueError(f"Position {position_id} not found")
            
            # Update stop loss
            if update.stop_loss is not None:
                position_info.stop_loss = update.stop_loss
            
            # Update take profit
            if update.take_profit is not None:
                position_info.take_profit = update.take_profit
            
            # Update trailing stop
            if update.trailing_stop_percent is not None:
                position_info.trailing_stop_distance = update.trailing_stop_percent
            
            # Add to position
            if update.add_quantity:
                await self._add_to_position(position_id, update.add_quantity)
            
            # Reduce position
            if update.reduce_quantity:
                await self.close_position(position_id, update.reduce_quantity)
            
            # Update metadata
            if update.notes is not None:
                position_info.notes = update.notes
            
            if update.tags is not None:
                position_info.tags = update.tags
            
            position_info.last_updated = datetime.now()
            
            # Update database
            async with get_db() as session:
                position = await session.get(Position, int(position_id))
                if position:
                    # Update database fields
                    # Note: Some fields like stop_loss might need to be added to the model
                    await session.commit()
            
            logger.info(f"Updated position {position_id}")
            
            return position_info
            
        except Exception as e:
            logger.error(f"Failed to update position: {str(e)}")
            raise
    
    async def get_position(self, position_id: str) -> Optional[PositionInfo]:
        """Get position by ID"""
        return self.positions_cache.get(position_id)
    
    async def get_portfolio_positions(
        self,
        portfolio_id: int,
        status: Optional[PositionStatus] = None
    ) -> List[PositionInfo]:
        """Get all positions for a portfolio"""
        positions = [
            p for p in self.positions_cache.values()
            if p.portfolio_id == portfolio_id
        ]
        
        if status:
            # Filter by status if needed
            pass
        
        return positions
    
    async def get_portfolio_summary(self, portfolio_id: int) -> PortfolioSummary:
        """Get portfolio summary"""
        try:
            async with get_db() as session:
                portfolio = await session.get(Portfolio, portfolio_id)
                if not portfolio:
                    raise ValueError(f"Portfolio {portfolio_id} not found")
                
                # Get all positions
                positions = await self.get_portfolio_positions(portfolio_id)
                
                # Calculate metrics
                total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
                total_realized_pnl = sum(p.realized_pnl for p in positions)
                positions_value = sum(p.position_value for p in positions)
                total_exposure = sum(p.exposure for p in positions)
                margin_used = sum(p.margin_required for p in positions)
                
                # Position stats
                open_positions = len([p for p in positions if p.quantity > 0])
                winning_positions = len([p for p in positions if p.total_pnl > 0])
                losing_positions = len([p for p in positions if p.total_pnl < 0])
                
                # Performance metrics
                win_rate = winning_positions / max(open_positions, 1)
                winning_pnls = [p.total_pnl for p in positions if p.total_pnl > 0]
                losing_pnls = [p.total_pnl for p in positions if p.total_pnl < 0]
                avg_win = np.mean(winning_pnls) if winning_pnls else 0
                avg_loss = np.mean(losing_pnls) if losing_pnls else 0
                
                # Calculate Sharpe ratio (simplified)
                if positions:
                    returns = [p.pnl_percentage for p in positions]
                    if len(returns) > 1:
                        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                    else:
                        sharpe_ratio = 0
                else:
                    sharpe_ratio = 0
                
                # Total value
                total_value = portfolio.cash_balance + positions_value
                total_return_pct = ((total_value - portfolio.initial_capital) / portfolio.initial_capital * 100) if portfolio.initial_capital > 0 else 0
                
                summary = PortfolioSummary(
                    portfolio_id=portfolio_id,
                    name=portfolio.name,
                    total_value=total_value,
                    cash_balance=portfolio.cash_balance,
                    positions_value=positions_value,
                    total_unrealized_pnl=total_unrealized_pnl,
                    total_realized_pnl=total_realized_pnl,
                    daily_pnl=0,  # TODO: Calculate from history
                    total_return_pct=total_return_pct,
                    total_exposure=total_exposure,
                    margin_used=margin_used,
                    margin_available=portfolio.cash_balance - margin_used,
                    leverage=total_exposure / total_value if total_value > 0 else 0,
                    open_positions=open_positions,
                    winning_positions=winning_positions,
                    losing_positions=losing_positions,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=0,  # TODO: Calculate from history
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    created_at=portfolio.created_at,
                    last_updated=datetime.now()
                )
                
                # Cache summary
                self.portfolios_cache[portfolio_id] = summary
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {str(e)}")
            raise
    
    async def check_stop_loss_take_profit(self) -> None:
        """Check and execute stop loss/take profit orders"""
        for position_id, position in list(self.positions_cache.items()):
            try:
                # Get current price
                market_data = await market_data_aggregator.get_latest_price(position.symbol)
                current_price = market_data.get("price", 0)
                
                if not current_price:
                    continue
                
                should_close = False
                close_reason = ""
                
                # Check stop loss
                if position.stop_loss:
                    if position.position_type == PositionType.LONG:
                        if current_price <= position.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                    else:
                        if current_price >= position.stop_loss:
                            should_close = True
                            close_reason = "stop_loss"
                
                # Check take profit
                if position.take_profit and not should_close:
                    if position.position_type == PositionType.LONG:
                        if current_price >= position.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                    else:
                        if current_price <= position.take_profit:
                            should_close = True
                            close_reason = "take_profit"
                
                # Check trailing stop
                if position.trailing_stop_distance and not should_close:
                    trailing_stop_price = self._calculate_trailing_stop(
                        position,
                        current_price
                    )
                    
                    if position.position_type == PositionType.LONG:
                        if current_price <= trailing_stop_price:
                            should_close = True
                            close_reason = "trailing_stop"
                    else:
                        if current_price >= trailing_stop_price:
                            should_close = True
                            close_reason = "trailing_stop"
                
                # Execute close if triggered
                if should_close:
                    logger.info(
                        f"Closing position {position_id} due to {close_reason} "
                        f"at price {current_price}"
                    )
                    
                    result = await self.close_position(
                        position_id,
                        order_type=OrderType.MARKET
                    )
                    
                    # Publish alert
                    await event_bus.publish(
                        "position.auto_closed",
                        data={
                            "position_id": position_id,
                            "reason": close_reason,
                            "price": current_price,
                            "pnl": result["pnl"]
                        }
                    )
                    
            except Exception as e:
                logger.error(f"Error checking stops for position {position_id}: {str(e)}")
    
    async def update_position_prices(self) -> None:
        """Update current prices and P&L for all positions"""
        for position_id, position in self.positions_cache.items():
            try:
                # Get current price
                market_data = await market_data_aggregator.get_latest_price(position.symbol)
                current_price = market_data.get("price", 0)
                
                if not current_price or current_price == position.current_price:
                    continue
                
                # Update price
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.position_type == PositionType.LONG:
                    position.unrealized_pnl = (
                        (current_price - position.avg_entry_price) * position.quantity
                    )
                else:
                    position.unrealized_pnl = (
                        (position.avg_entry_price - current_price) * position.quantity
                    )
                
                # Update totals
                position.total_pnl = position.unrealized_pnl + position.realized_pnl
                position.pnl_percentage = (
                    position.total_pnl / (position.avg_entry_price * position.quantity) * 100
                    if position.avg_entry_price > 0 else 0
                )
                
                # Update position value
                position.position_value = abs(position.quantity * current_price)
                position.exposure = position.position_value
                
                # Track max profit/loss
                position.max_profit = max(position.max_profit, position.unrealized_pnl)
                position.max_loss = min(position.max_loss, position.unrealized_pnl)
                
                # Update timing
                position.last_updated = datetime.now()
                position.holding_period = position.last_updated - position.opened_at
                
            except Exception as e:
                logger.error(f"Error updating price for position {position_id}: {str(e)}")
    
    async def start_monitoring(self) -> None:
        """Start position monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._price_update_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Position monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop position monitoring"""
        self._monitoring_active = False
        
        if self._price_update_task:
            self._price_update_task.cancel()
            try:
                await self._price_update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Position monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Update prices
                await self.update_position_prices()
                
                # Check stops
                await self.check_stop_loss_take_profit()
                
                # Sleep
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _load_positions_from_db(self) -> None:
        """Load existing open positions from database"""
        try:
            async with get_db() as session:
                # Query open positions
                result = await session.execute(
                    select(Position).where(
                        Position.status.in_([PositionStatus.OPEN.value, PositionStatus.PARTIALLY_CLOSED.value])
                    )
                )
                positions = result.scalars().all()
                
                for position in positions:
                    # Convert to PositionInfo
                    position_info = PositionInfo(
                        position_id=str(position.id),
                        portfolio_id=position.portfolio_id,
                        symbol=position.symbol,
                        position_type=PositionType(position.position_type),
                        quantity=abs(position.quantity),
                        avg_entry_price=position.avg_entry_price,
                        current_price=position.current_price or position.avg_entry_price,
                        unrealized_pnl=position.unrealized_pnl or 0,
                        realized_pnl=position.realized_pnl or 0,
                        total_pnl=(position.unrealized_pnl or 0) + (position.realized_pnl or 0),
                        pnl_percentage=0,
                        position_value=abs(position.quantity * (position.current_price or position.avg_entry_price)),
                        exposure=abs(position.quantity * (position.current_price or position.avg_entry_price)),
                        margin_required=abs(position.quantity * (position.current_price or position.avg_entry_price)) * self.margin_requirement,
                        leverage=1.0,
                        opened_at=position.opened_at,
                        last_updated=datetime.now(),
                        holding_period=datetime.now() - position.opened_at
                    )
                    
                    self.positions_cache[str(position.id)] = position_info
                
                logger.info(f"Loaded {len(positions)} positions from database")
                
        except Exception as e:
            logger.error(f"Failed to load positions from database: {str(e)}")
    
    async def _validate_position_request(self, portfolio_id: int, request: PositionRequest) -> None:
        """Validate position request"""
        # Check portfolio exists
        async with get_db() as session:
            portfolio = await session.get(Portfolio, portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Check sufficient funds
            required_capital = request.quantity * (request.limit_price or 0)
            if required_capital > portfolio.cash_balance:
                raise ValueError(f"Insufficient funds: required {required_capital}, available {portfolio.cash_balance}")
    
    async def _check_risk_limits(
        self,
        portfolio_id: int,
        symbol: str,
        quantity: float,
        price: float
    ) -> None:
        """Check risk limits before opening position"""
        # Get portfolio summary
        summary = await self.get_portfolio_summary(portfolio_id)
        
        # Check position size limit
        position_value = quantity * price
        if position_value > summary.total_value * self.max_position_size_pct:
            raise ValueError(
                f"Position size {position_value} exceeds limit of "
                f"{summary.total_value * self.max_position_size_pct}"
            )
        
        # Check leverage limit
        new_exposure = summary.total_exposure + position_value
        new_leverage = new_exposure / summary.total_value
        if new_leverage > self.max_leverage:
            raise ValueError(
                f"New leverage {new_leverage:.2f} would exceed limit of {self.max_leverage}"
            )
    
    def _calculate_execution_price(
        self,
        order_type: OrderType,
        side: OrderSide,
        current_price: float,
        limit_price: Optional[float],
        stop_price: Optional[float]
    ) -> float:
        """Calculate execution price based on order type"""
        if order_type == OrderType.MARKET:
            # Add small slippage for market orders
            slippage = 0.001  # 0.1%
            if side == OrderSide.BUY:
                return current_price * (1 + slippage)
            else:
                return current_price * (1 - slippage)
        
        elif order_type == OrderType.LIMIT:
            return limit_price or current_price
        
        elif order_type == OrderType.STOP:
            return stop_price or current_price
        
        else:
            return current_price
    
    def _calculate_trailing_stop(self, position: PositionInfo, current_price: float) -> float:
        """Calculate trailing stop price"""
        if not position.trailing_stop_distance:
            return 0
        
        distance_pct = position.trailing_stop_distance / 100
        
        if position.position_type == PositionType.LONG:
            # For long positions, trail below the highest price
            highest_price = max(position.avg_entry_price, current_price)
            return highest_price * (1 - distance_pct)
        else:
            # For short positions, trail above the lowest price
            lowest_price = min(position.avg_entry_price, current_price)
            return lowest_price * (1 + distance_pct)
    
    async def _add_to_position(self, position_id: str, add_quantity: float) -> None:
        """Add to existing position"""
        position = self.positions_cache.get(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        # Get current price
        market_data = await market_data_aggregator.get_latest_price(position.symbol)
        current_price = market_data.get("price", 0)
        
        if not current_price:
            raise ValueError(f"Unable to get price for {position.symbol}")
        
        # Update average entry price
        total_cost = (position.avg_entry_price * position.quantity) + (current_price * add_quantity)
        new_quantity = position.quantity + add_quantity
        position.avg_entry_price = total_cost / new_quantity
        position.quantity = new_quantity
        
        # Update position value
        position.position_value = new_quantity * current_price
        position.exposure = position.position_value
        position.margin_required = position.position_value * self.margin_requirement
        
        logger.info(f"Added {add_quantity} to position {position_id}")
    
    async def _on_market_data(self, event) -> None:
        """Handle market data updates"""
        # Price updates are handled in the monitoring loop
        pass
    
    async def _on_trade_executed(self, event) -> None:
        """Handle trade execution events"""
        # Trade execution is handled in open/close position methods
        pass


# Global position manager instance
position_manager = PositionManager()