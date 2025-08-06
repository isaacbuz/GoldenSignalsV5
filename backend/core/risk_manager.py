"""
Risk Management System
Comprehensive risk limits, controls, and monitoring
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from decimal import Decimal
import numpy as np
import pandas as pd
import uuid

from pydantic import BaseModel, Field
from sqlalchemy import select, and_, func

from database.models import Portfolio, Position, Trade, Alert
from database.connection import get_db
from core.events.bus import event_bus, EventTypes
from core.logging import get_logger
from core.position_manager import position_manager, PositionInfo

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    BETA = "beta"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    MARGIN = "margin"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"


class RiskAction(Enum):
    """Risk control actions"""
    BLOCK_TRADE = "block_trade"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    HEDGE_POSITION = "hedge_position"
    ALERT_ONLY = "alert_only"
    FREEZE_TRADING = "freeze_trading"
    REDUCE_LEVERAGE = "reduce_leverage"


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Position limits
    max_position_size: float = 0.1  # 10% of portfolio
    max_position_count: int = 20
    max_concentration: float = 0.3  # 30% in single asset
    
    # Loss limits
    max_daily_loss: float = 0.02  # 2% daily loss
    max_weekly_loss: float = 0.05  # 5% weekly loss
    max_monthly_loss: float = 0.1  # 10% monthly loss
    max_drawdown: float = 0.15  # 15% max drawdown
    
    # Leverage limits
    max_leverage: float = 3.0
    max_margin_usage: float = 0.8  # 80% margin utilization
    min_free_margin: float = 0.2  # 20% free margin
    
    # Risk metrics limits
    max_var_95: float = 0.05  # 5% VaR at 95% confidence
    max_var_99: float = 0.1  # 10% VaR at 99% confidence
    min_sharpe_ratio: float = 0.5
    max_correlation: float = 0.8  # Position correlation
    
    # Volatility limits
    max_portfolio_volatility: float = 0.3  # 30% annualized
    max_position_volatility: float = 0.5  # 50% annualized
    
    # Trading limits
    max_trades_per_day: int = 100
    max_orders_per_minute: int = 10
    min_time_between_trades: int = 1  # seconds
    
    # Exposure limits
    max_long_exposure: float = 2.0  # 200% of capital
    max_short_exposure: float = 1.0  # 100% of capital
    max_net_exposure: float = 1.5  # 150% net


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    portfolio_id: int
    timestamp: datetime
    
    # Portfolio metrics
    total_value: float
    total_exposure: float
    net_exposure: float
    leverage: float
    margin_used: float
    margin_available: float
    
    # P&L metrics
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    max_drawdown: float
    current_drawdown: float
    
    # Risk metrics
    var_95: float  # 1-day VaR at 95%
    var_99: float  # 1-day VaR at 99%
    cvar_95: float  # Conditional VaR
    sharpe_ratio: float
    sortino_ratio: float
    portfolio_beta: float
    
    # Volatility
    portfolio_volatility: float
    avg_position_volatility: float
    
    # Concentration
    largest_position_pct: float
    top_5_concentration: float
    sector_concentration: Dict[str, float] = field(default_factory=dict)
    
    # Correlation
    avg_correlation: float
    max_correlation: float
    correlation_matrix: Optional[pd.DataFrame] = None
    
    # Trading activity
    trades_today: int
    orders_per_minute: float
    
    # Risk level
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    violations: List[str] = field(default_factory=list)


class RiskAlert(BaseModel):
    """Risk alert notification"""
    alert_id: str
    timestamp: datetime
    risk_level: RiskLevel
    metric_type: RiskMetricType
    current_value: float
    limit_value: float
    message: str
    recommended_action: RiskAction
    portfolio_id: Optional[int] = None
    position_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskControl(BaseModel):
    """Risk control rule"""
    rule_id: str
    name: str
    description: str
    metric_type: RiskMetricType
    condition: str  # e.g., "greater_than", "less_than"
    threshold: float
    action: RiskAction
    enabled: bool = True
    auto_execute: bool = False
    cooldown_minutes: int = 60
    last_triggered: Optional[datetime] = None


class RiskManager:
    """
    Centralized risk management system
    """
    
    def __init__(self):
        self.risk_limits = RiskLimits()
        self.risk_controls: List[RiskControl] = []
        self.current_metrics: Dict[int, RiskMetrics] = {}
        self.alert_history: List[RiskAlert] = []
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Risk calculation parameters
        self.var_confidence_levels = [0.95, 0.99]
        self.lookback_days = 252  # 1 year for historical calculations
        self.correlation_window = 20  # days
        
        # Circuit breakers
        self.circuit_breaker_triggered = False
        self.trading_frozen = False
        self.frozen_until: Optional[datetime] = None
        
        # Initialize default risk controls
        self._initialize_default_controls()
        
        logger.info("Risk Manager initialized")
    
    def _initialize_default_controls(self) -> None:
        """Initialize default risk control rules"""
        self.risk_controls = [
            RiskControl(
                rule_id="max_drawdown",
                name="Maximum Drawdown Control",
                description="Freeze trading when max drawdown exceeded",
                metric_type=RiskMetricType.MAX_DRAWDOWN,
                condition="greater_than",
                threshold=self.risk_limits.max_drawdown,
                action=RiskAction.FREEZE_TRADING,
                auto_execute=True
            ),
            RiskControl(
                rule_id="daily_loss",
                name="Daily Loss Limit",
                description="Stop trading when daily loss limit reached",
                metric_type=RiskMetricType.VAR,
                condition="greater_than",
                threshold=self.risk_limits.max_daily_loss,
                action=RiskAction.FREEZE_TRADING,
                auto_execute=True,
                cooldown_minutes=1440  # 24 hours
            ),
            RiskControl(
                rule_id="leverage_limit",
                name="Leverage Control",
                description="Reduce positions when leverage too high",
                metric_type=RiskMetricType.LEVERAGE,
                condition="greater_than",
                threshold=self.risk_limits.max_leverage,
                action=RiskAction.REDUCE_LEVERAGE,
                auto_execute=False
            ),
            RiskControl(
                rule_id="margin_call",
                name="Margin Call Protection",
                description="Close positions when margin critically low",
                metric_type=RiskMetricType.MARGIN,
                condition="less_than",
                threshold=0.1,  # 10% free margin
                action=RiskAction.CLOSE_POSITION,
                auto_execute=True
            ),
            RiskControl(
                rule_id="concentration",
                name="Concentration Risk",
                description="Alert on high concentration in single asset",
                metric_type=RiskMetricType.CONCENTRATION,
                condition="greater_than",
                threshold=self.risk_limits.max_concentration,
                action=RiskAction.ALERT_ONLY,
                auto_execute=False
            ),
            RiskControl(
                rule_id="volatility_spike",
                name="Volatility Spike Protection",
                description="Reduce exposure during high volatility",
                metric_type=RiskMetricType.VOLATILITY,
                condition="greater_than",
                threshold=self.risk_limits.max_portfolio_volatility * 1.5,
                action=RiskAction.REDUCE_POSITION,
                auto_execute=False
            )
        ]
    
    async def check_risk_limits(
        self,
        portfolio_id: int,
        proposed_trade: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if risk limits are satisfied
        
        Args:
            portfolio_id: Portfolio to check
            proposed_trade: Optional proposed trade to evaluate
            
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        try:
            # Check if trading is frozen
            if self.trading_frozen:
                if self.frozen_until and datetime.now() < self.frozen_until:
                    return False, ["Trading is frozen due to risk limits"]
                else:
                    self.trading_frozen = False
                    self.frozen_until = None
            
            # Calculate current metrics
            metrics = await self.calculate_risk_metrics(portfolio_id)
            
            violations = []
            
            # Check each risk limit
            if metrics.leverage > self.risk_limits.max_leverage:
                violations.append(f"Leverage {metrics.leverage:.2f} exceeds limit {self.risk_limits.max_leverage}")
            
            if metrics.current_drawdown > self.risk_limits.max_drawdown:
                violations.append(f"Drawdown {metrics.current_drawdown:.2%} exceeds limit {self.risk_limits.max_drawdown:.2%}")
            
            if abs(metrics.daily_pnl) > self.risk_limits.max_daily_loss * metrics.total_value:
                violations.append(f"Daily loss exceeds limit")
            
            if metrics.margin_available < metrics.total_value * self.risk_limits.min_free_margin:
                violations.append(f"Insufficient free margin: {metrics.margin_available:.2f}")
            
            if metrics.var_95 > self.risk_limits.max_var_95:
                violations.append(f"VaR(95%) {metrics.var_95:.2%} exceeds limit {self.risk_limits.max_var_95:.2%}")
            
            if metrics.largest_position_pct > self.risk_limits.max_concentration:
                violations.append(f"Position concentration {metrics.largest_position_pct:.2%} exceeds limit")
            
            # Check proposed trade if provided
            if proposed_trade:
                trade_violations = await self._check_trade_limits(portfolio_id, proposed_trade, metrics)
                violations.extend(trade_violations)
            
            # Store metrics
            self.current_metrics[portfolio_id] = metrics
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Risk limit check failed: {str(e)}")
            return False, [f"Risk check error: {str(e)}"]
    
    async def calculate_risk_metrics(self, portfolio_id: int) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for portfolio
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Risk metrics
        """
        try:
            # Get portfolio and positions
            positions = await position_manager.get_portfolio_positions(portfolio_id)
            portfolio_summary = await position_manager.get_portfolio_summary(portfolio_id)
            
            # Calculate exposures
            long_exposure = sum(p.position_value for p in positions if p.position_type.value == "long")
            short_exposure = sum(p.position_value for p in positions if p.position_type.value == "short")
            net_exposure = long_exposure - short_exposure
            total_exposure = long_exposure + short_exposure
            
            # Calculate leverage
            leverage = total_exposure / portfolio_summary.total_value if portfolio_summary.total_value > 0 else 0
            
            # Calculate P&L metrics
            daily_pnl = await self._calculate_period_pnl(portfolio_id, 1)
            weekly_pnl = await self._calculate_period_pnl(portfolio_id, 7)
            monthly_pnl = await self._calculate_period_pnl(portfolio_id, 30)
            
            # Calculate VaR and CVaR
            var_95, var_99, cvar_95 = await self._calculate_var(portfolio_id, positions)
            
            # Calculate volatility
            portfolio_vol, avg_position_vol = await self._calculate_volatility(positions)
            
            # Calculate Sharpe and Sortino ratios
            sharpe, sortino = await self._calculate_risk_adjusted_returns(portfolio_id)
            
            # Calculate concentration
            concentration = self._calculate_concentration(positions, portfolio_summary.total_value)
            
            # Calculate correlation
            correlation_data = await self._calculate_correlations(positions)
            
            # Calculate drawdown
            max_dd, current_dd = await self._calculate_drawdown(portfolio_id)
            
            # Trading activity
            trades_today = await self._count_trades_today(portfolio_id)
            
            # Determine overall risk level
            risk_score = self._calculate_risk_score({
                "leverage": leverage,
                "drawdown": current_dd,
                "var_95": var_95,
                "volatility": portfolio_vol,
                "concentration": concentration["largest_position_pct"]
            })
            
            risk_level = self._determine_risk_level(risk_score)
            
            # Find violations
            violations = []
            if leverage > self.risk_limits.max_leverage:
                violations.append("Leverage exceeds limit")
            if current_dd > self.risk_limits.max_drawdown:
                violations.append("Drawdown exceeds limit")
            if var_95 > self.risk_limits.max_var_95:
                violations.append("VaR exceeds limit")
            
            return RiskMetrics(
                portfolio_id=portfolio_id,
                timestamp=datetime.now(),
                total_value=portfolio_summary.total_value,
                total_exposure=total_exposure,
                net_exposure=net_exposure,
                leverage=leverage,
                margin_used=portfolio_summary.margin_used,
                margin_available=portfolio_summary.margin_available,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                monthly_pnl=monthly_pnl,
                unrealized_pnl=portfolio_summary.total_unrealized_pnl,
                realized_pnl=portfolio_summary.total_realized_pnl,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                portfolio_beta=0,  # TODO: Calculate beta
                portfolio_volatility=portfolio_vol,
                avg_position_volatility=avg_position_vol,
                largest_position_pct=concentration["largest_position_pct"],
                top_5_concentration=concentration["top_5_concentration"],
                sector_concentration=concentration.get("sector_concentration", {}),
                avg_correlation=correlation_data["avg_correlation"],
                max_correlation=correlation_data["max_correlation"],
                correlation_matrix=correlation_data.get("matrix"),
                trades_today=trades_today,
                orders_per_minute=0,  # TODO: Calculate from recent activity
                overall_risk_level=risk_level,
                risk_score=risk_score,
                violations=violations
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {str(e)}")
            raise
    
    async def execute_risk_control(
        self,
        control: RiskControl,
        portfolio_id: int,
        current_value: float
    ) -> bool:
        """
        Execute risk control action
        
        Args:
            control: Risk control rule
            portfolio_id: Portfolio ID
            current_value: Current metric value
            
        Returns:
            Success status
        """
        try:
            logger.warning(
                f"Executing risk control {control.name}: {control.action.value} "
                f"for portfolio {portfolio_id}"
            )
            
            if control.action == RiskAction.FREEZE_TRADING:
                self.trading_frozen = True
                self.frozen_until = datetime.now() + timedelta(minutes=control.cooldown_minutes)
                await self._send_alert(
                    RiskLevel.CRITICAL,
                    control.metric_type,
                    current_value,
                    control.threshold,
                    f"Trading frozen until {self.frozen_until}",
                    control.action,
                    portfolio_id
                )
                
            elif control.action == RiskAction.REDUCE_POSITION:
                # Reduce largest positions by 50%
                positions = await position_manager.get_portfolio_positions(portfolio_id)
                positions_sorted = sorted(positions, key=lambda p: p.position_value, reverse=True)
                
                for position in positions_sorted[:3]:  # Reduce top 3 positions
                    reduce_qty = position.quantity * 0.5
                    await position_manager.close_position(
                        position.position_id,
                        quantity=reduce_qty
                    )
                    logger.info(f"Reduced position {position.position_id} by 50%")
                
            elif control.action == RiskAction.CLOSE_POSITION:
                # Close all positions
                positions = await position_manager.get_portfolio_positions(portfolio_id)
                for position in positions:
                    await position_manager.close_position(position.position_id)
                    logger.info(f"Closed position {position.position_id} due to risk control")
                
            elif control.action == RiskAction.HEDGE_POSITION:
                # Create hedges for largest positions
                positions = await position_manager.get_portfolio_positions(portfolio_id)
                positions_sorted = sorted(positions, key=lambda p: p.position_value, reverse=True)
                
                for position in positions_sorted[:2]:  # Hedge top 2 positions
                    # This would create opposite positions
                    logger.info(f"Would hedge position {position.position_id}")
                
            elif control.action == RiskAction.REDUCE_LEVERAGE:
                # Reduce positions to bring leverage down
                target_leverage = self.risk_limits.max_leverage * 0.8
                await self._reduce_to_target_leverage(portfolio_id, target_leverage)
                
            elif control.action == RiskAction.ALERT_ONLY:
                # Just send alert
                await self._send_alert(
                    RiskLevel.HIGH,
                    control.metric_type,
                    current_value,
                    control.threshold,
                    f"Risk limit breached: {control.name}",
                    control.action,
                    portfolio_id
                )
            
            # Update last triggered
            control.last_triggered = datetime.now()
            
            # Publish event
            await event_bus.publish(
                "risk.control_executed",
                data={
                    "control_id": control.rule_id,
                    "action": control.action.value,
                    "portfolio_id": portfolio_id,
                    "metric_value": current_value,
                    "threshold": control.threshold
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute risk control: {str(e)}")
            return False
    
    async def monitor_risk(self) -> None:
        """
        Continuous risk monitoring loop
        """
        while self._monitoring_active:
            try:
                # Monitor each portfolio
                async with get_db() as session:
                    result = await session.execute(
                        select(Portfolio).where(Portfolio.is_active == True)
                    )
                    portfolios = result.scalars().all()
                
                for portfolio in portfolios:
                    # Calculate metrics
                    metrics = await self.calculate_risk_metrics(portfolio.id)
                    
                    # Check controls
                    for control in self.risk_controls:
                        if not control.enabled:
                            continue
                        
                        # Check if control should trigger
                        should_trigger = await self._should_trigger_control(
                            control,
                            metrics
                        )
                        
                        if should_trigger:
                            # Check cooldown
                            if control.last_triggered:
                                time_since = datetime.now() - control.last_triggered
                                if time_since.total_seconds() < control.cooldown_minutes * 60:
                                    continue
                            
                            # Get current metric value
                            metric_value = self._get_metric_value(control.metric_type, metrics)
                            
                            if control.auto_execute:
                                # Execute control action
                                await self.execute_risk_control(
                                    control,
                                    portfolio.id,
                                    metric_value
                                )
                            else:
                                # Just send alert
                                await self._send_alert(
                                    RiskLevel.HIGH,
                                    control.metric_type,
                                    metric_value,
                                    control.threshold,
                                    f"Risk control triggered: {control.name}",
                                    control.action,
                                    portfolio.id
                                )
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {str(e)}")
                await asyncio.sleep(30)
    
    async def start_monitoring(self) -> None:
        """Start risk monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self.monitor_risk())
        
        # Subscribe to events
        await event_bus.subscribe(EventTypes.POSITION_OPENED, self._on_position_opened)
        await event_bus.subscribe(EventTypes.POSITION_CLOSED, self._on_position_closed)
        await event_bus.subscribe(EventTypes.TRADE_EXECUTED, self._on_trade_executed)
        
        logger.info("Risk monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop risk monitoring"""
        self._monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk monitoring stopped")
    
    def set_risk_limits(self, limits: RiskLimits) -> None:
        """Update risk limits"""
        self.risk_limits = limits
        logger.info("Risk limits updated")
    
    def add_risk_control(self, control: RiskControl) -> None:
        """Add new risk control rule"""
        self.risk_controls.append(control)
        logger.info(f"Added risk control: {control.name}")
    
    def get_risk_summary(self, portfolio_id: int) -> Dict[str, Any]:
        """Get risk summary for portfolio"""
        metrics = self.current_metrics.get(portfolio_id)
        if not metrics:
            return {"error": "No metrics available"}
        
        return {
            "portfolio_id": portfolio_id,
            "risk_level": metrics.overall_risk_level.value,
            "risk_score": metrics.risk_score,
            "leverage": metrics.leverage,
            "var_95": metrics.var_95,
            "current_drawdown": metrics.current_drawdown,
            "violations": metrics.violations,
            "trading_frozen": self.trading_frozen,
            "frozen_until": self.frozen_until,
            "timestamp": metrics.timestamp
        }
    
    async def _calculate_period_pnl(self, portfolio_id: int, days: int) -> float:
        """Calculate P&L for period"""
        try:
            async with get_db() as session:
                start_date = datetime.now() - timedelta(days=days)
                
                # Query trades in period
                result = await session.execute(
                    select(func.sum(Trade.total_value)).where(
                        and_(
                            Trade.portfolio_id == portfolio_id,
                            Trade.executed_at >= start_date
                        )
                    )
                )
                
                pnl = result.scalar() or 0
                return pnl
                
        except Exception as e:
            logger.error(f"Failed to calculate period P&L: {str(e)}")
            return 0
    
    async def _calculate_var(
        self,
        portfolio_id: int,
        positions: List[PositionInfo]
    ) -> Tuple[float, float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        try:
            if not positions:
                return 0, 0, 0
            
            # Get historical returns (simplified)
            returns = []
            for position in positions:
                # Simulate returns based on position volatility
                position_returns = np.random.normal(
                    0,
                    position.volatility / np.sqrt(252),
                    self.lookback_days
                )
                returns.append(position_returns * position.position_value)
            
            # Portfolio returns
            portfolio_returns = np.sum(returns, axis=0)
            
            # Calculate VaR
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            
            # Calculate CVaR (expected shortfall)
            cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
            
            # Convert to positive values and percentages
            portfolio_value = sum(p.position_value for p in positions)
            if portfolio_value > 0:
                var_95 = abs(var_95) / portfolio_value
                var_99 = abs(var_99) / portfolio_value
                cvar_95 = abs(cvar_95) / portfolio_value
            else:
                var_95 = var_99 = cvar_95 = 0
            
            return var_95, var_99, cvar_95
            
        except Exception as e:
            logger.error(f"Failed to calculate VaR: {str(e)}")
            return 0, 0, 0
    
    async def _calculate_volatility(
        self,
        positions: List[PositionInfo]
    ) -> Tuple[float, float]:
        """Calculate portfolio and average position volatility"""
        try:
            if not positions:
                return 0, 0
            
            # Position volatilities (simplified)
            position_vols = [p.volatility for p in positions if p.volatility > 0]
            
            if not position_vols:
                return 0, 0
            
            # Average position volatility
            avg_position_vol = np.mean(position_vols)
            
            # Portfolio volatility (simplified - ignoring correlations)
            weights = [p.position_value for p in positions]
            total_value = sum(weights)
            
            if total_value > 0:
                weights = [w / total_value for w in weights]
                portfolio_vol = np.sqrt(sum(w**2 * v**2 for w, v in zip(weights, position_vols)))
            else:
                portfolio_vol = 0
            
            return portfolio_vol, avg_position_vol
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility: {str(e)}")
            return 0, 0
    
    async def _calculate_risk_adjusted_returns(
        self,
        portfolio_id: int
    ) -> Tuple[float, float]:
        """Calculate Sharpe and Sortino ratios"""
        try:
            # Get historical returns (simplified)
            returns = []
            
            async with get_db() as session:
                # Query historical P&L
                result = await session.execute(
                    select(Trade).where(
                        Trade.portfolio_id == portfolio_id
                    ).order_by(Trade.executed_at.desc()).limit(self.lookback_days)
                )
                trades = result.scalars().all()
            
            if not trades:
                return 0, 0
            
            # Calculate daily returns (simplified)
            for trade in trades:
                # This is simplified - would need actual daily P&L
                returns.append(0.001)  # Placeholder
            
            returns = np.array(returns)
            
            # Sharpe ratio
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            else:
                sharpe = 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            else:
                sortino = sharpe  # If no downside, use Sharpe
            
            return sharpe, sortino
            
        except Exception as e:
            logger.error(f"Failed to calculate risk-adjusted returns: {str(e)}")
            return 0, 0
    
    def _calculate_concentration(
        self,
        positions: List[PositionInfo],
        total_value: float
    ) -> Dict[str, float]:
        """Calculate concentration metrics"""
        if not positions or total_value <= 0:
            return {
                "largest_position_pct": 0,
                "top_5_concentration": 0,
                "sector_concentration": {}
            }
        
        # Sort by value
        positions_sorted = sorted(positions, key=lambda p: p.position_value, reverse=True)
        
        # Largest position
        largest_pct = positions_sorted[0].position_value / total_value if positions_sorted else 0
        
        # Top 5 concentration
        top_5_value = sum(p.position_value for p in positions_sorted[:5])
        top_5_pct = top_5_value / total_value
        
        # Sector concentration (simplified - would need sector mapping)
        sector_concentration = {}  # TODO: Implement sector mapping
        
        return {
            "largest_position_pct": largest_pct,
            "top_5_concentration": top_5_pct,
            "sector_concentration": sector_concentration
        }
    
    async def _calculate_correlations(
        self,
        positions: List[PositionInfo]
    ) -> Dict[str, Any]:
        """Calculate position correlations"""
        try:
            if len(positions) < 2:
                return {
                    "avg_correlation": 0,
                    "max_correlation": 0,
                    "matrix": None
                }
            
            # This would fetch historical price data and calculate correlations
            # Simplified version
            n = len(positions)
            correlation_matrix = np.random.uniform(0.3, 0.7, (n, n))
            np.fill_diagonal(correlation_matrix, 1.0)
            
            # Make symmetric
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            
            # Extract upper triangle (excluding diagonal)
            upper_triangle = correlation_matrix[np.triu_indices(n, k=1)]
            
            return {
                "avg_correlation": np.mean(upper_triangle),
                "max_correlation": np.max(upper_triangle),
                "matrix": pd.DataFrame(
                    correlation_matrix,
                    index=[p.symbol for p in positions],
                    columns=[p.symbol for p in positions]
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {str(e)}")
            return {"avg_correlation": 0, "max_correlation": 0, "matrix": None}
    
    async def _calculate_drawdown(
        self,
        portfolio_id: int
    ) -> Tuple[float, float]:
        """Calculate maximum and current drawdown"""
        try:
            # Get portfolio value history (simplified)
            async with get_db() as session:
                portfolio = await session.get(Portfolio, portfolio_id)
                if not portfolio:
                    return 0, 0
                
                # Calculate from initial capital (simplified)
                current_value = portfolio.current_value or portfolio.initial_capital
                peak_value = portfolio.initial_capital * 1.2  # Placeholder
                
                if peak_value > 0:
                    current_dd = (peak_value - current_value) / peak_value
                    max_dd = 0.1  # Placeholder
                else:
                    current_dd = max_dd = 0
                
                return max_dd, max(current_dd, 0)
                
        except Exception as e:
            logger.error(f"Failed to calculate drawdown: {str(e)}")
            return 0, 0
    
    async def _count_trades_today(self, portfolio_id: int) -> int:
        """Count trades executed today"""
        try:
            async with get_db() as session:
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                result = await session.execute(
                    select(func.count(Trade.id)).where(
                        and_(
                            Trade.portfolio_id == portfolio_id,
                            Trade.executed_at >= today
                        )
                    )
                )
                
                return result.scalar() or 0
                
        except Exception as e:
            logger.error(f"Failed to count trades: {str(e)}")
            return 0
    
    def _calculate_risk_score(self, factors: Dict[str, float]) -> float:
        """Calculate overall risk score"""
        # Weighted risk scoring
        weights = {
            "leverage": 0.25,
            "drawdown": 0.25,
            "var_95": 0.2,
            "volatility": 0.15,
            "concentration": 0.15
        }
        
        score = 0
        for factor, value in factors.items():
            if factor in weights:
                # Normalize to 0-1 scale
                if factor == "leverage":
                    normalized = min(value / self.risk_limits.max_leverage, 1)
                elif factor == "drawdown":
                    normalized = min(value / self.risk_limits.max_drawdown, 1)
                elif factor == "var_95":
                    normalized = min(value / self.risk_limits.max_var_95, 1)
                elif factor == "volatility":
                    normalized = min(value / self.risk_limits.max_portfolio_volatility, 1)
                elif factor == "concentration":
                    normalized = min(value / self.risk_limits.max_concentration, 1)
                else:
                    normalized = 0
                
                score += normalized * weights[factor]
        
        return min(score, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    async def _check_trade_limits(
        self,
        portfolio_id: int,
        proposed_trade: Dict[str, Any],
        metrics: RiskMetrics
    ) -> List[str]:
        """Check if proposed trade violates limits"""
        violations = []
        
        # Check position size
        trade_value = proposed_trade.get("quantity", 0) * proposed_trade.get("price", 0)
        if trade_value > metrics.total_value * self.risk_limits.max_position_size:
            violations.append(f"Trade size exceeds position limit")
        
        # Check if would exceed leverage
        new_exposure = metrics.total_exposure + trade_value
        new_leverage = new_exposure / metrics.total_value
        if new_leverage > self.risk_limits.max_leverage:
            violations.append(f"Trade would exceed leverage limit")
        
        # Check daily trade count
        if metrics.trades_today >= self.risk_limits.max_trades_per_day:
            violations.append(f"Daily trade limit reached")
        
        return violations
    
    async def _should_trigger_control(
        self,
        control: RiskControl,
        metrics: RiskMetrics
    ) -> bool:
        """Check if control should trigger"""
        metric_value = self._get_metric_value(control.metric_type, metrics)
        
        if control.condition == "greater_than":
            return metric_value > control.threshold
        elif control.condition == "less_than":
            return metric_value < control.threshold
        elif control.condition == "equals":
            return abs(metric_value - control.threshold) < 0.001
        else:
            return False
    
    def _get_metric_value(self, metric_type: RiskMetricType, metrics: RiskMetrics) -> float:
        """Get metric value from metrics object"""
        mapping = {
            RiskMetricType.VAR: metrics.var_95,
            RiskMetricType.CVAR: metrics.cvar_95,
            RiskMetricType.SHARPE: metrics.sharpe_ratio,
            RiskMetricType.SORTINO: metrics.sortino_ratio,
            RiskMetricType.MAX_DRAWDOWN: metrics.current_drawdown,
            RiskMetricType.LEVERAGE: metrics.leverage,
            RiskMetricType.MARGIN: metrics.margin_available / metrics.total_value if metrics.total_value > 0 else 0,
            RiskMetricType.VOLATILITY: metrics.portfolio_volatility,
            RiskMetricType.CONCENTRATION: metrics.largest_position_pct
        }
        
        return mapping.get(metric_type, 0)
    
    async def _reduce_to_target_leverage(self, portfolio_id: int, target_leverage: float) -> None:
        """Reduce positions to achieve target leverage"""
        try:
            positions = await position_manager.get_portfolio_positions(portfolio_id)
            portfolio_summary = await position_manager.get_portfolio_summary(portfolio_id)
            
            current_exposure = portfolio_summary.total_exposure
            target_exposure = portfolio_summary.total_value * target_leverage
            
            if current_exposure <= target_exposure:
                return
            
            reduction_needed = current_exposure - target_exposure
            
            # Sort positions by size
            positions_sorted = sorted(positions, key=lambda p: p.position_value, reverse=True)
            
            for position in positions_sorted:
                if reduction_needed <= 0:
                    break
                
                # Reduce position
                reduce_amount = min(position.position_value, reduction_needed)
                reduce_qty = (reduce_amount / position.position_value) * position.quantity
                
                await position_manager.close_position(
                    position.position_id,
                    quantity=reduce_qty
                )
                
                reduction_needed -= reduce_amount
                
        except Exception as e:
            logger.error(f"Failed to reduce leverage: {str(e)}")
    
    async def _send_alert(
        self,
        level: RiskLevel,
        metric_type: RiskMetricType,
        current_value: float,
        limit_value: float,
        message: str,
        action: RiskAction,
        portfolio_id: Optional[int] = None,
        position_id: Optional[str] = None
    ) -> None:
        """Send risk alert"""
        alert = RiskAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            risk_level=level,
            metric_type=metric_type,
            current_value=current_value,
            limit_value=limit_value,
            message=message,
            recommended_action=action,
            portfolio_id=portfolio_id,
            position_id=position_id
        )
        
        # Store alert
        self.alert_history.append(alert)
        
        # Store in database
        async with get_db() as session:
            db_alert = Alert(
                alert_type="risk",
                severity=level.value,
                source="risk_manager",
                title=f"Risk Alert: {metric_type.value}",
                message=message,
                metadata=alert.dict()
            )
            session.add(db_alert)
            await session.commit()
        
        # Publish event
        await event_bus.publish(
            "risk.alert",
            data=alert.dict()
        )
        
        logger.warning(f"Risk alert: {message}")
    
    async def _on_position_opened(self, event) -> None:
        """Handle position opened event"""
        try:
            data = event.data
            portfolio_id = data.get("portfolio_id")
            
            if portfolio_id:
                # Check risk after position opened
                is_ok, violations = await self.check_risk_limits(portfolio_id)
                
                if not is_ok:
                    logger.warning(f"Risk violations after position open: {violations}")
                    
        except Exception as e:
            logger.error(f"Error handling position opened: {str(e)}")
    
    async def _on_position_closed(self, event) -> None:
        """Handle position closed event"""
        # Update metrics after position closed
        pass
    
    async def _on_trade_executed(self, event) -> None:
        """Handle trade executed event"""
        # Track trading activity
        pass


# Global risk manager instance
risk_manager = RiskManager()