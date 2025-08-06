"""
Backtesting Engine V5
Comprehensive backtesting framework with agent integration
Follows V5 architecture patterns with context-aware execution
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, AsyncIterator
import pandas as pd
import numpy as np
from collections import defaultdict

from pydantic import BaseModel, Field

from agents.base import BaseAgent, AgentContext
from services.market_data_aggregator import market_data_aggregator
from core.logging import get_logger
from rag.core.engine import rag_engine

logger = get_logger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes"""
    PAPER = "paper"            # Simulated trading without real execution
    HISTORICAL = "historical"  # Pure historical data replay
    WALK_FORWARD = "walk_forward"  # Walk-forward optimization
    MONTE_CARLO = "monte_carlo"    # Monte Carlo simulation
    STRESS_TEST = "stress_test"    # Stress testing scenarios


class OrderType(Enum):
    """Order types for backtesting"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1d"
    mode: BacktestMode = BacktestMode.HISTORICAL
    
    # Execution parameters
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.001    # 0.1% slippage
    spread: float = 0.0001     # 0.01% spread
    
    # Risk parameters
    max_position_size: float = 0.1  # 10% max per position
    max_leverage: float = 1.0       # No leverage by default
    stop_loss: Optional[float] = 0.02  # 2% stop loss
    take_profit: Optional[float] = 0.05  # 5% take profit
    
    # Advanced parameters
    allow_shorting: bool = False
    reinvest_profits: bool = True
    use_kelly_criterion: bool = False
    
    # Agent parameters
    agents_to_test: List[str] = field(default_factory=list)
    agent_weights: Dict[str, float] = field(default_factory=dict)
    
    # Data parameters
    data_quality: str = "minute"
    include_extended_hours: bool = False
    
    # Performance tracking
    track_metrics: bool = True
    save_trades: bool = True
    save_signals: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "mode": self.mode.value,
            "commission": self.commission,
            "slippage": self.slippage,
            "agents": self.agents_to_test
        }


@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    order_type: OrderType
    commission: float
    slippage_cost: float
    
    # Execution details
    signal_source: str  # Which agent generated signal
    signal_strength: float
    
    # P&L tracking
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    # Risk metrics
    position_size: float = 0.0
    risk_amount: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type.value,
            "commission": self.commission,
            "slippage_cost": self.slippage_cost,
            "signal_source": self.signal_source,
            "signal_strength": self.signal_strength,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent
        }


@dataclass
class Position:
    """Current position tracking"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    entry_time: datetime
    
    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    def update_price(self, price: float):
        """Update position with current price"""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl
        }


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Execution metrics
    avg_holding_period: timedelta = timedelta(0)
    turnover_rate: float = 0.0
    
    # Cost analysis
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Advanced metrics
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    
    def calculate_metrics(self, equity_curve: pd.Series, trades: List[Trade]):
        """Calculate all performance metrics"""
        if equity_curve.empty:
            return
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        self.total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        # Annual return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        self.annual_return = ((1 + self.total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(returns) > 0 and returns.std() > 0:
            self.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns**2))
            if downside_std > 0:
                self.sortino_ratio = np.sqrt(252) * returns.mean() / downside_std
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        self.max_drawdown = drawdown.min() * 100
        
        # VaR and CVaR
        if len(returns) > 0:
            self.var_95 = np.percentile(returns, 5) * 100
            self.cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Trading metrics
        self.total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]
        
        self.winning_trades = len(winning)
        self.losing_trades = len(losing)
        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        if winning:
            self.avg_win = np.mean([t.pnl for t in winning])
        if losing:
            self.avg_loss = abs(np.mean([t.pnl for t in losing]))
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calmar ratio
        if self.max_drawdown < 0:
            self.calmar_ratio = self.annual_return / abs(self.max_drawdown)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": round(self.total_return, 2),
            "annual_return": round(self.annual_return, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "calmar_ratio": round(self.calmar_ratio, 2)
        }


class BacktestEngine:
    """
    Main backtesting engine
    Integrates with agents and market data providers
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.signals: List[Dict[str, Any]] = []
        
        # Portfolio tracking
        self.cash = config.initial_capital
        self.equity_curve = []
        self.portfolio_value_history = []
        
        # Agent integration
        self.agents: Dict[str, BaseAgent] = {}
        self._trade_counter = 0
        
        # Performance tracking
        self.metrics = BacktestMetrics()
        
        # Event callbacks
        self.on_trade_callbacks: List[Callable] = []
        self.on_signal_callbacks: List[Callable] = []
        
        logger.info(f"Initialized backtest engine: {config.start_date} to {config.end_date}")
    
    async def run(self) -> BacktestMetrics:
        """
        Run the backtest
        
        Returns:
            BacktestMetrics with performance results
        """
        try:
            logger.info("Starting backtest...")
            
            # Load agents
            await self._load_agents()
            
            # Fetch historical data
            market_data = await self._fetch_market_data()
            
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for backtest period")
            
            # Run backtest based on mode
            if self.config.mode == BacktestMode.HISTORICAL:
                await self._run_historical_backtest(market_data)
            elif self.config.mode == BacktestMode.WALK_FORWARD:
                await self._run_walk_forward_backtest(market_data)
            elif self.config.mode == BacktestMode.MONTE_CARLO:
                await self._run_monte_carlo_backtest(market_data)
            else:
                await self._run_historical_backtest(market_data)
            
            # Calculate final metrics
            self._calculate_final_metrics()
            
            # Store results in RAG for learning
            await self._store_backtest_results()
            
            logger.info(f"Backtest complete: {self.metrics.total_return:.2f}% return")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise
    
    async def _run_historical_backtest(self, market_data: pd.DataFrame):
        """Run standard historical backtest"""
        
        # Group data by timestamp for time-series iteration
        grouped = market_data.groupby('timestamp')
        
        for timestamp, data in grouped:
            current_prices = {}
            
            # Update prices for all symbols
            for _, row in data.iterrows():
                symbol = row.get('symbol')
                if symbol:
                    current_prices[symbol] = row['close']
            
            # Update existing positions
            self._update_positions(current_prices)
            
            # Generate signals from agents
            signals = await self._generate_signals(timestamp, data)
            
            # Execute trades based on signals
            await self._execute_signals(signals, current_prices)
            
            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices)
            self.equity_curve.append({
                'timestamp': timestamp,
                'value': portfolio_value
            })
            
            # Check stop loss and take profit
            self._check_exits(current_prices)
    
    async def _run_walk_forward_backtest(self, market_data: pd.DataFrame):
        """
        Run walk-forward optimization backtest
        Periodically re-optimize parameters
        """
        # Split data into windows
        window_size = 30  # days
        optimization_period = 90  # days
        
        timestamps = market_data['timestamp'].unique()
        
        for i in range(optimization_period, len(timestamps), window_size):
            # Optimization window
            opt_start = max(0, i - optimization_period)
            opt_data = market_data[
                (market_data['timestamp'] >= timestamps[opt_start]) &
                (market_data['timestamp'] < timestamps[i])
            ]
            
            # Optimize agent parameters
            await self._optimize_agents(opt_data)
            
            # Test window
            test_end = min(i + window_size, len(timestamps))
            test_data = market_data[
                (market_data['timestamp'] >= timestamps[i]) &
                (market_data['timestamp'] < timestamps[test_end])
            ]
            
            # Run backtest on test window
            await self._run_historical_backtest(test_data)
    
    async def _run_monte_carlo_backtest(self, market_data: pd.DataFrame, n_simulations: int = 100):
        """
        Run Monte Carlo simulation
        Tests strategy robustness with random variations
        """
        all_metrics = []
        
        for i in range(n_simulations):
            # Reset state
            self._reset()
            
            # Add random noise to data
            noisy_data = self._add_noise_to_data(market_data, noise_level=0.01)
            
            # Randomly vary parameters
            self._randomize_parameters()
            
            # Run backtest
            await self._run_historical_backtest(noisy_data)
            
            # Store metrics
            all_metrics.append(self.metrics.to_dict())
        
        # Calculate statistics across simulations
        self._calculate_monte_carlo_statistics(all_metrics)
    
    async def _generate_signals(self, timestamp: datetime, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals from agents"""
        signals = []
        
        for agent_name, agent in self.agents.items():
            try:
                # Create context for agent
                context = AgentContext(
                    symbol=data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN",
                    timeframe=self.config.timeframe,
                    market_data=data.to_dict()
                )
                
                # Get signal from agent
                result = await agent.analyze(context)
                
                if result and result.get('signal'):
                    signal = {
                        'timestamp': timestamp,
                        'agent': agent_name,
                        'symbol': context.symbol,
                        'action': result['signal'],
                        'confidence': result.get('confidence', 0.5),
                        'metadata': result.get('metadata', {})
                    }
                    
                    signals.append(signal)
                    self.signals.append(signal)
                    
                    # Trigger callbacks
                    for callback in self.on_signal_callbacks:
                        await callback(signal)
                        
            except Exception as e:
                logger.warning(f"Agent {agent_name} failed to generate signal: {str(e)}")
        
        return signals
    
    async def _execute_signals(self, signals: List[Dict[str, Any]], current_prices: Dict[str, float]):
        """Execute trades based on signals"""
        
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal['confidence']
            
            # Check if we should trade based on confidence and weights
            min_confidence = 0.6
            if confidence < min_confidence:
                continue
            
            # Get agent weight
            agent_weight = self.config.agent_weights.get(signal['agent'], 1.0)
            if agent_weight <= 0:
                continue
            
            # Determine position size
            position_size = self._calculate_position_size(symbol, confidence, agent_weight)
            
            # Execute trade
            if action in ['BUY', 'STRONG_BUY']:
                await self._execute_buy(symbol, position_size, current_prices.get(symbol), signal)
            elif action in ['SELL', 'STRONG_SELL']:
                await self._execute_sell(symbol, position_size, current_prices.get(symbol), signal)
    
    async def _execute_buy(self, symbol: str, size: float, price: float, signal: Dict[str, Any]):
        """Execute buy order"""
        if not price or size <= 0:
            return
        
        # Calculate costs
        cost = size * price
        commission = cost * self.config.commission
        slippage = price * self.config.slippage
        actual_price = price + slippage
        total_cost = cost + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            return
        
        # Create trade
        self._trade_counter += 1
        trade = Trade(
            trade_id=f"T{self._trade_counter:06d}",
            timestamp=signal['timestamp'],
            symbol=symbol,
            side='buy',
            quantity=size,
            price=actual_price,
            order_type=OrderType.MARKET,
            commission=commission,
            slippage_cost=slippage * size,
            signal_source=signal['agent'],
            signal_strength=signal['confidence'],
            entry_price=actual_price
        )
        
        # Update position
        if symbol in self.positions:
            position = self.positions[symbol]
            # Average up
            total_quantity = position.quantity + size
            position.avg_price = (position.avg_price * position.quantity + actual_price * size) / total_quantity
            position.quantity = total_quantity
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=size,
                avg_price=actual_price,
                current_price=actual_price,
                entry_time=signal['timestamp']
            )
        
        # Update cash
        self.cash -= total_cost
        
        # Record trade
        self.trades.append(trade)
        
        # Trigger callbacks
        for callback in self.on_trade_callbacks:
            await callback(trade)
    
    async def _execute_sell(self, symbol: str, size: float, price: float, signal: Dict[str, Any]):
        """Execute sell order"""
        if not price or size <= 0:
            return
        
        # Check if we have position
        if symbol not in self.positions:
            if not self.config.allow_shorting:
                return
        
        position = self.positions.get(symbol)
        if position:
            # Limit size to position
            size = min(size, position.quantity)
        
        # Calculate proceeds
        commission = size * price * self.config.commission
        slippage = price * self.config.slippage
        actual_price = price - slippage
        proceeds = size * actual_price - commission
        
        # Create trade
        self._trade_counter += 1
        trade = Trade(
            trade_id=f"T{self._trade_counter:06d}",
            timestamp=signal['timestamp'],
            symbol=symbol,
            side='sell',
            quantity=size,
            price=actual_price,
            order_type=OrderType.MARKET,
            commission=commission,
            slippage_cost=slippage * size,
            signal_source=signal['agent'],
            signal_strength=signal['confidence'],
            exit_price=actual_price
        )
        
        # Calculate P&L if closing position
        if position:
            trade.pnl = (actual_price - position.avg_price) * size - commission
            trade.pnl_percent = (trade.pnl / (position.avg_price * size)) * 100
            
            # Update position
            position.quantity -= size
            position.realized_pnl += trade.pnl
            
            if position.quantity <= 0:
                del self.positions[symbol]
        
        # Update cash
        self.cash += proceeds
        
        # Record trade
        self.trades.append(trade)
        
        # Trigger callbacks
        for callback in self.on_trade_callbacks:
            await callback(trade)
    
    def _calculate_position_size(self, symbol: str, confidence: float, agent_weight: float) -> float:
        """Calculate position size using Kelly Criterion or fixed sizing"""
        
        if self.config.use_kelly_criterion:
            # Kelly Criterion sizing
            win_rate = 0.55  # Estimate from historical performance
            avg_win_loss_ratio = 1.5  # Estimate
            
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_value = self.cash * kelly_fraction * confidence * agent_weight
        else:
            # Fixed percentage sizing
            position_value = self.cash * self.config.max_position_size * confidence * agent_weight
        
        # Convert to shares (simplified - would need current price)
        return position_value / 100  # Placeholder
    
    def _update_positions(self, current_prices: Dict[str, float]):
        """Update position prices"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol])
    
    def _check_exits(self, current_prices: Dict[str, float]):
        """Check stop loss and take profit levels"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Check stop loss
            if self.config.stop_loss:
                stop_price = position.avg_price * (1 - self.config.stop_loss)
                if current_price <= stop_price:
                    positions_to_close.append((symbol, 'stop_loss'))
            
            # Check take profit
            if self.config.take_profit:
                target_price = position.avg_price * (1 + self.config.take_profit)
                if current_price >= target_price:
                    positions_to_close.append((symbol, 'take_profit'))
        
        # Close positions
        for symbol, reason in positions_to_close:
            # Create exit signal
            exit_signal = {
                'timestamp': datetime.now(),
                'agent': f'risk_management_{reason}',
                'symbol': symbol,
                'action': 'SELL',
                'confidence': 1.0,
                'metadata': {'reason': reason}
            }
            
            # Execute sell
            asyncio.create_task(
                self._execute_sell(
                    symbol,
                    self.positions[symbol].quantity,
                    current_prices[symbol],
                    exit_signal
                )
            )
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            position.quantity * current_prices.get(symbol, position.current_price)
            for symbol, position in self.positions.items()
        )
        return self.cash + positions_value
    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics"""
        if not self.equity_curve:
            return
        
        # Convert to pandas Series for easier calculation
        equity_df = pd.DataFrame(self.equity_curve)
        equity_series = pd.Series(
            equity_df['value'].values,
            index=pd.to_datetime(equity_df['timestamp'])
        )
        
        # Calculate metrics
        self.metrics.calculate_metrics(equity_series, self.trades)
        
        # Add costs
        self.metrics.total_commission = sum(t.commission for t in self.trades)
        self.metrics.total_slippage = sum(t.slippage_cost for t in self.trades)
    
    async def _load_agents(self):
        """Load and initialize agents for backtesting"""
        # This would load actual agent instances
        # For now, using placeholder
        logger.info(f"Loading agents: {self.config.agents_to_test}")
    
    async def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch historical market data for backtesting"""
        return await market_data_aggregator.fetch_data(
            symbols=self.config.symbols,
            timeframe=self.config.timeframe,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            use_cache=True
        )
    
    async def _optimize_agents(self, data: pd.DataFrame):
        """Optimize agent parameters on historical data"""
        # This would implement parameter optimization
        logger.debug("Optimizing agent parameters...")
    
    def _add_noise_to_data(self, data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """Add random noise to data for Monte Carlo simulation"""
        noisy_data = data.copy()
        
        for col in ['open', 'high', 'low', 'close']:
            if col in noisy_data.columns:
                noise = np.random.normal(0, noise_level, len(noisy_data))
                noisy_data[col] *= (1 + noise)
        
        return noisy_data
    
    def _randomize_parameters(self):
        """Randomize parameters for Monte Carlo simulation"""
        # Vary commission and slippage
        self.config.commission *= np.random.uniform(0.8, 1.2)
        self.config.slippage *= np.random.uniform(0.8, 1.2)
    
    def _calculate_monte_carlo_statistics(self, all_metrics: List[Dict[str, Any]]):
        """Calculate statistics across Monte Carlo simulations"""
        df = pd.DataFrame(all_metrics)
        
        # Calculate percentiles
        stats = {
            'mean_return': df['total_return'].mean(),
            'std_return': df['total_return'].std(),
            'percentile_5': df['total_return'].quantile(0.05),
            'percentile_95': df['total_return'].quantile(0.95),
            'mean_sharpe': df['sharpe_ratio'].mean(),
            'mean_max_dd': df['max_drawdown'].mean()
        }
        
        logger.info(f"Monte Carlo results: {stats}")
    
    def _reset(self):
        """Reset engine state"""
        self.positions.clear()
        self.trades.clear()
        self.signals.clear()
        self.cash = self.config.initial_capital
        self.equity_curve.clear()
        self.metrics = BacktestMetrics()
    
    async def _store_backtest_results(self):
        """Store backtest results in RAG for learning"""
        try:
            result_doc = {
                "type": "backtest_result",
                "config": self.config.to_dict(),
                "metrics": self.metrics.to_dict(),
                "trade_count": len(self.trades),
                "signal_count": len(self.signals),
                "timestamp": datetime.now().isoformat()
            }
            
            await rag_engine.add_document(result_doc)
            
        except Exception as e:
            logger.debug(f"Failed to store backtest results: {str(e)}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive backtest results"""
        return {
            "metrics": self.metrics.to_dict(),
            "trades": [t.to_dict() for t in self.trades[-100:]],  # Last 100 trades
            "equity_curve": self.equity_curve,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "config": self.config.to_dict()
        }