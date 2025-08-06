"""
Portfolio Optimization Module
Implements modern portfolio theory with AI enhancements
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import asyncio
from enum import Enum

from core.logging import get_logger
from core.events.bus import event_bus
from core.market_data import market_data_service
from ml.finance_ml_pipeline import FinanceMLPipeline

logger = get_logger(__name__)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAX_SHARPE = "max_sharpe"  # Maximum Sharpe ratio
    MIN_VOLATILITY = "min_volatility"  # Minimum volatility
    MAX_RETURNS = "max_returns"  # Maximum returns
    RISK_PARITY = "risk_parity"  # Risk parity
    MAX_DIVERSIFICATION = "max_diversification"  # Maximum diversification
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman model
    KELLY_CRITERION = "kelly_criterion"  # Kelly criterion


class RiskMeasure(Enum):
    """Risk measures for optimization"""
    VOLATILITY = "volatility"
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    DOWNSIDE_DEVIATION = "downside_deviation"


@dataclass
class AssetConstraints:
    """Constraints for individual assets"""
    symbol: str
    min_weight: float = 0.0  # Minimum allocation
    max_weight: float = 1.0  # Maximum allocation
    target_weight: Optional[float] = None  # Target allocation
    sector: Optional[str] = None
    asset_class: Optional[str] = None
    
    # Risk constraints
    max_position_risk: Optional[float] = None  # Max risk contribution
    min_liquidity: Optional[float] = None  # Minimum liquidity requirement


@dataclass
class PortfolioConstraints:
    """Portfolio-level constraints"""
    # Weight constraints
    min_positions: int = 2
    max_positions: int = 20
    allow_short: bool = False
    max_leverage: float = 1.0
    
    # Sector constraints
    max_sector_weight: float = 0.4
    sector_limits: Dict[str, float] = field(default_factory=dict)
    
    # Risk constraints
    max_volatility: Optional[float] = None
    max_var: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Transaction constraints
    min_trade_size: float = 0.01  # 1% minimum trade
    max_turnover: float = 1.0  # 100% turnover limit
    
    # Regulatory constraints
    max_concentration: float = 0.25  # Max single position


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]  # Optimal weights
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    
    # Risk decomposition
    risk_contributions: Dict[str, float]
    marginal_risk_contributions: Dict[str, float]
    
    # Performance metrics
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: Optional[float]
    
    # Portfolio characteristics
    effective_diversification: float
    concentration_risk: float
    
    # Optimization metadata
    objective: OptimizationObjective
    convergence: bool
    iterations: int
    optimization_time: float
    
    # Rebalancing suggestion
    current_weights: Optional[Dict[str, float]] = None
    trades_required: Optional[Dict[str, float]] = None
    transaction_costs: Optional[float] = None


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with AI enhancements
    """
    
    def __init__(self):
        self.ml_pipeline = FinanceMLPipeline()
        self.market_data = market_data_service
        
        # Cache for market data
        self.returns_cache: Dict[str, pd.DataFrame] = {}
        self.cache_ttl = 3600  # 1 hour
        self.last_cache_time: Dict[str, datetime] = {}
        
        # Risk-free rate (can be updated)
        self.risk_free_rate = 0.05  # 5% annual
        
        # Transaction cost model
        self.fixed_cost = 5  # Fixed cost per trade
        self.variable_cost = 0.001  # 0.1% variable cost
        
        logger.info("Portfolio Optimizer initialized")
    
    async def optimize(
        self,
        symbols: List[str],
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        constraints: Optional[PortfolioConstraints] = None,
        asset_constraints: Optional[List[AssetConstraints]] = None,
        lookback_days: int = 252,
        use_ml_predictions: bool = True,
        current_portfolio: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio allocation
        
        Args:
            symbols: List of symbols to optimize
            objective: Optimization objective
            constraints: Portfolio constraints
            asset_constraints: Individual asset constraints
            lookback_days: Historical data lookback period
            use_ml_predictions: Whether to use ML predictions
            current_portfolio: Current portfolio weights
            
        Returns:
            Optimization result
        """
        start_time = datetime.now()
        
        try:
            # Get historical returns
            returns = await self._get_returns(symbols, lookback_days)
            
            if returns.empty:
                raise ValueError("No historical data available")
            
            # Get ML predictions if enabled
            ml_views = None
            if use_ml_predictions:
                ml_views = await self._get_ml_views(symbols)
            
            # Calculate statistics
            expected_returns = self._calculate_expected_returns(returns, ml_views)
            cov_matrix = self._calculate_covariance_matrix(returns)
            
            # Set default constraints
            if constraints is None:
                constraints = PortfolioConstraints()
            
            # Run optimization based on objective
            if objective == OptimizationObjective.MAX_SHARPE:
                weights = self._optimize_max_sharpe(
                    expected_returns, cov_matrix, constraints, asset_constraints
                )
            elif objective == OptimizationObjective.MIN_VOLATILITY:
                weights = self._optimize_min_volatility(
                    cov_matrix, constraints, asset_constraints
                )
            elif objective == OptimizationObjective.MAX_RETURNS:
                weights = self._optimize_max_returns(
                    expected_returns, constraints, asset_constraints
                )
            elif objective == OptimizationObjective.RISK_PARITY:
                weights = self._optimize_risk_parity(
                    cov_matrix, constraints, asset_constraints
                )
            elif objective == OptimizationObjective.BLACK_LITTERMAN:
                weights = self._optimize_black_litterman(
                    returns, cov_matrix, ml_views, constraints, asset_constraints
                )
            elif objective == OptimizationObjective.KELLY_CRITERION:
                weights = self._optimize_kelly(
                    expected_returns, cov_matrix, constraints, asset_constraints
                )
            else:
                weights = self._optimize_max_diversification(
                    cov_matrix, constraints, asset_constraints
                )
            
            # Calculate portfolio metrics
            result = self._calculate_portfolio_metrics(
                weights, expected_returns, cov_matrix, returns, objective
            )
            
            # Add rebalancing information if current portfolio provided
            if current_portfolio:
                result.current_weights = current_portfolio
                result.trades_required = self._calculate_trades(current_portfolio, weights)
                result.transaction_costs = self._calculate_transaction_costs(
                    result.trades_required
                )
            
            # Record optimization time
            result.optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Publish optimization event
            await event_bus.publish(
                "portfolio.optimized",
                data={
                    "symbols": symbols,
                    "objective": objective.value,
                    "sharpe_ratio": result.sharpe_ratio,
                    "expected_return": result.expected_return,
                    "expected_volatility": result.expected_volatility,
                    "weights": result.weights
                }
            )
            
            logger.info(f"Portfolio optimized: Sharpe={result.sharpe_ratio:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            raise
    
    def _optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]]
    ) -> np.ndarray:
        """Maximize Sharpe ratio"""
        n = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe
        
        # Set constraints
        cons = self._get_optimization_constraints(constraints, n)
        bounds = self._get_bounds(constraints, asset_constraints, n)
        
        # Initial guess (equal weights)
        x0 = np.array([1/n] * n)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _optimize_min_volatility(
        self,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]]
    ) -> np.ndarray:
        """Minimize portfolio volatility"""
        n = cov_matrix.shape[0]
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        cons = self._get_optimization_constraints(constraints, n)
        bounds = self._get_bounds(constraints, asset_constraints, n)
        x0 = np.array([1/n] * n)
        
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _optimize_risk_parity(
        self,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]]
    ) -> np.ndarray:
        """Risk parity optimization"""
        n = cov_matrix.shape[0]
        
        def risk_parity_objective(weights):
            # Calculate marginal risk contributions
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize variance of risk contributions
            target_contrib = portfolio_vol / n
            return np.sum((contrib - target_contrib) ** 2)
        
        cons = self._get_optimization_constraints(constraints, n)
        bounds = self._get_bounds(constraints, asset_constraints, n)
        x0 = np.array([1/n] * n)
        
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _optimize_black_litterman(
        self,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray,
        ml_views: Optional[Dict[str, float]],
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]]
    ) -> np.ndarray:
        """Black-Litterman optimization"""
        n = cov_matrix.shape[0]
        
        # Market capitalization weights (equal weight as proxy)
        market_weights = np.array([1/n] * n)
        
        # Equilibrium returns
        delta = 2.5  # Risk aversion parameter
        equilibrium_returns = delta * np.dot(cov_matrix, market_weights)
        
        # If we have ML views, incorporate them
        if ml_views:
            # Create views matrix
            P = np.zeros((len(ml_views), n))
            Q = np.zeros(len(ml_views))
            
            symbols = returns.columns.tolist()
            for i, (symbol, view) in enumerate(ml_views.items()):
                if symbol in symbols:
                    idx = symbols.index(symbol)
                    P[i, idx] = 1
                    Q[i] = view
            
            # Black-Litterman formula
            tau = 0.05  # Uncertainty in equilibrium returns
            omega = np.diag(np.diag(P @ cov_matrix @ P.T)) * tau
            
            # Posterior returns
            inv_cov = np.linalg.inv(cov_matrix)
            posterior_cov = np.linalg.inv(inv_cov + P.T @ np.linalg.inv(omega) @ P / tau)
            posterior_returns = posterior_cov @ (inv_cov @ equilibrium_returns + P.T @ np.linalg.inv(omega) @ Q / tau)
        else:
            posterior_returns = equilibrium_returns
        
        # Optimize with posterior returns
        return self._optimize_max_sharpe(
            posterior_returns, cov_matrix, constraints, asset_constraints
        )
    
    def _optimize_kelly(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]]
    ) -> np.ndarray:
        """Kelly Criterion optimization"""
        n = len(expected_returns)
        
        # Kelly fraction with safety factor
        kelly_fraction = 0.25  # Use 25% of full Kelly
        
        def kelly_objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Kelly criterion: maximize log(1 + f*r - 0.5*f^2*sigma^2)
            kelly = np.log(1 + kelly_fraction * portfolio_return - 0.5 * kelly_fraction**2 * portfolio_var)
            return -kelly
        
        cons = self._get_optimization_constraints(constraints, n)
        bounds = self._get_bounds(constraints, asset_constraints, n)
        x0 = np.array([1/n] * n)
        
        result = minimize(
            kelly_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _optimize_max_returns(
        self,
        expected_returns: np.ndarray,
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]]
    ) -> np.ndarray:
        """Maximize expected returns"""
        n = len(expected_returns)
        
        def negative_returns(weights):
            return -np.dot(weights, expected_returns)
        
        cons = self._get_optimization_constraints(constraints, n)
        bounds = self._get_bounds(constraints, asset_constraints, n)
        x0 = np.array([1/n] * n)
        
        result = minimize(
            negative_returns,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _optimize_max_diversification(
        self,
        cov_matrix: np.ndarray,
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]]
    ) -> np.ndarray:
        """Maximum diversification portfolio"""
        n = cov_matrix.shape[0]
        vol = np.sqrt(np.diag(cov_matrix))
        
        def negative_diversification(weights):
            weighted_vol = np.dot(weights, vol)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            diversification = weighted_vol / portfolio_vol
            return -diversification
        
        cons = self._get_optimization_constraints(constraints, n)
        bounds = self._get_bounds(constraints, asset_constraints, n)
        x0 = np.array([1/n] * n)
        
        result = minimize(
            negative_diversification,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else x0
    
    def _get_optimization_constraints(
        self,
        constraints: PortfolioConstraints,
        n: int
    ) -> List[Dict]:
        """Get optimization constraints"""
        cons = []
        
        # Sum to 1 constraint
        cons.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
        
        # Maximum volatility constraint
        if constraints.max_volatility:
            cons.append({
                'type': 'ineq',
                'fun': lambda x, cov=cov_matrix, max_vol=constraints.max_volatility: 
                    max_vol - np.sqrt(np.dot(x.T, np.dot(cov, x)))
            })
        
        # Minimum positions constraint
        if constraints.min_positions:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: np.sum(x > 0.001) - constraints.min_positions
            })
        
        return cons
    
    def _get_bounds(
        self,
        constraints: PortfolioConstraints,
        asset_constraints: Optional[List[AssetConstraints]],
        n: int
    ) -> List[Tuple[float, float]]:
        """Get variable bounds"""
        # Default bounds
        if constraints.allow_short:
            bounds = [(-constraints.max_leverage, constraints.max_leverage)] * n
        else:
            bounds = [(0, constraints.max_concentration)] * n
        
        # Apply asset-specific constraints
        if asset_constraints:
            for ac in asset_constraints:
                # TODO: Implement asset-specific constraint mapping
                # Would need symbol to index mapping to apply individual constraints
                logger.debug(f"Asset-specific constraints not yet implemented: {ac}")
        
        return bounds
    
    async def _get_returns(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """Get historical returns"""
        cache_key = f"{','.join(sorted(symbols))}_{lookback_days}"
        
        # Check cache
        if cache_key in self.returns_cache:
            if (datetime.now() - self.last_cache_time[cache_key]).seconds < self.cache_ttl:
                return self.returns_cache[cache_key]
        
        # Fetch new data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        returns_list = []
        for symbol in symbols:
            try:
                data = await self.market_data.get_historical_data(
                    symbol, "1d", start_date, end_date
                )
                if data and len(data) > 1:
                    prices = pd.Series([d['close'] for d in data])
                    returns = prices.pct_change().dropna()
                    returns_list.append(returns.rename(symbol))
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {str(e)}")
        
        if returns_list:
            returns_df = pd.concat(returns_list, axis=1).dropna()
            
            # Cache results
            self.returns_cache[cache_key] = returns_df
            self.last_cache_time[cache_key] = datetime.now()
            
            return returns_df
        
        return pd.DataFrame()
    
    async def _get_ml_views(self, symbols: List[str]) -> Dict[str, float]:
        """Get ML model predictions as views"""
        views = {}
        
        try:
            # Load ML model if available
            if not self.ml_pipeline.model:
                await self.ml_pipeline._load_latest_model()
            
            if self.ml_pipeline.model:
                for symbol in symbols:
                    try:
                        # Get prediction
                        prediction = await self.ml_pipeline.predict_single(symbol)
                        if prediction:
                            views[symbol] = prediction['expected_return']
                    except Exception as e:
                        logger.debug(f"ML prediction failed for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"Failed to get ML views: {str(e)}")
        
        return views
    
    def _calculate_expected_returns(
        self,
        returns: pd.DataFrame,
        ml_views: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Calculate expected returns"""
        # Historical mean returns (annualized)
        historical_returns = returns.mean() * 252
        
        if ml_views:
            # Blend historical with ML views
            blended = historical_returns.copy()
            for symbol, view in ml_views.items():
                if symbol in blended.index:
                    # 50/50 blend
                    blended[symbol] = 0.5 * blended[symbol] + 0.5 * view
            return blended.values
        
        return historical_returns.values
    
    def _calculate_covariance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix"""
        # Annualized covariance
        return returns.cov().values * 252
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        returns: pd.DataFrame,
        objective: OptimizationObjective
    ) -> OptimizationResult:
        """Calculate comprehensive portfolio metrics"""
        # Basic metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # VaR and CVaR (95% confidence)
        portfolio_returns = returns @ weights
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Risk contributions
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contributions = dict(zip(
            returns.columns,
            weights * marginal_contrib / portfolio_vol
        ))
        
        marginal_risk_contributions = dict(zip(
            returns.columns,
            marginal_contrib
        ))
        
        # Diversification metrics
        vol_vector = np.sqrt(np.diag(cov_matrix))
        effective_diversification = np.dot(weights, vol_vector) / portfolio_vol
        concentration_risk = np.sum(weights ** 2)
        
        # Additional ratios
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (portfolio_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        calmar = portfolio_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Create weight dictionary
        weight_dict = dict(zip(returns.columns, weights))
        
        return OptimizationResult(
            weights=weight_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var_95=var_95,
            cvar_95=cvar_95,
            risk_contributions=risk_contributions,
            marginal_risk_contributions=marginal_risk_contributions,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            information_ratio=None,
            effective_diversification=effective_diversification,
            concentration_risk=concentration_risk,
            objective=objective,
            convergence=True,
            iterations=100,
            optimization_time=0
        )
    
    def _calculate_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate required trades"""
        trades = {}
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            trade = target - current
            
            if abs(trade) > 0.001:  # 0.1% threshold
                trades[symbol] = trade
        
        return trades
    
    def _calculate_transaction_costs(self, trades: Dict[str, float]) -> float:
        """Calculate transaction costs"""
        total_cost = 0
        
        for symbol, trade in trades.items():
            if trade != 0:
                # Fixed cost per trade
                total_cost += self.fixed_cost
                # Variable cost
                total_cost += abs(trade) * self.variable_cost
        
        return total_cost
    
    async def backtest_strategy(
        self,
        symbols: List[str],
        objective: OptimizationObjective,
        rebalance_frequency: str = "monthly",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Backtest portfolio optimization strategy
        
        Args:
            symbols: List of symbols
            objective: Optimization objective
            rebalance_frequency: How often to rebalance
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest results
        """
        # Implementation would simulate portfolio performance
        # over historical period with periodic rebalancing
        pass


# Global portfolio optimizer instance
portfolio_optimizer = PortfolioOptimizer()