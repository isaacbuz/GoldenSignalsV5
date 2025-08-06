"""
Risk Management Agent V5
Comprehensive risk analytics, portfolio monitoring, and position management
Enhanced from archive with V5 architecture integration
"""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
from scipy import stats

from core.logging import get_logger

logger = get_logger(__name__)


class RiskMetric(Enum):
    """Available risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration_risk"


class PortfolioType(Enum):
    """Portfolio types"""
    EQUITY = "equity"
    OPTIONS = "options"
    MIXED = "mixed"
    CRYPTO = "crypto"
    FUTURES = "futures"


class AlertLevel(Enum):
    """Risk alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskLevel(Enum):
    """Overall risk levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class Position:
    """Represents a portfolio position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    position_type: str  # long, short, call, put
    asset_class: str = "equity"
    sector: str = ""
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def notional_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        if self.position_type == "long":
            return (self.current_price - self.entry_price) * self.quantity
        elif self.position_type == "short":
            return (self.entry_price - self.current_price) * abs(self.quantity)
        return 0
    
    @property
    def pnl_percent(self) -> float:
        if self.entry_price == 0:
            return 0
        return (self.pnl / (self.entry_price * abs(self.quantity))) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'market_value': self.market_value,
            'notional_value': self.notional_value,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent
        }


@dataclass
class Portfolio:
    """Portfolio structure"""
    id: str
    name: str
    positions: List[Position]
    cash: float
    portfolio_type: PortfolioType
    created_at: datetime
    benchmark: str = "SPY"
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(pos.market_value for pos in self.positions)
    
    @property
    def total_pnl(self) -> float:
        return sum(pos.pnl for pos in self.positions)
    
    @property
    def gross_exposure(self) -> float:
        return sum(pos.market_value for pos in self.positions)
    
    @property
    def net_exposure(self) -> float:
        return sum(pos.notional_value for pos in self.positions)
    
    @property
    def leverage(self) -> float:
        return self.gross_exposure / self.total_value if self.total_value > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'total_value': self.total_value,
            'total_pnl': self.total_pnl,
            'gross_exposure': self.gross_exposure,
            'net_exposure': self.net_exposure,
            'leverage': self.leverage,
            'position_count': len(self.positions),
            'cash': self.cash,
            'portfolio_type': self.portfolio_type.value,
            'benchmark': self.benchmark,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class RiskAlert:
    """Risk alert structure"""
    id: str
    portfolio_id: str
    metric: RiskMetric
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'metric': self.metric.value,
            'level': self.level.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    portfolio_id: str
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100
    metrics: Dict[str, Any]
    alerts: List[RiskAlert]
    recommendations: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'portfolio_id': self.portfolio_id,
            'overall_risk_level': self.overall_risk_level.value,
            'risk_score': self.risk_score,
            'metrics': self.metrics,
            'alerts': [alert.to_dict() for alert in self.alerts],
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class RiskManagementAgent:
    """
    V5 Risk Management Agent
    Comprehensive portfolio risk analytics and monitoring
    """
    
    def __init__(self):
        """Initialize the risk management agent"""
        # Portfolio storage
        self.portfolios: Dict[str, Portfolio] = {}
        
        # Historical data cache
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.returns_cache: Dict[str, np.ndarray] = {}
        
        # Risk alerts
        self.alerts: deque = deque(maxlen=1000)
        self.alert_thresholds = self._default_thresholds()
        
        # Performance tracking
        self.calculation_times: Dict[str, List[float]] = defaultdict(list)
        self.risk_history: deque = deque(maxlen=100)
        
        # Circuit breakers
        self.circuit_breakers = {
            'max_drawdown': 0.20,  # 20% max drawdown
            'daily_loss': 0.05,    # 5% daily loss limit
            'position_concentration': 0.25,  # 25% max single position
            'leverage': 3.0        # 3x max leverage
        }
        
        # Risk-free rate (can be updated)
        self.risk_free_rate = 0.02  # 2%
        
        logger.info("Risk Management Agent V5 initialized")
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default risk alert thresholds"""
        return {
            RiskMetric.VAR.value: {
                AlertLevel.WARNING.value: 0.05,   # 5% VaR
                AlertLevel.CRITICAL.value: 0.10,  # 10% VaR
                AlertLevel.EMERGENCY.value: 0.20  # 20% VaR
            },
            RiskMetric.MAX_DRAWDOWN.value: {
                AlertLevel.WARNING.value: 0.10,
                AlertLevel.CRITICAL.value: 0.20,
                AlertLevel.EMERGENCY.value: 0.30
            },
            RiskMetric.VOLATILITY.value: {
                AlertLevel.WARNING.value: 0.25,   # 25% annualized vol
                AlertLevel.CRITICAL.value: 0.40,
                AlertLevel.EMERGENCY.value: 0.60
            },
            RiskMetric.CONCENTRATION.value: {
                AlertLevel.WARNING.value: 0.20,
                AlertLevel.CRITICAL.value: 0.30,
                AlertLevel.EMERGENCY.value: 0.40
            }
        }
    
    async def create_portfolio(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a portfolio"""
        try:
            portfolio_id = portfolio_data.get('id', str(uuid.uuid4()))
            
            positions = []
            for pos_data in portfolio_data.get('positions', []):
                positions.append(Position(
                    symbol=pos_data.get('symbol'),
                    quantity=float(pos_data.get('quantity', 0)),
                    entry_price=float(pos_data.get('entry_price', 0)),
                    current_price=float(pos_data.get('current_price', 0)),
                    position_type=pos_data.get('position_type', 'long'),
                    asset_class=pos_data.get('asset_class', 'equity'),
                    sector=pos_data.get('sector', '')
                ))
            
            portfolio = Portfolio(
                id=portfolio_id,
                name=portfolio_data.get('name', f'Portfolio {portfolio_id[:8]}'),
                positions=positions,
                cash=float(portfolio_data.get('cash', 0)),
                portfolio_type=PortfolioType(portfolio_data.get('type', 'mixed')),
                created_at=datetime.now(),
                benchmark=portfolio_data.get('benchmark', 'SPY')
            )
            
            self.portfolios[portfolio_id] = portfolio
            
            logger.info(f"Created portfolio {portfolio_id} with {len(positions)} positions")
            
            return {
                "portfolio_id": portfolio_id,
                "status": "created",
                **portfolio.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {str(e)}")
            raise
    
    async def calculate_var(self, portfolio_id: str, confidence_level: float = 0.95,
                          time_horizon: int = 1, method: str = 'historical') -> Dict[str, Any]:
        """Calculate Value at Risk"""
        try:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Get portfolio returns
            returns = await self._get_portfolio_returns(portfolio, lookback_days=252)
            
            if len(returns) < 30:
                raise ValueError("Insufficient data for VaR calculation")
            
            # Scale returns to time horizon
            scaled_returns = returns * np.sqrt(time_horizon)
            
            # Calculate VaR based on method
            if method == 'historical':
                var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                mean = np.mean(scaled_returns)
                std = np.std(scaled_returns)
                var = mean + std * stats.norm.ppf(1 - confidence_level)
            elif method == 'monte_carlo':
                var = await self._monte_carlo_var(returns, confidence_level, time_horizon)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            # Calculate CVaR (Expected Shortfall)
            cvar = np.mean(scaled_returns[scaled_returns <= var])
            
            # Check for alerts
            await self._check_var_alert(portfolio_id, abs(var), confidence_level)
            
            result = {
                "portfolio_id": portfolio_id,
                "var": abs(var),
                "cvar": abs(cvar),
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon,
                "method": method,
                "portfolio_value": portfolio.total_value,
                "var_amount": abs(var) * portfolio.total_value,
                "var_percent": abs(var) * 100
            }
            
            logger.info(f"Calculated VaR for {portfolio_id}: {abs(var)*100:.2f}% ({method})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating VaR for {portfolio_id}: {str(e)}")
            raise
    
    async def assess_portfolio_risk(self, portfolio_id: str) -> RiskAssessment:
        """Comprehensive portfolio risk assessment"""
        try:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Calculate all risk metrics
            returns = await self._get_portfolio_returns(portfolio)
            
            metrics = {
                "total_value": portfolio.total_value,
                "total_pnl": portfolio.total_pnl,
                "return_percent": (portfolio.total_pnl / portfolio.total_value) * 100 if portfolio.total_value > 0 else 0,
                "leverage": portfolio.leverage,
                "gross_exposure": portfolio.gross_exposure,
                "net_exposure": portfolio.net_exposure,
                "volatility": self._calculate_volatility(returns),
                "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                "sortino_ratio": self._calculate_sortino_ratio(returns),
                "max_drawdown": self._calculate_max_drawdown(returns),
                "concentration_risk": self._calculate_concentration_risk(portfolio),
                "largest_position": self._get_largest_position(portfolio),
                "sector_exposure": self._calculate_sector_exposure(portfolio),
                "position_count": len(portfolio.positions)
            }
            
            # Calculate VaR
            var_result = await self.calculate_var(portfolio_id, confidence_level=0.95)
            metrics["var_95"] = var_result["var"]
            metrics["cvar_95"] = var_result["cvar"]
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(metrics)
            
            # Determine risk level
            overall_risk_level = self._determine_risk_level(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(portfolio, metrics)
            
            # Get relevant alerts
            relevant_alerts = [alert for alert in self.alerts if alert.portfolio_id == portfolio_id][-10:]
            
            assessment = RiskAssessment(
                portfolio_id=portfolio_id,
                overall_risk_level=overall_risk_level,
                risk_score=risk_score,
                metrics=metrics,
                alerts=relevant_alerts,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.risk_history.append(assessment)
            
            logger.info(f"Risk assessment for {portfolio_id}: {overall_risk_level.value} ({risk_score:.1f}/100)")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk {portfolio_id}: {str(e)}")
            raise
    
    async def stress_test(self, portfolio_id: str, scenarios: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run stress test scenarios"""
        try:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            # Default scenarios if none provided
            if not scenarios:
                scenarios = [
                    {"name": "Market Crash", "market_shock": -0.20, "vol_shock": 2.0, "correlation_shock": 0.8},
                    {"name": "Flash Crash", "market_shock": -0.10, "vol_shock": 3.0, "correlation_shock": 0.9},
                    {"name": "Interest Rate Spike", "market_shock": -0.05, "vol_shock": 1.5, "correlation_shock": 0.6},
                    {"name": "Black Swan", "market_shock": -0.30, "vol_shock": 4.0, "correlation_shock": 0.95},
                    {"name": "Sector Rotation", "market_shock": 0.0, "vol_shock": 1.8, "correlation_shock": 0.3},
                    {"name": "Liquidity Crisis", "market_shock": -0.15, "vol_shock": 2.5, "correlation_shock": 0.85}
                ]
            
            results = []
            
            for scenario in scenarios:
                scenario_result = await self._simulate_scenario(portfolio, scenario)
                results.append(scenario_result)
            
            # Find worst case
            worst_case = min(results, key=lambda x: x['portfolio_value'])
            best_case = max(results, key=lambda x: x['portfolio_value'])
            
            # Calculate survival metrics
            survival_probability = self._calculate_survival_probability(results)
            
            stress_result = {
                "portfolio_id": portfolio_id,
                "current_value": portfolio.total_value,
                "scenarios": results,
                "worst_case": worst_case,
                "best_case": best_case,
                "survival_probability": survival_probability,
                "avg_loss": np.mean([r['loss_pct'] for r in results]),
                "stress_summary": {
                    "scenarios_tested": len(scenarios),
                    "scenarios_survived": sum(1 for r in results if r['survival']),
                    "max_loss_scenario": worst_case['scenario'],
                    "max_loss_percent": worst_case['loss_pct']
                }
            }
            
            logger.info(f"Stress test for {portfolio_id}: {len(scenarios)} scenarios, {survival_probability:.1%} survival rate")
            return stress_result
            
        except Exception as e:
            logger.error(f"Error running stress test for {portfolio_id}: {str(e)}")
            raise
    
    async def check_circuit_breakers(self, portfolio_id: str) -> Dict[str, Any]:
        """Check if portfolio breaches circuit breaker thresholds"""
        try:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            breaches = []
            
            # Check leverage
            if portfolio.leverage > self.circuit_breakers['leverage']:
                breaches.append({
                    'type': 'leverage',
                    'current': portfolio.leverage,
                    'threshold': self.circuit_breakers['leverage'],
                    'severity': 'critical'
                })
            
            # Check concentration
            concentration = self._calculate_concentration_risk(portfolio)
            if concentration > self.circuit_breakers['position_concentration']:
                breaches.append({
                    'type': 'concentration',
                    'current': concentration,
                    'threshold': self.circuit_breakers['position_concentration'],
                    'severity': 'warning'
                })
            
            # Check drawdown
            returns = await self._get_portfolio_returns(portfolio, lookback_days=20)
            if len(returns) > 0:
                drawdown = self._calculate_max_drawdown(returns)
                if drawdown > self.circuit_breakers['max_drawdown']:
                    breaches.append({
                        'type': 'drawdown',
                        'current': drawdown,
                        'threshold': self.circuit_breakers['max_drawdown'],
                        'severity': 'emergency'
                    })
            
            # Check daily loss
            if len(returns) > 0:
                daily_return = returns[-1]
                if daily_return < -self.circuit_breakers['daily_loss']:
                    breaches.append({
                        'type': 'daily_loss',
                        'current': abs(daily_return),
                        'threshold': self.circuit_breakers['daily_loss'],
                        'severity': 'critical'
                    })
            
            return {
                'portfolio_id': portfolio_id,
                'breaches': breaches,
                'breach_count': len(breaches),
                'action_required': len(breaches) > 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers for {portfolio_id}: {str(e)}")
            raise
    
    async def analyze_position_risk(self, portfolio_id: str, symbol: str) -> Dict[str, Any]:
        """Analyze risk for a specific position"""
        try:
            portfolio = self.portfolios.get(portfolio_id)
            if not portfolio:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            position = next((p for p in portfolio.positions if p.symbol == symbol), None)
            if not position:
                raise ValueError(f"Position {symbol} not found in portfolio")
            
            # Get position returns
            returns = await self._get_symbol_returns(symbol)
            
            # Calculate position-specific metrics
            position_weight = position.market_value / portfolio.total_value if portfolio.total_value > 0 else 0
            position_var = np.percentile(returns, 5) * position.market_value
            volatility = np.std(returns) * np.sqrt(252)
            
            # Risk contribution to portfolio
            risk_contribution = position_weight * volatility
            
            # Estimate liquidation cost
            liquidation_cost = self._estimate_liquidation_cost(position)
            
            # Generate recommendations
            recommendations = self._get_position_recommendations(position, position_weight)
            
            return {
                "portfolio_id": portfolio_id,
                "symbol": symbol,
                "position_details": position.to_dict(),
                "risk_metrics": {
                    "position_weight": position_weight * 100,
                    "position_var_95": abs(position_var),
                    "volatility_annual": volatility * 100,
                    "risk_contribution": risk_contribution * 100,
                    "beta": await self._calculate_position_beta(symbol, portfolio.benchmark),
                    "liquidation_cost": liquidation_cost,
                    "sharpe_ratio": self._calculate_position_sharpe(returns) if len(returns) > 20 else 0
                },
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing position risk {symbol} in {portfolio_id}: {str(e)}")
            raise
    
    def get_alerts(self, portfolio_id: Optional[str] = None, level: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Get risk alerts"""
        alerts = list(self.alerts)
        
        # Filter by portfolio
        if portfolio_id:
            alerts = [a for a in alerts if a.portfolio_id == portfolio_id]
        
        # Filter by level
        if level:
            try:
                alert_level = AlertLevel(level)
                alerts = [a for a in alerts if a.level == alert_level]
            except ValueError:
                pass
        
        # Sort by timestamp descending
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return {
            "alerts": [a.to_dict() for a in alerts[:limit]],
            "total": len(alerts),
            "critical_count": len([a for a in alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]),
            "timestamp": datetime.now().isoformat()
        }
    
    def set_alert_threshold(self, metric: str, level: str, threshold: float) -> Dict[str, Any]:
        """Set custom alert threshold"""
        try:
            # Validate metric and level
            metric_enum = RiskMetric(metric)
            level_enum = AlertLevel(level)
            
            if metric_enum.value not in self.alert_thresholds:
                self.alert_thresholds[metric_enum.value] = {}
            
            self.alert_thresholds[metric_enum.value][level_enum.value] = threshold
            
            logger.info(f"Updated alert threshold: {metric} {level} = {threshold}")
            
            return {
                "metric": metric,
                "level": level,
                "threshold": threshold,
                "status": "updated",
                "timestamp": datetime.now().isoformat()
            }
            
        except ValueError as e:
            raise ValueError(f"Invalid metric or level: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        total_portfolios = len(self.portfolios)
        total_positions = sum(len(p.positions) for p in self.portfolios.values())
        total_alerts = len(self.alerts)
        critical_alerts = len([a for a in self.alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]])
        
        # Calculate average calculation times
        avg_calc_times = {}
        for metric, times in self.calculation_times.items():
            if times:
                avg_calc_times[f"{metric}_avg_ms"] = np.mean(times) * 1000
        
        return {
            "portfolios_managed": total_portfolios,
            "total_positions": total_positions,
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "risk_assessments": len(self.risk_history),
            "calculation_times": avg_calc_times,
            "circuit_breakers": self.circuit_breakers,
            "alert_thresholds": {
                metric: thresholds for metric, thresholds in self.alert_thresholds.items()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    # Helper methods (internal)
    
    async def _get_portfolio_returns(self, portfolio: Portfolio, lookback_days: int = 252) -> np.ndarray:
        """Get historical returns for portfolio"""
        # In production, this would fetch real historical data
        # For now, generate synthetic returns based on portfolio characteristics
        base_return = 0.0005  # 0.05% daily
        base_vol = 0.015      # 1.5% daily vol
        
        # Adjust based on portfolio leverage and composition
        vol_adjustment = 1.0 + (portfolio.leverage - 1.0) * 0.3
        
        returns = np.random.normal(base_return, base_vol * vol_adjustment, lookback_days)
        
        # Add some correlation and momentum effects
        for i in range(1, len(returns)):
            returns[i] += returns[i-1] * 0.05  # Small momentum effect
        
        return returns
    
    async def _get_symbol_returns(self, symbol: str, lookback_days: int = 252) -> np.ndarray:
        """Get historical returns for a symbol"""
        # Cache returns for efficiency
        cache_key = f"{symbol}_{lookback_days}"
        if cache_key in self.returns_cache:
            return self.returns_cache[cache_key]
        
        # In production, fetch real data
        # For now, generate synthetic returns based on symbol characteristics
        base_vol = {
            'AAPL': 0.02, 'MSFT': 0.018, 'GOOGL': 0.022, 'AMZN': 0.025,
            'TSLA': 0.035, 'NVDA': 0.03, 'SPY': 0.015, 'QQQ': 0.018
        }.get(symbol, 0.025)  # Default 2.5% daily vol
        
        returns = np.random.normal(0.0005, base_vol, lookback_days)
        
        # Cache the result
        self.returns_cache[cache_key] = returns
        return returns
    
    async def _monte_carlo_var(self, returns: np.ndarray, confidence_level: float,
                             time_horizon: int, simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Run simulations
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            simulations
        )
        
        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return var
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        return (np.mean(excess_returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.001
        return (np.mean(excess_returns) / downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_concentration_risk(self, portfolio: Portfolio) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        if not portfolio.positions or portfolio.total_value == 0:
            return 0
        
        weights = [pos.market_value / portfolio.total_value for pos in portfolio.positions]
        return sum(w**2 for w in weights)
    
    def _get_largest_position(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get largest position in portfolio"""
        if not portfolio.positions:
            return {"symbol": "None", "weight": 0}
        
        largest = max(portfolio.positions, key=lambda p: p.market_value)
        return {
            "symbol": largest.symbol,
            "weight": (largest.market_value / portfolio.total_value) * 100 if portfolio.total_value > 0 else 0,
            "value": largest.market_value
        }
    
    def _calculate_sector_exposure(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate sector exposure percentages"""
        if not portfolio.positions or portfolio.total_value == 0:
            return {}
        
        sector_values = defaultdict(float)
        for position in portfolio.positions:
            sector = position.sector or 'Unknown'
            sector_values[sector] += position.market_value
        
        return {
            sector: (value / portfolio.total_value) * 100 
            for sector, value in sector_values.items()
        }
    
    def _calculate_risk_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-100)"""
        score = 50  # Base score
        
        # Volatility impact
        vol = metrics.get('volatility', 0.15)
        if vol > 0.30:
            score += 20
        elif vol > 0.20:
            score += 10
        elif vol < 0.10:
            score -= 10
        
        # Leverage impact
        leverage = metrics.get('leverage', 1.0)
        if leverage > 2.0:
            score += 15
        elif leverage > 1.5:
            score += 8
        
        # Concentration impact
        concentration = metrics.get('concentration_risk', 0)
        if concentration > 0.3:
            score += 15
        elif concentration > 0.2:
            score += 8
        
        # Drawdown impact
        drawdown = metrics.get('max_drawdown', 0)
        if drawdown > 0.20:
            score += 20
        elif drawdown > 0.10:
            score += 10
        
        # VaR impact
        var = metrics.get('var_95', 0)
        if var > 0.15:
            score += 15
        elif var > 0.10:
            score += 8
        
        # Sharpe ratio benefit
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            score -= 10
        elif sharpe > 1.0:
            score -= 5
        elif sharpe < 0:
            score += 10
        
        return max(0, min(100, score))
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score >= 80:
            return RiskLevel.EXTREME
        elif risk_score >= 65:
            return RiskLevel.HIGH
        elif risk_score >= 45:
            return RiskLevel.MODERATE
        elif risk_score >= 25:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _generate_risk_recommendations(self, portfolio: Portfolio, metrics: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Leverage recommendations
        if metrics.get('leverage', 1.0) > 2.0:
            recommendations.append("Reduce leverage - consider deleveraging positions")
        
        # Concentration recommendations
        largest_pos = metrics.get('largest_position', {})
        if largest_pos.get('weight', 0) > 25:
            recommendations.append(f"Reduce concentration in {largest_pos.get('symbol', 'largest position')} (currently {largest_pos.get('weight', 0):.1f}%)")
        
        # Volatility recommendations
        if metrics.get('volatility', 0) > 0.30:
            recommendations.append("High portfolio volatility - consider adding defensive positions")
        
        # Drawdown recommendations
        if metrics.get('max_drawdown', 0) > 0.15:
            recommendations.append("Significant drawdown detected - review risk management strategy")
        
        # Sharpe ratio recommendations
        if metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Poor risk-adjusted returns - consider portfolio rebalancing")
        
        # VaR recommendations
        if metrics.get('var_95', 0) > 0.10:
            recommendations.append("High Value at Risk - consider hedging strategies")
        
        # Diversification recommendations
        if len(portfolio.positions) < 10 and portfolio.total_value > 100000:
            recommendations.append("Consider increasing diversification across more positions")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    async def _simulate_scenario(self, portfolio: Portfolio, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a stress scenario"""
        market_shock = scenario.get('market_shock', 0)
        vol_shock = scenario.get('vol_shock', 1)
        correlation_shock = scenario.get('correlation_shock', 0.5)
        
        # Base shock to portfolio
        shocked_value = portfolio.total_value * (1 + market_shock)
        
        # Additional volatility impact
        vol_impact = shocked_value * vol_shock * 0.05  # 5% of shocked value per vol unit
        
        # Correlation impact (higher correlation = more systematic risk)
        corr_impact = shocked_value * correlation_shock * 0.03
        
        final_value = shocked_value - vol_impact - corr_impact
        loss = portfolio.total_value - final_value
        
        return {
            "scenario": scenario['name'],
            "portfolio_value": final_value,
            "loss": loss,
            "loss_pct": (loss / portfolio.total_value) * 100 if portfolio.total_value > 0 else 0,
            "survival": final_value > portfolio.total_value * 0.5,  # 50% survival threshold
            "market_shock": market_shock * 100,
            "vol_shock": vol_shock,
            "correlation_shock": correlation_shock
        }
    
    def _calculate_survival_probability(self, stress_results: List[Dict[str, Any]]) -> float:
        """Calculate probability of surviving stress scenarios"""
        if not stress_results:
            return 1.0
        survivals = [r['survival'] for r in stress_results]
        return sum(survivals) / len(survivals)
    
    def _estimate_liquidation_cost(self, position: Position) -> float:
        """Estimate cost to liquidate position"""
        # Base cost depends on asset class and size
        base_costs = {
            'equity': 0.001,    # 10 bps
            'options': 0.005,   # 50 bps
            'futures': 0.0005,  # 5 bps
            'crypto': 0.002     # 20 bps
        }
        
        base_cost = base_costs.get(position.asset_class, 0.001)
        
        # Size impact
        size_factor = min(position.market_value / 1000000, 1)  # Cap at $1M
        
        return position.market_value * (base_cost + size_factor * 0.004)
    
    def _get_position_recommendations(self, position: Position, weight: float) -> List[str]:
        """Get recommendations for position management"""
        recommendations = []
        
        if weight > 0.25:
            recommendations.append(f"Position too concentrated ({weight*100:.1f}%). Consider reducing.")
        
        if position.pnl_percent < -20:
            recommendations.append("Position down >20%. Review stop-loss strategy.")
        
        if position.position_type == "short" and position.pnl < 0:
            recommendations.append("Short position moving against you. Monitor closely.")
        
        if position.asset_class == "options" and abs(position.pnl_percent) > 50:
            recommendations.append("Options position showing high volatility. Consider taking profits/cutting losses.")
        
        return recommendations
    
    def _calculate_position_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio for position"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    async def _calculate_position_beta(self, symbol: str, benchmark: str) -> float:
        """Calculate position beta vs benchmark"""
        try:
            symbol_returns = await self._get_symbol_returns(symbol, lookback_days=100)
            benchmark_returns = await self._get_symbol_returns(benchmark, lookback_days=100)
            
            covariance = np.cov(symbol_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        except:
            return 1.0  # Default beta
    
    async def _check_var_alert(self, portfolio_id: str, var_value: float, confidence: float):
        """Check if VaR breaches alert thresholds"""
        thresholds = self.alert_thresholds.get(RiskMetric.VAR.value, {})
        
        for level, threshold in thresholds.items():
            if var_value >= threshold:
                alert = RiskAlert(
                    id=str(uuid.uuid4()),
                    portfolio_id=portfolio_id,
                    metric=RiskMetric.VAR,
                    level=AlertLevel(level),
                    message=f"VaR ({confidence:.0%}) exceeds threshold: {var_value:.2%} > {threshold:.2%}",
                    value=var_value,
                    threshold=threshold,
                    timestamp=datetime.now()
                )
                
                self.alerts.append(alert)
                logger.warning(f"VaR Alert: {alert.message}")
                break


# Create global instance
risk_management_agent = RiskManagementAgent()