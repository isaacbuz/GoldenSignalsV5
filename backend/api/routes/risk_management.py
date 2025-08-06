"""
Risk Management API Routes
Comprehensive portfolio risk analysis and monitoring endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from core.logging import get_logger
from agents.risk_management_agent import risk_management_agent, RiskLevel, AlertLevel, RiskMetric

logger = get_logger(__name__)
router = APIRouter(tags=["Risk Management"])


@router.post("/risk/portfolio")
async def create_portfolio(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create or update a portfolio for risk monitoring.
    
    Args:
        portfolio_data: Portfolio details including positions, cash, and metadata
    
    Returns:
        Portfolio creation confirmation with basic statistics
    """
    try:
        logger.info(f"Creating portfolio: {portfolio_data.get('name', 'Unnamed')}")
        
        result = await risk_management_agent.create_portfolio(portfolio_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Portfolio creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/portfolio/{portfolio_id}/assessment")
async def assess_portfolio_risk(portfolio_id: str) -> Dict[str, Any]:
    """
    Get comprehensive risk assessment for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
    
    Returns:
        Complete risk assessment with metrics, alerts, and recommendations
    """
    try:
        logger.info(f"Assessing risk for portfolio: {portfolio_id}")
        
        assessment = await risk_management_agent.assess_portfolio_risk(portfolio_id)
        
        return assessment.to_dict()
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Risk assessment failed for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/portfolio/{portfolio_id}/var")
async def calculate_var(
    portfolio_id: str,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    method: str = "historical"
) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        confidence_level: Confidence level (0.90, 0.95, 0.99)
        time_horizon: Time horizon in days
        method: Calculation method (historical, parametric, monte_carlo)
    
    Returns:
        VaR calculation results including CVaR and portfolio impact
    """
    try:
        # Validate inputs
        if not 0.5 <= confidence_level <= 0.999:
            raise HTTPException(status_code=400, detail="Confidence level must be between 0.5 and 0.999")
        
        if not 1 <= time_horizon <= 252:
            raise HTTPException(status_code=400, detail="Time horizon must be between 1 and 252 days")
        
        if method not in ['historical', 'parametric', 'monte_carlo']:
            raise HTTPException(status_code=400, detail="Method must be historical, parametric, or monte_carlo")
        
        logger.info(f"Calculating VaR for {portfolio_id}: {confidence_level:.1%} confidence, {time_horizon}d horizon, {method}")
        
        result = await risk_management_agent.calculate_var(
            portfolio_id=portfolio_id,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=method
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"VaR calculation failed for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/portfolio/{portfolio_id}/stress-test")
async def run_stress_test(
    portfolio_id: str,
    scenarios: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Run stress tests on a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        scenarios: Custom stress scenarios (optional, uses defaults if not provided)
    
    Returns:
        Stress test results with scenario outcomes and survival analysis
    """
    try:
        logger.info(f"Running stress test for portfolio: {portfolio_id}")
        
        result = await risk_management_agent.stress_test(
            portfolio_id=portfolio_id,
            scenarios=scenarios
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Stress test failed for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/portfolio/{portfolio_id}/circuit-breakers")
async def check_circuit_breakers(portfolio_id: str) -> Dict[str, Any]:
    """
    Check if portfolio breaches circuit breaker thresholds.
    
    Args:
        portfolio_id: Portfolio identifier
    
    Returns:
        Circuit breaker status and any breaches detected
    """
    try:
        logger.info(f"Checking circuit breakers for portfolio: {portfolio_id}")
        
        result = await risk_management_agent.check_circuit_breakers(portfolio_id)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Circuit breaker check failed for {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/portfolio/{portfolio_id}/position/{symbol}")
async def analyze_position_risk(
    portfolio_id: str,
    symbol: str
) -> Dict[str, Any]:
    """
    Analyze risk for a specific position within a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        symbol: Position symbol
    
    Returns:
        Position-specific risk analysis and recommendations
    """
    try:
        logger.info(f"Analyzing position risk: {symbol} in {portfolio_id}")
        
        result = await risk_management_agent.analyze_position_risk(
            portfolio_id=portfolio_id,
            symbol=symbol
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Position risk analysis failed for {symbol} in {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/alerts")
async def get_risk_alerts(
    portfolio_id: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get risk alerts with optional filtering.
    
    Args:
        portfolio_id: Filter by portfolio (optional)
        level: Filter by alert level (info, warning, critical, emergency)
        limit: Maximum number of alerts to return
    
    Returns:
        Risk alerts with counts and summaries
    """
    try:
        if level and level not in ['info', 'warning', 'critical', 'emergency']:
            raise HTTPException(status_code=400, detail="Invalid alert level")
        
        if not 1 <= limit <= 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
        
        result = risk_management_agent.get_alerts(
            portfolio_id=portfolio_id,
            level=level,
            limit=limit
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/alerts/threshold")
async def set_alert_threshold(
    metric: str,
    level: str,
    threshold: float
) -> Dict[str, Any]:
    """
    Set custom alert threshold for a risk metric.
    
    Args:
        metric: Risk metric name
        level: Alert level (warning, critical, emergency)
        threshold: Threshold value
    
    Returns:
        Confirmation of threshold update
    """
    try:
        # Validate inputs
        valid_metrics = [m.value for m in RiskMetric]
        if metric not in valid_metrics:
            raise HTTPException(status_code=400, detail=f"Invalid metric. Must be one of: {valid_metrics}")
        
        valid_levels = [l.value for l in AlertLevel if l != AlertLevel.INFO]
        if level not in valid_levels:
            raise HTTPException(status_code=400, detail=f"Invalid level. Must be one of: {valid_levels}")
        
        if not 0 <= threshold <= 1:
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
        
        result = risk_management_agent.set_alert_threshold(
            metric=metric,
            level=level,
            threshold=threshold
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to set alert threshold: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/performance")
async def get_risk_agent_performance() -> Dict[str, Any]:
    """
    Get risk management agent performance metrics.
    
    Returns:
        Agent performance statistics and configuration
    """
    try:
        performance = risk_management_agent.get_performance_metrics()
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/test")
async def run_risk_test(
    test_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Run risk management system tests.
    
    Args:
        test_type: Type of test (comprehensive, var, stress, alerts)
    
    Returns:
        Test results and validation
    """
    try:
        if test_type not in ['comprehensive', 'var', 'stress', 'alerts']:
            raise HTTPException(status_code=400, detail="Invalid test type")
        
        logger.info(f"Running risk management test: {test_type}")
        
        # Create test portfolio
        test_portfolio = {
            "name": "Risk Test Portfolio",
            "cash": 100000,
            "type": "mixed",
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "entry_price": 150.0,
                    "current_price": 155.0,
                    "position_type": "long",
                    "asset_class": "equity",
                    "sector": "Technology"
                },
                {
                    "symbol": "GOOGL",
                    "quantity": 50,
                    "entry_price": 2800.0,
                    "current_price": 2750.0,
                    "position_type": "long",
                    "asset_class": "equity",
                    "sector": "Technology"
                },
                {
                    "symbol": "TSLA",
                    "quantity": -20,  # Short position
                    "entry_price": 900.0,
                    "current_price": 920.0,
                    "position_type": "short",
                    "asset_class": "equity",
                    "sector": "Consumer Discretionary"
                }
            ]
        }
        
        # Create portfolio
        portfolio_result = await risk_management_agent.create_portfolio(test_portfolio)
        portfolio_id = portfolio_result["portfolio_id"]
        
        test_results = {
            "test_type": test_type,
            "portfolio_created": portfolio_result,
            "tests_run": []
        }
        
        if test_type in ['comprehensive', 'var']:
            # Test VaR calculation
            var_result = await risk_management_agent.calculate_var(portfolio_id)
            test_results["tests_run"].append({
                "test": "VaR Calculation",
                "result": var_result,
                "status": "passed" if var_result["var"] > 0 else "failed"
            })
        
        if test_type in ['comprehensive', 'stress']:
            # Test stress scenarios
            stress_result = await risk_management_agent.stress_test(portfolio_id)
            test_results["tests_run"].append({
                "test": "Stress Test",
                "result": stress_result,
                "status": "passed" if len(stress_result["scenarios"]) > 0 else "failed"
            })
        
        if test_type in ['comprehensive', 'alerts']:
            # Test risk assessment (generates alerts)
            assessment_result = await risk_management_agent.assess_portfolio_risk(portfolio_id)
            test_results["tests_run"].append({
                "test": "Risk Assessment",
                "result": assessment_result,
                "status": "passed" if assessment_result["risk_score"] > 0 else "failed"
            })
        
        # Circuit breaker test
        if test_type == 'comprehensive':
            circuit_result = await risk_management_agent.check_circuit_breakers(portfolio_id)
            test_results["tests_run"].append({
                "test": "Circuit Breakers",
                "result": circuit_result,
                "status": "passed"
            })
        
        # Position analysis test
        if test_type == 'comprehensive':
            position_result = await risk_management_agent.analyze_position_risk(portfolio_id, "AAPL")
            test_results["tests_run"].append({
                "test": "Position Analysis",
                "result": position_result,
                "status": "passed" if position_result["risk_metrics"]["position_weight"] > 0 else "failed"
            })
        
        test_results["summary"] = {
            "total_tests": len(test_results["tests_run"]),
            "passed_tests": len([t for t in test_results["tests_run"] if t["status"] == "passed"]),
            "test_portfolio_id": portfolio_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return test_results
        
    except Exception as e:
        logger.error(f"Risk test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/scenarios/templates")
async def get_stress_scenario_templates() -> Dict[str, Any]:
    """
    Get predefined stress test scenario templates.
    
    Returns:
        Available stress test scenarios with descriptions
    """
    try:
        scenarios = {
            "market_crash": {
                "name": "Market Crash",
                "description": "Major market decline with elevated volatility",
                "market_shock": -0.20,
                "vol_shock": 2.0,
                "correlation_shock": 0.8,
                "typical_frequency": "Once every 5-10 years"
            },
            "flash_crash": {
                "name": "Flash Crash",
                "description": "Sudden, sharp market decline with extreme volatility",
                "market_shock": -0.10,
                "vol_shock": 3.0,
                "correlation_shock": 0.9,
                "typical_frequency": "Once every 2-3 years"
            },
            "interest_rate_spike": {
                "name": "Interest Rate Spike",
                "description": "Sharp increase in interest rates affecting valuations",
                "market_shock": -0.05,
                "vol_shock": 1.5,
                "correlation_shock": 0.6,
                "typical_frequency": "Once every 3-5 years"
            },
            "black_swan": {
                "name": "Black Swan Event",
                "description": "Extreme rare event with severe market impact",
                "market_shock": -0.30,
                "vol_shock": 4.0,
                "correlation_shock": 0.95,
                "typical_frequency": "Once every 10-20 years"
            },
            "sector_rotation": {
                "name": "Sector Rotation",
                "description": "Major rotation between market sectors",
                "market_shock": 0.0,
                "vol_shock": 1.8,
                "correlation_shock": 0.3,
                "typical_frequency": "Ongoing cyclical process"
            },
            "liquidity_crisis": {
                "name": "Liquidity Crisis",
                "description": "Market-wide liquidity shortage",
                "market_shock": -0.15,
                "vol_shock": 2.5,
                "correlation_shock": 0.85,
                "typical_frequency": "Once every 7-10 years"
            },
            "currency_crisis": {
                "name": "Currency Crisis",
                "description": "Major currency devaluation or crisis",
                "market_shock": -0.12,
                "vol_shock": 2.2,
                "correlation_shock": 0.7,
                "typical_frequency": "Varies by currency and region"
            }
        }
        
        return {
            "scenarios": scenarios,
            "total_scenarios": len(scenarios),
            "usage_instructions": {
                "description": "Use these templates in stress test API calls",
                "example_usage": "Pass scenarios array with modified parameters to /risk/portfolio/{id}/stress-test"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get scenario templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/metrics/definitions")
async def get_risk_metrics_definitions() -> Dict[str, Any]:
    """
    Get definitions and explanations of all risk metrics.
    
    Returns:
        Comprehensive risk metrics documentation
    """
    try:
        metrics_definitions = {
            "value_at_risk": {
                "name": "Value at Risk (VaR)",
                "description": "Maximum expected loss over a given time horizon at a specified confidence level",
                "formula": "VaR = Portfolio Value × Percentile of Return Distribution",
                "interpretation": "There is a X% confidence that losses will not exceed VaR amount",
                "typical_confidence_levels": [0.95, 0.99],
                "time_horizons": ["1 day", "5 days", "20 days"]
            },
            "conditional_value_at_risk": {
                "name": "Conditional VaR (CVaR) / Expected Shortfall",
                "description": "Expected loss given that losses exceed the VaR threshold",
                "formula": "CVaR = E[Loss | Loss > VaR]",
                "interpretation": "Average loss in worst-case scenarios beyond VaR",
                "importance": "More informative than VaR for tail risk assessment"
            },
            "sharpe_ratio": {
                "name": "Sharpe Ratio",
                "description": "Risk-adjusted return metric comparing excess return to volatility",
                "formula": "(Portfolio Return - Risk-free Rate) / Portfolio Volatility",
                "interpretation": ">1.0 = Good, >2.0 = Very Good, >3.0 = Excellent",
                "annualized": True
            },
            "sortino_ratio": {
                "name": "Sortino Ratio",
                "description": "Risk-adjusted return using only downside volatility",
                "formula": "(Portfolio Return - Risk-free Rate) / Downside Deviation",
                "interpretation": "Better than Sharpe for asymmetric return distributions",
                "advantage": "Only penalizes negative volatility"
            },
            "max_drawdown": {
                "name": "Maximum Drawdown",
                "description": "Largest peak-to-trough decline in portfolio value",
                "formula": "Max((Peak - Trough) / Peak)",
                "interpretation": "Worst loss experienced by investor",
                "typical_thresholds": {"acceptable": 0.1, "concerning": 0.2, "severe": 0.3}
            },
            "volatility": {
                "name": "Volatility (Standard Deviation)",
                "description": "Measure of price variation, typically annualized",
                "formula": "Standard Deviation of Returns × √252",
                "interpretation": "Higher volatility = higher risk",
                "typical_equity_range": "15% - 30% annually"
            },
            "beta": {
                "name": "Beta",
                "description": "Sensitivity of portfolio returns to market movements",
                "formula": "Covariance(Portfolio, Market) / Variance(Market)",
                "interpretation": "1.0 = Market risk, >1.0 = More volatile, <1.0 = Less volatile",
                "benchmark": "Usually vs S&P 500"
            },
            "alpha": {
                "name": "Alpha",
                "description": "Excess return above what beta-adjusted market return would predict",
                "formula": "Portfolio Return - (Risk-free Rate + Beta × Market Risk Premium)",
                "interpretation": "Positive alpha indicates outperformance",
                "skill_indicator": True
            },
            "concentration_risk": {
                "name": "Concentration Risk (Herfindahl Index)",
                "description": "Measure of portfolio concentration across positions",
                "formula": "Sum of squared position weights",
                "interpretation": "0 = Perfect diversification, 1 = Single position",
                "typical_thresholds": {"diversified": 0.1, "concentrated": 0.25, "risky": 0.4}
            }
        }
        
        return {
            "risk_metrics": metrics_definitions,
            "calculation_notes": {
                "returns_frequency": "Daily returns typically used, annualized where appropriate",
                "risk_free_rate": "Current 10-year Treasury rate or equivalent",
                "confidence_levels": "95% and 99% most common for VaR",
                "lookback_periods": "252 trading days (1 year) typical for most metrics"
            },
            "interpretation_guide": {
                "low_risk": "Conservative portfolio with stable returns",
                "moderate_risk": "Balanced portfolio with reasonable volatility",
                "high_risk": "Aggressive portfolio with high return potential and volatility"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics definitions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))