"""
Volatility API Routes
Comprehensive volatility analysis and forecasting endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import numpy as np

from core.logging import get_logger
from agents.volatility_agent import volatility_agent, VolatilityRegime, VolatilitySignal, IVRegime

logger = get_logger(__name__)
router = APIRouter(tags=["Volatility Analysis"])


@router.post("/volatility/analyze")
async def analyze_volatility(
    symbol: str,
    market_data: Dict[str, Any],
    include_iv: bool = False,
    forecast_horizons: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Comprehensive volatility analysis for a symbol.
    
    Args:
        symbol: Trading symbol
        market_data: OHLC price data and optional implied volatility data
        include_iv: Whether to include implied volatility analysis
        forecast_horizons: Days to forecast (default: [1, 5, 21])
    
    Returns:
        Complete volatility analysis with regime, patterns, signals, and forecasts
    """
    try:
        # Validate inputs
        if not market_data.get('close_prices'):
            raise HTTPException(status_code=400, detail="close_prices required in market_data")
        
        if len(market_data['close_prices']) < 31:
            raise HTTPException(status_code=400, detail="Minimum 31 data points required for analysis")
        
        if forecast_horizons and any(h < 1 or h > 252 for h in forecast_horizons):
            raise HTTPException(status_code=400, detail="Forecast horizons must be between 1 and 252 days")
        
        logger.info(f"Analyzing volatility for {symbol}")
        
        analysis = await volatility_agent.analyze_volatility(
            symbol=symbol,
            market_data=market_data,
            include_iv=include_iv,
            forecast_horizons=forecast_horizons
        )
        
        return analysis.to_dict()
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Volatility analysis failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/volatility/regime")
async def analyze_volatility_regime(
    symbol: str,
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze current volatility regime for a symbol.
    
    Args:
        symbol: Trading symbol
        market_data: OHLC price data
    
    Returns:
        Volatility regime analysis with percentiles and classifications
    """
    try:
        if not market_data.get('close_prices'):
            raise HTTPException(status_code=400, detail="close_prices required in market_data")
        
        logger.info(f"Analyzing volatility regime for {symbol}")
        
        # Get full analysis but focus on regime
        analysis = await volatility_agent.analyze_volatility(
            symbol=symbol,
            market_data=market_data,
            include_iv=False,
            forecast_horizons=[1]  # Minimal forecasting for speed
        )
        
        metrics = analysis.metrics
        
        # Enhanced regime analysis
        regime_analysis = {
            'symbol': symbol,
            'current_regime': analysis.current_regime.value,
            'volatility_metrics': {
                'current_vol_annualized': metrics.annualized_vol,
                'short_term_vol': metrics.short_term_vol,
                'long_term_vol': metrics.long_term_vol,
                'vol_ratio': metrics.vol_ratio,
                'vol_percentile': metrics.vol_percentile,
                'atr_percentile': metrics.atr_percentile
            },
            'regime_characteristics': {
                'is_low_vol_environment': analysis.current_regime in [VolatilityRegime.LOW, VolatilityRegime.EXTREMELY_LOW],
                'is_high_vol_environment': analysis.current_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH],
                'vol_expansion': metrics.vol_ratio > 1.2,
                'vol_contraction': metrics.vol_ratio < 0.8,
                'extreme_readings': metrics.vol_percentile > 90 or metrics.vol_percentile < 10
            },
            'trading_implications': {
                'volatility_mean_reversion_likely': analysis.current_regime in [VolatilityRegime.EXTREMELY_HIGH, VolatilityRegime.EXTREMELY_LOW],
                'breakout_potential': analysis.current_regime == VolatilityRegime.EXTREMELY_LOW and metrics.vol_percentile < 15,
                'premium_selling_opportunity': analysis.current_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH],
                'premium_buying_opportunity': analysis.current_regime in [VolatilityRegime.LOW, VolatilityRegime.EXTREMELY_LOW]
            },
            'timestamp': analysis.timestamp.isoformat()
        }
        
        return regime_analysis
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Volatility regime analysis failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/volatility/forecast")
async def forecast_volatility(
    symbol: str,
    market_data: Dict[str, Any],
    horizons: List[int] = [1, 5, 10, 21, 63]
) -> Dict[str, Any]:
    """
    Forecast volatility for multiple time horizons.
    
    Args:
        symbol: Trading symbol
        market_data: OHLC price data
        horizons: Days to forecast
    
    Returns:
        Volatility forecasts with confidence intervals
    """
    try:
        # Validate inputs
        if not market_data.get('close_prices'):
            raise HTTPException(status_code=400, detail="close_prices required in market_data")
        
        if any(h < 1 or h > 252 for h in horizons):
            raise HTTPException(status_code=400, detail="Horizons must be between 1 and 252 days")
        
        logger.info(f"Forecasting volatility for {symbol} across {len(horizons)} horizons")
        
        analysis = await volatility_agent.analyze_volatility(
            symbol=symbol,
            market_data=market_data,
            include_iv=False,
            forecast_horizons=horizons
        )
        
        current_price = market_data['close_prices'][-1]
        
        return {
            'symbol': symbol,
            'current_volatility': analysis.metrics.annualized_vol,
            'current_regime': analysis.current_regime.value,
            'forecasts': [forecast.to_dict() for forecast in analysis.forecasts],
            'forecast_summary': {
                'short_term_vol_1d': next((f.forecasted_vol for f in analysis.forecasts if f.horizon_days == 1), None),
                'medium_term_vol_21d': next((f.forecasted_vol for f in analysis.forecasts if f.horizon_days == 21), None),
                'long_term_vol_63d': next((f.forecasted_vol for f in analysis.forecasts if f.horizon_days == 63), None),
                'vol_trend': 'increasing' if analysis.metrics.vol_ratio > 1.1 else 'decreasing' if analysis.metrics.vol_ratio < 0.9 else 'stable'
            },
            'expected_moves': {
                f'{horizon}d_move': f.expected_move for f in analysis.forecasts for horizon in [f.horizon_days]
            },
            'timestamp': analysis.timestamp.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Volatility forecasting failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/volatility/patterns")
async def detect_volatility_patterns(
    symbol: str,
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect volatility patterns and trading opportunities.
    
    Args:
        symbol: Trading symbol
        market_data: OHLC price data
    
    Returns:
        Detected volatility patterns with trading implications
    """
    try:
        if not market_data.get('close_prices'):
            raise HTTPException(status_code=400, detail="close_prices required in market_data")
        
        logger.info(f"Detecting volatility patterns for {symbol}")
        
        analysis = await volatility_agent.analyze_volatility(
            symbol=symbol,
            market_data=market_data,
            include_iv=False,
            forecast_horizons=[1]
        )
        
        return {
            'symbol': symbol,
            'patterns_detected': len(analysis.patterns),
            'patterns': [pattern.to_dict() for pattern in analysis.patterns],
            'dominant_pattern': analysis.patterns[0].to_dict() if analysis.patterns else None,
            'pattern_summary': {
                'squeeze_detected': any(p.pattern_type == 'volatility_squeeze' for p in analysis.patterns),
                'spike_detected': any(p.pattern_type == 'volatility_spike' for p in analysis.patterns),
                'expansion_detected': any(p.pattern_type == 'volatility_expansion' for p in analysis.patterns),
                'clustering_detected': any(p.pattern_type == 'volatility_clustering' for p in analysis.patterns),
                'skew_pattern': any(p.pattern_type == 'negative_skew' for p in analysis.patterns)
            },
            'trading_signals': {
                'primary_signal': analysis.primary_signal.value,
                'signal_strength': analysis.signal_strength,
                'actionable_patterns': len([p for p in analysis.patterns if p.strength > 0.6])
            },
            'timestamp': analysis.timestamp.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pattern detection failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/volatility/iv-analysis")
async def analyze_implied_volatility(
    symbol: str,
    market_data: Dict[str, Any],
    current_iv: float
) -> Dict[str, Any]:
    """
    Analyze implied volatility rank and relationships.
    
    Args:
        symbol: Trading symbol
        market_data: OHLC price data with historical IV if available
        current_iv: Current implied volatility
    
    Returns:
        IV analysis with rank, percentile, and HV-IV relationship
    """
    try:
        if not market_data.get('close_prices'):
            raise HTTPException(status_code=400, detail="close_prices required in market_data")
        
        if not 0.01 <= current_iv <= 5.0:
            raise HTTPException(status_code=400, detail="Current IV must be between 0.01 and 5.0")
        
        # Add current IV to market data
        market_data_with_iv = market_data.copy()
        market_data_with_iv['implied_volatility'] = current_iv
        
        logger.info(f"Analyzing implied volatility for {symbol}: {current_iv:.1%}")
        
        analysis = await volatility_agent.analyze_volatility(
            symbol=symbol,
            market_data=market_data_with_iv,
            include_iv=True,
            forecast_horizons=[1, 5, 21]
        )
        
        metrics = analysis.metrics
        
        return {
            'symbol': symbol,
            'current_iv': current_iv,
            'current_hv': metrics.annualized_vol,
            'iv_analysis': {
                'iv_rank': metrics.iv_rank,
                'iv_percentile': metrics.iv_percentile,
                'iv_regime': analysis.iv_regime.value if analysis.iv_regime else None,
                'hv_iv_spread': metrics.hv_iv_spread,
                'iv_premium': metrics.hv_iv_spread > 0 if metrics.hv_iv_spread else None
            },
            'trading_implications': {
                'iv_expensive': analysis.iv_regime in [IVRegime.OVERVALUED, IVRegime.EXTREMELY_OVERVALUED] if analysis.iv_regime else False,
                'iv_cheap': analysis.iv_regime == IVRegime.UNDERVALUED if analysis.iv_regime else False,
                'sell_premium_opportunity': analysis.iv_regime in [IVRegime.OVERVALUED, IVRegime.EXTREMELY_OVERVALUED] if analysis.iv_regime else False,
                'buy_premium_opportunity': analysis.iv_regime == IVRegime.UNDERVALUED if analysis.iv_regime else False,
                'mean_reversion_likely': metrics.iv_rank and (metrics.iv_rank > 80 or metrics.iv_rank < 20)
            },
            'recommendations': [rec for rec in analysis.recommendations if 'IV' in rec or 'options' in rec],
            'timestamp': analysis.timestamp.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"IV analysis failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility/performance")
async def get_volatility_agent_performance() -> Dict[str, Any]:
    """
    Get volatility agent performance metrics.
    
    Returns:
        Agent performance statistics and analysis history
    """
    try:
        performance = volatility_agent.get_performance_metrics()
        
        return {
            'agent_performance': performance,
            'capabilities': {
                'volatility_metrics': [
                    'realized_volatility', 'ewma_volatility', 'yang_zhang_volatility',
                    'atr_analysis', 'volatility_skew', 'vol_of_vol'
                ],
                'regime_classification': [r.value for r in VolatilityRegime],
                'signal_types': [s.value for s in VolatilitySignal],
                'pattern_detection': [
                    'volatility_squeeze', 'volatility_spike', 'volatility_expansion',
                    'volatility_clustering', 'iv_premium', 'negative_skew'
                ],
                'forecasting_methods': ['mean_reversion', 'garch_like', 'historical_simulation']
            },
            'analysis_features': {
                'supports_iv_analysis': True,
                'multiple_forecast_horizons': True,
                'pattern_recognition': True,
                'regime_classification': True,
                'real_time_analysis': True,
                'performance_tracking': True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/volatility/test")
async def run_volatility_test(
    test_scenario: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Run volatility analysis system tests.
    
    Args:
        test_scenario: Type of test (comprehensive, regime, patterns, forecast, iv)
    
    Returns:
        Test results and validation
    """
    try:
        if test_scenario not in ['comprehensive', 'regime', 'patterns', 'forecast', 'iv']:
            raise HTTPException(status_code=400, detail="Invalid test scenario")
        
        logger.info(f"Running volatility test: {test_scenario}")
        
        # Create test data scenarios
        test_scenarios = {
            'low_vol': {
                'symbol': 'LOW_VOL_TEST',
                'close_prices': [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(100)],
                'description': 'Low volatility environment'
            },
            'high_vol': {
                'symbol': 'HIGH_VOL_TEST', 
                'close_prices': [100 + i * 0.1 + np.random.normal(0, 3.0) for i in range(100)],
                'description': 'High volatility environment'
            },
            'vol_spike': {
                'symbol': 'SPIKE_TEST',
                'close_prices': [100 + i * 0.05 + (np.random.normal(0, 5.0) if i > 70 else np.random.normal(0, 0.5)) for i in range(100)],
                'description': 'Volatility spike scenario'
            }
        }
        
        test_results = {
            'test_scenario': test_scenario,
            'test_cases': []
        }
        
        for scenario_name, scenario_data in test_scenarios.items():
            if test_scenario in ['comprehensive', 'regime', 'patterns']:
                try:
                    # Run volatility analysis
                    analysis = await volatility_agent.analyze_volatility(
                        symbol=scenario_data['symbol'],
                        market_data={'close_prices': scenario_data['close_prices']},
                        include_iv=False,
                        forecast_horizons=[1, 5, 21]
                    )
                    
                    test_case = {
                        'scenario': scenario_name,
                        'description': scenario_data['description'],
                        'results': {
                            'volatility_regime': analysis.current_regime.value,
                            'volatility_percentile': analysis.metrics.vol_percentile,
                            'patterns_detected': len(analysis.patterns),
                            'primary_signal': analysis.primary_signal.value,
                            'signal_strength': analysis.signal_strength
                        },
                        'status': 'passed'
                    }
                    
                except Exception as e:
                    test_case = {
                        'scenario': scenario_name,
                        'description': scenario_data['description'],
                        'error': str(e),
                        'status': 'failed'
                    }
                
                test_results['test_cases'].append(test_case)
        
        # IV test (if requested)
        if test_scenario in ['comprehensive', 'iv']:
            iv_test_data = test_scenarios['low_vol'].copy()
            iv_test_data['implied_volatility'] = 0.25  # 25% IV
            
            try:
                iv_analysis = await volatility_agent.analyze_volatility(
                    symbol='IV_TEST',
                    market_data=iv_test_data,
                    include_iv=True,
                    forecast_horizons=[1]
                )
                
                test_results['test_cases'].append({
                    'scenario': 'iv_analysis',
                    'description': 'Implied volatility analysis test',
                    'results': {
                        'iv_rank': iv_analysis.metrics.iv_rank,
                        'iv_regime': iv_analysis.iv_regime.value if iv_analysis.iv_regime else None,
                        'hv_iv_spread': iv_analysis.metrics.hv_iv_spread
                    },
                    'status': 'passed'
                })
                
            except Exception as e:
                test_results['test_cases'].append({
                    'scenario': 'iv_analysis',
                    'description': 'Implied volatility analysis test',
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Summary
        total_tests = len(test_results['test_cases'])
        passed_tests = len([tc for tc in test_results['test_cases'] if tc.get('status') == 'passed'])
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
            'timestamp': datetime.now().isoformat()
        }
        
        return test_results
        
    except Exception as e:
        logger.error(f"Volatility test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility/education")
async def get_volatility_education() -> Dict[str, Any]:
    """
    Get educational content about volatility analysis.
    
    Returns:
        Comprehensive volatility education and definitions
    """
    try:
        return {
            'volatility_basics': {
                'definition': 'Volatility measures the degree of price variation over time',
                'importance': 'Critical for options pricing, risk management, and market timing',
                'measurement': 'Typically expressed as annualized standard deviation of returns'
            },
            'volatility_types': {
                'historical_volatility': {
                    'description': 'Realized volatility based on past price movements',
                    'calculation': 'Standard deviation of historical returns, annualized',
                    'uses': 'Risk assessment, model calibration, regime identification'
                },
                'implied_volatility': {
                    'description': 'Market expectation of future volatility derived from option prices',
                    'calculation': 'Backed out from option prices using Black-Scholes model',
                    'uses': 'Options trading, market sentiment, forward-looking volatility'
                },
                'ewma_volatility': {
                    'description': 'Exponentially weighted moving average giving more weight to recent data',
                    'advantage': 'More responsive to recent market changes',
                    'uses': 'Dynamic risk management, trend identification'
                },
                'yang_zhang_volatility': {
                    'description': 'Estimator using OHLC data to capture overnight and intraday moves',
                    'advantage': 'More accurate than close-to-close for markets with gaps',
                    'uses': 'Professional volatility estimation, academic research'
                }
            },
            'volatility_regimes': {
                'extremely_low': 'Below 10th percentile - expect expansion, buy volatility',
                'low': '10th-25th percentile - potential volatility opportunities',
                'normal': '25th-75th percentile - typical market conditions',
                'high': '75th-90th percentile - consider selling premium',
                'extremely_high': 'Above 90th percentile - strong mean reversion candidate'
            },
            'iv_analysis': {
                'iv_rank': {
                    'definition': '(Current IV - Min IV) / (Max IV - Min IV) over lookback period',
                    'interpretation': 'Shows where current IV sits in historical range (0-100%)',
                    'trading_use': 'High IV rank (>75) suggests selling premium, low IV rank (<25) suggests buying'
                },
                'iv_percentile': {
                    'definition': 'Percentage of days current IV was below historical values',
                    'interpretation': 'More robust than IV rank as it accounts for distribution',
                    'trading_use': 'Similar to IV rank but less affected by extreme outliers'
                },
                'hv_iv_relationship': {
                    'normal_premium': 'IV typically trades at premium to HV (5-15%)',
                    'convergence': 'IV tends to converge toward realized volatility over time',
                    'divergence_opportunities': 'Large spreads create trading opportunities'
                }
            },
            'pattern_recognition': {
                'volatility_squeeze': {
                    'definition': 'Period of unusually low volatility',
                    'identification': 'Low vol percentile + tight ATR + low vol ratio',
                    'implication': 'Often precedes significant directional moves'
                },
                'volatility_spike': {
                    'definition': 'Sudden increase in volatility',
                    'identification': 'High vol percentile + elevated ATR + high vol ratio',
                    'implication': 'Mean reversion opportunity, consider selling premium'
                },
                'volatility_clustering': {
                    'definition': 'Tendency for high/low volatility to persist (GARCH effect)',
                    'identification': 'High vol-of-vol + persistent regime',
                    'implication': 'Current volatility regime likely to continue'
                }
            },
            'trading_strategies': {
                'long_volatility': {
                    'when': 'Low IV rank, volatility squeeze, expected events',
                    'instruments': 'Long straddles, strangles, VIX calls, options buying',
                    'risks': 'Time decay, volatility doesn\'t expand as expected'
                },
                'short_volatility': {
                    'when': 'High IV rank, volatility spike, stable market expected',
                    'instruments': 'Short straddles, iron condors, covered calls, VIX puts',
                    'risks': 'Unlimited loss potential, sudden volatility expansion'
                },
                'volatility_arbitrage': {
                    'when': 'HV-IV divergence, mispriced options',
                    'instruments': 'Delta-neutral options positions, volatility swaps',
                    'risks': 'Model risk, transaction costs, gamma hedging'
                }
            },
            'risk_management': {
                'position_sizing': 'Size positions based on current volatility regime',
                'stop_losses': 'Wider stops in high vol, tighter in low vol environments',
                'hedging': 'Use volatility products as portfolio hedge',
                'timing': 'Enter positions during favorable volatility regimes',
                'monitoring': 'Continuously monitor regime changes and adjust accordingly'
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get volatility education: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))