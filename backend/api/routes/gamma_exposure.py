"""
Gamma Exposure API Routes
Provides endpoints for gamma analysis and options Greeks
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from core.logging import get_logger
from agents.gamma_exposure_agent import gamma_exposure_agent

logger = get_logger(__name__)
router = APIRouter(tags=["Gamma Exposure"])


@router.post("/gamma/analyze")
async def analyze_gamma_exposure(
    symbol: str,
    spot_price: float,
    options_data: List[Dict[str, Any]],
    historical_volatility: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze gamma exposure for given options chain.
    
    Args:
        symbol: Trading symbol
        spot_price: Current spot price
        options_data: List of options with strikes, OI, volume, etc.
        historical_volatility: Historical volatility for comparison
    
    Returns:
        Comprehensive gamma analysis including:
        - Dealer gamma exposure
        - Pinning analysis
        - Volatility impact
        - Trading signals
    """
    try:
        logger.info(f"Analyzing gamma exposure for {symbol} at spot {spot_price}")
        
        # Validate options data
        if not options_data:
            raise ValueError("No options data provided")
        
        # Perform analysis
        result = await gamma_exposure_agent.analyze(
            options_data=options_data,
            spot_price=spot_price,
            historical_volatility=historical_volatility
        )
        
        # Add symbol to result
        result['symbol'] = symbol
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Gamma analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gamma/greeks")
async def calculate_greeks(
    spot: float = Query(..., description="Spot price"),
    strike: float = Query(..., description="Strike price"),
    time_to_expiry: float = Query(..., description="Time to expiry in years"),
    volatility: float = Query(..., description="Implied volatility"),
    option_type: str = Query("call", description="Option type (call/put)"),
    risk_free_rate: float = Query(0.05, description="Risk-free rate")
) -> Dict[str, Any]:
    """
    Calculate Black-Scholes Greeks for a single option.
    
    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    try:
        # Set risk-free rate
        gamma_exposure_agent.risk_free_rate = risk_free_rate
        
        # Convert option type
        from agents.gamma_exposure_agent import OptionType
        opt_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
        
        # Calculate Greeks
        greeks = gamma_exposure_agent.calculate_black_scholes_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            option_type=opt_type
        )
        
        return {
            'spot': spot,
            'strike': strike,
            'time_to_expiry': time_to_expiry,
            'volatility': volatility,
            'option_type': option_type,
            'greeks': greeks
        }
        
    except Exception as e:
        logger.error(f"Greeks calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gamma/exposure")
async def calculate_dealer_exposure(
    spot_price: float,
    options_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate dealer gamma exposure across strikes.
    
    Args:
        spot_price: Current spot price
        options_data: Options chain data
    
    Returns:
        Dealer gamma and delta exposure by strike
    """
    try:
        exposure = gamma_exposure_agent.calculate_dealer_gamma_exposure(
            options_data=options_data,
            spot_price=spot_price
        )
        
        return exposure
        
    except Exception as e:
        logger.error(f"Exposure calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gamma/pinning")
async def analyze_pinning(
    spot_price: float,
    options_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze gamma pinning risk.
    
    Returns:
        Pinning analysis with key levels and strength
    """
    try:
        # First calculate gamma exposure
        gamma_data = gamma_exposure_agent.calculate_dealer_gamma_exposure(
            options_data=options_data,
            spot_price=spot_price
        )
        
        # Analyze pinning
        pinning = gamma_exposure_agent.analyze_gamma_pinning(
            gamma_data=gamma_data,
            spot_price=spot_price
        )
        
        return pinning
        
    except Exception as e:
        logger.error(f"Pinning analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gamma/volatility-impact")
async def analyze_volatility_impact(
    spot_price: float,
    options_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze how gamma exposure impacts volatility.
    
    Returns:
        Volatility impact analysis and regime classification
    """
    try:
        # Calculate gamma exposure
        gamma_data = gamma_exposure_agent.calculate_dealer_gamma_exposure(
            options_data=options_data,
            spot_price=spot_price
        )
        
        # Analyze volatility impact
        vol_impact = gamma_exposure_agent.calculate_volatility_impact(gamma_data)
        
        return vol_impact
        
    except Exception as e:
        logger.error(f"Volatility impact analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gamma/test-data")
async def get_test_data(
    symbol: str = "SPY",
    spot_price: float = 450.0
) -> Dict[str, Any]:
    """
    Generate test options data for demonstration.
    
    Returns:
        Sample options chain with realistic data
    """
    try:
        # Generate test strikes around spot
        strikes = np.arange(
            spot_price * 0.9,
            spot_price * 1.1,
            5.0
        )
        
        # Generate test options data
        options_data = []
        
        for strike in strikes:
            # Generate realistic open interest (higher ATM)
            distance_from_atm = abs(strike - spot_price) / spot_price
            oi_multiplier = np.exp(-distance_from_atm * 10)
            
            # Call option
            call_oi = int(10000 * oi_multiplier * np.random.uniform(0.8, 1.2))
            call_volume = int(call_oi * np.random.uniform(0.1, 0.3))
            
            options_data.append({
                'strike': float(strike),
                'type': 'call',
                'open_interest': call_oi,
                'volume': call_volume,
                'time_to_expiry': 0.08,  # ~30 days
                'implied_volatility': 0.15 + distance_from_atm * 0.1  # Vol smile
            })
            
            # Put option
            put_oi = int(10000 * oi_multiplier * np.random.uniform(0.7, 1.1))
            put_volume = int(put_oi * np.random.uniform(0.1, 0.3))
            
            options_data.append({
                'strike': float(strike),
                'type': 'put',
                'open_interest': put_oi,
                'volume': put_volume,
                'time_to_expiry': 0.08,
                'implied_volatility': 0.15 + distance_from_atm * 0.12  # Put skew
            })
        
        return {
            'symbol': symbol,
            'spot_price': spot_price,
            'options_data': options_data,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test data generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gamma/signal")
async def generate_gamma_signal(
    spot_price: float,
    options_data: List[Dict[str, Any]],
    historical_volatility: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate trading signal based on gamma analysis.
    
    Returns:
        Trading signal with action, confidence, and reasoning
    """
    try:
        # Perform full analysis
        analysis = await gamma_exposure_agent.analyze(
            options_data=options_data,
            spot_price=spot_price,
            historical_volatility=historical_volatility
        )
        
        # Extract signal
        signal = analysis.get('signal', {})
        
        # Add key metrics
        signal['key_metrics'] = {
            'net_gamma': analysis.get('gamma_exposure', {}).get('net_gamma_exposure', 0),
            'regime': analysis.get('gamma_exposure', {}).get('regime', 'neutral'),
            'pinning_detected': analysis.get('pinning_analysis', {}).get('pinning_detected', False),
            'volatility_impact': analysis.get('volatility_impact', {}).get('volatility_impact', 'neutral')
        }
        
        return signal
        
    except Exception as e:
        logger.error(f"Signal generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))