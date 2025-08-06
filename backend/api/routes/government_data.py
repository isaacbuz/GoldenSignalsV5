"""
Government Data API Routes
Endpoints for accessing official government economic data
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from datetime import datetime

from services.government_data_service import (
    government_data_service, 
    GovernmentDataSource
)
from agents.government_data_agent import government_data_agent

router = APIRouter(prefix="/api/government", tags=["Government Data"])


@router.get("/economic-data")
async def get_economic_data():
    """
    Get comprehensive economic data from government sources
    
    Returns:
        Dict containing economic indicators from FRED, Treasury, BLS
    """
    try:
        data = await government_data_service.get_comprehensive_economic_data()
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{indicator_name}")
async def get_specific_indicator(
    indicator_name: str,
    source: Optional[str] = Query(default="fred", description="Data source: fred, treasury, bls")
):
    """
    Get a specific economic indicator
    
    Args:
        indicator_name: Name of the indicator (e.g., fed_funds_rate, unemployment_rate)
        source: Data source to use
    
    Returns:
        Specific indicator data
    """
    try:
        source_enum = GovernmentDataSource[source.upper()]
        data = await government_data_service.get_specific_indicator(
            indicator_name, 
            source_enum
        )
        
        if data:
            return {
                "status": "success",
                "data": {
                    "indicator": data.indicator,
                    "value": data.value,
                    "timestamp": data.timestamp.isoformat(),
                    "source": data.source.value,
                    "unit": data.unit,
                    "frequency": data.frequency,
                    "metadata": data.metadata
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Indicator not found")
            
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid source: {source}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/series")
async def get_fred_series():
    """
    Get list of available FRED series
    
    Returns:
        List of available FRED economic series
    """
    return {
        "status": "success",
        "series": government_data_service.fred_series,
        "total": len(government_data_service.fred_series)
    }


@router.get("/treasury/yields")
async def get_treasury_yields():
    """
    Get current Treasury yields
    
    Returns:
        Current US Treasury yields for various maturities
    """
    try:
        async with aiohttp.ClientSession() as session:
            yields = await government_data_service._fetch_treasury_yields(session)
            
            return {
                "status": "success",
                "yields": {
                    maturity: {
                        "yield": y.yield_value,
                        "date": y.date.isoformat(),
                        "change_daily": y.change_daily,
                        "change_weekly": y.change_weekly
                    }
                    for maturity, y in yields.items()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bls/labor-stats")
async def get_labor_statistics():
    """
    Get labor statistics from BLS
    
    Returns:
        Current labor market statistics including unemployment, CPI, etc.
    """
    try:
        async with aiohttp.ClientSession() as session:
            stats = await government_data_service._fetch_bls_data(session)
            
            return {
                "status": "success",
                "statistics": {
                    metric: {
                        "value": stat.value,
                        "date": stat.date.isoformat(),
                        "seasonally_adjusted": stat.seasonally_adjusted,
                        "year_over_year_change": stat.year_over_year_change
                    }
                    for metric, stat in stats.items()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}")
async def get_government_data_analysis(symbol: str):
    """
    Get trading signal based on government economic data
    
    Args:
        symbol: Stock symbol to analyze
    
    Returns:
        Trading signal based on government data analysis
    """
    try:
        market_data = {"symbol": symbol, "price": 100.0}  # Placeholder price
        signal = await government_data_agent.analyze(market_data)
        
        if signal:
            return {
                "status": "success",
                "signal": {
                    "symbol": signal.symbol,
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                    "reasoning": signal.reasoning,
                    "features": signal.features,
                    "market_conditions": signal.market_conditions
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "no_signal",
                "message": "No clear signal from government data",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-regime")
async def get_market_regime():
    """
    Determine current market regime based on government data
    
    Returns:
        Current market regime and supporting indicators
    """
    try:
        data = await government_data_service.get_comprehensive_economic_data()
        analysis = data.get('analysis', {})
        
        return {
            "status": "success",
            "regime": analysis.get('economic_regime', 'uncertain'),
            "yield_curve": analysis.get('yield_curve', {}),
            "inflation_outlook": analysis.get('inflation_outlook', {}),
            "employment_health": analysis.get('employment_health', {}),
            "monetary_policy": analysis.get('monetary_policy', {}),
            "fiscal_health": analysis.get('fiscal_health', {}),
            "market_implications": data.get('market_implications', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/priority")
async def get_priority_indicators():
    """
    Get priority economic indicators for quick market assessment
    
    Returns:
        Key economic indicators that most impact markets
    """
    try:
        # Priority indicators for quick assessment
        priority = [
            'fed_funds_rate',
            'unemployment_rate', 
            'cpi',
            '10_year_treasury',
            'gdp_growth_rate',
            'vix',
            'dollar_index'
        ]
        
        async with aiohttp.ClientSession() as session:
            indicators = await government_data_service._fetch_fred_indicators(session)
            
            priority_data = {
                name: indicators.get(name, {"error": "Not available"})
                for name in priority
            }
            
            return {
                "status": "success",
                "indicators": priority_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))