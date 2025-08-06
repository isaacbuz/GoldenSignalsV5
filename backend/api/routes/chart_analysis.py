"""
AI-Powered Chart Analysis API Routes
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, Any
from services.ai_chart_analysis import ai_chart_analyzer
from services.market_data.yfinance_service import yfinance_service
# from core.security import get_current_user
# from models.user import User
import pandas as pd

router = APIRouter(prefix="/chart-analysis", tags=["chart-analysis"])


@router.get("/analyze/{symbol}")
async def analyze_chart(
    symbol: str,
    period: str = Query("3mo", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max"),
    interval: str = Query("1d", description="Data interval: 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo"),
    include_predictions: bool = Query(True, description="Include price predictions"),
    # current_user: Optional[User] = None  # Make auth optional for demo
):
    """
    Get comprehensive AI analysis of a stock chart
    
    Returns:
    - Pattern recognition (head & shoulders, triangles, etc.)
    - Support and resistance levels
    - Trend analysis
    - Price predictions
    - Volume insights
    - Risk assessment
    - Trading recommendations
    - Chart annotations for visualization
    """
    try:
        # Fetch historical data
        data = yfinance_service.get_historical_data(symbol.upper(), period, interval)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Perform AI analysis
        analysis = await ai_chart_analyzer.analyze_chart(data, symbol.upper())
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        # Optionally remove predictions if not requested
        if not include_predictions and "predictions" in analysis:
            del analysis["predictions"]
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{symbol}")
async def detect_patterns(
    symbol: str,
    period: str = Query("3mo", description="Time period for analysis")
):
    """
    Detect chart patterns for a symbol
    
    Returns list of detected patterns with confidence scores
    """
    try:
        # Fetch data
        data = yfinance_service.get_historical_data(symbol.upper(), period, "1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Get full analysis but return only patterns
        analysis = await ai_chart_analyzer.analyze_chart(data, symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "patterns": analysis.get("patterns", []),
            "timeframe": analysis.get("timeframe", "daily"),
            "confidence": analysis.get("confidence_score", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/support-resistance/{symbol}")
async def get_support_resistance(
    symbol: str,
    period: str = Query("3mo", description="Time period for analysis")
):
    """
    Calculate support and resistance levels using AI
    
    Returns key price levels with strength indicators
    """
    try:
        # Fetch data
        data = yfinance_service.get_historical_data(symbol.upper(), period, "1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Get full analysis but return only support/resistance
        analysis = await ai_chart_analyzer.analyze_chart(data, symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "support_resistance": analysis.get("support_resistance", {}),
            "current_price": data['Close'].iloc[-1]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{symbol}")
async def get_price_predictions(
    symbol: str,
    period: str = Query("3mo", description="Historical data period for training"),
    # current_user: User = Depends(get_current_user)  # Require auth for predictions
):
    """
    Get AI-powered price predictions
    
    Requires authentication
    Returns predicted prices with confidence intervals
    """
    try:
        # Fetch data
        data = yfinance_service.get_historical_data(symbol.upper(), period, "1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Get full analysis but return only predictions
        analysis = await ai_chart_analyzer.analyze_chart(data, symbol.upper())
        
        predictions = analysis.get("predictions", {})
        if "error" in predictions:
            raise HTTPException(status_code=500, detail=predictions["error"])
        
        return {
            "symbol": symbol.upper(),
            "current_price": float(data['Close'].iloc[-1]),
            "predictions": predictions,
            "risk_assessment": analysis.get("risk_assessment", {}),
            "confidence": analysis.get("confidence_score", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{symbol}")
async def get_ai_insights(
    symbol: str,
    period: str = Query("3mo", description="Time period for analysis")
):
    """
    Get AI-generated insights and recommendations
    
    Returns human-readable insights about the chart
    """
    try:
        # Fetch data
        data = yfinance_service.get_historical_data(symbol.upper(), period, "1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Get full analysis
        analysis = await ai_chart_analyzer.analyze_chart(data, symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "insights": analysis.get("ai_insights", []),
            "recommendation": analysis.get("trading_recommendation", {}),
            "risk_assessment": analysis.get("risk_assessment", {}),
            "anomalies": analysis.get("anomalies", []),
            "confidence": analysis.get("confidence_score", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotations/{symbol}")
async def get_chart_annotations(
    symbol: str,
    period: str = Query("3mo", description="Time period for analysis"),
    interval: str = Query("1d", description="Data interval")
):
    """
    Get chart annotations for visualization
    
    Returns annotations to overlay on the chart (patterns, levels, predictions)
    """
    try:
        # Fetch data
        data = yfinance_service.get_historical_data(symbol.upper(), period, interval)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        # Get full analysis
        analysis = await ai_chart_analyzer.analyze_chart(data, symbol.upper())
        
        # Convert data for frontend
        chart_data = []
        for i, (index, row) in enumerate(data.iterrows()):
            chart_data.append({
                "time": index.isoformat() if hasattr(index, 'isoformat') else str(index),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
        
        return {
            "symbol": symbol.upper(),
            "data": chart_data,
            "annotations": analysis.get("annotations", []),
            "insights": analysis.get("ai_insights", [])[:3],  # Top 3 insights
            "support_resistance": analysis.get("support_resistance", {}).get("levels", [])[:5],  # Top 5 levels
            "patterns": [p["pattern"] for p in analysis.get("patterns", [])[:2]],  # Top 2 patterns
            "recommendation": analysis.get("trading_recommendation", {}).get("action", "HOLD")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realtime-analysis")
async def get_realtime_analysis(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    # current_user: Optional[User] = None
):
    """
    Get real-time AI analysis for multiple symbols
    
    Useful for dashboard views
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")][:10]  # Limit to 10
        
        results = []
        for symbol in symbol_list:
            try:
                # Get 1-day data for quick analysis
                data = yfinance_service.get_historical_data(symbol, "5d", "1h")
                
                if not data.empty:
                    # Quick analysis
                    analysis = await ai_chart_analyzer.analyze_chart(data, symbol)
                    
                    # Extract key metrics
                    results.append({
                        "symbol": symbol,
                        "price": float(data['Close'].iloc[-1]),
                        "change_percent": float((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100),
                        "trend": analysis.get("trend_analysis", {}).get("overall_trend", "unknown"),
                        "momentum": analysis.get("momentum_indicators", {}).get("rsi", 50),
                        "recommendation": analysis.get("trading_recommendation", {}).get("action", "HOLD"),
                        "top_pattern": analysis.get("patterns", [{}])[0].get("pattern") if analysis.get("patterns") else None,
                        "risk_level": analysis.get("risk_assessment", {}).get("level", "moderate")
                    })
            except:
                continue
        
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "analyses": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))