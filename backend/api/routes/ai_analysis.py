"""
AI Analysis API Routes
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import pandas as pd
import logging

from agents.chart_analyst import chart_analyst
from services.market_data_service import market_data_service
from services.prediction_engine import PredictionEngine
from services.ai_signal_generator import ai_signal_generator

router = APIRouter(prefix="/ai-analysis", tags=["ai-analysis"])
logger = logging.getLogger(__name__)

# Service instances
prediction_engine = PredictionEngine()


@router.get("/chart/{symbol}")
async def get_ai_chart_analysis(
    symbol: str,
    timeframe: str = Query("5m", description="Timeframe: 1m, 5m, 15m, 30m, 1h, 1d"),
    period: str = Query("1d", description="Historical period: 1d, 5d, 1mo, 3mo")
) -> Dict[str, Any]:
    """
    Get comprehensive AI-powered chart analysis for a symbol
    """
    try:
        # Fetch historical data
        df = await market_data_service.get_historical_data(symbol, period, timeframe)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Run AI analysis
        analysis = await chart_analyst.analyze_chart(symbol, df, timeframe)
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        return {
            "success": True,
            "symbol": symbol,
            "analysis": analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI chart analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals/{symbol}")
async def get_trading_signals(
    symbol: str,
    timeframe: str = Query("15m", description="Timeframe for signal generation")
) -> Dict[str, Any]:
    """
    Get AI-generated trading signals for a symbol
    """
    try:
        # Generate AI signal
        signal = await ai_signal_generator.generate_signal(symbol, timeframe)
        
        # Get additional analysis if signal exists
        if signal:
            df = await market_data_service.get_historical_data(symbol, "1d", timeframe)
            if df is not None and not df.empty:
                analysis = await chart_analyst.analyze_chart(symbol, df, timeframe)
                signal["analysis"] = analysis
        
        return {
            "success": True,
            "symbol": symbol,
            "signal": signal,
            "signals": [signal] if signal else [],
            "timestamp": pd.Timestamp.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Signal generation error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-regime/{symbol}")
async def get_market_regime(symbol: str) -> Dict[str, Any]:
    """
    Get current market regime analysis
    """
    try:
        df = await market_data_service.get_historical_data(symbol, "5d", "15m")
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        analysis = await chart_analyst.analyze_chart(symbol, df, "15m")
        
        return {
            "success": True,
            "symbol": symbol,
            "regime": analysis.get("market_regime", {}),
            "patterns": analysis.get("patterns", []),
            "institutional_activity": analysis.get("institutional_activity", {})
        }
        
    except Exception as e:
        logger.error(f"Market regime analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/smart-levels/{symbol}")
async def get_smart_levels(symbol: str) -> Dict[str, Any]:
    """
    Get AI-calculated support/resistance levels
    """
    try:
        df = await market_data_service.get_historical_data(symbol, "1mo", "1h")
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        analysis = await chart_analyst.analyze_chart(symbol, df, "1h")
        
        return {
            "success": True,
            "symbol": symbol,
            "levels": analysis.get("support_resistance", {}),
            "volume_profile": analysis.get("volume_profile", {})
        }
        
    except Exception as e:
        logger.error(f"Smart levels calculation error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/order-flow/{symbol}")
async def get_order_flow_analysis(symbol: str) -> Dict[str, Any]:
    """
    Get order flow and smart money analysis
    """
    try:
        df = await market_data_service.get_historical_data(symbol, "1d", "5m")
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        analysis = await chart_analyst.analyze_chart(symbol, df, "5m")
        
        return {
            "success": True,
            "symbol": symbol,
            "order_flow": analysis.get("order_flow", {}),
            "options_flow": analysis.get("options_flow", {}),
            "institutional_activity": analysis.get("institutional_activity", {})
        }
        
    except Exception as e:
        logger.error(f"Order flow analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prediction/{symbol}")
async def get_price_prediction(
    symbol: str,
    timeframe: str = Query("15m", description="Timeframe for prediction")
) -> Dict[str, Any]:
    """
    Get AI price prediction
    """
    try:
        df = await market_data_service.get_historical_data(symbol, "5d", timeframe)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        analysis = await chart_analyst.analyze_chart(symbol, df, timeframe)
        
        return {
            "success": True,
            "symbol": symbol,
            "prediction": analysis.get("price_prediction", {}),
            "ml_indicators": analysis.get("ml_indicators", {}),
            "confidence_metrics": {
                "regime_confidence": analysis.get("market_regime", {}).get("confidence", 0),
                "prediction_confidence": analysis.get("price_prediction", {}).get("confidence", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Price prediction error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-period-prediction/{symbol}")
async def get_multi_period_prediction(
    symbol: str,
    timeframe: str = Query("15min", description="Timeframe: 1min, 5min, 15min, 30min, 1h, 4h, 1d, 1w"),
    periods: int = Query(24, description="Number of periods to predict ahead (max 168 for 1 week)"),
    confidence_bands: bool = Query(True, description="Include confidence bands")
) -> Dict[str, Any]:
    """
    Get AI-powered multi-period price predictions with confidence intervals
    
    Features:
    - Ensemble of ML models (Linear Regression, Random Forest, ARIMA)
    - Technical analysis pattern matching
    - Historical pattern recognition using RAG
    - AI agent consensus integration
    - Real-time confidence adjustment
    """
    try:
        # Validate inputs
        if periods > 168:  # Max 1 week of hourly predictions
            raise HTTPException(status_code=400, detail="Maximum 168 periods allowed")
        
        valid_timeframes = ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w"]
        if timeframe not in valid_timeframes:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe. Must be one of: {valid_timeframes}")
        
        # Generate predictions
        prediction_result = await prediction_engine.get_multi_period_prediction(
            symbol=symbol,
            timeframe=timeframe,
            periods_ahead=periods,
            include_confidence_bands=confidence_bands
        )
        
        return {
            "success": True,
            "data": prediction_result
        }
        
    except Exception as e:
        logger.error(f"Multi-period prediction error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prediction-confidence/{symbol}")
async def get_prediction_confidence(
    symbol: str,
    timeframe: str = Query("15min", description="Timeframe for confidence analysis")
) -> Dict[str, Any]:
    """
    Get prediction confidence metrics and model performance
    """
    try:
        # This would typically involve backtesting and model validation
        # For now, return basic confidence metrics
        
        historical_data = await market_data_service.get_historical_data(symbol, "1mo", timeframe)
        if historical_data is None or len(historical_data) < 50:
            raise HTTPException(status_code=404, detail="Insufficient historical data")
        
        # Calculate volatility as a confidence factor
        returns = historical_data['Close'].pct_change().dropna()
        volatility = returns.std() * 100  # As percentage
        
        # Calculate trend strength
        sma_20 = historical_data['Close'].rolling(20).mean()
        sma_50 = historical_data['Close'].rolling(50).mean()
        trend_strength = abs((sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
        
        # Base confidence calculation
        base_confidence = max(0.3, min(0.9, 0.8 - (volatility / 10)))
        trend_confidence = min(0.95, base_confidence + (trend_strength * 0.2))
        
        return {
            "success": True,
            "symbol": symbol,
            "confidence_metrics": {
                "overall_confidence": round(trend_confidence, 3),
                "volatility": round(volatility, 2),
                "trend_strength": round(trend_strength, 3),
                "data_quality": "high" if len(historical_data) > 100 else "medium",
                "prediction_horizon": {
                    "short_term": round(trend_confidence, 3),  # 1-6 periods
                    "medium_term": round(trend_confidence * 0.85, 3),  # 6-24 periods
                    "long_term": round(trend_confidence * 0.7, 3)   # 24+ periods
                }
            },
            "recommendation": {
                "use_predictions": trend_confidence > 0.6,
                "confidence_level": "high" if trend_confidence > 0.8 else "medium" if trend_confidence > 0.6 else "low"
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction confidence error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))