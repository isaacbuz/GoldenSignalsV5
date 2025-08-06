"""
AI Prediction Engine - Multi-period price prediction with confidence intervals

This service provides:
- Multi-timeframe predictions (up to 1 week)
- Confidence bands using ensemble methods
- Real-time prediction updates
- Historical pattern matching with RAG
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import json

from services.market_data_service import MarketDataService
from rag.core.engine import RAGEngine
from agents.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

class PredictionEngine:
    """Advanced prediction engine using ensemble methods and historical patterns"""
    
    def __init__(self):
        self.market_data_service = MarketDataService()
        self.rag_engine = RAGEngine()
        self.orchestrator = AgentOrchestrator()
        
    async def get_multi_period_prediction(
        self,
        symbol: str,
        timeframe: str = "15min",
        periods_ahead: int = 24,
        include_confidence_bands: bool = True
    ) -> Dict[str, Any]:
        """Generate multi-period price predictions with confidence intervals"""
        
        try:
            # Get historical data for analysis
            historical_data = await self._get_extended_historical_data(symbol, timeframe)
            if historical_data is None or len(historical_data) < 100:
                raise ValueError(f"Insufficient historical data for {symbol}")
            
            # Run multiple prediction models
            predictions = await self._run_ensemble_predictions(
                historical_data, 
                periods_ahead,
                timeframe
            )
            
            # Get AI agent consensus for market context
            agent_analysis = await self._get_agent_analysis(symbol, timeframe)
            
            # Retrieve similar historical patterns using RAG
            similar_patterns = await self._find_similar_patterns(
                symbol,
                historical_data,
                timeframe
            )
            
            # Combine predictions with confidence intervals
            final_predictions = self._combine_predictions(
                predictions,
                agent_analysis,
                similar_patterns,
                include_confidence_bands
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                predictions,
                agent_analysis,
                similar_patterns
            )
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "last_price": float(historical_data['Close'].iloc[-1]),
                "predictions": final_predictions,
                "overall_confidence": overall_confidence,
                "prediction_metadata": {
                    "models_used": ["linear_regression", "random_forest", "arima", "pattern_matching"],
                    "historical_patterns_found": len(similar_patterns),
                    "agent_consensus": agent_analysis.get("consensus", 0),
                    "market_regime": agent_analysis.get("market_regime", "unknown"),
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            raise
    
    async def _get_extended_historical_data(
        self, 
        symbol: str, 
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Get extended historical data based on timeframe"""
        
        # Determine period based on timeframe
        period_map = {
            "1min": "5d",
            "5min": "1mo",
            "15min": "3mo",
            "30min": "3mo",
            "1h": "6mo",
            "4h": "1y",
            "1d": "2y",
            "1w": "5y"
        }
        
        period = period_map.get(timeframe, "3mo")
        
        # Get data from market data service
        data = await self.market_data_service.get_historical_data(
            symbol, 
            period=period, 
            interval=self._convert_timeframe_to_interval(timeframe)
        )
        
        return data
    
    def _convert_timeframe_to_interval(self, timeframe: str) -> str:
        """Convert our timeframe format to yfinance interval format"""
        conversion_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1h": "1h",
            "4h": "1h",  # yfinance doesn't have 4h, we'll aggregate
            "1d": "1d",
            "1w": "1wk"
        }
        return conversion_map.get(timeframe, "15m")
    
    async def _run_ensemble_predictions(
        self,
        historical_data: pd.DataFrame,
        periods_ahead: int,
        timeframe: str
    ) -> Dict[str, List[Dict[str, float]]]:
        """Run multiple prediction models and return results"""
        
        predictions = {}
        
        # Prepare features
        features, targets = self._prepare_features(historical_data)
        
        # 1. Linear Regression (Trend following)
        lr_predictions = self._linear_regression_predict(
            features, targets, periods_ahead
        )
        predictions['linear_regression'] = lr_predictions
        
        # 2. Random Forest (Non-linear patterns)
        rf_predictions = self._random_forest_predict(
            features, targets, periods_ahead
        )
        predictions['random_forest'] = rf_predictions
        
        # 3. ARIMA (Time series specific)
        arima_predictions = self._arima_predict(
            historical_data['Close'], periods_ahead
        )
        predictions['arima'] = arima_predictions
        
        # 4. Technical Analysis Based Prediction
        ta_predictions = self._technical_analysis_predict(
            historical_data, periods_ahead
        )
        predictions['technical_analysis'] = ta_predictions
        
        return predictions
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML models"""
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['Price_Change'] = data['Close'].pct_change()
        data['Volatility'] = data['Price_Change'].rolling(20).std()
        
        # Drop NaN values
        data = data.dropna()
        
        # Features: OHLCV + indicators
        features = data[['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 
                        'RSI', 'Volume_Ratio', 'Volatility']].values
        
        # Target: Next close price
        targets = data['Close'].shift(-1).dropna().values
        features = features[:-1]  # Align with targets
        
        return features, targets
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _linear_regression_predict(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        periods_ahead: int
    ) -> List[Dict[str, float]]:
        """Linear regression prediction"""
        
        model = LinearRegression()
        model.fit(features, targets)
        
        predictions = []
        last_features = features[-1:].copy()
        
        for i in range(periods_ahead):
            pred = model.predict(last_features)[0]
            
            # Simple confidence based on recent volatility
            volatility = np.std(targets[-20:])
            
            predictions.append({
                "price": float(pred),
                "upper_bound": float(pred + 1.96 * volatility),
                "lower_bound": float(pred - 1.96 * volatility),
                "confidence": 0.7  # Linear models have moderate confidence
            })
            
            # Update features for next prediction (simplified)
            last_features[0, 0] = pred  # Update open with predicted close
            last_features[0, 1] = pred * 1.001  # Simplified high
            last_features[0, 2] = pred * 0.999  # Simplified low
            
        return predictions
    
    def _random_forest_predict(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        periods_ahead: int
    ) -> List[Dict[str, float]]:
        """Random Forest prediction with uncertainty estimates"""
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features, targets)
        
        predictions = []
        last_features = features[-1:].copy()
        
        for i in range(periods_ahead):
            # Get predictions from all trees for uncertainty
            tree_predictions = np.array([
                tree.predict(last_features)[0] 
                for tree in model.estimators_
            ])
            
            pred = np.mean(tree_predictions)
            std = np.std(tree_predictions)
            
            predictions.append({
                "price": float(pred),
                "upper_bound": float(pred + 1.96 * std),
                "lower_bound": float(pred - 1.96 * std),
                "confidence": 0.8  # RF generally has good confidence
            })
            
            # Update features
            last_features[0, 0] = pred
            last_features[0, 1] = pred * 1.001
            last_features[0, 2] = pred * 0.999
            
        return predictions
    
    def _arima_predict(
        self,
        prices: pd.Series,
        periods_ahead: int
    ) -> List[Dict[str, float]]:
        """ARIMA time series prediction"""
        
        try:
            # Fit ARIMA model
            model = ARIMA(prices.values, order=(5, 1, 2))
            fitted_model = model.fit()
            
            # Generate predictions
            forecast = fitted_model.forecast(steps=periods_ahead)
            
            # Get prediction intervals
            forecast_df = fitted_model.get_forecast(steps=periods_ahead)
            confidence_intervals = forecast_df.conf_int(alpha=0.05)
            
            predictions = []
            for i in range(periods_ahead):
                predictions.append({
                    "price": float(forecast[i]),
                    "upper_bound": float(confidence_intervals.iloc[i, 1]),
                    "lower_bound": float(confidence_intervals.iloc[i, 0]),
                    "confidence": 0.75  # ARIMA confidence
                })
            
            return predictions
            
        except Exception as e:
            logger.warning(f"ARIMA prediction failed: {e}")
            # Return simple fallback predictions
            last_price = float(prices.iloc[-1])
            return [{
                "price": last_price,
                "upper_bound": last_price * 1.02,
                "lower_bound": last_price * 0.98,
                "confidence": 0.5
            } for _ in range(periods_ahead)]
    
    def _technical_analysis_predict(
        self,
        data: pd.DataFrame,
        periods_ahead: int
    ) -> List[Dict[str, float]]:
        """Prediction based on technical analysis patterns"""
        
        # Calculate support/resistance levels
        support = data['Low'].rolling(20).min().iloc[-1]
        resistance = data['High'].rolling(20).max().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Calculate trend
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(50).mean().iloc[-1]
        trend_strength = (sma_20 - sma_50) / sma_50
        
        predictions = []
        for i in range(periods_ahead):
            # Project price based on trend and levels
            if trend_strength > 0:
                # Uptrend - price moves toward resistance
                target = current_price + (resistance - current_price) * 0.1 * (i + 1)
            else:
                # Downtrend - price moves toward support
                target = current_price - (current_price - support) * 0.1 * (i + 1)
            
            # Add mean reversion component
            mean_price = data['Close'].rolling(50).mean().iloc[-1]
            target = 0.7 * target + 0.3 * mean_price
            
            predictions.append({
                "price": float(target),
                "upper_bound": float(target * (1 + 0.02 * (i + 1))),
                "lower_bound": float(target * (1 - 0.02 * (i + 1))),
                "confidence": 0.65
            })
            
            current_price = target
        
        return predictions
    
    async def _get_agent_analysis(
        self, 
        symbol: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """Get analysis from AI agents"""
        
        try:
            # For now, return mock analysis since orchestrator needs market data
            # In a full implementation, we'd pass historical data to the orchestrator
            return {
                "consensus": 0.7,  # Mock consensus value
                "signal_type": "HOLD",  # Default signal type
                "market_regime": "trending",  # Mock regime
                "agent_signals": []
            }
        except Exception as e:
            logger.error(f"Agent analysis failed: {e}")
            return {"consensus": 0.5, "signal_type": "HOLD", "market_regime": "unknown"}
    
    async def _find_similar_patterns(
        self,
        symbol: str,
        current_data: pd.DataFrame,
        timeframe: str
    ) -> List[Dict[str, Any]]:
        """Find similar historical patterns using RAG"""
        
        try:
            # Get recent price pattern
            recent_prices = current_data['Close'].tail(20).tolist()
            
            # Create pattern description
            pattern_context = {
                "symbol": symbol,
                "timeframe": timeframe,
                "recent_prices": recent_prices,
                "current_rsi": float(self._calculate_rsi(current_data['Close']).iloc[-1]),
                "volume_trend": "high" if current_data['Volume'].tail(5).mean() > current_data['Volume'].mean() else "normal"
            }
            
            # Search for similar patterns
            similar_patterns = await self.rag_engine.search_similar_contexts(
                json.dumps(pattern_context),
                top_k=5
            )
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Pattern matching failed: {e}")
            return []
    
    def _combine_predictions(
        self,
        model_predictions: Dict[str, List[Dict[str, float]]],
        agent_analysis: Dict[str, Any],
        similar_patterns: List[Dict[str, Any]],
        include_confidence_bands: bool
    ) -> List[Dict[str, float]]:
        """Combine predictions from multiple models with confidence weighting"""
        
        combined_predictions = []
        
        # Get number of periods
        num_periods = len(list(model_predictions.values())[0])
        
        for i in range(num_periods):
            # Collect predictions from all models for this period
            prices = []
            confidences = []
            upper_bounds = []
            lower_bounds = []
            
            for model_name, predictions in model_predictions.items():
                pred = predictions[i]
                prices.append(pred['price'])
                confidences.append(pred['confidence'])
                upper_bounds.append(pred['upper_bound'])
                lower_bounds.append(pred['lower_bound'])
            
            # Weight by confidence
            weights = np.array(confidences) / np.sum(confidences)
            weighted_price = np.sum(np.array(prices) * weights)
            
            # Adjust based on agent analysis
            if agent_analysis['signal_type'] == 'BUY':
                weighted_price *= 1.002  # Slight upward bias
            elif agent_analysis['signal_type'] == 'SELL':
                weighted_price *= 0.998  # Slight downward bias
            
            # Calculate confidence bounds
            if include_confidence_bands:
                # Use weighted average of bounds
                upper = np.sum(np.array(upper_bounds) * weights)
                lower = np.sum(np.array(lower_bounds) * weights)
                
                # Widen bounds based on prediction distance
                uncertainty_factor = 1 + (i * 0.02)  # 2% more uncertainty per period
                upper = weighted_price + (upper - weighted_price) * uncertainty_factor
                lower = weighted_price - (weighted_price - lower) * uncertainty_factor
            else:
                upper = weighted_price * 1.02
                lower = weighted_price * 0.98
            
            # Calculate combined confidence
            base_confidence = np.mean(confidences)
            
            # Boost confidence if we found similar patterns
            if similar_patterns:
                pattern_boost = min(0.1, len(similar_patterns) * 0.02)
                base_confidence = min(0.95, base_confidence + pattern_boost)
            
            # Decay confidence over time
            time_decay = 0.95 ** i
            final_confidence = base_confidence * time_decay
            
            combined_predictions.append({
                "price": float(weighted_price),
                "upper_bound": float(upper),
                "lower_bound": float(lower),
                "confidence": float(final_confidence),
                "period": i + 1
            })
        
        return combined_predictions
    
    def _calculate_overall_confidence(
        self,
        model_predictions: Dict[str, List[Dict[str, float]]],
        agent_analysis: Dict[str, Any],
        similar_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall prediction confidence"""
        
        # Average confidence from models
        all_confidences = []
        for predictions in model_predictions.values():
            all_confidences.extend([p['confidence'] for p in predictions[:5]])  # First 5 periods
        
        model_confidence = np.mean(all_confidences)
        
        # Agent consensus confidence
        agent_confidence = agent_analysis.get('consensus', 0.5)
        
        # Pattern matching confidence
        pattern_confidence = min(0.9, 0.5 + len(similar_patterns) * 0.08)
        
        # Weighted combination
        overall_confidence = (
            0.4 * model_confidence +
            0.4 * agent_confidence +
            0.2 * pattern_confidence
        )
        
        return float(overall_confidence)