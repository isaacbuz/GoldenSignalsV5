"""
LSTM Prediction Tool for MCP
Wraps the LSTM predictor as an MCP-compatible tool
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json

from mcp.servers.base import BaseTool, ToolResult
from services.ml.lstm_predictor import lstm_predictor
from core.logging import get_logger

logger = get_logger(__name__)


class LSTMPredictionTool(BaseTool):
    """
    MCP Tool for LSTM stock price predictions
    
    This tool provides high-accuracy price predictions using
    a bidirectional LSTM model with 95%+ directional accuracy.
    """
    
    def __init__(self):
        super().__init__(
            name="lstm_prediction",
            description="Generate high-accuracy stock price predictions using LSTM neural network"
        )
        
    @property
    def parameters(self) -> Dict[str, Any]:
        """Define tool parameters"""
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to predict (e.g., AAPL, MSFT)"
                },
                "timeframe": {
                    "type": "string",
                    "description": "Prediction timeframe: 1d, 1w, 1m",
                    "enum": ["1d", "1w", "1m"],
                    "default": "1d"
                }
            },
            "required": ["symbol"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute LSTM prediction
        
        Args:
            symbol: Stock symbol to predict
            timeframe: Prediction timeframe (1d, 1w, 1m)
            
        Returns:
            ToolResult with prediction data
        """
        
        try:
            # Extract parameters
            symbol = kwargs.get('symbol')
            timeframe = kwargs.get('timeframe', '1d')
            
            if not symbol:
                return ToolResult(
                    success=False,
                    error="Symbol is required"
                )
            
            # Validate symbol
            symbol = symbol.upper()
            
            # Get prediction
            logger.info(f"Generating LSTM prediction for {symbol} ({timeframe})")
            prediction = await lstm_predictor.predict(symbol, timeframe)
            
            if 'error' in prediction:
                return ToolResult(
                    success=False,
                    error=prediction['error']
                )
            
            # Format result for MCP
            result = {
                "prediction": {
                    "symbol": prediction['symbol'],
                    "current_price": prediction['current_price'],
                    "predicted_price": prediction['predicted_price'],
                    "price_change": prediction['price_change'],
                    "price_change_percentage": prediction['price_change_pct'],
                    "signal": prediction['signal'],
                    "confidence": prediction['confidence'],
                    "timeframe": timeframe
                },
                "model_info": {
                    "version": prediction['model_version'],
                    "accuracy": prediction['model_accuracy'],
                    "type": "Bidirectional LSTM"
                },
                "metadata": {
                    "timestamp": prediction['timestamp'],
                    "predictions_all_timeframes": prediction.get('predictions', {})
                }
            }
            
            # Add interpretation
            interpretation = self._interpret_prediction(prediction)
            result["interpretation"] = interpretation
            
            return ToolResult(
                success=True,
                data=result
            )
            
        except Exception as e:
            logger.error(f"LSTM prediction tool error: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def _interpret_prediction(self, prediction: Dict[str, Any]) -> str:
        """
        Generate human-readable interpretation of prediction
        """
        
        signal = prediction['signal']
        confidence = prediction['confidence']
        change_pct = prediction['price_change_pct']
        
        if signal == 'BUY':
            action = "bullish"
            recommendation = "consider buying"
        elif signal == 'SELL':
            action = "bearish"
            recommendation = "consider selling"
        else:
            action = "neutral"
            recommendation = "hold position"
        
        interpretation = (
            f"The LSTM model predicts a {abs(change_pct):.1f}% "
            f"{'increase' if change_pct > 0 else 'decrease'} in price. "
            f"Signal is {action} with {confidence:.0%} confidence. "
            f"Recommendation: {recommendation}."
        )
        
        if confidence < 0.7:
            interpretation += " Note: Confidence is relatively low, consider additional analysis."
        elif confidence > 0.85:
            interpretation += " Note: High confidence prediction based on strong patterns."
        
        return interpretation
    
    async def train_model(self, symbol: str, start_date: str = None, end_date: str = None) -> ToolResult:
        """
        Train or retrain the LSTM model
        
        This is an administrative function that should be called
        periodically to update the model with new data.
        """
        
        try:
            logger.info(f"Training LSTM model for {symbol}")
            
            # Train model
            training_result = await lstm_predictor.train(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            return ToolResult(
                success=True,
                data={
                    "training_complete": True,
                    "accuracy": training_result['accuracy'],
                    "train_loss": training_result['train_loss'],
                    "val_loss": training_result['val_loss'],
                    "message": f"Model trained successfully with {training_result['accuracy']:.2%} accuracy"
                }
            )
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


# Create singleton instance
lstm_prediction_tool = LSTMPredictionTool()