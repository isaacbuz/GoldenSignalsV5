"""
ML-Enhanced Trading Agent
Combines traditional technical analysis with ML predictions
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import numpy as np

from agents.base import BaseAgent, Signal, AgentCapability, AgentContext, AgentConfig
from ml.finance_ml_pipeline import FinanceMLPipeline, ModelType
from core.events.bus import event_bus
from core.logging import get_logger

logger = get_logger(__name__)


class MLEnhancedAgent(BaseAgent):
    """
    Trading agent that combines ML predictions with traditional analysis
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.ml_pipeline = FinanceMLPipeline()
        self.model_path: Optional[str] = None
        self.model_loaded = False
        self.prediction_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # ML configuration
        self.ml_weight = 0.5  # Weight given to ML predictions vs traditional signals
        self.min_ml_confidence = 0.6
        self.use_ensemble = True
        self.models: List[FinanceMLPipeline] = []
        
        # Initialize capabilities
        self.capabilities = [
            AgentCapability.TECHNICAL_ANALYSIS,
            AgentCapability.PATTERN_RECOGNITION,
            AgentCapability.ML_PREDICTION
        ]
    
    async def initialize(self) -> None:
        """Initialize the ML-enhanced agent"""
        await super().initialize()
        
        # Try to load the latest model
        await self._load_latest_model()
        
        # Subscribe to ML events
        await event_bus.subscribe("ml.training.completed", self._on_model_trained)
        
        logger.info("ML-Enhanced Agent initialized")
    
    async def _load_latest_model(self) -> None:
        """Load the most recent trained model"""
        try:
                        model_dir = "models"
            
            if os.path.exists(model_dir):
                models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                if models:
                    # Get the most recent model
                    models.sort(key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
                    latest_model = models[-1]
                    self.model_path = os.path.join(model_dir, latest_model)
                    
                    await self.ml_pipeline.load_model(self.model_path)
                    self.model_loaded = True
                    logger.info(f"Loaded ML model: {self.model_path}")
                else:
                    logger.warning("No ML models found")
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {str(e)}")
    
    async def analyze(self, context: AgentContext) -> Optional[Signal]:
        """
        Analyze market data using both ML and traditional methods
        
        Args:
            context: Agent context with market data
            
        Returns:
            Trading signal or None
        """
        try:
            symbol = context.symbol
            timeframe = context.timeframe
            
            # Get traditional technical analysis signal
            ta_signal = await self._get_technical_signal(context)
            
            # Get ML prediction if model is loaded
            ml_signal = None
            if self.model_loaded:
                ml_signal = await self._get_ml_prediction(symbol, timeframe)
            
            # Combine signals
            final_signal = await self._combine_signals(ta_signal, ml_signal, context)
            
            if final_signal:
                # Cache the signal
                await self._cache_signal(final_signal)
                
                # Publish signal event
                await event_bus.publish(
                    "agent.signal.generated",
                    data={
                        "agent_id": self.config.agent_id,
                        "signal": final_signal.dict(),
                        "ml_used": ml_signal is not None
                    }
                )
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return None
    
    async def _get_technical_signal(self, context: AgentContext) -> Optional[Signal]:
        """
        Get signal from traditional technical analysis
        
        Args:
            context: Agent context
            
        Returns:
            Technical analysis signal
        """
        try:
            market_data = context.market_data
            
            if not market_data or 'ohlcv' not in market_data:
                return None
            
            ohlcv = market_data['ohlcv']
            if len(ohlcv) < 50:  # Need enough data for indicators
                return None
            
            # Calculate technical indicators
            closes = [candle['close'] for candle in ohlcv]
            volumes = [candle['volume'] for candle in ohlcv]
            
            # Simple moving averages
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:])
            
            # RSI
            rsi = self._calculate_rsi(closes, 14)
            
            # MACD
            macd, signal_line = self._calculate_macd(closes)
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_surge = current_volume > avg_volume * 1.5
            
            # Generate signal based on indicators
            current_price = closes[-1]
            signal_strength = 0
            action = None
            
            # Trend following
            if sma_20 > sma_50 and current_price > sma_20:
                signal_strength += 0.3
                action = "buy"
            elif sma_20 < sma_50 and current_price < sma_20:
                signal_strength += 0.3
                action = "sell"
            
            # Momentum
            if rsi < 30:  # Oversold
                signal_strength += 0.2
                if action != "sell":
                    action = "buy"
            elif rsi > 70:  # Overbought
                signal_strength += 0.2
                if action != "buy":
                    action = "sell"
            
            # MACD crossover
            if macd > signal_line:
                signal_strength += .2
                if not action:
                    action = "buy"
            elif macd < signal_line:
                signal_strength += 0.2
                if not action:
                    action = "sell"
            
            # Volume confirmation
            if volume_surge:
                signal_strength += 0.1
            
            # Create signal if strong enough
            if signal_strength >= 0.5 and action:
                # Calculate stop loss and take profit
                atr = self._calculate_atr(ohlcv, 14)
                
                if action == "buy":
                    stop_loss = current_price - (2 * atr)
                    take_profit = current_price + (3 * atr)
                else:
                    stop_loss = current_price + (2 * atr)
                    take_profit = current_price - (3 * atr)
                
                return Signal(
                    symbol=context.symbol,
                    action=action,
                    confidence=min(signal_strength, 1.0),
                    reasoning="Technical indicators alignment",
                    metadata={
                        "entry_price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "rsi": rsi,
                        "sma_20": sma_20,
                        "sma_50": sma_50,
                        "volume_surge": volume_surge
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            return None
    
    async def _get_ml_prediction(self, symbol: str, timeframe: str) -> Optional[Signal]:
        """
        Get ML model prediction
        
        Args:
            symbol: Trading symbol
            timeframe: Time frame
            
        Returns:
            ML-based signal
        """
        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.prediction_cache:
                cached = self.prediction_cache[cache_key]
                if (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
                    return cached['signal']
            
            # Prepare data for prediction
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            await self.ml_pipeline.prepare_data(
                [symbol],
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if self.ml_pipeline.data.empty:
                return None
            
            # Get latest features
            latest_features = self.ml_pipeline.data.iloc[-1][self.ml_pipeline.feature_columns]
            
            # Make prediction
            prediction = await self.ml_pipeline.predict(
                latest_features.values.reshape(1, -1)
            )
            
            if prediction is None or len(prediction) == 0:
                return None
            
            pred_value = prediction[0]
            
            # Calculate confidence
            confidence = self.ml_pipeline.calculate_prediction_confidence(pred_value)
            
            if confidence < self.min_ml_confidence:
                return None
            
            # Determine action based on prediction
            current_price = self.ml_pipeline.data.iloc[-1]['close']
            predicted_return = (pred_value - current_price) / current_price
            
            # Generate signal if prediction is significant
            if abs(predicted_return) > 0.01:  # 1% threshold
                action = "buy" if predicted_return > 0 else "sell"
                
                # Calculate targets
                atr = self._calculate_atr_from_df(self.ml_pipeline.data)
                
                if action == "buy":
                    stop_loss = current_price - (2 * atr)
                    take_profit = pred_value  # Use prediction as target
                else:
                    stop_loss = current_price + (2 * atr)
                    take_profit = pred_value
                
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    reasoning=f"ML prediction: {predicted_return:.2%} return expected",
                    metadata={
                        "ml_prediction": pred_value,
                        "predicted_return": predicted_return,
                        "entry_price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "model_type": self.ml_pipeline.model_type.value if self.ml_pipeline.model_type else "unknown"
                    }
                )
                
                # Cache the prediction
                self.prediction_cache[cache_key] = {
                    'signal': signal,
                    'timestamp': datetime.now()
                }
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            return None
    
    async def _combine_signals(
        self,
        ta_signal: Optional[Signal],
        ml_signal: Optional[Signal],
        context: AgentContext
    ) -> Optional[Signal]:
        """
        Combine technical and ML signals
        
        Args:
            ta_signal: Technical analysis signal
            ml_signal: ML prediction signal
            context: Agent context
            
        Returns:
            Combined signal
        """
        try:
            # If only one signal exists, use it with adjusted confidence
            if ta_signal and not ml_signal:
                ta_signal.confidence *= (1 - self.ml_weight)
                ta_signal.metadata['signal_type'] = 'technical_only'
                return ta_signal if ta_signal.confidence >= 0.5 else None
            
            if ml_signal and not ta_signal:
                ml_signal.confidence *= self.ml_weight
                ml_signal.metadata['signal_type'] = 'ml_only'
                return ml_signal if ml_signal.confidence >= 0.5 else None
            
            if not ta_signal and not ml_signal:
                return None
            
            # Both signals exist - combine them
            if ta_signal.action == ml_signal.action:
                # Signals agree - strong signal
                combined_confidence = (
                    ta_signal.confidence * (1 - self.ml_weight) +
                    ml_signal.confidence * self.ml_weight
                ) * 1.2  # Boost for agreement
                
                combined_confidence = min(combined_confidence, 1.0)
                
                # Use ML targets if available, otherwise TA
                stop_loss = ml_signal.metadata.get('stop_loss', ta_signal.metadata.get('stop_loss'))
                take_profit = ml_signal.metadata.get('take_profit', ta_signal.metadata.get('take_profit'))
                
                return Signal(
                    symbol=context.symbol,
                    action=ta_signal.action,
                    confidence=combined_confidence,
                    reasoning=f"ML and TA agree: {ta_signal.reasoning} | {ml_signal.reasoning}",
                    metadata={
                        'signal_type': 'combined_agreement',
                        'ta_confidence': ta_signal.confidence,
                        'ml_confidence': ml_signal.confidence,
                        'entry_price': ta_signal.metadata.get('entry_price'),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        **ta_signal.metadata,
                        **{f"ml_{k}": v for k, v in ml_signal.metadata.items()}
                    }
                )
            else:
                # Signals disagree - use weighted average or skip
                ta_weight = ta_signal.confidence * (1 - self.ml_weight)
                ml_weight = ml_signal.confidence * self.ml_weight
                
                if ta_weight > ml_weight:
                    dominant_signal = ta_signal
                    dominant_signal.confidence = ta_weight - ml_weight
                else:
                    dominant_signal = ml_signal
                    dominant_signal.confidence = ml_weight - ta_weight
                
                dominant_signal.metadata['signal_type'] = 'combined_disagreement'
                dominant_signal.reasoning = f"Mixed signals (reduced confidence): {dominant_signal.reasoning}"
                
                # Only return if confidence is still high enough
                return dominant_signal if dominant_signal.confidence >= 0.3 else None
                
        except Exception as e:
            logger.error(f"Signal combination failed: {str(e)}")
            return ta_signal  # Fallback to TA signal
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50  # Neutral
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return 0, 0
        
        # Calculate EMAs
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd = ema_12 - ema_26
        signal = self._calculate_ema([macd], 9) if len(prices) > 35 else macd
        
        return macd, signal
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate exponential moving average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        
        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_atr(self, ohlcv: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(ohlcv) < period + 1:
            return 0
        
        true_ranges = []
        for i in range(1, min(period + 1, len(ohlcv))):
            high = ohlcv[-i]['high']
            low = ohlcv[-i]['low']
            prev_close = ohlcv[-i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return np.mean(true_ranges) if true_ranges else 0
    
    def _calculate_atr_from_df(self, df) -> float:
        """Calculate ATR from dataframe"""
        if len(df) < 15:
            return df['high'].iloc[-1] - df['low'].iloc[-1]
        
        high = df['high'].iloc[-14:].values
        low = df['low'].iloc[-14:].values
        close = df['close'].iloc[-15:-1].values
        
        tr1 = high - low
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr)
    
    async def _cache_signal(self, signal: Signal) -> None:
        """Cache the generated signal"""
        self.last_signal = signal
        self.last_analysis_time = datetime.now()
    
    async def _on_model_trained(self, event) -> None:
        """Handle new model trained event"""
        try:
            data = event.data
            model_path = data.get('model_path')
            
            if model_path:
                logger.info(f"Loading newly trained model: {model_path}")
                self.model_path = model_path
                await self.ml_pipeline.load_model(model_path)
                self.model_loaded = True
                
                # Clear prediction cache with new model
                self.prediction_cache.clear()
                
        except Exception as e:
            logger.error(f"Failed to load new model: {str(e)}")
    
    def get_required_data_types(self) -> List[str]:
        """Get required data types for analysis"""
        return ["ohlcv", "market_depth", "trades"]
    
    async def train_model(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        model_type: ModelType = ModelType.LIGHTGBM
    ) -> Dict[str, Any]:
        """
        Train a new ML model
        
        Args:
            symbols: Symbols to train on
            start_date: Training start date
            end_date: Training end date
            model_type: Type of model to train
            
        Returns:
            Training results
        """
        try:
            # Configure pipeline
            self.ml_pipeline.model_type = model_type
            self.ml_pipeline.optimize_hyperparams = True
            
            # Prepare data
            await self.ml_pipeline.prepare_data(symbols, start_date, end_date)
            
            # Train model
            metrics = await self.ml_pipeline.train_model()
            
            # Evaluate
            evaluation = await self.ml_pipeline.evaluate_model()
            
            # Save model
            model_name = f"ml_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = await self.ml_pipeline.save_model(f"models/{model_name}")
            
            # Load the new model
            self.model_path = model_path
            self.model_loaded = True
            
            return {
                "model_path": model_path,
                "metrics": metrics,
                "evaluation": evaluation,
                "symbols": symbols,
                "date_range": f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise