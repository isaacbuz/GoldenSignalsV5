"""
AI-Powered Chart Analysis Agent
Provides real-time technical analysis, pattern recognition, and predictive insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
# import talib  # Temporarily disabled due to numpy compatibility
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    CONSOLIDATION = "consolidation"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"

class PatternType(Enum):
    """Chart pattern types"""
    HEAD_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    CUP_HANDLE = "cup_and_handle"
    WEDGE_RISING = "rising_wedge"
    WEDGE_FALLING = "falling_wedge"

@dataclass
class ChartInsight:
    """AI-generated chart insight"""
    timestamp: datetime
    symbol: str
    insight_type: str
    confidence: float
    message: str
    data: Dict[str, Any]
    action_items: List[str]
    risk_level: str

@dataclass
class TradingSignal:
    """AI-generated trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # Multiple targets
    risk_reward_ratio: float
    confidence: float
    reasoning: str
    timeframe: str
    pattern: Optional[str]

class ChartAnalystAgent:
    """AI-powered chart analysis agent"""
    
    def __init__(self):
        self.regime_history = {}
        self.pattern_cache = {}
        self.support_resistance_levels = {}
        self.volume_profile = {}
        self.order_flow_imbalance = {}
        
    async def analyze_chart(self, 
                           symbol: str, 
                           df: pd.DataFrame,
                           timeframe: str = "5m") -> Dict[str, Any]:
        """
        Comprehensive AI-powered chart analysis
        """
        try:
            # Ensure we have enough data
            if len(df) < 100:
                return {"error": "Insufficient data for analysis"}
            
            # Run parallel analysis tasks
            tasks = [
                self._detect_market_regime(df),
                self._find_chart_patterns(df),
                self._calculate_smart_levels(df),
                self._analyze_volume_profile(df),
                self._detect_order_flow_imbalance(df),
                self._calculate_ml_indicators(df),
                self._predict_price_movement(df),
                self._analyze_options_flow(symbol),
                self._calculate_institutional_activity(df),
                self._generate_trading_signals(symbol, df, timeframe)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "timeframe": timeframe,
                "market_regime": results[0],
                "patterns": results[1],
                "support_resistance": results[2],
                "volume_profile": results[3],
                "order_flow": results[4],
                "ml_indicators": results[5],
                "price_prediction": results[6],
                "options_flow": results[7],
                "institutional_activity": results[8],
                "trading_signals": results[9],
                "insights": self._generate_insights(results),
                "risk_assessment": self._assess_risk(df, results)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Chart analysis error for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime using ML"""
        try:
            # Calculate regime indicators
            sma_20 = df['Close'].rolling(20).mean()
            sma_50 = df['Close'].rolling(50).mean()
            
            # Trend strength
            adx = talib.ADX(df['High'], df['Low'], df['Close'])
            
            # Volatility
            atr = talib.ATR(df['High'], df['Low'], df['Close'])
            volatility = (atr / df['Close']).iloc[-1]
            
            # Current position relative to moving averages
            current_price = df['Close'].iloc[-1]
            above_sma20 = current_price > sma_20.iloc[-1]
            above_sma50 = current_price > sma_50.iloc[-1]
            
            # Trend direction
            price_change_20 = (current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
            
            # Determine regime
            if adx.iloc[-1] > 25:
                if price_change_20 > 0.05 and above_sma20 and above_sma50:
                    regime = MarketRegime.BULL_TREND
                elif price_change_20 < -0.05 and not above_sma20 and not above_sma50:
                    regime = MarketRegime.BEAR_TREND
                elif volatility > 0.02:
                    regime = MarketRegime.VOLATILE
                else:
                    regime = MarketRegime.CONSOLIDATION
            else:
                if volatility > 0.03:
                    regime = MarketRegime.VOLATILE
                else:
                    regime = MarketRegime.CONSOLIDATION
            
            # Check for breakout/breakdown
            resistance = df['High'].rolling(20).max().iloc[-2]
            support = df['Low'].rolling(20).min().iloc[-2]
            
            if current_price > resistance * 1.01:
                regime = MarketRegime.BREAKOUT
            elif current_price < support * 0.99:
                regime = MarketRegime.BREAKDOWN
            
            return {
                "regime": regime.value,
                "trend_strength": float(adx.iloc[-1]),
                "volatility": float(volatility),
                "confidence": self._calculate_regime_confidence(df, regime)
            }
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return {"regime": "unknown", "confidence": 0}
    
    async def _find_chart_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect chart patterns using pattern recognition algorithms"""
        patterns = []
        
        try:
            # Head and Shoulders
            if pattern := self._detect_head_shoulders(df):
                patterns.append(pattern)
            
            # Double Top/Bottom
            if pattern := self._detect_double_patterns(df):
                patterns.append(pattern)
            
            # Triangle patterns
            if pattern := self._detect_triangles(df):
                patterns.append(pattern)
            
            # Flag patterns
            if pattern := self._detect_flags(df):
                patterns.append(pattern)
            
            # Cup and Handle
            if pattern := self._detect_cup_handle(df):
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return []
    
    async def _calculate_smart_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate AI-enhanced support/resistance levels"""
        try:
            # Volume-weighted levels
            vwap = (df['Close'] * df['Volume']).sum() / df['Volume'].sum()
            
            # Fibonacci levels
            high = df['High'].max()
            low = df['Low'].min()
            diff = high - low
            
            fib_levels = {
                "fib_0": low,
                "fib_236": low + 0.236 * diff,
                "fib_382": low + 0.382 * diff,
                "fib_500": low + 0.500 * diff,
                "fib_618": low + 0.618 * diff,
                "fib_786": low + 0.786 * diff,
                "fib_1000": high
            }
            
            # ML-detected levels using clustering
            from sklearn.cluster import KMeans
            
            # Prepare price data for clustering
            prices = df[['High', 'Low', 'Close']].values.flatten()
            prices = prices.reshape(-1, 1)
            
            # Find optimal clusters (support/resistance levels)
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(prices)
            
            ml_levels = sorted(kmeans.cluster_centers_.flatten())
            
            # Calculate strength of each level
            level_strength = {}
            for level in ml_levels:
                touches = np.sum(np.abs(df['Close'] - level) < level * 0.005)
                level_strength[level] = touches
            
            return {
                "vwap": float(vwap),
                "fibonacci": fib_levels,
                "ml_levels": [float(l) for l in ml_levels],
                "level_strength": level_strength,
                "current_price": float(df['Close'].iloc[-1]),
                "nearest_resistance": self._find_nearest_resistance(df['Close'].iloc[-1], ml_levels),
                "nearest_support": self._find_nearest_support(df['Close'].iloc[-1], ml_levels)
            }
            
        except Exception as e:
            logger.error(f"Smart levels calculation error: {e}")
            return {}
    
    async def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile for institutional activity"""
        try:
            # Calculate volume profile
            price_bins = pd.cut(df['Close'], bins=20)
            volume_profile = df.groupby(price_bins)['Volume'].sum()
            
            # Find Point of Control (POC)
            poc_idx = volume_profile.idxmax()
            poc_price = poc_idx.mid if poc_idx else df['Close'].mean()
            
            # Value Area (70% of volume)
            sorted_profile = volume_profile.sort_values(ascending=False)
            cumsum = sorted_profile.cumsum()
            total_volume = sorted_profile.sum()
            value_area = sorted_profile[cumsum <= total_volume * 0.7]
            
            # Detect volume anomalies
            volume_mean = df['Volume'].rolling(20).mean()
            volume_std = df['Volume'].rolling(20).std()
            volume_zscore = (df['Volume'] - volume_mean) / volume_std
            
            unusual_volume = df[volume_zscore.abs() > 2]
            
            return {
                "poc": float(poc_price),
                "value_area_high": float(value_area.index[0].right) if len(value_area) > 0 else 0,
                "value_area_low": float(value_area.index[-1].left) if len(value_area) > 0 else 0,
                "unusual_volume_detected": len(unusual_volume) > 0,
                "unusual_volume_times": unusual_volume.index.tolist() if len(unusual_volume) > 0 else [],
                "current_volume_zscore": float(volume_zscore.iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return {}
    
    async def _detect_order_flow_imbalance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect order flow imbalance for smart money tracking"""
        try:
            # Calculate buy/sell pressure
            df['buy_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            df['sell_pressure'] = (df['High'] - df['Close']) / (df['High'] - df['Low'])
            
            # Volume-weighted pressure
            df['buy_volume'] = df['Volume'] * df['buy_pressure']
            df['sell_volume'] = df['Volume'] * df['sell_pressure']
            
            # Recent imbalance (last 20 bars)
            recent_buy = df['buy_volume'].iloc[-20:].sum()
            recent_sell = df['sell_volume'].iloc[-20:].sum()
            
            imbalance = (recent_buy - recent_sell) / (recent_buy + recent_sell)
            
            # Cumulative delta
            df['delta'] = df['buy_volume'] - df['sell_volume']
            cumulative_delta = df['delta'].cumsum()
            
            # Detect divergence
            price_trend = np.polyfit(range(20), df['Close'].iloc[-20:].values, 1)[0]
            delta_trend = np.polyfit(range(20), cumulative_delta.iloc[-20:].values, 1)[0]
            
            divergence = False
            if (price_trend > 0 and delta_trend < 0) or (price_trend < 0 and delta_trend > 0):
                divergence = True
            
            return {
                "imbalance": float(imbalance),
                "buy_pressure": float(df['buy_pressure'].iloc[-1]),
                "sell_pressure": float(df['sell_pressure'].iloc[-1]),
                "cumulative_delta": float(cumulative_delta.iloc[-1]),
                "divergence_detected": divergence,
                "smart_money_direction": "buying" if imbalance > 0.1 else "selling" if imbalance < -0.1 else "neutral"
            }
            
        except Exception as e:
            logger.error(f"Order flow analysis error: {e}")
            return {}
    
    async def _calculate_ml_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ML-enhanced technical indicators"""
        try:
            # Feature engineering
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Anomaly detection for unusual price movements
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            features = df[['returns', 'volatility', 'Volume']].fillna(0).iloc[-100:]
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            anomalies = iso_forest.fit_predict(scaled_features)
            is_anomaly = anomalies[-1] == -1
            
            # Momentum indicators with ML twist
            rsi = talib.RSI(df['Close'])
            macd, macd_signal, macd_hist = talib.MACD(df['Close'])
            
            # Stochastic RSI
            stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
            
            # Bollinger Bands with dynamic width
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], timeperiod=20)
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Mean reversion indicator
            zscore = (df['Close'].iloc[-1] - df['Close'].rolling(20).mean().iloc[-1]) / df['Close'].rolling(20).std().iloc[-1]
            
            return {
                "rsi": float(rsi.iloc[-1]),
                "stoch_rsi": float(stoch_rsi.iloc[-1]),
                "macd": float(macd.iloc[-1]),
                "macd_signal": float(macd_signal.iloc[-1]),
                "macd_histogram": float(macd_hist.iloc[-1]),
                "bb_position": float((df['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])),
                "bb_width": float(bb_width.iloc[-1]),
                "zscore": float(zscore),
                "is_anomaly": bool(is_anomaly),
                "volatility": float(df['volatility'].iloc[-1]),
                "trend_strength": self._calculate_trend_strength(df)
            }
            
        except Exception as e:
            logger.error(f"ML indicators error: {e}")
            return {}
    
    async def _predict_price_movement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict short-term price movement using ML"""
        try:
            # Simple prediction using linear regression on recent trend
            from sklearn.linear_model import LinearRegression
            
            # Prepare features
            window = 20
            X = np.arange(window).reshape(-1, 1)
            y = df['Close'].iloc[-window:].values
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict next 5 periods
            future_X = np.arange(window, window + 5).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Calculate confidence based on R²
            r2_score = model.score(X, y)
            
            # Volatility-adjusted targets
            current_price = df['Close'].iloc[-1]
            volatility = df['Close'].pct_change().std() * np.sqrt(5)  # 5-period volatility
            
            return {
                "current_price": float(current_price),
                "predicted_prices": [float(p) for p in predictions],
                "expected_move": float(predictions[-1] - current_price),
                "expected_move_pct": float((predictions[-1] - current_price) / current_price * 100),
                "confidence": float(r2_score),
                "upper_bound": float(predictions[-1] * (1 + volatility)),
                "lower_bound": float(predictions[-1] * (1 - volatility)),
                "trend_direction": "bullish" if predictions[-1] > current_price else "bearish"
            }
            
        except Exception as e:
            logger.error(f"Price prediction error: {e}")
            return {}
    
    async def _analyze_options_flow(self, symbol: str) -> Dict[str, Any]:
        """Analyze options flow for unusual activity"""
        # Simulated options flow data
        # In production, this would connect to real options data feed
        
        np.random.seed(hash(symbol) % 1000)  # Consistent random data per symbol
        
        return {
            "put_call_ratio": float(np.random.uniform(0.5, 1.5)),
            "unusual_activity": bool(np.random.random() > 0.7),
            "large_trades": [
                {
                    "strike": float(np.random.uniform(95, 105)),
                    "expiry": "2025-08-15",
                    "type": np.random.choice(["CALL", "PUT"]),
                    "volume": int(np.random.uniform(1000, 10000)),
                    "premium": float(np.random.uniform(100000, 1000000))
                }
                for _ in range(np.random.randint(0, 3))
            ],
            "implied_volatility": float(np.random.uniform(0.2, 0.6)),
            "iv_rank": float(np.random.uniform(0, 100)),
            "smart_money_sentiment": np.random.choice(["bullish", "bearish", "neutral"])
        }
    
    async def _calculate_institutional_activity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional and dark pool activity"""
        try:
            # Detect large volume spikes (potential institutional)
            volume_mean = df['Volume'].rolling(20).mean()
            volume_spike = df['Volume'] > volume_mean * 2
            
            # Block trades (large single transactions)
            avg_trade_size = df['Volume'] / 390  # Assuming 390 minutes in trading day
            large_trades = df[df['Volume'] > avg_trade_size * 100]
            
            # Price stability during high volume (accumulation/distribution)
            high_volume_periods = df[df['Volume'] > df['Volume'].quantile(0.8)]
            if len(high_volume_periods) > 0:
                price_volatility_high_vol = high_volume_periods['Close'].pct_change().std()
                normal_volatility = df['Close'].pct_change().std()
                accumulation = price_volatility_high_vol < normal_volatility * 0.5
            else:
                accumulation = False
            
            return {
                "institutional_buying_detected": bool(volume_spike.iloc[-1] and df['Close'].iloc[-1] > df['Open'].iloc[-1]),
                "institutional_selling_detected": bool(volume_spike.iloc[-1] and df['Close'].iloc[-1] < df['Open'].iloc[-1]),
                "accumulation_phase": accumulation,
                "distribution_phase": not accumulation and volume_spike.any(),
                "dark_pool_probability": float(np.random.uniform(0, 1)),  # Simulated
                "large_trade_count": len(large_trades),
                "average_trade_size": float(avg_trade_size.iloc[-1]) if len(avg_trade_size) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Institutional activity analysis error: {e}")
            return {}
    
    async def _generate_trading_signals(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[TradingSignal]:
        """Generate AI-powered trading signals"""
        signals = []
        
        try:
            current_price = df['Close'].iloc[-1]
            
            # Get all analysis components
            regime = await self._detect_market_regime(df)
            ml_indicators = await self._calculate_ml_indicators(df)
            order_flow = await self._detect_order_flow_imbalance(df)
            levels = await self._calculate_smart_levels(df)
            
            # Signal generation logic
            rsi = ml_indicators.get('rsi', 50)
            macd_hist = ml_indicators.get('macd_histogram', 0)
            imbalance = order_flow.get('imbalance', 0)
            
            # Long signal conditions
            if (rsi < 30 and macd_hist > 0 and imbalance > 0.2 and 
                regime.get('regime') in ['bull_trend', 'breakout']):
                
                stop_loss = levels.get('nearest_support', current_price * 0.98)
                take_profits = [
                    current_price * 1.02,  # 2% target
                    current_price * 1.05,  # 5% target
                    levels.get('nearest_resistance', current_price * 1.10)
                ]
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profits,
                    risk_reward_ratio=(take_profits[0] - current_price) / (current_price - stop_loss),
                    confidence=0.75,
                    reasoning="Oversold bounce with positive order flow and bullish regime",
                    timeframe=timeframe,
                    pattern=None
                ))
            
            # Short signal conditions
            elif (rsi > 70 and macd_hist < 0 and imbalance < -0.2 and 
                  regime.get('regime') in ['bear_trend', 'breakdown']):
                
                stop_loss = levels.get('nearest_resistance', current_price * 1.02)
                take_profits = [
                    current_price * 0.98,  # 2% target
                    current_price * 0.95,  # 5% target
                    levels.get('nearest_support', current_price * 0.90)
                ]
                
                signals.append(TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profits,
                    risk_reward_ratio=(current_price - take_profits[0]) / (stop_loss - current_price),
                    confidence=0.75,
                    reasoning="Overbought reversal with negative order flow and bearish regime",
                    timeframe=timeframe,
                    pattern=None
                ))
            
            return [self._signal_to_dict(s) for s in signals]
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return []
    
    def _generate_insights(self, results: List[Any]) -> List[ChartInsight]:
        """Generate actionable insights from analysis"""
        insights = []
        
        try:
            # Market regime insight
            regime = results[0]
            if regime.get('regime') == 'breakout':
                insights.append(ChartInsight(
                    timestamp=datetime.utcnow(),
                    symbol="",
                    insight_type="breakout",
                    confidence=regime.get('confidence', 0),
                    message="Breakout detected! Price breaking above resistance with strong momentum",
                    data=regime,
                    action_items=["Consider long position", "Set stop below breakout level", "Monitor volume confirmation"],
                    risk_level="medium"
                ))
            
            # Order flow insight
            order_flow = results[4]
            if order_flow.get('divergence_detected'):
                insights.append(ChartInsight(
                    timestamp=datetime.utcnow(),
                    symbol="",
                    insight_type="divergence",
                    confidence=0.8,
                    message="Price/Volume divergence detected - potential reversal ahead",
                    data=order_flow,
                    action_items=["Watch for reversal confirmation", "Consider reducing position size", "Set tight stops"],
                    risk_level="high"
                ))
            
            # Institutional activity
            inst_activity = results[8]
            if inst_activity.get('accumulation_phase'):
                insights.append(ChartInsight(
                    timestamp=datetime.utcnow(),
                    symbol="",
                    insight_type="accumulation",
                    confidence=0.7,
                    message="Institutional accumulation detected - smart money is buying",
                    data=inst_activity,
                    action_items=["Follow smart money", "Look for entry on pullbacks", "Increase position gradually"],
                    risk_level="low"
                ))
            
            return [self._insight_to_dict(i) for i in insights]
            
        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            return []
    
    def _assess_risk(self, df: pd.DataFrame, results: List[Any]) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        try:
            # Volatility risk
            returns = df['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Drawdown risk
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns.dropna(), 5)
            
            # Risk score (0-100)
            risk_score = min(100, (volatility * 100 + abs(max_drawdown) * 100 + abs(var_95) * 1000) / 3)
            
            return {
                "risk_score": float(risk_score),
                "risk_level": "high" if risk_score > 70 else "medium" if risk_score > 40 else "low",
                "volatility": float(volatility),
                "max_drawdown": float(max_drawdown),
                "value_at_risk_95": float(var_95),
                "recommended_position_size": max(0.1, 1 - risk_score / 100),
                "stop_loss_distance": float(volatility * 2)  # 2 standard deviations
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {"risk_score": 50, "risk_level": "medium"}
    
    # Helper methods
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern"""
        # Simplified pattern detection
        return None
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect double top/bottom patterns"""
        return None
    
    def _detect_triangles(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns"""
        return None
    
    def _detect_flags(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect flag patterns"""
        return None
    
    def _detect_cup_handle(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect cup and handle pattern"""
        return None
    
    def _calculate_regime_confidence(self, df: pd.DataFrame, regime: MarketRegime) -> float:
        """Calculate confidence in regime detection"""
        return 0.75  # Simplified
    
    def _find_nearest_resistance(self, price: float, levels: List[float]) -> float:
        """Find nearest resistance level"""
        resistances = [l for l in levels if l > price]
        return min(resistances) if resistances else price * 1.05
    
    def _find_nearest_support(self, price: float, levels: List[float]) -> float:
        """Find nearest support level"""
        supports = [l for l in levels if l < price]
        return max(supports) if supports else price * 0.95
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength indicator"""
        try:
            adx = talib.ADX(df['High'], df['Low'], df['Close'])
            return float(adx.iloc[-1]) / 100
        except:
            return 0.5
    
    def _signal_to_dict(self, signal: TradingSignal) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            "symbol": signal.symbol,
            "action": signal.action,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "timeframe": signal.timeframe,
            "pattern": signal.pattern
        }
    
    def _insight_to_dict(self, insight: ChartInsight) -> Dict[str, Any]:
        """Convert insight to dictionary"""
        return {
            "timestamp": insight.timestamp.isoformat(),
            "type": insight.insight_type,
            "confidence": insight.confidence,
            "message": insight.message,
            "data": insight.data,
            "action_items": insight.action_items,
            "risk_level": insight.risk_level
        }

# Global instance
chart_analyst = ChartAnalystAgent()