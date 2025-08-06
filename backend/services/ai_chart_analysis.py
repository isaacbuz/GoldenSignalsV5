"""
AI-Powered Chart Analysis Service
Provides intelligent insights, pattern recognition, and predictive analytics for charts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AIChartAnalyzer:
    """AI-powered chart analysis with pattern recognition and predictions"""
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle': self._detect_triangle,
            'wedge': self._detect_wedge,
            'flag': self._detect_flag,
            'channel': self._detect_channel,
            'cup_and_handle': self._detect_cup_and_handle
        }
        
        self.ml_model = LinearRegression()
        self.scaler = StandardScaler()
    
    async def analyze_chart(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Comprehensive AI analysis of chart data
        
        Returns:
            - Pattern recognition results
            - Support/resistance levels
            - Price predictions
            - Trend analysis
            - Volume insights
            - Risk metrics
            - Trading recommendations
        """
        try:
            if data.empty or len(data) < 20:
                return {"error": "Insufficient data for analysis"}
            
            # Ensure we have all required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                return {"error": "Missing required OHLCV data"}
            
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "data_points": len(data),
                "timeframe": self._detect_timeframe(data)
            }
            
            # Core Analysis
            analysis["patterns"] = self._detect_patterns(data)
            analysis["support_resistance"] = self._calculate_support_resistance(data)
            analysis["trend_analysis"] = self._analyze_trend(data)
            analysis["predictions"] = self._generate_predictions(data)
            analysis["volume_analysis"] = self._analyze_volume(data)
            analysis["volatility_analysis"] = self._analyze_volatility(data)
            analysis["momentum_indicators"] = self._calculate_momentum(data)
            analysis["anomalies"] = self._detect_anomalies(data)
            
            # AI Insights
            analysis["ai_insights"] = self._generate_ai_insights(analysis)
            analysis["risk_assessment"] = self._assess_risk(data, analysis)
            analysis["trading_recommendation"] = self._generate_recommendation(analysis)
            
            # Chart Annotations
            analysis["annotations"] = self._generate_annotations(data, analysis)
            
            # Confidence Score
            analysis["confidence_score"] = self._calculate_confidence(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI chart analysis: {e}")
            return {"error": str(e)}
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect the timeframe of the data"""
        if len(data) < 2:
            return "unknown"
        
        # Calculate average time between data points
        if hasattr(data.index, 'to_pydatetime'):
            time_diffs = pd.Series(data.index).diff().dropna()
            avg_diff = time_diffs.mean()
            
            if avg_diff <= pd.Timedelta(minutes=5):
                return "intraday"
            elif avg_diff <= pd.Timedelta(hours=1):
                return "hourly"
            elif avg_diff <= pd.Timedelta(days=1):
                return "daily"
            elif avg_diff <= pd.Timedelta(days=7):
                return "weekly"
            else:
                return "monthly"
        
        return "daily"  # Default
    
    def _detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect chart patterns using AI"""
        detected_patterns = []
        
        for pattern_name, detector_func in self.patterns.items():
            result = detector_func(data)
            if result and result.get('detected'):
                detected_patterns.append({
                    "pattern": pattern_name,
                    "confidence": result.get('confidence', 0),
                    "position": result.get('position'),
                    "description": result.get('description'),
                    "implications": result.get('implications')
                })
        
        return sorted(detected_patterns, key=lambda x: x['confidence'], reverse=True)
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect head and shoulders pattern"""
        if len(data) < 50:
            return {"detected": False}
        
        prices = data['Close'].values
        highs = data['High'].values
        
        # Find peaks
        peaks, properties = find_peaks(highs, distance=5, prominence=prices.std() * 0.5)
        
        if len(peaks) >= 3:
            # Check for head and shoulders formation
            for i in range(len(peaks) - 2):
                left_shoulder = highs[peaks[i]]
                head = highs[peaks[i + 1]]
                right_shoulder = highs[peaks[i + 2]]
                
                # Classic head and shoulders pattern
                if (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                    
                    return {
                        "detected": True,
                        "confidence": 0.75,
                        "position": int(peaks[i + 1]),
                        "description": "Head and Shoulders pattern detected",
                        "implications": "Bearish reversal pattern. Potential trend reversal from bullish to bearish."
                    }
        
        return {"detected": False}
    
    def _detect_double_top(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect double top pattern"""
        if len(data) < 30:
            return {"detected": False}
        
        highs = data['High'].values
        peaks, _ = find_peaks(highs, distance=5, prominence=highs.std() * 0.3)
        
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1 = highs[peaks[i]]
                peak2 = highs[peaks[i + 1]]
                
                # Check if peaks are similar height (within 2%)
                if abs(peak1 - peak2) / peak1 < 0.02:
                    # Check for valley between peaks
                    valley_start = peaks[i] + 1
                    valley_end = peaks[i + 1]
                    valley = highs[valley_start:valley_end].min()
                    
                    if valley < peak1 * 0.95:  # Valley at least 5% below peaks
                        return {
                            "detected": True,
                            "confidence": 0.70,
                            "position": int(peaks[i + 1]),
                            "description": "Double Top pattern detected",
                            "implications": "Bearish reversal pattern. Indicates potential downward price movement."
                        }
        
        return {"detected": False}
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect double bottom pattern"""
        if len(data) < 30:
            return {"detected": False}
        
        lows = data['Low'].values
        troughs = argrelextrema(lows, np.less, order=5)[0]
        
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1 = lows[troughs[i]]
                trough2 = lows[troughs[i + 1]]
                
                # Check if troughs are similar depth (within 2%)
                if abs(trough1 - trough2) / trough1 < 0.02:
                    # Check for peak between troughs
                    peak_start = troughs[i] + 1
                    peak_end = troughs[i + 1]
                    peak = lows[peak_start:peak_end].max()
                    
                    if peak > trough1 * 1.05:  # Peak at least 5% above troughs
                        return {
                            "detected": True,
                            "confidence": 0.70,
                            "position": int(troughs[i + 1]),
                            "description": "Double Bottom pattern detected",
                            "implications": "Bullish reversal pattern. Indicates potential upward price movement."
                        }
        
        return {"detected": False}
    
    def _detect_triangle(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(data) < 20:
            return {"detected": False}
        
        highs = data['High'].values[-20:]
        lows = data['Low'].values[-20:]
        
        # Fit trend lines
        x = np.arange(len(highs))
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]
        
        # Detect pattern type
        if abs(high_slope) < highs.std() * 0.01 and low_slope > 0:
            return {
                "detected": True,
                "confidence": 0.65,
                "position": len(data) - 10,
                "description": "Ascending Triangle pattern detected",
                "implications": "Bullish continuation pattern. Breakout likely to the upside."
            }
        elif high_slope < 0 and abs(low_slope) < lows.std() * 0.01:
            return {
                "detected": True,
                "confidence": 0.65,
                "position": len(data) - 10,
                "description": "Descending Triangle pattern detected",
                "implications": "Bearish continuation pattern. Breakdown likely to the downside."
            }
        elif high_slope < 0 and low_slope > 0 and abs(high_slope + low_slope) < 0.01:
            return {
                "detected": True,
                "confidence": 0.60,
                "position": len(data) - 10,
                "description": "Symmetrical Triangle pattern detected",
                "implications": "Neutral pattern. Breakout direction uncertain, watch for volume confirmation."
            }
        
        return {"detected": False}
    
    def _detect_wedge(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect wedge patterns"""
        if len(data) < 20:
            return {"detected": False}
        
        highs = data['High'].values[-20:]
        lows = data['Low'].values[-20:]
        
        # Fit trend lines
        x = np.arange(len(highs))
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]
        
        # Both lines moving in same direction but converging
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
            return {
                "detected": True,
                "confidence": 0.60,
                "position": len(data) - 10,
                "description": "Rising Wedge pattern detected",
                "implications": "Bearish reversal pattern. Price likely to break down."
            }
        elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            return {
                "detected": True,
                "confidence": 0.60,
                "position": len(data) - 10,
                "description": "Falling Wedge pattern detected",
                "implications": "Bullish reversal pattern. Price likely to break up."
            }
        
        return {"detected": False}
    
    def _detect_flag(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect flag patterns"""
        if len(data) < 15:
            return {"detected": False}
        
        # Look for strong move followed by consolidation
        prices = data['Close'].values
        
        # Check last 15 periods
        initial_move = prices[-15:-10]
        consolidation = prices[-10:]
        
        initial_change = (initial_move[-1] - initial_move[0]) / initial_move[0]
        consolidation_range = consolidation.max() - consolidation.min()
        consolidation_avg_range = consolidation_range / consolidation.mean()
        
        if abs(initial_change) > 0.05 and consolidation_avg_range < 0.02:
            if initial_change > 0:
                pattern_type = "Bull Flag"
                implication = "Bullish continuation pattern. Uptrend likely to continue."
            else:
                pattern_type = "Bear Flag"
                implication = "Bearish continuation pattern. Downtrend likely to continue."
            
            return {
                "detected": True,
                "confidence": 0.65,
                "position": len(data) - 5,
                "description": f"{pattern_type} pattern detected",
                "implications": implication
            }
        
        return {"detected": False}
    
    def _detect_channel(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect channel patterns"""
        if len(data) < 20:
            return {"detected": False}
        
        highs = data['High'].values[-20:]
        lows = data['Low'].values[-20:]
        
        # Fit parallel trend lines
        x = np.arange(len(highs))
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]
        
        # Check if slopes are parallel (within 10%)
        if abs(high_slope - low_slope) / abs(high_slope + 0.0001) < 0.1:
            if abs(high_slope) < 0.01:
                channel_type = "Horizontal Channel"
                implication = "Range-bound trading. Buy at support, sell at resistance."
            elif high_slope > 0:
                channel_type = "Ascending Channel"
                implication = "Uptrend channel. Buy at lower trendline, sell at upper trendline."
            else:
                channel_type = "Descending Channel"
                implication = "Downtrend channel. Sell at upper trendline, buy at lower trendline."
            
            return {
                "detected": True,
                "confidence": 0.70,
                "position": len(data) - 10,
                "description": f"{channel_type} pattern detected",
                "implications": implication
            }
        
        return {"detected": False}
    
    def _detect_cup_and_handle(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect cup and handle pattern"""
        if len(data) < 50:
            return {"detected": False}
        
        prices = data['Close'].values
        
        # Look for U-shape followed by small consolidation
        # Simplified detection
        first_third = prices[:len(prices)//3]
        middle_third = prices[len(prices)//3:2*len(prices)//3]
        last_third = prices[2*len(prices)//3:]
        
        # Check for U-shape
        if (first_third.mean() > middle_third.mean() and 
            last_third.mean() > middle_third.mean() and
            abs(first_third.mean() - last_third.mean()) / first_third.mean() < 0.05):
            
            # Check for handle (small consolidation at the end)
            handle = prices[-10:]
            handle_range = (handle.max() - handle.min()) / handle.mean()
            
            if handle_range < 0.03:
                return {
                    "detected": True,
                    "confidence": 0.55,
                    "position": len(data) - 5,
                    "description": "Cup and Handle pattern detected",
                    "implications": "Bullish continuation pattern. Breakout expected above handle resistance."
                }
        
        return {"detected": False}
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels using AI"""
        prices = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find local maxima and minima
        peaks = argrelextrema(highs, np.greater, order=5)[0]
        troughs = argrelextrema(lows, np.less, order=5)[0]
        
        # Cluster analysis for support/resistance
        resistance_levels = []
        support_levels = []
        
        if len(peaks) > 0:
            peak_prices = highs[peaks]
            # Cluster nearby peaks
            resistance_levels = self._cluster_levels(peak_prices)
        
        if len(troughs) > 0:
            trough_prices = lows[troughs]
            # Cluster nearby troughs
            support_levels = self._cluster_levels(trough_prices)
        
        # Add psychological levels (round numbers)
        current_price = prices[-1]
        psychological_levels = self._get_psychological_levels(current_price)
        
        # Calculate strength of each level
        levels_with_strength = []
        
        for level in resistance_levels[:3]:  # Top 3 resistance
            strength = self._calculate_level_strength(data, level, 'resistance')
            levels_with_strength.append({
                "type": "resistance",
                "price": float(level),
                "strength": strength,
                "distance_from_current": float((level - current_price) / current_price * 100)
            })
        
        for level in support_levels[:3]:  # Top 3 support
            strength = self._calculate_level_strength(data, level, 'support')
            levels_with_strength.append({
                "type": "support",
                "price": float(level),
                "strength": strength,
                "distance_from_current": float((level - current_price) / current_price * 100)
            })
        
        # Add key psychological levels
        for level in psychological_levels:
            if abs(level - current_price) / current_price < 0.1:  # Within 10%
                levels_with_strength.append({
                    "type": "psychological",
                    "price": float(level),
                    "strength": 0.5,
                    "distance_from_current": float((level - current_price) / current_price * 100)
                })
        
        return {
            "current_price": float(current_price),
            "levels": sorted(levels_with_strength, key=lambda x: abs(x['distance_from_current'])),
            "nearest_resistance": min([l for l in levels_with_strength if l['price'] > current_price], 
                                     key=lambda x: x['price'], default=None),
            "nearest_support": max([l for l in levels_with_strength if l['price'] < current_price], 
                                  key=lambda x: x['price'], default=None)
        }
    
    def _cluster_levels(self, prices: np.ndarray, threshold: float = 0.02) -> List[float]:
        """Cluster nearby price levels"""
        if len(prices) == 0:
            return []
        
        sorted_prices = np.sort(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if abs(price - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(price)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def _get_psychological_levels(self, price: float) -> List[float]:
        """Get psychological price levels (round numbers)"""
        levels = []
        
        # Determine the scale
        if price < 10:
            step = 1
        elif price < 100:
            step = 5
        elif price < 1000:
            step = 50
        else:
            step = 100
        
        # Get levels around current price
        base = (price // step) * step
        for i in range(-2, 3):
            level = base + (i * step)
            if level > 0:
                levels.append(level)
        
        return levels
    
    def _calculate_level_strength(self, data: pd.DataFrame, level: float, level_type: str) -> float:
        """Calculate the strength of a support/resistance level"""
        prices = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        touches = 0
        bounces = 0
        
        for i in range(1, len(prices) - 1):
            # Count touches
            if level_type == 'resistance':
                if highs[i] >= level * 0.995 and highs[i] <= level * 1.005:
                    touches += 1
                    if prices[i + 1] < prices[i]:
                        bounces += 1
            else:  # support
                if lows[i] >= level * 0.995 and lows[i] <= level * 1.005:
                    touches += 1
                    if prices[i + 1] > prices[i]:
                        bounces += 1
        
        # Calculate strength (0-1)
        touch_score = min(touches / 10, 1.0)
        bounce_score = bounces / max(touches, 1)
        recency_score = 0.5  # Could be enhanced with time-based calculation
        
        strength = (touch_score * 0.4 + bounce_score * 0.4 + recency_score * 0.2)
        return float(min(strength, 1.0))
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using multiple methods"""
        prices = data['Close'].values
        
        # Short, medium, long term trends
        short_term = self._calculate_trend(prices[-20:]) if len(prices) >= 20 else None
        medium_term = self._calculate_trend(prices[-50:]) if len(prices) >= 50 else None
        long_term = self._calculate_trend(prices[-100:]) if len(prices) >= 100 else None
        
        # Moving average analysis
        ma_analysis = self._analyze_moving_averages(data)
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(prices)
        
        # ADX (Average Directional Index) approximation
        adx = self._calculate_adx(data) if len(data) >= 14 else None
        
        return {
            "short_term": short_term,
            "medium_term": medium_term,
            "long_term": long_term,
            "moving_averages": ma_analysis,
            "trend_strength": trend_strength,
            "adx": adx,
            "overall_trend": self._determine_overall_trend(short_term, medium_term, long_term)
        }
    
    def _calculate_trend(self, prices: np.ndarray) -> Dict[str, Any]:
        """Calculate trend for given price array"""
        if len(prices) < 2:
            return None
        
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend
        price_change_pct = (prices[-1] - prices[0]) / prices[0] * 100
        
        if abs(price_change_pct) < 1:
            trend = "neutral"
        elif price_change_pct > 0:
            trend = "bullish"
        else:
            trend = "bearish"
        
        return {
            "direction": trend,
            "slope": float(slope),
            "angle": float(np.degrees(np.arctan(slope))),
            "r_squared": float(r_squared),
            "change_percent": float(price_change_pct)
        }
    
    def _analyze_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze moving averages"""
        prices = data['Close'].values
        current_price = prices[-1]
        
        analysis = {}
        
        # Calculate various MAs
        periods = [20, 50, 200]
        for period in periods:
            if len(prices) >= period:
                ma = prices[-period:].mean()
                analysis[f"ma_{period}"] = {
                    "value": float(ma),
                    "position": "above" if current_price > ma else "below",
                    "distance_percent": float((current_price - ma) / ma * 100)
                }
        
        # Golden/Death cross detection
        if len(prices) >= 200:
            ma_50 = prices[-50:].mean()
            ma_200 = prices[-200:].mean()
            ma_50_prev = prices[-51:-1].mean()
            ma_200_prev = prices[-201:-1].mean()
            
            if ma_50_prev <= ma_200_prev and ma_50 > ma_200:
                analysis["cross_signal"] = "golden_cross"
            elif ma_50_prev >= ma_200_prev and ma_50 < ma_200:
                analysis["cross_signal"] = "death_cross"
            else:
                analysis["cross_signal"] = None
        
        return analysis
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength (0-1)"""
        if len(prices) < 2:
            return 0.0
        
        # Method 1: Directional consistency
        price_changes = np.diff(prices)
        positive_changes = np.sum(price_changes > 0)
        consistency = abs(positive_changes - len(price_changes)/2) / (len(price_changes)/2)
        
        # Method 2: Smoothness (lower volatility in trend direction)
        volatility = np.std(price_changes) / np.mean(np.abs(prices))
        smoothness = 1 / (1 + volatility * 10)
        
        # Combine metrics
        strength = (consistency * 0.6 + smoothness * 0.4)
        return float(min(max(strength, 0), 1))
    
    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """Calculate Average Directional Index (simplified)"""
        if len(data) < 14:
            return None
        
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        # Calculate True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        
        # Calculate directional movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with EMA
        period = 14
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        pos_di = 100 * pd.Series(pos_dm).ewm(span=period, adjust=False).mean().values / atr
        neg_di = 100 * pd.Series(neg_dm).ewm(span=period, adjust=False).mean().values / atr
        
        # Calculate ADX
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 0.0001)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values[-1]
        
        return float(adx)
    
    def _determine_overall_trend(self, short, medium, long) -> str:
        """Determine overall trend from multiple timeframes"""
        trends = []
        weights = []
        
        if short:
            trends.append(short['direction'])
            weights.append(0.3)
        if medium:
            trends.append(medium['direction'])
            weights.append(0.4)
        if long:
            trends.append(long['direction'])
            weights.append(0.3)
        
        if not trends:
            return "unknown"
        
        # Weight the trends
        trend_scores = {"bullish": 1, "neutral": 0, "bearish": -1}
        weighted_score = sum(trend_scores.get(t, 0) * w for t, w in zip(trends, weights))
        
        if weighted_score > 0.3:
            return "bullish"
        elif weighted_score < -0.3:
            return "bearish"
        else:
            return "neutral"
    
    def _generate_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate price predictions using ML"""
        try:
            prices = data['Close'].values
            
            if len(prices) < 30:
                return {"error": "Insufficient data for predictions"}
            
            # Prepare features
            features = self._prepare_ml_features(data)
            
            # Train model on historical data
            X = features[:-1]
            y = prices[1:]
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.ml_model.fit(X_scaled, y)
            
            # Make predictions
            last_features = features[-1].reshape(1, -1)
            last_features_scaled = self.scaler.transform(last_features)
            
            # Next price prediction
            next_price = self.ml_model.predict(last_features_scaled)[0]
            
            # Confidence based on model score
            score = self.ml_model.score(X_scaled, y)
            
            # Extended predictions (simplified - would use more sophisticated methods in production)
            predictions = []
            current_features = last_features_scaled
            
            for i in range(1, 6):  # 5 period predictions
                pred = self.ml_model.predict(current_features)[0]
                predictions.append({
                    "period": i,
                    "price": float(pred),
                    "confidence": float(max(0, score - i * 0.1))  # Confidence decreases with time
                })
                
                # Update features for next prediction (simplified)
                current_features[0, 0] = pred
            
            # Calculate price targets
            current_price = prices[-1]
            volatility = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
            
            return {
                "next_price": float(next_price),
                "confidence": float(score),
                "predictions": predictions,
                "targets": {
                    "optimistic": float(current_price + 2 * volatility),
                    "realistic": float(next_price),
                    "pessimistic": float(current_price - 2 * volatility)
                },
                "expected_range": {
                    "high": float(next_price + volatility),
                    "low": float(next_price - volatility)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {"error": str(e)}
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        features = []
        
        prices = data['Close'].values
        volumes = data['Volume'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        for i in range(len(prices)):
            feature_vector = [
                prices[i],
                volumes[i] / volumes.mean() if volumes.mean() > 0 else 1,
                (highs[i] - lows[i]) / prices[i] if prices[i] > 0 else 0,  # Range
            ]
            
            # Add technical indicators if enough data
            if i >= 5:
                feature_vector.append(prices[i-5:i].mean())  # 5-period MA
            else:
                feature_vector.append(prices[i])
            
            if i >= 10:
                feature_vector.append(np.std(prices[i-10:i]))  # 10-period volatility
            else:
                feature_vector.append(0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns with AI"""
        volumes = data['Volume'].values
        prices = data['Close'].values
        
        if len(volumes) < 2:
            return {"error": "Insufficient volume data"}
        
        current_volume = volumes[-1]
        avg_volume = volumes.mean()
        
        # Volume trend
        volume_trend = "increasing" if volumes[-5:].mean() > volumes[-20:].mean() else "decreasing"
        
        # Price-volume correlation
        if len(prices) >= 20:
            correlation = np.corrcoef(prices[-20:], volumes[-20:])[0, 1]
        else:
            correlation = 0
        
        # Detect volume spikes
        volume_std = volumes.std()
        spikes = []
        for i in range(max(0, len(volumes) - 10), len(volumes)):
            if volumes[i] > avg_volume + 2 * volume_std:
                spikes.append({
                    "index": i,
                    "volume": float(volumes[i]),
                    "multiplier": float(volumes[i] / avg_volume)
                })
        
        # Volume-based signals
        signals = []
        if current_volume > avg_volume * 1.5 and prices[-1] > prices[-2]:
            signals.append("Bullish volume surge")
        elif current_volume > avg_volume * 1.5 and prices[-1] < prices[-2]:
            signals.append("Bearish volume surge")
        elif current_volume < avg_volume * 0.5:
            signals.append("Low volume warning")
        
        return {
            "current": float(current_volume),
            "average": float(avg_volume),
            "ratio_to_average": float(current_volume / avg_volume) if avg_volume > 0 else 1,
            "trend": volume_trend,
            "price_correlation": float(correlation),
            "recent_spikes": spikes,
            "signals": signals
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility patterns"""
        prices = data['Close'].values
        
        if len(prices) < 2:
            return {"error": "Insufficient data for volatility analysis"}
        
        # Calculate various volatility measures
        returns = np.diff(prices) / prices[:-1]
        
        # Historical volatility
        hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Recent vs historical volatility
        if len(returns) >= 20:
            recent_vol = np.std(returns[-5:]) * np.sqrt(252)
            historical_vol = np.std(returns[-20:]) * np.sqrt(252)
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        else:
            recent_vol = hist_vol
            vol_ratio = 1
        
        # Volatility trend
        if len(returns) >= 30:
            vol_trend = "increasing" if np.std(returns[-10:]) > np.std(returns[-30:-20]) else "decreasing"
        else:
            vol_trend = "stable"
        
        # Risk level
        if hist_vol < 0.15:
            risk_level = "low"
        elif hist_vol < 0.25:
            risk_level = "moderate"
        elif hist_vol < 0.35:
            risk_level = "high"
        else:
            risk_level = "very high"
        
        return {
            "historical": float(hist_vol),
            "recent": float(recent_vol),
            "ratio": float(vol_ratio),
            "trend": vol_trend,
            "risk_level": risk_level,
            "daily_range": float((data['High'].values[-1] - data['Low'].values[-1]) / prices[-1] * 100)
        }
    
    def _calculate_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        prices = data['Close'].values
        
        indicators = {}
        
        # RSI
        if len(prices) >= 14:
            indicators["rsi"] = float(self._calculate_rsi(prices))
        
        # MACD
        if len(prices) >= 26:
            macd_result = self._calculate_macd(prices)
            indicators["macd"] = macd_result
        
        # Stochastic
        if len(data) >= 14:
            indicators["stochastic"] = self._calculate_stochastic(data)
        
        # Momentum
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / prices[-10] * 100
            indicators["momentum_10"] = float(momentum)
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate MACD"""
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            "macd": float(macd.iloc[-1]),
            "signal": float(signal.iloc[-1]),
            "histogram": float(histogram.iloc[-1])
        }
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calculate Stochastic Oscillator"""
        high = data['High'].values[-period:]
        low = data['Low'].values[-period:]
        close = data['Close'].values[-1]
        
        lowest_low = low.min()
        highest_high = high.max()
        
        if highest_high - lowest_low != 0:
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        else:
            k_percent = 50
        
        return {
            "k": float(k_percent),
            "d": float(k_percent)  # Simplified - normally would be 3-period SMA of K
        }
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in price and volume"""
        anomalies = []
        
        prices = data['Close'].values
        volumes = data['Volume'].values
        
        if len(prices) < 20:
            return anomalies
        
        # Price anomalies (using z-score)
        price_mean = prices[-20:].mean()
        price_std = prices[-20:].std()
        
        for i in range(max(0, len(prices) - 5), len(prices)):
            z_score = (prices[i] - price_mean) / price_std if price_std > 0 else 0
            if abs(z_score) > 2.5:
                anomalies.append({
                    "type": "price",
                    "index": i,
                    "value": float(prices[i]),
                    "z_score": float(z_score),
                    "description": f"Unusual price {'spike' if z_score > 0 else 'drop'}"
                })
        
        # Volume anomalies
        volume_mean = volumes[-20:].mean()
        volume_std = volumes[-20:].std()
        
        for i in range(max(0, len(volumes) - 5), len(volumes)):
            z_score = (volumes[i] - volume_mean) / volume_std if volume_std > 0 else 0
            if abs(z_score) > 2.5:
                anomalies.append({
                    "type": "volume",
                    "index": i,
                    "value": float(volumes[i]),
                    "z_score": float(z_score),
                    "description": "Unusual volume spike"
                })
        
        # Gap detection
        for i in range(1, min(5, len(prices))):
            gap = (data['Open'].values[-i] - data['Close'].values[-i-1]) / data['Close'].values[-i-1]
            if abs(gap) > 0.02:  # 2% gap
                anomalies.append({
                    "type": "gap",
                    "index": len(prices) - i,
                    "value": float(gap * 100),
                    "description": f"{'Gap up' if gap > 0 else 'Gap down'} of {abs(gap)*100:.1f}%"
                })
        
        return anomalies
    
    def _generate_ai_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate AI-powered insights from analysis"""
        insights = []
        
        # Pattern insights
        if analysis.get("patterns"):
            top_pattern = analysis["patterns"][0] if analysis["patterns"] else None
            if top_pattern:
                insights.append(f"🎯 {top_pattern['pattern'].replace('_', ' ').title()} pattern detected with {top_pattern['confidence']*100:.0f}% confidence")
        
        # Trend insights
        trend = analysis.get("trend_analysis", {}).get("overall_trend")
        if trend:
            trend_strength = analysis.get("trend_analysis", {}).get("trend_strength", 0)
            if trend_strength > 0.7:
                insights.append(f"💪 Strong {trend} trend in progress")
            elif trend == "neutral":
                insights.append("➡️ Market is range-bound, consider range trading strategies")
        
        # Support/Resistance insights
        sr = analysis.get("support_resistance", {})
        if sr.get("nearest_resistance"):
            resistance = sr["nearest_resistance"]
            insights.append(f"🔴 Resistance at ${resistance['price']:.2f} ({abs(resistance['distance_from_current']):.1f}% away)")
        if sr.get("nearest_support"):
            support = sr["nearest_support"]
            insights.append(f"🟢 Support at ${support['price']:.2f} ({abs(support['distance_from_current']):.1f}% away)")
        
        # Volume insights
        volume = analysis.get("volume_analysis", {})
        if volume.get("signals"):
            for signal in volume["signals"][:1]:
                insights.append(f"📊 {signal}")
        
        # Volatility insights
        volatility = analysis.get("volatility_analysis", {})
        if volatility.get("risk_level") == "very high":
            insights.append("⚠️ Very high volatility - consider reducing position size")
        elif volatility.get("risk_level") == "low":
            insights.append("✅ Low volatility environment - favorable for trend following")
        
        # Momentum insights
        momentum = analysis.get("momentum_indicators", {})
        rsi = momentum.get("rsi")
        if rsi:
            if rsi > 70:
                insights.append(f"📈 RSI overbought at {rsi:.1f} - potential pullback ahead")
            elif rsi < 30:
                insights.append(f"📉 RSI oversold at {rsi:.1f} - potential bounce ahead")
        
        # Prediction insights
        predictions = analysis.get("predictions", {})
        if predictions.get("next_price"):
            current = analysis.get("support_resistance", {}).get("current_price", 0)
            next_price = predictions["next_price"]
            change_pct = (next_price - current) / current * 100 if current > 0 else 0
            if abs(change_pct) > 1:
                direction = "increase" if change_pct > 0 else "decrease"
                insights.append(f"🔮 AI predicts {abs(change_pct):.1f}% {direction} to ${next_price:.2f}")
        
        # Anomaly insights
        anomalies = analysis.get("anomalies", [])
        if anomalies:
            insights.append(f"🚨 {len(anomalies)} unusual market events detected")
        
        return insights[:7]  # Limit to top 7 insights
    
    def _assess_risk(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess trading risk"""
        volatility = analysis.get("volatility_analysis", {})
        trend = analysis.get("trend_analysis", {})
        
        risk_score = 50  # Start neutral
        
        # Volatility risk
        if volatility.get("risk_level") == "very high":
            risk_score += 30
        elif volatility.get("risk_level") == "high":
            risk_score += 15
        elif volatility.get("risk_level") == "low":
            risk_score -= 10
        
        # Trend risk
        trend_strength = trend.get("trend_strength", 0.5)
        if trend_strength < 0.3:
            risk_score += 10  # Weak trend = higher risk
        elif trend_strength > 0.7:
            risk_score -= 10  # Strong trend = lower risk
        
        # Pattern risk
        patterns = analysis.get("patterns", [])
        if patterns:
            # Reversal patterns increase risk
            reversal_patterns = ["head_and_shoulders", "double_top", "double_bottom"]
            if any(p["pattern"] in reversal_patterns for p in patterns):
                risk_score += 15
        
        # Normalize to 0-100
        risk_score = max(0, min(100, risk_score))
        
        # Determine risk level
        if risk_score < 30:
            level = "low"
            color = "green"
        elif risk_score < 50:
            level = "moderate"
            color = "yellow"
        elif risk_score < 70:
            level = "high"
            color = "orange"
        else:
            level = "very high"
            color = "red"
        
        # Suggested position size (inverse of risk)
        suggested_position = max(0.1, 1.0 - risk_score / 100)
        
        return {
            "score": risk_score,
            "level": level,
            "color": color,
            "suggested_position_size": f"{suggested_position * 100:.0f}%",
            "factors": {
                "volatility": volatility.get("risk_level", "unknown"),
                "trend_strength": f"{trend_strength:.2f}",
                "pattern_risk": len([p for p in patterns if p["pattern"] in ["head_and_shoulders", "double_top", "double_bottom"]])
            }
        }
    
    def _generate_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendation based on all analysis"""
        score = 0
        factors = []
        
        # Pattern score
        patterns = analysis.get("patterns", [])
        if patterns:
            top_pattern = patterns[0]
            if "bullish" in top_pattern.get("implications", "").lower():
                score += 20
                factors.append(f"Bullish {top_pattern['pattern']} pattern")
            elif "bearish" in top_pattern.get("implications", "").lower():
                score -= 20
                factors.append(f"Bearish {top_pattern['pattern']} pattern")
        
        # Trend score
        trend = analysis.get("trend_analysis", {}).get("overall_trend")
        if trend == "bullish":
            score += 15
            factors.append("Bullish trend")
        elif trend == "bearish":
            score -= 15
            factors.append("Bearish trend")
        
        # Momentum score
        momentum = analysis.get("momentum_indicators", {})
        rsi = momentum.get("rsi", 50)
        if rsi < 30:
            score += 10
            factors.append("Oversold RSI")
        elif rsi > 70:
            score -= 10
            factors.append("Overbought RSI")
        
        # Volume score
        volume = analysis.get("volume_analysis", {})
        if "Bullish volume surge" in volume.get("signals", []):
            score += 10
            factors.append("Bullish volume")
        elif "Bearish volume surge" in volume.get("signals", []):
            score -= 10
            factors.append("Bearish volume")
        
        # Support/Resistance score
        sr = analysis.get("support_resistance", {})
        current_price = sr.get("current_price", 0)
        if sr.get("nearest_support"):
            support_distance = abs(sr["nearest_support"]["distance_from_current"])
            if support_distance < 2:
                score += 5
                factors.append("Near support")
        if sr.get("nearest_resistance"):
            resistance_distance = abs(sr["nearest_resistance"]["distance_from_current"])
            if resistance_distance < 2:
                score -= 5
                factors.append("Near resistance")
        
        # Determine recommendation
        if score >= 20:
            action = "STRONG BUY"
            confidence = min(0.9, 0.5 + score / 100)
        elif score >= 10:
            action = "BUY"
            confidence = min(0.75, 0.5 + score / 100)
        elif score <= -20:
            action = "STRONG SELL"
            confidence = min(0.9, 0.5 + abs(score) / 100)
        elif score <= -10:
            action = "SELL"
            confidence = min(0.75, 0.5 + abs(score) / 100)
        else:
            action = "HOLD"
            confidence = 0.5
        
        return {
            "action": action,
            "confidence": confidence,
            "score": score,
            "factors": factors,
            "timeframe": "Short to medium term",
            "risk_level": analysis.get("risk_assessment", {}).get("level", "moderate")
        }
    
    def _generate_annotations(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate chart annotations for visualization"""
        annotations = []
        
        # Pattern annotations
        for pattern in analysis.get("patterns", [])[:3]:  # Top 3 patterns
            if pattern.get("position") is not None:
                annotations.append({
                    "type": "pattern",
                    "x": pattern["position"],
                    "y": data['High'].values[min(pattern["position"], len(data)-1)] if pattern["position"] < len(data) else data['High'].values[-1],
                    "text": pattern["pattern"].replace("_", " ").title(),
                    "color": "#FFD700"
                })
        
        # Support/Resistance lines
        sr = analysis.get("support_resistance", {})
        for level in sr.get("levels", [])[:5]:  # Top 5 levels
            annotations.append({
                "type": "horizontal_line",
                "y": level["price"],
                "text": f"{level['type'].title()} ${level['price']:.2f}",
                "color": "#00FF00" if level["type"] == "support" else "#FF0000",
                "style": "dashed"
            })
        
        # Anomaly markers
        for anomaly in analysis.get("anomalies", [])[:3]:  # Top 3 anomalies
            if anomaly.get("index") is not None:
                annotations.append({
                    "type": "marker",
                    "x": anomaly["index"],
                    "y": data['High'].values[min(anomaly["index"], len(data)-1)] if anomaly["index"] < len(data) else data['High'].values[-1],
                    "text": "⚠️",
                    "tooltip": anomaly["description"]
                })
        
        # Prediction arrow
        predictions = analysis.get("predictions", {})
        if predictions.get("next_price"):
            current_price = data['Close'].values[-1]
            next_price = predictions["next_price"]
            annotations.append({
                "type": "arrow",
                "x1": len(data) - 1,
                "y1": current_price,
                "x2": len(data),
                "y2": next_price,
                "color": "#00FFFF",
                "text": f"Target: ${next_price:.2f}"
            })
        
        # Volume spike annotations
        volume_analysis = analysis.get("volume_analysis", {})
        for spike in volume_analysis.get("recent_spikes", [])[:2]:
            annotations.append({
                "type": "volume_marker",
                "x": spike["index"],
                "text": f"{spike['multiplier']:.1f}x vol",
                "color": "#FFA500"
            })
        
        return annotations
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis"""
        confidence_factors = []
        
        # Data quality
        data_points = analysis.get("data_points", 0)
        if data_points >= 200:
            confidence_factors.append(1.0)
        elif data_points >= 100:
            confidence_factors.append(0.8)
        elif data_points >= 50:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Pattern confidence
        patterns = analysis.get("patterns", [])
        if patterns:
            pattern_confidence = patterns[0].get("confidence", 0)
            confidence_factors.append(pattern_confidence)
        
        # Trend clarity
        trend_analysis = analysis.get("trend_analysis", {})
        trend_strength = trend_analysis.get("trend_strength", 0)
        confidence_factors.append(trend_strength)
        
        # Prediction confidence
        predictions = analysis.get("predictions", {})
        if predictions.get("confidence"):
            confidence_factors.append(predictions["confidence"])
        
        # Calculate weighted average
        if confidence_factors:
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            overall_confidence = 0.5
        
        return float(min(max(overall_confidence, 0), 1))


# Global service instance
ai_chart_analyzer = AIChartAnalyzer()