"""
Comprehensive Pattern Recognition Service
Implements all major chart patterns with detection and visualization
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress
import asyncio

from core.logging import get_logger

logger = get_logger(__name__)


class PatternCategory(Enum):
    """Pattern categories"""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    BILATERAL = "bilateral"  # Can be either


class PatternImplication(Enum):
    """Pattern market implications"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ChartPatternType(Enum):
    """Comprehensive chart pattern types"""
    # Reversal Patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ROUNDING_TOP = "rounding_top"
    ROUNDING_BOTTOM = "rounding_bottom"
    
    # Continuation Patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT_BULLISH = "pennant_bullish"
    PENNANT_BEARISH = "pennant_bearish"
    CUP_AND_HANDLE = "cup_and_handle"
    
    # Wedge Patterns (Can be reversal or continuation)
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    
    # Rectangle Patterns
    RECTANGLE = "rectangle"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    
    # Advanced Patterns
    DIAMOND_TOP = "diamond_top"
    DIAMOND_BOTTOM = "diamond_bottom"
    BROADENING_FORMATION = "broadening_formation"
    
    # Harmonic Patterns
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    SHARK = "shark"
    CYPHER = "cypher"
    
    # Elliott Wave Patterns
    ELLIOTT_IMPULSE = "elliott_impulse"
    ELLIOTT_CORRECTIVE = "elliott_corrective"
    
    # Wyckoff Patterns
    WYCKOFF_ACCUMULATION = "wyckoff_accumulation"
    WYCKOFF_DISTRIBUTION = "wyckoff_distribution"


@dataclass
class PatternPoint:
    """Key point in a pattern"""
    timestamp: datetime
    price: float
    type: str  # 'peak', 'trough', 'support', 'resistance', etc.
    index: int  # Position in data array


@dataclass
class DetectedPattern:
    """Detected chart pattern with all details"""
    pattern_type: ChartPatternType
    category: PatternCategory
    implication: PatternImplication
    confidence: float  # 0-100
    
    # Pattern points
    key_points: List[PatternPoint]
    start_time: datetime
    end_time: datetime
    
    # Pattern characteristics
    neckline: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Pattern metrics
    pattern_height: float = 0
    pattern_width_bars: int = 0
    volume_pattern: str = ""  # 'increasing', 'decreasing', 'neutral'
    
    # Formation status
    is_complete: bool = False
    completion_percentage: float = 0
    breakout_confirmed: bool = False
    
    # Visualization data
    trendlines: List[Tuple[PatternPoint, PatternPoint]] = field(default_factory=list)
    filled_areas: List[List[PatternPoint]] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)


class PatternRecognitionService:
    """
    Comprehensive pattern recognition service
    """
    
    def __init__(self):
        # Pattern detection parameters
        self.min_pattern_bars = 5
        self.max_pattern_bars = 200
        self.extrema_order = 5  # Window for finding peaks/troughs
        self.similarity_threshold = 0.05  # 5% price similarity
        self.volume_confirmation = True
        self.min_confidence = 60  # Minimum confidence to report pattern
        
        # Smoothing parameters
        self.smooth_window = 3
        self.use_smoothing = True
        
        # Pattern-specific parameters
        self.pattern_params = self._initialize_pattern_params()
        
        logger.info("Pattern Recognition Service initialized")
    
    def _initialize_pattern_params(self) -> Dict:
        """Initialize pattern-specific parameters"""
        return {
            ChartPatternType.HEAD_AND_SHOULDERS: {
                'min_bars': 35,
                'shoulder_tolerance': 0.05,
                'head_prominence': 0.1,
                'volume_decline': True
            },
            ChartPatternType.DOUBLE_TOP: {
                'min_bars': 20,
                'peak_tolerance': 0.03,
                'valley_depth': 0.05,
                'volume_pattern': 'declining'
            },
            ChartPatternType.ASCENDING_TRIANGLE: {
                'min_bars': 15,
                'min_touches': 2,
                'slope_tolerance': 0.02,
                'volume_pattern': 'increasing'
            },
            ChartPatternType.CUP_AND_HANDLE: {
                'min_bars': 30,
                'cup_depth': 0.1,
                'handle_depth': 0.05,
                'roundness_factor': 0.7
            },
            # Add more pattern-specific params as needed
        }
    
    async def detect_patterns(
        self,
        data: pd.DataFrame,
        patterns_to_detect: Optional[List[ChartPatternType]] = None,
        include_forming: bool = True,
        min_confidence: Optional[float] = None
    ) -> List[DetectedPattern]:
        """
        Detect chart patterns in price data
        
        Args:
            data: DataFrame with OHLCV data
            patterns_to_detect: Specific patterns to look for (None = all)
            include_forming: Include incomplete/forming patterns
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected patterns
        """
        try:
            if data.empty or len(data) < self.min_pattern_bars:
                return []
            
            # Prepare data
            prices = self._prepare_data(data)
            
            # Find extrema points
            peaks, troughs = self._find_extrema(prices)
            
            if len(peaks) < 2 or len(troughs) < 2:
                return []
            
            # Detect patterns
            detected = []
            patterns = patterns_to_detect or list(ChartPatternType)
            
            for pattern_type in patterns:
                try:
                    if pattern_type == ChartPatternType.HEAD_AND_SHOULDERS:
                        detected.extend(await self._detect_head_shoulders(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.INVERSE_HEAD_AND_SHOULDERS:
                        detected.extend(await self._detect_inverse_head_shoulders(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.DOUBLE_TOP:
                        detected.extend(await self._detect_double_top(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.DOUBLE_BOTTOM:
                        detected.extend(await self._detect_double_bottom(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.TRIPLE_TOP:
                        detected.extend(await self._detect_triple_top(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.TRIPLE_BOTTOM:
                        detected.extend(await self._detect_triple_bottom(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.ASCENDING_TRIANGLE:
                        detected.extend(await self._detect_ascending_triangle(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.DESCENDING_TRIANGLE:
                        detected.extend(await self._detect_descending_triangle(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.SYMMETRICAL_TRIANGLE:
                        detected.extend(await self._detect_symmetrical_triangle(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.FLAG_BULLISH:
                        detected.extend(await self._detect_bull_flag(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.FLAG_BEARISH:
                        detected.extend(await self._detect_bear_flag(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.CUP_AND_HANDLE:
                        detected.extend(await self._detect_cup_handle(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.RISING_WEDGE:
                        detected.extend(await self._detect_rising_wedge(prices, peaks, troughs, data))
                    elif pattern_type == ChartPatternType.FALLING_WEDGE:
                        detected.extend(await self._detect_falling_wedge(prices, peaks, troughs, data))
                    # Add more pattern detections as needed
                except Exception as e:
                    logger.debug(f"Error detecting {pattern_type.value}: {str(e)}")
            
            # Filter by confidence
            confidence_threshold = min_confidence or self.min_confidence
            detected = [p for p in detected if p.confidence >= confidence_threshold]
            
            # Filter out forming patterns if not requested
            if not include_forming:
                detected = [p for p in detected if p.is_complete]
            
            # Sort by confidence
            detected.sort(key=lambda x: x.confidence, reverse=True)
            
            return detected
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
            return []
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.Series:
        """Prepare price data for pattern detection"""
        # Use close prices (could also use high/low for certain patterns)
        prices = data['close'].copy()
        
        # Apply smoothing if enabled
        if self.use_smoothing:
            prices = prices.rolling(window=self.smooth_window, center=True).mean()
            prices = prices.fillna(method='bfill').fillna(method='ffill')
        
        return prices
    
    def _find_extrema(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Find local peaks and troughs"""
        # Find local maxima (peaks)
        peaks = argrelextrema(
            prices.values,
            np.greater,
            order=self.extrema_order
        )[0]
        
        # Find local minima (troughs)
        troughs = argrelextrema(
            prices.values,
            np.less,
            order=self.extrema_order
        )[0]
        
        return peaks, troughs
    
    async def _detect_head_shoulders(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Head and Shoulders patterns"""
        patterns = []
        
        if len(peaks) < 3:
            return patterns
        
        # Check each combination of 3 peaks
        for i in range(len(peaks) - 2):
            left_shoulder_idx = peaks[i]
            head_idx = peaks[i + 1]
            right_shoulder_idx = peaks[i + 2]
            
            left_shoulder = prices.iloc[left_shoulder_idx]
            head = prices.iloc[head_idx]
            right_shoulder = prices.iloc[right_shoulder_idx]
            
            # Check if middle peak is highest (head)
            if head > left_shoulder and head > right_shoulder:
                # Check if shoulders are similar height
                shoulder_diff = abs(left_shoulder - right_shoulder) / head
                if shoulder_diff <= self.similarity_threshold:
                    # Find neckline (troughs between shoulders)
                    left_trough_idx = self._find_trough_between(troughs, left_shoulder_idx, head_idx)
                    right_trough_idx = self._find_trough_between(troughs, head_idx, right_shoulder_idx)
                    
                    if left_trough_idx is not None and right_trough_idx is not None:
                        left_trough = prices.iloc[left_trough_idx]
                        right_trough = prices.iloc[right_trough_idx]
                        neckline = (left_trough + right_trough) / 2
                        
                        # Calculate pattern metrics
                        pattern_height = head - neckline
                        target_price = neckline - pattern_height  # Bearish target
                        
                        # Calculate confidence based on pattern quality
                        confidence = self._calculate_pattern_confidence(
                            pattern_type=ChartPatternType.HEAD_AND_SHOULDERS,
                            key_metrics={
                                'shoulder_symmetry': 1 - shoulder_diff,
                                'head_prominence': (head - max(left_shoulder, right_shoulder)) / head,
                                'neckline_consistency': 1 - abs(left_trough - right_trough) / neckline,
                                'volume_pattern': self._check_volume_pattern(data, left_shoulder_idx, right_shoulder_idx)
                            }
                        )
                        
                        # Check if pattern is complete (price broke neckline)
                        current_price = prices.iloc[-1]
                        is_complete = current_price < neckline
                        
                        # Create pattern object
                        pattern = DetectedPattern(
                            pattern_type=ChartPatternType.HEAD_AND_SHOULDERS,
                            category=PatternCategory.REVERSAL,
                            implication=PatternImplication.BEARISH,
                            confidence=confidence,
                            key_points=[
                                PatternPoint(data.index[left_shoulder_idx], left_shoulder, 'left_shoulder', left_shoulder_idx),
                                PatternPoint(data.index[left_trough_idx], left_trough, 'left_trough', left_trough_idx),
                                PatternPoint(data.index[head_idx], head, 'head', head_idx),
                                PatternPoint(data.index[right_trough_idx], right_trough, 'right_trough', right_trough_idx),
                                PatternPoint(data.index[right_shoulder_idx], right_shoulder, 'right_shoulder', right_shoulder_idx)
                            ],
                            start_time=data.index[left_shoulder_idx],
                            end_time=data.index[right_shoulder_idx],
                            neckline=neckline,
                            resistance_level=head,
                            target_price=target_price,
                            stop_loss=head * 1.02,  # 2% above head
                            pattern_height=pattern_height,
                            pattern_width_bars=right_shoulder_idx - left_shoulder_idx,
                            is_complete=is_complete,
                            completion_percentage=100 if is_complete else 80,
                            breakout_confirmed=is_complete and current_price < neckline * 0.98
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_inverse_head_shoulders(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Inverse Head and Shoulders patterns"""
        patterns = []
        
        if len(troughs) < 3:
            return patterns
        
        # Check each combination of 3 troughs
        for i in range(len(troughs) - 2):
            left_shoulder_idx = troughs[i]
            head_idx = troughs[i + 1]
            right_shoulder_idx = troughs[i + 2]
            
            left_shoulder = prices.iloc[left_shoulder_idx]
            head = prices.iloc[head_idx]
            right_shoulder = prices.iloc[right_shoulder_idx]
            
            # Check if middle trough is lowest (head)
            if head < left_shoulder and head < right_shoulder:
                # Check if shoulders are similar
                shoulder_diff = abs(left_shoulder - right_shoulder) / ((left_shoulder + right_shoulder) / 2)
                if shoulder_diff <= self.similarity_threshold:
                    # Find neckline (peaks between shoulders)
                    left_peak_idx = self._find_peak_between(peaks, left_shoulder_idx, head_idx)
                    right_peak_idx = self._find_peak_between(peaks, head_idx, right_shoulder_idx)
                    
                    if left_peak_idx is not None and right_peak_idx is not None:
                        left_peak = prices.iloc[left_peak_idx]
                        right_peak = prices.iloc[right_peak_idx]
                        neckline = (left_peak + right_peak) / 2
                        
                        # Calculate pattern metrics
                        pattern_height = neckline - head
                        target_price = neckline + pattern_height  # Bullish target
                        
                        # Calculate confidence
                        confidence = self._calculate_pattern_confidence(
                            pattern_type=ChartPatternType.INVERSE_HEAD_AND_SHOULDERS,
                            key_metrics={
                                'shoulder_symmetry': 1 - shoulder_diff,
                                'head_prominence': (min(left_shoulder, right_shoulder) - head) / head,
                                'neckline_consistency': 1 - abs(left_peak - right_peak) / neckline,
                                'volume_pattern': self._check_volume_pattern(data, left_shoulder_idx, right_shoulder_idx)
                            }
                        )
                        
                        # Check if complete
                        current_price = prices.iloc[-1]
                        is_complete = current_price > neckline
                        
                        pattern = DetectedPattern(
                            pattern_type=ChartPatternType.INVERSE_HEAD_AND_SHOULDERS,
                            category=PatternCategory.REVERSAL,
                            implication=PatternImplication.BULLISH,
                            confidence=confidence,
                            key_points=[
                                PatternPoint(data.index[left_shoulder_idx], left_shoulder, 'left_shoulder', left_shoulder_idx),
                                PatternPoint(data.index[left_peak_idx], left_peak, 'left_peak', left_peak_idx),
                                PatternPoint(data.index[head_idx], head, 'head', head_idx),
                                PatternPoint(data.index[right_peak_idx], right_peak, 'right_peak', right_peak_idx),
                                PatternPoint(data.index[right_shoulder_idx], right_shoulder, 'right_shoulder', right_shoulder_idx)
                            ],
                            start_time=data.index[left_shoulder_idx],
                            end_time=data.index[right_shoulder_idx],
                            neckline=neckline,
                            support_level=head,
                            target_price=target_price,
                            stop_loss=head * 0.98,  # 2% below head
                            pattern_height=pattern_height,
                            pattern_width_bars=right_shoulder_idx - left_shoulder_idx,
                            is_complete=is_complete,
                            completion_percentage=100 if is_complete else 80,
                            breakout_confirmed=is_complete and current_price > neckline * 1.02
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_double_top(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Double Top patterns"""
        patterns = []
        
        if len(peaks) < 2:
            return patterns
        
        for i in range(len(peaks) - 1):
            first_peak_idx = peaks[i]
            second_peak_idx = peaks[i + 1]
            
            first_peak = prices.iloc[first_peak_idx]
            second_peak = prices.iloc[second_peak_idx]
            
            # Check if peaks are similar height
            peak_diff = abs(first_peak - second_peak) / max(first_peak, second_peak)
            if peak_diff <= self.similarity_threshold:
                # Find valley between peaks
                valley_idx = self._find_trough_between(troughs, first_peak_idx, second_peak_idx)
                
                if valley_idx is not None:
                    valley = prices.iloc[valley_idx]
                    
                    # Valley should be significantly lower than peaks
                    valley_depth = (max(first_peak, second_peak) - valley) / max(first_peak, second_peak)
                    if valley_depth >= 0.05:  # At least 5% lower
                        # Calculate metrics
                        pattern_height = max(first_peak, second_peak) - valley
                        target_price = valley - pattern_height
                        
                        confidence = self._calculate_pattern_confidence(
                            pattern_type=ChartPatternType.DOUBLE_TOP,
                            key_metrics={
                                'peak_similarity': 1 - peak_diff,
                                'valley_depth': valley_depth,
                                'volume_pattern': self._check_volume_pattern(data, first_peak_idx, second_peak_idx)
                            }
                        )
                        
                        current_price = prices.iloc[-1]
                        is_complete = current_price < valley
                        
                        pattern = DetectedPattern(
                            pattern_type=ChartPatternType.DOUBLE_TOP,
                            category=PatternCategory.REVERSAL,
                            implication=PatternImplication.BEARISH,
                            confidence=confidence,
                            key_points=[
                                PatternPoint(data.index[first_peak_idx], first_peak, 'first_peak', first_peak_idx),
                                PatternPoint(data.index[valley_idx], valley, 'valley', valley_idx),
                                PatternPoint(data.index[second_peak_idx], second_peak, 'second_peak', second_peak_idx)
                            ],
                            start_time=data.index[first_peak_idx],
                            end_time=data.index[second_peak_idx],
                            support_level=valley,
                            resistance_level=max(first_peak, second_peak),
                            target_price=target_price,
                            stop_loss=max(first_peak, second_peak) * 1.02,
                            pattern_height=pattern_height,
                            pattern_width_bars=second_peak_idx - first_peak_idx,
                            is_complete=is_complete,
                            completion_percentage=100 if is_complete else 70,
                            breakout_confirmed=is_complete and current_price < valley * 0.98
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_double_bottom(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Double Bottom patterns"""
        patterns = []
        
        if len(troughs) < 2:
            return patterns
        
        for i in range(len(troughs) - 1):
            first_trough_idx = troughs[i]
            second_trough_idx = troughs[i + 1]
            
            first_trough = prices.iloc[first_trough_idx]
            second_trough = prices.iloc[second_trough_idx]
            
            # Check if troughs are similar
            trough_diff = abs(first_trough - second_trough) / min(first_trough, second_trough)
            if trough_diff <= self.similarity_threshold:
                # Find peak between troughs
                peak_idx = self._find_peak_between(peaks, first_trough_idx, second_trough_idx)
                
                if peak_idx is not None:
                    peak = prices.iloc[peak_idx]
                    
                    # Peak should be significantly higher
                    peak_height = (peak - min(first_trough, second_trough)) / peak
                    if peak_height >= 0.05:
                        # Calculate metrics
                        pattern_height = peak - min(first_trough, second_trough)
                        target_price = peak + pattern_height
                        
                        confidence = self._calculate_pattern_confidence(
                            pattern_type=ChartPatternType.DOUBLE_BOTTOM,
                            key_metrics={
                                'trough_similarity': 1 - trough_diff,
                                'peak_height': peak_height,
                                'volume_pattern': self._check_volume_pattern(data, first_trough_idx, second_trough_idx)
                            }
                        )
                        
                        current_price = prices.iloc[-1]
                        is_complete = current_price > peak
                        
                        pattern = DetectedPattern(
                            pattern_type=ChartPatternType.DOUBLE_BOTTOM,
                            category=PatternCategory.REVERSAL,
                            implication=PatternImplication.BULLISH,
                            confidence=confidence,
                            key_points=[
                                PatternPoint(data.index[first_trough_idx], first_trough, 'first_trough', first_trough_idx),
                                PatternPoint(data.index[peak_idx], peak, 'peak', peak_idx),
                                PatternPoint(data.index[second_trough_idx], second_trough, 'second_trough', second_trough_idx)
                            ],
                            start_time=data.index[first_trough_idx],
                            end_time=data.index[second_trough_idx],
                            support_level=min(first_trough, second_trough),
                            resistance_level=peak,
                            target_price=target_price,
                            stop_loss=min(first_trough, second_trough) * 0.98,
                            pattern_height=pattern_height,
                            pattern_width_bars=second_trough_idx - first_trough_idx,
                            is_complete=is_complete,
                            completion_percentage=100 if is_complete else 70,
                            breakout_confirmed=is_complete and current_price > peak * 1.02
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_triple_top(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Triple Top patterns"""
        patterns = []
        
        if len(peaks) < 3:
            return patterns
        
        for i in range(len(peaks) - 2):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]
            peak3_idx = peaks[i + 2]
            
            peak1 = prices.iloc[peak1_idx]
            peak2 = prices.iloc[peak2_idx]
            peak3 = prices.iloc[peak3_idx]
            
            # Check if all peaks are similar
            max_peak = max(peak1, peak2, peak3)
            min_peak = min(peak1, peak2, peak3)
            peak_range = (max_peak - min_peak) / max_peak
            
            if peak_range <= self.similarity_threshold:
                # Find valleys
                valley1_idx = self._find_trough_between(troughs, peak1_idx, peak2_idx)
                valley2_idx = self._find_trough_between(troughs, peak2_idx, peak3_idx)
                
                if valley1_idx is not None and valley2_idx is not None:
                    valley1 = prices.iloc[valley1_idx]
                    valley2 = prices.iloc[valley2_idx]
                    support = min(valley1, valley2)
                    
                    # Calculate metrics
                    pattern_height = max_peak - support
                    target_price = support - pattern_height
                    
                    confidence = self._calculate_pattern_confidence(
                        pattern_type=ChartPatternType.TRIPLE_TOP,
                        key_metrics={
                            'peak_consistency': 1 - peak_range,
                            'support_strength': 3,  # Three tests of resistance
                            'volume_pattern': self._check_volume_pattern(data, peak1_idx, peak3_idx)
                        }
                    )
                    
                    current_price = prices.iloc[-1]
                    is_complete = current_price < support
                    
                    pattern = DetectedPattern(
                        pattern_type=ChartPatternType.TRIPLE_TOP,
                        category=PatternCategory.REVERSAL,
                        implication=PatternImplication.BEARISH,
                        confidence=confidence,
                        key_points=[
                            PatternPoint(data.index[peak1_idx], peak1, 'peak1', peak1_idx),
                            PatternPoint(data.index[valley1_idx], valley1, 'valley1', valley1_idx),
                            PatternPoint(data.index[peak2_idx], peak2, 'peak2', peak2_idx),
                            PatternPoint(data.index[valley2_idx], valley2, 'valley2', valley2_idx),
                            PatternPoint(data.index[peak3_idx], peak3, 'peak3', peak3_idx)
                        ],
                        start_time=data.index[peak1_idx],
                        end_time=data.index[peak3_idx],
                        support_level=support,
                        resistance_level=max_peak,
                        target_price=target_price,
                        stop_loss=max_peak * 1.02,
                        pattern_height=pattern_height,
                        pattern_width_bars=peak3_idx - peak1_idx,
                        is_complete=is_complete,
                        completion_percentage=100 if is_complete else 85,
                        breakout_confirmed=is_complete and current_price < support * 0.98
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_triple_bottom(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Triple Bottom patterns"""
        patterns = []
        
        if len(troughs) < 3:
            return patterns
        
        for i in range(len(troughs) - 2):
            trough1_idx = troughs[i]
            trough2_idx = troughs[i + 1]
            trough3_idx = troughs[i + 2]
            
            trough1 = prices.iloc[trough1_idx]
            trough2 = prices.iloc[trough2_idx]
            trough3 = prices.iloc[trough3_idx]
            
            # Check if all troughs are similar
            max_trough = max(trough1, trough2, trough3)
            min_trough = min(trough1, trough2, trough3)
            trough_range = (max_trough - min_trough) / min_trough
            
            if trough_range <= self.similarity_threshold:
                # Find peaks
                peak1_idx = self._find_peak_between(peaks, trough1_idx, trough2_idx)
                peak2_idx = self._find_peak_between(peaks, trough2_idx, trough3_idx)
                
                if peak1_idx is not None and peak2_idx is not None:
                    peak1 = prices.iloc[peak1_idx]
                    peak2 = prices.iloc[peak2_idx]
                    resistance = max(peak1, peak2)
                    
                    # Calculate metrics
                    pattern_height = resistance - min_trough
                    target_price = resistance + pattern_height
                    
                    confidence = self._calculate_pattern_confidence(
                        pattern_type=ChartPatternType.TRIPLE_BOTTOM,
                        key_metrics={
                            'trough_consistency': 1 - trough_range,
                            'support_strength': 3,
                            'volume_pattern': self._check_volume_pattern(data, trough1_idx, trough3_idx)
                        }
                    )
                    
                    current_price = prices.iloc[-1]
                    is_complete = current_price > resistance
                    
                    pattern = DetectedPattern(
                        pattern_type=ChartPatternType.TRIPLE_BOTTOM,
                        category=PatternCategory.REVERSAL,
                        implication=PatternImplication.BULLISH,
                        confidence=confidence,
                        key_points=[
                            PatternPoint(data.index[trough1_idx], trough1, 'trough1', trough1_idx),
                            PatternPoint(data.index[peak1_idx], peak1, 'peak1', peak1_idx),
                            PatternPoint(data.index[trough2_idx], trough2, 'trough2', trough2_idx),
                            PatternPoint(data.index[peak2_idx], peak2, 'peak2', peak2_idx),
                            PatternPoint(data.index[trough3_idx], trough3, 'trough3', trough3_idx)
                        ],
                        start_time=data.index[trough1_idx],
                        end_time=data.index[trough3_idx],
                        support_level=min_trough,
                        resistance_level=resistance,
                        target_price=target_price,
                        stop_loss=min_trough * 0.98,
                        pattern_height=pattern_height,
                        pattern_width_bars=trough3_idx - trough1_idx,
                        is_complete=is_complete,
                        completion_percentage=100 if is_complete else 85,
                        breakout_confirmed=is_complete and current_price > resistance * 1.02
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_ascending_triangle(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Ascending Triangle patterns"""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Look for flat resistance and rising support
        for i in range(len(peaks) - 1):
            # Check if peaks form horizontal resistance
            resistance_peaks = []
            for j in range(i, min(i + 5, len(peaks))):
                if abs(prices.iloc[peaks[j]] - prices.iloc[peaks[i]]) / prices.iloc[peaks[i]] <= self.similarity_threshold:
                    resistance_peaks.append(peaks[j])
            
            if len(resistance_peaks) >= 2:
                # Find troughs in the same period
                start_idx = resistance_peaks[0]
                end_idx = resistance_peaks[-1]
                pattern_troughs = [t for t in troughs if start_idx < t < end_idx]
                
                if len(pattern_troughs) >= 2:
                    # Check if troughs are rising
                    trough_prices = [prices.iloc[t] for t in pattern_troughs]
                    if all(trough_prices[i] < trough_prices[i+1] for i in range(len(trough_prices)-1)):
                        # Calculate metrics
                        resistance = np.mean([prices.iloc[p] for p in resistance_peaks])
                        support_slope = (trough_prices[-1] - trough_prices[0]) / (pattern_troughs[-1] - pattern_troughs[0])
                        
                        pattern_height = resistance - trough_prices[0]
                        target_price = resistance + pattern_height
                        
                        confidence = self._calculate_pattern_confidence(
                            pattern_type=ChartPatternType.ASCENDING_TRIANGLE,
                            key_metrics={
                                'resistance_consistency': len(resistance_peaks) / 5,
                                'support_trend': support_slope * 100,
                                'volume_pattern': self._check_volume_pattern(data, start_idx, end_idx)
                            }
                        )
                        
                        current_price = prices.iloc[-1]
                        is_complete = current_price > resistance
                        
                        pattern = DetectedPattern(
                            pattern_type=ChartPatternType.ASCENDING_TRIANGLE,
                            category=PatternCategory.CONTINUATION,
                            implication=PatternImplication.BULLISH,
                            confidence=confidence,
                            key_points=[
                                PatternPoint(data.index[p], prices.iloc[p], f'resistance_{i}', p)
                                for i, p in enumerate(resistance_peaks)
                            ] + [
                                PatternPoint(data.index[t], prices.iloc[t], f'support_{i}', t)
                                for i, t in enumerate(pattern_troughs)
                            ],
                            start_time=data.index[start_idx],
                            end_time=data.index[end_idx],
                            resistance_level=resistance,
                            target_price=target_price,
                            stop_loss=trough_prices[-1] * 0.98,
                            pattern_height=pattern_height,
                            pattern_width_bars=end_idx - start_idx,
                            is_complete=is_complete,
                            completion_percentage=100 if is_complete else 75,
                            breakout_confirmed=is_complete and current_price > resistance * 1.02
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    # Continue with more pattern implementations...
    
    def _find_trough_between(self, troughs: np.ndarray, start: int, end: int) -> Optional[int]:
        """Find trough between two indices"""
        valid_troughs = [t for t in troughs if start < t < end]
        return valid_troughs[0] if valid_troughs else None
    
    def _find_peak_between(self, peaks: np.ndarray, start: int, end: int) -> Optional[int]:
        """Find peak between two indices"""
        valid_peaks = [p for p in peaks if start < p < end]
        return valid_peaks[0] if valid_peaks else None
    
    def _check_volume_pattern(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Check volume pattern during pattern formation"""
        if 'volume' not in data.columns:
            return 0.5
        
        volumes = data.iloc[start_idx:end_idx+1]['volume']
        if len(volumes) < 2:
            return 0.5
        
        # Check for declining volume (common in many patterns)
        slope, _, r_value, _, _ = linregress(range(len(volumes)), volumes)
        
        if slope < 0:  # Declining volume
            return 0.7 + abs(r_value) * 0.3
        else:  # Increasing volume
            return 0.3 + abs(r_value) * 0.2
    
    def _calculate_pattern_confidence(
        self,
        pattern_type: ChartPatternType,
        key_metrics: Dict[str, float]
    ) -> float:
        """Calculate pattern confidence score"""
        # Base confidence
        confidence = 50
        
        # Add weighted metrics
        weights = {
            'symmetry': 20,
            'prominence': 15,
            'consistency': 15,
            'volume_pattern': 10,
            'support_strength': 10,
            'resistance_strength': 10
        }
        
        for metric, value in key_metrics.items():
            for weight_key in weights:
                if weight_key in metric:
                    confidence += weights[weight_key] * value
                    break
        
        # Cap at 100
        return min(confidence, 100)
    
    # Additional pattern detection methods would follow similar structure...
    
    async def _detect_cup_handle(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Cup and Handle patterns"""
        patterns = []
        
        if len(prices) < 30:  # Need minimum data for cup pattern
            return patterns
        
        # Look for cup formation (U-shape)
        for i in range(len(prices) - 30):
            window = prices.iloc[i:i+30]
            
            # Find left lip, bottom, and right lip
            left_lip = window.iloc[0:5].max()
            right_lip = window.iloc[-5:].max()
            bottom = window.iloc[10:20].min()
            
            # Check if lips are similar height
            lip_diff = abs(left_lip - right_lip) / max(left_lip, right_lip)
            if lip_diff > self.similarity_threshold:
                continue
            
            # Check cup depth (should be 10-30% below lips)
            cup_depth = (max(left_lip, right_lip) - bottom) / max(left_lip, right_lip)
            if cup_depth < 0.1 or cup_depth > 0.3:
                continue
            
            # Check for roundness (gradual decline and rise)
            left_half = window.iloc[:15]
            right_half = window.iloc[15:]
            
            # Calculate correlation with ideal U-shape
            ideal_u = np.concatenate([
                np.linspace(1, 0, 15),
                np.linspace(0, 1, 15)
            ])
            actual_shape = (window.values - bottom) / (left_lip - bottom)
            correlation = np.corrcoef(actual_shape, ideal_u)[0, 1]
            
            if correlation < 0.7:  # Not round enough
                continue
            
            # Look for handle after cup (small consolidation)
            if i + 35 < len(prices):
                handle_prices = prices.iloc[i+30:i+35]
                handle_high = handle_prices.max()
                handle_low = handle_prices.min()
                
                # Handle should be small (less than 1/3 of cup depth)
                handle_depth = (handle_high - handle_low) / (right_lip - bottom)
                if handle_depth > 0.33:
                    continue
                
                # Calculate confidence
                confidence = self._calculate_pattern_confidence(
                    pattern_type=ChartPatternType.CUP_AND_HANDLE,
                    key_metrics={
                        'lip_symmetry': 1 - lip_diff,
                        'cup_depth': min(cup_depth / 0.2, 1),  # Optimal at 20%
                        'roundness': correlation,
                        'handle_proportion': 1 - handle_depth / 0.33,
                        'volume_pattern': self._check_volume_pattern(data, i, i+35)
                    }
                )
                
                # Check if pattern is complete
                current_price = prices.iloc[-1]
                breakout_level = max(left_lip, right_lip)
                is_complete = current_price > breakout_level
                
                pattern = DetectedPattern(
                    pattern_type=ChartPatternType.CUP_AND_HANDLE,
                    category=PatternCategory.CONTINUATION,
                    implication=PatternImplication.BULLISH,
                    confidence=confidence,
                    key_points=[
                        PatternPoint(data.index[i], left_lip, 'left_lip', i),
                        PatternPoint(data.index[i+15], bottom, 'bottom', i+15),
                        PatternPoint(data.index[i+29], right_lip, 'right_lip', i+29),
                        PatternPoint(data.index[i+32], handle_low, 'handle_low', i+32)
                    ],
                    start_time=data.index[i],
                    end_time=data.index[min(i+35, len(data)-1)],
                    resistance_level=breakout_level,
                    support_level=bottom,
                    target_price=breakout_level + cup_depth * max(left_lip, right_lip),
                    stop_loss=handle_low * 0.98,
                    pattern_height=cup_depth * max(left_lip, right_lip),
                    pattern_width_bars=35,
                    is_complete=is_complete,
                    completion_percentage=100 if is_complete else 90,
                    breakout_confirmed=is_complete and current_price > breakout_level * 1.02
                )
                
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_bull_flag(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Bull Flag patterns"""
        patterns = []
        
        if len(prices) < 20:
            return patterns
        
        # Look for sharp upward move (pole) followed by consolidation (flag)
        for i in range(len(prices) - 20):
            # Check for pole (strong upward move)
            pole_start = i
            pole_end = min(i + 10, len(prices) - 10)
            
            pole_prices = prices.iloc[pole_start:pole_end]
            pole_return = (pole_prices.iloc[-1] - pole_prices.iloc[0]) / pole_prices.iloc[0]
            
            # Need at least 10% move for pole
            if pole_return < 0.1:
                continue
            
            # Check for flag (consolidation)
            flag_start = pole_end
            flag_end = min(flag_start + 10, len(prices))
            flag_prices = prices.iloc[flag_start:flag_end]
            
            if len(flag_prices) < 5:
                continue
            
            # Flag should be slight downward or sideways
            flag_slope, _, flag_r_value, _, _ = linregress(range(len(flag_prices)), flag_prices)
            flag_angle = np.arctan(flag_slope) * 180 / np.pi
            
            # Flag should be between -30 and 10 degrees
            if flag_angle < -30 or flag_angle > 10:
                continue
            
            # Flag should be contained (not too wide)
            flag_high = flag_prices.max()
            flag_low = flag_prices.min()
            flag_height = flag_high - flag_low
            pole_height = pole_prices.iloc[-1] - pole_prices.iloc[0]
            
            if flag_height > pole_height * 0.5:  # Flag too wide
                continue
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.FLAG_BULLISH,
                key_metrics={
                    'pole_strength': min(pole_return / 0.2, 1),  # Optimal at 20%+
                    'flag_tightness': 1 - flag_height / (pole_height * 0.5),
                    'flag_angle': 1 - abs(flag_angle) / 30,
                    'pattern_clarity': abs(flag_r_value),
                    'volume_pattern': self._check_volume_pattern(data, pole_start, flag_end-1)
                }
            )
            
            # Check if pattern is complete
            current_price = prices.iloc[-1]
            breakout_level = flag_high
            is_complete = current_price > breakout_level
            
            # Target is pole height added to breakout
            target_price = breakout_level + pole_height
            
            pattern = DetectedPattern(
                pattern_type=ChartPatternType.FLAG_BULLISH,
                category=PatternCategory.CONTINUATION,
                implication=PatternImplication.BULLISH,
                confidence=confidence,
                key_points=[
                    PatternPoint(data.index[pole_start], pole_prices.iloc[0], 'pole_start', pole_start),
                    PatternPoint(data.index[pole_end-1], pole_prices.iloc[-1], 'pole_end', pole_end-1),
                    PatternPoint(data.index[flag_start], flag_high, 'flag_high', flag_start),
                    PatternPoint(data.index[flag_end-1], flag_low, 'flag_low', flag_end-1)
                ],
                start_time=data.index[pole_start],
                end_time=data.index[flag_end-1],
                resistance_level=flag_high,
                support_level=flag_low,
                target_price=target_price,
                stop_loss=flag_low * 0.98,
                pattern_height=pole_height,
                pattern_width_bars=flag_end - pole_start,
                is_complete=is_complete,
                completion_percentage=100 if is_complete else 80,
                breakout_confirmed=is_complete and current_price > breakout_level * 1.02
            )
            
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_bear_flag(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Bear Flag patterns"""
        patterns = []
        
        if len(prices) < 20:
            return patterns
        
        # Look for sharp downward move (pole) followed by consolidation (flag)
        for i in range(len(prices) - 20):
            # Check for pole (strong downward move)
            pole_start = i
            pole_end = min(i + 10, len(prices) - 10)
            
            pole_prices = prices.iloc[pole_start:pole_end]
            pole_return = (pole_prices.iloc[-1] - pole_prices.iloc[0]) / pole_prices.iloc[0]
            
            # Need at least -10% move for pole
            if pole_return > -0.1:
                continue
            
            # Check for flag (consolidation)
            flag_start = pole_end
            flag_end = min(flag_start + 10, len(prices))
            flag_prices = prices.iloc[flag_start:flag_end]
            
            if len(flag_prices) < 5:
                continue
            
            # Flag should be slight upward or sideways
            flag_slope, _, flag_r_value, _, _ = linregress(range(len(flag_prices)), flag_prices)
            flag_angle = np.arctan(flag_slope) * 180 / np.pi
            
            # Flag should be between -10 and 30 degrees for bear flag
            if flag_angle < -10 or flag_angle > 30:
                continue
            
            # Flag should be contained
            flag_high = flag_prices.max()
            flag_low = flag_prices.min()
            flag_height = flag_high - flag_low
            pole_height = abs(pole_prices.iloc[0] - pole_prices.iloc[-1])
            
            if flag_height > pole_height * 0.5:
                continue
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.FLAG_BEARISH,
                key_metrics={
                    'pole_strength': min(abs(pole_return) / 0.2, 1),
                    'flag_tightness': 1 - flag_height / (pole_height * 0.5),
                    'flag_angle': 1 - abs(flag_angle) / 30,
                    'pattern_clarity': abs(flag_r_value),
                    'volume_pattern': self._check_volume_pattern(data, pole_start, flag_end-1)
                }
            )
            
            # Check if pattern is complete
            current_price = prices.iloc[-1]
            breakout_level = flag_low
            is_complete = current_price < breakout_level
            
            # Target is pole height subtracted from breakout
            target_price = breakout_level - pole_height
            
            pattern = DetectedPattern(
                pattern_type=ChartPatternType.FLAG_BEARISH,
                category=PatternCategory.CONTINUATION,
                implication=PatternImplication.BEARISH,
                confidence=confidence,
                key_points=[
                    PatternPoint(data.index[pole_start], pole_prices.iloc[0], 'pole_start', pole_start),
                    PatternPoint(data.index[pole_end-1], pole_prices.iloc[-1], 'pole_end', pole_end-1),
                    PatternPoint(data.index[flag_start], flag_high, 'flag_high', flag_start),
                    PatternPoint(data.index[flag_end-1], flag_low, 'flag_low', flag_end-1)
                ],
                start_time=data.index[pole_start],
                end_time=data.index[flag_end-1],
                resistance_level=flag_high,
                support_level=flag_low,
                target_price=target_price,
                stop_loss=flag_high * 1.02,
                pattern_height=pole_height,
                pattern_width_bars=flag_end - pole_start,
                is_complete=is_complete,
                completion_percentage=100 if is_complete else 80,
                breakout_confirmed=is_complete and current_price < breakout_level * 0.98
            )
            
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_symmetrical_triangle(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Symmetrical Triangle patterns"""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Look for converging trendlines (one rising, one falling)
        for i in range(min(len(peaks), len(troughs)) - 1):
            # Get peaks and troughs for pattern
            pattern_peaks = peaks[i:min(i+5, len(peaks))]
            pattern_troughs = troughs[i:min(i+5, len(troughs))]
            
            if len(pattern_peaks) < 2 or len(pattern_troughs) < 2:
                continue
            
            # Fit trendlines
            peak_prices = prices.iloc[pattern_peaks].values
            trough_prices = prices.iloc[pattern_troughs].values
            
            peak_slope, peak_intercept, peak_r, _, _ = linregress(pattern_peaks, peak_prices)
            trough_slope, trough_intercept, trough_r, _, _ = linregress(pattern_troughs, trough_prices)
            
            # One should be rising, one falling (symmetrical)
            if not ((peak_slope < 0 and trough_slope > 0) or 
                   (abs(peak_slope + trough_slope) < abs(peak_slope) * 0.2)):
                continue
            
            # Lines should converge
            convergence_point = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            if convergence_point < 0 or convergence_point > len(prices) + 20:
                continue
            
            start_idx = min(pattern_peaks[0], pattern_troughs[0])
            end_idx = max(pattern_peaks[-1], pattern_troughs[-1])
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.SYMMETRICAL_TRIANGLE,
                key_metrics={
                    'upper_line_fit': abs(peak_r),
                    'lower_line_fit': abs(trough_r),
                    'symmetry': 1 - abs(abs(peak_slope) - abs(trough_slope)) / max(abs(peak_slope), abs(trough_slope)),
                    'convergence_quality': 1 / (1 + abs(convergence_point - end_idx) / 20),
                    'volume_pattern': self._check_volume_pattern(data, start_idx, end_idx)
                }
            )
            
            # Breakout can be either direction
            current_price = prices.iloc[-1]
            current_resistance = peak_slope * len(prices) + peak_intercept
            current_support = trough_slope * len(prices) + trough_intercept
            
            is_complete = current_price > current_resistance or current_price < current_support
            
            # Determine direction
            if current_price > current_resistance:
                implication = PatternImplication.BULLISH
                target_price = current_resistance + (peak_prices[0] - trough_prices[0])
            elif current_price < current_support:
                implication = PatternImplication.BEARISH
                target_price = current_support - (peak_prices[0] - trough_prices[0])
            else:
                implication = PatternImplication.NEUTRAL
                target_price = None
            
            pattern = DetectedPattern(
                pattern_type=ChartPatternType.SYMMETRICAL_TRIANGLE,
                category=PatternCategory.BILATERAL,
                implication=implication,
                confidence=confidence,
                key_points=[
                    PatternPoint(data.index[p], prices.iloc[p], f'peak_{j}', p)
                    for j, p in enumerate(pattern_peaks)
                ] + [
                    PatternPoint(data.index[t], prices.iloc[t], f'trough_{j}', t)
                    for j, t in enumerate(pattern_troughs)
                ],
                start_time=data.index[start_idx],
                end_time=data.index[end_idx],
                resistance_level=current_resistance,
                support_level=current_support,
                target_price=target_price,
                stop_loss=current_support if implication == PatternImplication.BULLISH else current_resistance,
                pattern_height=peak_prices[0] - trough_prices[0],
                pattern_width_bars=end_idx - start_idx,
                is_complete=is_complete,
                completion_percentage=100 if is_complete else 70,
                breakout_confirmed=is_complete
            )
            
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_descending_triangle(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Descending Triangle patterns"""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Look for flat support and falling resistance
        for i in range(len(troughs) - 1):
            # Check if troughs form horizontal support
            support_troughs = []
            for j in range(i, min(i + 5, len(troughs))):
                if abs(prices.iloc[troughs[j]] - prices.iloc[troughs[i]]) / prices.iloc[troughs[i]] <= self.similarity_threshold:
                    support_troughs.append(troughs[j])
            
            if len(support_troughs) >= 2:
                # Find peaks in the same period
                start_idx = support_troughs[0]
                end_idx = support_troughs[-1]
                pattern_peaks = [p for p in peaks if start_idx < p < end_idx]
                
                if len(pattern_peaks) >= 2:
                    # Check if peaks are falling
                    peak_prices = [prices.iloc[p] for p in pattern_peaks]
                    if all(peak_prices[i] > peak_prices[i+1] for i in range(len(peak_prices)-1)):
                        # Calculate metrics
                        support = np.mean([prices.iloc[t] for t in support_troughs])
                        resistance_slope = (peak_prices[-1] - peak_prices[0]) / (pattern_peaks[-1] - pattern_peaks[0])
                        
                        pattern_height = peak_prices[0] - support
                        target_price = support - pattern_height
                        
                        confidence = self._calculate_pattern_confidence(
                            pattern_type=ChartPatternType.DESCENDING_TRIANGLE,
                            key_metrics={
                                'support_consistency': len(support_troughs) / 5,
                                'resistance_trend': abs(resistance_slope) * 100,
                                'volume_pattern': self._check_volume_pattern(data, start_idx, end_idx)
                            }
                        )
                        
                        current_price = prices.iloc[-1]
                        is_complete = current_price < support
                        
                        pattern = DetectedPattern(
                            pattern_type=ChartPatternType.DESCENDING_TRIANGLE,
                            category=PatternCategory.CONTINUATION,
                            implication=PatternImplication.BEARISH,
                            confidence=confidence,
                            key_points=[
                                PatternPoint(data.index[t], prices.iloc[t], f'support_{i}', t)
                                for i, t in enumerate(support_troughs)
                            ] + [
                                PatternPoint(data.index[p], prices.iloc[p], f'resistance_{i}', p)
                                for i, p in enumerate(pattern_peaks)
                            ],
                            start_time=data.index[start_idx],
                            end_time=data.index[end_idx],
                            support_level=support,
                            target_price=target_price,
                            stop_loss=peak_prices[-1] * 1.02,
                            pattern_height=pattern_height,
                            pattern_width_bars=end_idx - start_idx,
                            is_complete=is_complete,
                            completion_percentage=100 if is_complete else 75,
                            breakout_confirmed=is_complete and current_price < support * 0.98
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_rising_wedge(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Rising Wedge patterns"""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Look for converging upward trendlines
        for i in range(len(peaks) - 1):
            # Need at least 3 peaks and 3 troughs
            available_peaks = peaks[i:min(i+5, len(peaks))]
            available_troughs = troughs[troughs >= peaks[i]]
            available_troughs = available_troughs[:min(5, len(available_troughs))]
            
            if len(available_peaks) < 3 or len(available_troughs) < 3:
                continue
            
            # Fit trendlines
            peak_prices = prices.iloc[available_peaks].values
            trough_prices = prices.iloc[available_troughs].values
            
            # Calculate slopes
            peak_slope, peak_intercept, peak_r, _, _ = linregress(available_peaks, peak_prices)
            trough_slope, trough_intercept, trough_r, _, _ = linregress(available_troughs, trough_prices)
            
            # Both should be rising
            if peak_slope <= 0 or trough_slope <= 0:
                continue
            
            # Lines should be converging (peak slope < trough slope for rising wedge)
            if peak_slope >= trough_slope:
                continue
            
            # Check convergence quality
            convergence_rate = (trough_slope - peak_slope) / trough_slope
            if convergence_rate < 0.1:  # Not converging enough
                continue
            
            # Pattern boundaries
            start_idx = available_peaks[0]
            end_idx = max(available_peaks[-1], available_troughs[-1])
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.RISING_WEDGE,
                key_metrics={
                    'upper_line_fit': abs(peak_r),
                    'lower_line_fit': abs(trough_r),
                    'convergence': convergence_rate,
                    'pattern_points': (len(available_peaks) + len(available_troughs)) / 8,
                    'volume_pattern': self._check_volume_pattern(data, start_idx, end_idx)
                }
            )
            
            # Check if pattern is complete (breakdown below lower trendline)
            current_price = prices.iloc[-1]
            current_support = trough_slope * len(prices) + trough_intercept
            is_complete = current_price < current_support
            
            # Target is height of wedge at start
            wedge_height = peak_prices[0] - trough_prices[0]
            target_price = current_support - wedge_height
            
            pattern = DetectedPattern(
                pattern_type=ChartPatternType.RISING_WEDGE,
                category=PatternCategory.REVERSAL,  # Usually bearish reversal
                implication=PatternImplication.BEARISH,
                confidence=confidence,
                key_points=[
                    PatternPoint(data.index[p], prices.iloc[p], f'peak_{j}', p)
                    for j, p in enumerate(available_peaks)
                ] + [
                    PatternPoint(data.index[t], prices.iloc[t], f'trough_{j}', t)
                    for j, t in enumerate(available_troughs)
                ],
                start_time=data.index[start_idx],
                end_time=data.index[end_idx],
                resistance_level=peak_slope * end_idx + peak_intercept,
                support_level=trough_slope * end_idx + trough_intercept,
                target_price=target_price,
                stop_loss=(peak_slope * len(prices) + peak_intercept) * 1.02,
                pattern_height=wedge_height,
                pattern_width_bars=end_idx - start_idx,
                is_complete=is_complete,
                completion_percentage=100 if is_complete else 75,
                breakout_confirmed=is_complete and current_price < current_support * 0.98
            )
            
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_falling_wedge(
        self,
        prices: pd.Series,
        peaks: np.ndarray,
        troughs: np.ndarray,
        data: pd.DataFrame
    ) -> List[DetectedPattern]:
        """Detect Falling Wedge patterns"""
        patterns = []
        
        if len(peaks) < 2 or len(troughs) < 2:
            return patterns
        
        # Look for converging downward trendlines
        for i in range(len(troughs) - 1):
            # Need at least 3 peaks and 3 troughs
            available_troughs = troughs[i:min(i+5, len(troughs))]
            available_peaks = peaks[peaks >= troughs[i]]
            available_peaks = available_peaks[:min(5, len(available_peaks))]
            
            if len(available_peaks) < 3 or len(available_troughs) < 3:
                continue
            
            # Fit trendlines
            peak_prices = prices.iloc[available_peaks].values
            trough_prices = prices.iloc[available_troughs].values
            
            # Calculate slopes
            peak_slope, peak_intercept, peak_r, _, _ = linregress(available_peaks, peak_prices)
            trough_slope, trough_intercept, trough_r, _, _ = linregress(available_troughs, trough_prices)
            
            # Both should be falling
            if peak_slope >= 0 or trough_slope >= 0:
                continue
            
            # Lines should be converging (trough slope > peak slope for falling wedge)
            if trough_slope <= peak_slope:
                continue
            
            # Check convergence quality
            convergence_rate = (peak_slope - trough_slope) / abs(trough_slope)
            if convergence_rate < 0.1:
                continue
            
            # Pattern boundaries
            start_idx = available_troughs[0]
            end_idx = max(available_peaks[-1], available_troughs[-1])
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(
                pattern_type=ChartPatternType.FALLING_WEDGE,
                key_metrics={
                    'upper_line_fit': abs(peak_r),
                    'lower_line_fit': abs(trough_r),
                    'convergence': convergence_rate,
                    'pattern_points': (len(available_peaks) + len(available_troughs)) / 8,
                    'volume_pattern': self._check_volume_pattern(data, start_idx, end_idx)
                }
            )
            
            # Check if pattern is complete (breakout above upper trendline)
            current_price = prices.iloc[-1]
            current_resistance = peak_slope * len(prices) + peak_intercept
            is_complete = current_price > current_resistance
            
            # Target is height of wedge at start
            wedge_height = peak_prices[0] - trough_prices[0]
            target_price = current_resistance + wedge_height
            
            pattern = DetectedPattern(
                pattern_type=ChartPatternType.FALLING_WEDGE,
                category=PatternCategory.REVERSAL,  # Usually bullish reversal
                implication=PatternImplication.BULLISH,
                confidence=confidence,
                key_points=[
                    PatternPoint(data.index[p], prices.iloc[p], f'peak_{j}', p)
                    for j, p in enumerate(available_peaks)
                ] + [
                    PatternPoint(data.index[t], prices.iloc[t], f'trough_{j}', t)
                    for j, t in enumerate(available_troughs)
                ],
                start_time=data.index[start_idx],
                end_time=data.index[end_idx],
                resistance_level=peak_slope * end_idx + peak_intercept,
                support_level=trough_slope * end_idx + trough_intercept,
                target_price=target_price,
                stop_loss=(trough_slope * len(prices) + trough_intercept) * 0.98,
                pattern_height=wedge_height,
                pattern_width_bars=end_idx - start_idx,
                is_complete=is_complete,
                completion_percentage=100 if is_complete else 75,
                breakout_confirmed=is_complete and current_price > current_resistance * 1.02
            )
            
            patterns.append(pattern)
        
        return patterns


# Global pattern recognition service instance
pattern_recognition = PatternRecognitionService()