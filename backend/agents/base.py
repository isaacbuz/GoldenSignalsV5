"""
Base Agent Architecture for GoldenSignalsAI
Provides foundation for all trading agents with performance tracking
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from models.signal import Signal as SignalModel, SignalAction
from services.signal_service import SignalService

logger = get_logger(__name__)


class AgentCapability(str, Enum):
    """Agent capability types"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SOCIAL_MEDIA_MONITORING = "social_media_monitoring"
    NEWS_ANALYSIS = "news_analysis"
    OPTIONS_ANALYSIS = "options_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    RISK_MANAGEMENT = "risk_management"
    PATTERN_RECOGNITION = "pattern_recognition"
    MARKET_REGIME_DETECTION = "market_regime_detection"


class AgentContext(BaseModel):
    """Context information passed to agents"""
    symbol: str
    timeframe: str = "1d"
    market_data: Dict[str, Any] = Field(default_factory=dict)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SignalStrength(str, Enum):
    """Signal strength levels"""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class Signal(BaseModel):
    """Trading signal with confidence and metadata"""
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    strength: SignalStrength
    source: str
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: List[str] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    indicators: Dict[str, float] = Field(default_factory=dict)
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_db_model(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert to database model format"""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "source": self.source,
            "agent_id": agent_id,
            "metadata": {
                "signal_id": self.signal_id,
                "strength": self.strength.value,
                "current_price": self.current_price,
                "target_price": self.target_price,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "risk_score": self.risk_score,
                "reasoning": self.reasoning,
                "features": self.features,
                "indicators": self.indicators,
                "market_conditions": self.market_conditions
            }
        }


class AgentConfig(BaseModel):
    """Configuration for an individual agent"""
    name: str
    version: str = "1.0.0"
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    timeout: int = Field(default=30, gt=0)  # seconds
    max_retries: int = Field(default=3, ge=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    min_data_points: int = Field(default=50, gt=0)
    
    class Config:
        extra = "allow"  # Allow additional fields for agent-specific config


class AgentPerformance(BaseModel):
    """Performance metrics for an agent"""
    agent_id: str
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_execution_time: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def update_accuracy(self) -> float:
        """Update and return current accuracy"""
        if self.total_signals > 0:
            self.accuracy = self.correct_signals / self.total_signals
            self.win_rate = self.accuracy * 100
        return self.accuracy


class BaseAgent(ABC):
    """
    Base class for all trading agents in GoldenSignalsAI
    
    Features:
    - Asynchronous execution
    - Performance tracking and metrics
    - Adaptive confidence scoring
    - Error handling and retries
    - Self-learning capabilities
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.performance = AgentPerformance(agent_id=self.agent_id)
        self._is_running = False
        self._stop_event = asyncio.Event()
        
        # Agent state for persistence
        self._state = {
            "model_parameters": {},
            "learned_patterns": {},
            "performance_history": [],
            "last_analysis_time": None,
            "cached_calculations": {}
        }
        
        # Track execution times for performance monitoring
        self._execution_times: List[float] = []
        self._max_execution_history = 100
        
        logger.info(f"Initialized {config.name} agent with ID: {self.agent_id}")
    
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Core analysis method that each agent must implement
        
        Args:
            market_data: Market data for analysis including price, volume, indicators
            
        Returns:
            Signal: Trading signal with confidence and metadata, or None
        """
        raise NotImplementedError("Subclasses must implement analyze_market_data")
    
    @abstractmethod
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for this agent
        
        Returns:
            List of data type strings (e.g., ['price', 'volume', 'indicators'])
        """
        raise NotImplementedError("Subclasses must implement get_required_data_types")
    
    async def execute_with_monitoring(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Execute analysis with comprehensive monitoring and error handling
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Signal if successful, None if failed
        """
        if not self.config.enabled:
            logger.debug(f"Agent {self.config.name} is disabled")
            return None
        
        # Validate required data
        if not self._validate_market_data(market_data):
            logger.warning(f"Agent {self.config.name}: Invalid or insufficient market data")
            return None
        
        start_time = time.time()
        attempt = 0
        
        while attempt < self.config.max_retries:
            try:
                # Execute with timeout
                signal = await asyncio.wait_for(
                    self.analyze(market_data),
                    timeout=self.config.timeout
                )
                
                if signal:
                    # Apply confidence threshold
                    if signal.confidence < self.config.confidence_threshold:
                        logger.debug(
                            f"Agent {self.config.name} signal confidence "
                            f"{signal.confidence:.3f} below threshold {self.config.confidence_threshold}"
                        )
                        return None
                    
                    # Record successful execution
                    execution_time = time.time() - start_time
                    await self._record_success(signal, execution_time)
                    
                    logger.debug(
                        f"Agent {self.config.name} generated signal: "
                        f"{signal.action.value} with confidence {signal.confidence:.3f}"
                    )
                    
                return signal
                
            except asyncio.TimeoutError:
                logger.warning(f"Agent {self.config.name} timed out on attempt {attempt + 1}")
                attempt += 1
                
            except Exception as e:
                logger.error(f"Agent {self.config.name} failed: {str(e)}", exc_info=True)
                attempt += 1
                
                if attempt >= self.config.max_retries:
                    await self._record_failure(e)
                    break
                
                # Exponential backoff
                await asyncio.sleep(min(2 ** attempt, 10))
        
        return None
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Validate that market data contains required fields"""
        required_types = self.get_required_data_types()
        
        # Check for basic required fields
        if 'symbol' not in market_data:
            return False
        
        # Check data types
        for data_type in required_types:
            if data_type == 'price' and 'price' not in market_data:
                return False
            elif data_type == 'volume' and 'volume' not in market_data:
                return False
            elif data_type == 'indicators' and 'indicators' not in market_data:
                return False
            elif data_type == 'ohlcv' and not all(k in market_data for k in ['open', 'high', 'low', 'close']):
                return False
        
        return True
    
    async def _record_success(self, signal: Signal, execution_time: float) -> None:
        """Record successful signal generation"""
        self.performance.total_signals += 1
        
        # Update execution time tracking
        self._execution_times.append(execution_time)
        if len(self._execution_times) > self._max_execution_history:
            self._execution_times.pop(0)
        
        # Update performance metrics
        self.performance.avg_execution_time = sum(self._execution_times) / len(self._execution_times)
        self.performance.avg_confidence = (
            (self.performance.avg_confidence * (self.performance.total_signals - 1) + signal.confidence)
            / self.performance.total_signals
        )
        self.performance.last_updated = datetime.utcnow()
        
        # Update state
        self._state["last_analysis_time"] = datetime.utcnow().isoformat()
        
        logger.info(
            f"Agent {self.config.name} signal recorded: "
            f"total={self.performance.total_signals}, "
            f"avg_time={self.performance.avg_execution_time:.3f}s"
        )
    
    async def _record_failure(self, error: Exception) -> None:
        """Record failed signal generation"""
        logger.error(f"Agent {self.config.name} failed after retries: {str(error)}")
        
        # Could track failure metrics here
        self._state["last_error"] = {
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def update_performance_feedback(self, signal_id: str, was_correct: bool, profit: Optional[float] = None) -> None:
        """
        Update agent performance based on signal outcome
        
        Args:
            signal_id: ID of the signal to update
            was_correct: Whether the signal prediction was correct
            profit: Profit/loss from the signal (optional)
        """
        if was_correct:
            self.performance.correct_signals += 1
        
        # Update accuracy
        self.performance.update_accuracy()
        
        # Track performance history
        self._state["performance_history"].append({
            "signal_id": signal_id,
            "was_correct": was_correct,
            "profit": profit,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only recent history
        if len(self._state["performance_history"]) > 1000:
            self._state["performance_history"] = self._state["performance_history"][-1000:]
        
        # Adaptive learning: adjust confidence threshold based on performance
        if self.performance.total_signals >= 10:  # Need minimum signals
            if self.performance.accuracy < 0.4:  # Performing poorly
                # Increase confidence threshold
                self.config.confidence_threshold = min(0.9, self.config.confidence_threshold + self.config.learning_rate)
            elif self.performance.accuracy > 0.7:  # Performing well
                # Decrease confidence threshold slightly
                self.config.confidence_threshold = max(0.5, self.config.confidence_threshold - self.config.learning_rate / 2)
        
        logger.debug(
            f"Agent {self.config.name} performance updated: "
            f"accuracy={self.performance.accuracy:.3f}, "
            f"confidence_threshold={self.config.confidence_threshold:.3f}"
        )
    
    def adjust_weight(self, new_weight: float) -> None:
        """Adjust agent weight based on performance"""
        self.config.weight = max(0.0, min(1.0, new_weight))
        logger.info(f"Agent {self.config.name} weight adjusted to {self.config.weight:.3f}")
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "accuracy": self.performance.accuracy,
            "total_signals": self.performance.total_signals,
            "correct_signals": self.performance.correct_signals,
            "avg_confidence": self.performance.avg_confidence,
            "avg_execution_time": self.performance.avg_execution_time,
            "weight": self.config.weight,
            "confidence_threshold": self.config.confidence_threshold,
            "enabled": self.config.enabled,
            "win_rate": self.performance.win_rate
        }
    
    async def start(self) -> None:
        """Start the agent"""
        self._is_running = True
        self._stop_event.clear()
        logger.info(f"Agent {self.config.name} started")
    
    async def stop(self) -> None:
        """Stop the agent"""
        self._is_running = False
        self._stop_event.set()
        logger.info(f"Agent {self.config.name} stopped")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        await self.stop()
        logger.info(f"Agent {self.config.name} shutdown complete")
    
    @property
    def is_running(self) -> bool:
        """Check if agent is running"""
        return self._is_running
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, weight={self.config.weight:.2f})"
    
    # Optional lifecycle hooks for subclasses
    async def initialize(self) -> None:
        """
        Optional async initialization hook.
        Subclasses can override to load models, warm caches, etc.
        """
        pass
    
    async def on_market_open(self) -> None:
        """Hook called when market opens"""
        pass
    
    async def on_market_close(self) -> None:
        """Hook called when market closes"""
        pass