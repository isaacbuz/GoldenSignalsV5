"""
Agent Performance Monitoring System
Tracks and analyzes agent performance metrics with learning capabilities
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json

from pydantic import BaseModel, Field
from sqlalchemy import select, and_, func

from agents.base import BaseAgent, Signal
from core.logging import get_logger
from core.events.bus import event_bus, EventTypes
from database.models import AgentPerformance, SignalHistory, TradeOutcome
from database.connection import get_db

logger = get_logger(__name__)


class PerformanceMetric(Enum):
    """Performance metric types"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    AVG_RETURN = "avg_return"
    SIGNAL_QUALITY = "signal_quality"
    RESPONSE_TIME = "response_time"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class AgentMetrics:
    """Comprehensive agent performance metrics"""
    agent_id: str
    agent_type: str
    
    # Accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Trading metrics
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Financial metrics
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Timing metrics
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    
    # Confidence metrics
    avg_confidence: float = 0.0
    confidence_calibration: float = 0.0  # How well confidence matches actual accuracy
    overconfidence_ratio: float = 0.0
    
    # Trend metrics
    performance_trend: str = "stable"  # improving, declining, stable
    recent_performance: float = 0.0
    
    # Meta metrics
    last_updated: datetime = field(default_factory=datetime.now)
    evaluation_period: str = "1d"
    sample_size: int = 0


class SignalOutcome(BaseModel):
    """Signal outcome tracking"""
    signal_id: str
    agent_id: str
    symbol: str
    signal_type: str  # buy, sell, hold
    confidence: float
    timestamp: datetime
    
    # Outcome
    outcome: Optional[str] = None  # success, failure, partial
    pnl: Optional[float] = None
    actual_return: Optional[float] = None
    expected_return: Optional[float] = None
    
    # Timing
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    holding_period: Optional[timedelta] = None
    
    # Context
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class PerformanceTracker:
    """
    Tracks individual agent performance
    """
    
    def __init__(self, agent_id: str, window_size: int = 100):
        self.agent_id = agent_id
        self.window_size = window_size
        
        # Rolling windows for recent performance
        self.recent_signals = deque(maxlen=window_size)
        self.recent_outcomes = deque(maxlen=window_size)
        self.recent_returns = deque(maxlen=window_size)
        self.recent_response_times = deque(maxlen=window_size)
        
        # Cumulative stats
        self.total_signals = 0
        self.successful_signals = 0
        self.failed_signals = 0
        self.cumulative_pnl = 0.0
        
        # Performance history
        self.performance_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[float] = []
        
        # Confidence calibration
        self.confidence_buckets = defaultdict(lambda: {"total": 0, "correct": 0})
    
    def record_signal(self, signal: Signal, response_time: float) -> None:
        """Record a new signal"""
        self.recent_signals.append(signal)
        self.recent_response_times.append(response_time)
        self.total_signals += 1
        
        # Track confidence buckets for calibration
        bucket = int(signal.confidence * 10) / 10  # Round to nearest 0.1
        self.confidence_buckets[bucket]["total"] += 1
    
    def record_outcome(self, outcome: SignalOutcome) -> None:
        """Record signal outcome"""
        self.recent_outcomes.append(outcome)
        
        if outcome.pnl:
            self.recent_returns.append(outcome.pnl)
            self.cumulative_pnl += outcome.pnl
        
        if outcome.outcome == "success":
            self.successful_signals += 1
            bucket = int(outcome.confidence * 10) / 10
            self.confidence_buckets[bucket]["correct"] += 1
        elif outcome.outcome == "failure":
            self.failed_signals += 1
        
        # Update performance history
        current_performance = self.calculate_performance_score()
        self.performance_history.append((datetime.now(), current_performance))
        
        # Update drawdown
        if self.cumulative_pnl > 0:
            peak = max([p for _, p in self.performance_history] + [0])
            drawdown = (peak - self.cumulative_pnl) / peak if peak > 0 else 0
            self.drawdown_history.append(drawdown)
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score"""
        if not self.recent_outcomes:
            return 0.5
        
        # Combine multiple factors
        win_rate = len([o for o in self.recent_outcomes if o.outcome == "success"]) / len(self.recent_outcomes)
        avg_return = np.mean(self.recent_returns) if self.recent_returns else 0
        consistency = 1 - np.std(self.recent_returns) if len(self.recent_returns) > 1 else 0.5
        
        # Weighted score
        score = (win_rate * 0.4 + 
                min(avg_return / 0.1, 1) * 0.4 +  # Normalize to [0, 1]
                consistency * 0.2)
        
        return min(max(score, 0), 1)  # Clamp to [0, 1]
    
    def get_metrics(self) -> AgentMetrics:
        """Get current performance metrics"""
        metrics = AgentMetrics(
            agent_id=self.agent_id,
            agent_type="unknown",  # Will be set by monitor
            total_signals=self.total_signals
        )
        
        if self.recent_outcomes:
            # Win rate
            successes = [o for o in self.recent_outcomes if o.outcome == "success"]
            metrics.winning_signals = len(successes)
            metrics.losing_signals = len([o for o in self.recent_outcomes if o.outcome == "failure"])
            metrics.win_rate = metrics.winning_signals / len(self.recent_outcomes)
            
            # Accuracy metrics (treating as binary classification)
            metrics.accuracy = metrics.win_rate
            metrics.precision = metrics.win_rate  # For trading, precision = win rate
            metrics.recall = 1.0  # Assuming we take all signals
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall) if (metrics.precision + metrics.recall) > 0 else 0
        
        if self.recent_returns:
            returns = list(self.recent_returns)
            
            # Financial metrics
            metrics.total_pnl = self.cumulative_pnl
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            
            metrics.avg_win = np.mean(wins) if wins else 0
            metrics.avg_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Sharpe ratio (annualized)
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                metrics.sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    downside_std = np.std(downside_returns)
                    metrics.sortino_ratio = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            
            # Max drawdown
            metrics.max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0
        
        # Timing metrics
        if self.recent_response_times:
            metrics.avg_response_time = np.mean(self.recent_response_times)
            metrics.max_response_time = max(self.recent_response_times)
        
        # Confidence metrics
        if self.recent_signals:
            metrics.avg_confidence = np.mean([s.confidence for s in self.recent_signals])
            
            # Calibration: how well confidence matches accuracy
            calibration_error = 0
            total_buckets = 0
            
            for confidence, stats in self.confidence_buckets.items():
                if stats["total"] > 0:
                    actual_accuracy = stats["correct"] / stats["total"]
                    calibration_error += abs(confidence - actual_accuracy) * stats["total"]
                    total_buckets += stats["total"]
            
            metrics.confidence_calibration = 1 - (calibration_error / total_buckets) if total_buckets > 0 else 0.5
            
            # Overconfidence ratio
            if metrics.win_rate > 0:
                metrics.overconfidence_ratio = metrics.avg_confidence / metrics.win_rate
        
        # Performance trend
        if len(self.performance_history) >= 10:
            recent = [p for _, p in self.performance_history[-10:]]
            older = [p for _, p in self.performance_history[-20:-10]] if len(self.performance_history) >= 20 else recent
            
            recent_avg = np.mean(recent)
            older_avg = np.mean(older)
            
            if recent_avg > older_avg * 1.1:
                metrics.performance_trend = "improving"
            elif recent_avg < older_avg * 0.9:
                metrics.performance_trend = "declining"
            else:
                metrics.performance_trend = "stable"
            
            metrics.recent_performance = recent_avg
        
        metrics.sample_size = len(self.recent_outcomes)
        
        return metrics


class AgentPerformanceMonitor:
    """
    Centralized agent performance monitoring system
    """
    
    def __init__(self):
        self.trackers: Dict[str, PerformanceTracker] = {}
        self.agent_registry: Dict[str, BaseAgent] = {}
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Performance thresholds
        self.thresholds = {
            "min_accuracy": 0.55,
            "min_sharpe": 0.5,
            "max_drawdown": 0.2,
            "min_signals": 10  # Minimum signals before evaluation
        }
        
        # Alert callbacks
        self.alert_handlers: List[Callable] = []
        
        # ML model for performance prediction
        self.performance_predictor = None
        
        logger.info("Agent Performance Monitor initialized")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for monitoring"""
        agent_id = agent.config.name
        self.agent_registry[agent_id] = agent
        
        if agent_id not in self.trackers:
            self.trackers[agent_id] = PerformanceTracker(agent_id)
        
        logger.info(f"Registered agent for monitoring: {agent_id}")
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Subscribe to events
        await event_bus.subscribe(EventTypes.SIGNAL_GENERATED, self._on_signal_generated)
        await event_bus.subscribe(EventTypes.TRADE_EXECUTED, self._on_trade_executed)
        await event_bus.subscribe(EventTypes.POSITION_CLOSED, self._on_position_closed)
        
        # Start monitoring loop
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Evaluate all agents
                for agent_id, tracker in self.trackers.items():
                    metrics = tracker.get_metrics()
                    
                    # Check for performance issues
                    await self._check_performance_thresholds(agent_id, metrics)
                    
                    # Store metrics in database
                    await self._store_metrics(agent_id, metrics)
                    
                    # Predict future performance
                    if self.performance_predictor:
                        prediction = await self._predict_performance(agent_id, metrics)
                        if prediction["risk"] == "high":
                            await self._send_alert(
                                agent_id,
                                "Performance Risk",
                                f"High risk of performance degradation predicted: {prediction['reason']}"
                            )
                
                # Publish aggregated metrics
                await self._publish_metrics()
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _on_signal_generated(self, event) -> None:
        """Handle signal generation events"""
        try:
            data = event.data
            agent_id = data.get("agent_id")
            signal = data.get("signal")
            response_time = data.get("response_time", 0)
            
            if agent_id and agent_id in self.trackers:
                self.trackers[agent_id].record_signal(signal, response_time)
                
        except Exception as e:
            logger.error(f"Error handling signal event: {str(e)}")
    
    async def _on_trade_executed(self, event) -> None:
        """Handle trade execution events"""
        try:
            data = event.data
            signal_id = data.get("signal_id")
            agent_id = data.get("agent_id")
            
            # Link trade to signal for outcome tracking
            if agent_id:
                # Store trade-signal mapping for later outcome calculation
                pass
                
        except Exception as e:
            logger.error(f"Error handling trade event: {str(e)}")
    
    async def _on_position_closed(self, event) -> None:
        """Handle position closure events"""
        try:
            data = event.data
            agent_id = data.get("agent_id")
            signal_id = data.get("signal_id")
            pnl = data.get("pnl")
            
            if agent_id and agent_id in self.trackers:
                # Create outcome
                outcome = SignalOutcome(
                    signal_id=signal_id or "unknown",
                    agent_id=agent_id,
                    symbol=data.get("symbol", "unknown"),
                    signal_type=data.get("signal_type", "unknown"),
                    confidence=data.get("confidence", 0.5),
                    timestamp=datetime.now(),
                    outcome="success" if pnl > 0 else "failure",
                    pnl=pnl,
                    actual_return=data.get("return", 0),
                    entry_price=data.get("entry_price"),
                    exit_price=data.get("exit_price")
                )
                
                self.trackers[agent_id].record_outcome(outcome)
                
        except Exception as e:
            logger.error(f"Error handling position closed event: {str(e)}")
    
    async def _check_performance_thresholds(self, agent_id: str, metrics: AgentMetrics) -> None:
        """Check if agent performance meets thresholds"""
        if metrics.sample_size < self.thresholds["min_signals"]:
            return  # Not enough data
        
        alerts = []
        
        # Check accuracy
        if metrics.accuracy < self.thresholds["min_accuracy"]:
            alerts.append(f"Low accuracy: {metrics.accuracy:.2%}")
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.thresholds["min_sharpe"]:
            alerts.append(f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        
        # Check drawdown
        if metrics.max_drawdown > self.thresholds["max_drawdown"]:
            alerts.append(f"High drawdown: {metrics.max_drawdown:.2%}")
        
        # Check confidence calibration
        if metrics.overconfidence_ratio > 1.5:
            alerts.append(f"Overconfident: confidence {metrics.avg_confidence:.2%} vs accuracy {metrics.accuracy:.2%}")
        
        # Check trend
        if metrics.performance_trend == "declining" and metrics.recent_performance < 0.4:
            alerts.append(f"Declining performance trend")
        
        # Send alerts
        if alerts:
            await self._send_alert(
                agent_id,
                "Performance Alert",
                "\n".join(alerts)
            )
    
    async def _send_alert(self, agent_id: str, alert_type: str, message: str) -> None:
        """Send performance alert"""
        alert = {
            "agent_id": agent_id,
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now()
        }
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {str(e)}")
        
        # Publish event
        await event_bus.publish(
            "performance.alert",
            data=alert
        )
        
        logger.warning(f"Performance alert for {agent_id}: {message}")
    
    async def _store_metrics(self, agent_id: str, metrics: AgentMetrics) -> None:
        """Store metrics in database"""
        try:
            async with get_db() as session:
                record = AgentPerformance(
                    agent_id=agent_id,
                    timestamp=datetime.now(),
                    accuracy=metrics.accuracy,
                    win_rate=metrics.win_rate,
                    sharpe_ratio=metrics.sharpe_ratio,
                    total_pnl=metrics.total_pnl,
                    max_drawdown=metrics.max_drawdown,
                    avg_confidence=metrics.avg_confidence,
                    total_signals=metrics.total_signals,
                    metrics_json=json.dumps(metrics.__dict__, default=str)
                )
                
                session.add(record)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {str(e)}")
    
    async def _predict_performance(self, agent_id: str, metrics: AgentMetrics) -> Dict[str, Any]:
        """Predict future agent performance"""
        # This would use ML model to predict performance
        # For now, use simple heuristics
        
        prediction = {
            "risk": "low",
            "expected_performance": metrics.recent_performance,
            "confidence": 0.7,
            "reason": ""
        }
        
        # Check for warning signs
        if metrics.performance_trend == "declining":
            prediction["risk"] = "medium"
            prediction["reason"] = "Declining performance trend"
        
        if metrics.max_drawdown > 0.15:
            prediction["risk"] = "high"
            prediction["reason"] = "High drawdown risk"
        
        if metrics.overconfidence_ratio > 2:
            prediction["risk"] = "high"
            prediction["reason"] = "Severe overconfidence"
        
        return prediction
    
    async def _publish_metrics(self) -> None:
        """Publish aggregated metrics"""
        all_metrics = {}
        
        for agent_id, tracker in self.trackers.items():
            metrics = tracker.get_metrics()
            all_metrics[agent_id] = {
                "accuracy": metrics.accuracy,
                "win_rate": metrics.win_rate,
                "sharpe_ratio": metrics.sharpe_ratio,
                "total_pnl": metrics.total_pnl,
                "trend": metrics.performance_trend,
                "recent_performance": metrics.recent_performance
            }
        
        await event_bus.publish(
            "performance.metrics",
            data=all_metrics
        )
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for specific agent"""
        if agent_id in self.trackers:
            return self.trackers[agent_id].get_metrics()
        return None
    
    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents"""
        return {
            agent_id: tracker.get_metrics()
            for agent_id, tracker in self.trackers.items()
        }
    
    async def get_historical_metrics(
        self,
        agent_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical metrics from database"""
        async with get_db() as session:
            query = select(AgentPerformance).where(
                and_(
                    AgentPerformance.agent_id == agent_id,
                    AgentPerformance.timestamp >= start_date,
                    AgentPerformance.timestamp <= end_date
                )
            ).order_by(AgentPerformance.timestamp)
            
            result = await session.execute(query)
            records = result.scalars().all()
            
            if records:
                data = [
                    {
                        "timestamp": r.timestamp,
                        "accuracy": r.accuracy,
                        "win_rate": r.win_rate,
                        "sharpe_ratio": r.sharpe_ratio,
                        "total_pnl": r.total_pnl,
                        "max_drawdown": r.max_drawdown
                    }
                    for r in records
                ]
                return pd.DataFrame(data)
            
            return pd.DataFrame()
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update performance thresholds"""
        self.thresholds.update(thresholds)
        logger.info(f"Updated performance thresholds: {thresholds}")


# Global monitor instance
performance_monitor = AgentPerformanceMonitor()