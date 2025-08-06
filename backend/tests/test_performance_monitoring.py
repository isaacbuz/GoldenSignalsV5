"""
Tests for Agent Performance Monitoring System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from monitoring.agent_performance import (
    AgentPerformanceMonitor,
    PerformanceTracker,
    AgentMetrics,
    SignalOutcome,
    PerformanceMetric
)
from agents.base import Signal, AgentConfig
from core.events.bus import Event, EventPriority


class TestPerformanceTracker:
    """Test PerformanceTracker class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.tracker = PerformanceTracker("test_agent", window_size=10)
    
    def test_record_signal(self):
        """Test recording signals"""
        signal = Signal(
            symbol="AAPL",
            action="buy",
            confidence=0.8,
            metadata={}
        )
        
        self.tracker.record_signal(signal, response_time=0.5)
        
        assert self.tracker.total_signals == 1
        assert len(self.tracker.recent_signals) == 1
        assert self.tracker.recent_signals[0] == signal
        assert self.tracker.recent_response_times[0] == 0.5
    
    def test_record_outcome(self):
        """Test recording signal outcomes"""
        outcome = SignalOutcome(
            signal_id="sig_001",
            agent_id="test_agent",
            symbol="AAPL",
            signal_type="buy",
            confidence=0.8,
            timestamp=datetime.now(),
            outcome="success",
            pnl=100.0
        )
        
        self.tracker.record_outcome(outcome)
        
        assert self.tracker.successful_signals == 1
        assert self.tracker.cumulative_pnl == 100.0
        assert len(self.tracker.recent_outcomes) == 1
        assert len(self.tracker.recent_returns) == 1
    
    def test_calculate_performance_score(self):
        """Test performance score calculation"""
        # Add some outcomes
        for i in range(5):
            outcome = SignalOutcome(
                signal_id=f"sig_{i}",
                agent_id="test_agent",
                symbol="AAPL",
                signal_type="buy",
                confidence=0.7,
                timestamp=datetime.now(),
                outcome="success" if i < 3 else "failure",
                pnl=50.0 if i < 3 else -30.0
            )
            self.tracker.record_outcome(outcome)
        
        score = self.tracker.calculate_performance_score()
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be positive with 3/5 wins
    
    def test_get_metrics(self):
        """Test getting comprehensive metrics"""
        # Add signals and outcomes
        for i in range(10):
            signal = Signal(
                symbol="AAPL",
                action="buy" if i % 2 == 0 else "sell",
                confidence=0.6 + (i * 0.03),
                metadata={}
            )
            self.tracker.record_signal(signal, response_time=0.1 * (i + 1))
            
            outcome = SignalOutcome(
                signal_id=f"sig_{i}",
                agent_id="test_agent",
                symbol="AAPL",
                signal_type=signal.action,
                confidence=signal.confidence,
                timestamp=datetime.now(),
                outcome="success" if i < 6 else "failure",
                pnl=100.0 if i < 6 else -50.0
            )
            self.tracker.record_outcome(outcome)
        
        metrics = self.tracker.get_metrics()
        
        assert isinstance(metrics, AgentMetrics)
        assert metrics.agent_id == "test_agent"
        assert metrics.total_signals == 10
        assert metrics.winning_signals == 6
        assert metrics.losing_signals == 4
        assert metrics.win_rate == 0.6
        assert metrics.total_pnl == 400.0  # 6*100 - 4*50
        assert metrics.avg_response_time > 0
        assert metrics.avg_confidence > 0
        assert metrics.sample_size == 10
    
    def test_confidence_calibration(self):
        """Test confidence calibration calculation"""
        # Add signals with varying confidence
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        outcomes = ["success", "success", "failure", "success", "failure"]
        
        for conf, outcome in zip(confidences, outcomes):
            signal = Signal(
                symbol="AAPL",
                action="buy",
                confidence=conf,
                metadata={}
            )
            self.tracker.record_signal(signal, response_time=0.1)
            
            signal_outcome = SignalOutcome(
                signal_id=f"sig_{conf}",
                agent_id="test_agent",
                symbol="AAPL",
                signal_type="buy",
                confidence=conf,
                timestamp=datetime.now(),
                outcome=outcome,
                pnl=10 if outcome == "success" else -10
            )
            self.tracker.record_outcome(signal_outcome)
        
        metrics = self.tracker.get_metrics()
        
        # Calibration should be between 0 and 1
        assert 0 <= metrics.confidence_calibration <= 1
        # Overconfidence ratio should be calculated
        assert metrics.overconfidence_ratio > 0


class TestAgentPerformanceMonitor:
    """Test AgentPerformanceMonitor class"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance"""
        return AgentPerformanceMonitor()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent"""
        agent = Mock()
        agent.config = AgentConfig(
            name="test_agent",
            description="Test agent"
        )
        return agent
    
    def test_register_agent(self, monitor, mock_agent):
        """Test agent registration"""
        monitor.register_agent(mock_agent)
        
        assert "test_agent" in monitor.agent_registry
        assert "test_agent" in monitor.trackers
        assert monitor.agent_registry["test_agent"] == mock_agent
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring"""
        with patch('core.events.bus.event_bus.subscribe', new_callable=AsyncMock):
            await monitor.start_monitoring()
            assert monitor.monitoring_active
            assert monitor._monitor_task is not None
            
            await monitor.stop_monitoring()
            assert not monitor.monitoring_active
    
    @pytest.mark.asyncio
    async def test_on_signal_generated(self, monitor, mock_agent):
        """Test handling signal generation events"""
        monitor.register_agent(mock_agent)
        
        event = Event(
            type="signal.generated",
            data={
                "agent_id": "test_agent",
                "signal": Signal(
                    symbol="AAPL",
                    action="buy",
                    confidence=0.75,
                    metadata={}
                ),
                "response_time": 0.25
            }
        )
        
        await monitor._on_signal_generated(event)
        
        tracker = monitor.trackers["test_agent"]
        assert tracker.total_signals == 1
        assert len(tracker.recent_signals) == 1
    
    @pytest.mark.asyncio
    async def test_on_position_closed(self, monitor, mock_agent):
        """Test handling position closure events"""
        monitor.register_agent(mock_agent)
        
        event = Event(
            type="position.closed",
            data={
                "agent_id": "test_agent",
                "signal_id": "sig_001",
                "symbol": "AAPL",
                "pnl": 150.0,
                "return": 0.05,
                "entry_price": 100.0,
                "exit_price": 105.0
            }
        )
        
        await monitor._on_position_closed(event)
        
        tracker = monitor.trackers["test_agent"]
        assert tracker.successful_signals == 1
        assert tracker.cumulative_pnl == 150.0
    
    @pytest.mark.asyncio
    async def test_check_performance_thresholds(self, monitor, mock_agent):
        """Test performance threshold checking"""
        monitor.register_agent(mock_agent)
        
        # Create metrics that violate thresholds
        metrics = AgentMetrics(
            agent_id="test_agent",
            agent_type="test",
            accuracy=0.45,  # Below threshold
            sharpe_ratio=0.3,  # Below threshold
            max_drawdown=0.25,  # Above threshold
            sample_size=20,
            overconfidence_ratio=2.0  # Overconfident
        )
        
        alerts = []
        
        async def capture_alert(alert):
            alerts.append(alert)
        
        monitor.add_alert_handler(capture_alert)
        
        await monitor._check_performance_thresholds("test_agent", metrics)
        
        # Should have generated alerts
        assert len(alerts) > 0
        assert any("Low accuracy" in alert["message"] for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_predict_performance(self, monitor):
        """Test performance prediction"""
        metrics = AgentMetrics(
            agent_id="test_agent",
            agent_type="test",
            performance_trend="declining",
            max_drawdown=0.18,
            overconfidence_ratio=2.5,
            recent_performance=0.3
        )
        
        prediction = await monitor._predict_performance("test_agent", metrics)
        
        assert "risk" in prediction
        assert "expected_performance" in prediction
        assert "confidence" in prediction
        assert "reason" in prediction
        
        # Should identify high risk
        assert prediction["risk"] == "high"
    
    def test_get_agent_metrics(self, monitor, mock_agent):
        """Test getting metrics for specific agent"""
        monitor.register_agent(mock_agent)
        
        # Add some data to tracker
        tracker = monitor.trackers["test_agent"]
        for i in range(5):
            outcome = SignalOutcome(
                signal_id=f"sig_{i}",
                agent_id="test_agent",
                symbol="AAPL",
                signal_type="buy",
                confidence=0.7,
                timestamp=datetime.now(),
                outcome="success",
                pnl=50.0
            )
            tracker.record_outcome(outcome)
        
        metrics = monitor.get_agent_metrics("test_agent")
        
        assert metrics is not None
        assert metrics.agent_id == "test_agent"
        assert metrics.total_pnl == 250.0
    
    def test_get_all_metrics(self, monitor):
        """Test getting metrics for all agents"""
        # Register multiple agents
        for i in range(3):
            agent = Mock()
            agent.config = AgentConfig(
                name=f"agent_{i}",
                description=f"Test agent {i}"
            )
            monitor.register_agent(agent)
        
        all_metrics = monitor.get_all_metrics()
        
        assert len(all_metrics) == 3
        assert "agent_0" in all_metrics
        assert "agent_1" in all_metrics
        assert "agent_2" in all_metrics
    
    @pytest.mark.asyncio
    async def test_store_metrics(self, monitor):
        """Test storing metrics to database"""
        metrics = AgentMetrics(
            agent_id="test_agent",
            agent_type="test",
            accuracy=0.65,
            win_rate=0.6,
            sharpe_ratio=1.2,
            total_pnl=500.0
        )
        
        with patch('database.connection.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            
            await monitor._store_metrics("test_agent", metrics)
            
            # Should have added record and committed
            assert mock_session.add.called
            assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_publish_metrics(self, monitor, mock_agent):
        """Test publishing aggregated metrics"""
        monitor.register_agent(mock_agent)
        
        with patch('core.events.bus.event_bus.publish', new_callable=AsyncMock) as mock_publish:
            await monitor._publish_metrics()
            
            # Should have published metrics event
            mock_publish.assert_called_once()
            call_args = mock_publish.call_args
            assert call_args[0][0] == "performance.metrics"
            assert "test_agent" in call_args[1]["data"]


class TestPerformanceIntegration:
    """Integration tests for performance monitoring"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self):
        """Test complete monitoring flow"""
        monitor = AgentPerformanceMonitor()
        
        # Create and register agent
        agent = Mock()
        agent.config = AgentConfig(
            name="integration_agent",
            description="Integration test agent"
        )
        monitor.register_agent(agent)
        
        # Simulate signal generation and outcomes
        tracker = monitor.trackers["integration_agent"]
        
        # Generate signals
        for i in range(20):
            signal = Signal(
                symbol="AAPL",
                action="buy" if i % 2 == 0 else "sell",
                confidence=0.5 + (i * 0.02),
                metadata={"iteration": i}
            )
            tracker.record_signal(signal, response_time=0.1)
            
            # Simulate outcome after some time
            outcome = SignalOutcome(
                signal_id=f"sig_{i}",
                agent_id="integration_agent",
                symbol="AAPL",
                signal_type=signal.action,
                confidence=signal.confidence,
                timestamp=datetime.now(),
                outcome="success" if np.random.random() > 0.4 else "failure",
                pnl=np.random.normal(50, 30)
            )
            tracker.record_outcome(outcome)
        
        # Get final metrics
        metrics = monitor.get_agent_metrics("integration_agent")
        
        # Verify metrics are calculated
        assert metrics.total_signals == 20
        assert metrics.winning_signals + metrics.losing_signals == 20
        assert 0 <= metrics.win_rate <= 1
        assert metrics.avg_confidence > 0
        assert metrics.sample_size == 20
        
        # Test performance trend detection
        assert metrics.performance_trend in ["improving", "declining", "stable"]
    
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test alert generation for poor performance"""
        monitor = AgentPerformanceMonitor()
        alerts_received = []
        
        async def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_handler(alert_handler)
        
        # Create agent with poor performance
        agent = Mock()
        agent.config = AgentConfig(
            name="poor_agent",
            description="Poorly performing agent"
        )
        monitor.register_agent(agent)
        
        tracker = monitor.trackers["poor_agent"]
        
        # Generate mostly losing trades
        for i in range(15):
            outcome = SignalOutcome(
                signal_id=f"sig_{i}",
                agent_id="poor_agent",
                symbol="AAPL",
                signal_type="buy",
                confidence=0.8,  # High confidence
                timestamp=datetime.now(),
                outcome="failure" if i > 3 else "success",  # Only 4 wins
                pnl=-100 if i > 3 else 50
            )
            tracker.record_outcome(outcome)
        
        # Check thresholds
        metrics = tracker.get_metrics()
        await monitor._check_performance_thresholds("poor_agent", metrics)
        
        # Should have generated alerts
        assert len(alerts_received) > 0
        
        # Check alert content
        alert = alerts_received[0]
        assert alert["agent_id"] == "poor_agent"
        assert alert["type"] == "Performance Alert"
        assert "Low accuracy" in alert["message"] or "Overconfident" in alert["message"]