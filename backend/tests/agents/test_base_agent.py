"""
Test cases for BaseAgent functionality
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from agents.base import BaseAgent, AgentConfig, Signal, SignalAction, SignalStrength


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.analyze_mock = AsyncMock()
        
    async def analyze(self, market_data):
        return await self.analyze_mock(market_data)
    
    def get_required_data_types(self):
        return ['price', 'volume']


@pytest.fixture
def agent_config():
    """Basic agent configuration"""
    return AgentConfig(
        name="TestAgent",
        version="1.0.0",
        enabled=True,
        weight=1.0,
        confidence_threshold=0.7,
        timeout=10,
        max_retries=3
    )


@pytest.fixture
def test_agent(agent_config):
    """Test agent instance"""
    return TestAgent(agent_config)


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'symbol': 'AAPL',
        'price': 150.0,
        'volume': 1000000,
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_signal():
    """Sample signal for testing"""
    return Signal(
        symbol="AAPL",
        action=SignalAction.BUY,
        confidence=0.8,
        strength=SignalStrength.MODERATE,
        source="TestAgent",
        current_price=150.0,
        reasoning=["Test signal"]
    )


class TestBaseAgent:
    """Test BaseAgent functionality"""
    
    def test_agent_initialization(self, agent_config):
        """Test agent initialization"""
        agent = TestAgent(agent_config)
        
        assert agent.config.name == "TestAgent"
        assert agent.config.enabled is True
        assert agent.performance.total_signals == 0
        assert not agent.is_running
    
    def test_validate_market_data_valid(self, test_agent, sample_market_data):
        """Test market data validation with valid data"""
        assert test_agent._validate_market_data(sample_market_data) is True
    
    def test_validate_market_data_missing_symbol(self, test_agent):
        """Test market data validation with missing symbol"""
        data = {'price': 150.0, 'volume': 1000000}
        assert test_agent._validate_market_data(data) is False
    
    def test_validate_market_data_missing_required_fields(self, test_agent):
        """Test market data validation with missing required fields"""
        data = {'symbol': 'AAPL'}  # Missing price and volume
        assert test_agent._validate_market_data(data) is False
    
    @pytest.mark.asyncio
    async def test_execute_with_monitoring_success(self, test_agent, sample_market_data, sample_signal):
        """Test successful execution with monitoring"""
        test_agent.analyze_mock.return_value = sample_signal
        
        result = await test_agent.execute_with_monitoring(sample_market_data)
        
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.action == SignalAction.BUY
        assert test_agent.performance.total_signals == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_monitoring_disabled(self, agent_config, sample_market_data):
        """Test execution with disabled agent"""
        agent_config.enabled = False
        agent = TestAgent(agent_config)
        
        result = await agent.execute_with_monitoring(sample_market_data)
        
        assert result is None
        assert agent.performance.total_signals == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_monitoring_low_confidence(self, test_agent, sample_market_data, sample_signal):
        """Test execution with low confidence signal"""
        sample_signal.confidence = 0.5  # Below threshold
        test_agent.analyze_mock.return_value = sample_signal
        
        result = await test_agent.execute_with_monitoring(sample_market_data)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_with_monitoring_timeout(self, test_agent, sample_market_data):
        """Test execution timeout handling"""
        test_agent.analyze_mock.side_effect = asyncio.TimeoutError()
        test_agent.config.timeout = 0.1
        test_agent.config.max_retries = 1
        
        result = await test_agent.execute_with_monitoring(sample_market_data)
        
        assert result is None
    
    @pytest.mark.asyncio 
    async def test_execute_with_monitoring_exception(self, test_agent, sample_market_data):
        """Test execution exception handling"""
        test_agent.analyze_mock.side_effect = ValueError("Test error")
        test_agent.config.max_retries = 1
        
        result = await test_agent.execute_with_monitoring(sample_market_data)
        
        assert result is None
    
    def test_update_performance_feedback_correct(self, test_agent):
        """Test performance feedback for correct signal"""
        test_agent.performance.total_signals = 10
        
        test_agent.update_performance_feedback("test_id", True, 100.0)
        
        assert test_agent.performance.correct_signals == 1
        assert test_agent.performance.accuracy == 0.1  # 1/10
    
    def test_update_performance_feedback_incorrect(self, test_agent):
        """Test performance feedback for incorrect signal"""
        test_agent.performance.total_signals = 10
        
        test_agent.update_performance_feedback("test_id", False, -50.0)
        
        assert test_agent.performance.correct_signals == 0
        assert test_agent.performance.accuracy == 0.0
    
    def test_adaptive_learning_poor_performance(self, test_agent):
        """Test adaptive learning with poor performance"""
        test_agent.performance.total_signals = 20
        test_agent.performance.correct_signals = 5  # 25% accuracy
        original_threshold = test_agent.config.confidence_threshold
        
        test_agent.update_performance_feedback("test_id", False)
        
        # Threshold should increase
        assert test_agent.config.confidence_threshold > original_threshold
    
    def test_adaptive_learning_good_performance(self, test_agent):
        """Test adaptive learning with good performance"""
        test_agent.performance.total_signals = 20
        test_agent.performance.correct_signals = 15  # 75% accuracy
        original_threshold = test_agent.config.confidence_threshold
        
        test_agent.update_performance_feedback("test_id", True)
        
        # Threshold should decrease slightly
        assert test_agent.config.confidence_threshold < original_threshold
    
    def test_adjust_weight(self, test_agent):
        """Test weight adjustment"""
        test_agent.adjust_weight(0.5)
        assert test_agent.config.weight == 0.5
        
        # Test bounds
        test_agent.adjust_weight(-0.1)
        assert test_agent.config.weight == 0.0
        
        test_agent.adjust_weight(1.5)
        assert test_agent.config.weight == 1.0
    
    def test_get_current_performance(self, test_agent):
        """Test performance metrics retrieval"""
        performance = test_agent.get_current_performance()
        
        assert 'agent_id' in performance
        assert 'name' in performance
        assert 'accuracy' in performance
        assert 'total_signals' in performance
        assert performance['name'] == "TestAgent"
    
    @pytest.mark.asyncio
    async def test_lifecycle_methods(self, test_agent):
        """Test agent lifecycle methods"""
        assert not test_agent.is_running
        
        await test_agent.start()
        assert test_agent.is_running
        
        await test_agent.stop()
        assert not test_agent.is_running
        
        await test_agent.shutdown()
        assert not test_agent.is_running
    
    def test_repr(self, test_agent):
        """Test string representation"""
        repr_str = repr(test_agent)
        assert "TestAgent" in repr_str
        assert "name=TestAgent" in repr_str
        assert "weight=1.00" in repr_str