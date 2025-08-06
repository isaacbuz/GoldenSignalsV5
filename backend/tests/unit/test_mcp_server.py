"""
Unit Tests for MCP Server
Comprehensive testing of Model Context Protocol server functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from services.mcp.mcp_server import (
    MCPServer, MCPMessage, MCPResponse, MCPTool,
    MarketDataTool, TradingSignalTool, PortfolioAnalysisTool
)


class TestMCPMessage:
    """Test MCP message structure"""
    
    def test_mcp_message_creation(self):
        """Test MCP message creation"""
        msg = MCPMessage(
            id="test_id",
            method="test_method",
            params={"key": "value"},
            timestamp=datetime.now()
        )
        
        assert msg.id == "test_id"
        assert msg.method == "test_method"
        assert msg.params["key"] == "value"
        assert isinstance(msg.timestamp, datetime)


class TestMCPResponse:
    """Test MCP response structure"""
    
    def test_mcp_response_success(self):
        """Test successful MCP response"""
        response = MCPResponse(
            id="test_id",
            result={"data": "success"},
            timestamp=datetime.now()
        )
        
        assert response.id == "test_id"
        assert response.result["data"] == "success"
        assert response.error is None
    
    def test_mcp_response_error(self):
        """Test error MCP response"""
        response = MCPResponse(
            id="test_id",
            result=None,
            error="Test error",
            timestamp=datetime.now()
        )
        
        assert response.error == "Test error"
        assert response.result is None


class TestMarketDataTool:
    """Test MarketDataTool functionality"""
    
    @pytest.fixture
    def market_tool(self):
        """Create market data tool instance"""
        return MarketDataTool()
    
    def test_tool_properties(self, market_tool):
        """Test tool name and description"""
        assert market_tool.name == "get_market_data"
        assert "real-time market data" in market_tool.description.lower()
    
    @pytest.mark.asyncio
    async def test_execute_with_real_data(self, market_tool):
        """Test execute with mocked real data provider"""
        with patch('services.mcp.mcp_server.LiveDataProvider') as mock_provider:
            # Mock provider instance
            mock_instance = Mock()
            mock_instance.get_current_price = AsyncMock(return_value={
                "price": 150.25,
                "change": 2.5,
                "change_percent": 1.69,
                "volume": 1000000
            })
            mock_provider.return_value = mock_instance
            
            result = await market_tool.execute({
                "symbol": "AAPL",
                "timeframe": "1d"
            })
            
            assert result["symbol"] == "AAPL"
            assert result["price"] == 150.25
            assert result["status"] == "success"
            assert result["timeframe"] == "1d"
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, market_tool):
        """Test execute with fallback data"""
        with patch('services.mcp.mcp_server.LiveDataProvider') as mock_provider:
            # Mock provider that returns None
            mock_instance = Mock()
            mock_instance.get_current_price = AsyncMock(return_value=None)
            mock_provider.return_value = mock_instance
            
            result = await market_tool.execute({
                "symbol": "AAPL",
                "timeframe": "1h"
            })
            
            assert result["symbol"] == "AAPL"
            assert result["status"] == "mock_data"
            assert result["price"] == 150.25  # Fallback price
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, market_tool):
        """Test execute with error handling"""
        with patch('services.mcp.mcp_server.LiveDataProvider', side_effect=Exception("API Error")):
            
            result = await market_tool.execute({
                "symbol": "AAPL"
            })
            
            assert result["symbol"] == "AAPL"
            assert result["status"] == "error_fallback"
            assert "error" in result
            assert result["error"] == "API Error"


class TestTradingSignalTool:
    """Test TradingSignalTool functionality"""
    
    @pytest.fixture
    def signal_tool(self):
        """Create trading signal tool instance"""
        return TradingSignalTool()
    
    def test_tool_properties(self, signal_tool):
        """Test tool name and description"""
        assert signal_tool.name == "generate_signal"
        assert "ai-powered trading signals" in signal_tool.description.lower()
    
    @pytest.mark.asyncio
    async def test_execute_with_real_signal(self, signal_tool):
        """Test execute with mocked signal generator"""
        with patch('services.mcp.mcp_server.AISignalGenerator') as mock_generator:
            # Mock generator instance
            mock_instance = Mock()
            mock_instance.generate_signal = AsyncMock(return_value={
                "action": "BUY",
                "confidence": 0.85,
                "price": 150.0,
                "stop_loss": 145.0,
                "target_price": 160.0,
                "reasoning": "Strong bullish momentum",
                "risk_level": "MEDIUM"
            })
            mock_generator.return_value = mock_instance
            
            result = await signal_tool.execute({
                "symbol": "AAPL",
                "strategy": "momentum",
                "timeframe": "1d"
            })
            
            assert result["symbol"] == "AAPL"
            assert result["signal"] == "BUY"
            assert result["confidence"] == 0.85
            assert result["status"] == "success"
            assert "reasoning" in result
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, signal_tool):
        """Test execute with fallback signal"""
        with patch('services.mcp.mcp_server.AISignalGenerator') as mock_generator:
            # Mock generator that returns None
            mock_instance = Mock()
            mock_instance.generate_signal = AsyncMock(return_value=None)
            mock_generator.return_value = mock_instance
            
            result = await signal_tool.execute({
                "symbol": "AAPL",
                "strategy": "momentum"
            })
            
            assert result["symbol"] == "AAPL"
            assert result["signal"] == "HOLD"  # Fallback
            assert result["status"] == "fallback"
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, signal_tool):
        """Test execute with error handling"""
        with patch('services.mcp.mcp_server.AISignalGenerator', side_effect=Exception("Generator Error")):
            
            result = await signal_tool.execute({
                "symbol": "AAPL",
                "strategy": "momentum"
            })
            
            assert result["symbol"] == "AAPL"
            assert result["signal"] == "HOLD"
            assert result["status"] == "error"
            assert result["confidence"] == 0.0
            assert "error" in result


class TestPortfolioAnalysisTool:
    """Test PortfolioAnalysisTool functionality"""
    
    @pytest.fixture
    def portfolio_tool(self):
        """Create portfolio analysis tool instance"""
        return PortfolioAnalysisTool()
    
    def test_tool_properties(self, portfolio_tool):
        """Test tool name and description"""
        assert portfolio_tool.name == "analyze_portfolio"
        assert "portfolio performance" in portfolio_tool.description.lower()
    
    @pytest.mark.asyncio
    async def test_execute_with_positions(self, portfolio_tool):
        """Test execute with position list"""
        with patch('services.mcp.mcp_server.PortfolioOptimizer') as mock_optimizer:
            # Mock optimizer instance
            mock_instance = Mock()
            mock_instance.analyze_portfolio = AsyncMock(return_value={
                "total_value": 100000,
                "daily_pnl": 2500,
                "volatility": 0.15,
                "sharpe_ratio": 0.8,
                "max_drawdown": 5.0
            })
            mock_optimizer.return_value = mock_instance
            
            positions = [
                {"symbol": "AAPL", "quantity": 100, "price": 150.0},
                {"symbol": "TSLA", "quantity": 50, "price": 800.0}
            ]
            
            result = await portfolio_tool.execute({
                "positions": positions
            })
            
            assert result["total_value"] == 100000
            assert result["daily_pnl"] == 2500
            assert result["status"] == "success"
            assert isinstance(result["recommendations"], list)
    
    @pytest.mark.asyncio
    async def test_execute_with_user_id(self, portfolio_tool):
        """Test execute with user ID"""
        with patch('services.mcp.mcp_server.get_session') as mock_session, \
             patch('services.mcp.mcp_server.Portfolio') as mock_portfolio_model:
            
            # Mock database session
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            # Mock portfolio query
            mock_portfolio = Mock()
            mock_portfolio.total_value = 50000
            mock_portfolio.daily_pnl = 1250
            mock_portfolio.volatility = 0.2
            mock_portfolio.sharpe_ratio = 0.6
            mock_portfolio.max_drawdown = 10.0
            mock_portfolio.win_rate = 65.5
            mock_portfolio.total_trades = 25
            
            mock_query = Mock()
            mock_query.filter.return_value.first = AsyncMock(return_value=mock_portfolio)
            mock_session_instance.query.return_value = mock_query
            
            result = await portfolio_tool.execute({
                "user_id": 123
            })
            
            assert result["total_value"] == 50000
            assert result["daily_pnl"] == 1250
            assert result["status"] == "success"
            assert result["win_rate"] == 65.5
    
    @pytest.mark.asyncio
    async def test_execute_empty_portfolio(self, portfolio_tool):
        """Test execute with empty portfolio"""
        result = await portfolio_tool.execute({})
        
        assert result["total_value"] == 0
        assert result["status"] == "empty_portfolio"
        assert "Start by adding positions" in result["recommendations"][0]
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, portfolio_tool):
        """Test execute with error handling"""
        with patch('services.mcp.mcp_server.PortfolioOptimizer', side_effect=Exception("Analysis Error")):
            
            result = await portfolio_tool.execute({
                "positions": [{"symbol": "AAPL"}]
            })
            
            assert result["status"] == "error"
            assert result["total_value"] == 0
            assert "error" in result


class TestMCPServer:
    """Test MCP Server functionality"""
    
    @pytest.fixture
    def mcp_server(self):
        """Create MCP server instance"""
        return MCPServer()
    
    def test_server_initialization(self, mcp_server):
        """Test server initialization"""
        assert len(mcp_server.tools) == 3  # 3 default tools
        assert "get_market_data" in mcp_server.tools
        assert "generate_signal" in mcp_server.tools
        assert "analyze_portfolio" in mcp_server.tools
    
    def test_register_tool(self, mcp_server):
        """Test tool registration"""
        class CustomTool(MCPTool):
            @property
            def name(self):
                return "custom_tool"
            
            @property  
            def description(self):
                return "Custom test tool"
            
            async def execute(self, params):
                return {"result": "custom"}
        
        custom_tool = CustomTool()
        mcp_server.register_tool(custom_tool)
        
        assert "custom_tool" in mcp_server.tools
        assert mcp_server.tools["custom_tool"] == custom_tool
    
    def test_register_handler(self, mcp_server):
        """Test custom handler registration"""
        async def custom_handler(params):
            return {"custom": "response"}
        
        mcp_server.register_handler("custom_method", custom_handler)
        
        assert "custom_method" in mcp_server.handlers
    
    @pytest.mark.asyncio
    async def test_handle_tool_execution(self, mcp_server):
        """Test tool execution handling"""
        # Mock tool execution
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(return_value={"price": 150.0})
        mcp_server.tools["test_tool"] = mock_tool
        
        message = MCPMessage(
            id="test_msg",
            method="tools/execute",
            params={"tool": "test_tool", "params": {"symbol": "AAPL"}},
            timestamp=datetime.now()
        )
        
        response = await mcp_server.handle_message(message)
        
        assert response.id == "test_msg"
        assert response.result["price"] == 150.0
        assert response.error is None
        mock_tool.execute.assert_called_once_with({"symbol": "AAPL"})
    
    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, mcp_server):
        """Test handling unknown tool"""
        message = MCPMessage(
            id="test_msg",
            method="tools/execute",
            params={"tool": "nonexistent_tool"},
            timestamp=datetime.now()
        )
        
        response = await mcp_server.handle_message(message)
        
        assert response.id == "test_msg"
        assert response.result is None
        assert "Unknown tool" in response.error
    
    @pytest.mark.asyncio
    async def test_handle_custom_method(self, mcp_server):
        """Test custom method handling"""
        async def custom_handler(params):
            return {"handled": True, "params": params}
        
        mcp_server.register_handler("custom_method", custom_handler)
        
        message = MCPMessage(
            id="test_msg",
            method="custom_method",
            params={"test": "data"},
            timestamp=datetime.now()
        )
        
        response = await mcp_server.handle_message(message)
        
        assert response.id == "test_msg"
        assert response.result["handled"] is True
        assert response.result["params"]["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_server):
        """Test tools listing"""
        message = MCPMessage(
            id="test_msg",
            method="tools/list",
            params={},
            timestamp=datetime.now()
        )
        
        response = await mcp_server.handle_message(message)
        
        assert response.id == "test_msg"
        assert len(response.result) == 3
        assert "get_market_data" in response.result
        assert response.result["get_market_data"]["name"] == "get_market_data"
    
    @pytest.mark.asyncio
    async def test_unknown_method(self, mcp_server):
        """Test unknown method handling"""
        message = MCPMessage(
            id="test_msg", 
            method="unknown_method",
            params={},
            timestamp=datetime.now()
        )
        
        response = await mcp_server.handle_message(message)
        
        assert response.id == "test_msg"
        assert response.result is None
        assert "Unknown method" in response.error


@pytest.mark.integration
class TestMCPServerIntegration:
    """Integration tests for MCP server"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete MCP workflow"""
        server = MCPServer()
        
        # Test market data -> signal generation -> portfolio analysis
        
        # 1. Get market data
        market_msg = MCPMessage(
            id="market_1",
            method="tools/execute", 
            params={"tool": "get_market_data", "params": {"symbol": "AAPL"}},
            timestamp=datetime.now()
        )
        
        market_response = await server.handle_message(market_msg)
        assert market_response.error is None
        
        # 2. Generate signal
        signal_msg = MCPMessage(
            id="signal_1",
            method="tools/execute",
            params={"tool": "generate_signal", "params": {"symbol": "AAPL", "strategy": "momentum"}},
            timestamp=datetime.now()
        )
        
        signal_response = await server.handle_message(signal_msg)
        assert signal_response.error is None
        
        # 3. Analyze portfolio
        portfolio_msg = MCPMessage(
            id="portfolio_1", 
            method="tools/execute",
            params={"tool": "analyze_portfolio", "params": {"positions": [{"symbol": "AAPL", "quantity": 100}]}},
            timestamp=datetime.now()
        )
        
        portfolio_response = await server.handle_message(portfolio_msg)
        assert portfolio_response.error is None
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation through MCP layers"""
        server = MCPServer()
        
        # Mock tool that raises exception
        mock_tool = Mock()
        mock_tool.execute = AsyncMock(side_effect=Exception("Tool error"))
        server.tools["error_tool"] = mock_tool
        
        message = MCPMessage(
            id="error_test",
            method="tools/execute",
            params={"tool": "error_tool"},
            timestamp=datetime.now()
        )
        
        response = await server.handle_message(message)
        assert "Tool error" in response.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])