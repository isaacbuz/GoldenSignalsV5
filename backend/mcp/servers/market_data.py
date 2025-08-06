"""
Market Data MCP Server
Provides standardized access to market data through MCP
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import random

from mcp.servers.base import BaseMCPServer, MCPTool, MCPResource, ToolType
from services.market_data_unified import unified_market_service as MarketDataService
from core.logging import get_logger

logger = get_logger(__name__)


class MarketDataMCPServer(BaseMCPServer):
    """
    MCP server for market data access
    
    Provides tools for:
    - Real-time quotes
    - Historical data
    - Market analysis
    - Technical indicators
    - Market summaries
    """
    
    def __init__(self):
        super().__init__("GoldenSignals-MarketData", "1.0.0")
        self.market_service = MarketDataService()
        
        # Popular symbols for examples
        self.popular_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ',
            'AMZN', 'META', 'NFLX', 'AMD', 'BTC-USD', 'ETH-USD'
        ]
        
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        """Register all available tools"""
        
        # Get Quote Tool
        self.register_tool(MCPTool(
            name="get_quote",
            description="Get real-time quote for a stock or crypto symbol",
            tool_type=ToolType.QUERY,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock or crypto symbol (e.g., AAPL, BTC-USD)"
                    }
                },
                "required": ["symbol"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "price": {"type": "number"},
                    "change": {"type": "number"},
                    "changePercent": {"type": "number"},
                    "volume": {"type": "integer"},
                    "timestamp": {"type": "string"}
                }
            },
            examples=[
                {"input": {"symbol": "AAPL"}, "output": {"symbol": "AAPL", "price": 195.89}},
                {"input": {"symbol": "BTC-USD"}, "output": {"symbol": "BTC-USD", "price": 45230.50}}
            ],
            rate_limit=120  # 120 requests per minute
        ))
        
        # Get Historical Data Tool
        self.register_tool(MCPTool(
            name="get_historical_data",
            description="Get historical OHLCV data for analysis",
            tool_type=ToolType.QUERY,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {
                        "type": "string",
                        "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)",
                        "default": "1mo"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk)",
                        "default": "1d"
                    }
                },
                "required": ["symbol"]
            },
            rate_limit=60
        ))
        
        # Get Technical Indicators Tool
        self.register_tool(MCPTool(
            name="get_technical_indicators",
            description="Calculate technical indicators for a symbol",
            tool_type=ToolType.ANALYSIS,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of indicators (sma, ema, rsi, macd, bb)",
                        "default": ["sma_20", "rsi", "macd"]
                    }
                },
                "required": ["symbol"]
            },
            rate_limit=60
        ))
        
        # Get Market Summary Tool
        self.register_tool(MCPTool(
            name="get_market_summary",
            description="Get market overview with indices and top movers",
            tool_type=ToolType.QUERY,
            input_schema={
                "type": "object",
                "properties": {
                    "include_sectors": {
                        "type": "boolean",
                        "description": "Include sector performance",
                        "default": True
                    }
                }
            },
            rate_limit=30
        ))
        
        # Compare Symbols Tool
        self.register_tool(MCPTool(
            name="compare_symbols",
            description="Compare multiple symbols side by side",
            tool_type=ToolType.ANALYSIS,
            input_schema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of symbols to compare (max 10)"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to compare",
                        "default": ["price", "change_percent", "volume", "market_cap"]
                    }
                },
                "required": ["symbols"]
            },
            rate_limit=30
        ))
        
        # Search Symbols Tool
        self.register_tool(MCPTool(
            name="search_symbols",
            description="Search for symbols by name or ticker",
            tool_type=ToolType.QUERY,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
            rate_limit=60
        ))
    
    def _register_resources(self):
        """Register available resources"""
        
        # Market indices resource
        self.register_resource(MCPResource(
            uri="market://indices",
            name="Market Indices",
            description="Real-time data for major market indices",
            metadata={
                "update_frequency": "1 minute",
                "indices": ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000"]
            }
        ))
        
        # Popular stocks resource
        self.register_resource(MCPResource(
            uri="market://popular-stocks",
            name="Popular Stocks",
            description="Most actively traded stocks",
            metadata={
                "update_frequency": "5 minutes",
                "count": 50
            }
        ))
        
        # Crypto markets resource
        self.register_resource(MCPResource(
            uri="market://crypto",
            name="Cryptocurrency Markets",
            description="Top cryptocurrencies by market cap",
            metadata={
                "update_frequency": "1 minute",
                "currencies": ["BTC", "ETH", "SOL", "BNB"]
            }
        ))
    
    async def initialize(self) -> None:
        """Initialize the MCP server"""
        logger.info("Initializing Market Data MCP Server...")
        # Market service is already initialized in __init__
        logger.info("Market Data MCP Server initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the MCP server"""
        logger.info("Shutting down Market Data MCP Server...")
        # Cleanup if needed
        logger.info("Market Data MCP Server shutdown complete")
    
    async def _execute_tool_logic(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool-specific logic"""
        
        if tool_name == "get_quote":
            return await self._get_quote(arguments["symbol"])
            
        elif tool_name == "get_historical_data":
            return await self._get_historical_data(
                arguments["symbol"],
                arguments.get("period", "1mo"),
                arguments.get("interval", "1d")
            )
            
        elif tool_name == "get_technical_indicators":
            return await self._get_technical_indicators(
                arguments["symbol"],
                arguments.get("indicators", ["sma_20", "rsi", "macd"])
            )
            
        elif tool_name == "get_market_summary":
            return await self._get_market_summary(
                arguments.get("include_sectors", True)
            )
            
        elif tool_name == "compare_symbols":
            return await self._compare_symbols(
                arguments["symbols"],
                arguments.get("metrics", ["price", "change_percent", "volume", "market_cap"])
            )
            
        elif tool_name == "search_symbols":
            return await self._search_symbols(
                arguments["query"],
                arguments.get("limit", 10)
            )
            
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote"""
        quote = await self.market_service.get_quote(symbol)
        
        if not quote:
            return {
                "error": f"No data found for symbol: {symbol}",
                "symbol": symbol
            }
        
        return {
            "success": True,
            "data": quote,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_historical_data(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        """Get historical data"""
        data = await self.market_service.get_historical_data(symbol, period, interval)
        
        if not data:
            return {
                "error": f"No historical data found for {symbol}",
                "symbol": symbol,
                "period": period,
                "interval": interval
            }
        
        # Calculate statistics
        if data:
            prices = [d.get("close", 0) for d in data]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
            
            statistics = {
                "count": len(data),
                "high": max(prices) if prices else 0,
                "low": min(prices) if prices else 0,
                "average": sum(prices) / len(prices) if prices else 0,
                "volatility": self._calculate_volatility(returns) if returns else 0,
                "total_return": ((prices[-1] / prices[0] - 1) * 100) if len(prices) > 1 and prices[0] > 0 else 0
            }
        else:
            statistics = {}
        
        return {
            "success": True,
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": data,
            "statistics": statistics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_technical_indicators(self, symbol: str, indicators: List[str]) -> Dict[str, Any]:
        """Get technical indicators"""
        result = await self.market_service.get_technical_indicators(symbol)
        
        if not result:
            return {
                "error": f"Failed to calculate indicators for {symbol}",
                "symbol": symbol
            }
        
        # Filter requested indicators
        filtered_indicators = {}
        for indicator in indicators:
            if indicator in result:
                filtered_indicators[indicator] = result[indicator]
        
        return {
            "success": True,
            "symbol": symbol,
            "indicators": filtered_indicators,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_market_summary(self, include_sectors: bool) -> Dict[str, Any]:
        """Get market summary"""
        
        # Get major indices
        indices = [
            {"symbol": "^GSPC", "name": "S&P 500"},
            {"symbol": "^DJI", "name": "Dow Jones"},
            {"symbol": "^IXIC", "name": "NASDAQ"},
            {"symbol": "^VIX", "name": "VIX"}
        ]
        
        index_data = []
        for index in indices:
            quote = await self.market_service.get_quote(index["symbol"])
            if quote:
                index_data.append({
                    "symbol": index["symbol"],
                    "name": index["name"],
                    "price": quote.get("price", 0),
                    "change": quote.get("change", 0),
                    "changePercent": quote.get("change_percent", 0)
                })
        
        # Get top movers (mock for now)
        top_gainers = [
            {"symbol": "NVDA", "changePercent": 5.2},
            {"symbol": "TSLA", "changePercent": 3.8},
            {"symbol": "AMD", "changePercent": 3.1}
        ]
        
        top_losers = [
            {"symbol": "INTC", "changePercent": -2.5},
            {"symbol": "DIS", "changePercent": -2.1},
            {"symbol": "BA", "changePercent": -1.8}
        ]
        
        summary = {
            "success": True,
            "indices": index_data,
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "market_status": "open" if datetime.utcnow().hour >= 9 and datetime.utcnow().hour < 16 else "closed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if include_sectors:
            summary["sectors"] = [
                {"name": "Technology", "change": 1.2},
                {"name": "Healthcare", "change": 0.8},
                {"name": "Finance", "change": -0.5},
                {"name": "Energy", "change": 2.1}
            ]
        
        return summary
    
    async def _compare_symbols(self, symbols: List[str], metrics: List[str]) -> Dict[str, Any]:
        """Compare multiple symbols"""
        if len(symbols) > 10:
            return {
                "error": "Maximum 10 symbols allowed for comparison",
                "requested": len(symbols)
            }
        
        comparison_data = []
        
        for symbol in symbols:
            quote = await self.market_service.get_quote(symbol)
            if quote:
                symbol_data = {"symbol": symbol}
                
                # Add requested metrics
                metric_mapping = {
                    "price": "price",
                    "change": "change",
                    "change_percent": "change_percent",
                    "volume": "volume",
                    "market_cap": "market_cap",
                    "pe_ratio": "pe_ratio",
                    "dividend_yield": "dividend_yield"
                }
                
                for metric in metrics:
                    if metric in metric_mapping:
                        symbol_data[metric] = quote.get(metric_mapping[metric], "N/A")
                
                comparison_data.append(symbol_data)
        
        return {
            "success": True,
            "symbols": symbols,
            "metrics": metrics,
            "data": comparison_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _search_symbols(self, query: str, limit: int) -> Dict[str, Any]:
        """Search for symbols"""
        # Mock search results for now
        # In production, integrate with a symbol search API
        
        results = []
        
        # Simple matching against known symbols
        all_symbols = [
            {"symbol": "AAPL", "name": "Apple Inc.", "type": "stock"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "type": "stock"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "stock"},
            {"symbol": "BTC-USD", "name": "Bitcoin USD", "type": "crypto"},
            {"symbol": "ETH-USD", "name": "Ethereum USD", "type": "crypto"}
        ]
        
        query_lower = query.lower()
        for item in all_symbols:
            if query_lower in item["symbol"].lower() or query_lower in item["name"].lower():
                results.append(item)
                if len(results) >= limit:
                    break
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_resource_content(self, uri: str) -> Any:
        """Get resource content"""
        
        if uri == "market://indices":
            # Return current index data
            indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
            data = []
            for symbol in indices:
                quote = await self.market_service.get_quote(symbol)
                if quote:
                    data.append(quote)
            return data
            
        elif uri == "market://popular-stocks":
            # Return popular stocks
            data = []
            for symbol in self.popular_symbols[:10]:
                quote = await self.market_service.get_quote(symbol)
                if quote:
                    data.append(quote)
            return data
            
        elif uri == "market://crypto":
            # Return crypto data
            cryptos = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
            data = []
            for symbol in cryptos:
                quote = await self.market_service.get_quote(symbol)
                if quote:
                    data.append(quote)
            return data
            
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate volatility from returns"""
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return (variance ** 0.5) * 100  # Return as percentage