"""
Example Agent Plugin
Demonstrates how to create a plugin for the system
"""

from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

from core.plugins.base import (
    AgentPlugin,
    PluginMetadata,
    PluginType,
    PluginContext
)
from core.events.bus import event_bus, EventTypes
from core.logging import get_logger

logger = get_logger(__name__)


class ExampleTradingAgentPlugin(AgentPlugin):
    """
    Example trading agent implemented as a plugin
    Shows how to create modular, extensible agents
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        return PluginMetadata(
            name="ExampleTradingAgent",
            version="1.0.0",
            type=PluginType.AGENT,
            author="GoldenSignalsAI",
            description="Example agent plugin demonstrating plugin architecture",
            dependencies=[],  # List other plugins this depends on
            tags=["example", "momentum", "trading"]
        )
    
    async def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin"""
        self._context = context
        
        # Get configuration
        self.threshold = context.config.settings.get("signal_threshold", 0.7)
        self.lookback = context.config.settings.get("lookback_period", 20)
        
        # Initialize agent state
        self.agent = None
        self._price_history = []
        
        # Subscribe to events
        await context.event_bus.subscribe(
            EventTypes.MARKET_DATA_RECEIVED,
            self._on_market_data
        )
        
        logger.info(f"Initialized {self.metadata.name} plugin")
    
    async def start(self) -> None:
        """Start the plugin"""
        # Create agent instance
        self.agent = self._create_agent()
        
        # Start background tasks if needed
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_markets())
        
        logger.info(f"Started {self.metadata.name}")
    
    async def stop(self) -> None:
        """Stop the plugin"""
        self._running = False
        
        if hasattr(self, '_monitor_task'):
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped {self.metadata.name}")
    
    async def shutdown(self) -> None:
        """Cleanup resources"""
        # Unsubscribe from events
        await self._context.event_bus.unsubscribe(
            EventTypes.MARKET_DATA_RECEIVED,
            self._on_market_data
        )
        
        # Clear state
        self._price_history.clear()
        self.agent = None
        
        logger.info(f"Shutdown {self.metadata.name}")
    
    async def get_agent(self) -> Any:
        """Get the agent instance"""
        return self.agent
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis on market data
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Analysis results with signals
        """
        try:
            # Add to price history
            if 'close' in market_data:
                self._price_history.append(market_data['close'])
                if len(self._price_history) > self.lookback:
                    self._price_history.pop(0)
            
            # Simple momentum strategy
            signal = await self._calculate_momentum_signal()
            
            # Generate result
            result = {
                "signal": signal['action'],
                "confidence": signal['confidence'],
                "analysis": {
                    "momentum": signal['momentum'],
                    "trend": signal['trend']
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish signal event if strong enough
            if signal['confidence'] >= self.threshold:
                await self._context.event_bus.publish(
                    EventTypes.SIGNAL_GENERATED,
                    data={
                        "source": self.metadata.name,
                        "signal": result
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _calculate_momentum_signal(self) -> Dict[str, Any]:
        """Calculate momentum-based signal"""
        if len(self._price_history) < self.lookback:
            return {
                "action": "NEUTRAL",
                "confidence": 0.0,
                "momentum": 0.0,
                "trend": "insufficient_data"
            }
        
        # Calculate simple momentum
        current_price = self._price_history[-1]
        past_price = self._price_history[0]
        momentum = (current_price - past_price) / past_price
        
        # Calculate trend strength
        import numpy as np
        prices = np.array(self._price_history)
        trend = np.polyfit(range(len(prices)), prices, 1)[0]
        
        # Generate signal
        if momentum > 0.05 and trend > 0:
            action = "BUY"
            confidence = min(abs(momentum) * 10, 1.0)
        elif momentum < -0.05 and trend < 0:
            action = "SELL"
            confidence = min(abs(momentum) * 10, 1.0)
        else:
            action = "NEUTRAL"
            confidence = 0.3
        
        return {
            "action": action,
            "confidence": confidence,
            "momentum": momentum,
            "trend": "bullish" if trend > 0 else "bearish"
        }
    
    async def _on_market_data(self, event) -> None:
        """Handle market data events"""
        try:
            market_data = event.data
            
            # Analyze if we have data for our symbols
            if self._should_analyze(market_data):
                await self.analyze(market_data)
                
        except Exception as e:
            logger.error(f"Error handling market data: {str(e)}")
    
    def _should_analyze(self, market_data: Dict[str, Any]) -> bool:
        """Check if we should analyze this data"""
        # Add logic to filter relevant data
        symbols = self._context.config.settings.get("symbols", [])
        if not symbols:
            return True
        
        data_symbol = market_data.get("symbol")
        return data_symbol in symbols
    
    async def _monitor_markets(self) -> None:
        """Background task to monitor markets"""
        while self._running:
            try:
                # Perform periodic checks
                await asyncio.sleep(60)  # Check every minute
                
                # Could fetch data, check conditions, etc.
                logger.debug(f"{self.metadata.name} monitoring cycle")
                
            except Exception as e:
                logger.error(f"Monitor error: {str(e)}")
    
    def _create_agent(self) -> Any:
        """Create the actual agent instance"""
        # This would create your actual agent
        # For now, return a mock object
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.active = True
        
        return MockAgent(self.metadata.name)
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check"""
        base_health = await super().health_check()
        
        # Add custom health metrics
        base_health.update({
            "price_history_size": len(self._price_history),
            "agent_active": self.agent is not None,
            "last_analysis": self._context.get_state("last_analysis_time")
        })
        
        return base_health


class ExampleIndicatorPlugin(AgentPlugin):
    """
    Example technical indicator plugin
    Shows how to create modular indicators
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="CustomRSI",
            version="1.0.0",
            type=PluginType.INDICATOR,
            author="GoldenSignalsAI",
            description="Custom RSI indicator with adaptive periods",
            dependencies=[],
            tags=["indicator", "rsi", "momentum"]
        )
    
    async def initialize(self, context: PluginContext) -> None:
        """Initialize the indicator"""
        self._context = context
        self.period = context.config.settings.get("period", 14)
        self.overbought = context.config.settings.get("overbought", 70)
        self.oversold = context.config.settings.get("oversold", 30)
        
        logger.info(f"Initialized {self.metadata.name}")
    
    async def start(self) -> None:
        """Start the indicator"""
        logger.info(f"Started {self.metadata.name}")
    
    async def stop(self) -> None:
        """Stop the indicator"""
        logger.info(f"Stopped {self.metadata.name}")
    
    async def shutdown(self) -> None:
        """Cleanup"""
        logger.info(f"Shutdown {self.metadata.name}")
    
    async def get_agent(self) -> Any:
        """Get indicator instance"""
        return self
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate RSI"""
        prices = market_data.get("prices", [])
        
        if len(prices) < self.period:
            return {"rsi": 50, "signal": "neutral"}
        
        # Calculate RSI (simplified)
        import numpy as np
        prices = np.array(prices[-self.period:])
        deltas = np.diff(prices)
        gains = deltas[deltas > 0].sum() / self.period
        losses = -deltas[deltas < 0].sum() / self.period
        
        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        
        # Generate signal
        if rsi > self.overbought:
            signal = "overbought"
        elif rsi < self.oversold:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {
            "rsi": rsi,
            "signal": signal,
            "overbought": self.overbought,
            "oversold": self.oversold
        }