"""
WebSocket Orchestrator Service
Integrates real-time WebSocket broadcasting with agent orchestration
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from services.websocket_manager import ws_manager, SignalUpdate
from agents.orchestrator import AgentOrchestrator
from agents.base import Signal, SignalAction
from services.market_data_service import MarketDataService
from services.signal_service import SignalService
from core.logging import get_logger
from core.database import get_db

logger = get_logger(__name__)


@dataclass
class AgentActivity:
    """Represents agent processing activity"""
    agent_id: str
    agent_name: str
    status: str  # processing, completed, failed
    signal: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[List[str]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class WebSocketOrchestrator:
    """
    Orchestrates WebSocket communication with agent processing
    Provides real-time updates on agent activities and signal generation
    """
    
    def __init__(self):
        self.agent_orchestrator = AgentOrchestrator()
        self.market_data_service = MarketDataService()
        self.signal_service = SignalService()
        
        # Track active analysis sessions
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info("WebSocket Orchestrator initialized")
    
    async def initialize(self):
        """Initialize the orchestrator and its dependencies"""
        # Initialize agent orchestrator with default agents
        await self.agent_orchestrator.initialize_default_agents()
        
        # Start background tasks
        await self.start()
        
        logger.info("WebSocket Orchestrator fully initialized")
    
    async def start(self):
        """Start background processing tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start market monitoring task
        self._tasks.append(
            asyncio.create_task(self._market_monitoring_loop())
        )
        
        # Start signal processing task
        self._tasks.append(
            asyncio.create_task(self._signal_processing_loop())
        )
        
        logger.info("WebSocket Orchestrator started")
    
    async def stop(self):
        """Stop all background tasks"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()
        
        logger.info("WebSocket Orchestrator stopped")
    
    async def analyze_symbol(self, symbol: str, client_id: Optional[str] = None):
        """
        Trigger real-time analysis for a symbol
        Broadcasts agent activities and results via WebSocket
        """
        analysis_id = f"{symbol}_{datetime.now().timestamp()}"
        
        try:
            # Mark analysis as active
            self.active_analyses[analysis_id] = {
                "symbol": symbol,
                "client_id": client_id,
                "started_at": datetime.now(),
                "status": "fetching_data"
            }
            
            # Broadcast analysis start
            await ws_manager.broadcast_agent_update(
                symbol=symbol,
                agent_name="Orchestrator",
                signal="ANALYZING",
                confidence=0.0
            )
            
            # Fetch market data
            market_data = await self._fetch_market_data(symbol)
            if not market_data:
                raise ValueError(f"Failed to fetch market data for {symbol}")
            
            # Update analysis status
            self.active_analyses[analysis_id]["status"] = "analyzing"
            
            # Create agent activity tracker
            agent_activities: Dict[str, AgentActivity] = {}
            
            # Hook into agent execution to broadcast updates
            async def agent_callback(agent_id: str, agent_name: str, status: str, result: Any = None):
                """Callback for agent activity updates"""
                activity = AgentActivity(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status=status
                )
                
                if status == "completed" and isinstance(result, Signal):
                    activity.signal = result.action.value
                    activity.confidence = result.confidence
                    activity.reasoning = result.reasoning[:3]  # First 3 reasons
                
                agent_activities[agent_id] = activity
                
                # Broadcast agent update
                await ws_manager.broadcast_agent_update(
                    symbol=symbol,
                    agent_name=agent_name,
                    signal=activity.signal or status.upper(),
                    confidence=activity.confidence or 0.0
                )
            
            # Execute analysis with callback
            signal = await self._analyze_with_callbacks(market_data, agent_callback)
            
            if signal:
                # Create comprehensive signal update
                signal_update = SignalUpdate(
                    symbol=symbol,
                    signal_id=signal.id if hasattr(signal, 'id') else analysis_id,
                    action=signal.action.value,
                    confidence=signal.confidence,
                    price=signal.current_price,
                    agents_consensus={
                        "total_agents": len(agent_activities),
                        "completed_agents": sum(1 for a in agent_activities.values() if a.status == "completed"),
                        "consensus_strength": signal.strength.value,
                        "agent_details": {
                            aid: {
                                "name": act.agent_name,
                                "signal": act.signal,
                                "confidence": act.confidence
                            }
                            for aid, act in agent_activities.items()
                            if act.status == "completed"
                        }
                    },
                    timestamp=datetime.now(),
                    metadata={
                        "reasoning": signal.reasoning[:5],
                        "indicators": signal.indicators,
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss
                    }
                )
                
                # Broadcast final signal
                await ws_manager.broadcast_signal(signal_update)
                
                # Store signal in database
                async for db in get_db():
                    await self.signal_service.create_signal(
                        db=db,
                        symbol=signal.symbol,
                        action=signal.action.value,
                        confidence=signal.confidence,
                        source="WebSocketOrchestrator",
                        metadata=asdict(signal_update)
                    )
                    break
                
                logger.info(f"Analysis completed for {symbol}: {signal.action.value} (confidence: {signal.confidence:.2f})")
            else:
                # Broadcast no signal result
                await ws_manager.broadcast_decision(
                    symbol=symbol,
                    decision={
                        "action": "NO_SIGNAL",
                        "reason": "Insufficient consensus among agents",
                        "agent_summary": {
                            aid: act.signal for aid, act in agent_activities.items()
                            if act.signal
                        }
                    }
                )
            
            # Mark analysis as complete
            self.active_analyses[analysis_id]["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            
            # Broadcast error
            await ws_manager.broadcast_decision(
                symbol=symbol,
                decision={
                    "action": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Mark analysis as failed
            if analysis_id in self.active_analyses:
                self.active_analyses[analysis_id]["status"] = "failed"
        
        finally:
            # Clean up after delay
            await asyncio.sleep(60)  # Keep record for 1 minute
            self.active_analyses.pop(analysis_id, None)
    
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive market data for analysis"""
        try:
            # Get current quote
            quote = await self.market_data_service.get_quote(symbol)
            if not quote:
                return None
            
            # Get historical data
            historical = await self.market_data_service.get_historical_data(
                symbol, period="1mo", interval="1d"
            )
            
            # Get intraday data
            intraday = await self.market_data_service.get_intraday_data(symbol)
            
            # Compile market data
            market_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": quote.get("price", 0),
                "volume": quote.get("volume", 0),
                "change": quote.get("change", 0),
                "change_percent": quote.get("change_percent", 0),
                "bid": quote.get("bid", 0),
                "ask": quote.get("ask", 0),
                "day_high": quote.get("day_high", 0),
                "day_low": quote.get("day_low", 0),
                "historical_prices": historical,
                "intraday_prices": intraday,
                "market_cap": quote.get("market_cap", 0),
                "pe_ratio": quote.get("pe_ratio", 0)
            }
            
            # Broadcast price update
            await ws_manager.broadcast_price_update(
                symbol=symbol,
                price=market_data["price"],
                volume=market_data["volume"]
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def _analyze_with_callbacks(self, market_data: Dict[str, Any], callback) -> Optional[Signal]:
        """
        Analyze market data with agent activity callbacks
        This wraps the orchestrator's analyze_market method to add real-time updates
        """
        # Get list of active agents
        active_agents = [
            agent for agent in self.agent_orchestrator.agents.values()
            if agent.config.enabled
        ]
        
        # Create tasks with callbacks
        tasks = []
        for agent in active_agents:
            # Notify processing start
            await callback(
                agent.agent_id,
                agent.config.name,
                "processing"
            )
            
            # Create wrapped task
            async def wrapped_execute(agent_instance):
                try:
                    result = await agent_instance.execute_with_monitoring(market_data)
                    await callback(
                        agent_instance.agent_id,
                        agent_instance.config.name,
                        "completed",
                        result
                    )
                    return result
                except Exception as e:
                    await callback(
                        agent_instance.agent_id,
                        agent_instance.config.name,
                        "failed",
                        str(e)
                    )
                    raise
            
            tasks.append(wrapped_execute(agent))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid signals
        valid_signals = []
        for agent, result in zip(active_agents, results):
            if isinstance(result, Signal):
                valid_signals.append((agent, result))
        
        # Aggregate signals using orchestrator
        if valid_signals:
            return await self.agent_orchestrator._aggregate_signals(valid_signals, market_data)
        
        return None
    
    async def _market_monitoring_loop(self):
        """
        Background task that monitors market for significant changes
        and triggers automatic analysis
        """
        monitored_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ"]
        last_prices = {}
        
        while self._running:
            try:
                for symbol in monitored_symbols:
                    # Skip if analysis is already active
                    if any(a["symbol"] == symbol and a["status"] == "analyzing" 
                          for a in self.active_analyses.values()):
                        continue
                    
                    # Get current quote
                    quote = await self.market_data_service.get_quote(symbol)
                    if not quote:
                        continue
                    
                    current_price = quote.get("price", 0)
                    
                    # Check for significant price change
                    if symbol in last_prices:
                        price_change = abs(current_price - last_prices[symbol]) / last_prices[symbol]
                        
                        # Trigger analysis on 1% change
                        if price_change >= 0.01:
                            logger.info(f"Significant price change for {symbol}: {price_change:.2%}")
                            asyncio.create_task(self.analyze_symbol(symbol))
                    
                    last_prices[symbol] = current_price
                    
                    # Broadcast price update
                    await ws_manager.broadcast_price_update(
                        symbol=symbol,
                        price=current_price,
                        volume=quote.get("volume", 0)
                    )
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Market monitoring error: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _signal_processing_loop(self):
        """
        Background task that processes signals from the queue
        and broadcasts them via WebSocket
        """
        while self._running:
            try:
                # Check for new signals in database
                async for db in get_db():
                    recent_signals = await self.signal_service.get_recent_signals(
                        db=db,
                        limit=10,
                        minutes=1  # Last minute
                    )
                    
                    for signal in recent_signals:
                        # Skip if already broadcasted (check metadata)
                        if signal.metadata and signal.metadata.get("broadcasted"):
                            continue
                        
                        # Create signal update
                        signal_update = SignalUpdate(
                            symbol=signal.symbol,
                            signal_id=str(signal.id),
                            action=signal.action,
                            confidence=signal.confidence,
                            price=signal.price,
                            agents_consensus=signal.metadata.get("agents_consensus", {}),
                            timestamp=signal.timestamp,
                            metadata=signal.metadata
                        )
                        
                        # Broadcast signal
                        await ws_manager.broadcast_signal(signal_update)
                        
                        # Mark as broadcasted
                        signal.metadata = signal.metadata or {}
                        signal.metadata["broadcasted"] = True
                        await db.commit()
                    
                    break
                
                # Sleep for 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
                await asyncio.sleep(10)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "running": self._running,
            "active_analyses": len(self.active_analyses),
            "active_agents": len([
                a for a in self.agent_orchestrator.agents.values()
                if a.config.enabled
            ]),
            "websocket_connections": ws_manager.get_connection_count(),
            "websocket_metrics": ws_manager.get_metrics(),
            "current_analyses": [
                {
                    "symbol": a["symbol"],
                    "status": a["status"],
                    "started_at": a["started_at"].isoformat()
                }
                for a in self.active_analyses.values()
            ]
        }


# Create singleton instance
websocket_orchestrator = WebSocketOrchestrator()