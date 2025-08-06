"""
Event Bus Implementation
Centralized event handling for loose coupling between components
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import inspect
from collections import defaultdict
import weakref

from core.logging import get_logger

logger = get_logger(__name__)


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Base event class"""
    type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Centralized event bus for publish-subscribe pattern
    Supports both sync and async handlers with priority
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._weak_handlers: Dict[str, List[weakref.ref]] = defaultdict(list)
        self._priority_queues: Dict[EventPriority, asyncio.Queue] = {}
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._middleware: List[Callable] = []
        self._event_history: List[Event] = []
        self._max_history = 1000
        
        # Statistics
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'handlers_registered': 0
        }
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        weak: bool = False,
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """
        Subscribe to an event type
        
        Args:
            event_type: Type of event to subscribe to (supports wildcards)
            handler: Callback function
            weak: Use weak reference (handler can be garbage collected)
            priority: Handler priority
        """
        if weak:
            self._weak_handlers[event_type].append(weakref.ref(handler))
        elif asyncio.iscoroutinefunction(handler):
            self._async_handlers[event_type].append(handler)
        else:
            self._handlers[event_type].append(handler)
        
        self._stats['handlers_registered'] += 1
        logger.debug(f"Registered handler for {event_type}: {handler.__name__}")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type"""
        if handler in self._handlers.get(event_type, []):
            self._handlers[event_type].remove(handler)
        elif handler in self._async_handlers.get(event_type, []):
            self._async_handlers[event_type].remove(handler)
        
        # Clean up weak references
        self._clean_weak_handlers(event_type)
        
        logger.debug(f"Unregistered handler for {event_type}: {handler.__name__}")
    
    async def publish(
        self,
        event_type: str,
        data: Any = None,
        priority: EventPriority = EventPriority.NORMAL,
        **kwargs
    ) -> None:
        """
        Publish an event
        
        Args:
            event_type: Type of event
            data: Event data
            priority: Event priority
            **kwargs: Additional event metadata
        """
        event = Event(
            type=event_type,
            data=data,
            priority=priority,
            **kwargs
        )
        
        # Apply middleware
        for middleware in self._middleware:
            event = await self._apply_middleware(middleware, event)
            if event is None:
                return  # Event filtered out
        
        # Add to history
        self._add_to_history(event)
        
        # Process immediately if high priority
        if priority == EventPriority.CRITICAL:
            await self._process_event(event)
        else:
            # Add to queue for processing
            if self._running:
                queue = self._get_priority_queue(priority)
                await queue.put(event)
        
        self._stats['events_published'] += 1
        logger.debug(f"Published event: {event_type}")
    
    async def emit(self, event: Event) -> None:
        """Emit a pre-constructed event"""
        await self.publish(
            event.type,
            event.data,
            event.priority,
            source=event.source,
            correlation_id=event.correlation_id,
            metadata=event.metadata
        )
    
    async def _process_event(self, event: Event) -> None:
        """Process a single event"""
        try:
            # Get all matching handlers (including wildcards)
            handlers = self._get_matching_handlers(event.type)
            
            # Process sync handlers
            for handler in handlers['sync']:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Sync handler {handler.__name__} failed: {str(e)}")
                    self._stats['events_failed'] += 1
            
            # Process async handlers
            tasks = []
            for handler in handlers['async']:
                tasks.append(self._call_async_handler(handler, event))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Async handler failed: {str(result)}")
                        self._stats['events_failed'] += 1
            
            # Process weak handlers
            for handler_ref in handlers['weak']:
                handler = handler_ref()
                if handler:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Weak handler failed: {str(e)}")
                        self._stats['events_failed'] += 1
            
            self._stats['events_processed'] += 1
            
        except Exception as e:
            logger.error(f"Event processing failed for {event.type}: {str(e)}")
            self._stats['events_failed'] += 1
    
    async def _call_async_handler(self, handler: Callable, event: Event) -> Any:
        """Call async handler with error handling"""
        try:
            return await handler(event)
        except Exception as e:
            logger.error(f"Async handler {handler.__name__} failed: {str(e)}")
            raise
    
    def _get_matching_handlers(self, event_type: str) -> Dict[str, List]:
        """Get all handlers matching the event type (supports wildcards)"""
        matching = {
            'sync': [],
            'async': [],
            'weak': []
        }
        
        # Exact match
        if event_type in self._handlers:
            matching['sync'].extend(self._handlers[event_type])
        if event_type in self._async_handlers:
            matching['async'].extend(self._async_handlers[event_type])
        if event_type in self._weak_handlers:
            matching['weak'].extend(self._weak_handlers[event_type])
        
        # Wildcard matching (e.g., "trade.*" matches "trade.executed")
        for pattern in self._handlers.keys():
            if self._matches_pattern(event_type, pattern) and pattern != event_type:
                matching['sync'].extend(self._handlers[pattern])
        
        for pattern in self._async_handlers.keys():
            if self._matches_pattern(event_type, pattern) and pattern != event_type:
                matching['async'].extend(self._async_handlers[pattern])
        
        for pattern in self._weak_handlers.keys():
            if self._matches_pattern(event_type, pattern) and pattern != event_type:
                matching['weak'].extend(self._weak_handlers[pattern])
        
        return matching
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches pattern (supports * wildcard)"""
        if '*' not in pattern:
            return event_type == pattern
        
        # Convert pattern to regex
        import re
        regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
        return re.match(f"^{regex_pattern}$", event_type) is not None
    
    def _clean_weak_handlers(self, event_type: str) -> None:
        """Remove dead weak references"""
        if event_type in self._weak_handlers:
            self._weak_handlers[event_type] = [
                ref for ref in self._weak_handlers[event_type]
                if ref() is not None
            ]
    
    def _get_priority_queue(self, priority: EventPriority) -> asyncio.Queue:
        """Get or create priority queue"""
        if priority not in self._priority_queues:
            self._priority_queues[priority] = asyncio.Queue()
        return self._priority_queues[priority]
    
    async def _apply_middleware(self, middleware: Callable, event: Event) -> Optional[Event]:
        """Apply middleware to event"""
        if asyncio.iscoroutinefunction(middleware):
            return await middleware(event)
        else:
            return middleware(event)
    
    def _add_to_history(self, event: Event) -> None:
        """Add event to history"""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware for event processing"""
        self._middleware.append(middleware)
        logger.debug(f"Added middleware: {middleware.__name__}")
    
    async def start(self) -> None:
        """Start event processing"""
        if self._running:
            return
        
        self._running = True
        
        # Initialize priority queues
        for priority in EventPriority:
            self._priority_queues[priority] = asyncio.Queue()
        
        # Start processor task
        self._processor_task = asyncio.create_task(self._process_events())
        
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop event processing"""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    async def _process_events(self) -> None:
        """Main event processing loop"""
        while self._running:
            try:
                # Process events by priority
                for priority in sorted(EventPriority, key=lambda p: p.value, reverse=True):
                    queue = self._priority_queues.get(priority)
                    if queue and not queue.empty():
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=0.1)
                            await self._process_event(event)
                        except asyncio.TimeoutError:
                            continue
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Event processing loop error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self._stats,
            'handlers_count': sum([
                len(self._handlers),
                len(self._async_handlers),
                len(self._weak_handlers)
            ]),
            'history_size': len(self._event_history),
            'queued_events': sum(
                q.qsize() for q in self._priority_queues.values()
            )
        }
    
    def get_history(self, limit: int = 100) -> List[Event]:
        """Get recent event history"""
        return self._event_history[-limit:]


# Global event bus instance
event_bus = EventBus()


# Common event types
class EventTypes:
    """Standard event types"""
    
    # Market events
    MARKET_DATA_RECEIVED = "market.data.received"
    MARKET_DATA_ERROR = "market.data.error"
    MARKET_OPEN = "market.open"
    MARKET_CLOSE = "market.close"
    
    # Trading events
    SIGNAL_GENERATED = "signal.generated"
    TRADE_EXECUTED = "trade.executed"
    TRADE_FAILED = "trade.failed"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    
    # Agent events
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"
    AGENT_SIGNAL = "agent.signal"
    
    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    
    # Backtest events
    BACKTEST_STARTED = "backtest.started"
    BACKTEST_COMPLETED = "backtest.completed"
    BACKTEST_FAILED = "backtest.failed"
    
    # WebSocket events
    WS_CONNECTED = "websocket.connected"
    WS_DISCONNECTED = "websocket.disconnected"
    WS_MESSAGE = "websocket.message"


# Event decorators for easy subscription
def on_event(event_type: str, priority: EventPriority = EventPriority.NORMAL):
    """Decorator to subscribe a method to an event"""
    def decorator(func):
        event_bus.subscribe(event_type, func, priority=priority)
        return func
    return decorator


def emit_event(event_type: str, priority: EventPriority = EventPriority.NORMAL):
    """Decorator to emit an event after method execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await event_bus.publish(
                event_type,
                data={'result': result, 'args': args, 'kwargs': kwargs},
                priority=priority,
                source=func.__name__
            )
            return result
        return wrapper
    return decorator