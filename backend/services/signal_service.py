"""
Signal Service - Business logic for trading signal operations

This service provides:
- Signal storage and retrieval
- Performance tracking
- Real-time signal streaming
- Analytics and metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import uuid
import redis
import json
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_async_db, AsyncSessionLocal
from core.config import settings
from models.signal import Signal, SignalAction, SignalStatus, RiskLevel
from services.websocket_manager import ws_manager, broadcast_new_signal

logger = logging.getLogger(__name__)


class SignalService:
    """Service for managing trading signals and their lifecycle"""

    def __init__(self):
        # Redis connection for pub/sub
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established for SignalService")
        except Exception as e:
            logger.warning(f"Redis not available for SignalService: {e}")
            self.redis_client = None

        # Pub/Sub for real-time signals
        self.pubsub = self.redis_client.pubsub() if self.redis_client else None

    async def create_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        reasoning: str,
        agents_consensus: Dict[str, Any],
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_level: str = "medium",
        timeframe: str = "1d",
        expires_in_hours: int = 24
    ) -> Signal:
        """
        Create a new trading signal
        
        Args:
            symbol: Stock symbol
            action: BUY, SELL, or HOLD
            confidence: Confidence level (0-1)
            price: Current price
            reasoning: AI reasoning for the signal
            agents_consensus: Individual agent votes
            stop_loss: Stop loss price
            take_profit: Take profit price
            risk_level: low, medium, or high
            timeframe: Trading timeframe
            expires_in_hours: Hours until signal expires
            
        Returns:
            Created Signal object
        """
        try:
            async with AsyncSessionLocal() as session:
                # Calculate consensus strength
                consensus_strength = self._calculate_consensus_strength(agents_consensus)
                
                # Create signal
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction[action.upper()],
                    confidence=confidence,
                    price=price,
                    target_price=take_profit,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_level=RiskLevel[risk_level.upper()],
                    timeframe=timeframe,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_in_hours),
                    reasoning=reasoning,
                    consensus_strength=consensus_strength,
                    agents_consensus=agents_consensus,
                    indicators={},  # Will be populated by agents
                    signal_source="ai_consensus",
                    status=SignalStatus.ACTIVE
                )
                
                # Save to database
                session.add(signal)
                await session.commit()
                await session.refresh(signal)
                
                # Broadcast to WebSocket subscribers
                signal_data = signal.to_dict()
                await broadcast_new_signal(signal_data)
                
                # Publish to Redis pub/sub
                if self.redis_client:
                    await self._publish_signal(signal_data)
                
                logger.info(f"Created signal {signal.id} for {symbol} - {action}")
                return signal
                
        except Exception as e:
            logger.error(f"Failed to create signal: {e}")
            raise

    async def get_signals(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        action: Optional[str] = None,
        min_confidence: Optional[float] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Signal]:
        """
        Get signals with filtering
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            action: Filter by action
            min_confidence: Minimum confidence threshold
            since: Filter signals created after this time
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of Signal objects
        """
        try:
            async with AsyncSessionLocal() as session:
                # Build query
                query = select(Signal)
                
                # Apply filters
                conditions = []
                if symbol:
                    conditions.append(Signal.symbol == symbol.upper())
                if status:
                    conditions.append(Signal.status == SignalStatus[status.upper()])
                if action:
                    conditions.append(Signal.action == SignalAction[action.upper()])
                if min_confidence:
                    conditions.append(Signal.confidence >= min_confidence)
                if since:
                    conditions.append(Signal.created_at >= since)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                # Order and limit
                query = query.order_by(Signal.created_at.desc()).limit(limit).offset(offset)
                
                # Execute
                result = await session.execute(query)
                signals = result.scalars().all()
                
                return signals
                
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            raise

    async def get_signal_by_id(self, signal_id: str) -> Optional[Signal]:
        """Get a specific signal by ID"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Signal).where(Signal.id == signal_id)
                )
                return result.scalar_one_or_none()
                
        except Exception as e:
            logger.error(f"Failed to get signal by ID: {e}")
            raise

    async def update_signal_status(
        self,
        signal_id: str,
        status: str,
        execution_price: Optional[float] = None
    ) -> Optional[Signal]:
        """
        Update signal status
        
        Args:
            signal_id: Signal ID
            status: New status
            execution_price: Price at which signal was executed
            
        Returns:
            Updated Signal object
        """
        try:
            async with AsyncSessionLocal() as session:
                # Get signal
                result = await session.execute(
                    select(Signal).where(Signal.id == signal_id)
                )
                signal = result.scalar_one_or_none()
                
                if not signal:
                    return None
                
                # Update status
                signal.status = SignalStatus[status.upper()]
                
                if execution_price:
                    signal.execution_price = execution_price
                    signal.executed_at = datetime.now(timezone.utc)
                
                await session.commit()
                await session.refresh(signal)
                
                return signal
                
        except Exception as e:
            logger.error(f"Failed to update signal status: {e}")
            raise

    async def update_signal_performance(
        self,
        signal_id: str,
        current_price: float
    ) -> Optional[Signal]:
        """Update signal performance metrics based on current price"""
        try:
            async with AsyncSessionLocal() as session:
                # Get signal
                result = await session.execute(
                    select(Signal).where(Signal.id == signal_id)
                )
                signal = result.scalar_one_or_none()
                
                if not signal:
                    return None
                
                # Update performance
                signal.update_performance(current_price)
                
                await session.commit()
                await session.refresh(signal)
                
                return signal
                
        except Exception as e:
            logger.error(f"Failed to update signal performance: {e}")
            raise

    async def get_signal_analytics(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get signal analytics and performance metrics
        
        Args:
            symbol: Optional symbol filter
            days: Number of days to analyze
            
        Returns:
            Analytics dictionary
        """
        try:
            async with AsyncSessionLocal() as session:
                # Base query
                since_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                # Total signals
                total_query = select(func.count(Signal.id)).where(
                    Signal.created_at >= since_date
                )
                if symbol:
                    total_query = total_query.where(Signal.symbol == symbol.upper())
                
                total_result = await session.execute(total_query)
                total_signals = total_result.scalar()
                
                # Signals by action
                action_query = select(
                    Signal.action,
                    func.count(Signal.id)
                ).where(
                    Signal.created_at >= since_date
                ).group_by(Signal.action)
                
                if symbol:
                    action_query = action_query.where(Signal.symbol == symbol.upper())
                
                action_result = await session.execute(action_query)
                signals_by_action = {
                    action.value: count for action, count in action_result
                }
                
                # Average confidence
                avg_confidence_query = select(
                    func.avg(Signal.confidence)
                ).where(Signal.created_at >= since_date)
                
                if symbol:
                    avg_confidence_query = avg_confidence_query.where(
                        Signal.symbol == symbol.upper()
                    )
                
                avg_confidence_result = await session.execute(avg_confidence_query)
                avg_confidence = avg_confidence_result.scalar() or 0
                
                # Profitable signals
                profitable_query = select(func.count(Signal.id)).where(
                    and_(
                        Signal.created_at >= since_date,
                        Signal.pnl > 0
                    )
                )
                if symbol:
                    profitable_query = profitable_query.where(Signal.symbol == symbol.upper())
                
                profitable_result = await session.execute(profitable_query)
                profitable_signals = profitable_result.scalar()
                
                # Calculate win rate
                executed_query = select(func.count(Signal.id)).where(
                    and_(
                        Signal.created_at >= since_date,
                        Signal.status == SignalStatus.EXECUTED
                    )
                )
                if symbol:
                    executed_query = executed_query.where(Signal.symbol == symbol.upper())
                
                executed_result = await session.execute(executed_query)
                executed_signals = executed_result.scalar()
                
                win_rate = (profitable_signals / executed_signals * 100) if executed_signals > 0 else 0
                
                return {
                    "period_days": days,
                    "symbol": symbol,
                    "total_signals": total_signals,
                    "signals_by_action": signals_by_action,
                    "average_confidence": round(avg_confidence, 3),
                    "executed_signals": executed_signals,
                    "profitable_signals": profitable_signals,
                    "win_rate": round(win_rate, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get signal analytics: {e}")
            raise

    async def cleanup_expired_signals(self) -> int:
        """Clean up expired signals and update their status"""
        try:
            async with AsyncSessionLocal() as session:
                # Find expired active signals
                result = await session.execute(
                    select(Signal).where(
                        and_(
                            Signal.status == SignalStatus.ACTIVE,
                            Signal.expires_at < datetime.now(timezone.utc)
                        )
                    )
                )
                expired_signals = result.scalars().all()
                
                # Update status
                for signal in expired_signals:
                    signal.status = SignalStatus.EXPIRED
                
                await session.commit()
                
                count = len(expired_signals)
                if count > 0:
                    logger.info(f"Cleaned up {count} expired signals")
                
                return count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired signals: {e}")
            raise

    def _calculate_consensus_strength(self, agents_consensus: Dict[str, Any]) -> float:
        """Calculate consensus strength from agent votes"""
        if not agents_consensus:
            return 0.0
        
        # Count votes
        buy_votes = sum(1 for vote in agents_consensus.values() if vote.get("action") == "BUY")
        sell_votes = sum(1 for vote in agents_consensus.values() if vote.get("action") == "SELL")
        hold_votes = sum(1 for vote in agents_consensus.values() if vote.get("action") == "HOLD")
        
        total_votes = buy_votes + sell_votes + hold_votes
        if total_votes == 0:
            return 0.0
        
        # Calculate consensus as percentage of agents agreeing with majority
        max_votes = max(buy_votes, sell_votes, hold_votes)
        consensus = max_votes / total_votes
        
        return round(consensus, 3)

    async def _publish_signal(self, signal_data: Dict[str, Any]) -> None:
        """Publish signal to Redis pub/sub"""
        if not self.redis_client:
            return
        
        try:
            channel = f"signals:{signal_data['symbol']}"
            message = json.dumps(signal_data)
            self.redis_client.publish(channel, message)
            
        except Exception as e:
            logger.error(f"Failed to publish signal to Redis: {e}")

    async def subscribe_to_signals(self, symbol: str) -> None:
        """Subscribe to real-time signals for a symbol"""
        if not self.pubsub:
            logger.warning("Redis pub/sub not available")
            return
        
        try:
            channel = f"signals:{symbol}"
            self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to signals for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to signals: {e}")


# Global service instance
signal_service = SignalService()