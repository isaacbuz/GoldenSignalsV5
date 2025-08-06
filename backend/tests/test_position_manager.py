"""
Tests for Position Management System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
import uuid

from core.position_manager import (
    PositionManager,
    PositionInfo,
    PositionRequest,
    PositionUpdate,
    PositionType,
    PositionStatus,
    OrderType,
    OrderSide,
    PortfolioSummary
)
from database.models import Position, Portfolio, Trade


class TestPositionManager:
    """Test PositionManager class"""
    
    @pytest.fixture
    def manager(self):
        """Create PositionManager instance"""
        return PositionManager()
    
    @pytest.fixture
    def position_request(self):
        """Create sample position request"""
        return PositionRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            stop_loss=145.0,
            take_profit=155.0,
            signal_id="sig_001",
            agent_id="test_agent",
            notes="Test position",
            tags=["test", "long"]
        )
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio"""
        portfolio = Mock(spec=Portfolio)
        portfolio.id = 1
        portfolio.name = "Test Portfolio"
        portfolio.initial_capital = 100000.0
        portfolio.cash_balance = 90000.0
        portfolio.created_at = datetime.now()
        return portfolio
    
    @pytest.mark.asyncio
    async def test_open_position(self, manager, position_request, mock_portfolio):
        """Test opening a new position"""
        with patch('database.connection.get_db') as mock_get_db:
            # Setup mock database session
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value = mock_portfolio
            
            # Mock market data
            with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
                mock_price.return_value = {"price": 150.0}
                
                # Open position
                position = await manager.open_position(1, position_request)
                
                # Verify position created
                assert isinstance(position, PositionInfo)
                assert position.symbol == "AAPL"
                assert position.position_type == PositionType.LONG
                assert position.quantity == 100
                assert position.avg_entry_price > 0
                assert position.stop_loss == 145.0
                assert position.take_profit == 155.0
                assert position.source_signal_id == "sig_001"
                assert position.source_agent_id == "test_agent"
                
                # Verify database operations
                assert mock_session.add.called
                assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_close_position(self, manager):
        """Test closing a position"""
        # Create a position in cache
        position_info = PositionInfo(
            position_id="pos_001",
            portfolio_id=1,
            symbol="AAPL",
            position_type=PositionType.LONG,
            quantity=100,
            avg_entry_price=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=0,
            total_pnl=500.0,
            pnl_percentage=3.33,
            position_value=15500.0,
            exposure=15500.0,
            margin_required=3875.0,
            leverage=1.0,
            opened_at=datetime.now() - timedelta(hours=2),
            last_updated=datetime.now(),
            holding_period=timedelta(hours=2)
        )
        manager.positions_cache["pos_001"] = position_info
        
        with patch('database.connection.get_db') as mock_get_db:
            # Setup mock database
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            
            mock_position = Mock(spec=Position)
            mock_position.id = 1
            mock_session.get.return_value = mock_position
            
            mock_portfolio = Mock(spec=Portfolio)
            mock_portfolio.cash_balance = 90000.0
            mock_session.get.return_value = mock_portfolio
            
            # Mock market data
            with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
                mock_price.return_value = {"price": 155.0}
                
                # Close position
                result = await manager.close_position("pos_001")
                
                # Verify result
                assert result["position_id"] == "pos_001"
                assert result["quantity_closed"] == 100
                assert result["pnl"] > 0  # Should have profit
                assert result["fully_closed"] == True
                
                # Verify position removed from cache
                assert "pos_001" not in manager.positions_cache
    
    @pytest.mark.asyncio
    async def test_partial_close(self, manager):
        """Test partially closing a position"""
        # Create position
        position_info = PositionInfo(
            position_id="pos_002",
            portfolio_id=1,
            symbol="TSLA",
            position_type=PositionType.LONG,
            quantity=200,
            avg_entry_price=250.0,
            current_price=260.0,
            unrealized_pnl=2000.0,
            realized_pnl=0,
            total_pnl=2000.0,
            pnl_percentage=4.0,
            position_value=52000.0,
            exposure=52000.0,
            margin_required=13000.0,
            leverage=1.0,
            opened_at=datetime.now() - timedelta(days=1),
            last_updated=datetime.now(),
            holding_period=timedelta(days=1)
        )
        manager.positions_cache["pos_002"] = position_info
        
        with patch('database.connection.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            
            mock_position = Mock(spec=Position)
            mock_position.id = 2
            mock_session.get.return_value = mock_position
            
            with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
                mock_price.return_value = {"price": 260.0}
                
                # Partially close position (50%)
                result = await manager.close_position("pos_002", quantity=100)
                
                # Verify result
                assert result["quantity_closed"] == 100
                assert result["remaining_quantity"] == 100
                assert result["fully_closed"] == False
                
                # Verify position still in cache with reduced quantity
                assert "pos_002" in manager.positions_cache
                assert manager.positions_cache["pos_002"].quantity == 100
    
    @pytest.mark.asyncio
    async def test_update_position(self, manager):
        """Test updating position parameters"""
        # Create position
        position_info = PositionInfo(
            position_id="pos_003",
            portfolio_id=1,
            symbol="GOOGL",
            position_type=PositionType.LONG,
            quantity=50,
            avg_entry_price=2800.0,
            current_price=2850.0,
            unrealized_pnl=2500.0,
            realized_pnl=0,
            total_pnl=2500.0,
            pnl_percentage=1.79,
            position_value=142500.0,
            exposure=142500.0,
            margin_required=35625.0,
            leverage=1.0,
            opened_at=datetime.now(),
            last_updated=datetime.now(),
            holding_period=timedelta(0)
        )
        manager.positions_cache["pos_003"] = position_info
        
        # Update position
        update = PositionUpdate(
            stop_loss=2750.0,
            take_profit=2900.0,
            trailing_stop_percent=2.0,
            notes="Updated stop loss",
            tags=["updated", "trailing"]
        )
        
        with patch('database.connection.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            
            updated = await manager.update_position("pos_003", update)
            
            # Verify updates
            assert updated.stop_loss == 2750.0
            assert updated.take_profit == 2900.0
            assert updated.trailing_stop_distance == 2.0
            assert updated.notes == "Updated stop loss"
            assert "updated" in updated.tags
    
    @pytest.mark.asyncio
    async def test_check_stop_loss(self, manager):
        """Test stop loss execution"""
        # Create position with stop loss
        position_info = PositionInfo(
            position_id="pos_004",
            portfolio_id=1,
            symbol="AAPL",
            position_type=PositionType.LONG,
            quantity=100,
            avg_entry_price=150.0,
            current_price=148.0,  # Below stop loss
            stop_loss=149.0,  # Stop loss triggered
            take_profit=155.0,
            unrealized_pnl=-200.0,
            realized_pnl=0,
            total_pnl=-200.0,
            pnl_percentage=-1.33,
            position_value=14800.0,
            exposure=14800.0,
            margin_required=3700.0,
            leverage=1.0,
            opened_at=datetime.now(),
            last_updated=datetime.now(),
            holding_period=timedelta(0)
        )
        manager.positions_cache["pos_004"] = position_info
        
        with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
            mock_price.return_value = {"price": 148.0}
            
            with patch.object(manager, 'close_position', new_callable=AsyncMock) as mock_close:
                mock_close.return_value = {
                    "position_id": "pos_004",
                    "pnl": -200.0,
                    "fully_closed": True
                }
                
                # Check stops
                await manager.check_stop_loss_take_profit()
                
                # Verify stop loss triggered
                mock_close.assert_called_once()
                call_args = mock_close.call_args
                assert call_args[0][0] == "pos_004"  # Position ID
                assert call_args[1]["order_type"] == OrderType.MARKET
    
    @pytest.mark.asyncio
    async def test_check_take_profit(self, manager):
        """Test take profit execution"""
        # Create position with take profit
        position_info = PositionInfo(
            position_id="pos_005",
            portfolio_id=1,
            symbol="TSLA",
            position_type=PositionType.LONG,
            quantity=50,
            avg_entry_price=250.0,
            current_price=265.0,  # Above take profit
            stop_loss=240.0,
            take_profit=260.0,  # Take profit triggered
            unrealized_pnl=750.0,
            realized_pnl=0,
            total_pnl=750.0,
            pnl_percentage=6.0,
            position_value=13250.0,
            exposure=13250.0,
            margin_required=3312.5,
            leverage=1.0,
            opened_at=datetime.now(),
            last_updated=datetime.now(),
            holding_period=timedelta(0)
        )
        manager.positions_cache["pos_005"] = position_info
        
        with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
            mock_price.return_value = {"price": 265.0}
            
            with patch.object(manager, 'close_position', new_callable=AsyncMock) as mock_close:
                mock_close.return_value = {
                    "position_id": "pos_005",
                    "pnl": 750.0,
                    "fully_closed": True
                }
                
                # Check stops
                await manager.check_stop_loss_take_profit()
                
                # Verify take profit triggered
                mock_close.assert_called_once()
                call_args = mock_close.call_args
                assert call_args[0][0] == "pos_005"  # Position ID
    
    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self, manager, mock_portfolio):
        """Test getting portfolio summary"""
        # Add positions to cache
        positions = [
            PositionInfo(
                position_id=f"pos_{i}",
                portfolio_id=1,
                symbol=f"STOCK{i}",
                position_type=PositionType.LONG,
                quantity=100,
                avg_entry_price=100.0 + i * 10,
                current_price=105.0 + i * 10,
                unrealized_pnl=500.0 * (1 if i % 2 == 0 else -1),
                realized_pnl=200.0 * i,
                total_pnl=500.0 * (1 if i % 2 == 0 else -1) + 200.0 * i,
                pnl_percentage=5.0 * (1 if i % 2 == 0 else -1),
                position_value=10500.0 + i * 1000,
                exposure=10500.0 + i * 1000,
                margin_required=2625.0 + i * 250,
                leverage=1.0,
                opened_at=datetime.now() - timedelta(days=i),
                last_updated=datetime.now(),
                holding_period=timedelta(days=i)
            )
            for i in range(3)
        ]
        
        for pos in positions:
            manager.positions_cache[pos.position_id] = pos
        
        with patch('database.connection.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value = mock_portfolio
            
            # Get summary
            summary = await manager.get_portfolio_summary(1)
            
            # Verify summary
            assert isinstance(summary, PortfolioSummary)
            assert summary.portfolio_id == 1
            assert summary.name == "Test Portfolio"
            assert summary.open_positions == 3
            assert summary.cash_balance == 90000.0
            assert summary.positions_value > 0
            assert summary.total_value > 0
    
    @pytest.mark.asyncio
    async def test_risk_limits(self, manager, mock_portfolio):
        """Test risk limit validation"""
        # Set strict risk limits
        manager.max_position_size_pct = 0.05  # 5% max per position
        manager.max_leverage = 2.0
        
        # Try to open a position that exceeds limits
        large_request = PositionRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000,  # Large quantity
            order_type=OrderType.MARKET
        )
        
        with patch('database.connection.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value = mock_portfolio
            
            with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
                mock_price.return_value = {"price": 150.0}
                
                # Mock get_portfolio_summary to return current state
                with patch.object(manager, 'get_portfolio_summary') as mock_summary:
                    mock_summary.return_value = PortfolioSummary(
                        portfolio_id=1,
                        name="Test Portfolio",
                        total_value=100000.0,
                        cash_balance=90000.0,
                        positions_value=10000.0,
                        total_unrealized_pnl=0,
                        total_realized_pnl=0,
                        daily_pnl=0,
                        total_return_pct=0,
                        total_exposure=10000.0,
                        margin_used=2500.0,
                        margin_available=87500.0,
                        leverage=0.1,
                        open_positions=1,
                        winning_positions=0,
                        losing_positions=0,
                        sharpe_ratio=0,
                        max_drawdown=0,
                        win_rate=0,
                        avg_win=0,
                        avg_loss=0,
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    
                    # Should raise error due to position size limit
                    with pytest.raises(ValueError, match="Position size.*exceeds limit"):
                        await manager.open_position(1, large_request)
    
    @pytest.mark.asyncio
    async def test_update_position_prices(self, manager):
        """Test updating position prices and P&L"""
        # Create position
        position = PositionInfo(
            position_id="pos_006",
            portfolio_id=1,
            symbol="MSFT",
            position_type=PositionType.LONG,
            quantity=100,
            avg_entry_price=300.0,
            current_price=300.0,
            unrealized_pnl=0,
            realized_pnl=0,
            total_pnl=0,
            pnl_percentage=0,
            position_value=30000.0,
            exposure=30000.0,
            margin_required=7500.0,
            leverage=1.0,
            opened_at=datetime.now(),
            last_updated=datetime.now(),
            holding_period=timedelta(0)
        )
        manager.positions_cache["pos_006"] = position
        
        with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
            # Price goes up
            mock_price.return_value = {"price": 310.0}
            
            # Update prices
            await manager.update_position_prices()
            
            # Verify P&L updated
            updated = manager.positions_cache["pos_006"]
            assert updated.current_price == 310.0
            assert updated.unrealized_pnl == 1000.0  # (310-300) * 100
            assert updated.total_pnl == 1000.0
            assert updated.position_value == 31000.0
            assert updated.max_profit == 1000.0
    
    @pytest.mark.asyncio
    async def test_short_position(self, manager, mock_portfolio):
        """Test handling short positions"""
        # Create short position request
        short_request = PositionRequest(
            symbol="GME",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.MARKET,
            stop_loss=25.0,  # Stop loss above entry for short
            take_profit=15.0  # Take profit below entry for short
        )
        
        with patch('database.connection.get_db') as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value = mock_portfolio
            
            with patch('core.market_data.aggregator.market_data_aggregator.get_latest_price') as mock_price:
                mock_price.return_value = {"price": 20.0}
                
                # Open short position
                position = await manager.open_position(1, short_request)
                
                # Verify short position
                assert position.position_type == PositionType.SHORT
                assert position.quantity == 50
                assert position.stop_loss == 25.0
                assert position.take_profit == 15.0