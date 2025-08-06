"""
Trading-related Pydantic schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SignalAction(str, Enum):
    """Trading signal actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class OrderType(str, Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


class SignalRequest(BaseModel):
    """Request model for generating trading signal"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    agents: Optional[List[str]] = Field(None, description="Specific agents to use")
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()


class SignalResponse(BaseModel):
    """Response model for trading signal"""
    id: str
    symbol: str
    action: SignalAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    agents_consensus: Dict[str, Any]
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "sig_123456",
                "symbol": "AAPL",
                "action": "BUY",
                "confidence": 0.75,
                "timestamp": "2024-01-01T12:00:00Z",
                "entry_price": 150.00,
                "target_price": 155.00,
                "stop_loss": 147.00,
                "agents_consensus": {
                    "ma_crossover": {"signal": "BUY", "confidence": 0.8},
                    "sentiment": {"signal": "BUY", "confidence": 0.7}
                },
                "reasoning": "Strong bullish signals from multiple indicators",
                "metadata": {"volume_confirmation": True}
            }
        }


class PositionRequest(BaseModel):
    """Request model for creating a position"""
    symbol: str = Field(..., min_length=1, max_length=10)
    quantity: float = Field(..., gt=0)
    entry_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    signal_id: Optional[str] = None
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()


class PositionResponse(BaseModel):
    """Response model for position"""
    id: int
    symbol: str
    quantity: float
    entry_price: float
    current_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    unrealized_pnl: float
    realized_pnl: float
    opened_at: datetime
    status: str
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "symbol": "AAPL",
                "quantity": 100,
                "entry_price": 150.00,
                "current_price": 152.00,
                "stop_loss": 147.00,
                "take_profit": 155.00,
                "unrealized_pnl": 200.00,
                "realized_pnl": 0.00,
                "opened_at": "2024-01-01T09:30:00Z",
                "status": "OPEN"
            }
        }


class OrderRequest(BaseModel):
    """Request model for placing an order"""
    symbol: str = Field(..., min_length=1, max_length=10)
    order_type: OrderType
    side: OrderSide
    quantity: float = Field(..., gt=0)
    limit_price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    position_id: Optional[int] = None
    signal_id: Optional[str] = None
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()
    
    @validator('limit_price')
    def validate_limit_price(cls, v, values):
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Limit price required for LIMIT and STOP_LIMIT orders')
        return v
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        if values.get('order_type') in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Stop price required for STOP and STOP_LIMIT orders')
        return v


class OrderResponse(BaseModel):
    """Response model for order"""
    id: int
    symbol: str
    order_type: str
    side: str
    quantity: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    executed_price: Optional[float]
    status: str
    filled_quantity: float
    created_at: datetime
    executed_at: Optional[datetime]
    broker_order_id: Optional[str]
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "id": 1,
                "symbol": "AAPL",
                "order_type": "LIMIT",
                "side": "BUY",
                "quantity": 100,
                "limit_price": 150.00,
                "stop_price": None,
                "executed_price": 149.95,
                "status": "FILLED",
                "filled_quantity": 100,
                "created_at": "2024-01-01T09:30:00Z",
                "executed_at": "2024-01-01T09:30:15Z",
                "broker_order_id": "BRK123456"
            }
        }


class PortfolioResponse(BaseModel):
    """Response model for portfolio"""
    total_value: float
    cash_balance: float
    invested_value: float
    total_return: float
    daily_return: float
    positions: List[PositionResponse]
    performance_metrics: Dict[str, float]
    
    class Config:
        schema_extra = {
            "example": {
                "total_value": 100000.00,
                "cash_balance": 50000.00,
                "invested_value": 50000.00,
                "total_return": 5.25,
                "daily_return": 0.75,
                "positions": [],
                "performance_metrics": {
                    "sharpe_ratio": 1.25,
                    "max_drawdown": -5.50,
                    "win_rate": 0.65
                }
            }
        }