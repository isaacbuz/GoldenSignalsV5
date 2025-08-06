"""
Standardized Exception Handling System
Provides consistent error handling patterns across the application
"""

import traceback
from typing import Any, Dict, Optional, Type, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes"""
    
    # General errors (1000-1999)
    UNKNOWN_ERROR = 1000
    VALIDATION_ERROR = 1001
    AUTHENTICATION_ERROR = 1002
    AUTHORIZATION_ERROR = 1003
    NOT_FOUND = 1004
    CONFLICT = 1005
    RATE_LIMITED = 1006
    
    # Market data errors (2000-2999)
    MARKET_DATA_UNAVAILABLE = 2000
    MARKET_DATA_INVALID = 2001
    MARKET_DATA_TIMEOUT = 2002
    SYMBOL_NOT_FOUND = 2003
    PROVIDER_ERROR = 2004
    
    # Agent/AI errors (3000-3999)
    AGENT_EXECUTION_ERROR = 3000
    AGENT_TIMEOUT = 3001
    AGENT_INITIALIZATION_ERROR = 3002
    MODEL_ERROR = 3003
    PREDICTION_ERROR = 3004
    
    # Trading errors (4000-4999)
    TRADING_ERROR = 4000
    INSUFFICIENT_FUNDS = 4001
    INVALID_ORDER = 4002
    ORDER_REJECTED = 4003
    
    # System errors (5000-5999)
    DATABASE_ERROR = 5000
    CACHE_ERROR = 5001
    NETWORK_ERROR = 5002
    CONFIGURATION_ERROR = 5003
    SERVICE_UNAVAILABLE = 5004


@dataclass
class ErrorContext:
    """Additional context for errors"""
    
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    symbol: Optional[str] = None
    agent_id: Optional[str] = None
    service: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseGoldenSignalsException(Exception):
    """Base exception for all GoldenSignals errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization"""
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "timestamp": self.timestamp.isoformat(),
                "recoverable": self.recoverable,
                "context": {
                    "user_id": self.context.user_id,
                    "session_id": self.context.session_id,
                    "request_id": self.context.request_id,
                    "symbol": self.context.symbol,
                    "agent_id": self.context.agent_id,
                    "service": self.context.service,
                    "metadata": self.context.metadata
                },
                "caused_by": str(self.cause) if self.cause else None
            }
        }


class MarketDataException(BaseGoldenSignalsException):
    """Market data related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.MARKET_DATA_UNAVAILABLE,
        symbol: Optional[str] = None,
        provider: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        context = ErrorContext(
            symbol=symbol,
            service="market_data",
            metadata={"provider": provider} if provider else None
        )
        super().__init__(message, error_code, context, cause, recoverable=True)


class AgentException(BaseGoldenSignalsException):
    """Agent/AI related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.AGENT_EXECUTION_ERROR,
        agent_id: Optional[str] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True
    ):
        context = ErrorContext(
            agent_id=agent_id,
            service="agent_orchestration"
        )
        super().__init__(message, error_code, context, cause, recoverable)


class TradingException(BaseGoldenSignalsException):
    """Trading related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.TRADING_ERROR,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        context = ErrorContext(
            symbol=symbol,
            service="trading",
            metadata={"order_id": order_id} if order_id else None
        )
        super().__init__(message, error_code, context, cause, recoverable=False)


class SystemException(BaseGoldenSignalsException):
    """System level errors"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        service: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        context = ErrorContext(service=service)
        super().__init__(message, error_code, context, cause, recoverable=True)


class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        log_level: int = logging.ERROR
    ) -> BaseGoldenSignalsException:
        """
        Handle any exception and convert to standardized format
        
        Args:
            exception: The original exception
            context: Additional error context
            log_level: Logging level for the error
            
        Returns:
            Standardized GoldenSignals exception
        """
        # Convert to standardized exception
        if isinstance(exception, BaseGoldenSignalsException):
            gs_exception = exception
        else:
            gs_exception = self._convert_to_gs_exception(exception, context)
        
        # Log the error
        self._log_error(gs_exception, log_level)
        
        # Track error metrics
        self._track_error(gs_exception)
        
        return gs_exception
    
    def _convert_to_gs_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None
    ) -> BaseGoldenSignalsException:
        """Convert standard exceptions to GoldenSignals exceptions"""
        
        if isinstance(exception, HTTPException):
            if exception.status_code == 404:
                error_code = ErrorCode.NOT_FOUND
            elif exception.status_code == 401:
                error_code = ErrorCode.AUTHENTICATION_ERROR
            elif exception.status_code == 403:
                error_code = ErrorCode.AUTHORIZATION_ERROR
            elif exception.status_code == 409:
                error_code = ErrorCode.CONFLICT
            elif exception.status_code == 429:
                error_code = ErrorCode.RATE_LIMITED
            else:
                error_code = ErrorCode.UNKNOWN_ERROR
            
            return BaseGoldenSignalsException(
                message=exception.detail,
                error_code=error_code,
                context=context,
                cause=exception
            )
        
        elif isinstance(exception, ValueError):
            return BaseGoldenSignalsException(
                message=str(exception),
                error_code=ErrorCode.VALIDATION_ERROR,
                context=context,
                cause=exception,
                recoverable=True
            )
        
        elif isinstance(exception, ConnectionError):
            return SystemException(
                message="Network connection error",
                error_code=ErrorCode.NETWORK_ERROR,
                cause=exception
            )
        
        elif isinstance(exception, TimeoutError):
            return SystemException(
                message="Operation timed out",
                error_code=ErrorCode.NETWORK_ERROR,
                cause=exception
            )
        
        else:
            # Generic exception
            return BaseGoldenSignalsException(
                message=str(exception),
                error_code=ErrorCode.UNKNOWN_ERROR,
                context=context,
                cause=exception
            )
    
    def _log_error(
        self,
        exception: BaseGoldenSignalsException,
        log_level: int = logging.ERROR
    ):
        """Log error with structured format"""
        
        log_data = {
            "error_code": exception.error_code.value,
            "message": exception.message,
            "recoverable": exception.recoverable,
            "timestamp": exception.timestamp.isoformat(),
        }
        
        # Add context information
        if exception.context:
            log_data.update({
                "user_id": exception.context.user_id,
                "session_id": exception.context.session_id,
                "symbol": exception.context.symbol,
                "agent_id": exception.context.agent_id,
                "service": exception.context.service,
            })
        
        # Add stack trace for system errors
        if exception.error_code.value >= 5000:
            log_data["stack_trace"] = traceback.format_exception(
                type(exception.cause) if exception.cause else type(exception),
                exception.cause or exception,
                exception.cause.__traceback__ if exception.cause else exception.__traceback__
            )
        
        logger.log(log_level, "Error occurred", extra=log_data)
    
    def _track_error(self, exception: BaseGoldenSignalsException):
        """Track error metrics"""
        error_key = f"{exception.error_code.value}:{exception.context.service or 'unknown'}"
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        
        self.error_counts[error_key] += 1
        
        # Keep recent error history
        self.error_history.append({
            "timestamp": exception.timestamp,
            "error_code": exception.error_code.value,
            "service": exception.context.service,
            "recoverable": exception.recoverable
        })
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error metrics and statistics"""
        return {
            "error_counts": self.error_counts,
            "total_errors": len(self.error_history),
            "recent_errors": self.error_history[-10:],  # Last 10 errors
            "error_rate": self._calculate_error_rate(),
            "top_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 most frequent errors
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate per hour"""
        if not self.error_history:
            return 0.0
        
        now = datetime.now()
        recent_errors = [
            error for error in self.error_history
            if (now - error["timestamp"]).total_seconds() < 3600  # Last hour
        ]
        
        return len(recent_errors)  # Errors per hour


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(func):
    """Decorator for standardized error handling"""
    
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            gs_exception = error_handler.handle_exception(e)
            raise HTTPException(
                status_code=500,
                detail=gs_exception.to_dict()
            )
    
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            gs_exception = error_handler.handle_exception(e)
            raise HTTPException(
                status_code=500,
                detail=gs_exception.to_dict()
            )
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Custom exception handler for FastAPI
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for FastAPI"""
    
    context = ErrorContext(
        request_id=request.headers.get("X-Request-ID"),
        metadata={"path": str(request.url.path), "method": request.method}
    )
    
    gs_exception = error_handler.handle_exception(exc, context)
    
    # Map error codes to HTTP status codes
    status_code_mapping = {
        ErrorCode.NOT_FOUND: 404,
        ErrorCode.AUTHENTICATION_ERROR: 401,
        ErrorCode.AUTHORIZATION_ERROR: 403,
        ErrorCode.VALIDATION_ERROR: 422,
        ErrorCode.CONFLICT: 409,
        ErrorCode.RATE_LIMITED: 429,
    }
    
    http_status = status_code_mapping.get(gs_exception.error_code, 500)
    
    return JSONResponse(
        status_code=http_status,
        content=gs_exception.to_dict()
    )


# Utility functions for common error scenarios
def raise_market_data_error(
    message: str,
    symbol: Optional[str] = None,
    provider: Optional[str] = None,
    cause: Optional[Exception] = None
):
    """Raise standardized market data error"""
    raise MarketDataException(message, symbol=symbol, provider=provider, cause=cause)


def raise_agent_error(
    message: str,
    agent_id: Optional[str] = None,
    cause: Optional[Exception] = None
):
    """Raise standardized agent error"""
    raise AgentException(message, agent_id=agent_id, cause=cause)


def raise_trading_error(
    message: str,
    symbol: Optional[str] = None,
    order_id: Optional[str] = None,
    cause: Optional[Exception] = None
):
    """Raise standardized trading error"""
    raise TradingException(message, symbol=symbol, order_id=order_id, cause=cause)


def raise_system_error(
    message: str,
    service: Optional[str] = None,
    cause: Optional[Exception] = None
):
    """Raise standardized system error"""
    raise SystemException(message, service=service, cause=cause)