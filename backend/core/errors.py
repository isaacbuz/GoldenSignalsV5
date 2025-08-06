"""
Legacy Error Handling - DEPRECATED
Use core.exceptions for new error handling patterns
"""

from typing import Any, Dict, Optional, Union
from fastapi import Request, status
from fastapi.responses import JSONResponse
from datetime import datetime

# Re-export from the new exceptions module for backward compatibility
from core.exceptions import (
    BaseGoldenSignalsException as BaseAPIException,
    ErrorCode,
    ErrorContext,
    MarketDataException,
    AgentException,
    TradingException,
    SystemException,
    error_handler,
    handle_errors,
    global_exception_handler,
    raise_market_data_error,
    raise_agent_error,
    raise_trading_error,
    raise_system_error
)

# Legacy exception classes for backward compatibility
class ValidationError(BaseAPIException):
    """Legacy validation error - use core.exceptions"""
    def __init__(self, detail: str, field: str = None):
        context = ErrorContext(metadata={"field": field} if field else None)
        super().__init__(detail, ErrorCode.VALIDATION_ERROR, context, recoverable=True)


class NotFoundError(BaseAPIException):
    """Legacy not found error - use core.exceptions"""
    def __init__(self, resource: str, identifier: Union[str, int]):
        context = ErrorContext(metadata={"resource": resource, "identifier": str(identifier)})
        super().__init__(f"{resource} not found: {identifier}", ErrorCode.NOT_FOUND, context)


# Legacy setup function - now uses new exception handler
def setup_exception_handlers(app):
    """Setup global exception handlers for FastAPI"""
    from core.logging import get_logger
    
    logger = get_logger(__name__)
    logger.info("Exception handlers configured using new standardized system")
    
    # Use the new global exception handler
    app.add_exception_handler(Exception, global_exception_handler)
    
    # Import here to avoid circular imports
    from core.exceptions import BaseGoldenSignalsException
    
    # Custom handler for BaseGoldenSignalsException
    @app.exception_handler(BaseGoldenSignalsException)
    async def goldensignals_exception_handler(request: Request, exc: BaseGoldenSignalsException):
        """Handler for standardized GoldenSignals exceptions"""
        
        # Map error codes to HTTP status codes
        status_code_mapping = {
            ErrorCode.NOT_FOUND: 404,
            ErrorCode.AUTHENTICATION_ERROR: 401,
            ErrorCode.AUTHORIZATION_ERROR: 403,
            ErrorCode.VALIDATION_ERROR: 422,
            ErrorCode.CONFLICT: 409,
            ErrorCode.RATE_LIMITED: 429,
            ErrorCode.MARKET_DATA_UNAVAILABLE: 503,
            ErrorCode.AGENT_EXECUTION_ERROR: 500,
            ErrorCode.TRADING_ERROR: 400,
            ErrorCode.DATABASE_ERROR: 503,
            ErrorCode.SERVICE_UNAVAILABLE: 503,
        }
        
        http_status = status_code_mapping.get(exc.error_code, 500)
        
        return JSONResponse(
            status_code=http_status,
            content=exc.to_dict()
        )