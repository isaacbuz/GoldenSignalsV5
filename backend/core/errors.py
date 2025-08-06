"""
Centralized Error Handling
Custom exceptions and error handlers for the application
"""

from typing import Any, Dict, Optional, Union
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from datetime import datetime
import traceback

from core.logging import get_logger

logger = get_logger(__name__)


class BaseAPIException(HTTPException):
    """Base exception for all API errors"""
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "Internal server error",
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.utcnow()


class ValidationError(BaseAPIException):
    """Raised when input validation fails"""
    
    def __init__(self, detail: str, field: Optional[str] = None):
        context = {"field": field} if field else {}
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="VALIDATION_ERROR",
            context=context
        )


class NotFoundError(BaseAPIException):
    """Raised when a resource is not found"""
    
    def __init__(self, resource: str, identifier: Union[str, int]):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found",
            error_code="NOT_FOUND",
            context={"resource": resource, "identifier": str(identifier)}
        )


class AuthenticationError(BaseAPIException):
    """Raised when authentication fails"""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(BaseAPIException):
    """Raised when user lacks permission"""
    
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="AUTHORIZATION_ERROR"
        )


class RateLimitError(BaseAPIException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            context={"retry_after": retry_after}
        )


class ServiceUnavailableError(BaseAPIException):
    """Raised when a service is temporarily unavailable"""
    
    def __init__(self, service: str, detail: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail or f"{service} is temporarily unavailable",
            error_code="SERVICE_UNAVAILABLE",
            context={"service": service}
        )


class MarketDataError(BaseAPIException):
    """Raised when market data operations fail"""
    
    def __init__(self, detail: str, symbol: Optional[str] = None):
        context = {"symbol": symbol} if symbol else {}
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="MARKET_DATA_ERROR",
            context=context
        )


class SignalGenerationError(BaseAPIException):
    """Raised when signal generation fails"""
    
    def __init__(self, detail: str, agent: Optional[str] = None):
        context = {"agent": agent} if agent else {}
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="SIGNAL_GENERATION_ERROR",
            context=context
        )


class DatabaseError(BaseAPIException):
    """Raised when database operations fail"""
    
    def __init__(self, detail: str, operation: Optional[str] = None):
        context = {"operation": operation} if operation else {}
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="DATABASE_ERROR",
            context=context
        )


async def api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """Handle API exceptions with consistent format"""
    error_id = f"ERR-{exc.timestamp.strftime('%Y%m%d%H%M%S')}-{exc.error_code}"
    
    # Log the error
    logger.error(
        f"API Error {error_id}: {exc.detail}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "context": exc.context,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # Return formatted error response
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "id": error_id,
                "code": exc.error_code,
                "message": exc.detail,
                "context": exc.context,
                "timestamp": exc.timestamp.isoformat(),
                "path": request.url.path
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    error_id = f"ERR-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-INTERNAL"
    
    # Log the full exception
    logger.error(
        f"Unhandled Exception {error_id}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # In production, hide internal error details
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "id": error_id,
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path
            }
        }
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle validation exceptions from Pydantic"""
    errors = []
    
    if hasattr(exc, 'errors'):
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "errors": errors,
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path
            }
        }
    )


def setup_exception_handlers(app):
    """Setup exception handlers for the FastAPI app"""
    from fastapi.exceptions import RequestValidationError
    
    # Custom exception handlers
    app.add_exception_handler(BaseAPIException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("Exception handlers configured")