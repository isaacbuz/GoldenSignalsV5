"""
Structured Logging Configuration
Provides consistent logging across the application
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger
from pathlib import Path

from core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['service'] = 'goldensignals-ai'
        log_record['environment'] = settings.ENVIRONMENT
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add context if available
        if hasattr(record, 'context'):
            log_record['context'] = record.context
        
        # Add error details if exception
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


class ContextFilter(logging.Filter):
    """Add request context to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add request ID if available (set by middleware)
        from contextvars import ContextVar
        request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
        request_id = request_id_var.get()
        
        if request_id:
            record.request_id = request_id
        
        return True


def setup_logging() -> None:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level based on environment
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    root_logger.setLevel(log_level)
    
    # Console Handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s',
        rename_fields={'msg': 'message'}
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)
    
    # File Handler for errors
    error_file_handler = logging.FileHandler(log_dir / "errors.log")
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(console_formatter)
    error_file_handler.addFilter(ContextFilter())
    root_logger.addHandler(error_file_handler)
    
    # File Handler for all logs (rotated daily)
    from logging.handlers import TimedRotatingFileHandler
    
    all_file_handler = TimedRotatingFileHandler(
        log_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding="utf-8"
    )
    all_file_handler.setFormatter(console_formatter)
    all_file_handler.addFilter(ContextFilter())
    root_logger.addHandler(all_file_handler)
    
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log startup message
    root_logger.info(
        "Logging initialized",
        extra={
            "log_level": logging.getLevelName(log_level),
            "environment": settings.ENVIRONMENT,
            "log_dir": str(log_dir.absolute())
        }
    )


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter with context support"""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        # Add extra context to all log messages
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def get_logger(name: str, **context) -> LoggerAdapter:
    """
    Get a logger instance with optional context
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to include in all logs
        
    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


class LoggingMiddleware:
    """Middleware to add request context to logs"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate request ID
            import uuid
            from contextvars import ContextVar
            
            request_id = str(uuid.uuid4())
            request_id_var: ContextVar[str] = ContextVar('request_id')
            token = request_id_var.set(request_id)
            
            # Log request
            logger = get_logger("api.request")
            logger.info(
                "Request started",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "query_string": scope["query_string"].decode(),
                    "request_id": request_id
                }
            )
            
            # Track response status
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    logger.info(
                        "Request completed",
                        extra={
                            "status_code": message["status"],
                            "request_id": request_id
                        }
                    )
                await send(message)
            
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                request_id_var.reset(token)
        else:
            await self.app(scope, receive, send)


# Performance logging utilities

class timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str, logger: Optional[LoggerAdapter] = None):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        self.logger.info(
            f"Operation completed: {self.name}",
            extra={
                "operation": self.name,
                "duration_ms": duration_ms,
                "success": exc_type is None
            }
        )


def log_performance(func):
    """Decorator to log function performance"""
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        with timer(f"{func.__module__}.{func.__name__}", logger):
            return await func(*args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        with timer(f"{func.__module__}.{func.__name__}", logger):
            return func(*args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Initialize logging on module import
import asyncio
setup_logging()