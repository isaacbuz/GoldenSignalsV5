"""
GoldenSignalsAI V5 - Main Application
Clean architecture implementation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

# Setup structured logging
from core.logging import get_logger, LoggingMiddleware
from core.errors import setup_exception_handlers

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting GoldenSignalsAI V5...")
    
    # Initialize database
    from core.database import init_database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. Running without persistence.")
    
    # Initialize service registry
    from core.dependencies import service_registry
    await service_registry.initialize_async_services()
    
    # Initialize Unified Orchestrator
    from core.orchestrator import unified_orchestrator
    logger.info("Initializing unified orchestrator...")
    await unified_orchestrator.initialize()
    
    # Validate settings
    from core.config import validate_settings
    validate_settings()
    
    yield
    
    # Cleanup
    logger.info("Shutting down GoldenSignalsAI V5...")
    
    # Stop Unified Orchestrator (if needed)
    # The unified orchestrator doesn't need explicit stop
    
    # Shutdown services
    await service_registry.shutdown_async_services()
    
    # Close database
    from core.database import db_manager
    await db_manager.async_close()

# Create application
app = FastAPI(
    title="GoldenSignalsAI V5",
    description="AI-Powered Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Setup exception handlers
setup_exception_handlers(app)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Configure CORS with security in mind
from core.config import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Health check
@app.get("/api/v1/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "GoldenSignalsAI V5"}

# Detailed health check
@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    """Detailed health check with all service statuses"""
    from core.dependencies import check_database_health, check_redis_health, check_services_health
    
    health_status = {
        "status": "healthy",
        "service": "GoldenSignalsAI V5",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Check database
    db_health = await check_database_health()
    health_status["checks"]["database"] = db_health
    
    # Check Redis
    redis_health = await check_redis_health()
    health_status["checks"]["redis"] = redis_health
    
    # Check services
    services_health = await check_services_health()
    health_status["checks"]["services"] = services_health
    
    # Determine overall status
    all_healthy = all(
        "healthy" in str(check) 
        for checks in health_status["checks"].values() 
        for check in (checks.values() if isinstance(checks, dict) else [checks])
    )
    
    health_status["status"] = "healthy" if all_healthy else "degraded"
    
    return health_status

# Import routers
from api.routes import rag, mcp, market_data, signals, agents, auth, ai_analysis, chart_analysis, options_flow, universal_data, meta_consensus, ml_training, gamma_exposure, market_regime, smart_execution, liquidity_prediction, risk_management, volatility, unified_dashboard, alerts, ml, government_data, alternative_data
from api.websocket import market_ws

# Include routers
app.include_router(auth.router, prefix="/api/v1")  # Auth routes first
app.include_router(rag.router, prefix="/api/v1")
app.include_router(mcp.router, prefix="/api/v1")
# app.include_router(mcp_v2.router, prefix="/api/v2")  # Enhanced MCP routes - TODO: implement v2
app.include_router(market_data.router, prefix="/api/v1")
app.include_router(signals.router, prefix="/api/v1")
app.include_router(agents.router, prefix="/api/v1")
app.include_router(ai_analysis.router, prefix="/api/v1")
app.include_router(chart_analysis.router, prefix="/api/v1")  # AI Chart Analysis
app.include_router(options_flow.router, prefix="/api/v1")  # Options Flow Intelligence
app.include_router(universal_data.router, prefix="/api/v1")  # Universal Market Data
app.include_router(meta_consensus.router, prefix="/api/v1")  # Meta Signal Orchestrator
app.include_router(ml_training.router, prefix="/api/v1")  # ML Training Pipeline
app.include_router(gamma_exposure.router, prefix="/api/v1")  # Gamma Exposure Agent
app.include_router(market_regime.router, prefix="/api/v1")  # Market Regime Agent
app.include_router(smart_execution.router, prefix="/api/v1")  # Smart Execution Agent
app.include_router(liquidity_prediction.router, prefix="/api/v1")  # Liquidity Prediction Agent
app.include_router(risk_management.router, prefix="/api/v1")  # Risk Management Agent
app.include_router(volatility.router, prefix="/api/v1")  # Volatility Agent
app.include_router(unified_dashboard.router, prefix="/api/v1")  # Unified Trading Dashboard
app.include_router(alerts.router)  # Alert Management System (uses its own prefix)
app.include_router(ml.router)  # Machine Learning Pipeline (uses its own prefix)
app.include_router(government_data.router)  # Government Economic Data APIs
app.include_router(alternative_data.router)  # Alternative Data Sources (News, Social, Weather, Commodities)

# Include WebSocket routers
app.include_router(market_ws.router)  # Market data WebSocket endpoints

# WebSocket endpoint
from fastapi import WebSocket, WebSocketDisconnect
from services.websocket_manager import ws_manager

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    client_id = await ws_manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            await ws_manager.handle_message(client_id, data)
    except WebSocketDisconnect:
        await ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
