"""Main application entry point - ARI Processing Server"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.core.logging import setup_logging
from app.infrastructure.mcp.mcp_service import mcp_service
from app.routers.api import router as api_router

# Setup logging
setup_logging()
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting ARI Processing Server...")
    try:
        # Initialize MCP service
        await mcp_service.initialize()
        logger.info("MCP service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP service: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ARI Processing Server...")
    try:
        await mcp_service.shutdown()
        logger.info("MCP service shutdown completed")
    except Exception as e:
        logger.error(f"Error during service shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="ARI Processing Server",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "ari-processing-server"}

# Include routers
app.include_router(api_router, prefix="/api")

# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
