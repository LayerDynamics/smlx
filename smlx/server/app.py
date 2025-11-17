# Copyright (c) 2025 SMLX Project

"""
SMLX Server - FastAPI-based inference server for small MLX models.

Provides OpenAI-compatible API endpoints for text generation, chat, audio transcription,
and embeddings. Optimized for Apple Silicon (M1/M2/M3/M4) with unified memory.

Example:
    Start the server:
    $ python -m smlx.server.app

    Or with uvicorn:
    $ uvicorn smlx.server.app:app --host 0.0.0.0 --port 8000

    Start with automatic quantization:
    $ SMLX_AUTO_QUANTIZE=4bit python -m smlx.server.app

    Start with GPTQ quantization:
    $ SMLX_AUTO_QUANTIZE=gptq SMLX_QUANTIZE_GROUP_SIZE=64 python -m smlx.server.app
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from smlx.config.memory import get_default_config
from smlx.utils.memory import (
    get_active_memory_gb,
    get_cache_memory_gb,
    get_device_info,
    get_peak_memory_gb,
)
from smlx.utils.watchdog import MemoryWatchdog

from .middleware import ErrorHandlingMiddleware, LoggingMiddleware, RateLimitMiddleware
from .model_manager import ModelManager
from .routes import audio, chat, completions, embeddings
from .routes import models as models_route

# Global instances
model_manager: Optional[ModelManager] = None
memory_watchdog: Optional[MemoryWatchdog] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown.

    On startup: Initialize model manager and memory watchdog
    On shutdown: Cleanup resources
    """
    global model_manager, memory_watchdog

    # Startup
    print("=> Starting SMLX Server...")

    # Read quantization configuration from environment
    auto_quantize = os.getenv("SMLX_AUTO_QUANTIZE")
    quantization_config = {}

    if auto_quantize:
        # Validate quantization method
        valid_methods = ["auto", "4bit", "8bit", "gptq", "awq", "dwq"]
        if auto_quantize not in valid_methods:
            print(f"  Warning: Invalid SMLX_AUTO_QUANTIZE value '{auto_quantize}'. "
                  f"Valid options: {', '.join(valid_methods)}")
            auto_quantize = None
        else:
            print(f"  Auto-quantization enabled: {auto_quantize}")

            # Read quantization parameters
            if bits := os.getenv("SMLX_QUANTIZE_BITS"):
                quantization_config["bits"] = int(bits)
            if group_size := os.getenv("SMLX_QUANTIZE_GROUP_SIZE"):
                quantization_config["group_size"] = int(group_size)

            if quantization_config:
                print(f"    Config: {quantization_config}")

    # Initialize model manager with quantization settings
    model_manager = ModelManager(
        auto_quantize=auto_quantize,
        quantization_config=quantization_config,
    )

    # Initialize memory watchdog if enabled
    config = get_default_config()
    if config.watchdog_enabled:
        try:
            memory_watchdog = MemoryWatchdog(
                warning_threshold=config.warning_threshold,
                critical_threshold=config.critical_threshold,
                check_interval=config.watchdog_interval,
                auto_cleanup=config.auto_cleanup,
            )
            memory_watchdog.start()
            print(f"  Memory watchdog enabled (warning={config.warning_threshold:.0%}, "
                  f"critical={config.critical_threshold:.0%})")
        except ImportError as e:
            print(f"  Warning: Memory watchdog disabled - {e}")
            memory_watchdog = None
    else:
        print("  Memory watchdog disabled (set SMLX_WATCHDOG_ENABLED=true to enable)")

    # Optionally preload models here
    # await model_manager.load_model("mlx-community/SmolLM2-135M-Instruct")

    print("  SMLX Server ready!")

    yield

    # Shutdown
    print("=> Shutting down SMLX Server...")

    # Stop watchdog
    if memory_watchdog:
        memory_watchdog.stop()
        print("  Memory watchdog stopped")

    # Cleanup model manager
    if model_manager:
        await model_manager.cleanup()

    print("  Cleanup complete")


# Create FastAPI application
app = FastAPI(
    title="SMLX Server",
    description="OpenAI-compatible inference server for small MLX models",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)


# ============================================================================
# Health Check Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint - server health check."""
    return {
        "name": "SMLX Server",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": len(model_manager.loaded_models) if model_manager else 0,
    }


@app.get("/memory")
async def memory_status():
    """
    Memory status endpoint.

    Returns current memory usage and watchdog status.
    """
    device_info = get_device_info()
    active_gb = get_active_memory_gb()
    cache_gb = get_cache_memory_gb()
    peak_gb = get_peak_memory_gb()
    total_gb = active_gb + cache_gb

    max_gb = device_info['max_recommended_working_set_size_gb']
    utilization = total_gb / max_gb if max_gb > 0 else 0

    return {
        "active_gb": round(active_gb, 2),
        "cache_gb": round(cache_gb, 2),
        "peak_gb": round(peak_gb, 2),
        "total_gb": round(total_gb, 2),
        "max_gb": round(max_gb, 2),
        "utilization": round(utilization, 3),
        "watchdog_enabled": memory_watchdog is not None,
        "timestamp": time.time(),
    }


# ============================================================================
# OpenAI-Compatible API Routes
# ============================================================================

# Include route modules
app.include_router(completions.router, prefix="/v1", tags=["completions"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(audio.router, prefix="/v1", tags=["audio"])
app.include_router(embeddings.router, prefix="/v1", tags=["embeddings"])
app.include_router(models_route.router, prefix="/v1", tags=["models"])


# ============================================================================
# Helper Functions
# ============================================================================


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    return model_manager


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "smlx.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info",
    )
