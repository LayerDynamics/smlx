# Copyright © 2025 SMLX Project

"""
Shared dependencies for FastAPI routes.
"""

from fastapi import HTTPException

from .model_manager import ModelManager


def get_model_manager() -> ModelManager:
    """
    Dependency to get the global model manager instance.

    This centralized dependency can be easily overridden in tests.
    """
    from .app import model_manager

    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    return model_manager
