# Copyright © 2025 SMLX Project

"""
Model listing endpoints (OpenAI compatible).
"""

import time

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_model_manager
from ..model_manager import ModelManager
from ..schemas import ModelInfo, ModelList

router = APIRouter()


@router.get("/models", response_model=ModelList)
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """
    List available models.

    Returns all supported models and their metadata.
    """
    supported_models = manager.list_supported_models()

    model_data = []
    for model_id, model_type in supported_models.items():
        model_data.append(
            ModelInfo(
                id=model_id,
                created=int(time.time()),
                owned_by="smlx",
            )
        )

    return ModelList(data=model_data)


@router.get("/models/{model_id:path}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Get information about a specific model.
    """
    supported_models = manager.list_supported_models()

    if model_id not in supported_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelInfo(
        id=model_id,
        created=int(time.time()),
        owned_by="smlx",
    )


@router.delete("/models/{model_id:path}")
async def unload_model(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Unload a model from memory.

    Custom endpoint (not in OpenAI spec) for resource management.
    """
    success = await manager.unload_model(model_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")

    return {"status": "success", "message": f"Model {model_id} unloaded"}
