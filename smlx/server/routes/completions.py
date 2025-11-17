# Copyright © 2025 SMLX Project

"""
Text completion endpoints (OpenAI compatible).
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from smlx.models.registry import infer_model_type
from smlx.models.smlx_router import get_router

from ..dependencies import get_model_manager
from ..model_manager import ModelManager
from ..schemas import CompletionChoice, CompletionRequest, CompletionResponse, FinishReason, Usage

router = APIRouter()


@router.post("/completions")
async def create_completion(
    request: CompletionRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Create a text completion.

    OpenAI-compatible endpoint for text generation.
    """
    try:
        # Load model
        model, tokenizer = await manager.load_model(request.model)

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_completion(model, tokenizer, request),
                media_type="text/event-stream",
            )

        # Non-streaming completion
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        choices = []

        for idx, prompt in enumerate(prompts):
            # Generate text
            generated_text = await generate_text(
                model=model,
                tokenizer=tokenizer,
                model_id=request.model,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
            )

            # Add echo if requested
            if request.echo:
                generated_text = prompt + generated_text

            choices.append(
                CompletionChoice(
                    index=idx,
                    text=generated_text,
                    finish_reason=FinishReason.STOP,
                )
            )

        # Calculate usage
        prompt_tokens = sum(len(tokenizer.encode(p)) for p in prompts)
        completion_tokens = sum(len(tokenizer.encode(c.text)) for c in choices)

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_text(
    model,
    tokenizer,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    stop: str | list[str] | None,
) -> str:
    """Generate text using the model."""
    # Get router and infer model type
    router = get_router()
    model_type = infer_model_type(model_id)

    if model_type is None:
        raise ValueError(f"Could not infer model type from: {model_id}")

    # Convert stop to list if it's a string
    stop_strings = None
    if stop:
        stop_strings = [stop] if isinstance(stop, str) else stop

    loop = asyncio.get_event_loop()

    # Run generation in thread pool
    result = await loop.run_in_executor(
        None,
        lambda: router.route_text_generation(
            model_type=model_type,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k or 0,
            stop_strings=stop_strings,
            verbose=False,
        ),
    )

    return result


async def stream_completion(
    model, tokenizer, request: CompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream completion tokens as SSE."""
    try:
        # Get router and infer model type
        router = get_router()
        model_type = infer_model_type(request.model)

        if model_type is None:
            raise ValueError(f"Could not infer model type from: {request.model}")

        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        completion_id = f"cmpl-{uuid.uuid4()}"

        # Convert stop to list if needed
        stop_strings = None
        if request.stop:
            stop_strings = [request.stop] if isinstance(request.stop, str) else request.stop

        # Stream tokens
        loop = asyncio.get_event_loop()

        # Create generator in executor
        def generate_tokens():
            return router.route_streaming_generation(
                model_type=model_type,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k or 0,
                stop_strings=stop_strings,
            )

        # Get the generator
        token_stream = await loop.run_in_executor(None, generate_tokens)

        # Stream each token
        for token in token_stream:
            chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": token,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {chunk}\n\n"

        # Send final chunk
        final_chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {final_chunk}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {error_chunk}\n\n"
