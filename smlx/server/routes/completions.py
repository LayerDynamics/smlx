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
        bm = await manager.load_model(request.model)

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_completion(bm, request),
                media_type="text/event-stream",
            )

        # Non-streaming completion
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        choices = []

        for idx, prompt in enumerate(prompts):
            # Generate text
            generated_text = await generate_text(
                bm=bm,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
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
        prompt_tokens = sum(len(bm.processor.encode(p)) for p in prompts)
        completion_tokens = sum(len(bm.processor.encode(c.text)) for c in choices)

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
    bm,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate raw-text completion through mlx-lm."""
    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: lm_generate(
            bm.model, bm.processor, prompt, max_tokens=max_tokens, sampler=sampler, verbose=False
        ),
    )


async def stream_completion(bm, request: CompletionRequest) -> AsyncGenerator[str, None]:
    """Stream completion tokens as SSE (mlx-lm)."""
    try:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        completion_id = f"cmpl-{uuid.uuid4()}"
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)

        for resp in stream_generate(
            bm.model, bm.processor, prompt, max_tokens=request.max_tokens, sampler=sampler
        ):
            token = resp.text
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
