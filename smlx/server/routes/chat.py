# Copyright © 2025 SMLX Project

"""
Chat completion endpoints (OpenAI compatible).
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies import get_model_manager
from ..model_manager import ModelManager
from ..schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FinishReason,
    Message,
    Role,
    Usage,
)

router = APIRouter()


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Create a chat completion.

    OpenAI-compatible endpoint for chat-based generation.
    """
    try:
        # Load model
        bm = await manager.load_model(request.model)

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(bm, request),
                media_type="text/event-stream",
            )

        # Non-streaming chat completion
        response_text = await generate_chat_response(
            bm=bm,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Calculate usage
        prompt_text = format_messages_as_prompt(request.messages)
        prompt_tokens = len(bm.processor.encode(prompt_text))
        completion_tokens = len(bm.processor.encode(response_text))

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content=response_text),
                    finish_reason=FinishReason.STOP,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def format_messages_as_prompt(messages: list[Message]) -> str:
    """
    Format chat messages into a prompt string.

    Uses a simple format: <role>: <content>
    """
    prompt_parts = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == Role.USER:
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == Role.ASSISTANT:
            prompt_parts.append(f"Assistant: {msg.content}")

    # Add assistant prefix for response
    prompt_parts.append("Assistant:")

    return "\n\n".join(prompt_parts)


def _chat_prompt(bm, messages: list[Message]) -> str:
    message_dicts = [{"role": msg.role.value, "content": msg.content} for msg in messages]
    return bm.processor.apply_chat_template(message_dicts, add_generation_prompt=True)


async def generate_chat_response(
    bm,
    messages: list[Message],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate a chat response through mlx-lm."""
    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = _chat_prompt(bm, messages)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: lm_generate(
            bm.model, bm.processor, prompt, max_tokens=max_tokens, sampler=sampler, verbose=False
        ),
    )
    return result.strip()


async def stream_chat_completion(bm, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens as SSE (mlx-lm)."""
    try:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = _chat_prompt(bm, request.messages)
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)

        for resp in stream_generate(
            bm.model, bm.processor, prompt, max_tokens=request.max_tokens, sampler=sampler
        ):
            token = resp.text
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token, "role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
