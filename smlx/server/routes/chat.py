# Copyright © 2025 SMLX Project

"""
Chat completion endpoints (OpenAI compatible).
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from smlx.models.smlx_router import get_router
from smlx.models.registry import infer_model_type

from ..dependencies import get_model_manager
from ..model_manager import ModelManager
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    Message,
    Usage,
    FinishReason,
    Role,
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
        model, tokenizer = await manager.load_model(request.model)

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(model, tokenizer, request),
                media_type="text/event-stream",
            )

        # Non-streaming chat completion
        response_text = await generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            messages=request.messages,
            model_id=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        # Calculate usage
        prompt_text = format_messages_as_prompt(request.messages)
        prompt_tokens = len(tokenizer.encode(prompt_text))
        completion_tokens = len(tokenizer.encode(response_text))

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


async def generate_chat_response(
    model,
    tokenizer,
    messages: list[Message],
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> str:
    """Generate chat response using the model."""
    # Get router and infer model type
    router = get_router()
    model_type = infer_model_type(model_id)

    if model_type is None:
        raise ValueError(f"Could not infer model type from: {model_id}")

    # Convert messages to dict format for router
    message_dicts = [{"role": msg.role.value, "content": msg.content} for msg in messages]

    # Run generation in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: router.route_chat(
            model_type=model_type,
            model=model,
            tokenizer=tokenizer,
            messages=message_dicts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k or 0,
            verbose=False,
        ),
    )

    # Extract only the assistant's response
    response = result.strip()

    return response


async def stream_chat_completion(
    model, tokenizer, request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens as SSE."""
    try:
        # Get router and infer model type
        router = get_router()
        model_type = infer_model_type(request.model)

        if model_type is None:
            raise ValueError(f"Could not infer model type from: {request.model}")

        # Convert messages to dict format for router
        message_dicts = [{"role": msg.role.value, "content": msg.content} for msg in request.messages]

        # Format prompt for streaming (use router's chat method expects messages)
        prompt = format_messages_as_prompt(request.messages)
        completion_id = f"chatcmpl-{uuid.uuid4()}"

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
            )

        # Get the generator
        token_stream = await loop.run_in_executor(None, generate_tokens)

        # Stream each token
        for token in token_stream:
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
