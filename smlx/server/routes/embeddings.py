# Copyright © 2025 SMLX Project

"""
Embedding endpoints (OpenAI compatible).
"""


from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_model_manager
from ..model_manager import ModelManager
from ..schemas import Embedding, EmbeddingRequest, EmbeddingResponse, Usage

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """
    Create embeddings for input text(s).

    OpenAI-compatible endpoint for text embeddings.
    """
    try:
        # Load embedding model
        model, tokenizer = await manager.load_model(request.model, model_type="embedding")

        # Handle single or multiple inputs
        inputs = [request.input] if isinstance(request.input, str) else request.input

        # Generate embeddings
        embeddings_data = []
        total_tokens = 0

        for idx, text in enumerate(inputs):
            # Get embedding
            embedding_vector = await generate_embedding(
                model=model,
                tokenizer=tokenizer,
                text=text,
            )

            embeddings_data.append(
                Embedding(
                    embedding=embedding_vector.tolist(),
                    index=idx,
                )
            )

            # Count tokens
            total_tokens += len(tokenizer.encode(text))

        return EmbeddingResponse(
            data=embeddings_data,
            model=request.model,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            ),
        )

    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Embedding models not yet implemented. Coming soon!",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_embedding(model, tokenizer, text: str):
    """
    Generate embedding for text.

    Args:
        model: Loaded embedding model
        tokenizer: Tokenizer
        text: Input text

    Returns:
        Embedding vector as numpy array
    """
    import asyncio

    from smlx.models.MiniLM import encode_single

    # Run encoding in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None, lambda: encode_single(model, tokenizer, text, normalize=True)
    )

    return embedding
