# Copyright © 2025 SMLX Project

"""
Model Manager for SMLX Server.

Handles loading, caching, and lifecycle management of MLX models.
Supports multiple model types: language models, vision-language, audio, embeddings.
"""

import asyncio
from typing import Any, Literal, Optional

import mlx.core as mx

QuantizePreset = Literal["auto", "4bit", "8bit", "gptq", "awq", "dwq"]


class ModelManager:
    """
    Manages loading and caching of MLX models.

    Features:
    - Lazy loading: Models loaded on first request
    - Caching: Keep models in memory for fast inference
    - Multi-model support: Handle different model types
    - Resource management: Cleanup on shutdown
    - Automatic quantization: Load models with quantization for memory efficiency
    """

    def __init__(
        self,
        cache_size: int = 3,
        auto_quantize: Optional[QuantizePreset] = None,
        quantization_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize model manager.

        Args:
            cache_size: Maximum number of models to keep in cache
            auto_quantize: Automatically quantize models on load (None, "auto", "4bit", etc.)
            quantization_config: Configuration dict for quantization
        """
        self.cache_size = cache_size
        self.auto_quantize = auto_quantize
        self.quantization_config = quantization_config or {}
        self.loaded_models: dict[str, tuple[Any, Any]] = {}  # model_id -> (model, tokenizer)
        self.model_types: dict[str, str] = {}  # model_id -> model_type
        self.load_lock = asyncio.Lock()

        # Supported model mappings
        self.supported_models = {
            # Language Models
            "mlx-community/SmolLM2-135M-Instruct": "smollm",
            "mlx-community/SmolLM2-360M-Instruct": "smollm",
            "SmolLM2-135M": "smollm",
            "SmolLM2-360M": "smollm",
            # Audio Models
            "whisper-tiny": "whisper",
            "mlx-community/whisper-tiny": "whisper",
            # Embedding Models (for future)
            "all-MiniLM-L6-v2": "embedding",
            "minilm": "embedding",
            # VLMs (for future)
            "SmolVLM-256M": "vlm",
            "SmolVLM-500M": "vlm",
        }

    async def load_model(self, model_id: str, model_type: Optional[str] = None) -> tuple[Any, Any]:
        """
        Load a model and tokenizer.

        Args:
            model_id: Model identifier (HuggingFace ID or alias)
            model_type: Optional explicit model type

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ValueError: If model type is not supported
        """
        # Check if already loaded
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        async with self.load_lock:
            # Double-check after acquiring lock
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]

            # Determine model type
            if model_type is None:
                model_type = self._infer_model_type(model_id)

            print(f"📦 Loading model: {model_id} (type: {model_type})")

            bm = await self._load_backend(model_id, model_type)

            # Cache management (LRU eviction)
            if len(self.loaded_models) >= self.cache_size:
                lru_model_id = next(iter(self.loaded_models))
                del self.loaded_models[lru_model_id]
                del self.model_types[lru_model_id]

            self.loaded_models[model_id] = bm
            self.model_types[model_id] = model_type
            print(f"✅ Model loaded: {model_id}")
            return bm

    async def _load_backend(self, model_id: str, model_type: str):
        from smlx.models import mlx_backend as B

        if model_type == "whisper":
            repo, backend = "whisper-tiny", None
        elif model_type == "embedding":
            repo, backend = "minilm", None
        elif model_type == "smollm":
            repo = {
                "SmolLM2-135M": "mlx-community/SmolLM2-135M-Instruct",
                "SmolLM2-360M": "mlx-community/SmolLM2-360M-Instruct",
            }.get(model_id, model_id)
            backend = B.Backend.MLX_LM
        elif model_type == "vlm":
            repo, backend = model_id, B.Backend.MLX_VLM
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        quantize = self.auto_quantize if self.auto_quantize in ("4bit", "8bit") else None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: B.load(repo, backend=backend, quantize=quantize)
        )

    def _infer_model_type(self, model_id: str) -> str:
        """
        Infer model type from model ID.

        Args:
            model_id: Model identifier

        Returns:
            Model type string

        Raises:
            ValueError: If model type cannot be inferred
        """
        model_id_lower = model_id.lower()

        # Check known models
        if model_id in self.supported_models:
            return self.supported_models[model_id]

        # Infer from ID
        if "smollm" in model_id_lower or "135m" in model_id_lower or "360m" in model_id_lower:
            return "smollm"
        elif "whisper" in model_id_lower:
            return "whisper"
        elif "minilm" in model_id_lower or "embedding" in model_id_lower:
            return "embedding"
        elif "vlm" in model_id_lower or "vision" in model_id_lower:
            return "vlm"
        else:
            raise ValueError(f"Cannot infer model type for: {model_id}")

    def get_model(self, model_id: str) -> Optional[tuple[Any, Any]]:
        """
        Get a cached model without loading.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, tokenizer) if cached, None otherwise
        """
        return self.loaded_models.get(model_id)

    def list_loaded_models(self) -> list[str]:
        """
        List currently loaded models.

        Returns:
            List of model IDs
        """
        return list(self.loaded_models.keys())

    def list_supported_models(self) -> dict[str, str]:
        """
        List all supported models.

        Returns:
            Dictionary mapping model IDs to model types
        """
        return self.supported_models.copy()

    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from cache.

        Args:
            model_id: Model identifier

        Returns:
            True if model was unloaded, False if not found
        """
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            del self.model_types[model_id]
            print(f"🗑️  Unloaded model: {model_id}")
            return True
        return False

    async def cleanup(self):
        """Cleanup all loaded models."""
        print("🧹 Cleaning up models...")
        self.loaded_models.clear()
        self.model_types.clear()
        # Force garbage collection
        mx.clear_cache()
        print("✅ Cleanup complete")
