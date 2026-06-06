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

            # Load model based on type
            if model_type == "smollm":
                model, tokenizer = await self._load_smollm(model_id)
            elif model_type == "whisper":
                model, tokenizer = await self._load_whisper(model_id)
            elif model_type == "embedding":
                model, tokenizer = await self._load_embedding(model_id)
            elif model_type == "vlm":
                model, tokenizer = await self._load_vlm(model_id)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Cache management
            if len(self.loaded_models) >= self.cache_size:
                # Remove least recently used model
                lru_model_id = next(iter(self.loaded_models))
                print(f"🗑️  Evicting model from cache: {lru_model_id}")
                del self.loaded_models[lru_model_id]
                del self.model_types[lru_model_id]

            # Cache the model
            self.loaded_models[model_id] = (model, tokenizer)
            self.model_types[model_id] = model_type

            print(f"✅ Model loaded: {model_id}")

            return model, tokenizer

    async def _load_smollm(self, model_id: str) -> tuple[Any, Any]:
        """Load SmolLM model with optional quantization."""
        # Normalize model ID
        if model_id == "SmolLM2-135M":
            model_id = "mlx-community/SmolLM2-135M-Instruct"
        elif model_id == "SmolLM2-360M":
            model_id = "mlx-community/SmolLM2-360M-Instruct"

        # Import and load
        if "135M" in model_id:
            from smlx.models.SmolLM2_135M import load
        elif "360M" in model_id:
            from smlx.models.SmolLM2_360M import load
        else:
            raise ValueError(f"Unknown SmolLM variant: {model_id}")

        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        # Apply quantization if auto_quantize is set
        if self.auto_quantize:
            print(f"   Applying {self.auto_quantize} quantization...")
            model, tokenizer = await loop.run_in_executor(
                None,
                lambda: load(
                    model_id,
                    quantize=self.auto_quantize,
                    quantization_config=self.quantization_config,
                ),
            )
        else:
            model, tokenizer = await loop.run_in_executor(None, load, model_id)

        return model, tokenizer

    async def _load_whisper(self, model_id: str) -> tuple[Any, Any]:
        """Load Whisper model."""
        from smlx.models.Whisper_tiny import load

        loop = asyncio.get_event_loop()
        model, tokenizer = await loop.run_in_executor(None, load, model_id)

        return model, tokenizer

    async def _load_embedding(self, model_id: str) -> tuple[Any, Any]:
        """Load embedding model (MiniLM-based sentence transformers)."""
        # Normalize model ID to supported variants
        if model_id in ["all-MiniLM-L6-v2", "minilm", "embedding"]:
            variant = "all-MiniLM-L6-v2"
        elif "minilm" in model_id.lower():
            # Use the model_id as-is (might be a HuggingFace path)
            variant = model_id
        else:
            # Assume it's a sentence-transformers model
            variant = model_id

        from smlx.models.MiniLM import load

        loop = asyncio.get_event_loop()
        model, tokenizer = await loop.run_in_executor(None, load, variant)

        return model, tokenizer

    async def _load_vlm(self, model_id: str) -> tuple[Any, Any]:
        """Load vision-language model (returns model, processor)."""
        model_id_lower = model_id.lower()

        # Determine which VLM to load based on model ID
        if "smolvlm-256m" in model_id_lower or model_id == "SmolVLM-256M":
            from smlx.models.SmolVLM_256M import load

            default_repo = "HuggingFaceTB/SmolVLM-256M-Instruct"
        elif "smolvlm-500m" in model_id_lower or model_id == "SmolVLM-500M":
            from smlx.models.SmolVLM_500M_Instruct import load

            default_repo = "HuggingFaceTB/SmolVLM-500M-Instruct"
        elif "nanovlm" in model_id_lower:
            from smlx.models.nanoVLM import load

            default_repo = "lusxvr/nanoVLM-222M"
        elif "moondream" in model_id_lower:
            from smlx.models.Moondream2 import load

            default_repo = "vikhyatk/moondream2"
        elif "tinyllava" in model_id_lower:
            from smlx.models.TinyLLaVA import load

            default_repo = "bczhou/TinyLLaVA-1.5B"
        else:
            raise ValueError(
                f"Unknown VLM model: {model_id}. "
                f"Supported: SmolVLM-256M, SmolVLM-500M, nanoVLM, Moondream2, TinyLLaVA"
            )

        # Use full HF path if provided, otherwise use default
        repo = model_id if "/" in model_id else default_repo

        # Load model in thread pool
        loop = asyncio.get_event_loop()
        model, processor = await loop.run_in_executor(None, load, repo)

        return model, processor

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
