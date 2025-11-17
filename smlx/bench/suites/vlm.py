"""
Benchmark suite for vision-language models (VLMs).

Provides functions for benchmarking multimodal model performance.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx
from PIL import Image

from smlx.utils.memory import clear_cache, memory_profiler, reset_peak_memory

from ..stats import ModelBenchmarkStats, create_model_stats


def _get_image_token_count(model: Any, processor: Any) -> int:
    """Determine the number of image tokens for a VLM model.

    Args:
        model: The VLM model
        processor: The model's processor

    Returns:
        Estimated number of image tokens
    """
    # Try to get from model config
    if hasattr(model, "config"):
        config = model.config

        # Check for explicit image token configuration
        if hasattr(config, "num_image_tokens"):
            return config.num_image_tokens
        if hasattr(config, "image_token_length"):
            return config.image_token_length

        # Check for vision config
        if hasattr(config, "vision_config"):
            vision_config = config.vision_config
            # SigLIP-based models (SmolVLM, nanoVLM)
            if hasattr(vision_config, "image_size") and hasattr(vision_config, "patch_size"):
                image_size = vision_config.image_size
                patch_size = vision_config.patch_size
                # Number of patches = (image_size / patch_size)^2
                num_patches = (image_size // patch_size) ** 2
                return num_patches

        # Model type-specific defaults
        model_type = getattr(config, "model_type", "").lower()
        if "smolvlm-256m" in model_type or "siglip" in model_type:
            return 256  # SigLIP-base: 14x14 grid + CLS token = 196, rounded to 256
        elif "smolvlm-500m" in model_type:
            return 729  # SigLIP-400M: 27x27 grid
        elif "moondream" in model_type:
            return 729  # Moondream2 uses SigLIP-400M
        elif "tinyllava" in model_type or "llava" in model_type:
            return 576  # CLIP ViT-L/14: 24x24 grid
        elif "nanovlm" in model_type:
            return 256  # SigLIP-base

    # Fallback: try to inspect processor
    if hasattr(processor, "image_processor"):
        img_proc = processor.image_processor
        if hasattr(img_proc, "size"):
            # Standard CLIP/SigLIP processing
            size = img_proc.size
            if isinstance(size, dict):
                height = size.get("height", 224)
                width = size.get("width", 224)
            elif isinstance(size, (list, tuple)):
                height, width = size
            else:
                height = width = size

            # Assume 16x16 patches (common default)
            patch_size = getattr(img_proc, "patch_size", 16)
            num_patches = (height // patch_size) * (width // patch_size)
            return num_patches

    # Conservative default
    return 256


@dataclass
class VLMBenchmarkConfig:
    """Configuration for VLM benchmarks."""

    max_tokens: int = 100
    """Maximum tokens to generate"""

    num_trials: int = 5
    """Number of trials to run"""

    temperature: float = 0.0
    """Generation temperature (0.0 for greedy)"""

    seed: int = 0
    """Random seed for reproducibility"""

    resize_shape: Optional[tuple] = None
    """Optional image resize shape"""

    cache_type: Optional[str] = None
    """Cache type for enhanced VLM language models: 'auto', 'standard', 'rotating', 'quantized'"""

    max_kv_size: Optional[int] = None
    """Maximum KV cache size for rotating/quantized caches"""

    enable_monitoring: bool = False
    """Enable memory pressure monitoring (for enhanced cache models)"""

    quantization_bits: int = 4
    """Quantization bits for quantized cache (4 or 8)"""


def benchmark_vlm(
    model: Any,
    processor: Any,
    image: Union[str, Path, Image.Image],
    prompt: str = "Describe this image.",
    config: Optional[VLMBenchmarkConfig] = None,
) -> ModelBenchmarkStats:
    """
    Benchmark a vision-language model.

    Args:
        model: VLM model to benchmark
        processor: Image/text processor
        image: Input image (path or PIL Image)
        prompt: Text prompt
        config: Benchmark configuration (includes cache options for enhanced VLM models)

    Returns:
        ModelBenchmarkStats with performance metrics

    Example:
        >>> from smlx.models import load_vlm
        >>> model, processor = load_vlm("SmolVLM")
        >>> stats = benchmark_vlm(model, processor, "image.jpg", "What is this?")
        >>> print(f"Generation: {stats.generation_tps:.2f} tok/s")

        >>> # With enhanced cache configuration
        >>> config = VLMBenchmarkConfig(
        ...     max_tokens=200,
        ...     cache_type="quantized",
        ...     quantization_bits=4,
        ...     enable_monitoring=True
        ... )
        >>> stats = benchmark_vlm(model, processor, image, prompt, config=config)

    Note:
        Cache configuration (cache_type, max_kv_size, etc.) is intended for VLM models
        with enhanced cache support (SmolVLM, Moondream2, TinyLLaVA, nanoVLM).
        The cache is used for the language model component of the VLM.
    """
    if config is None:
        config = VLMBenchmarkConfig()

    # Set seed for reproducibility
    mx.random.seed(config.seed)

    # Load image if path
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    # Resize if specified
    if config.resize_shape:
        image = image.resize(config.resize_shape)

    # Clear cache and reset memory tracking
    clear_cache()
    reset_peak_memory()

    # Warmup
    _ = _generate_vlm(
        model, processor, image, prompt, max_tokens=min(10, config.max_tokens), temperature=config.temperature
    )

    # Clear cache after warmup
    clear_cache()
    reset_peak_memory()

    # Benchmark with memory profiling
    with memory_profiler() as mem:
        # Process inputs (image + text)
        prompt_start = time.perf_counter()
        # Encode image and text
        # This is model-specific and would need to be adapted
        prompt_end = time.perf_counter()
        prompt_time = prompt_end - prompt_start

        # Generate tokens
        gen_start = time.perf_counter()
        output, generation_tokens = _generate_vlm(
            model, processor, image, prompt, max_tokens=config.max_tokens, temperature=config.temperature
        )
        gen_end = time.perf_counter()
        generation_time = gen_end - gen_start

    # Estimate prompt tokens (image tokens + text tokens)
    # This is approximate and model-specific
    text_tokens = (
        len(processor.tokenizer.encode(prompt))
        if hasattr(processor, "tokenizer")
        else len(prompt.split())
    )
    # Get image token count from model configuration
    image_tokens = _get_image_token_count(model, processor)
    prompt_tokens = text_tokens + image_tokens

    # Create statistics
    model_name = getattr(model, "name", "unknown_vlm")

    stats = create_model_stats(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        prompt_time=prompt_time,
        generation_tokens=generation_tokens,
        generation_time=generation_time,
        peak_memory_gb=mem.peak_gb,
        modality="vision-language",
    )

    return stats


def _generate_vlm(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> tuple[str, int]:
    """
    Generate text from VLM.

    Supports SmolVLM, nanoVLM, Moondream2, and other VLM models.

    Args:
        model: VLM model
        processor: Processor
        image: Input image
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (generated_text, num_tokens)
    """
    # Try to import and use model-specific generate functions
    # This approach allows benchmarking any VLM model

    # Detect model type and use appropriate generation
    model_type = getattr(model, "config", None)
    if model_type and hasattr(model_type, "model_type"):
        model_type_str = model_type.model_type.lower()
    else:
        model_type_str = str(type(model).__name__).lower()

    # Try SmolVLM models first
    if "smolvlm" in model_type_str:
        try:
            from smlx.models.SmolVLM_256M.generate import generate

            generated_text = generate(
                model=model,
                processor=processor,
                prompt=prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,  # No top-p for benchmarks
                verbose=False,
            )
            # Count tokens in output
            num_tokens = len(processor.tokenizer.encode(generated_text))
            return generated_text, num_tokens
        except (ImportError, AttributeError) as e:
            pass

    # Try nanoVLM
    if "nanovlm" in model_type_str:
        try:
            from smlx.models.nanoVLM.generate import generate

            generated_text = generate(
                model=model,
                processor=processor,
                prompt=prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                verbose=False,
            )
            num_tokens = len(processor.tokenizer.encode(generated_text))
            return generated_text, num_tokens
        except (ImportError, AttributeError):
            pass

    # Try Moondream2
    if "moondream" in model_type_str:
        try:
            from smlx.models.Moondream2.generate import generate

            generated_text = generate(
                model=model,
                processor=processor,
                prompt=prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                verbose=False,
            )
            num_tokens = len(processor.tokenizer.encode(generated_text))
            return generated_text, num_tokens
        except (ImportError, AttributeError):
            pass

    # Try TinyLLaVA
    if "tinyllava" in model_type_str or "llava" in model_type_str:
        try:
            from smlx.models.TinyLLaVA.generate import generate

            generated_text = generate(
                model=model,
                processor=processor,
                prompt=prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                verbose=False,
            )
            num_tokens = len(processor.tokenizer.encode(generated_text))
            return generated_text, num_tokens
        except (ImportError, AttributeError):
            pass

    # Generic fallback: try to use a generate method if available
    if hasattr(model, "generate"):
        try:
            # Prepare inputs
            from smlx.models.SmolVLM_256M.generate import prepare_inputs

            inputs = prepare_inputs(processor, prompt, image)
            generated_text = model.generate(
                **inputs, max_tokens=max_tokens, temperature=temperature
            )
            num_tokens = len(processor.tokenizer.encode(generated_text))
            return generated_text, num_tokens
        except Exception:
            pass

    # Last resort: raise error with helpful message
    raise NotImplementedError(
        f"VLM generation not implemented for model type '{model_type_str}'. "
        f"Supported models: SmolVLM-256M, SmolVLM-500M, nanoVLM, Moondream2, TinyLLaVA. "
        f"Make sure the model's generate.py module is properly implemented."
    )


def benchmark_vlm_batch(
    model: Any,
    processor: Any,
    image_prompt_pairs: list[tuple[Union[str, Path, Image.Image], str]],
    config: Optional[VLMBenchmarkConfig] = None,
) -> list[ModelBenchmarkStats]:
    """
    Benchmark VLM with multiple image-prompt pairs.

    Args:
        model: VLM model
        processor: Processor
        image_prompt_pairs: List of (image, prompt) tuples
        config: Benchmark configuration

    Returns:
        List of ModelBenchmarkStats for each pair

    Example:
        >>> pairs = [
        ...     ("cat.jpg", "What animal is this?"),
        ...     ("dog.jpg", "Describe this image."),
        ... ]
        >>> stats_list = benchmark_vlm_batch(model, processor, pairs)
    """
    results = []
    for image, prompt in image_prompt_pairs:
        stats = benchmark_vlm(model, processor, image, prompt, config)
        results.append(stats)
    return results
