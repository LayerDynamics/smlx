"""Unified model backend — correct upstream MLX implementations + SMLX value-add.

SMLX does NOT re-implement model forward passes (that is a per-model bug farm).
Instead it loads and runs each model through the maintained, correct MLX
implementation — ``mlx-lm`` for language models and ``mlx-vlm`` for
vision-language models — and layers SMLX's actual differentiators on top:
quantization (:mod:`smlx.quant`), a curated "smol" zoo with one consistent API,
and the bench/eval/verify trust layer.

This module is that backend: a curated registry routes a model to the right
upstream loader, ``load`` returns a uniform :class:`BackendModel`, and
``generate`` exposes one text/VLM generation call regardless of backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from PIL import Image


class Backend(str, Enum):
    """Which upstream library runs a model's forward pass."""

    MLX_LM = "mlx_lm"  # language models (llama-family, etc.)
    MLX_VLM = "mlx_vlm"  # vision-language models


@dataclass(frozen=True)
class ZooEntry:
    """A curated model: its HF repo, modality, and backend."""

    key: str
    repo: str
    modality: str  # "language" | "vlm"
    backend: Backend
    params: str  # human-readable size, informational only (perf-based inclusion)


# Curated "smol" zoo. Keys are short aliases; repos are the canonical MLX-ready
# checkpoints. Membership is by performance (see smlx.config.inclusion_policy),
# not a hard parameter cap — params here are informational.
ZOO: dict[str, ZooEntry] = {
    # --- Language models (mlx-lm) ---
    "smollm2-135m": ZooEntry(
        "smollm2-135m", "mlx-community/SmolLM2-135M-Instruct", "language", Backend.MLX_LM, "135M"
    ),
    "smollm2-360m": ZooEntry(
        "smollm2-360m", "mlx-community/SmolLM2-360M-Instruct", "language", Backend.MLX_LM, "360M"
    ),
    "smollm2-1.7b": ZooEntry(
        "smollm2-1.7b", "mlx-community/SmolLM2-1.7B-Instruct", "language", Backend.MLX_LM, "1.7B"
    ),
    "qwen2.5-0.5b": ZooEntry(
        "qwen2.5-0.5b",
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "language",
        Backend.MLX_LM,
        "0.5B",
    ),
    # --- Vision-language models (mlx-vlm) ---
    "smolvlm-256m": ZooEntry(
        "smolvlm-256m", "HuggingFaceTB/SmolVLM-256M-Instruct", "vlm", Backend.MLX_VLM, "256M"
    ),
    "smolvlm-500m": ZooEntry(
        "smolvlm-500m", "HuggingFaceTB/SmolVLM-500M-Instruct", "vlm", Backend.MLX_VLM, "500M"
    ),
    "smolvlm2-2.2b": ZooEntry(
        "smolvlm2-2.2b", "mlx-community/SmolVLM2-2.2B-Instruct-4bit", "vlm", Backend.MLX_VLM, "2.2B"
    ),
    "qwen2-vl-2b": ZooEntry(
        "qwen2-vl-2b", "mlx-community/Qwen2-VL-2B-Instruct-4bit", "vlm", Backend.MLX_VLM, "2B"
    ),
}

# VLM architecture hints for auto-detecting an unregistered repo id.
_VLM_HINTS = (
    "vlm",
    "llava",
    "moondream",
    "idefics",
    "paligemma",
    "pixtral",
    "florence",
    "qwen2-vl",
    "qwen2.5-vl",
    "internvl",
    "smolvlm",
    "nanovlm",
    "aya-vision",
    "gemma-3",
)


@dataclass
class BackendModel:
    """A loaded model plus the metadata needed to run it uniformly."""

    model: Any
    processor: Any  # tokenizer (LM) or processor (VLM)
    backend: Backend
    repo: str
    config: Any | None = None  # mlx-vlm config (needed for chat templating)
    quantized: bool = False


def resolve(model_id: str) -> tuple[str, Backend | None, str | None]:
    """Resolve an alias/repo to (repo, backend, modality).

    A zoo alias yields its full entry; a raw HF repo id is auto-classified by
    architecture hints (backend/modality may be None if undetermined, in which
    case the caller must pass an explicit backend).
    """
    entry = ZOO.get(model_id.lower())
    if entry is not None:
        return entry.repo, entry.backend, entry.modality
    low = model_id.lower()
    if any(h in low for h in _VLM_HINTS):
        return model_id, Backend.MLX_VLM, "vlm"
    return model_id, Backend.MLX_LM, "language"


def load(
    model_id: str,
    *,
    backend: Backend | None = None,
    quantize: str | None = None,
    lazy: bool = False,
) -> BackendModel:
    """Load a model through its correct upstream MLX implementation.

    Args:
        model_id: A zoo alias (e.g. "smolvlm-256m") or an HF repo id.
        backend: Force a backend; otherwise auto-detected.
        quantize: Optional SMLX quant preset ("4bit", "8bit", "gptq", "awq",
            "dwq") applied to the loaded model via :mod:`smlx.quant`.
        lazy: Pass-through lazy weight loading.

    Returns:
        A :class:`BackendModel`.
    """
    repo, detected, _modality = resolve(model_id)
    backend = backend or detected or Backend.MLX_LM

    if backend is Backend.MLX_VLM:
        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config

        model, processor = vlm_load(repo, lazy=lazy)
        config = load_config(repo)
        bm = BackendModel(model, processor, backend, repo, config=config)
    else:
        from mlx_lm import load as lm_load

        model, tokenizer = lm_load(repo)
        bm = BackendModel(model, tokenizer, backend, repo)

    if quantize:
        apply_quantization(bm, quantize)

    return bm


def apply_quantization(bm: BackendModel, preset: str) -> BackendModel:
    """Apply an SMLX quantization preset to a backend-loaded model in place.

    This is SMLX's value-add over the upstream backends: the same
    :mod:`smlx.quant` routines (4/8-bit, GPTQ, AWQ, DWQ) that work on any MLX
    module, now applied to correctly-implemented upstream models.
    """
    from smlx.quant import quantize_4bit, quantize_8bit

    preset = preset.lower()
    if preset in ("4bit", "int4", "4"):
        quantize_4bit(bm.model)
    elif preset in ("8bit", "int8", "8"):
        quantize_8bit(bm.model)
    else:
        # GPTQ/AWQ/DWQ need calibration data; route through the high-level helper
        # so callers get a clear error if they omit it rather than a silent no-op.
        raise ValueError(
            f"Quant preset {preset!r} needs calibration data; call smlx.quant."
            f"{preset}_quantize directly with calibration_data, or use '4bit'/'8bit'."
        )
    bm.quantized = True
    return bm


def generate(
    bm: BackendModel,
    prompt: str,
    *,
    image: str | Image.Image | list | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    verbose: bool = False,
) -> str:
    """Generate text from a backend model (LM or VLM) with one uniform call."""
    if bm.backend is Backend.MLX_VLM:
        return _generate_vlm(bm, prompt, image, max_tokens, temperature, verbose)
    return _generate_lm(bm, prompt, max_tokens, temperature, verbose)


def _generate_lm(bm, prompt, max_tokens, temperature, verbose) -> str:
    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    tokenizer = bm.processor
    # Apply the instruct chat template when available.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
    sampler = make_sampler(temp=temperature)
    return lm_generate(
        bm.model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler, verbose=verbose
    )


def _generate_vlm(bm, prompt, image, max_tokens, temperature, verbose) -> str:
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    if image is None:
        images: list = []
    elif isinstance(image, list):
        images = image
    else:
        images = [image]
    # mlx-vlm accepts file paths or PIL images; normalise PIL to RGB.
    norm_images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in images]

    formatted = apply_chat_template(bm.processor, bm.config, prompt, num_images=len(norm_images))
    result = vlm_generate(
        bm.model,
        bm.processor,
        formatted,
        image=norm_images,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
    )
    # mlx-vlm returns a GenerationResult (.text) on newer versions, a str on older.
    return getattr(result, "text", result)
