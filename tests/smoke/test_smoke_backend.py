"""Smoke tests for the unified upstream backend (the architecture pivot).

These prove SMLX's real value proposition end to end: models run through the
correct upstream MLX implementations (mlx-lm / mlx-vlm) via one unified API, and
SMLX's quantization applies on top — no hand-reimplemented forward passes.
"""

from __future__ import annotations

import pytest

from tests.smoke import assertions as A

pytestmark = [pytest.mark.smoke, pytest.mark.requires_model]


def test_backend_lm_real_text():
    from smlx.models import mlx_backend as B

    lm = B.load("smollm2-135m")
    out = B.generate(lm, "Name three primary colors.", max_tokens=32)
    A.assert_text_coherent(out, context="backend mlx-lm")


def test_backend_vlm_describes_giraffes(giraffe_image_path):
    from smlx.models import mlx_backend as B

    vlm = B.load("smolvlm-256m")
    out = B.generate(vlm, "What is in this image?", image=str(giraffe_image_path), max_tokens=40)
    A.assert_text_coherent(out, context="backend mlx-vlm")
    A.assert_contains_any(out, ["giraffe", "giraffes", "animal", "tree"], context="backend vlm")


def test_backend_quantized_lm_stays_correct():
    """SMLX's differentiator: quantize a correctly-loaded upstream model and it
    still produces correct output."""
    from smlx.models import mlx_backend as B

    lm = B.load("smollm2-360m", quantize="4bit")
    assert lm.quantized is True
    out = B.generate(lm, "What is the capital of France?", max_tokens=24)
    A.assert_text_coherent(out, context="backend 4bit")
    A.assert_contains_any(out, ["paris"], context="backend 4bit answer")
