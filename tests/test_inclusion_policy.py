"""Tests for the performance-based model inclusion policy.

Encodes the decision that zoo membership is gated by performance (memory + speed
+ correctness) on the M4 target, not by a hard parameter-count cap — so a larger
model that fits memory and runs/works is admitted as a performance exception.
"""

from __future__ import annotations

import pytest

from smlx.config.inclusion_policy import (
    DEFAULT_GATES,
    Comparison,
    InclusionGates,
    SpeedFloor,
    evaluate_inclusion,
)


@pytest.mark.unit
def test_large_model_admitted_when_it_fits_and_works():
    """A ~1.8B VLM (e.g. Moondream2) is NOT excluded by size: if its peak memory
    fits the budget and it passes the real-output smoke, it qualifies."""
    v = evaluate_inclusion(
        "Moondream2-1.8B",
        "vlm",
        peak_memory_gb=4.5,  # ~1.8B fp16 + activations — well under the 30GB ceiling
        smoke_passed=True,
        measured_speed=12.0,  # tok/s (speed floor uncalibrated -> provisional pass)
    )
    assert v.memory_ok is True
    assert v.correctness_ok is True
    assert v.qualifies is True


@pytest.mark.unit
def test_excluded_when_smoke_fails():
    """Fitting memory is not enough — a model that fails the real-output smoke
    (placeholder/garbled output) is excluded regardless of size."""
    v = evaluate_inclusion(
        "TinyLLaVA-1.5B",
        "vlm",
        peak_memory_gb=3.6,
        smoke_passed=False,  # currently echoes the prompt
        measured_speed=10.0,
    )
    assert v.memory_ok is True
    assert v.correctness_ok is False
    assert v.qualifies is False


@pytest.mark.unit
def test_excluded_when_over_memory_budget():
    v = evaluate_inclusion(
        "Huge-7B",
        "vlm",
        peak_memory_gb=40.0,  # exceeds 36 - 6 = 30GB ceiling
        smoke_passed=True,
        measured_speed=5.0,
    )
    assert v.memory_ok is False
    assert v.qualifies is False


@pytest.mark.unit
def test_calibrated_speed_floor_enforced():
    """Once a speed floor is calibrated (WS-3), a too-slow model is excluded."""
    gates = InclusionGates(
        speed_floors={
            "language": SpeedFloor("decode_tokens_per_s", "tok/s", Comparison.AT_LEAST, floor=20.0)
        }
    )
    slow = evaluate_inclusion(
        "Slow-LM",
        "language",
        peak_memory_gb=1.0,
        smoke_passed=True,
        measured_speed=8.0,
        gates=gates,
    )
    fast = evaluate_inclusion(
        "Fast-LM",
        "language",
        peak_memory_gb=1.0,
        smoke_passed=True,
        measured_speed=50.0,
        gates=gates,
    )
    assert slow.speed_ok is False and slow.qualifies is False
    assert fast.speed_ok is True and fast.qualifies is True


@pytest.mark.unit
def test_memory_ceiling_is_budget_minus_headroom():
    assert DEFAULT_GATES.memory_ceiling_gb == pytest.approx(30.0)
