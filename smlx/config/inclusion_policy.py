"""Performance-based model inclusion policy.

SMLX admits a model to the zoo by *performance on the M4 target*, not by a hard
parameter-count cap. A model qualifies iff it passes three gates:

1. **Memory** — peak inference memory fits the unified-memory budget (with
   headroom) on the reference machine.
2. **Speed** — it meets its modality's performance floor (the WS-3 bench gate).
3. **Correctness** — it produces real, correct output (the WS-0 smoke assertion).

Parameter count is a *guideline* (target < 500M, prefer < 1B), not a gate: a
larger model that still fits memory and runs acceptably is admitted as a
documented performance exception (e.g. TinyLLaVA ~1.5B, Moondream2 ~1.8B).

This module is the single source of truth for that decision. WS-3 (bench) and
WS-4 (`smlx models verify`) feed measurements in; ``evaluate_inclusion`` returns
the verdict and the per-gate rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Comparison(str, Enum):
    """How a measured value is compared to its floor/ceiling."""

    AT_LEAST = "at_least"  # measured >= floor (throughput-like: tok/s, embeds/s)
    AT_MOST = "at_most"  # measured <= floor (latency-like: ms, real-time-factor)


@dataclass(frozen=True)
class SpeedFloor:
    """A per-modality performance floor enforced on the reference machine.

    Floors are policy thresholds, calibrated against measured baselines in WS-3;
    until calibrated, ``floor`` is ``None`` (the speed gate then passes on any
    finite measurement and records that it is uncalibrated).
    """

    metric: str  # e.g. "decode_tokens_per_s", "first_token_latency_ms", "real_time_factor"
    unit: str
    comparison: Comparison
    floor: float | None = None  # None => not yet calibrated (WS-3)


@dataclass(frozen=True)
class InclusionGates:
    """The performance gates a model must clear to be included."""

    # Reference machine: M4 with 36 GB unified memory.
    memory_budget_gb: float = 36.0
    # Reserve for the OS, framework, and activation spikes not in steady-state peak.
    memory_headroom_gb: float = 6.0
    # Per-modality speed floors (calibrated in WS-3). Modality keys match the
    # smoke/bench modality labels.
    speed_floors: dict[str, SpeedFloor] = field(
        default_factory=lambda: {
            # Calibrated from the verified zoo on the M4 reference (slowest member,
            # halved with margin so the gate is meaningful but not CI-flaky on a
            # loaded machine). Slowest verified: LM 30 tok/s (SmolLM2-1.7B),
            # VLM 14 tok/s (SmolVLM2-2.2B), embeddings 594 sent/s (MiniLM).
            "language": SpeedFloor("decode_tokens_per_s", "tok/s", Comparison.AT_LEAST, 10.0),
            "vlm": SpeedFloor("decode_tokens_per_s", "tok/s", Comparison.AT_LEAST, 5.0),
            "asr": SpeedFloor("real_time_factor", "xRT", Comparison.AT_MOST),
            "tts": SpeedFloor("real_time_factor", "xRT", Comparison.AT_MOST),
            "embeddings": SpeedFloor("sentences_per_s", "sent/s", Comparison.AT_LEAST, 100.0),
            "ocr": SpeedFloor("images_per_s", "img/s", Comparison.AT_LEAST),
            "audio_cls": SpeedFloor("clips_per_s", "clip/s", Comparison.AT_LEAST),
            "vad": SpeedFloor("real_time_factor", "xRT", Comparison.AT_MOST),
            "cad": SpeedFloor("decode_tokens_per_s", "tok/s", Comparison.AT_LEAST),
        }
    )

    @property
    def memory_ceiling_gb(self) -> float:
        """Maximum allowed peak inference memory."""
        return self.memory_budget_gb - self.memory_headroom_gb


DEFAULT_GATES = InclusionGates()


@dataclass
class InclusionVerdict:
    """Result of evaluating a model against the inclusion gates."""

    model_name: str
    modality: str
    qualifies: bool
    memory_ok: bool
    speed_ok: bool
    correctness_ok: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "INCLUDED" if self.qualifies else "EXCLUDED"
        return f"{self.model_name} [{self.modality}]: {status} — " + "; ".join(self.reasons)


def evaluate_inclusion(
    model_name: str,
    modality: str,
    *,
    peak_memory_gb: float,
    smoke_passed: bool,
    measured_speed: float | None = None,
    gates: InclusionGates = DEFAULT_GATES,
) -> InclusionVerdict:
    """Decide whether a model qualifies for the zoo from real measurements.

    Args:
        model_name: Identifier (e.g. "TinyLLaVA-1.5B").
        modality: One of the keys in ``gates.speed_floors``.
        peak_memory_gb: Measured peak inference memory on the reference machine.
        smoke_passed: Whether the model passed its WS-0 real-output assertion.
        measured_speed: Measured value for the modality's speed metric (the units
            in ``gates.speed_floors[modality]``). ``None`` if not yet measured.
        gates: The gate thresholds to apply.

    Returns:
        An :class:`InclusionVerdict` with per-gate booleans and human reasons.
    """
    reasons: list[str] = []

    # --- Gate 1: memory -------------------------------------------------------
    memory_ok = peak_memory_gb <= gates.memory_ceiling_gb
    reasons.append(
        f"memory {peak_memory_gb:.1f}GB {'<=' if memory_ok else '>'} "
        f"ceiling {gates.memory_ceiling_gb:.1f}GB"
    )

    # --- Gate 2: speed --------------------------------------------------------
    floor = gates.speed_floors.get(modality)
    if floor is None:
        speed_ok = False
        reasons.append(f"unknown modality {modality!r} — no speed floor")
    elif floor.floor is None:
        # Not yet calibrated (WS-3). Require a finite measurement but don't fail
        # on an uncalibrated threshold; record that it's provisional.
        speed_ok = measured_speed is not None
        reasons.append(
            f"speed {floor.metric} uncalibrated (WS-3); " f"measured={measured_speed} {floor.unit}"
            if measured_speed is not None
            else f"speed {floor.metric} not measured"
        )
    elif measured_speed is None:
        speed_ok = False
        reasons.append(f"speed {floor.metric} not measured (floor {floor.floor} {floor.unit})")
    else:
        if floor.comparison is Comparison.AT_LEAST:
            speed_ok = measured_speed >= floor.floor
            op = ">=" if speed_ok else "<"
        else:
            speed_ok = measured_speed <= floor.floor
            op = "<=" if speed_ok else ">"
        reasons.append(f"speed {measured_speed:.2f} {op} floor {floor.floor:.2f} {floor.unit}")

    # --- Gate 3: correctness --------------------------------------------------
    correctness_ok = bool(smoke_passed)
    reasons.append(f"real-output smoke {'passed' if correctness_ok else 'FAILED'}")

    qualifies = memory_ok and speed_ok and correctness_ok
    return InclusionVerdict(
        model_name=model_name,
        modality=modality,
        qualifies=qualifies,
        memory_ok=memory_ok,
        speed_ok=speed_ok,
        correctness_ok=correctness_ok,
        reasons=reasons,
    )
