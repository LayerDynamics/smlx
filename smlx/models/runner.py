"""Unified model runner — produce real output from every implemented SMLX model.

SMLX ships 18 model packages across 9 modalities. The curated :mod:`mlx_backend`
zoo runs a verified subset (LM/VLM/ASR/embeddings) through upstream MLX impls.
This module adds one consistent way to *run any of them* — including the legacy
custom implementations (TTS, OCR, VAD, audio-classification, CAD, hand-written
VLMs) — and get real output: text, audio, labels, boxes, embeddings, CAD.

Design:
- A registry (:data:`REGISTRY`) maps a short alias to a :class:`RunEntry` whose
  ``runner`` adapter calls that model's *own real* inference function (no
  reimplementation).
- :func:`produce` loads the model, runs the adapter, writes any non-text artifact
  to ``data/output/``, and returns a :class:`RunResult`.
- Output honesty is first-class: every result carries a runtime-derived
  :class:`WeightStatus` (``TRAINED`` / ``TRAINED-WEIGHTS`` gap / ``PIPELINE-ONLY``
  / ``NO-WEIGHTS``). ``ok=True`` means *the pipeline produced output*, NOT that the
  output is trained-quality — the status field carries quality. Nothing prints a
  fake "success" for an untrained pipeline.

The adapters and registry live in :mod:`smlx.models.runner_adapters` (WS-B); this
module is the framework they plug into.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# Inputs the runner understands; an entry declares which it needs.
INPUT_KINDS = ("text", "image", "audio", "document")

DEFAULT_OUTPUT_DIR = "data/output"


class WeightStatus(str, Enum):
    """How trustworthy a run's output is, derived from real load signals."""

    TRAINED = "TRAINED"  # real public weights -> meaningful output
    TRAINED_GAP = "TRAINED-WEIGHTS"  # real weights but a known defect (partial/repetitive)
    PIPELINE_ONLY = "PIPELINE-ONLY"  # random/partial weights: output happens, not meaningful
    NO_WEIGHTS = "NO-WEIGHTS"  # could not load any weights at all
    SKIPPED = "SKIPPED"  # not run: a required input was not supplied
    ERROR = "ERROR"  # the pipeline raised


@dataclass
class RunOutput:
    """What a model adapter returns: the real output plus honest status.

    Exactly one payload field is populated according to ``kind``. ``status`` and
    ``reason`` are determined by the adapter from the model's real load signals.
    """

    kind: str  # text|audio|labels|boxes|segments|embeddings|cad
    status: WeightStatus
    reason: str = ""
    text: str | None = None
    audio: tuple | None = None  # (np.ndarray waveform, sample_rate)
    data: Any = None  # labels / boxes / segments / embeddings / cad payload


@dataclass
class RunResult:
    """Outcome of running one model through :func:`produce`."""

    model: str
    modality: str
    status: WeightStatus
    reason: str
    output_repr: str  # human-readable summary of the output
    ok: bool  # pipeline produced output (NOT a quality judgement)
    elapsed_s: float
    artifact_path: str | None = None  # data/output/<model>.{wav,json,py} for non-text
    error: str | None = None  # set (and ok=False) when the pipeline raised

    def status_line(self) -> str:
        """One-line CLI rendering: model, status(+reason), output/artifact."""
        tag = self.status.value
        detail = f" ({self.reason})" if self.reason else ""
        if self.error:
            body = f"ERROR: {self.error}"
        elif self.artifact_path:
            body = f"-> {self.artifact_path}  [{self.output_repr}]"
        else:
            body = self.output_repr
        return f"{self.model:<22} {tag + detail:<46} {body}"


@dataclass(frozen=True)
class RunEntry:
    """A runnable model: its modality, required inputs, loader, and adapter."""

    key: str
    modality: str  # language|vlm|asr|tts|ocr|embeddings|vad|audio_cls|cad
    needs: tuple  # subset of INPUT_KINDS required to run, e.g. ("image", "text")
    loader: Callable[[], Any]  # returns the model's native load() result
    runner: Callable[
        ..., RunOutput
    ]  # (loaded, *, text, image, audio, document, **opts) -> RunOutput
    note: str = ""  # short human description


# Populated by smlx.models.runner_adapters at import time.
REGISTRY: dict[str, RunEntry] = {}

# Process-lifetime cache of loaded models, keyed by registry alias, so repeated
# runs (and `--all`) don't reload the same heavy model.
_LOAD_CACHE: dict[str, Any] = {}


def _ensure_registry() -> None:
    """Import the adapters module so REGISTRY is populated (idempotent)."""
    if not REGISTRY:
        from . import runner_adapters  # noqa: F401  (import for side effect: registration)


def register(entry: RunEntry) -> None:
    """Add an entry to the global registry (called by the adapters module)."""
    REGISTRY[entry.key] = entry


def list_entries() -> list[RunEntry]:
    """All registered run entries, sorted by modality then key."""
    _ensure_registry()
    return sorted(REGISTRY.values(), key=lambda e: (e.modality, e.key))


def load_cached(key: str) -> Any:
    """Load (and cache) a model by registry alias."""
    _ensure_registry()
    if key not in REGISTRY:
        raise KeyError(f"Unknown model {key!r}. Try: smlx run --list")
    if key not in _LOAD_CACHE:
        _LOAD_CACHE[key] = REGISTRY[key].loader()
    return _LOAD_CACHE[key]


# --------------------------------------------------------------------------- #
# Artifact writers — turn a RunOutput payload into a real file on disk.        #
# --------------------------------------------------------------------------- #


def _write_audio(path: Path, audio_tuple: tuple) -> str:
    import numpy as np
    import soundfile as sf

    waveform, sample_rate = audio_tuple
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    sf.write(str(path), arr, int(sample_rate))
    dur = len(arr) / float(sample_rate) if sample_rate else 0.0
    return f"audio {dur:.2f}s ({len(arr)} samples @ {int(sample_rate)}Hz)"


def _write_json(path: Path, payload: Any) -> str:
    text = json.dumps(payload, indent=2, default=str)
    path.write_text(text)
    n = len(payload) if isinstance(payload, (list, dict)) else 1
    return f"{n} item(s) -> json"


def _write_cad(path_base: Path, payload: dict) -> tuple[str, str]:
    """Write CAD as both .json (sequence) and .py (CadQuery). Returns (json_path, repr)."""
    seq = payload.get("sequence_json")
    py = payload.get("python")
    json_path = path_base.with_suffix(".json")
    json_path.write_text(seq if isinstance(seq, str) else json.dumps(seq, indent=2, default=str))
    if py:
        path_base.with_suffix(".py").write_text(py)
    n = payload.get("n_commands", "?")
    return str(json_path), f"CAD sequence ({n} commands) + CadQuery .py"


# --------------------------------------------------------------------------- #
# The dispatcher.                                                              #
# --------------------------------------------------------------------------- #


def produce(
    model_id: str,
    *,
    text: str | None = None,
    image: str | None = None,
    audio: Any | None = None,
    document: str | None = None,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    **opts: Any,
) -> RunResult:
    """Run one model's real inference pipeline and return an honest result.

    Args:
        model_id: A registry alias (see :func:`list_entries` / ``smlx run --list``).
        text/image/audio/document: Per-modality inputs; only those in the entry's
            ``needs`` are required.
        out_dir: Where non-text artifacts (wav/json/py) are written.
        **opts: Passed through to the adapter (e.g. max_tokens, temperature).

    Returns:
        A :class:`RunResult`. ``ok`` reflects whether the pipeline produced output;
        ``status`` reflects whether that output is trustworthy.
    """
    import time

    _ensure_registry()
    if model_id not in REGISTRY:
        raise KeyError(f"Unknown model {model_id!r}. Try: smlx run --list")
    entry = REGISTRY[model_id]

    provided = {"text": text, "image": image, "audio": audio, "document": document}
    missing = [k for k in entry.needs if provided.get(k) in (None, "")]
    if missing:
        return RunResult(
            model=entry.key,
            modality=entry.modality,
            status=WeightStatus.SKIPPED,
            reason=f"needs {', '.join(missing)}",
            output_repr="-",
            ok=False,
            elapsed_s=0.0,
        )

    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        loaded = load_cached(entry.key)
        result = entry.runner(
            loaded, text=text, image=image, audio=audio, document=document, **opts
        )
    except Exception as e:  # surface the real blocker; never swallow it
        return RunResult(
            model=entry.key,
            modality=entry.modality,
            status=WeightStatus.ERROR,
            reason="pipeline error",
            output_repr="",
            ok=False,
            elapsed_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )

    elapsed = time.perf_counter() - t0
    artifact_path: str | None = None
    base = out_path_dir / entry.key

    if result.kind == "text":
        output_repr = " ".join((result.text or "").split())[:80] or "(empty)"
    elif result.kind == "audio":
        path = base.with_suffix(".wav")
        output_repr = _write_audio(path, result.audio)
        artifact_path = str(path)
    elif result.kind == "cad":
        artifact_path, output_repr = _write_cad(base, result.data)
    elif result.kind in ("labels", "boxes", "segments"):
        path = base.with_suffix(".json")
        output_repr = _write_json(path, result.data)
        artifact_path = str(path)
    elif result.kind == "embeddings":
        import numpy as np

        arr = np.asarray(result.data)
        path = base.with_suffix(".json")
        # Save a compact, honest summary (shape + first vector head), not the full
        # matrix, so the artifact stays readable.
        head = arr.reshape(arr.shape[0], -1)[0, :8].tolist() if arr.size else []
        _write_json(path, {"shape": list(arr.shape), "first_vector_head": head})
        output_repr = f"embeddings shape {tuple(arr.shape)}"
        artifact_path = str(path)
    else:
        output_repr = f"<{result.kind}>"

    return RunResult(
        model=entry.key,
        modality=entry.modality,
        status=result.status,
        reason=result.reason,
        output_repr=output_repr,
        ok=True,
        elapsed_s=elapsed,
        artifact_path=artifact_path,
    )


def produce_all(
    *,
    text: str | None = None,
    image: str | None = None,
    audio: Any | None = None,
    document: str | None = None,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    only_modalities: set | None = None,
    **opts: Any,
) -> list[RunResult]:
    """Run every registered model whose required inputs are satisfied.

    Models missing a required input are returned as SKIP results (ok=False,
    output_repr starting with "SKIP"); they are not errors.
    """
    _ensure_registry()
    results: list[RunResult] = []
    for entry in list_entries():
        if only_modalities and entry.modality not in only_modalities:
            continue
        results.append(
            produce(
                entry.key,
                text=text,
                image=image,
                audio=audio,
                document=document,
                out_dir=out_dir,
                **opts,
            )
        )
    return results


# Helper for adapters: classify weights from a "loaded ok" + "has gap" signal.
def classify(
    *, loaded: bool, gap: bool = False, gap_reason: str = "", untrained_reason: str = ""
) -> tuple[WeightStatus, str]:
    """Map real load signals to a (WeightStatus, reason)."""
    if not loaded:
        return WeightStatus.PIPELINE_ONLY, untrained_reason or "random weights"
    if gap:
        return WeightStatus.TRAINED_GAP, gap_reason
    return WeightStatus.TRAINED, ""


__all__ = [
    "WeightStatus",
    "RunOutput",
    "RunResult",
    "RunEntry",
    "REGISTRY",
    "register",
    "list_entries",
    "load_cached",
    "produce",
    "produce_all",
    "classify",
    "INPUT_KINDS",
]
