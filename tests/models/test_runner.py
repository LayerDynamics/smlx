"""Tests for the unified model runner (smlx.models.runner).

Two tiers:
- Offline unit tests exercise the registry, the SKIP path, error handling, and
  status rendering WITHOUT loading any model (registration uses lazy closures, so
  importing the runner pulls no heavy deps). These run in CI.
- `requires_model` tests actually run a model's pipeline and assert that output
  *happens* and that the reported WeightStatus matches reality. The CAD/TTS ones
  use random-weight pipelines (offline-capable but marked requires_model because
  they pull tokenizers); the LM/embeddings ones download real checkpoints.
"""

from __future__ import annotations

import pytest

from smlx.models import runner
from smlx.models.runner import WeightStatus

# The verified-real runner registry. Every entry routes to a real upstream impl
# or a real deterministic one; bespoke garbage (Orpheus/Chatterbox/TrOCR/Donut/
# YAMNet/untrained smolGenCad/degenerate moondream2) is quarantined, NOT listed.
EXPECTED_KEYS = {
    "smollm2-135m",
    "smollm2-360m",
    "smolvlm-256m",
    "smolvlm-500m",
    "nanovlm",
    "tinyllava",
    "whisper-tiny",
    "kokoro",
    "ocr",
    "minilm",
    "all-minilm-l6-v2",
    "silero-vad",
    "ast",
    "cad",
}
# Models that must NOT be wired into the runner (they produce garbage).
QUARANTINED = {
    "orpheus-150m",
    "chatterbox",
    "trocr-small",
    "donut-base",
    "yamnet",
    "smolgencad",
    "moondream2",
}
EXPECTED_MODALITIES = {
    "language",
    "vlm",
    "asr",
    "tts",
    "ocr",
    "embeddings",
    "vad",
    "audio_cls",
    "cad",
}


# --------------------------------------------------------------------------- #
# Offline unit tests — no model is loaded.                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_registry_covers_every_model():
    keys = {e.key for e in runner.list_entries()}
    assert keys == EXPECTED_KEYS, f"missing/extra: {keys ^ EXPECTED_KEYS}"


@pytest.mark.unit
def test_every_modality_is_represented():
    mods = {e.modality for e in runner.list_entries()}
    assert mods == EXPECTED_MODALITIES


@pytest.mark.unit
def test_needs_are_valid_input_kinds():
    for e in runner.list_entries():
        assert e.needs, f"{e.key} declares no required inputs"
        for kind in e.needs:
            assert kind in runner.INPUT_KINDS, f"{e.key} needs unknown input {kind!r}"


@pytest.mark.unit
def test_missing_input_is_skipped_not_loaded():
    """A model whose required input is absent returns SKIPPED without loading."""
    # whisper-tiny needs audio; supplying only text must SKIP before any load.
    r = runner.produce("whisper-tiny", text="hello")
    assert r.status is WeightStatus.SKIPPED
    assert r.ok is False
    assert r.error is None
    assert "audio" in r.reason


@pytest.mark.unit
def test_vlm_missing_image_is_skipped():
    r = runner.produce("smolvlm-256m", text="describe")  # no image
    assert r.status is WeightStatus.SKIPPED
    assert r.ok is False
    assert "image" in r.reason


@pytest.mark.unit
def test_unknown_model_raises():
    with pytest.raises(KeyError):
        runner.produce("does-not-exist", text="x")


@pytest.mark.unit
def test_status_line_renders_status_and_output():
    from smlx.models.runner import RunResult

    r = RunResult(
        model="demo",
        modality="language",
        status=WeightStatus.TRAINED,
        reason="",
        output_repr="hello world",
        ok=True,
        elapsed_s=0.1,
    )
    line = r.status_line()
    assert "demo" in line and "TRAINED" in line and "hello world" in line


@pytest.mark.unit
def test_pipeline_error_is_captured_not_raised(monkeypatch):
    """An adapter that raises yields an ERROR result, not an exception."""

    def boom(_loaded, **_kw):
        raise RuntimeError("kaboom")

    entry = runner.REGISTRY["smollm2-135m"]
    monkeypatch.setitem(
        runner.REGISTRY,
        "smollm2-135m",
        runner.RunEntry(entry.key, entry.modality, entry.needs, lambda: object(), boom),
    )
    r = runner.produce("smollm2-135m", text="hi")
    assert r.status is WeightStatus.ERROR
    assert r.ok is False
    assert "kaboom" in (r.error or "")


@pytest.mark.unit
def test_quarantined_models_are_not_wired():
    """The bespoke garbage models must NOT appear in the runner registry."""
    keys = {e.key for e in runner.list_entries()}
    leaked = keys & QUARANTINED
    assert not leaked, f"quarantined models wired into the runner: {leaked}"


@pytest.mark.unit
def test_no_bespoke_forward_imported_by_adapters():
    """No adapter may import a bespoke smlx.models.<Model> forward pass.

    Every entry must route to a real upstream impl (mlx_backend / mlx_whisper /
    mlx_embeddings / mlx_audio / onnxruntime / transformers) or a real
    deterministic module (text_to_cad). This guards against silently
    re-introducing a hand-written forward.
    """
    import pathlib

    src = pathlib.Path(runner.__file__).with_name("runner_adapters.py").read_text()
    bespoke = [
        "from smlx.models.SmolLM2",
        "from smlx.models.Whisper_tiny",
        "from smlx.models.MiniLM",
        "from smlx.models.all_MiniLM",
        "from smlx.models.nanoVLM",
        "from smlx.models.SmolVLM",
        "from smlx.models.TinyLLaVA",
        "from smlx.models.Moondream2",
        "from smlx.models.TrOCR_small",
        "from smlx.models.Donut_base",
        "from smlx.models.YAMNet",
        "from smlx.models.Orpheus",
        "from smlx.models.Chatterbox",
        "from smlx.models.SileroVAD",
    ]
    found = [b for b in bespoke if b in src]
    assert not found, f"adapters import bespoke forwards: {found}"


# --------------------------------------------------------------------------- #
# requires_model — run real pipelines and assert correct output.              #
# --------------------------------------------------------------------------- #


@pytest.mark.requires_model
def test_produce_cad_real_cadquery(tmp_path):
    """`cad` produces real, valid CadQuery with the expected bounding box."""
    r = runner.produce(
        "cad", text="a cylinder with radius 5mm and height 10mm", out_dir=str(tmp_path)
    )
    assert r.ok is True
    assert r.status is WeightStatus.TRAINED
    assert "(10.0, 10.0, 10.0)" in r.output_repr  # verified bbox
    import os

    assert r.artifact_path and os.path.exists(r.artifact_path.replace(".json", ".py"))


@pytest.mark.requires_model
def test_produce_lm_trained_text(tmp_path):
    r = runner.produce("smollm2-135m", text="What is the capital of France?", out_dir=str(tmp_path))
    assert r.ok is True
    assert r.status is WeightStatus.TRAINED
    assert "paris" in r.output_repr.lower()  # real, correct answer


@pytest.mark.requires_model
def test_produce_embeddings_trained(tmp_path):
    r = runner.produce("minilm", text="a cat sleeps", out_dir=str(tmp_path))
    assert r.ok is True
    assert r.status is WeightStatus.TRAINED
    assert "embeddings shape" in r.output_repr


@pytest.mark.requires_model
def test_verify_gate_passes_for_real_models():
    """The fail-closed correctness gate passes for representative real models."""
    from smlx.models import runner_verify

    results, all_ok = runner_verify.verify(["smollm2-135m", "cad", "minilm"])
    assert all_ok, [(r.model, r.detail) for r in results if not r.ok]
