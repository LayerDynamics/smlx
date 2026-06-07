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

# The complete set of implemented model packages the runner must cover.
EXPECTED_KEYS = {
    "smollm2-135m",
    "smollm2-360m",
    "smolvlm-256m",
    "smolvlm-500m",
    "nanovlm",
    "moondream2",
    "tinyllava",
    "whisper-tiny",
    "orpheus-150m",
    "chatterbox",
    "trocr-small",
    "donut-base",
    "minilm",
    "all-minilm-l6-v2",
    "silero-vad",
    "yamnet",
    "smolgencad",
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


# --------------------------------------------------------------------------- #
# requires_model — actually run pipelines and assert output happens.          #
# --------------------------------------------------------------------------- #


@pytest.mark.requires_model
def test_produce_cad_pipeline_output_happens(tmp_path):
    """smolGenCad has no public weights -> PIPELINE-ONLY, but output must happen."""
    r = runner.produce("smolgencad", text="Create a cylinder radius 5", out_dir=str(tmp_path))
    assert r.ok is True
    assert r.status is WeightStatus.PIPELINE_ONLY
    assert r.artifact_path and r.artifact_path.endswith(".json")
    import os

    assert os.path.exists(r.artifact_path)
    assert os.path.exists(r.artifact_path.replace(".json", ".py"))  # CadQuery python too


@pytest.mark.requires_model
def test_produce_tts_pipeline_output_happens(tmp_path):
    """Orpheus has random weights -> PIPELINE-ONLY noise, but a wav must be written."""
    r = runner.produce("orpheus-150m", text="Hello world", out_dir=str(tmp_path))
    assert r.ok is True
    assert r.status is WeightStatus.PIPELINE_ONLY
    assert r.artifact_path and r.artifact_path.endswith(".wav")
    import os

    assert os.path.exists(r.artifact_path)


@pytest.mark.requires_model
def test_produce_lm_trained_text(tmp_path):
    r = runner.produce("smollm2-135m", text="In one sentence, what is MLX?", out_dir=str(tmp_path))
    assert r.ok is True
    assert r.status is WeightStatus.TRAINED
    assert len(r.output_repr.split()) >= 4  # real, multi-word answer


@pytest.mark.requires_model
def test_produce_embeddings_trained(tmp_path):
    r = runner.produce("minilm", text="a cat sleeps", out_dir=str(tmp_path))
    assert r.ok is True
    assert r.status is WeightStatus.TRAINED
    assert "embeddings shape" in r.output_repr
