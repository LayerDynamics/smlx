"""WS-9: the fail-closed gate covers the non-runner public surfaces too.

`smlx run --verify` verifies every runner registry entry; these tests pin the
*additional* coverage of the other public ways to run a model — the legacy
`load_model()` API, the `mlx_backend` ASR/embeddings entrypoints, and the server
audio/embeddings route handlers.

The unit tests are hermetic (no model download) — they assert the coverage is
wired and that `load_model` is fail-closed on removed aliases. The requires_model
test runs the public-path checks end to end (real models, real correctness).
"""

from __future__ import annotations

import inspect
import tempfile

import pytest


@pytest.mark.unit
def test_gate_exposes_public_path_coverage():
    """A full `verify()` run includes the public-path checks by default."""
    from smlx.models import runner_verify

    assert hasattr(runner_verify, "_check_public_paths")
    assert "include_public_paths" in inspect.signature(runner_verify.verify).parameters


@pytest.mark.unit
def test_load_model_is_fail_closed_on_removed_aliases():
    """Removed bespoke aliases report not-implemented (never resolve to noise)."""
    from smlx.models import is_model_implemented

    for removed in (
        "chatterbox",
        "orpheus-150m",
        "trocr-small",
        "donut-base",
        "yamnet",
        "smolgencad",
    ):
        assert not is_model_implemented(removed), removed
    # Real entries still resolve.
    assert is_model_implemented("smollm2-135m")
    assert is_model_implemented("whisper-tiny")


@pytest.mark.unit
def test_server_route_handlers_are_plain_callables():
    """The audio/embeddings handlers the gate drives need no FastAPI app/TestClient."""
    from smlx.server.routes.audio import transcribe_audio
    from smlx.server.routes.embeddings import generate_embedding

    assert callable(transcribe_audio)
    assert callable(generate_embedding)


@pytest.mark.requires_model
def test_public_path_checks_all_pass():
    """End-to-end: load_model, mlx_backend ASR/embeddings, and the server
    audio/embeddings handlers each produce real, correct output."""
    from smlx.models import runner_verify

    with tempfile.TemporaryDirectory() as tmp:
        fx = runner_verify._build_fixtures(tmp)
        results = runner_verify._check_public_paths(fx)

    assert results, "no public-path checks ran"
    expected = {
        ("load_model", "language"),
        ("load_model", "fail-closed"),
        ("mlx_backend", "asr"),
        ("mlx_backend", "embeddings"),
        ("server:audio", "asr"),
        ("server:embeddings", "embeddings"),
    }
    assert {(r.model, r.modality) for r in results} == expected
    failed = [(r.model, r.modality, r.detail) for r in results if not r.ok]
    assert not failed, f"public-path checks failed: {failed}"
