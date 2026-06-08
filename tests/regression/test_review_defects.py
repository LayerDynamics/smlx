"""Regression tests for the signature/reference defects found in the
2026-06 codebase review (called-but-missing methods and wrong kwargs that only
surfaced on convenience/integration paths bypassed by imports, unit tests, and
demos).

Each test fails on the original bug (ImportError / AttributeError / TypeError)
and passes once the call site is reconciled with the real signature.
"""

from __future__ import annotations

import inspect
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# #1 — `smlx convert` CLI must call the real convert2mlx API, not a missing name
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_convert_calls_real_convert(tmp_path):
    """The `convert` CLI command previously imported a non-existent
    `convert_model` (ImportError on invoke). It must now call convert2mlx.convert
    with mapped arguments."""
    from click.testing import CliRunner

    from smlx.main import cli

    src = tmp_path / "model"
    src.mkdir()
    out = tmp_path / "out"

    with mock.patch("smlx.tools.convert2mlx.convert") as m:
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", str(src), str(out), "--quantize", "8bit"])

    assert result.exit_code == 0, result.output
    m.assert_called_once()
    kwargs = m.call_args.kwargs
    assert kwargs["hf_path"] == str(src)
    assert kwargs["mlx_path"] == str(out)
    assert kwargs["quantize"] is True
    assert kwargs["q_bits"] == 8


@pytest.mark.unit
def test_cli_convert_no_quantize(tmp_path):
    """Without --quantize, convert() is called with quantize=False."""
    from click.testing import CliRunner

    from smlx.main import cli

    src = tmp_path / "model"
    src.mkdir()
    out = tmp_path / "out"

    with mock.patch("smlx.tools.convert2mlx.convert") as m:
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", str(src), str(out)])

    assert result.exit_code == 0, result.output
    assert m.call_args.kwargs["quantize"] is False


# ---------------------------------------------------------------------------
# #4 — KVCacheManager factories must accept the enable_monitoring alias
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory,kwargs",
    [
        ("create_standard", {"num_layers": 2}),
        ("create_rotating", {"num_layers": 2, "max_kv_size": 128}),
        ("create_quantized", {"num_layers": 2}),
        ("create_auto", {"num_layers": 2, "model_size_gb": 0.5}),
    ],
)
def test_kvcache_factories_accept_enable_monitoring_alias(factory, kwargs):
    """The 4 VLM cache modules call these factories with `enable_monitoring=`.
    Only create_standard accepted that alias before the fix; the others raised
    TypeError. All four must now accept it (and the canonical name)."""
    from smlx.kv_cache.kv_manager import KVCacheManager

    fn = getattr(KVCacheManager, factory)
    assert fn(enable_monitoring=True, **kwargs) is not None
    assert fn(enable_monitoring=False, **kwargs) is not None
    # Canonical name still works too.
    assert fn(enable_memory_monitoring=True, **kwargs) is not None


# ---------------------------------------------------------------------------
# #5 — smlx.utils.load_csv / save_csv must use csv.DictReader / DictWriter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_utils_csv_roundtrip(tmp_path):
    """load_csv/save_csv used csv.dictReader/dictWriter (don't exist ->
    AttributeError). A round-trip must now work."""
    from smlx.utils import load_csv, save_csv

    rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
    path = tmp_path / "t.csv"
    save_csv(rows, path)
    assert load_csv(path) == rows


# ---------------------------------------------------------------------------
# #6 — stream_generate must accept verbose; chat(stream=True) forwards it
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stream_generate_accepts_verbose():
    """chat(stream=True) forwards verbose= to stream_generate, which previously
    had no such parameter (TypeError)."""
    from smlx.utils.generation import stream_generate

    assert "verbose" in inspect.signature(stream_generate).parameters


@pytest.mark.unit
def test_chat_stream_forwards_verbose_without_typeerror():
    import smlx.utils.generation as gen

    captured = {}

    def fake_stream(**kwargs):
        captured.update(kwargs)
        yield "hi"

    with mock.patch.object(gen, "stream_generate", fake_stream):
        out = gen.chat(
            model=object(),
            tokenizer=object(),  # no apply_chat_template -> fallback formatting
            messages=[{"role": "user", "content": "x"}],
            stream=True,
            verbose=True,
        )
        assert list(out) == ["hi"]

    assert captured.get("verbose") is True


# ---------------------------------------------------------------------------
# #7 — minor latent: correct type references
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_base_dataset_protocol_declares_process():
    """All concrete datasets implement process(); the Protocol now declares it."""
    from smlx.data.datasets import BaseDataset

    assert hasattr(BaseDataset, "process")
