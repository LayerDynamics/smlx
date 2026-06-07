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
# #2 — server audio route must call transcribe(audio=...), not audio_path=...
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_server_transcribe_uses_audio_kwarg():
    """`transcribe_audio` previously called transcribe(audio_path=...), leaving
    the required `audio` argument unbound (TypeError). An autospec'd mock of the
    real transcribe enforces its signature, so the bound call must include
    `audio=` and never `audio_path=`."""
    import smlx.server.routes.audio as audio_mod
    from smlx.models.Whisper_tiny import transcribe as real_transcribe

    spec = mock.create_autospec(
        real_transcribe,
        return_value={"text": "hello", "language": "en", "duration": 1.0, "segments": []},
    )
    with mock.patch("smlx.models.Whisper_tiny.transcribe", spec):
        out = await audio_mod.transcribe_audio(
            model=object(),
            tokenizer=object(),
            audio_bytes=b"RIFFxxxxWAVE",
            language="en",
            prompt=None,
            temperature=0.0,
        )

    assert out["text"] == "hello"
    kwargs = spec.call_args.kwargs
    assert "audio" in kwargs and "audio_path" not in kwargs


# ---------------------------------------------------------------------------
# #3 — quant load presets must call quant fns with all required args
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("loader_mod", ["SmolLM2_135M", "SmolLM2_360M"])
@pytest.mark.parametrize("preset", ["gptq", "awq", "dwq"])
def test_quant_loader_presets_call_with_required_args(loader_mod, preset):
    """`_apply_quantization` previously called awq/dwq/gptq quantizers without
    their required `awq_config` / `calibration_data`. Autospec'd quantizers
    enforce the real signatures, so a mis-bound call raises TypeError."""
    import importlib

    import smlx.quant as quant
    import smlx.quant.utils as quant_utils
    from smlx.quant.awq import awq_quantize as real_awq
    from smlx.quant.dwq import dwq_quantize_simple as real_dwq
    from smlx.quant.gptq import gptq_quantize as real_gptq

    loader = importlib.import_module(f"smlx.models.{loader_mod}.loader")

    sentinel_model = object()
    specs = {
        "gptq_quantize": mock.create_autospec(real_gptq, return_value=sentinel_model),
        "awq_quantize": mock.create_autospec(real_awq, return_value=sentinel_model),
        "dwq_quantize_simple": mock.create_autospec(real_dwq, return_value=sentinel_model),
    }
    with (
        mock.patch.object(quant_utils, "load_calibration_data", return_value=object()),
        mock.patch.object(quant, "gptq_quantize", specs["gptq_quantize"]),
        mock.patch.object(quant, "awq_quantize", specs["awq_quantize"]),
        mock.patch.object(quant, "dwq_quantize_simple", specs["dwq_quantize_simple"]),
    ):
        # tokenizer is unused once load_calibration_data is mocked
        result = loader._apply_quantization(object(), object(), preset, None)

    assert result is sentinel_model
    if preset == "awq":
        # awq_config is the required arg that was missing before the fix
        called = specs["awq_quantize"].call_args
        bound = called.kwargs
        assert "awq_config" in bound and bound["awq_config"] is not None


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
def test_tinyllava_language_model_class_name():
    """cache.py's TYPE_CHECKING import aliased a non-existent `TinyLlama`; the
    real class is TinyLlamaModel."""
    from smlx.models.TinyLLaVA.language import TinyLlamaModel

    assert TinyLlamaModel is not None


@pytest.mark.unit
def test_base_dataset_protocol_declares_process():
    """All concrete datasets implement process(); the Protocol now declares it."""
    from smlx.data.datasets import BaseDataset

    assert hasattr(BaseDataset, "process")
