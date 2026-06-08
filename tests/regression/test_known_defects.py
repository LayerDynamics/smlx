"""Regression tests for the deceptive/wrong-code defects fixed in the audit pass.

Each test pins a specific fix so the defect cannot silently return. Tests are kept
fast and free of model downloads; where verifying full behavior would require a
heavy model, a source-level guard asserts the deceptive construct stays gone.

These avoid the mx.metal.* deprecation paths on purpose because pytest.ini sets
``filterwarnings = error`` (a stray DeprecationWarning would fail the test).
"""

from __future__ import annotations

import pathlib

import mlx.core as mx
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SMLX = REPO_ROOT / "smlx"


# ---------------------------------------------------------------------------
# 1. Invalid-UTF-8 source files
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_all_source_files_are_valid_utf8():
    """Every smlx/**/*.py must decode as UTF-8 (the bare-Latin-1 © regression)."""
    bad = []
    for path in SMLX.rglob("*.py"):
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            bad.append(str(path.relative_to(REPO_ROOT)))
    assert not bad, f"Files with invalid UTF-8: {bad}"


@pytest.mark.unit
def test_previously_broken_modules_import():
    """Modules that used to fail to import/compile now load cleanly."""
    import importlib

    for mod in (
        "smlx.utils.trace",
        "smlx.kv_cache.rope",
        "smlx.gym.envs.classic.cartpole",
        "smlx.gym.envs.classic.lunar_lander",
    ):
        importlib.import_module(mod)


# ---------------------------------------------------------------------------
# 2. Orphaned gym router is registered
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_gym_router_registered_in_app():
    """smlx.server.app must mount the gym router so /v1/gym/* is reachable."""
    src = (SMLX / "server" / "app.py").read_text()
    assert "include_router(gym.router" in src, "gym router not wired into app.py"


# ---------------------------------------------------------------------------
# 5. convert2mlx really quantizes (not a no-op)
# ---------------------------------------------------------------------------
def _fake_llm_weights():
    return {
        "model.embed_tokens.weight": mx.random.normal((256, 128)).astype(mx.float32),
        "model.layers.0.self_attn.q_proj.weight": mx.random.normal((128, 128)).astype(mx.float32),
        "model.layers.0.self_attn.v_proj.weight": mx.random.normal((128, 128)).astype(mx.float32),
        "model.layers.0.mlp.down_proj.weight": mx.random.normal((128, 256)).astype(mx.float32),
        "model.layers.0.input_layernorm.weight": mx.random.normal((128,)).astype(mx.float32),
        "lm_head.weight": mx.random.normal((256, 128)).astype(mx.float32),
        "model.layers.0.odd.weight": mx.random.normal((128, 100)).astype(mx.float32),
    }


def _nbytes(d):
    return sum(v.size * v.dtype.size for v in d.values())


@pytest.mark.unit
def test_convert2mlx_quantize_model_reduces_size_and_packs():
    from smlx.tools.convert2mlx import quantize_model

    weights = _fake_llm_weights()
    orig = _nbytes(weights)
    qw, qcfg = quantize_model(weights, {"num_hidden_layers": 1}, group_size=64, bits=4)

    # Real packing: weight becomes uint32 with scales+biases siblings.
    qk = "model.layers.0.self_attn.q_proj"
    assert qw[f"{qk}.weight"].dtype == mx.uint32
    assert f"{qk}.scales" in qw and f"{qk}.biases" in qw
    # 1D and non-divisible weights are left untouched.
    assert qw["model.layers.0.input_layernorm.weight"].dtype == mx.float32
    assert "model.layers.0.odd.scales" not in qw
    # Genuine memory savings (~4x for 4-bit).
    assert _nbytes(qw) < orig * 0.4
    assert qcfg["quantization"] == {"group_size": 64, "bits": 4}
    assert qk in qcfg["quantization_layers"]


@pytest.mark.unit
def test_convert2mlx_quantize_dequantize_roundtrip():
    from smlx.tools.convert2mlx import dequantize_model, quantize_model

    weights = _fake_llm_weights()
    qw, qcfg = quantize_model(weights, {"num_hidden_layers": 1}, group_size=64, bits=4)
    dw, dcfg = dequantize_model(qw, qcfg)
    assert dw["lm_head.weight"].shape == (256, 128)
    assert "quantization" not in dcfg


@pytest.mark.unit
def test_convert2mlx_mixed_bit_applies_per_layer_widths():
    from smlx.tools.convert2mlx import quantize_model_mixed

    weights = _fake_llm_weights()
    _, mcfg = quantize_model_mixed(weights, {"num_hidden_layers": 1}, "mixed_4_6", group_size=64)
    layers = mcfg["quantization_layers"]
    # Recipe: lm_head / v_proj / down_proj get high (6) bits, others low (4).
    assert layers["lm_head"] == 6
    assert layers["model.layers.0.self_attn.v_proj"] == 6
    assert layers["model.layers.0.self_attn.q_proj"] == 4


@pytest.mark.unit
def test_convert2mlx_infers_layer_count_from_keys():
    """Mixed-bit with no num_hidden_layers must infer it (exercises the `re` path)."""
    from smlx.tools.convert2mlx import _infer_num_layers, quantize_model_mixed

    weights = {
        "model.layers.0.self_attn.q_proj.weight": mx.random.normal((128, 128)).astype(mx.float32),
        "model.layers.5.mlp.down_proj.weight": mx.random.normal((128, 256)).astype(mx.float32),
        "lm_head.weight": mx.random.normal((256, 128)).astype(mx.float32),
    }
    assert _infer_num_layers(weights, {}) == 6  # max index 5 + 1
    # Must not raise NameError on `re` when config lacks num_hidden_layers.
    _, cfg = quantize_model_mixed(weights, {}, "mixed_4_6", group_size=64)
    assert cfg["quantization"]["recipe"] == "mixed_4_6"


# ---------------------------------------------------------------------------
# 8. autoquant MXFP probe must not unconditionally force ocp_microscaling=True
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_autoquant_does_not_force_microscaling_when_mxfp_unsupported(monkeypatch):
    import importlib

    # smlx.quant re-exports an `autoquant` function that shadows the submodule
    # attribute, so resolve the module explicitly.
    aq = importlib.import_module("smlx.quant.autoquant")

    real_quantize = mx.quantize

    def fake_quantize(*args, **kwargs):
        # Simulate a build where MXFP modes are unavailable.
        if kwargs.get("mode") in ("mxfp4", "mxfp8"):
            raise ValueError("mxfp not supported")
        return real_quantize(*args, **kwargs)

    # autoquant uses `mlx.core as mx`; patching the shared module object suffices.
    monkeypatch.setattr(mx, "quantize", fake_quantize)
    caps = aq.detect_hardware_capabilities()
    # With MXFP unsupported, microscaling must be reported False (the old bug
    # forced it True via a plain INT4 probe that always succeeds).
    assert caps["ocp_microscaling"] is False
    assert caps["supports_mxfp4"] is False
    assert caps["supports_mxfp8"] is False


# ---------------------------------------------------------------------------
# 9. DWQ dead refinement loop is gone
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dwq_has_no_dead_refinement_loop():
    src = (SMLX / "quant" / "dwq.py").read_text()
    assert "unused for now" not in src
    # The real gradient refinement is the Step 4 path.
    assert "gradient-based" in src


# ---------------------------------------------------------------------------
# 10. RL trainer _evaluate runs real episodes without disturbing training state
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_rl_evaluate_runs_real_episodes():
    gym = pytest.importorskip("gymnasium")
    from smlx.gym.algorithms.base import AgentConfig, RLAgent

    class RandomAgent(RLAgent):
        def select_action(self, observation):
            return self.env.action_space.sample()

        def train_step(self, batch):
            return {"loss": 0.0}

        def save(self, path):
            pass

        def load(self, path):
            pass

    env = gym.make("CartPole-v1")
    try:
        agent = RandomAgent(env, AgentConfig(eval_episodes=3, max_steps_per_episode=25, seed=0))
        steps_before = agent.total_steps
        result = agent._evaluate()
        assert result["eval_episodes"] == 3.0
        assert result["eval_average_return"] > 0
        # Evaluation must not pollute training counters/windows.
        assert agent.total_steps == steps_before
        assert agent._episode_returns == []
        assert agent.eval_mode is False
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 11. GGML model quantizers are honest about no runtime savings
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_ggml_quantizers_disclose_no_runtime_savings():
    for fname in ("q8_0.py", "q4_1.py", "q4_k_m.py", "q6_k.py"):
        src = (SMLX / "quant" / fname).read_text().lower()
        assert "no runtime memory savings" in src or "not reduce runtime memory" in src, fname


# ---------------------------------------------------------------------------
# 17. bench: no fabricated param count; manual loop stops on EOS
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_quant_bench_no_hardcoded_param_count():
    src = (SMLX / "bench" / "suites" / "quantization.py").read_text()
    assert "135_000_000" not in src
    # And the prefill/decode split is no longer a hardcoded 0.2/0.8 ratio.
    assert "total_time * 0.2" not in src


@pytest.mark.unit
def test_manual_generate_loop_stops_on_eos():
    from smlx.bench.suites.llm import _manual_generate_loop

    class FakeModel:
        eos_token_id = 7

        def __call__(self, toks):
            v = mx.zeros((len(toks), 10))
            v[-1, 7] = 100.0  # force EOS as argmax
            return v

    out = _manual_generate_loop(FakeModel(), [1, 2, 3], max_tokens=50, temperature=0.0)
    assert out[-1] == 7
    assert len(out) == 4  # 3 prompt + 1 generated EOS, stopped early


# ---------------------------------------------------------------------------
# 18. CLI is wired up as a console script entry point
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_cli_entry_point_configured_and_importable():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'smlx = "smlx.main:main"' in pyproject
    from smlx.main import cli, main

    assert callable(main)
    # The documented commands exist.
    assert set(cli.commands) >= {"generate", "server", "bench", "convert", "download", "transcribe"}
