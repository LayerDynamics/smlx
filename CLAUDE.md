# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMLX (Smol MLX) is a Python package for small, efficient models using Apple's MLX
framework, optimized for M4 chipsets. Focus areas: vision, voice, language, and
multimodal models.

**Critical Requirement — performance-based inclusion (not a hard parameter cap).**
A model qualifies for the SMLX zoo if it meets all three *performance* gates on the
M4 target, regardless of exact parameter count:

1. **Memory**: loads and runs inference within the 36 GB unified-memory budget (with
   headroom), using the presets in `smlx/config/model_profiles.py`.
2. **Speed**: meets its modality's performance floor (WS-3 bench gate — e.g. tok/s,
   first-token latency, RTF).
3. **Correctness**: produces real, correct output (passes the `tests/smoke`
   real-output assertion — no placeholder/noise).

Parameter count is a **guideline, not a gate**: target < 500M, prefer < 1B, but a
larger model (e.g. a ~1.5–1.8B VLM that still fits memory and runs acceptably) is
admitted as a documented **performance exception**. The name "smol" reflects
*on-device efficiency*, not a strict size limit.

> **Architecture pivot (current state).** SMLX no longer hand-implements model
> forward passes. The 17 bespoke model packages under `smlx/models/<Name>/` and the
> old `smlx_router`/`smlx_runner`/`smlx_manager` framework were **removed**; every
> curated model now runs through a maintained upstream library (mlx-lm / mlx-vlm /
> mlx-whisper / mlx-embeddings / mlx-audio / onnxruntime / transformers) or real
> deterministic code, behind the unified runner (`smlx run`, see *Model Execution
> Framework* below). The two "Known/Outstanding Defects" sections that follow are a
> **historical audit trail**: many entries cite files in those now-deleted packages.
> They are retained as a record — do not treat a defect in a removed package as live
> work; the modality it covered is served by a real backend and verified by
> `smlx run --verify` (15/15).

## Known Defects — Deceptive or Wrong Code (RESOLVED)

The items below were found by auditing the actual source (file:line cited) for code,
output, or claims that misrepresented what actually happens, plus things that were
broken/no-op. **All have now been fixed** (checked boxes); regression tests live in
`tests/regression/test_known_defects.py`. A few items have residual *by-design* caveats,
noted inline (e.g. GGML formats give no MLX runtime savings; datasets are LFS-tracked and
absent until pulled). Kept here as an audit trail — if you touch these areas, preserve the
fix and its test.

## Outstanding Defects — found in later review (not all resolved)

A second review pass (after the original 20) surfaced more issues. Tractable ones
were fixed; the rest are documented here with file:line and what they need. Fixed
items have regression coverage in the existing model/quant/utils test suites.

### Status: all resolved

Every item from this second review pass is now fixed (below). No outstanding
defects remain from this audit.

### Fixed in the later review pass

- [x] **GPTQ produced a degenerate model (output collapsed to EOS).**
  `smlx/quant/gptq.py` `gptq_quantize`: the attention `o_proj` inputs have dead
  (all-zero) activation channels, so `H = XᵀX` was singular and the Cholesky inverse
  came back NaN, corrupting those layers' weights — every prompt then emitted only
  EOS. (The per-layer GPTQ math is correct: on well-conditioned data its output
  error is ~2× better than plain quant.) Fixed `_compute_inverse_hessian` with the
  standard dead-feature guard (zero diagonals → 1) **plus** a finite-check fallback
  to a diagonal inverse when the Cholesky is still non-finite (that layer then
  degrades to plain per-group quant instead of NaN). GPTQ now generates coherent
  text. Guarded by `tests/quant/test_gptq.py::TestGPTQDeadFeatures` (unit) and
  `test_output_quality.py::test_gptq_no_gibberish` (real-model, FP-gated).
- [x] **TinyLLaVA multiple image tokens (now batch-correct).** `smlx/models/TinyLLaVA/model.py`
  `prepare_inputs_for_generation` replaces each `<image>` token, in order, with its
  own image's patch features (handles 1 or N images) and validates that the total
  number of `<image>` tokens equals the number of images. The earlier
  `np.where(...)[1]` flattened column positions across the batch, so it was correct
  only for batch size 1; it now groups positions **per row**, assigns images to rows
  in order, and splices each row independently before stacking, so batched prompts
  with images inject correctly. A ragged batch (rows with differing `<image>`-token
  counts → unequal expanded lengths) raises `ValueError` rather than silently
  mis-aligning, since the forward path threads no per-row padding mask. Note: the
  `generate()` decode loop is still single-sequence (`generate.py:284`); this fix
  covers the prompt-stage merge, not batched token-by-token decoding. Covered by
  `tests/models/test_tinyllava.py::TestImageTokenMerge`. (By-design: `loader.py:84`
  `FIXME` is a real upstream config/weights mismatch — HF config says 27 vision
  layers, weights have 26 — and is handled correctly.)
- [x] **Bench VLM path.** `smlx/bench/suites/vlm.py` gained a documented generic
  generation path (`model.generate(prompt, image, max_tokens, temperature) -> str`)
  so any model (incl. test doubles) can be benchmarked, not only the 4 built-in
  VLMs; a path-image `ResourceWarning` leak was fixed. The 7 unit-level VLM bench
  tests are un-skipped. The quantization bench function was already fully
  implemented (verified end-to-end with SmolLM2-135M); `test_basic_benchmark` is now
  a real `requires_model` test rather than a mock that cannot drive generation.
- [x] **smolGenCad vocab mismatch.** Decoder embedding / head were sized 1100 but the
  tokenizer emits ids up to 1103 (vocab 1104) → out-of-range tokens. Unified to a single
  `CAD_VOCAB_SIZE = 1104` constant in `tokenizer.py` (with a drift-guard assertion),
  referenced by `decoder.py`, `model.py`, and `loader.py`.
- [x] **smolGenCad CAD export dropped circle/rect coordinates.** `generate.py`
  `sequence_to_python` emitted `result.circle(r)` (always at the origin), ignoring the
  required `cx/cy`, and left `in_sketch`/`sketch_commands` dead. Now emits
  `moveTo(cx, cy).circle(r)` / `.rect(...)`, warns on geometry outside a sketch, and
  summarizes each closed sketch.
- [x] **MLP dropout silently ignored.** `smlx/models/common/mlp.py` accepted a `dropout`
  arg but did nothing (`if self.dropout > 0.0: pass`). Now uses `nn.Dropout`.
- [x] **nanoVLM attention was fully bidirectional.** The forward drove the language-model
  layers directly with `mask=None`, so SDPA applied no masking (text attended to future
  tokens). `model.py` now builds a causal mask (with image↔image bidirectional blocks,
  image positions found from `input_ids`) via the previously-dead `create_attention_mask`.
- [x] **nanoVLM generation re-ran the full forward each step AND dropped the image after
  token 0.** `smlx/models/nanoVLM/generate.py` `generate`/`stream_generate` rebuilt the
  whole `input_ids` and re-ran the entire forward every step (O(N³) over N tokens) while
  setting `pixel_values=None` after the first token — so from token 1 on the `49150` image
  markers were embedded as plain text and the attention fell back to a pure causal mask,
  silently dropping the image. The model + SmolLM2 layers already accepted `cache=` but the
  loop never used it (unlike every sibling VLM, which already prefill+cache). Now builds a
  per-layer KV cache (`smlx.utils.cache.make_cache`), prefills the prompt+image once, then
  decodes single tokens against the cache — linear *and* image-conditioned for every token.
  Numerically equivalent to a full re-forward that keeps the image (image-token K/V are
  frozen at their prefill values). Guarded by `tests/models/test_nanovlm.py::
  TestKVCacheGeneration` (text-only cache equivalence, image-influences-every-token, and a
  generate()-level prefill→single-token-decode shape check). (Verified at unit level with
  random-init models; the real-weights integration suite is `requires_model` and was not
  run here.)
- [x] **PyTorch `.bin` conversion implemented.** `smlx/tools/convert2mlx.py` previously
  raised `NotImplementedError` for `.bin`; it now loads the torch state_dict (sharded ok,
  bfloat16 upcast) and converts to MLX arrays.
- [x] **`verify_weights` crashed on MLX arrays** (`np.all` TypeError) and **B904** in
  `smlx/utils/loading.py` — fixed (`np.array(weight)`, `raise ... from e`).
- [x] **`timer` rejected zero iterations** (`smlx/utils/timing.py`) — now allowed.

### CRITICAL — broken / non-functional

- [x] **Invalid-UTF-8 source files crash on (re)compile.** 8 files carry a bare Latin-1
  byte (`©`=`0xA9`, `°`=`0xB0`, `±`=`0xB1`, or `0xF1`) in a comment with **no**
  `# -*- coding: utf-8 -*-` header. Python 3 raises
  `SyntaxError: Non-UTF-8 code starting with '\xNN'` on recompile. Fix the byte (use a
  real UTF-8 `©` or ASCII `(c)`) in every one:
  - `smlx/utils/trace.py` — **fails to import now** (14 bad bytes)
  - `smlx/gym/envs/classic/cartpole.py:21` and `smlx/gym/envs/classic/lunar_lander.py` —
    **fail to import**, and via `smlx/gym/envs/classic/__init__.py:23-24` they take the
    whole `smlx.gym.envs.classic` package down (valid `MountainCarEnv` becomes
    unimportable too)
  - `smlx/main.py:2` — `python smlx/main.py` dies immediately (CLI only runs via
    `python -m smlx.main` off cached `.pyc`)
  - `smlx/kv_cache/rope.py`, `smlx/server/__init__.py`,
    `smlx/server/middleware/__init__.py`, `smlx/server/routes/__init__.py` — import only
    via stale `.pyc`; break the moment they are edited/recompiled

- [x] **`smlx/server/routes/gym.py` is an orphaned router.** It defines 5 real endpoints
  (`POST/GET /v1/gym/envs...`, lines 179-362) but is **never** `include_router`'d in
  `smlx/server/app.py` (only the 5 OpenAI routers are wired, app.py:215-219). Every
  `/v1/gym/*` route is unreachable. Either register it or delete it.

### HIGH — fabricated output presented as real (most deceptive)

- [x] **`smlx/models/Orpheus_150M/synthesize.py:70-73`** prints
  `"✓ Generated …s of audio (HiFi-GAN V3 vocoder)"` even though `loader.py:113-121`
  admits the weights aren't on HF and returns `None`, so the forward pass runs on
  **random-init weights and emits noise**. Presents garbage as a successful synthesis.
  Make output honest/conditional on real weights being loaded.
- [x] **`smlx/models/Chatterbox/synthesize.py:191-195`** *unconditionally* prints
  `"Note: Generated …s of placeholder audio … Load pre-trained weights … for actual
  synthesis"` — wrong in the success case (`loader.py:76-86` can load real weights) and
  the inverse deception of Orpheus. Gate the message on whether weights actually loaded.
- [x] **`smlx/tools/convert2mlx.py:350-385` `quantize_model` is a no-op.** It writes a
  `quantization` block into the config then `return weights, quantized_config` with the
  tensors **unchanged** (`# For now, return weights as-is`). It's reachable via the
  `--quantize` CLI flag (line ~1085, help: "Apply quantization to model"), so users get
  an unquantized model **labeled as quantized**. The `--quant-recipe` mixed-bit path
  (lines 945-955) also logs a warning and falls through to this no-op. Wire it to the
  real `smlx.quant` module or remove the flags.
- [x] **`smlx/bench/suites/quantization.py:228-229`** fabricates the prefill/decode split:
  `prompt_time = total_time * 0.2` / `generation_time = total_time * 0.8` (hardcoded
  ratio), then reports `prompt_tps`/`generation_tps` as if measured. Measure the two
  phases or stop reporting them as distinct metrics.
- [x] **`smlx/bench/suites/llm.py:550-553` `benchmark_llm_streaming` is fake.** It
  `return stats, []` (empty token-times) while its docstring/example (lines 541-548)
  promise per-token timing and compute TTFT as `token_times[0]` → callers get an
  `IndexError`. Implement real streaming timing or delete the function + its claims.

### MEDIUM — misleading behavior / false capability claims

- [x] **`smlx/quant/autoquant.py:143-149`** — the "Test if MXFP quantization is available"
  probe calls plain **INT4** `mx.quantize` (no `mode="mxfp4"`), which always succeeds, so
  it **unconditionally sets `ocp_microscaling=True`**, overriding the correct
  M1/M2 → `False` detection at lines 98-102. Falsely advertises MXFP hardware support on
  M1/M2 and routes to unsupported strategies. Probe the real `mode="mxfp4"` path.
- [x] **`smlx/quant/dwq.py:383-413`** — the loop printed as "Step 4: Refining weights with
  knowledge distillation" computes `_total_loss` and **discards it**
  (`# unused for now`); no gradients, no optimizer — it recomputes the same output and
  prints the same loss every iteration. Real refinement happens only in Step 4.5
  (`_refine_with_gradients`, 415-426). Remove the dead display loop or make it train.
  (Related: `dwq.py:242-243` comment claims "Update only scales and biases (filter
  gradients)" but no filtering occurs.)
- [x] **`smlx/gym/algorithms/base.py:296-299` `_evaluate()` is `pass` + `# TODO`** yet it's
  called every `eval_interval` episodes in the live training loop (line 240). Configured
  evaluations silently do nothing. Implement it.
- [x] **`smlx/models/Donut_base`** — `model.py:782,870` docstrings call the
  **implemented** `BARTDecoder` a "placeholder … replaced in Phase 2", and
  `__init__.py:117` claims "✅ Pre-trained on large document datasets" while
  `__init__.py:124-130` admits no weights load. Fix the contradictory docstrings/claims.
- [x] **`smlx/models/SmolVLM_500M_Instruct/model.py:5,12`** — module docstring says
  "SmolVLM-**256M**-Instruct" and "Total parameters: **~256M**" (copy-pasted from the
  256M variant; this is the 500M model). Correct the figures.
- [x] **`smlx/models/Moondream2/generate.py:543-544`** emits a hardcoded `0.9` confidence
  for every text-parsed detection (fabricated scores). It's a documented fallback, but
  returns fake numbers — surface them as `None`/unknown instead.

### LOW — disclosed-but-misleading, naming mismatches, doc drift

- [x] **GGML `quantize_model_*` helpers dequantize back to FP16 → zero runtime savings**
  despite the name (`smlx/quant/q8_0.py:195`, `q4_1.py:227`,
  `q4_k_m.py:355,544,564`, `q6_k.py:374`). Disclosed in docstrings; rename or actually
  keep weights packed.
- [x] **`smlx/models/TrOCR_small/processor.py:181-238`** — `ord(c) % vocab_size` /
  `chr(t)` char-level tokenizer fallback silently produces garbage if the HF tokenizer
  load (line ~155) fails. Fail loudly instead.
- [x] **`smlx/models/SileroVAD/loader.py:123-128`** — ONNX (Silero's *native* format) →
  "Conversion to MLX not yet implemented" → random weights. The realistic path yields an
  untrained VAD.
- [x] **`smlx/bench/suites/quantization.py:128`** hardcodes `total_params = 135_000_000`
  fallback; **`smlx/bench/suites/llm.py:386-390`** `_manual_generate_loop` never breaks on
  EOS (token counts ignore natural stopping).
- [x] **CLI console script — now wired and verified.** `pyproject.toml [project.scripts]`
  has `smlx = "smlx.main:main"` (uncommented); after `pip install -e .` the `smlx …`
  command works (verified in a fresh venv from the built wheel). `python -m smlx.main`
  also still works. Note: importing `smlx.models` requires `psutil` and `safetensors`,
  which are now **core** dependencies (previously mis-filed under `[dev]`/`[tools]`, which
  made a clean `pip install smlx` crash on `import psutil` in `smlx_manager.py`).

### Documentation defects in THIS file (now reconciled)

- [x] **"Version Control" section** — the working tree was **not** a git repository. Fixed
  by `git init` (branch `main`); `git status` now works. No commits were made — the tree
  is untracked until you choose to commit.
- [x] **"Data Directory (Git LFS)" section** — there was no top-level `data/`. Fixed by
  creating the `data/{audio,benchmark,datasets,documents,images}/` skeleton (each with a
  `.gitkeep`). **Residual by-design:** the actual datasets are Git-LFS tracked and remain
  absent until `git lfs pull` against a remote that has them.
- [x] **"Model Implementation Pattern" overstated uniformity** — clarified in that section
  that the 6-file layout is the SmolLM2-style template, and audio models like
  `Whisper_tiny` use a different layout.
- [x] **`smlx/models/all_MiniLM_L6_v2/`** has no `model.py` (it's a thin wrapper around
  `MiniLM`) — the "Implemented Models → Embeddings" entry is corrected to say so.

## Essential Commands

### Python Environment

**Python path for all commands**: `/Users/ryanoboyle/miniforge3/envs/smlx/bin/python`

```bash
# Activate conda environment before running any commands
conda activate smlx

# Install package in editable mode
pip install -e .                    # Base install
pip install -e ".[dev]"             # With dev dependencies
pip install -e ".[all]"             # With all optional dependencies
```

### Testing

```bash
# Run all tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest

# Run specific test file
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/test_gptq.py -v
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/integration/test_smollm2_generation.py -v

# Run with markers
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m unit          # Only unit tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m "not slow"    # Skip slow tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m integration   # Integration tests

# Useful options
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -v               # Verbose
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -x               # Stop on first failure

# Parallel execution: -n auto is DELIBERATELY NOT enabled in pytest.ini.
# Integration tests use module-scoped fixtures and will exhaust memory if run in
# parallel. Only parallelize unit tests, and cap the worker count:
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -n 2 tests/unit/
```

**pytest.ini gotchas that bite (all configured in `addopts`/`[pytest]`):**

- `filterwarnings = error` — **any warning fails the test**. New code that emits a
  DeprecationWarning (etc.) will fail until the warning is fixed or explicitly ignored.
- `--timeout=60` (thread method) — a single test taking >60s is killed. Mark genuinely
  long tests `@pytest.mark.slow` and/or raise the timeout for them.
- `--strict-markers --strict-config` — using an unregistered marker is a hard error.
  Register every new marker in `pytest.ini` before using it.
- `asyncio_mode = auto` — async tests run without an explicit `@pytest.mark.asyncio`.

### Code Quality

```bash
# Format code (line length = 100)
black .

# Lint code
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m ruff check .
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m ruff check --fix .      # Auto-fix

# Type checking
mypy smlx/
```

### Running Models

Every curated model runs through one real entrypoint — `smlx run` (gate-verified):

```bash
PY=/Users/ryanoboyle/miniforge3/envs/smlx/bin/python
$PY -m smlx.main run --list                                   # every runnable model
$PY -m smlx.main run smollm2-135m --text "What is MLX?"       # language
$PY -m smlx.main run smolvlm-256m -i photo.jpg --text "What is this?"  # VLM
$PY -m smlx.main run whisper-tiny --audio clip.wav            # ASR
$PY -m smlx.main run ocr --document scan.png                  # OCR
$PY -m smlx.main run --verify                                 # fail-closed gate (15/15)
```

Standalone scripts under `examples/` cover the non-model subsystems (quant, gym/RL,
server, eval).

### Command-Line Interface (`smlx/main.py`)

There is a full Click-based CLI in `smlx/main.py` (commands: `generate`, `server`,
`bench`, `convert`, `download`, `transcribe`, `data`). It **is** wired up as a console
script — `smlx = "smlx.main:main"` in `pyproject.toml [project.scripts]` — so after
`pip install -e .` the `smlx` command works directly. The module form also works:

```bash
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.main generate SmolLM2-135M "Hello"
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.main generate SmolVLM-256M "What's this?" -i photo.jpg
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.main server --host 0.0.0.0 --port 8000
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.main transcribe audio.wav
```

> **Run it as a module (`-m smlx.main`), not as a script (`python smlx/main.py`).**
> Several source files (`smlx/main.py`, `smlx/utils/trace.py`, `smlx/kv_cache/rope.py`,
> `smlx/gym/envs/classic/{cartpole,lunar_lander}.py`, and the `smlx/server/`
> `__init__.py` files) contain a bare Latin-1 `©` byte (`0xA9`) in a comment with no
> `# -*- coding: utf-8 -*-` declaration. Python 3 raises
> `SyntaxError: Non-UTF-8 code starting with '\xa9'` whenever it **recompiles** such a
> file. Today these import only because a valid cached `.pyc` exists; `trace.py`,
> `cartpole.py`, and `lunar_lander.py` already fail to import outright, and
> `python smlx/main.py` fails immediately. When you edit any of these files, fix the
> byte (use a real UTF-8 `©` or ASCII `(c)`) in the same change, or the module stops
> importing.

## Architecture Overview

### Module Structure

```
smlx/
├── models/          # Model layer (curated models run via upstream impls)
│   ├── common/      # Shared model components (attention, MLP, MoE/switch layers)
│   ├── runner.py        # Unified runner: REGISTRY + produce()/produce_all()
│   ├── runner_adapters.py  # Registers each curated alias with its real backend
│   ├── runner_verify.py    # Fail-closed per-modality correctness gate
│   ├── mlx_backend.py   # load/generate/transcribe/embed over upstream libs
│   ├── registry.py      # Legacy load_model() API (delegates to mlx_backend)
│   └── cad.py           # Deterministic text-to-CAD parser
├── quant/           # Quantization (GPTQ, AWQ, LoRA, DoRA, 4-bit, 8-bit, etc.)
├── utils/           # Shared utilities (generation, sampling, loading, memory)
├── config/          # Per-model memory profiles & safe-parameter presets (OOM guards)
├── evals/           # Evaluation benchmarks (MathVista, MMMU, OCRBench)
├── bench/           # Performance benchmarking framework
├── server/          # FastAPI REST API (OpenAI-compatible)
├── agents/          # Agent system (ReAct, Chain-of-Thought)
├── gym/             # RL environment integrations
├── data/            # Data module for dataset loading
├── tools/           # CLI tools (download, convert, compare)
├── kv_cache/        # KV cache implementations
└── main.py          # Click CLI entry point (run via `python -m smlx.main`)
```

**`smlx/config/model_profiles.py`** holds per-model safe `max_tokens` / KV-cache /
batch-size presets to prevent OOM on the 36GB M4 target. Prefer pulling parameters from
here over hardcoding generation limits when adding a model.

### Model Execution Framework

SMLX does **not** re-implement model forward passes. Every curated model runs
through a maintained upstream implementation (or real deterministic code), behind
one unified layer:

1. **Runner** (`runner.py` + `runner_adapters.py`) — a `REGISTRY` mapping a short
   alias to a `RunEntry`; `produce(alias, text=/image=/audio=/document=)` runs the
   model's real pipeline and returns an honest `RunResult`. `smlx run` is the CLI.
2. **Backend** (`mlx_backend.py`) — `load()` returns a `BackendModel`
   (`.model` / `.processor` / `.backend` / `.repo` / `.modality` / `.quantized`);
   `generate()` / `transcribe()` / `embed()` are the one-call entrypoints over
   mlx-lm / mlx-vlm / mlx-whisper / mlx-embeddings.
3. **Gate** (`runner_verify.py`) — `smlx run --verify` loads every model, runs a
   real per-modality correctness check, and exits non-zero on any failure
   (currently 15/15).

The earlier per-model "Model Execution Framework" (`smlx_router.py` /
`smlx_runner.py` / `smlx_manager.py`) and the 17 bespoke model packages were
**removed** — they hand-reimplemented forward passes and several emitted garbage.

**Model loading**:

```python
from smlx.models import load, generate

m = load("smollm2-135m")              # BackendModel (mlx-lm)
generate(m, "What is MLX?")

m = load("smolvlm-256m", quantize="4bit")
generate(m, "Describe this image.", image="photo.jpg")

# Legacy alias: load_model(...) returns the same BackendModel
from smlx.models import load_model
bm = load_model("mlx-community/SmolLM2-135M-Instruct")
```

### Implemented Models (curated zoo — verified by `smlx run --verify`)

15 aliases, each routed to a real upstream backend (see `docs/MODEL_STATUS.md`):

- **Language** (mlx-lm): `smollm2-135m`, `smollm2-360m`
- **Vision-language** (mlx-vlm): `smolvlm-256m`, `smolvlm-500m`, `nanovlm`,
  `tinyllava`, `moondream3`
- **ASR** (mlx-whisper): `whisper-tiny`
- **TTS** (mlx-audio / Kokoro): `kokoro`
- **OCR** (SmolVLM via mlx-vlm): `ocr`
- **VAD** (onnxruntime / Silero): `silero-vad`
- **Audio classification** (transformers / AST): `ast`
- **Embeddings** (mlx-embeddings): `minilm`, `all-minilm-l6-v2`
- **CAD** (deterministic CadQuery parser): `cad`

Adding a model is a one-line entry in `runner_adapters.py` (alias → repo +
modality), then `smlx run --verify <alias>`. No model code is needed when
mlx-lm / mlx-vlm support the architecture.

### Server Architecture (FastAPI)

OpenAI-compatible REST API with:

- `POST /v1/chat/completions` - Chat with message history
- `POST /v1/completions` - Text completion
- `POST /v1/audio/transcriptions` - Audio transcription
- `POST /v1/embeddings` - Text embeddings
- `GET /v1/models` - List available models

**Running the server** — `smlx/server/app.py` does **not** parse CLI args; its
`__main__` block hardcodes `0.0.0.0:8000` with `reload=True`. To control host/port, use
the CLI command or invoke uvicorn directly:

```bash
# Preferred — Click CLI (accepts --host/--port/--reload/--log-level)
python -m smlx.main server --host 0.0.0.0 --port 8000

# Direct uvicorn (host/port via uvicorn flags)
uvicorn smlx.server.app:app --host 0.0.0.0 --port 8000 --reload

# Defaults only (0.0.0.0:8000, reload on) — no flags are honored here
python -m smlx.server.app
```

## Critical Rules

### 1. Never Import from `resources/`

The `resources/` directory contains reference implementations from MLX ecosystem projects (mlx-lm, mlx-vlm, etc.). **DO NOT** import from these modules directly.

**Correct approach**:

1. Study implementation patterns in `resources/`
2. Copy and adapt code into appropriate `smlx/` module
3. Ensure implementation is suitable for "smol" models

### 2. Don't Remove Stub Files

If something is called but missing, it should be **implemented, not removed**. Many files exist as placeholders for planned features.

### 3. Code Style

- Line length: 100 characters (configured in pyproject.toml)
- Use Black for formatting
- Use Ruff for linting
- Follow the runner/backend patterns (`runner_adapters.py`, `mlx_backend.py`), not
  hand-written per-model forward passes

### 4. Adding a Model

There are **no per-model packages** — do not hand-reimplement forward passes (the
17 bespoke packages were removed for exactly this reason). To add a model, register
one `RunEntry` in `smlx/models/runner_adapters.py` that points an alias at a real
upstream repo + modality, then prove it:

```python
# in runner_adapters.py — e.g. another mlx-vlm model
_VLM_BACKEND["my-vlm"] = ("org/My-VLM-mlx-4bit", "My-VLM (mlx-vlm)")
```

```bash
smlx run --verify my-vlm        # must pass the fail-closed correctness gate
```

**Key requirements**:

- Routes to a maintained upstream impl (mlx-lm / mlx-vlm / mlx-whisper /
  mlx-embeddings / mlx-audio / onnxruntime / transformers) or real deterministic code
- Fits the 36 GB M4 memory budget and passes `smlx run --verify` (real output)
- Param count is a guideline (prefer < 1B); a larger model is admitted as a
  documented performance exception (e.g. the ~1.57B `moondream3`)

### 5. Use Shared Utilities

Prefer utilities from `smlx/utils/` over reimplementing:

- `generation.py` - Text generation, streaming, chat
- `sampling.py` - Token sampling (temperature, top-p, top-k)
- `loading.py` - Model and tokenizer loading
- `memory.py` - Memory profiling and management
- `cache.py` - KV cache implementations

## Testing Guidelines

### Pytest Markers

Use markers to categorize tests (defined in pytest.ini):

```python
@pytest.mark.unit            # Fast, isolated unit tests
@pytest.mark.integration     # Integration tests (may require external services)
@pytest.mark.slow            # Slow-running tests (skip with -m "not slow")
@pytest.mark.benchmark       # Performance benchmarking
@pytest.mark.eval            # Evaluation tests (may require datasets)
@pytest.mark.gpu             # Requires GPU/MLX acceleration
@pytest.mark.requires_model  # Downloads models
@pytest.mark.requires_hf     # Requires the HuggingFace datasets library
@pytest.mark.heavy_memory    # Needs significant memory (>3GB)
@pytest.mark.streaming       # Uses streaming/buffering (may build up resources)
@pytest.mark.asyncio         # Async test (asyncio_mode=auto makes this optional)
```

The marker is `eval`, **not** `evaluation`. `--strict-markers` rejects any marker not
registered in `pytest.ini`. `tests/conftest.py` installs autouse fixtures that purge MLX
memory after each test and each module — this is why integration tests must run serially
(see the parallel-execution note above).

### Test Structure

```
tests/
├── unit/             # Fast unit tests
├── integration/      # Integration tests
├── quant/            # Quantization tests
├── evals/            # Evaluation tests
├── models/           # Model-specific tests
├── server/           # Server API tests
└── bench/            # Benchmark tests
```

## Data Directory (Git LFS)

> The `data/{audio,benchmark,datasets,documents,images}/` skeleton exists (each with a
> `.gitkeep`), but the **actual datasets are Git-LFS tracked and absent** until you
> `git lfs pull` from a remote that hosts them. Code that needs a specific dataset file
> will still fail until the files are pulled.

The `data/` directory is tracked with Git LFS and contains:

- `data/audio/` - Audio test files
- `data/benchmark/` - Benchmark datasets
- `data/datasets/` - Evaluation datasets (MathVista, MMMU, etc.)
- `data/documents/` - Document images for OCR
- `data/images/` - Test images for vision models

**Important**: Run `git lfs pull` after cloning to download large files.

## Quantization Support

The `smlx/quant/` module provides comprehensive quantization:

**Methods**:

- GPTQ - Post-training quantization for language models
- AWQ - Activation-aware weight quantization
- Dynamic - Mixed-precision based on layer sensitivity
- LoRA/DoRA - Parameter-efficient fine-tuning

**Bit-widths**:

- 4-bit, 6-bit, 8-bit integer quantization
- FP4, FP8 floating-point formats
- MXFP4, MXFP8 microscaling formats (OCP standard)
- GGML formats (Q4_0, Q4_1, Q4_K, Q8_0) - llama.cpp compatible

**Usage**:

```python
from smlx.quant import quantize_model, gptq_quantize, awq_quantize

# Simple quantization
model = quantize_model(model, bits=4, group_size=64)

# GPTQ (better quality preservation)
gptq_quantize(model, calibration_data)

# AWQ (activation-aware)
awq_quantize(model, calibration_data)
```

## Development Workflow

1. **Check existing stubs** - Many files exist as placeholders
2. **Reference resources** - Study patterns in `resources/` directory
3. **Implement in smlx/** - Write implementations in appropriate module
4. **Keep it small** - Remember: models must be "smol" (< 1B parameters)
5. **Test** - Add tests with appropriate markers
6. **Format and lint** - Run black and ruff before committing

## Configuration Files

- `pyproject.toml` - Package metadata, dependencies, tool configs (black, ruff, mypy)
- `pytest.ini` - Pytest configuration with custom markers
- `environment.yml` - Conda environment specification
- `setup.py` - Minimal wrapper for editable installs
- `.gitattributes` - Git LFS configuration for data files

## Version Control

> The repo was `git init`'d (branch `main`) during the defect-fix pass, so the commands
> below work. The working tree is currently **untracked** (no commits yet) — commit when
> ready. Git LFS commands require `git lfs` installed and a remote that hosts the data.

This project uses Git with Git LFS for large file tracking:

```bash
# Check git status
git status

# View LFS-tracked files
git lfs ls-files

# Pull LFS files
git lfs pull
```

## Key Principles

1. **Small models only** - Core requirement: < 1B parameters
2. **MLX-first** - Use MLX framework for Apple Silicon optimization
3. **M4 optimized** - Target M4 chipset with 36GB unified memory
4. **No direct resource imports** - Reference but don't import from `resources/`
5. **Quantization by default** - Models should support 4-bit/8-bit quantization
6. **Shared utilities** - Reuse code from `smlx/utils/`
7. **Consistent APIs** - Follow established patterns from SmolLM2_135M, Whisper_tiny
