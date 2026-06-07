# Plan: Production-Ready SMLX — Every Model Real, Fast, and Installable

**Date:** 2026-06-07
**Owner:** LayerDynamics
**Status:** Proposed (awaiting execution handoff)

---

## 1. What SMLX is (the thesis the plan serves)

SMLX is an **on-device "smol model" zoo for Apple Silicon** — every model is < 1B params, MLX-native, and runs locally with no cloud. The product promise is: *install it, pick any model from the zoo, and get real output fast on your Mac.* This plan makes that promise literally true for **all 17 models** — no placeholder/noise outputs, measured performance with CI-enforced floors, a verified install, and one source-of-truth doc.

## 2. Goal & success criteria (from the requirements)

| Requirement (your answer) | Concrete definition of done |
|---|---|
| **No placeholders — every model emits real, intended output** | Each model loads real pretrained weights (or trained-by-us weights) and passes a correctness assertion on a known input. Zero `random/placeholder` weight paths reachable in the default load path. |
| **Hard perf targets + CI gate** | Each modality has a numeric floor (tok/s, RTF, latency, embed/s). A bench run records per-model numbers; CI fails if a model regresses below its floor on the reference machine. |
| **Fresh-install + CLI verify** | `pip install smlx` (wheel) and `pip install -e .` in a clean env both succeed; `smlx --help` and one real command per modality run. Extras (`[audio]`, `[vision]`, …) resolve. |
| **Per-model smoke suite + `smlx models verify`** | `pytest -m smoke` and a `smlx models verify [--model X]` command load+run+assert every model; both are the canonical "does it work" gate. |
| **Model-status table + quickstarts** | One `docs/MODEL_STATUS.md` (model → works/fast/weights/params/cmd) as source of truth; a consolidated `QUICKSTART` reconciling the 40+ existing docs. |
| **CI matrix** | GitHub Actions runs lint + unit + smoke + bench (model downloads cached) on PRs. |
| **You verify via interaction** | An interactive harness (`smlx chat` / `smlx run <model>` REPL + the showcase demos) so you can manually exercise each model; documented in MODEL_STATUS. |
| **Rollout** | **Full zoo at once** — all 17 models in scope, sequenced by dependency, not deferred. |

## 3. Ground truth — current per-model status (verified this session + loader audit)

> Legend: ✅ verified real output · 🟡 real upstream weights exist, needs conversion/wiring + verify · 🟠 weights exist but a component is placeholder · 🔴 no public weights (needs sourcing or training)

| Model | Params | Modality | Status | What it needs |
|---|---|---|---|---|
| SmolLM2_135M | 135M | LM | ✅ | perf floor + smoke test only |
| SmolLM2_360M | 360M | LM | 🟡 | verify real weights load + smoke/perf |
| SmolVLM_256M | 256M | VLM | ✅ | perf floor + smoke |
| SmolVLM_500M_Instruct | 500M | VLM | 🟡 | verify + smoke/perf |
| nanoVLM | 222M | VLM | 🟡 | wire `lusxvr/nanoVLM-222M` weights → MLX (random fallback today) |
| Moondream2 | ~0.5–1.8B* | VLM | 🟡 | verify `vikhyatk/moondream2` load (*confirm a <1B variant) |
| TinyLLaVA | ~0.5B | VLM | 🟡 | verify weights; resolve 27-vs-26 vision-layer config mismatch |
| Whisper_tiny | 39M | ASR | ✅ | perf floor + smoke (WER) |
| MiniLM | 23M | Embeddings | ✅ | perf floor + smoke |
| all_MiniLM_L6_v2 | 23M | Embeddings | 🟡 | thin wrapper — verify parity with MiniLM |
| TrOCR_small | ~60M | OCR | 🟡 | convert `microsoft/trocr-small-printed` weights → MLX; fail-loud already done |
| Donut_base | ~200M | OCR/Doc | 🟡 | convert `naver-clova-ix/donut-base`; wire BARTDecoder weights |
| YAMNet | ~4M | Audio cls | 🟡 | confirm `mlx-community/yamnet` exists; convert (orig is TF) |
| SileroVAD | ~1.8M | VAD | 🟠 | implement ONNX→MLX conversion (downloads ONNX, conversion is a stub → random) |
| Chatterbox | ~500M | TTS | 🟠 | real model weights exist (`ResembleAI/chatterbox`, cached); wire real HiFi-GAN vocoder (placeholder today) |
| Orpheus_150M | 150M | TTS | 🔴 | source real Orpheus weights on HF (e.g. canopylabs/*) or mark blocked; vocoder = `nvidia/tts_hifigan` |
| smolGenCad | 158M | Text→CAD | 🔴 | **no weights exist — must be trained** (custom arch). Highest effort/risk. |

\* Moondream2 sizing must be confirmed against the < 1B "smol" requirement before guaranteeing it.

**Reality the plan must own:** "fix all, no placeholders" is **achievable for 14 models** (✅/🟡/🟠 — conversion/wiring work) but **2 are hard**: Orpheus_150M needs a real weights source we don't control, and **smolGenCad has no weights and must be trained from scratch** (dataset + training run). These are called out as gated decisions in §6, not silently deferred.

## 4. Workstreams

### WS-0 — Foundation: real-output test contract (do first; everything depends on it)
- Define `tests/smoke/` + a `@pytest.mark.smoke` marker (register in pytest.ini).
- A reusable `assert_real_output(model_type, output)` helper per modality:
  - LM/VLM: non-empty, decodes to coherent text, not all-EOS/whitespace.
  - ASR: WER below a ceiling on a known LibriSpeech clip.
  - Embeddings: correct dim + semantic ranking (known pair similarity > unrelated).
  - TTS: audio is **not** silence/noise — assert spectral energy + (optional) ASR round-trip intelligibility of the synthesized clip.
  - OCR: exact/normalized string match on a known image.
  - VAD: correct speech/non-speech segmentation on a labeled clip.
  - Audio cls / CAD: top-1 label / valid parseable CAD program on a known input.
- **This is how "no placeholder" is enforced**: a random-weight model fails `assert_real_output`.

### WS-1 — Packaging & install (make it usable by others)
- Verify `pip install -e .` and a built wheel install in a **fresh conda/venv**; fix any missing runtime deps, extras (`[audio]`,`[vision]`,`[server]`,`[all]`), and `py.typed`.
- Confirm the now-uncommented `smlx = "smlx.main:main"` console script works post-install (`smlx --help`, `smlx generate`, `smlx transcribe`, `smlx data list`).
- Reconcile CLAUDE.md (which still says the entry point is commented out) with reality.
- Weight auto-download UX: first run of any model prints a clear "downloading X from HF…" and caches; offline behavior is a clean error, never silent random init.

### WS-2 — Make every model emit real output (the core, no placeholders)
Per-model, in dependency order (see §5). For each: load real weights → run on a fixture → pass `assert_real_output` → record in MODEL_STATUS.
- **Conversions (🟡):** SmolLM2_360M, SmolVLM_500M, nanoVLM, Moondream2, TinyLLaVA, all_MiniLM_L6_v2, TrOCR_small, Donut_base, YAMNet — write/verify the HF→MLX weight mapping in each `loader.py`; remove the random fallback from the default path (keep only behind an explicit `allow_random=True` test flag).
- **Vocoder/format wiring (🟠):** SileroVAD (ONNX→MLX), Chatterbox (real HiFi-GAN vocoder + verify ResembleAI weights).
- **Sourcing/training (🔴):** Orpheus_150M (locate real weights; if none, escalate per §6), smolGenCad (train: assemble CAD dataset, training loop, eval; or source weights).

### WS-3 — Performance: hard targets + gate
- Extend `smlx/bench/suites/` to cover **audio (ASR/TTS/VAD/cls), OCR, embeddings, CAD** (today: llm/vlm/quant/ops/text_gen).
- Establish a **reference machine** (the M4 target) and record baseline numbers; set per-model floors (e.g. LM ≥ N tok/s, ASR RTF ≤ R, embeddings ≥ E sentences/s, VLM first-token ≤ L ms). Floors are calibrated from the baseline, not invented.
- Ensure fast-paths are on by default everywhere: KV-cache (prefill+decode, not full re-forward), quantization presets, batched where safe. Use `smlx/config/model_profiles.py` for OOM-safe presets.
- `bench` emits machine-readable JSON; a CI step compares against committed floors and fails on regression.

### WS-4 — Verification surfaces
- `pytest -m smoke` (WS-0) — automated per-model load+run+assert.
- `smlx models verify [--model X] [--all]` — live status command reusing the smoke assertions; prints a table (loads? real-output? tok/s).
- Interactive harness for **your** manual verification: `smlx run <model>` / `smlx chat`, plus the `examples/showcase/` demos extended to every modality.

### WS-5 — Docs: one source of truth
- `docs/MODEL_STATUS.md` — generated/maintained table (model → status, params, weights repo, perf, one-line run command). MUST match `smlx models verify` output.
- Consolidated `QUICKSTART` (install → pick a model → run) linking the per-modality docs; prune/merge contradictory entries among the 40+ files.
- Update README badges/claims to match measured reality.

### WS-6 — CI matrix
- GitHub Actions: lint (ruff+black) + `mypy` (with the mlx-stub workaround) + unit tests on every PR; smoke + bench on a self-hosted/Apple-Silicon runner (model downloads cached) — the only place real perf floors can be enforced.
- Document the runner requirement (MLX needs Apple Silicon; GitHub-hosted macOS runners are now arm64 — validate they suffice or use self-hosted).

## 5. Sequencing (full zoo, dependency-ordered)

1. **WS-0** real-output test contract + **WS-1** install/CLI (unblocks verifying everything).
2. **WS-2 conversions** for the ✅/🟡 core first (cheapest, highest coverage): LM ×2, VLM ×4, embeddings ×2, then OCR ×2, YAMNet.
3. **WS-2 wiring** for 🟠: SileroVAD, Chatterbox.
4. **WS-3 perf** harness + floors as each model goes green (perf test written alongside the smoke test).
5. **WS-2 hard cases (🔴):** Orpheus (source), smolGenCad (train) — longest lead time, started early in parallel given their risk.
6. **WS-5 docs** + **WS-6 CI** continuously, finalized once the matrix is green.

## 6. Risks & gated decisions (need your call when reached)

- **smolGenCad (🔴, highest):** no weights exist; "real output" = **train a model** (needs a parametric-CAD dataset, a training run, and an eval). This is a project of its own. *Decision when reached:* train it now (scope a dataset + budget), source third-party weights, or ship it flagged `experimental` despite the no-placeholder mandate.
- **Orpheus_150M (🔴):** depends on a real public weight source existing. *Decision:* if no usable weights, swap to a different smol TTS with weights, or flag blocked.
- **Moondream2 "smol" compliance:** the common Moondream2 is ~1.8B — confirm a < 1B variant exists or it violates the core requirement.
- **Perf floors are machine-specific:** numbers only mean something on the pinned reference M4; CI must run on equivalent hardware or floors become noise.
- **CI cost:** real-model smoke/bench downloads GBs; needs caching + an Apple-Silicon runner.
- **Conversion correctness:** a wrong HF→MLX mapping yields *plausible-looking* garbage — WS-0 assertions (coherence/WER/match), not just "non-empty", are the guard.

## 7. Definition of done
- `smlx models verify --all` → every model **loads + passes real-output assertion** (or is explicitly, visibly flagged per a §6 decision you approved).
- `pytest -m "smoke"` green; `bench` meets every committed perf floor on the reference machine; CI enforces both.
- Fresh-env `pip install` + `smlx` CLI works for all modalities.
- `docs/MODEL_STATUS.md` matches `smlx models verify`; README claims are true.
- You have interactively run each model and signed off.

---

### Appendix — verified-this-session evidence
SmolLM2-135M (wikitext ppl ≈ 26), SmolVLM-256M (correct coco8 captions), Whisper-tiny (WER ≈ 10%), MiniLM (correct semantic ranking) — the ✅ baseline the rest is measured against.
