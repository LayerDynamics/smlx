# Plan: Unified Model Runner — produce real output from ALL models

**Date:** 2026-06-07
**Branch (suggested):** `feat/unified-model-runner`
**Status:** Planned (awaiting execution)

## What this is

SMLX has **18 model packages** spanning 9 modalities. The curated `mlx_backend`
zoo already runs a verified subset (LM/VLM/ASR/embeddings) through upstream MLX
impls. This plan adds **one unified way to actually run every implemented model and
get real output from it** — including the legacy custom implementations (TTS, OCR,
VAD, audio-classification, CAD, and the hand-written VLMs) — through a single
registry, a `produce()` Python API, and a `smlx run` CLI.

It is explicitly **honest about weight status**: every run reports whether the
output came from real trained weights, trained weights with a known gap, or an
untrained/partial pipeline (random weights → structurally-real but not meaningful
output). No fabrication, no placeholder success messages.

## Goal

- `smlx run <model> [input...]` loads a model, runs its **real** inference pipeline,
  writes any non-text artifact to `data/output/`, and prints a one-line honest
  status + output snippet.
- `smlx run --all --text ... --image ... --audio ... --document ...` fans the
  provided inputs to every model of the matching modality; models whose required
  input was not supplied are **skipped with a clear reason** (inputs are
  user-provided — no bundled sample files).
- A `produce(model_id, inputs) -> RunResult` Python API backs the CLI.
- Every model is covered (all 18), each exercised by a test that asserts **output
  happens** and the reported weight-status is correct.

## Non-goals

- Training models or sourcing missing public weights (Orpheus, smolGenCad, etc.
  stay untrained — we run their pipelines honestly, not invent quality).
- Replacing the curated `mlx_backend` zoo or `smlx models verify` — this composes
  with them (LM/VLM/ASR/embeddings can delegate to the backend; the runner adds the
  legacy modalities the backend doesn't cover).
- Bundling new sample input assets (inputs are user-provided per the decision).

## Model inventory (ground truth from `smlx/models/*/__init__.py`)

| Model | Modality | load + inference fn | Input | Output |
|-------|----------|---------------------|-------|--------|
| SmolLM2_135M / _360M | language | `load`, `generate`/`chat` | text | text |
| SmolVLM_256M / _500M_Instruct | vlm | `load`, `generate` | image+text | text |
| nanoVLM | vlm | `load`, `generate`/`caption` | image+text | text |
| Moondream2 | vlm | `load`, `generate`/`caption`/`detect` | image+text | text/boxes |
| TinyLLaVA | vlm | `load`, `generate`/`caption` | image+text | text |
| Whisper_tiny | asr | `load`, `transcribe` (+`vad`) | audio | text |
| Chatterbox | tts | `load`, `synthesize` | text | audio (wav) |
| Orpheus_150M | tts | `load`, `synthesize` | text | audio (wav) |
| TrOCR_small | ocr | `load`, `recognize` | document image | text |
| Donut_base | ocr | `load`, `generate` | document image (+task) | text/json |
| MiniLM / all_MiniLM_L6_v2 | embeddings | `load`, `encode` | text(s) | vector(s) |
| SileroVAD | vad | `load`, `vad` | audio | speech segments |
| YAMNet | audio_cls | `load`, `classify` | audio | class labels |
| smolGenCad | cad | `load`, `generate` | text | CAD seq + Python/JSON |

## Architecture

### 1. Adapter registry — `smlx/models/runner.py` (new)

A `RunEntry` per model gives the runner a uniform handle without flattening each
model's real API:

```python
@dataclass(frozen=True)
class RunEntry:
    key: str                      # CLI alias, e.g. "orpheus-150m"
    modality: str                 # language|vlm|asr|tts|ocr|embeddings|vad|audio_cls|cad
    needs: tuple[str, ...]        # required inputs: ("text",) | ("image","text") | ("audio",) | ("document",)
    loader: Callable[[], Any]     # returns whatever that model's load() returns
    runner: Callable[..., RunOutput]  # adapter: takes loaded handle + inputs -> RunOutput
```

Each adapter is a thin function calling the model's **own real** `generate`/
`synthesize`/`recognize`/`classify`/`vad`/`encode`/`transcribe` — no
reimplementation. The registry is the single source of "all models".

### 2. Weight-status taxonomy (honest, computed at runtime — never hardcoded)

```python
class WeightStatus(Enum):
    TRAINED          = "TRAINED"           # real public weights, meaningful output
    TRAINED_GAP      = "TRAINED-WEIGHTS"   # real weights but a known defect (e.g. TrOCR decoder gap)
    PIPELINE_ONLY    = "PIPELINE-ONLY"     # random/partial weights: output happens, not meaningful
    NO_WEIGHTS       = "NO-WEIGHTS"        # could not load any weights at all
```

Status is **derived from real load signals**, e.g.:
- Orpheus exposes `model.weights_loaded` (loader.py sets it) → PIPELINE_ONLY when False.
- TrOCR loader reports "N params not in checkpoint" → TRAINED_GAP when >0 missing.
- smolGenCad / SileroVAD load random weights → PIPELINE_ONLY (their loaders already
  print the warning; the adapter reads the same condition).
- Each adapter returns the `reason` string shown in the status line.

### 3. `produce()` API + `RunResult`

```python
def produce(model_id: str, *, text=None, image=None, audio=None, document=None,
            out_dir="data/output", **opts) -> RunResult: ...

@dataclass
class RunResult:
    model: str; modality: str
    status: WeightStatus; reason: str
    output_repr: str           # text snippet, or "audio 0.14s (3328 samples)", etc.
    artifact_path: str | None  # data/output/<model>.wav|.json|.png if non-text
    elapsed_s: float
    ok: bool                   # pipeline ran and produced output (≠ trained-quality)
```

### 4. CLI — `smlx run` (in `smlx/main.py`)

```
smlx run smollm2-135m --text "What is MLX?"
smlx run orpheus-150m --text "Hello world"          # -> data/output/orpheus-150m.wav
smlx run trocr-small  --document scan.png
smlx run --all --text "Hello" --image cat.jpg --audio clip.wav --document scan.png
smlx run --list                                      # registry: model, modality, needs
```

`--all` runs every registered model whose `needs` are satisfied by the supplied
inputs; unsatisfied models print `SKIP (needs <inputs>)`. Exit non-zero only if a
model that *should* have run errored (never for SKIP, never for honest
PIPELINE-ONLY).

## Workstreams

### WS-A — Runner core (`smlx/models/runner.py`)
1. `RunEntry`, `RunOutput`, `RunResult`, `WeightStatus`.
2. `produce()` dispatcher: resolve entry → load (cached) → run adapter → classify
   status → write artifact → build `RunResult`.
3. Artifact writers: wav (soundfile), json/python (CAD), boxes/labels → json.

### WS-B — Adapters for all 18 models
One adapter per model calling its real inference fn. Group by modality:
- language (2), vlm (5: SmolVLM x2, nanoVLM, Moondream2, TinyLLaVA), asr (1),
  tts (2), ocr (2), embeddings (2), vad (2: SileroVAD + Whisper.vad), audio_cls (1),
  cad (1). LM/VLM/ASR/embeddings adapters may delegate to `mlx_backend` where a
  curated entry exists; legacy-only models call their package directly.
4. Each adapter determines and returns the real `WeightStatus` + reason.

### WS-C — CLI `smlx run` + `--all` + `--list`
5. Wire into `smlx/main.py`; per-modality input flags; honest status table;
   `data/output/` creation.

### WS-D — Tests (`tests/models/test_runner.py` + per-modality)
6. Unit: registry completeness (all 18 present; every modality covered).
7. `produce()` per model asserts **output happens** (non-empty artifact/text) and
   the reported `WeightStatus` matches the real load condition. Network-weight
   models marked `requires_model`; random-weight models run offline.
8. A `--list` snapshot test and an `--all` smoke (with provided dummy inputs)
   asserting SKIP logic for missing inputs.

### WS-E — Docs
9. Extend `docs/MODEL_STATUS.md` with a "Run any model" section + the live
   status taxonomy; add `smlx run` to README quickstart.

## Honesty rules (enforced)
- Never print "✓ generated" for PIPELINE_ONLY/NO_WEIGHTS — the status line states
  the truth (e.g. `PIPELINE-ONLY (random weights: noise, not speech)`).
- `ok=True` means *the pipeline produced output*, not *the output is correct*. The
  status field carries quality. Docs and CLI legend make this explicit.
- Status is read from real load signals; if a model's loader doesn't expose one,
  WS-B adds a real check (e.g. compare loaded keys vs `tree_flatten(params)`), not a
  guess.

## Risks / unknowns (resolved during execution by running real code)
- Some legacy adapters may error on load (missing optional deps, weight format).
  Per the "fix errors" rule, each is diagnosed and fixed or its real blocker
  surfaced as `NO_WEIGHTS` with the actual exception — not swallowed.
- YAMNet / Donut / Chatterbox / SileroVAD weight-load paths are unverified this
  session; WS-B runs each to determine real status before writing the table.
- `soundfile` for wav artifacts is already an audio dep; confirm it's declared core
  or add it (WS-A).

## Acceptance criteria
- `smlx run --list` shows all 18 models with modality + needs.
- For every model, `produce(...)` (with appropriate input) returns `ok=True`
  (pipeline produced output) and a `WeightStatus` that matches reality; artifacts
  land in `data/output/` for audio/CAD.
- `smlx run --all` with one input per modality runs all matching models and SKIPs
  the rest with reasons; no fabricated success.
- Tests in WS-D pass (offline ones in CI unit scope; weight-downloading ones under
  `requires_model`).
- README + MODEL_STATUS document `smlx run` and the status taxonomy.
