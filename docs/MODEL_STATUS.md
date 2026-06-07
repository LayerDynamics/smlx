# SMLX Model Status

Single source of truth for the curated model zoo. Every entry here is verified by
`smlx models verify` — it loads the model through its backend, runs it, and checks
the output is **real** (coherent / correct), not placeholder.

Regenerate / re-verify at any time:

```bash
smlx models verify            # all models
smlx models verify smolvlm-256m   # one model
smlx models list              # the curated zoo
```

## Run any model (`smlx run`)

The curated zoo above is the *verified, trained-quality* subset. To produce real
output from **every implemented model** (all 17 packages — including the legacy
TTS/OCR/VAD/audio-classification/CAD ones), use `smlx run`. Each model runs its own
real inference pipeline; audio/CAD/json artifacts are written to `data/output/`.

```bash
smlx run --list                                   # every runnable model + what it needs
smlx run smollm2-135m --text "What is MLX?"
smlx run orpheus-150m --text "Hello world"        # -> data/output/orpheus-150m.wav
smlx run trocr-small --document scan.png
smlx run --all --text "Hi" -i cat.jpg -a clip.wav -d scan.png
```

Inputs are **user-provided** (per-modality flags `--text/--image/--audio/--document`);
`--all` runs every model whose required inputs you supplied and **SKIPs** the rest.

### Honest weight status

Every run reports a runtime-derived status — `ok` means *the pipeline produced
output*, the status says whether that output is **trustworthy**:

| Status | Meaning |
|--------|---------|
| `TRAINED` | Real public weights → meaningful output (SmolLM2, SmolVLM, Whisper, MiniLM, Moondream2, TinyLLaVA, nanoVLM, YAMNet). |
| `TRAINED-WEIGHTS` | Real weights but a known defect (TrOCR: decoder MLP params absent from the checkpoint → repetitive). |
| `PIPELINE-ONLY` | Random/partial weights: output happens and is well-formed, but **not** meaningful (Orpheus, Chatterbox without local weights, smolGenCad, SileroVAD without converted weights). |
| `SKIPPED` | A required input was not supplied (not an error). |
| `ERROR` | The pipeline raised (the real exception is shown). |

Status comes from a real `model.weights_loaded` signal where the loader exposes one,
never a hardcoded guess. `smlx run` exits non-zero only on `ERROR` — `SKIPPED` and
honest `PIPELINE-ONLY` are not failures.

## Architecture

SMLX runs each model through the **correct upstream MLX implementation** and layers
its own value on top — it does **not** re-implement model forward passes:

- **mlx-lm** — language models
- **mlx-vlm** — vision-language models
- **SMLX-native** — ASR (Whisper) and embeddings (MiniLM), which mlx-lm/mlx-vlm don't cover

SMLX's differentiators: **quantization** (`smlx.quant`: 4/8-bit, GPTQ, AWQ, DWQ),
**smol curation** (this zoo), a **unified API** (`smlx.models.load` / `generate`),
and the **bench/eval/verify** trust layer. Inclusion is by *performance on the M4
target* (see `smlx/config/inclusion_policy.py`), not a hard parameter cap.

## Verified zoo

Status from `smlx models verify` on the M4 reference (tok/s is indicative, not a
committed floor — see WS-3):

| Alias | Modality | Backend | Params | Status | tok/s | Repo |
|-------|----------|---------|--------|--------|-------|------|
| `smollm2-135m` | language | mlx-lm | 135M | ✅ PASS | ~115 | mlx-community/SmolLM2-135M-Instruct |
| `smollm2-360m` | language | mlx-lm | 360M | ✅ PASS | ~114 | mlx-community/SmolLM2-360M-Instruct |
| `smollm2-1.7b` | language | mlx-lm | 1.7B | ✅ PASS | ~30 | mlx-community/SmolLM2-1.7B-Instruct |
| `qwen2.5-0.5b` | language | mlx-lm | 0.5B | ✅ PASS | ~111 | mlx-community/Qwen2.5-0.5B-Instruct-4bit |
| `smolvlm-256m` | vision-language | mlx-vlm | 256M | ✅ PASS | ~55 | HuggingFaceTB/SmolVLM-256M-Instruct |
| `smolvlm-500m` | vision-language | mlx-vlm | 500M | ✅ PASS | ~52 | HuggingFaceTB/SmolVLM-500M-Instruct |
| `smolvlm2-2.2b` | vision-language | mlx-vlm | 2.2B | ✅ PASS | ~14 | mlx-community/SmolVLM2-2.2B-Instruct-mlx |
| `qwen2-vl-2b` | vision-language | mlx-vlm | 2B | ✅ PASS | ~33 | mlx-community/Qwen2-VL-2B-Instruct-4bit |
| `whisper-tiny` | ASR | SMLX-native | 39M | ✅ PASS | 1.3s/clip | mlx-community/whisper-tiny |
| `minilm` | embeddings | SMLX-native | 23M | ✅ PASS | ~291 sent/s | sentence-transformers/all-MiniLM-L6-v2 |

**10/10 produce real, correct output**, and all clear `verify --enforce-perf`
(language/VLM/embeddings have calibrated speed floors; ASR/TTS RTF is reported but
not yet floor-gated). Latest full run: 10/10 PASS.

## Modality coverage (what's verified vs remaining)

The named target zoo spans seven modalities. Verified today:

- ✅ **Language**, ✅ **Vision-language**, ✅ **ASR**, ✅ **Embeddings** — in the zoo above.

Pipeline-verified (**output happens** end to end) but **not** trained-quality —
the generation path runs and emits real, well-formed output; the *content* is the
remaining gap, not the plumbing:

- 🟡 **TTS** (Orpheus) — `synthesize()` produces a finite audio waveform
  (`np.ndarray`, shape `(samples,)`); with the bundled random-init weights it is
  honestly labelled noise, not speech. Covered by
  `tests/integration/test_orpheus.py::test_basic_synthesis`.
- 🟡 **OCR** (TrOCR) — `load("printed")` + `recognize()` runs with real
  `microsoft/trocr-small-printed` weights and returns a decoded string. A residual
  decoder weight-mapping gap (75 MLP params absent from the checkpoint) makes the
  text repetitive, so content isn't yet correct. Covered by
  `tests/integration/test_trocr_small.py::{test_printed_model_loading,test_basic_recognition}`.
  (Loader bug fixed here: short variant names like `"printed"` were passed raw to
  the tokenizer instead of resolving to the HF repo id.)
- 🟡 **CAD** (smolGenCad) — SMLX-native ~147M text-to-CAD model. The **generation
  pipeline is verified to produce real, well-formed output** end to end (a valid
  CAD command sequence + parseable CadQuery Python + JSON), covered by
  `tests/models/smolGenCad/test_generate.py`. It ships with **random-initialised
  weights** (no public checkpoint), so the CAD *content* is not yet meaningful.

Fully blocked (no public weights at all): **Chatterbox** (TTS), **Donut** (OCR).

## Quantization (SMLX value-add)

Any zoo model can be quantized with SMLX's system at load time:

```bash
smlx generate smollm2-360m "What is the capital of France?" -q 4bit
```

```python
from smlx.models import load, generate
m = load("smolvlm-500m", quantize="4bit")   # correct upstream impl + SMLX 4-bit
generate(m, "Describe this image.", image="photo.jpg")
```

Verified: SmolLM2-360M quantized to 4-bit still answers correctly ("Paris").

## Adding a model

Add a one-line entry to `ZOO` in `smlx/models/mlx_backend.py` (alias → repo +
modality + backend), then `smlx models verify <alias>`. If mlx-lm/mlx-vlm support
the architecture, no model code is needed.

## Not in the curated zoo (and why)

- **Hand-written model implementations** under `smlx/models/<Name>/` are **legacy**.
  Some are correct (nanoVLM, SmolVLM-256M/500M, Whisper_tiny, MiniLM were fixed and
  verified); others had forward-path bugs. New work should use the backend zoo.
- **TTS (Orpheus, Chatterbox)** — gated on real public weights; not yet verified to
  produce intelligible speech.
- **OCR (TrOCR, Donut)** — TrOCR's custom impl had a multi-bug DeiT/BART encoder
  (8 fixed; one encoder-numerics bug remains). Prefer an mlx-vlm OCR model
  (florence2/deepseek-ocr) via the zoo when OCR is needed.
