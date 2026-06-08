# SMLX Model Status

Single source of truth for the curated model zoo. Every entry here is verified by
the fail-closed correctness gate — it loads the model through its real backend, runs
it, and checks the output is **real** (coherent / correct), not placeholder.

Re-verify at any time:

```bash
smlx run --verify                 # all models (non-zero exit on any failure)
smlx run --verify moondream3      # one model
smlx run --list                   # the curated zoo + what each needs
```

## Run any model (`smlx run`) — 100% real, gate-verified

**Every** entry in `smlx run --list` produces real, correct output. There are **no
bespoke hand-written forward passes** — each entry routes to a maintained upstream
implementation or a real deterministic one, and a **fail-closed correctness gate**
(`smlx run --verify`) proves it:

```bash
smlx run --list                                   # every runnable model + what it needs
smlx run --verify                                 # real per-modality correctness gate
smlx run smollm2-135m --text "What is the capital of France?"   # -> Paris
smlx run ocr --document scan.png                  # real OCR (SmolVLM via mlx-vlm)
smlx run kokoro --text "Hello world"              # real TTS -> data/output/kokoro.wav
smlx run cad  --text "cylinder radius 5mm height 10mm"          # real CadQuery
smlx run --all --text "Hi" -i cat.jpg -a clip.wav -d scan.png
```

### Verified entries (`smlx run --verify` → 15/15)

| Entry | Modality | Real backing | Correctness check |
|-------|----------|--------------|-------------------|
| `smollm2-135m`, `smollm2-360m` | language | mlx-lm | capital-of-France → "Paris" |
| `smolvlm-256m`, `smolvlm-500m`, `nanovlm`, `tinyllava`, `moondream3` | vision-language | mlx-vlm | colour discrimination (red→red, green→green) |
| `whisper-tiny` | ASR | mlx-whisper | transcribes real speech (keyword match) |
| `kokoro` | TTS | mlx-audio (Kokoro-82M) | synth → Whisper round-trips the text |
| `ocr` | OCR | mlx-vlm (SmolVLM-500M) | rendered text → exact match |
| `silero-vad` | VAD | onnxruntime (Silero v5) | real speech → speech segment |
| `ast` | audio-cls | transformers (AST AudioSet) | speech → "Speech" |
| `minilm`, `all-minilm-l6-v2` | embeddings | mlx-embeddings | paraphrase cos > unrelated cos |
| `cad` | CAD | deterministic parser → CadQuery | spec → solid with expected bbox |

The gate generates its speech fixture with the real Kokoro TTS, so ASR/VAD/audio-cls
are checked against genuine speech.

`moondream3` is the `beshkenadze/moondream3-preview-mlx-4bit` checkpoint (~1.57B,
within the documented VLM performance-exception band). Its tokenizer leaks a
`<|md_reserved_4|>` special-token prefix into the raw text via mlx-vlm; the colour
answer is correct and the gate passes.

### Quarantined (NOT wired into `smlx run` — would produce garbage)

The earlier bespoke SMLX implementations mishandled real weights (or had none); they
were **removed** rather than faked. Each modality is covered by a real replacement:

| Removed package | Why it was unusable | Real replacement |
|-----------------|---------------------|------------------|
| `Orpheus_150M`, `Chatterbox` (TTS) | no public checkpoint → noise | `kokoro` |
| `TrOCR_small`, `Donut_base` (OCR) | architecture diverged from real TrOCR/BART → gibberish | `ocr` |
| `YAMNet` (audio-cls) | log-mel front-end didn't match weights → wrong labels | `ast` |
| `smolGenCad` (CAD) | no trained text-to-CAD checkpoint → random | `cad` (deterministic) |
| `Moondream2` | original Moondream2 arch unsupported by mlx-vlm | `moondream3` (4-bit) |

## Architecture

SMLX runs each model through the **correct upstream implementation** and layers its
own value on top — it does **not** re-implement model forward passes:

- **mlx-lm** — language models
- **mlx-vlm** — vision-language models (incl. OCR via SmolVLM)
- **mlx-whisper** — ASR
- **mlx-embeddings** — sentence embeddings
- **mlx-audio** — TTS (Kokoro)
- **onnxruntime** — VAD (Silero)
- **transformers** — audio classification (AST AudioSet)
- **deterministic CadQuery parser** — text-to-CAD

SMLX's differentiators: **quantization** (`smlx.quant`: 4/8-bit, GPTQ, AWQ, DWQ),
**smol curation** (this zoo), a **unified API** (`smlx.models.load` / `generate` and
the legacy `load_model`), and the **bench/eval/verify** trust layer. Inclusion is by
*performance on the M4 target*, not a hard parameter cap — a larger model (e.g. the
~1.57B `moondream3`) is admitted as a documented performance exception when it fits
memory and passes the gate.

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

Add a one-line entry to the appropriate registration block in
`smlx/models/runner_adapters.py` (alias → repo + modality), then
`smlx run --verify <alias>`. If mlx-lm / mlx-vlm support the architecture, no model
code is needed — the entry routes straight to the upstream backend via
`smlx/models/mlx_backend.py`.
