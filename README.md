# SMLX (smol MLX)

**Small models, big performance** — vision, language, audio, and multimodal models that run on Apple Silicon through one unified, verified API.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

SMLX is a curated zoo of small, efficient models for Apple's MLX framework,
tuned for the M-series (especially M4) and its unified-memory architecture. It is
purpose-built for:

- **On-device inference** — no cloud required; your data stays private
- **Memory efficiency** — every model fits the 36 GB unified-memory budget
- **Real, verified output** — a fail-closed gate proves each model produces
  correct results, not placeholder noise
- **Practical deployment** — quantization, KV-cache management, an
  OpenAI-compatible server, agents, and an evaluation suite

### How it works — no hand-written forward passes

SMLX does **not** re-implement model forward passes. Every curated model routes to
a maintained upstream implementation (or real deterministic code), behind a single
runner:

- **mlx-lm** — language models
- **mlx-vlm** — vision-language models (and OCR via SmolVLM)
- **mlx-whisper** — speech recognition
- **mlx-embeddings** — sentence embeddings
- **mlx-audio** — text-to-speech (Kokoro)
- **onnxruntime** — voice-activity detection (Silero)
- **transformers** — audio classification (AST / AudioSet)
- **deterministic CadQuery parser** — text-to-CAD

SMLX's value sits on top: a curated "smol" zoo, quantization (`smlx.quant`), a
unified API, an OpenAI-compatible server, agents, and the bench/eval/verify trust
layer.

### "smol" is performance-based, not a hard cap

A model qualifies on three *performance* gates on the M4 target — **memory** (fits
36 GB with headroom), **speed** (meets its modality's floor), and **correctness**
(produces real output, verified by the gate). Parameter count is a *guideline*
(target < 500M, prefer < 1B); a larger model is admitted as a **documented
performance exception** when it still fits memory and runs acceptably (e.g. the
~1.57B `moondream3`, the 2.2B SmolVLM2, the 2B Qwen2-VL 4-bit).

## Key Features

- **Unified runner** — one entrypoint (`smlx run` / `smlx.models.load`) for every
  modality, with a fail-closed correctness gate (`smlx run --verify`).
- **Quantization** (`smlx.quant`) — GPTQ, AWQ, DWQ, 4/8-bit, LoRA/DoRA, applied on
  top of the correct upstream model at load time.
- **OpenAI-compatible server** — `/v1/chat/completions`, `/v1/completions`,
  `/v1/audio/transcriptions`, `/v1/embeddings`, `/v1/models`, with streaming.
- **Agents** — ReAct, Chain-of-Thought, and Self-Consistency with tool integration.
- **Evaluation suite** — MathVista, MMMU, MMStar, OCRBench, perplexity, WER.

## Supported Models

Two verification surfaces, both real and fail-closed:

- **`smlx run --verify`** — the **runner registry (15 entries)**, covering every
  modality. Currently **15/15** produce real, correct output.
- **`smlx models verify`** — the **backend ZOO (10 models)**: LM/VLM/ASR/embeddings
  through mlx-lm / mlx-vlm / mlx-whisper / mlx-embeddings, with optional
  `--enforce-perf` speed floors.

See [`docs/MODEL_STATUS.md`](docs/MODEL_STATUS.md) for the live board.

### Language (mlx-lm)

- **SmolLM2-135M / -360M / -1.7B** — chat-capable language models
- **Qwen2.5-0.5B-Instruct** — 4-bit language model

### Vision-language (mlx-vlm)

- **SmolVLM-256M / -500M-Instruct** — compact vision-language understanding
- **nanoVLM**, **TinyLLaVA** — compact VLMs
- **moondream3** (4-bit, ~1.57B) — documented performance exception
- **SmolVLM2-2.2B-Instruct**, **Qwen2-VL-2B-Instruct (4-bit)** — larger VLMs

### Audio

- **Whisper-tiny** — speech recognition (mlx-whisper)
- **Kokoro** — text-to-speech (mlx-audio)
- **Silero VAD** — voice-activity detection (onnxruntime)
- **AST (AudioSet)** — audio classification (transformers)

### Embeddings / OCR / CAD

- **MiniLM / all-MiniLM-L6-v2** — sentence embeddings (mlx-embeddings)
- **OCR** — document text recognition (SmolVLM via mlx-vlm)
- **CAD** — text-to-CAD parametric generation (deterministic CadQuery parser)

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.9, < 3.13
- Xcode Command Line Tools

### Install from source

```bash
git clone https://github.com/LayerDynamics/smlx.git
cd smlx

# Using Conda (recommended)
conda env create -f environment.yml
conda activate smlx
pip install -e .

# Or using pip with optional extras
pip install -e ".[all]"          # All features
pip install -e ".[dev]"          # Development tools
pip install -e ".[audio]"        # TTS / VAD / audio-classification stack
pip install -e ".[server]"       # API server
```

> After cloning, run `git lfs pull` to fetch the LFS-tracked datasets under `data/`.

## Quick Start

SMLX gives you **one API** over a curated zoo of verified small models. Each model
runs through its correct upstream implementation; SMLX quantization applies on top.

```python
from smlx.models import load, generate

# Any zoo alias auto-routes to the right backend
m = load("smollm2-360m")
print(generate(m, "Explain quantum computing in simple terms.", max_tokens=100))
```

### Vision-language understanding

```python
from smlx.models import load, generate

vlm = load("smolvlm-256m")
print(generate(vlm, "What is in this image?", image="photo.jpg", max_tokens=40))
```

### Audio transcription

```python
from smlx.models import load
from smlx.models.mlx_backend import transcribe

asr = load("whisper-tiny")
print(transcribe(asr, "audio.wav"))
```

### Embeddings

```python
from smlx.models import load
from smlx.models.mlx_backend import embed

emb = load("minilm")
vectors = embed(emb, ["a cat sat on the mat", "a feline rested on the rug"])
```

### Quantization (SMLX value-add)

Quantize any zoo model at load time — it still runs through its correct upstream
implementation:

```python
from smlx.models import load, generate

m = load("smollm2-360m", quantize="4bit")   # correct impl + SMLX 4-bit
print(generate(m, "What is the capital of France?", max_tokens=24))  # -> Paris
```

### Run any model from the CLI

`smlx run` produces **real, correct** output across every modality. Audio/CAD/JSON
artifacts land in `data/output/`.

```bash
smlx run --list                                # every runnable model + what it needs
smlx run --verify                              # fail-closed correctness gate (15/15)
smlx run smollm2-135m --text "What is the capital of France?"   # -> Paris
smlx run smolvlm-256m -i photo.jpg --text "What is this?"       # VLM
smlx run whisper-tiny --audio clip.wav         # ASR
smlx run kokoro --text "Hello world"           # TTS -> data/output/kokoro.wav
smlx run ocr --document scan.png               # OCR (SmolVLM via mlx-vlm)
smlx run cad  --text "cylinder radius 5mm height 10mm"          # real CadQuery
smlx run --all --text "Hi" -i cat.jpg -a clip.wav -d scan.png
```

Verify the backend zoo (with optional speed floors):

```bash
smlx models list                 # the curated backend zoo
smlx models verify               # load + run every zoo model, assert real output
smlx models verify --enforce-perf
```

### Agent with tools

```python
from smlx.agents import ReActAgent
from smlx.agents.tools import ToolRegistry, calculator, get_time
from smlx.models import load

# load() returns a BackendModel: .model / .processor / .backend / .repo / .modality
bm = load("smollm2-135m")
registry = ToolRegistry()
registry.register(calculator)
registry.register(get_time)

agent = ReActAgent(bm.model, bm.processor, registry)
response = agent.run("What is 15 * 23, and what time is it?")
print(response.content)
```

### REST API server

```bash
# Start the OpenAI-compatible server (host/port via the CLI)
smlx server --host 0.0.0.0 --port 8000

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2-135m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Documentation

Guides live under `docs/`:

- **[Model Status](docs/MODEL_STATUS.md)** — the verified zoo board and backends
- **[Quick Start](docs/QUICKSTART.md)** — a 5-minute walkthrough
- **[Server API](docs/Server.md)** — REST API reference and deployment
- **[Agent System](docs/Agents.md)** — agent types, tools, reasoning patterns
- **[CLI Tools](docs/Tools.md)** — download, conversion, benchmarking
- **[Quantization](docs/Quant.md)** — quantization techniques and usage
- **[Evaluation](docs/EVALUATION.md)** — benchmark suites and the eval framework
- **[Memory Management](docs/MemoryManagement.md)** / **[Enhanced KV Cache](docs/EnhancedCache.md)**

## Examples

Every model runs through one real entrypoint — `smlx run` (see **Quick Start**).
Standalone scripts under `examples/` cover the non-model subsystems:

```bash
python examples/quant/fp4_comparison.py            # quantization
python examples/gym/dqn_cartpole.py                # RL / gym
python examples/server/openai_compatible.py        # OpenAI-compatible server
python examples/eval/vlm_eval_example.py           # evaluation
```

## CLI Tools

```bash
# Download models / datasets
smlx download --model mlx-community/SmolLM2-135M-Instruct
smlx download --models            # all curated models
smlx download --datasets          # evaluation datasets
smlx download --all

# Convert a HuggingFace model to MLX (optionally quantized)
smlx convert ./hf-model ./mlx-out --quantize 4bit

# Transcribe audio (Whisper via mlx-whisper)
smlx transcribe audio.wav --format srt

# Inspect bundled datasets
smlx data list
smlx data validate

# Benchmarks
smlx bench --list                 # available suites
smlx bench llm -m smollm2-135m
smlx bench quantization
```

## Development

```bash
pip install -e ".[dev]"

# Tests (see pytest.ini — filterwarnings=error, 60s timeout, strict markers)
python -m pytest
python -m pytest -m unit                  # unit tests only
python -m pytest -m "not slow"            # skip slow tests
python -m pytest tests/quant/test_gptq.py -v

# Code quality
black .
ruff check .
ruff check --fix .
mypy smlx/
```

### Project structure

```text
smlx/
├── agents/              # Agent system (ReAct, CoT, tools)
├── bench/               # Performance benchmarking
├── config/              # Per-model memory/perf presets, inclusion policy
├── evals/               # Evaluation benchmarks (MathVista, MMMU, OCRBench, ...)
├── models/              # Model layer
│   ├── runner.py            # Unified runner: REGISTRY + produce()/produce_all()
│   ├── runner_adapters.py   # Registers each alias with its real backend
│   ├── runner_verify.py     # Fail-closed per-modality correctness gate
│   ├── mlx_backend.py       # load/generate/transcribe/embed over upstream libs
│   ├── registry.py          # Legacy load_model() API (delegates to mlx_backend)
│   ├── cad.py               # Deterministic text-to-CAD parser
│   └── common/              # Shared layers (attention, MLP, MoE/switch)
├── quant/               # Quantization (GPTQ, AWQ, DWQ, LoRA/DoRA, 4/8-bit, ...)
├── server/              # OpenAI-compatible REST API
├── tools/               # CLI utilities (download, convert)
├── utils/               # Shared utilities (generation, sampling, cache, memory)
└── main.py              # Click CLI entry point

docs/                    # Documentation
examples/                # Subsystem usage examples
tests/                   # Test suite
resources/               # Reference implementations (study, do not import)
```

### Adding a model

There are **no per-model packages** — do not hand-reimplement forward passes.
Register one entry in `smlx/models/runner_adapters.py` that points an alias at a
real upstream repo + modality, then prove it:

```python
# in runner_adapters.py — e.g. another mlx-vlm model
_VLM_BACKEND["my-vlm"] = ("org/My-VLM-mlx-4bit", "My-VLM (mlx-vlm)")
```

```bash
smlx run --verify my-vlm          # must pass the fail-closed correctness gate
```

A model is admitted when it routes to a maintained upstream impl (or real
deterministic code), fits the 36 GB budget, and passes `smlx run --verify`.
Parameter count is a guideline (prefer < 1B); larger models are documented
performance exceptions.

## Performance

Indicative throughput on M4 Pro (36 GB unified memory); not a committed floor —
run `smlx models verify --enforce-perf` for the calibrated speed gates.

| Model | Parameters | Backend | Tokens/sec |
|-------|-----------|---------|------------|
| SmolLM2-135M | 135M | mlx-lm | ~115 |
| SmolLM2-360M | 360M | mlx-lm | ~114 |
| SmolVLM-256M | 256M | mlx-vlm | ~55 |
| SmolVLM-500M | 500M | mlx-vlm | ~52 |
| Whisper-tiny | 39M | mlx-whisper | ~1.3 s/clip |
| MiniLM | 23M | mlx-embeddings | ~291 sent/s |

## Resources

The `resources/` directory contains reference implementations from MLX-ecosystem
projects (mlx, mlx-examples, mlx-lm, mlx-vlm, lightning-whisper-mlx) for learning
and pattern-borrowing.

**Important:** these are for reference only — do not import them directly. Study
the patterns and route through a maintained upstream library instead. See
[docs/RESOURCES_QUICK_START.md](docs/RESOURCES_QUICK_START.md) and
[docs/RESOURCES_REFERENCE_MAP.md](docs/RESOURCES_REFERENCE_MAP.md).

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes (add a runner entry for a new model — no bespoke forward pass)
4. Add tests and update documentation
5. Run `black`, `ruff`, and the relevant tests
6. Ensure `smlx run --verify <alias>` passes for any new model
7. Submit a pull request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

## Acknowledgments

- **Apple MLX Team** — for the MLX framework and the mlx-lm / mlx-vlm / mlx-whisper
  / mlx-embeddings / mlx-audio libraries
- **HuggingFace** — for model hosting, tokenizers, and `transformers`
- **MLX Community** — for the reference implementations under `resources/`

## Citation

```bibtex
@software{smlx,
  title = {SMLX: Small Models for Apple Silicon},
  author = {LayerDynamics},
  url = {https://github.com/LayerDynamics/smlx}
}
```

---

**Built for Apple Silicon**
