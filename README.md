# SMLX (smol MLX)

**Small models, big performance** - Vision, language, audio, and multimodal models optimized for Apple Silicon M4 with MLX.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

SMLX focuses exclusively on **small, efficient models** (< 1B parameters) that run exceptionally well on Apple's M4 chipset with unified memory architecture. Unlike general-purpose ML frameworks, SMLX is purpose-built for:

- **On-device inference** - No cloud required, your data stays private
- **Memory efficiency** - Models fit in 8-36GB unified memory
- **Optimized performance** - Built specifically for M4's architecture
- **Practical deployment** - Production-ready with quantization, caching, and batching

## Key Features

### 🎯 Small Model Focus

All models are "smol" (< 1B parameters), ensuring fast inference and low memory usage on consumer hardware.

### 🚀 MLX-Native

Built from the ground up using Apple's MLX framework for optimal performance on Apple Silicon.

### 🔧 Quantization Support

Built-in support for:

- **GPTQ** - Post-training quantization for language models
- **AWQ** - Activation-aware weight quantization
- **Dynamic Quantization** - Runtime weight quantization
- **LoRA/DoRA** - Parameter-efficient fine-tuning

### 🌐 Production-Ready

Complete server infrastructure with:

- OpenAI-compatible REST API
- Streaming responses
- Model management and caching
- Authentication and rate limiting
- Docker/Kubernetes deployment

### 🤖 Agent System

Sophisticated reasoning with:

- **ReAct** - Reasoning + Acting agents
- **Chain-of-Thought** - Step-by-step reasoning
- **Self-Consistency** - Multiple reasoning paths
- Tool integration and custom tool creation

### 📊 Evaluation Suite

Built-in benchmarks for:

- Math-Vision-Language tasks (MathVista)
- Multimodal understanding (MMMU, MMStar)
- OCR capabilities (OCRBench)
- Custom evaluation pipelines

## Supported Models

### Language Models (LLMs)

The **curated zoo** below is verified end to end by `smlx models verify` — every
entry loads, runs, and produces real (coherent / correct) output. See
[`docs/MODEL_STATUS.md`](docs/MODEL_STATUS.md) for the live board and tok/s.

- **SmolLM2-135M / -360M / -1.7B** ✓ — chat-capable language models (mlx-lm)
- **Qwen2.5-0.5B-Instruct** ✓ — 4-bit language model (mlx-lm)

### Vision-Language Models (VLMs)

- **SmolVLM-256M / -500M-Instruct** ✓ — compact vision-language understanding (mlx-vlm)
- **SmolVLM2-2.2B-Instruct** ✓ — larger multimodal model (mlx-vlm)
- **Qwen2-VL-2B-Instruct** ✓ — 4-bit VLM (mlx-vlm)

### Audio Models

- **Whisper-tiny** ✓ — speech recognition (SMLX-native MLX impl)

### Embedding Models

- **MiniLM / all-MiniLM-L6-v2** ✓ — sentence embeddings (SMLX-native MLX impl)

### Not yet in the verified zoo

Legacy hand-written implementations exist for **Moondream2, TinyLLaVA, nanoVLM,
TrOCR-small, Donut-base, Orpheus-150M (TTS), Chatterbox (TTS), YAMNet, Silero VAD**.
Some are correct and some have known forward-path issues — they are not yet covered
by `smlx models verify` and are documented in
[`docs/MODEL_STATUS.md`](docs/MODEL_STATUS.md). Prefer the curated zoo above.

## Installation

### Requirements

- Python >= 3.9, < 3.13
- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools

### Install from source

```bash
# Clone repository
git clone https://github.com/yourusername/smlx.git
cd smlx

# Using Conda (recommended)
conda env create -f environment.yml
conda activate smlx

# Or using pip
pip install -e .

# With optional dependencies
pip install -e ".[all]"          # All features
pip install -e ".[dev]"          # Development tools
pip install -e ".[evals]"        # Evaluation suite
pip install -e ".[server]"       # API server
```

## Quick Start

SMLX gives you **one API** over a curated zoo of verified small models. Each model
runs through its correct upstream MLX implementation (mlx-lm / mlx-vlm), and SMLX's
quantization applies on top. See [`docs/MODEL_STATUS.md`](docs/MODEL_STATUS.md) for the
full verified board (10/10 models pass `smlx models verify`).

```python
from smlx.models import load, generate

# Any zoo alias — auto-routes to the right backend
m = load("smollm2-360m")
print(generate(m, "Explain quantum computing in simple terms.", max_tokens=100))
```

### Vision-Language Understanding

```python
from smlx.models import load, generate

vlm = load("smolvlm-256m")
print(generate(vlm, "What is in this image?", image="photo.jpg", max_tokens=40))
```

### Audio Transcription

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

### Model Quantization (SMLX value-add)

Quantize any zoo model with SMLX's own system at load time — the model still runs
through its correct upstream implementation:

```python
from smlx.models import load, generate

m = load("smollm2-360m", quantize="4bit")   # correct impl + SMLX 4-bit
print(generate(m, "What is the capital of France?", max_tokens=24))  # -> Paris
```

### Verify it yourself

```bash
smlx models list              # the curated zoo
smlx models verify            # load + run every model, check output is REAL
smlx generate smolvlm-256m "What's this?" -i photo.jpg -q 4bit
```

### Run any model (`smlx run`)

`smlx run` produces **real, correct** output across every modality (language, VLM,
ASR, TTS, OCR, VAD, audio-classification, embeddings, CAD). There are **no bespoke
hand-written forward passes** — each entry routes to a maintained upstream impl
(mlx-lm / mlx-vlm / mlx-whisper / mlx-embeddings / mlx-audio / onnxruntime /
transformers) or a real deterministic one, and a **fail-closed correctness gate**
proves it. Audio/CAD/json artifacts land in `data/output/`. See
[`docs/MODEL_STATUS.md`](docs/MODEL_STATUS.md#run-any-model-smlx-run--100-real-gate-verified).

```bash
smlx run --list                                # every runnable model + what it needs
smlx run --verify                              # real per-modality correctness gate (14/14)
smlx run smollm2-135m --text "What is the capital of France?"   # -> Paris
smlx run kokoro --text "Hello world"           # real TTS -> data/output/kokoro.wav
smlx run ocr --document scan.png               # real OCR (SmolVLM via mlx-vlm)
smlx run cad  --text "cylinder radius 5mm height 10mm"          # real CadQuery
smlx run --all --text "Hi" -i cat.jpg -a clip.wav -d scan.png
```

### Agent with Tools

```python
from smlx.agents import ReActAgent
from smlx.agents.tools import ToolRegistry, calculator, get_time
from smlx.models.SmolLM2_135M import load

# Setup
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
registry = ToolRegistry()
registry.register(calculator)
registry.register(get_time)

# Create agent
agent = ReActAgent(model, tokenizer, registry)

# Run task
response = agent.run("What is 15 * 23, and what time is it?")
print(response.content)
```

### REST API Server

```bash
# Start server
python -m smlx.server.app --host 0.0.0.0 --port 8000

# Use with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Documentation

Comprehensive guides are available in the `docs/` directory:

- **[Model Implementations](docs/ModelImplementations.md)** - Guide to implementing new models
- **[Server API](docs/Server.md)** - REST API reference and deployment
- **[Agent System](docs/Agents.md)** - Agent types, tools, and reasoning patterns
- **[CLI Tools](docs/Tools.md)** - Model download, conversion, and benchmarking
- **[Quantization](docs/Quant.md)** - Quantization techniques and usage
- **[Evaluation](docs/Eval.md)** - Benchmark suites and evaluation framework

## Examples

Working examples demonstrating all features:

```bash
# Language model examples
python examples/models/smollm2_135m/smollm2_135m_example.py

# Audio transcription
python examples/whisper_tiny/basic_transcription.py

# OCR examples
python examples/models/trocr_small/trocr_example.py

# Quantization examples
python examples/quant/gptq_example.py
python examples/quant/lora_example.py

# Evaluation examples
python examples/eval/mmmu_eval.py

# Agent examples
python examples/agents/react_agent_example.py
```

## CLI Tools

### Download Models and Datasets

```bash
# Download specific model
python -m smlx.tools.download_data --model mlx-community/SmolLM2-135M-Instruct

# Download all models
python -m smlx.tools.download_data --models

# Download evaluation datasets
python -m smlx.tools.download_data --datasets

# Download everything
python -m smlx.tools.download_data --all
```

### Convert Models to MLX

```bash
# Convert with quantization
python -m smlx.tools.convert2mlx \
  --hf-path gpt2 \
  --output-path ./models/gpt2-4bit \
  --quantize \
  --bits 4 \
  --group-size 64
```

### Run Benchmarks

```bash
# Run all benchmarks
python -m smlx.bench.run

# Run specific suite
python -m smlx.bench.run --suite llm
python -m smlx.bench.run --suite vlm
python -m smlx.bench.run --suite quantization

# Compare results
python -m smlx.tools.compare_results \
  results/baseline.json \
  results/optimized.json
```

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Run specific test suite
python -m pytest tests/quant/test_gptq.py -v

# Run with markers
python -m pytest -m unit                 # Unit tests only
python -m pytest -m "not slow"           # Skip slow tests
python -m pytest -m integration          # Integration tests

# Code quality
black .                                  # Format code
ruff check .                             # Lint code
ruff check --fix .                       # Auto-fix issues
mypy smlx/                               # Type check
```

### Project Structure

```text
smlx/
├── agents/              # Agent system (ReAct, CoT, tools)
├── bench/               # Performance benchmarking
├── evals/               # Evaluation benchmarks
├── models/              # Model implementations
│   ├── SmolLM2_135M/    # Language models
│   ├── SmolVLM_256M/    # Vision-language models
│   ├── Whisper_tiny/    # Audio models
│   ├── TrOCR_small/     # Document models
│   └── MiniLM/          # Embedding models
├── quant/               # Quantization (GPTQ, AWQ, LoRA)
├── server/              # REST API server
├── tools/               # CLI utilities
└── utils/               # Shared utilities

docs/                    # Documentation
examples/                # Usage examples
tests/                   # Test suite
resources/               # Reference implementations (do not import)
```

### Adding New Models

See [docs/ModelImplementations.md](docs/ModelImplementations.md) for detailed guidelines. Quick checklist:

1. Create model directory in `smlx/models/YourModel/`
2. Implement core modules (config.py, model.py, loader.py, generate.py)
3. Add example in `examples/models/your_model/`
4. Add integration test in `tests/integration/`
5. Update documentation

**Requirements:**

- Must be "smol" (< 1B parameters preferred)
- Must use MLX operations
- Must support quantization
- Must follow existing API patterns

## Performance

SMLX models are optimized for Apple Silicon with impressive performance on M4:

| Model | Parameters | Memory | Tokens/sec | Quantization |
|-------|-----------|---------|------------|--------------|
| SmolLM2-135M | 135M | ~500MB | ~150 | 4-bit/8-bit |
| SmolLM2-360M | 360M | ~1.3GB | ~100 | 4-bit/8-bit |
| SmolVLM-256M | 256M | ~1GB | ~80 | 4-bit/8-bit |
| SmolVLM-500M | 500M | ~2GB | ~60 | 4-bit/8-bit |
| Whisper-tiny | 39M | ~150MB | Real-time | 8-bit |

*Benchmarks on M4 Pro with 36GB unified memory*

## Use Cases

### On-Device AI

- Privacy-sensitive applications
- Offline-first mobile/desktop apps
- Edge computing scenarios

### Rapid Prototyping

- Quick experimentation with small models
- Testing architectures before scaling
- Educational projects

### Production Deployment

- Low-latency inference APIs
- Cost-effective model serving
- Resource-constrained environments

## Resources

The `resources/` directory contains reference implementations from various MLX projects for learning and pattern-borrowing:

- `mlx/` - Core MLX framework
- `mlx-examples/` - MLX examples
- `mlx-lm/` - Language models
- `mlx-vlm/` - Vision-language models
- `lightning-whisper-mlx/` - Whisper implementation

**Important:** These are for reference only - do not import directly. Study patterns and adapt code into `smlx/` modules.

See [RESOURCES_QUICK_START.md](RESOURCES_QUICK_START.md) for a fast implementation guide, and [RESOURCES_REFERENCE_MAP.md](RESOURCES_REFERENCE_MAP.md) for exact code patterns.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-model`)
3. Implement your changes
4. Add tests and documentation
5. Run code quality checks (`black`, `ruff`, `mypy`)
6. Submit a pull request

**Model Contributions:** We only accept models that are "smol" (< 1B parameters). Please ensure your model:

- Is properly quantized and optimized
- Includes comprehensive tests
- Has working examples
- Follows the existing API patterns

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Apple MLX Team** - For the excellent MLX framework
- **HuggingFace** - For model hosting and tokenizers
- **MLX Community** - For reference implementations in mlx-examples, mlx-lm, mlx-vlm

## Citation

If you use SMLX in your research or project, please cite:

```bibtex
@software{smlx2024,
  title = {SMLX: Small Models for Apple Silicon},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/smlx}
}
```

---

**Built for Apple Silicon**
