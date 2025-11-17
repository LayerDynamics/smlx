# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ Important Notes - READ FIRST

**Critical Rules:**

1. **NEVER import from `resources/`** - The resources directory contains reference implementations ONLY. Study patterns, then copy and adapt code into `smlx/` modules.

2. **Always activate conda environment** - Before running any commands:

   ```bash
   conda activate smlx
   ```

3. **Python path for all commands**: `/Users/ryanoboyle/miniforge3/envs/smlx/bin/python`

4. **"Smol" models only** - All models must be < 1B parameters (< 500M preferred)

5. **Don't remove stub files** - If something is called but missing, it should be implemented, not removed (per global instructions)

6. **Code style**: Line length = 100 characters (configured in pyproject.toml for both black and ruff)

## Project Overview

SMLX (smol MLX) is a Python package focused on implementing small models (vision, voice, and multimodal) using Apple's MLX framework, specifically optimized for M4 chipsets with unified memory (36GB).

**Core Requirement**: All models in this project must be "smol" (small/lightweight).

## Architecture

### Module Structure

The project is organized into the following modules:

- **`smlx/agents/`** - Agent implementations with:
  - `base.py` - Base agent class (Message, AgentResponse, BaseAgent)
  - `react.py` - ReAct (Reasoning + Acting) agent
  - `cot.py` - Chain-of-Thought reasoning agent
  - `memory.py` - Agent memory management
  - `tools.py` - Tool registry and tool implementations

- **`smlx/bench/`** - Performance benchmarking with:
  - `cli.py` - Command-line interface for benchmarks
  - `runners.py` - Benchmark execution runners
  - `stats.py` - Statistical analysis
  - `system.py` - System information gathering
  - `report.py` - Benchmark report generation
  - `run.py` - Main benchmark orchestration
  - `suites/` - Benchmark suites:
    - `llm.py` - Language model benchmarks (prompt/generation tokens, TTFT)
    - `vlm.py` - Vision-language model benchmarks
    - `quantization.py` - Quantization performance tests
    - `text_generation.py` - Text generation quality/speed
    - `ops.py` - Low-level MLX operation benchmarks

- **`smlx/evals/`** - Evaluation benchmarks including:
  - `math_vista.py` - Math-Vision-Language tasks
  - `mmmu.py` - Multimodal understanding
  - `mmstar.py` - Multimodal reasoning
  - `ocrbench.py` - OCR benchmarks
  - `utils.py` - Evaluation utilities

- **`smlx/models/`** - Model implementations (see "Available Models" below)

- **`smlx/quant/`** - Quantization techniques:
  - `awq.py` - Activation-aware Weight Quantization
  - `gptq.py` - GPT Quantization
  - `dwq.py` - Dynamic Weight Quantization
  - `dynamic_quant.py` - Dynamic quantization
  - `lora.py` - Low-Rank Adaptation
  - `dora.py` - Weight-Decomposed Low-Rank Adaptation

- **`smlx/server/`** - FastAPI-based REST API server with:
  - `app.py` - Main FastAPI application with lifespan management
  - `model_manager.py` - Model loading and caching
  - `schemas.py` - Pydantic request/response schemas
  - `middleware/` - Server middleware:
    - `error_handling.py` - Global error handling
    - `logging.py` - Request/response logging
    - `rate_limit.py` - Rate limiting
  - `routes/` - API endpoints (OpenAI-compatible):
    - `chat.py` - Chat completions endpoint
    - `completions.py` - Text completions endpoint
    - `audio.py` - Audio transcription endpoint
    - `embeddings.py` - Text embeddings endpoint
    - `models.py` - Model listing endpoint

- **`smlx/tools/`** - Utility tools including:
  - `convert2mlx.py` - Convert models to MLX format
  - `download.py` - Core download utilities
  - `download_data.py` - CLI for downloading models and datasets
  - `compare_results.py` - Compare benchmark results

- **`smlx/utils/`** - General utilities (see below for details)

- **`smlx/gym/`** - Gymnasium/RL environment integrations (planned - see docs/GymResearch/Gym_Claude.md for design)

### Data Directory Structure

The `data/` directory is tracked with Git LFS and contains datasets, models, and test files:

- **`data/audio/`** - Audio files for testing (speech, TTS, environmental sounds)
- **`data/benchmark/`** - Benchmark datasets and reference results
- **`data/datasets/`** - Evaluation datasets (MathVista, MMMU, OCRBench, etc.)
- **`data/documents/`** - Document images for OCR testing
- **`data/images/`** - Test images for vision models

**Important**: These files are stored with Git LFS. Use `git lfs pull` after cloning to download them.

### Server Architecture (FastAPI)

The server follows a standard FastAPI pattern:

1. **Application Lifecycle** (`app.py`):
   - `lifespan()` context manager handles startup/shutdown
   - Global `ModelManager` instance for model caching
   - CORS middleware for cross-origin requests

2. **Model Management** (`model_manager.py`):
   - Lazy model loading (models loaded on first request)
   - Model caching to avoid reloading
   - Automatic model discovery from `smlx.models/`

3. **Request Flow**:

   ```text
   Client Request → Middleware (logging, rate limit, error handling)
                 → Route Handler → Model Manager → Model Inference
                 → Response (JSON or Streaming)
   ```

4. **API Endpoints** (OpenAI-compatible):
   - `POST /v1/chat/completions` - Chat with message history
   - `POST /v1/completions` - Simple text completion
   - `POST /v1/audio/transcriptions` - Audio transcription (Whisper)
   - `POST /v1/embeddings` - Text embeddings
   - `GET /v1/models` - List available models

5. **Running the Server**:

   ```bash
   python -m smlx.server.app --host 0.0.0.0 --port 8000
   # Or with uvicorn:
   uvicorn smlx.server.app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Agent System Architecture

The agent system provides autonomous task execution with tool use:

1. **Base Components** (`base.py`):
   - `Message` - Conversation messages (role, content, timestamp, metadata)
   - `AgentResponse` - Agent execution results (content, reasoning, tool_calls, success)
   - `BaseAgent` - Abstract base class for all agents

2. **Agent Types**:
   - **ReAct Agent** (`react.py`) - Reasoning + Acting pattern
     - Iteratively reasons about next action
     - Executes tools
     - Updates based on observations
   - **Chain-of-Thought** (`cot.py`) - Step-by-step reasoning
     - Zero-shot or few-shot prompting
     - Structured thinking process
   - **Self-Consistency CoT** - Multiple reasoning paths, voting

3. **Tool System** (`tools.py`):
   - `ToolRegistry` - Central tool registration and discovery
   - Built-in tools: calculator, get_time, wikipedia_search, etc.
   - Custom tool creation via decorators

4. **Memory Management** (`memory.py`):
   - Conversation history tracking
   - Context window management
   - Memory summarization for long interactions

### Benchmark Suite Structure

The benchmark system supports multiple specialized suites:

1. **LLM Benchmarks** (`suites/llm.py`):
   - Configurable prompt/generation tokens
   - Time to first token (TTFT) measurement
   - Tokens/second throughput
   - Batch processing benchmarks
   - Memory usage tracking

2. **VLM Benchmarks** (`suites/vlm.py`):
   - Vision-language model performance
   - Image processing overhead
   - Multimodal generation speed

3. **Quantization Benchmarks** (`suites/quantization.py`):
   - Compare FP16 vs 4-bit vs 8-bit performance
   - Memory reduction measurements
   - Quality degradation analysis

4. **Text Generation Benchmarks** (`suites/text_generation.py`):
   - Generation quality metrics
   - Different sampling strategies
   - Temperature/top-p effects

5. **Ops Benchmarks** (`suites/ops.py`):
   - Low-level MLX operation performance
   - Matrix operations, attention, layer norms
   - Metal GPU utilization

**Running Benchmarks**:

```bash
# List available benchmark suites
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run --list

# Display system information
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run system

# Specific suite
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run llm --model SmolLM2-135M
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run quantization --model SmolLM2-135M

# Operation benchmarks
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run ops --operation matmul --shape 1000,1000

# Run all benchmarks
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run --all --model SmolLM2-135M
```

### Utility Modules

The `smlx/utils/` directory provides reusable utilities across the project:

- `stats.py` - Statistical utilities for analysis
- `memory.py` - Memory profiling and optimization
- `timing.py` - Performance timing utilities
- `config.py` - Configuration management helpers
- `cache.py` - Caching utilities (shared KV cache implementations)
- `formatting.py` - Output formatting utilities
- `io.py` - Input/output helpers
- `sampling.py` - Token sampling utilities (temperature, top-p, top-k)
- `generation.py` - Text generation utilities
- `loading.py` - Model loading helpers
- `vision.py` - Vision utilities (image loading, preprocessing, batch processing)
- `batch.py` - Batch processing utilities
- `profiling.py` - Performance profiling utilities
- `quantization.py` - Quantization helper utilities

**When implementing new models**: Prefer using these shared utilities over reimplementing common functionality.

### Available Models

The project currently includes the following "smol" models:

**Fully Implemented (✓):**

**Language Models:**

- **SmolLM2_135M** - 135M parameter language model with generation, streaming, and chat
- **SmolLM2_360M** - 360M parameter variant with improved capabilities

**Vision-Language Models:**

- **SmolVLM_256M** - 256M parameter VLM with SigLIP vision + SmolLM2 language
- **SmolVLM_500M_Instruct** - 500M parameter VLM variant with larger capacity
- **nanoVLM** - 222M parameter minimal VLM (SigLIP-base + MLP + SmolLM2-135M)
- **Moondream2** - ~500M parameter VLM with region detection and spatial reasoning
- **TinyLLaVA** - Compact LLaVA variant for vision-language tasks

**Audio Models:**

- **Whisper_tiny** - Lightweight speech recognition with comprehensive support
- **Chatterbox** - 500M parameter TTS model with voice cloning and emotion control (reference implementation)
- **Orpheus_150M** - 150M parameter lightweight TTS model (reference implementation)
- **YAMNet** - Audio event classification
- **SileroVAD** - Voice activity detection

**Document/OCR Models:**

- **TrOCR_small** - Optical character recognition for printed and handwritten text
- **Donut_base** - OCR-free document understanding with Swin Transformer + BART (reference implementation)

**Embedding Models:**

- **MiniLM** - Efficient text embeddings
- **all-MiniLM-L6-v2** - Sentence embeddings

**Notes on Reference Implementations:**

Some models (Chatterbox, Orpheus_150M, Donut_base) are complete reference implementations showing proper API structure and architecture patterns. They require pre-trained weights from HuggingFace Hub to produce actual outputs. All have:
- Complete model architectures
- Full API implementations
- Comprehensive example files
- Integration tests

**Planned/New Models:**

- **smolGenCad** - CAD generation model (early development)

**Fully Implemented Model Structure (SmolLM2_135M as reference):**

- `model.py` - Core model architecture (Attention, MLP, TransformerBlock, NoPE)
- `loader.py` - Model and tokenizer loading from HuggingFace Hub
- `generate.py` - Text generation with streaming support and chat interface
- `cache.py` - KV cache implementations (standard and rotating)
- `config.py` - Configuration management and validation
- `__init__.py` - Public API exports

**Fully Implemented Audio Model Structure (Whisper-tiny as reference):**

- `model.py` - Core Whisper architecture
- `audio.py` - Audio preprocessing and feature extraction
- `transcribe.py` - Transcription interface
- `streaming.py` - Real-time streaming support
- `decoding.py` - Advanced decoding strategies
- `vad.py` - Voice activity detection
- `tokenizer.py` - Whisper tokenization
- `timing.py` - Timestamp alignment

### Resources Directory - CRITICAL RULE

The `resources/` directory contains reference implementations from various MLX projects:

- `mlx/` - Core MLX framework
- `mlx-examples/` - MLX example implementations
- `mlx-lm/` - MLX language models
- `mlx-vlm/` - MLX vision-language models
- `mflux/` - Flux implementation
- `chat-with-mlx/` - Chat interface
- `lightning-whisper-mlx/` - Whisper speech recognition
- `ml-aim/` - AIM models
- `transformerlab-app/` - Transformer lab application

**IMPORTANT**: Code in `resources/` is for reference and borrowing patterns ONLY. Do NOT import directly from these modules. Instead:

1. Study the implementation patterns
2. Adapt and copy relevant code into the appropriate `smlx/` module
3. Ensure the implementation is suitable for "smol" models

**Comprehensive Pattern Documentation:**

To effectively use the resources directory, consult these comprehensive pattern guides:

- **[RESOURCES_INDEX.md](RESOURCES_INDEX.md)** - Start here: Quick guide to using the three pattern documents
- **[RESOURCES_QUICK_START.md](RESOURCES_QUICK_START.md)** - Fast implementation checklist (5 min read)
- **[RESOURCES_REFERENCE_MAP.md](RESOURCES_REFERENCE_MAP.md)** - Exact code patterns with line numbers for copy-paste
- **[RESOURCES_PATTERNS.md](RESOURCES_PATTERNS.md)** - Deep dive into all architectural patterns (20 min read)

**Model Conversion Guides** (for converting external models to MLX):

- **[CONVERSION_INDEX.md](CONVERSION_INDEX.md)** - Step-by-step conversion workflow
- **[CONVERSION_PATTERNS.md](CONVERSION_PATTERNS.md)** - Common PyTorch→MLX patterns
- **[CONVERSION_REFERENCE.md](CONVERSION_REFERENCE.md)** - Detailed operation mapping reference

**When implementing new models**:

1. If converting from PyTorch/TensorFlow: Start with `CONVERSION_INDEX.md`
2. For MLX-native implementation: Study `RESOURCES_QUICK_START.md` first, then use `RESOURCES_REFERENCE_MAP.md` for exact code to copy

## Development Commands

### Environment Setup

**Requirements**: Python >=3.9, <3.13

```bash
# Using Conda (recommended for M4 Macs)
conda env create -f environment.yml
conda activate smlx

# Using pip
pip install -e .                      # Install base package in editable mode
pip install -e ".[dev]"               # Install with development dependencies
pip install -e ".[all]"               # Install with all optional dependencies
pip install -e ".[evals,server]"      # Install specific optional groups
```

Available optional dependency groups: `dev`, `evals`, `server`, `quant`, `agents`, `vision`, `audio`, `mlx-ecosystem`

### Build System

The project uses modern Python packaging with configuration primarily in `pyproject.toml`:

- **pyproject.toml** - All package metadata, dependencies, and tool configurations (black, ruff, mypy)
- **setup.py** - Minimal wrapper for backwards compatibility and editable installs
- **MANIFEST.in** - Controls which files are included in distributions (excludes `resources/`)

No separate build step is required - the package is installed directly via `pip install -e .`

### Testing

**Python Path:** `/Users/ryanoboyle/miniforge3/envs/smlx/bin/python`

```bash
# Run all tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest

# Run specific test file
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/test_gptq.py -v
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/integration/test_smollm2_generation.py -v

# Run tests matching pattern
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -k test_function_name

# Run tests with markers
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m unit                        # Run only unit tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m "not slow"                  # Skip slow tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m "not gpu"                   # Skip tests requiring GPU
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m integration                 # Run only integration tests

# Run specific test suites
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/ -v               # All quantization tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/evals/ -m eval          # Evaluation tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/integration/ -v         # Integration tests

# Useful test options
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -v                            # Verbose output
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -x                            # Stop on first failure
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest --durations=10                # Show 10 slowest tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -n auto                       # Parallel execution

# Coverage (when enabled in pytest.ini)
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest --cov=smlx --cov-report=html  # Generate coverage report
```

### Code Quality

**Code Style Configuration:**

- **Line length**: 100 characters (configured in pyproject.toml)
- **Formatter**: Black (line-length = 100)
- **Linter**: Ruff (line-length = 100)
- **Type checker**: MyPy

```bash
# Format code
black .                                                                # Format all Python files
black smlx/                                                            # Format only smlx package

# Lint code
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m ruff check .     # Check all files
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m ruff check smlx/ # Check only smlx package
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m ruff check --fix .  # Auto-fix issues

# Type checking
mypy smlx/                                                             # Run type checker on package
```

### Tools & Utilities

```bash
# Download models and datasets
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.tools.download_data --all
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.tools.download_data --model mlx-community/SmolLM2-135M-Instruct
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.tools.download_data --models --datasets

# Run benchmarks
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run --list     # List available benchmarks
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run system     # Display system info
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.run ops        # Run operation benchmarks
```

### Examples

**Working Examples** (fully functional):

```bash
# SmolLM2-135M examples (basic generation, streaming, chat)
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/models/smollm2_135m/smollm2_135m_example.py

# SmolVLM-256M examples (vision-language model)
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/vlm/smolvlm_256m/basic_vqa.py
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/vlm/smolvlm_256m/batch_processing.py
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/vlm/smolvlm_256m/streaming_example.py

# nanoVLM examples (minimal 222M VLM)
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/vlm/nanovlm/minimal_vlm.py

# Moondream2 examples (region detection and spatial reasoning)
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/vlm/moondream2/object_detection_example.py

# Whisper-tiny examples (audio transcription)
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/whisper_tiny/basic_transcription.py
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python examples/whisper_tiny/batch_transcription.py
```

**Example Capabilities Demonstrated:**

- Basic text generation with custom parameters
- Streaming output for real-time applications
- Chat-style interactions with conversation history
- Custom generation configurations
- **Visual question answering (VQA) with images**
- **Image captioning and description**
- **Batch image processing**
- **Streaming multimodal generation**
- **Object detection and spatial reasoning**
- Audio transcription (basic and batch processing)

**Example Directory Structure:**

- `examples/models/` - Model-specific examples organized by model type:
  - `smollm2_135m/` - SmolLM2-135M demonstrations (✓ working)
  - `smollm2_360m/` - SmolLM2-360M demonstrations
  - `smolvlm_256m/` - SmolVLM-256M demonstrations
  - `smolvlm_500m/` - SmolVLM-500M demonstrations
  - `moondream2/` - Moondream2 demonstrations
  - `nanovlm/` - nanoVLM demonstrations
  - `tinyllava/` - TinyLLaVA demonstrations
  - `whisper_tiny/` - Whisper audio transcription examples (✓ working)
  - `trocr_small/` - TrOCR OCR examples (✓ working)
  - `yamnet/` - YAMNet audio classification examples
  - `silero_vad/` - Silero VAD examples
  - `chatterbox/` - Chatterbox speech synthesis examples
  - `donut_base/` - Donut document understanding examples
  - `minilm/` - MiniLM embedding examples
  - `all_minilm_l6_v2/` - all-MiniLM-L6-v2 examples
  - `orpheus_150m/` - Orpheus audio processing examples
- `examples/vlm/` - Cross-model vision-language examples (✓ working)
  - `smolvlm_256m/` - Advanced VQA, batch processing, streaming
  - `nanovlm/` - Minimal VLM demonstrations
  - `moondream2/` - Object detection and spatial reasoning
- `examples/agents/` - Agent system examples
- `examples/quant/` - Quantization examples
- `examples/eval/` - Evaluation examples
- `examples/server/` - Server API examples
- `examples/performance/` - Performance benchmark examples

### Python Environment

This project uses Python >=3.9, <3.13 with the following configuration files:

- `pyproject.toml` - Project metadata, dependencies, and tool configurations (black, ruff, mypy)
- `setup.py` - Package setup configuration (enables editable installs)
- `environment.yml` - Conda environment specification
- `pytest.ini` - Pytest configuration with custom markers
- `MANIFEST.in` - Package manifest (excludes resources/ from distribution)

## Testing Guidelines

### Custom Pytest Markers

Use these markers to categorize tests (defined in [pytest.ini:39-47](pytest.ini#L39-L47)):

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests (may require external services)
- `@pytest.mark.slow` - Slow-running tests (skip with `-m "not slow"`)
- `@pytest.mark.benchmark` - Performance benchmarking tests
- `@pytest.mark.eval` - Evaluation tests (may require datasets)
- `@pytest.mark.gpu` - Tests requiring GPU/MLX acceleration
- `@pytest.mark.requires_model` - Tests that download models
- `@pytest.mark.asyncio` - Async tests (auto-configured via pytest-asyncio in pytest.ini)

Example:

```python
@pytest.mark.unit
def test_quantization_utils():
    # Fast unit test
    pass

@pytest.mark.slow
@pytest.mark.gpu
def test_model_inference():
    # Slow test requiring GPU
    pass
```

## Development Workflow

Since this is an early-stage project with many stub files:

1. **Check existing stubs**: Many module files exist but are empty placeholders
2. **Reference resources**: Look in `resources/` for implementation patterns
3. **Implement in smlx/**: Write actual implementations in the `smlx/` package
4. **Keep it small**: Remember the core requirement - models must be "smol"
5. **Test**: Add tests in `tests/` directory with appropriate markers
6. **Format and lint**: Run `black` and `ruff` before committing

### Version Control

This project uses Git with Git LFS for large file tracking. The repository is configured to track data files (models, datasets, audio files) using Git LFS.

**Important Git LFS Notes:**
- Large files in `data/` directory are tracked with Git LFS
- Run `git lfs install` if you haven't already
- Use `git lfs pull` to download large files after cloning

```bash
# Check git status
git status

# View LFS-tracked files
git lfs ls-files

# Pull LFS files
git lfs pull
```

## Implementation Patterns

### Implementing New Models

When adding a new "smol" model, follow the SmolLM2_135M structure as a reference:

1. **Create model directory:** `smlx/models/YourModel/`

2. **Implement core modules:**
   - `model.py` - Model architecture (inherit from `mlx.nn.Module`)
     - Define layers (Attention, MLP, TransformerBlock, etc.)
     - Implement forward pass with proper MLX operations
   - `loader.py` - Loading from HuggingFace Hub
     - `load()` - Load model and tokenizer
     - `load_model_from_path()` - Load from local path
     - `save_model()` - Save model weights
   - `generate.py` - Generation logic
     - `generate()` - Basic text generation
     - `stream_generate()` - Streaming generation
     - `chat()` - Chat interface (if applicable)
     - `sample()` - Token sampling strategies
   - `config.py` - Model configuration
     - Default config dictionary
     - Config validation and loading
   - `cache.py` - KV cache (if applicable for transformers)
     - Standard KV cache
     - Rotating KV cache for long sequences
   - `__init__.py` - Export public API

3. **Add example in `examples/your_model/`**
   - Demonstrate basic usage
   - Show streaming and chat (if applicable)
   - Include generation configuration examples

4. **Add integration test in `tests/integration/`**
   - Test model loading
   - Test generation
   - Mark with `@pytest.mark.integration` and `@pytest.mark.requires_model`

5. **Update this documentation**
   - Add model to Available Models list
   - Document any special features or requirements

**Key Requirements:**

- **Must be "smol"** (< 1B parameters preferred, < 500M ideal)
- **Must work with MLX** on Apple Silicon
- **Must support quantization** (4-bit/8-bit via `smlx.quant/`)
- **Must follow existing API patterns** (load, generate, stream_generate)
- **Use MLX operations** throughout (no PyTorch/TensorFlow)

**Reference Implementation:** See `smlx/models/SmolLM2_135M/` for a complete, working example.

### Working with Stub Files

Most files in `smlx/` are currently empty placeholders. When implementing:

1. **Don't remove stub files** - If something is called but missing, it should be implemented, not removed
2. **Study reference implementations** - Browse `resources/mlx-examples/`, `resources/mlx-lm/`, and `resources/mlx-vlm/` for patterns
3. **Copy and adapt** - Take relevant code from resources and adapt it for "smol" models
4. **Maintain module boundaries** - Keep quantization in `smlx/quant/`, models in `smlx/models/`, etc.

### MLX-Specific Patterns

When implementing MLX models:

- **Use MLX array operations** - Leverage `mlx.core` for tensor operations
- **Lazy evaluation** - MLX uses lazy evaluation; call `mx.eval()` when needed
- **Unified memory** - Take advantage of Apple's unified memory architecture
- **Quantization support** - Most models should support 4-bit and 8-bit quantization via `smlx.quant/`

### Quantization Architecture

The `smlx/quant/` module provides multiple quantization techniques:

- **AWQ** (`awq.py`) - Activation-aware Weight Quantization for preserving accuracy
- **GPTQ** (`gptq.py`) - Post-training quantization for language models
- **DWQ** (`dwq.py`) - Dynamic Weight Quantization
- **LoRA** (`lora.py`) - Low-Rank Adaptation for parameter-efficient fine-tuning
- **DoRA** (`dora.py`) - Weight-Decomposed Low-Rank Adaptation

Reference implementations can be found in `resources/mlx-examples/lora/` and `resources/mlx-lm/`.

### Evaluation Framework

The `smlx/evals/` module is designed for multimodal benchmarking:

- **Math-Vista** - Math reasoning with vision
- **MMMU** - Massive Multi-discipline Multimodal Understanding
- **MMStar** - Multimodal reasoning benchmark
- **OCRBench** - OCR capability evaluation

When implementing evals, follow patterns from `resources/mlx-vlm/` for vision-language tasks.

## Additional Documentation

### Module-Specific Documentation (in `docs/` directory)

**Available:**

- `Quant.md` - Quantization techniques and usage
- `Eval.md` / `EVALUATION.md` - Evaluation framework and benchmarks
- `PerformanceOptimization.md` - Performance tuning and optimization strategies

**Planned (empty stubs):**

- `ModelImplementations.md` - Model implementation patterns and guidelines
- `Agents.md` - Agent system design
- `Server.md` - Server architecture and API design
- `Tools.md` - Tool utilities and CLI commands

### Root-Level Documentation

Quick reference guides in the project root:

- **[QUICKSTART.md](QUICKSTART.md)** - Fast-track guide to getting started with SMLX
- **[BENCHMARKS.md](BENCHMARKS.md)** - Comprehensive benchmarking guide and performance metrics
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines and workflow

**Model Conversion Documentation** (PyTorch/TensorFlow → MLX):

- **[CONVERSION_INDEX.md](CONVERSION_INDEX.md)** - Overview of conversion guides and quick navigation
- **[CONVERSION_PATTERNS.md](CONVERSION_PATTERNS.md)** - Common PyTorch→MLX conversion patterns
- **[CONVERSION_REFERENCE.md](CONVERSION_REFERENCE.md)** - Detailed conversion reference
- **[CONVERSION_UPDATE_SUMMARY.md](CONVERSION_UPDATE_SUMMARY.md)** - Recent conversion updates

**When converting models from PyTorch/TensorFlow**: Start with `CONVERSION_INDEX.md` for a step-by-step guide to adapting models to MLX.

Consult available documents for module-specific and task-specific implementation details.

## Key Principles

1. **Small models only** - This is the primary requirement for any model added
2. **MLX-first** - All implementations should leverage MLX framework for Apple silicon
3. **M4 optimized** - Target M4 chipset with 36GB unified memory
4. **No direct resource imports** - Reference but don't import from `resources/`
5. **Quantization by default** - Models should support quantization to reduce memory footprint
