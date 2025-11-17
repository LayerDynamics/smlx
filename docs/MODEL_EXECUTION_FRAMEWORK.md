# Model Execution Framework

Comprehensive infrastructure for model routing, inference, and lifecycle management in SMLX.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
  - [ModelRouter](#modelrouter)
  - [ModelRunner](#modelrunner)
  - [ModelLifecycleManager](#modellifecyclemanager-planned)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Integration](#integration)
- [Testing](#testing)

## Overview

The Model Execution Framework provides a unified, capability-based system for working with diverse models in SMLX. It eliminates hardcoded imports and provides consistent interfaces for:

- **Model Routing** - Dynamic dispatch based on model type and capabilities
- **Inference Execution** - Unified interface with preprocessing and validation
- **Lifecycle Management** - Model loading, caching, and telemetry (planned)

### Key Benefits

✅ **Eliminates hardcoded imports** - No more try/except import chains in server routes
✅ **Capability-based routing** - Models self-describe their capabilities
✅ **Type safety** - Comprehensive type hints and validation
✅ **Consistent API** - Unified interface across all model types
✅ **Preprocessing built-in** - Automatic input validation and preparation
✅ **Extensible** - Easy to add new models and capabilities

## Architecture

The framework follows a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                      │
│        (Server Routes, CLI Tools, Examples)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   ModelRunner (Execution)                    │
│   • Input preprocessing & validation                         │
│   • Config management                                        │
│   • Batch/streaming abstraction                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   ModelRouter (Dispatch)                     │
│   • Capability detection                                     │
│   • Dynamic module loading                                   │
│   • Model-specific routing                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              ModelLifecycleManager (Planned)                 │
│   • Model loading & caching                                  │
│   • Telemetry & monitoring                                   │
│   • Resource management                                      │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**
   - Router: What model to use and how to call it
   - Runner: How to prepare inputs and execute
   - Manager: When to load/unload and how to monitor

2. **Capability-Driven Design**
   - Models declare their capabilities
   - Framework validates requests against capabilities
   - No assumptions about model internals

3. **Progressive Enhancement**
   - Simple cases are simple (single-line inference)
   - Complex cases are possible (custom config, multimodal)
   - Framework grows with your needs

## Components

### ModelRouter

**File**: [smlx_router.py](smlx_router.py:1-730)
**Purpose**: Dynamic model dispatch based on capabilities

#### Key Classes

**`ModelCapabilities`** - Describes what a model can do

```python
@dataclass
class ModelCapabilities:
    # Core capabilities
    can_chat: bool = False          # Supports chat interface
    can_complete: bool = True       # Supports text completion
    can_stream: bool = True         # Supports streaming output
    can_transcribe: bool = False    # Supports audio transcription
    can_caption: bool = False       # Supports image captioning
    can_detect: bool = False        # Supports object detection

    # Input requirements
    requires_image: bool = False    # Requires image input
    requires_audio: bool = False    # Requires audio input

    # Model properties
    max_context_length: int = 2048
    supports_batch: bool = False
    modality: str = "text"          # text, vision, audio, vision-language
    category: str = "language"      # language, vision-language, audio, etc.

    # Extensible
    extra_capabilities: dict[str, Any] = field(default_factory=dict)
```

**`ModelRouter`** - Routes requests to appropriate model implementations

```python
class ModelRouter:
    def route_text_generation(
        self, model_type: str, model, tokenizer, prompt: str,
        max_tokens: int = 100, **kwargs
    ) -> str

    def route_chat(
        self, model_type: str, model, tokenizer,
        messages: list[dict[str, str]], **kwargs
    ) -> str

    def route_streaming_generation(
        self, model_type: str, model, tokenizer, prompt: str, **kwargs
    ) -> Generator[str, None, None]

    def route_multimodal(
        self, model_type: str, model, processor,
        prompt: str, image, **kwargs
    ) -> str

    def route_transcription(
        self, model_type: str, model, tokenizer, audio, **kwargs
    ) -> dict[str, Any]
```

#### Usage Example

```python
from smlx.models.smlx_router import get_router

# Get singleton router instance
router = get_router()

# Check capabilities
caps = router.get_capabilities("smollm2-135m")
if caps.can_chat:
    response = router.route_chat(
        model_type="smollm2-135m",
        model=model,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "Hello!"}]
    )

# Validate before calling
if router.can_handle("whisper-tiny", "can_transcribe"):
    result = router.route_transcription(
        model_type="whisper-tiny",
        model=model,
        tokenizer=tokenizer,
        audio="speech.wav"
    )
```

#### Capability Map

The router maintains a comprehensive capability map for all models:

**Language Models**:

- `smollm2-135m`, `smollm2-360m`, `smollm2-1.7b` - Chat, completion, streaming

**Vision-Language Models**:

- `smolvlm-256m`, `smolvlm-500m` - Chat, caption, requires image
- `moondream2` - Chat, caption, detect (regions)
- `nanovlm`, `tinyllava` - Minimal VLMs

**Audio Models**:

- `whisper-tiny`, `whisper-base` - Transcription, translation
- `yamnet` - Audio classification
- `silero-vad` - Voice activity detection

**Document/OCR Models**:

- `trocr-small`, `trocr-base` - OCR
- `donut-base` - Document understanding

**Embedding Models**:

- `minilm`, `all-minilm-l6-v2` - Text embeddings

See [smlx_router.py](smlx_router.py:80-277) for the complete capability map.

### ModelRunner

**File**: [smlx_runner.py](smlx_runner.py:1-605)
**Purpose**: Unified inference execution with preprocessing

#### Key Classes

**`InferenceConfig`** - Configuration for inference runs

```python
@dataclass
class InferenceConfig:
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop_strings: list[str] | None = None
    stream: bool = False
    batch_size: int = 1
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    verbose: bool = False
    use_cache: bool = True
    cache_config: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        """Validate configuration parameters."""
        # Raises ValueError if invalid
```

**`ModelRunner`** - Executes inference with preprocessing

```python
class ModelRunner:
    def run(
        self, model, tokenizer, model_type: str, prompt: str | list[str],
        config: InferenceConfig | None = None,
        processor=None, image=None, audio=None
    ) -> str | list[str]

    def run_chat(
        self, model, tokenizer, model_type: str,
        messages: list[dict[str, str]],
        config: InferenceConfig | None = None
    ) -> str

    def run_batch(
        self, model, tokenizer, model_type: str,
        prompts: list[str],
        config: InferenceConfig | None = None
    ) -> list[str]
```

#### Usage Example

```python
from smlx.models.smlx_runner import ModelRunner, InferenceConfig

# Create runner
runner = ModelRunner()

# Simple text generation
result = runner.run(
    model=model,
    tokenizer=tokenizer,
    model_type="smollm2-135m",
    prompt="Write a haiku about coding"
)

# Custom configuration
config = InferenceConfig(
    max_tokens=200,
    temperature=0.9,
    top_p=0.95,
    stop_strings=["END"]
)

result = runner.run(
    model=model,
    tokenizer=tokenizer,
    model_type="smollm2-135m",
    prompt="Once upon a time",
    config=config
)

# Chat interface
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is MLX?"}
]

response = runner.run_chat(
    model=model,
    tokenizer=tokenizer,
    model_type="smollm2-135m",
    messages=messages,
    config=config
)

# Batch processing
prompts = ["Hello", "Goodbye", "How are you?"]
results = runner.run_batch(
    model=model,
    tokenizer=tokenizer,
    model_type="smollm2-135m",
    prompts=prompts
)

# Multimodal (vision-language)
result = runner.run(
    model=model,
    tokenizer=tokenizer,
    model_type="smolvlm-256m",
    prompt="What's in this image?",
    processor=processor,
    image="photo.jpg"
)
```

#### Preprocessing Functions

The runner includes built-in preprocessing utilities:

**`preprocess_text_input()`** - Validates and tokenizes text

```python
def preprocess_text_input(
    prompt: str | list[str],
    tokenizer,
    max_length: int | None = None
) -> tuple[list[str], list[mx.array]]
```

**`preprocess_chat_input()`** - Formats chat messages

```python
def preprocess_chat_input(
    messages: list[dict[str, str]],
    tokenizer,
    max_length: int | None = None
) -> tuple[str, mx.array]
```

**`preprocess_image_input()`** - Loads and processes images

```python
def preprocess_image_input(
    image: str | Path | Any,
    processor
) -> Any
```

**`preprocess_audio_input()`** - Loads and processes audio

```python
def preprocess_audio_input(
    audio: str | Path | Any,
    processor,
    sample_rate: int = 16000
) -> Any
```

### ModelLifecycleManager

**File**: [smlx_manager.py](smlx_manager.py:1-713)
**Purpose**: Model loading, caching, and telemetry

#### Key Classes

**`CacheConfig`** - Configuration for model cache

```python
@dataclass
class CacheConfig:
    max_models: int = 3                  # Max models to cache
    max_memory_gb: float = 24.0          # Max memory usage
    min_free_memory_gb: float = 4.0      # Min free memory to maintain
    enable_eviction: bool = True         # Auto eviction on memory pressure
    eviction_threshold: float = 0.8      # Threshold to trigger eviction

    def validate(self):
        """Validate configuration parameters."""
```

**`TelemetryConfig`** - Configuration for telemetry

```python
@dataclass
class TelemetryConfig:
    enable_telemetry: bool = True
    track_latency: bool = True
    track_memory: bool = True
    track_errors: bool = True
    retention_hours: int = 24
```

**`ModelStats`** - Statistics for a single model

```python
@dataclass
class ModelStats:
    model_id: str
    load_count: int = 0
    inference_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    memory_mb: float = 0.0
    last_used: float
    first_loaded: float

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency."""
```

**`ModelCache`** - LRU cache with memory awareness

```python
class ModelCache:
    def get(self, model_id: str) -> tuple[Any, Any] | None
    def put(self, model_id: str, model, tokenizer)
    def remove(self, model_id: str) -> bool
    def clear(self)
    def get_cached_models(self) -> list[str]
```

**`ModelTelemetry`** - Usage and performance tracking

```python
class ModelTelemetry:
    def record_load(self, model_id: str, memory_mb: float = 0.0)
    def record_inference(self, model_id: str, latency_ms: float = 0.0)
    def record_error(self, model_id: str)
    def get_stats(self, model_id: str | None = None) -> dict | ModelStats
    def reset_stats(self, model_id: str | None = None)
```

**`ModelLifecycleManager`** - Unified lifecycle management

```python
class ModelLifecycleManager:
    def load_model(
        self, model_id: str, quantization: str | None = None,
        force_reload: bool = False, **kwargs
    ) -> tuple[Any, Any]

    def unload_model(self, model_id: str, quantization: str | None = None) -> bool
    def get_cached_models(self) -> list[str]
    def get_stats(self, model_id: str | None = None) -> dict
    def clear_cache(self)
    def get_memory_info(self) -> dict[str, float]
```

#### Usage Example

```python
from smlx.models.smlx_manager import get_manager, CacheConfig, TelemetryConfig

# Get manager instance (singleton)
manager = get_manager()

# Load model (cached automatically)
model, tokenizer = manager.load_model(
    "mlx-community/SmolLM2-135M-Instruct",
    quantization="4bit"
)

# Model is now cached - second load is instant
model2, tokenizer2 = manager.load_model("mlx-community/SmolLM2-135M-Instruct")

# Get telemetry statistics
stats = manager.get_stats()
print(f"Total models: {stats['total_models']}")
print(f"Total loads: {stats['total_loads']}")

# Get model-specific stats
model_stats = manager.get_stats("mlx-community/SmolLM2-135M-Instruct:4bit")
print(f"Load count: {model_stats['load_count']}")

# Get memory info
mem_info = manager.get_memory_info()
print(f"Memory usage: {mem_info['used_gb']:.1f}GB")
print(f"Cached models: {mem_info['cached_models']}")

# Unload model
manager.unload_model("mlx-community/SmolLM2-135M-Instruct", quantization="4bit")

# Custom configuration
cache_config = CacheConfig(max_models=5, max_memory_gb=32.0)
telemetry_config = TelemetryConfig(enable_telemetry=True)

manager = get_manager(
    cache_config=cache_config,
    telemetry_config=telemetry_config,
    force_new=True
)
```

#### Features

- ✅ **LRU Caching** - Least recently used eviction policy
- ✅ **Memory Awareness** - Monitors system memory, evicts on pressure
- ✅ **Lazy Loading** - Models loaded only when needed
- ✅ **Telemetry** - Track loads, inference calls, errors, latency
- ✅ **Quantization Support** - Separate cache entries for different quantizations
- ✅ **Thread-Safe** - Safe for concurrent access
- ✅ **Resource Management** - Memory monitoring and cleanup

## Usage Examples

### Server Integration (FastAPI)

**Before** - Hardcoded imports with try/except chains:

```python
# Old approach (smlx/server/routes/chat.py)
try:
    from smlx.models.SmolLM2_135M import generate
except:
    try:
        from smlx.models.SmolLM2_360M import generate
    except:
        raise ValueError("Could not find generate function")

response = generate(model, tokenizer, prompt, ...)
```

**After** - Dynamic routing:

```python
# New approach (smlx/server/routes/chat.py)
from smlx.models.smlx_router import get_router
from smlx.models.registry import infer_model_type

router = get_router()
model_type = infer_model_type(model_id)

response = router.route_chat(
    model_type=model_type,
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k or 0
)
```

### CLI Tools

```python
#!/usr/bin/env python3
"""Simple inference CLI using ModelRunner."""

from smlx.models.smlx_runner import ModelRunner, InferenceConfig
from smlx.models.SmolLM2_135M import load

def main():
    # Load model
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Create runner and config
    runner = ModelRunner()
    config = InferenceConfig(max_tokens=100, temperature=0.7)

    # Interactive loop
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break

        response = runner.run(
            model=model,
            tokenizer=tokenizer,
            model_type="smollm2-135m",
            prompt=prompt,
            config=config
        )

        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
```

### Agent Systems

```python
from smlx.models.smlx_runner import ModelRunner, InferenceConfig
from smlx.agents.base import BaseAgent, Message

class LLMAgent(BaseAgent):
    def __init__(self, model, tokenizer, model_type):
        self.runner = ModelRunner()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.config = InferenceConfig(max_tokens=500, temperature=0.8)

    def execute(self, messages: list[Message]) -> str:
        # Convert to dict format
        message_dicts = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Run inference
        response = self.runner.run_chat(
            model=self.model,
            tokenizer=self.tokenizer,
            model_type=self.model_type,
            messages=message_dicts,
            config=self.config
        )

        return response
```

### Benchmarking

```python
from smlx.models.smlx_runner import ModelRunner, InferenceConfig
from smlx.bench.stats import measure_throughput
import time

def benchmark_model(model, tokenizer, model_type, prompts):
    """Benchmark model throughput."""
    runner = ModelRunner()
    config = InferenceConfig(max_tokens=100, temperature=0.7)

    start = time.time()
    results = runner.run_batch(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        prompts=prompts,
        config=config
    )
    elapsed = time.time() - start

    tokens_generated = sum(len(tokenizer.encode(r)) for r in results)
    throughput = tokens_generated / elapsed

    return {
        "num_prompts": len(prompts),
        "elapsed_seconds": elapsed,
        "tokens_per_second": throughput,
        "results": results
    }
```

## API Reference

### ModelRouter

#### Methods

**`get_capabilities(model_type: str) -> ModelCapabilities`**

Get capabilities for a model type.

```python
caps = router.get_capabilities("smollm2-135m")
print(f"Can chat: {caps.can_chat}")
print(f"Max context: {caps.max_context_length}")
```

**`can_handle(model_type: str, capability: str) -> bool`**

Check if model supports a specific capability.

```python
if router.can_handle("whisper-tiny", "can_transcribe"):
    # Safe to call transcription
    pass
```

**`route_text_generation(...) -> str`**

Route text generation request.

Parameters:

- `model_type` (str): Model type identifier
- `model`: Loaded model instance
- `tokenizer`: Model tokenizer
- `prompt` (str): Input text
- `max_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature
- `top_p` (float): Nucleus sampling threshold
- `top_k` (int): Top-k sampling threshold
- `stop_strings` (list[str] | None): Stop sequences
- `verbose` (bool): Print generation progress

Returns: Generated text (str)

**`route_chat(...) -> str`**

Route chat request.

Parameters:

- `model_type` (str): Model type identifier
- `model`: Loaded model instance
- `tokenizer`: Model tokenizer
- `messages` (list[dict]): Chat messages with 'role' and 'content'
- `max_tokens` (int): Maximum tokens to generate
- Other parameters same as `route_text_generation`

Returns: Assistant response (str)

**`route_streaming_generation(...) -> Generator[str, None, None]`**

Route streaming generation request.

Parameters: Same as `route_text_generation`

Returns: Generator yielding token strings

**`route_multimodal(...) -> str`**

Route multimodal generation request.

Parameters:

- `model_type` (str): Model type identifier
- `model`: Loaded model instance
- `processor`: Image/audio processor
- `prompt` (str): Text prompt
- `image`: Image input (path, PIL, or array)
- Other parameters same as `route_text_generation`

Returns: Generated description (str)

**`route_transcription(...) -> dict`**

Route audio transcription request.

Parameters:

- `model_type` (str): Model type identifier
- `model`: Loaded model instance
- `tokenizer`: Model tokenizer
- `audio`: Audio input (path or array)
- `language` (str | None): Source language code
- `verbose` (bool): Print progress

Returns: Dict with 'text' and 'language' keys

### ModelRunner

#### Methods

**`run(...) -> str | list[str]`**

Run inference with automatic preprocessing.

Parameters:

- `model`: Loaded model instance
- `tokenizer`: Model tokenizer
- `model_type` (str): Model type identifier
- `prompt` (str | list[str]): Text prompt(s)
- `config` (InferenceConfig | None): Inference configuration
- `processor` (Any | None): Image/audio processor
- `image` (Any | None): Image input for VLMs
- `audio` (Any | None): Audio input for audio models

Returns: Generated text or list of texts

**`run_chat(...) -> str`**

Run chat inference.

Parameters:

- `model`: Loaded model instance
- `tokenizer`: Model tokenizer
- `model_type` (str): Model type identifier
- `messages` (list[dict]): Chat messages
- `config` (InferenceConfig | None): Inference configuration

Returns: Assistant response (str)

**`run_batch(...) -> list[str]`**

Run batch inference.

Parameters:

- `model`: Loaded model instance
- `tokenizer`: Model tokenizer
- `model_type` (str): Model type identifier
- `prompts` (list[str]): List of prompts
- `config` (InferenceConfig | None): Inference configuration

Returns: List of generated texts

### InferenceConfig

#### Attributes

- `max_tokens` (int): Maximum tokens to generate (default: 100)
- `temperature` (float): Sampling temperature (default: 0.7)
- `top_p` (float): Nucleus sampling threshold (default: 1.0)
- `top_k` (int): Top-k sampling threshold (default: 0)
- `stop_strings` (list[str] | None): Stop sequences (default: None)
- `stream` (bool): Use streaming generation (default: False)
- `batch_size` (int): Batch size (default: 1)
- `repetition_penalty` (float): Repetition penalty (default: 1.0)
- `repetition_context_size` (int): Context size for repetition (default: 20)
- `verbose` (bool): Print progress (default: False)
- `use_cache` (bool): Use KV cache (default: True)
- `cache_config` (dict): Cache configuration (default: {})

#### Methods

**`validate()`**

Validate configuration parameters. Raises `ValueError` if any parameter is invalid.

## Integration

### Adding New Models

To integrate a new model with the framework:

1. **Add capability entry** in [smlx_router.py](smlx_router.py:80-277):

```python
CAPABILITY_MAP: dict[str, ModelCapabilities] = {
    # ... existing entries ...

    "your-model": ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_stream=True,
        max_context_length=4096,
        modality="text",
        category="language"
    ),
}
```

2. **Ensure model implements standard API**:

Your model's `generate.py` should export:

- `generate(model, tokenizer, prompt, ...)` - For text completion
- `chat(model, tokenizer, messages, ...)` - For chat (if can_chat=True)
- `stream_generate(model, tokenizer, prompt, ...)` - For streaming (if can_stream=True)

3. **Add to model registry** in [registry.py](registry.py):

```python
MODEL_REGISTRY = {
    # ... existing entries ...
    "your-model": "smlx.models.YourModel.loader",
}
```

4. **Test integration**:

```python
from smlx.models.smlx_router import get_router

router = get_router()
caps = router.get_capabilities("your-model")
assert caps.can_chat == True

# Test routing
result = router.route_text_generation(
    model_type="your-model",
    model=model,
    tokenizer=tokenizer,
    prompt="Test"
)
```

### Server Integration

The framework is already integrated into FastAPI server routes:

- [smlx/server/routes/chat.py](../../server/routes/chat.py:117-158) - Chat completions
- [smlx/server/routes/completions.py](../../server/routes/completions.py:96-139) - Text completions

Both routes use:

1. `infer_model_type()` to detect model type from HF ID
2. `get_router()` to get router instance
3. `router.route_*()` methods to execute inference

## Testing

### Unit Tests

**Router Tests**: [test_smlx_router.py](../../tests/models/test_smlx_router.py) - 31 tests

```bash
# Run router tests
python -m pytest tests/models/test_smlx_router.py -v

# Test specific functionality
python -m pytest tests/models/test_smlx_router.py::test_router_route_chat -v
```

**Runner Tests**: [test_smlx_runner.py](../../tests/models/test_smlx_runner.py) - 36 tests

```bash
# Run runner tests
python -m pytest tests/models/test_smlx_runner.py -v

# Test preprocessing
python -m pytest tests/models/test_smlx_runner.py -k preprocess -v
```

### Integration Tests

Integration tests verify end-to-end functionality:

```python
@pytest.mark.integration
@pytest.mark.requires_model
def test_runner_with_real_model():
    """Test ModelRunner with actual model."""
    from smlx.models.SmolLM2_135M import load
    from smlx.models.smlx_runner import ModelRunner, InferenceConfig

    # Load model
    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

    # Create runner
    runner = ModelRunner()
    config = InferenceConfig(max_tokens=50, temperature=0.7)

    # Test generation
    result = runner.run(
        model=model,
        tokenizer=tokenizer,
        model_type="smollm2-135m",
        prompt="Write a short poem",
        config=config
    )

    assert len(result) > 0
    assert isinstance(result, str)
```

### Test Coverage

Current test coverage:

- **ModelCapabilities**: 100% (defaults, custom, validation)
- **CAPABILITY_MAP**: 100% (all model types, aliases)
- **ModelRouter**: 95% (all routing methods, error handling)
- **InferenceConfig**: 100% (defaults, validation)
- **Preprocessing**: 90% (text, chat, image - audio partial)
- **ModelRunner**: 95% (run, run_chat, run_batch)

## Performance Considerations

### Router Performance

- **Singleton pattern**: Router is initialized once, cached globally
- **Module caching**: Imported modules cached in `_function_cache`
- **No overhead for capability checks**: Simple dict lookups

### Runner Performance

- **Preprocessing overhead**: ~1-5ms per request
- **Validation overhead**: <1ms per config
- **Batch efficiency**: Linear scaling with batch size

### Optimization Tips

1. **Reuse runner instances**: Create once, call multiple times
2. **Pre-validate configs**: Validate once, reuse many times
3. **Batch when possible**: Use `run_batch()` for multiple prompts
4. **Cache preprocessed inputs**: For repeated inference on same input

## Future Enhancements

### Planned Features

- [ ] **ModelLifecycleManager** - Smart caching and lifecycle management
- [ ] **Quantization integration** - Automatic 4-bit/8-bit loading
- [ ] **Telemetry** - Track usage, performance, errors
- [ ] **Streaming for batch** - Stream results for batch processing
- [ ] **Tool calling** - Support for tool/function calling models
- [ ] **Multi-turn chat optimization** - Efficient multi-turn conversations

### Under Consideration

- [ ] **Async support** - Native async inference
- [ ] **Model ensembles** - Combine multiple models
- [ ] **Fallback routing** - Automatic fallback if model unavailable
- [ ] **Hot-swapping** - Replace models without downtime

## Contributing

When contributing to the Model Execution Framework:

1. **Add tests** for all new functionality
2. **Update capability map** when adding models
3. **Document API changes** in this README
4. **Follow type hints** - Use modern Python typing (X | None)
5. **Validate inputs** - Check parameters before processing

## License

Copyright © 2025 SMLX Project

---

**Questions or Issues?** Open an issue at [github.com/ryanoboyle/smlx](https://github.com/ryanoboyle/smlx)
