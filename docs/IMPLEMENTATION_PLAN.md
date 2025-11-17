# SMLX Implementation Plan

**Generated**: 2025-11-12
**Status**: Comprehensive Planning Document

---

## Executive Summary

Based on comprehensive analysis of the SMLX codebase, resources directory, and 2025 MLX ecosystem patterns, this document provides a detailed implementation plan for:

1. **Models Module** - 16 stub models need implementation following SmolLM2_135M reference
2. **Utils Module** - Currently empty, needs 7 core utility modules
3. **Tools Module** - Well-implemented, needs 3 enhancements
4. **Data Requirements** - All models/datasets available via implemented download tools

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Models Module Planning](#models-module-planning)
3. [Utils Module Planning](#utils-module-planning)
4. [Tools Module Enhancements](#tools-module-enhancements)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Resource Mapping](#resource-mapping)
7. [Testing Strategy](#testing-strategy)
8. [Data Availability](#data-availability)

---

## Current State Assessment

### What's Working ✅

#### smlx/tools/ (Fully Functional)

- **download.py** (369 lines) - Complete HuggingFace Hub integration
- **download_data.py** (286 lines) - Full CLI for models/datasets
- **convert2mlx.py** (722 lines) - Model conversion with quantization prep
- **Cache management** - Organized at `~/.cache/smlx/`
- **Testing** - 26 tests covering core functionality

#### smlx/models/SmolLM2_135M/ (Reference Implementation)

- **Complete implementation** (1,834 lines across 6 files)
- **All patterns established**:
  - Model architecture with NoPE (No Positional Encoding)
  - Config management with validation
  - Loader with HuggingFace integration
  - Generation with streaming and chat
  - KV cache (standard + rotating)
- **Working examples** in `examples/smollm2_135m/`
- **Integration tests** passing

#### smlx/quant/ (Quantization System)

- AWQ, GPTQ, DWQ implementations
- LoRA and DoRA for fine-tuning
- Calibration data download support

#### Documentation

- Comprehensive READMEs for all models
- Clear CLAUDE.md with patterns
- pytest configuration with markers

### What's Missing ❌

#### smlx/utils/ (Empty - Priority 1)

- No shared utilities
- No common patterns extracted
- No helper functions

#### smlx/models/ (16 Stubs - Priority 2)

- Only 1 of 17 models implemented
- Missing models across 6 categories:
  - 1 Language Model (SmolLM2-360M)
  - 5 Vision-Language Models
  - 4 Audio Models
  - 2 OCR Models
  - 2 Embedding Models
  - 1 Chat Model

---

## Models Module Planning

### Model Categories & Implementation Strategy

#### Category 1: Language Models (Simple Transformers)

**SmolLM2-360M** - Priority: HIGH

- **Type**: Autoregressive language model
- **Size**: 360M parameters
- **Architecture**: Same as SmolLM2-135M (scaled up)
- **Implementation Effort**: 🟢 LOW (copy SmolLM2-135M patterns)
- **Key Changes**:
  - Larger hidden_size: 960 (vs 576)
  - More layers: 32 (vs 30)
  - More attention heads: 15 (vs 9)
  - Same NoPE pattern (every 4th layer)
- **Files Needed**:
  - `model.py` - Copy SmolLM2-135M, adjust config
  - `config.py` - Update DEFAULT_CONFIG
  - `loader.py` - Identical pattern
  - `generate.py` - Identical pattern
  - `cache.py` - Reuse existing
  - `__init__.py` - Standard exports
- **Resource Reference**: `/resources/mlx-lm/mlx_lm/models/`
- **HuggingFace Hub**: `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Estimated Time**: 4-6 hours

---

#### Category 2: Vision-Language Models (Multimodal)

**SmolVLM-256M-Instruct** - Priority: HIGH

- **Type**: Vision-Language Model (VLM)
- **Size**: 256M parameters (smallest VLM in world)
- **Architecture**: Vision encoder + language model + connector
- **Implementation Effort**: 🟡 MEDIUM (new vision components)
- **Components**:
  1. **Vision Encoder** (SigLIP-based)
     - Patch embedding: 14x14 patches
     - Vision transformer layers
     - Output: visual embeddings
  2. **Connector/Projector**
     - MLP to project vision → text space
     - Typically 2-3 linear layers
  3. **Language Model** (SmolLM-based)
     - Similar to SmolLM2-135M
     - Accepts text + vision tokens
  4. **Image Processor**
     - Resize, normalize, patch
     - Output: pixel_values tensor
- **Files Needed**:
  - `model.py` - VisionEncoder + Connector + LanguageModel
  - `vision.py` - Vision-specific components
  - `processor.py` - Image preprocessing
  - `config.py` - Multimodal config
  - `loader.py` - Load vision + language weights
  - `generate.py` - Image + text generation
  - `cache.py` - Reuse existing
  - `__init__.py` - Public API
- **Resource Reference**:
  - `/resources/mlx-vlm/mlx_vlm/models/` (best VLM patterns)
  - `/resources/mlx-vlm/mlx_vlm/models/smolvlm/` (if exists)
  - `/resources/mlx-vlm/mlx_vlm/models/llava/` (similar architecture)
- **Key Patterns**:

  ```python
  # Vision encoder
  vision_features = vision_encoder(pixel_values)  # [batch, patches, dim]

  # Project to language space
  vision_tokens = connector(vision_features)  # [batch, patches, hidden_size]

  # Combine with text embeddings
  input_embeds = combine_vision_text(vision_tokens, text_embeds)

  # Language model forward
  output = language_model(input_embeds)
  ```

- **HuggingFace Hub**: `HuggingFaceTB/SmolVLM-256M-Instruct`
- **Estimated Time**: 16-24 hours

**SmolVLM-500M-Instruct** - Priority: MEDIUM

- Same architecture as 256M, scaled up
- Larger vision encoder and language model
- Copy patterns from SmolVLM-256M
- **Estimated Time**: 8-12 hours

**Moondream2** - Priority: MEDIUM

- **Type**: Production-ready VLM
- **Size**: 1.8B parameters (at upper limit for "smol")
- **Architecture**: SigLIP + Phi-2 style
- **Special Features**: Optimized for mobile deployment
- **Resource Reference**: `/resources/mlx-vlm/mlx_vlm/models/moondream/`
- **Estimated Time**: 12-16 hours

**TinyLLaVA** - Priority: MEDIUM

- **Type**: LLaVA-style VLM
- **Size**: ~1.5B parameters
- **Architecture**: CLIP + LLaMA
- **Resource Reference**: `/resources/mlx-vlm/mlx_vlm/models/llava/`
- **Estimated Time**: 12-16 hours

**nanoVLM** - Priority: LOW

- **Type**: Lightweight VLM
- **Size**: 222M parameters
- **Architecture**: Similar to SmolVLM
- **Estimated Time**: 8-12 hours

---

#### Category 3: Audio Models (Speech/Audio)

**Whisper-tiny** - Priority: HIGH

- **Type**: Automatic Speech Recognition (ASR)
- **Size**: 39M parameters (multilingual)
- **Architecture**: Encoder-decoder transformer
- **Components**:
  1. **Audio Encoder**
     - Mel-spectrogram input (80 channels)
     - Sinusoidal positional embeddings
     - Conv1D layers for audio feature extraction
     - Transformer encoder layers
  2. **Text Decoder**
     - Transformer decoder with cross-attention
     - Attends to encoder output
     - Autoregressive text generation
  3. **Audio Processor**
     - Convert audio → mel-spectrogram
     - Normalization, padding
- **Files Needed**:
  - `model.py` - AudioEncoder + TextDecoder
  - `audio.py` - Audio-specific components
  - `processor.py` - Audio preprocessing (mel-spectrogram)
  - `config.py` - Whisper config
  - `loader.py` - Load encoder + decoder weights
  - `decode.py` - Beam search, language detection
  - `cache.py` - KV cache for decoder
  - `__init__.py` - Public API
- **Resource Reference**:
  - `/resources/lightning-whisper-mlx/` (optimized implementation)
  - `/resources/mlx-examples/whisper/` (canonical example)
- **Key Patterns**:

  ```python
  # Preprocess audio
  mel = audio_processor(audio_waveform)  # [batch, 80, time]

  # Encode audio
  encoder_output = audio_encoder(mel)  # [batch, time/2, dim]

  # Decode with cross-attention
  text_tokens = decoder(
      input_ids=decoder_input,
      encoder_hidden_states=encoder_output,
      cache=kv_cache
  )
  ```

- **Special Features**:
  - Multilingual (99 languages)
  - Timestamping support
  - Language detection
  - Beam search decoding
- **HuggingFace Hub**: `openai/whisper-tiny`
- **Estimated Time**: 20-28 hours

**Orpheus-150M** - Priority: LOW

- **Type**: Audio processing
- **Size**: 150M parameters
- **Estimated Time**: 16-20 hours

**YAMNet** - Priority: LOW

- **Type**: Audio classification
- **Size**: Small (mobilenet-based)
- **Estimated Time**: 8-12 hours

**Silero VAD** - Priority: LOW

- **Type**: Voice Activity Detection
- **Size**: Very small
- **Estimated Time**: 4-8 hours

---

#### Category 4: OCR Models (Document Understanding)

**TrOCR-small** - Priority: MEDIUM

- **Type**: Optical Character Recognition
- **Architecture**: Vision encoder + text decoder
- **Similar to**: Vision-language models
- **Resource Reference**: `/resources/mlx-vlm/` (vision encoder patterns)
- **Estimated Time**: 12-16 hours

**Donut-base** - Priority: MEDIUM

- **Type**: Document understanding + OCR
- **Architecture**: Swin Transformer + BART decoder
- **Estimated Time**: 16-20 hours

---

#### Category 5: Embedding Models (Sentence/Text)

**MiniLM** - Priority: LOW

- **Type**: Sentence embeddings
- **Architecture**: BERT-based encoder
- **Output**: Dense vectors for similarity
- **Implementation Effort**: 🟢 LOW (encoder-only)
- **Files Needed**:
  - `model.py` - Encoder transformer
  - `config.py` - BERT-style config
  - `loader.py` - Load encoder weights
  - `encode.py` - Encoding interface (no generation)
  - `__init__.py` - Public API
- **Resource Reference**: `/resources/mlx-examples/` (encoder patterns)
- **Estimated Time**: 6-10 hours

**all-MiniLM-L6-v2** - Priority: LOW

- Same architecture as MiniLM
- Specific pretrained weights
- **Estimated Time**: 4-6 hours

---

#### Category 6: Chat Models

**Chatterbox** - Priority: LOW

- **Type**: Chat-optimized model
- **Details**: Unclear from docs, may be application layer
- **Estimated Time**: TBD (need more research)

---

### Common Model Patterns (Can Be Shared)

All models share these patterns that could be extracted to `smlx/utils/`:

1. **BaseModelArgs** - Already in SmolLM2_135M
2. **Config loading/validation** - Standard across all models
3. **KV Cache** - Standard and rotating variants
4. **Sampling logic** - Temperature, top-p, top-k
5. **Generation loop** - Token-by-token with stopping criteria
6. **Weight loading** - Safetensors + NPZ support
7. **HuggingFace Hub integration** - Model/tokenizer downloading

---

## Utils Module Planning

The `smlx/utils/` module is currently **completely empty**. Based on patterns from SmolLM2_135M and common needs across all models, here's what should be implemented:

### Proposed Utils Module Structure

```
smlx/utils/
├── __init__.py                 # Public exports
├── config.py                   # Configuration utilities (PRIORITY 1)
├── loading.py                  # Model loading patterns (PRIORITY 1)
├── generation.py               # Generation utilities (PRIORITY 2)
├── cache.py                    # KV cache implementations (PRIORITY 2)
├── sampling.py                 # Sampling strategies (PRIORITY 3)
├── preprocessing.py            # Input preprocessing (PRIORITY 3)
├── tokenization.py             # Tokenizer utilities (PRIORITY 4)
├── attention.py                # Attention mechanisms (PRIORITY 4)
├── validation.py               # Model validation (PRIORITY 5)
└── metrics.py                  # Performance metrics (PRIORITY 5)
```

---

### 1. smlx/utils/config.py (PRIORITY 1)

**Purpose**: Shared configuration management for all models

**Functions to Implement**:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json
from pathlib import Path

@dataclass
class BaseModelArgs:
    """Base configuration class with common utilities."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """Create from HuggingFace config dict."""
        # Filter to only valid fields
        # Handle nested configs
        # Return instance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""

    def save(self, path: Path):
        """Save config to JSON file."""

    @classmethod
    def load(cls, path: Path):
        """Load config from JSON file."""

def load_config(
    config_path: Path,
    config_class: type = None
) -> BaseModelArgs:
    """Load config from path with automatic class detection."""

def validate_config(config: BaseModelArgs) -> None:
    """Validate configuration values."""
    # Check ranges
    # Check types
    # Check compatibility

def merge_configs(
    base_config: Dict,
    override_config: Dict
) -> Dict:
    """Merge two configs with override priority."""

def print_config(config: BaseModelArgs, estimate_params: bool = True):
    """Pretty-print config with parameter estimation."""

def estimate_parameters(config: BaseModelArgs) -> int:
    """Estimate model parameter count from config."""
    # Based on architecture type
    # Calculate embeddings, attention, MLP, output
```

**Patterns to Extract**:

- From `smlx/models/SmolLM2_135M/config.py`
- Generalize for all model types

---

### 2. smlx/utils/loading.py (PRIORITY 1)

**Purpose**: Standardized model loading across all model types

**Functions to Implement**:

```python
from pathlib import Path
from typing import Optional, Tuple, Union
import mlx.core as mx
from huggingface_hub import snapshot_download
from safetensors import safe_open

def resolve_model_path(
    model_name_or_path: str,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None
) -> Path:
    """Resolve model path from HuggingFace or local."""
    # Check if local path exists
    # If not, download from Hub
    # Return absolute path

def load_weights(
    model_path: Path,
    lazy: bool = False
) -> Dict[str, mx.array]:
    """Load model weights from safetensors or npz."""
    # Support .safetensors (preferred)
    # Support .npz (MLX format)
    # Support sharded weights (index.json)
    # Handle lazy loading

def save_weights(
    weights: Dict[str, mx.array],
    save_path: Path,
    max_shard_size_gb: float = 5.0
):
    """Save weights with automatic sharding."""
    # Shard if necessary
    # Create index file
    # Save safetensors

def sanitize_weights(
    weights: Dict[str, mx.array],
    remove_patterns: Optional[list] = None
) -> Dict[str, mx.array]:
    """Remove unnecessary keys from weights."""
    # Default patterns: rotary_emb.inv_freq, lm_head.*
    # Custom patterns

def load_tokenizer(
    model_path: Path,
    trust_remote_code: bool = False
):
    """Load transformers tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code
    )

def verify_weights(
    weights: Dict[str, mx.array],
    expected_keys: Optional[list] = None
) -> bool:
    """Verify weights are complete."""
    # Check for required keys
    # Check shapes
    # Check dtypes
```

**Patterns to Extract**:

- From `smlx/models/SmolLM2_135M/loader.py`
- From `smlx/tools/convert2mlx.py`
- From resources: `/resources/mlx-lm/mlx_lm/utils.py`

---

### 3. smlx/utils/generation.py (PRIORITY 2)

**Purpose**: Shared text generation utilities

**Functions to Implement**:

```python
from dataclasses import dataclass
from typing import Optional, List, Generator
import mlx.core as mx

@dataclass
class GenerationConfig:
    """Standard generation configuration."""
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    min_tokens: int = 0
    stop_strings: Optional[List[str]] = None

def create_stopping_criteria(
    stop_strings: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    eos_token_id: Optional[int] = None
):
    """Create callable stopping criteria function."""

def apply_repetition_penalty(
    logits: mx.array,
    generated_tokens: mx.array,
    penalty: float = 1.0
) -> mx.array:
    """Apply repetition penalty to logits."""
    # Reduce probability of already-generated tokens

def generate_step(
    model,
    prompt: str,
    tokenizer,
    config: GenerationConfig,
    cache = None
) -> Generator[str, None, None]:
    """Core generation loop as generator."""
    # Tokenize prompt
    # Run model forward
    # Sample token
    # Yield decoded token
    # Check stopping criteria

def generate(
    model,
    prompt: str,
    tokenizer,
    config: GenerationConfig,
    cache = None
) -> str:
    """Complete generation (blocking)."""

def stream_generate(
    model,
    prompt: str,
    tokenizer,
    config: GenerationConfig,
    cache = None
) -> Generator[str, None, None]:
    """Streaming generation (yields tokens)."""
```

**Patterns to Extract**:

- From `smlx/models/SmolLM2_135M/generate.py`
- From resources: `/resources/mlx-lm/mlx_lm/utils.py`

---

### 4. smlx/utils/cache.py (PRIORITY 2)

**Purpose**: KV cache implementations for all autoregressive models

**Classes to Implement**:

```python
from typing import Optional, Tuple
import mlx.core as mx

class KVCache:
    """Simple KV cache with dynamic growth."""

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
        self.step: int = 256

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Update cache and return all keys/values."""

    @property
    def state(self):
        """Get cache state for serialization."""

    @state.setter
    def state(self, value):
        """Set cache state from deserialization."""

class RotatingKVCache(KVCache):
    """KV cache with maximum size (for long sequences)."""

    def __init__(self, max_size: int, keep: int = 0):
        super().__init__()
        self.max_size = max_size
        self.keep = keep  # Preserve first N tokens

    def _trim(self, trim_size: int, v: mx.array, append: Optional[mx.array] = None):
        """Trim cache to size."""

    def _temporal_order(self, v: mx.array):
        """Restore temporal order after rotation."""

def make_cache(
    model_type: str = "standard",
    max_size: Optional[int] = None,
    keep: int = 0
):
    """Factory function for cache creation."""
    if max_size is not None:
        return RotatingKVCache(max_size, keep)
    return KVCache()
```

**Patterns to Extract**:

- From `smlx/models/SmolLM2_135M/cache.py`
- From resources: `/resources/mlx-lm/mlx_lm/models/cache.py` (1000 lines)

---

### 5. smlx/utils/sampling.py (PRIORITY 3)

**Purpose**: Token sampling strategies

**Functions to Implement**:

```python
import mlx.core as mx
from typing import Optional

def sample_with_temperature(
    logits: mx.array,
    temperature: float = 1.0
) -> mx.array:
    """Sample with temperature scaling."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    scaled_logits = logits / temperature
    return mx.random.categorical(scaled_logits)

def top_p_sampling(
    logits: mx.array,
    top_p: float = 0.9
) -> mx.array:
    """Nucleus sampling (top-p)."""
    # Sort probabilities
    # Calculate cumulative sum
    # Mask tokens beyond top_p threshold
    # Sample from remaining

def top_k_sampling(
    logits: mx.array,
    top_k: int = 50
) -> mx.array:
    """Top-k sampling."""
    # Get top k logits
    # Mask others
    # Sample

def sample(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None
) -> mx.array:
    """Combined sampling with all strategies."""
    # Apply temperature
    # Apply top-k if specified
    # Apply top-p
    # Sample final token

def beam_search(
    model,
    initial_tokens: mx.array,
    num_beams: int = 4,
    max_length: int = 100,
    length_penalty: float = 1.0
):
    """Beam search decoding (for Whisper, etc.)."""
```

**Patterns to Extract**:

- From `smlx/models/SmolLM2_135M/generate.py`
- From resources: `/resources/mlx-lm/mlx_lm/sample_utils.py`

---

### 6. smlx/utils/preprocessing.py (PRIORITY 3)

**Purpose**: Input preprocessing for multimodal models

**Functions to Implement**:

```python
import mlx.core as mx
from PIL import Image
import numpy as np

def preprocess_image(
    image: Image.Image,
    size: Tuple[int, int] = (224, 224),
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    convert_to_rgb: bool = True
) -> mx.array:
    """Preprocess image for vision models."""
    # Resize
    # Convert to RGB if needed
    # Normalize with mean/std
    # Convert to MLX array

def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160
) -> mx.array:
    """Preprocess audio to mel-spectrogram."""
    # Resample if needed
    # Compute STFT
    # Convert to mel scale
    # Log-scale
    # Return MLX array

def pad_sequence(
    sequences: List[mx.array],
    padding_value: int = 0,
    padding_side: str = "right"
) -> mx.array:
    """Pad sequences to same length."""

def create_attention_mask(
    input_ids: mx.array,
    pad_token_id: int = 0
) -> mx.array:
    """Create attention mask from input IDs."""
```

**Patterns to Extract**:

- From resources: `/resources/mlx-vlm/mlx_vlm/utils.py`
- From resources: `/resources/lightning-whisper-mlx/`

---

### 7. smlx/utils/tokenization.py (PRIORITY 4)

**Purpose**: Tokenizer utilities and helpers

**Functions to Implement**:

```python
def apply_chat_template(
    messages: List[Dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = True
) -> str:
    """Apply chat template to messages."""

def decode_with_skip_special_tokens(
    token_ids: mx.array,
    tokenizer,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True
) -> str:
    """Decode tokens with options."""

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""

def truncate_to_max_length(
    text: str,
    tokenizer,
    max_length: int,
    strategy: str = "end"  # "end", "start", "middle"
) -> str:
    """Truncate text to fit max tokens."""
```

---

### 8. smlx/utils/attention.py (PRIORITY 4)

**Purpose**: Reusable attention mechanisms

**Functions to Implement**:

```python
import mlx.core as mx

def create_causal_mask(
    seq_length: int,
    device: Optional[str] = None
) -> mx.array:
    """Create causal attention mask."""

def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None
) -> mx.array:
    """MLX fast attention."""

def initialize_rope(
    dim: int,
    max_position_embeddings: int = 2048,
    theta: float = 10000.0,
    traditional: bool = False,
    scaling_factor: float = 1.0
) -> Tuple[mx.array, mx.array]:
    """Initialize RoPE (Rotary Position Embeddings)."""

def apply_rope(
    query: mx.array,
    key: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_ids: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """Apply RoPE to query and key."""
```

**Patterns to Extract**:

- From `smlx/models/SmolLM2_135M/model.py`
- From resources: `/resources/mlx-lm/mlx_lm/models/base.py`

---

### 9. smlx/utils/validation.py (PRIORITY 5)

**Purpose**: Model and input validation

**Functions to Implement**:

```python
def validate_model_config(config: BaseModelArgs):
    """Validate model configuration."""

def validate_generation_config(config: GenerationConfig):
    """Validate generation parameters."""

def validate_input_shape(
    input_array: mx.array,
    expected_dims: int,
    expected_shape: Optional[Tuple] = None
):
    """Validate input tensor shape."""

def check_model_memory_requirements(
    config: BaseModelArgs,
    quantization_bits: Optional[int] = None
) -> float:
    """Estimate memory requirements in GB."""
```

---

### 10. smlx/utils/metrics.py (PRIORITY 5)

**Purpose**: Performance and evaluation metrics

**Functions to Implement**:

```python
import time
from contextlib import contextmanager

@contextmanager
def timing(name: str = "Operation"):
    """Context manager for timing operations."""

def calculate_tokens_per_second(
    num_tokens: int,
    elapsed_time: float
) -> float:
    """Calculate generation speed."""

def estimate_model_flops(config: BaseModelArgs, sequence_length: int) -> int:
    """Estimate FLOPs for forward pass."""

def calculate_perplexity(logits: mx.array, targets: mx.array) -> float:
    """Calculate perplexity."""
```

---

## Tools Module Enhancements

The `smlx/tools/` module is **well-implemented** but could benefit from these additions:

### 1. Full Quantization Implementation (HIGH PRIORITY)

**File**: `smlx/tools/convert2mlx.py`

**Current State**: Lines 444-505 have placeholder functions

```python
def quantize_model(weights, config, group_size=64, bits=4):
    # TODO: Implement actual quantization
    # Currently only updates config metadata
```

**Enhancement Needed**:

```python
from smlx.quant.gptq import quantize_weights as gptq_quantize
from smlx.quant.awq import quantize_weights as awq_quantize

def quantize_model(
    weights: Dict[str, mx.array],
    config: Dict,
    group_size: int = 64,
    bits: int = 4,
    method: str = "gptq",  # or "awq"
    calibration_data: Optional[str] = None
) -> Tuple[Dict[str, mx.array], Dict]:
    """Fully quantize model weights."""
    # Load calibration data if needed
    # Apply quantization method
    # Return quantized weights + updated config
```

---

### 2. Batch Processing Utilities (MEDIUM PRIORITY)

**New File**: `smlx/tools/batch_convert.py`

**Purpose**: Convert multiple models efficiently

```python
def batch_convert_models(
    model_list: List[str],
    output_dir: Path,
    quantize: bool = True,
    q_bits: int = 4,
    parallel: bool = True,
    max_workers: int = 4
):
    """Convert multiple models in batch."""

def create_conversion_report(
    conversions: List[Dict],
    output_path: Path
):
    """Generate summary report of conversions."""
```

---

### 3. Model Registry System (MEDIUM PRIORITY)

**New File**: `smlx/tools/registry.py`

**Purpose**: Central registry of available models

```python
@dataclass
class ModelInfo:
    name: str
    repo_id: str
    architecture: str
    size: str  # "135M", "256M", etc.
    type: str  # "language", "vision-language", "audio"
    quantized: bool = False
    local_path: Optional[Path] = None

class ModelRegistry:
    """Registry of all SMLX models."""

    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}

    def register(self, model_info: ModelInfo):
        """Register a model."""

    def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List available models."""

    def get_model_info(self, name: str) -> ModelInfo:
        """Get model information."""

    def is_downloaded(self, name: str) -> bool:
        """Check if model is cached locally."""

# Global registry instance
REGISTRY = ModelRegistry()

# Pre-register all known models
REGISTRY.register(ModelInfo(
    name="smollm2-135m",
    repo_id="mlx-community/SmolLM2-135M-Instruct",
    architecture="SmolLM3",
    size="135M",
    type="language"
))
# ... etc for all 17 models
```

---

### 4. Dataset Preprocessing Tools (LOW PRIORITY)

**New File**: `smlx/tools/prepare_dataset.py`

**Purpose**: Prepare datasets for fine-tuning/evaluation

```python
def prepare_for_finetuning(
    dataset_path: Path,
    output_path: Path,
    tokenizer,
    max_length: int = 2048,
    dataset_format: str = "jsonl"
):
    """Prepare dataset for fine-tuning."""

def prepare_for_evaluation(
    dataset_name: str,
    split: str = "test",
    cache_dir: Optional[Path] = None
):
    """Prepare evaluation dataset."""
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2) 🏗️

**Goal**: Build shared utilities and infrastructure

**Tasks**:

1. ✅ Implement `smlx/utils/config.py` (2 days)
   - BaseModelArgs with from_dict/to_dict
   - load_config, validate_config, merge_configs
   - estimate_parameters, print_config

2. ✅ Implement `smlx/utils/loading.py` (2 days)
   - resolve_model_path, load_weights, save_weights
   - sanitize_weights, load_tokenizer
   - verify_weights

3. ✅ Implement `smlx/utils/cache.py` (1 day)
   - Move KVCache and RotatingKVCache from SmolLM2_135M
   - Add factory function make_cache
   - Add tests

4. ✅ Implement `smlx/utils/generation.py` (2 days)
   - GenerationConfig dataclass
   - generate_step, generate, stream_generate
   - create_stopping_criteria, apply_repetition_penalty

5. ✅ Implement `smlx/utils/sampling.py` (1 day)
   - sample_with_temperature, top_p_sampling, top_k_sampling
   - Combined sample function
   - beam_search (for Whisper)

6. ✅ Write comprehensive tests (2 days)
   - Unit tests for all utils modules
   - Integration tests for loading pipeline
   - Test with SmolLM2_135M

**Deliverables**:

- Fully functional `smlx/utils/` module
- 50+ unit tests
- Documentation for each utility
- Updated CLAUDE.md with utils guidance

---

### Phase 2: Language Models (Week 3) 📝

**Goal**: Complete SmolLM2 family

**Tasks**:

1. ✅ Implement SmolLM2-360M (2 days)
   - Copy SmolLM2-135M structure
   - Update config for 360M parameters
   - Reuse all utils from Phase 1
   - Add integration tests
   - Create example script

2. ✅ Refactor SmolLM2-135M to use utils (1 day)
   - Replace local implementations with smlx.utils
   - Verify tests still pass
   - Clean up duplicate code

3. ✅ Documentation (1 day)
   - Update model READMEs
   - Add comparison guide (135M vs 360M)
   - Performance benchmarks on M4

**Deliverables**:

- SmolLM2-360M fully implemented
- SmolLM2-135M refactored
- Working examples for both

---

### Phase 3: Vision-Language Models (Weeks 4-6) 👁️

**Goal**: Implement core VLMs starting with SmolVLM

**Tasks**:

**Week 4: SmolVLM-256M** (flagship VLM)

1. ✅ Implement vision encoder (2 days)
   - `vision.py` with VisionEncoder class
   - Patch embedding layer
   - Vision transformer blocks
   - Study `/resources/mlx-vlm/mlx_vlm/models/`

2. ✅ Implement image processor (1 day)
   - `processor.py` with ImageProcessor class
   - Resize, normalize, patchify
   - Return pixel_values tensor

3. ✅ Implement connector/projector (1 day)
   - MLP projection from vision to text space
   - Integration with language model

4. ✅ Implement multimodal model (2 days)
   - Combine vision + language
   - Handle image + text inputs
   - Update generate.py for multimodal

5. ✅ Testing and examples (1 day)
   - Integration tests with sample images
   - Example scripts for common tasks
   - Benchmark on M4

**Week 5: SmolVLM-500M**

1. ✅ Copy SmolVLM-256M patterns (2 days)
   - Scale up components
   - Update configs
   - Add tests

2. ✅ Implement Moondream2 (3 days)
   - Study reference in `/resources/mlx-vlm/`
   - Implement SigLIP + Phi-2 architecture
   - Production optimizations
   - Tests and examples

**Week 6: More VLMs**

1. ✅ Implement TinyLLaVA (3 days)
   - Study `/resources/mlx-vlm/mlx_vlm/models/llava/`
   - CLIP encoder + LLaMA decoder
   - Tests and examples

2. ✅ Implement nanoVLM (2 days)
   - Similar to SmolVLM patterns
   - Tests and examples

**Deliverables**:

- 5 VLMs fully implemented
- Shared vision utilities in `smlx/utils/preprocessing.py`
- Comprehensive multimodal examples
- Vision-language generation guide

---

### Phase 4: Audio Models (Weeks 7-9) 🎵

**Goal**: Implement audio processing models

**Tasks**:

**Week 7-8: Whisper-tiny** (most important audio model)

1. ✅ Implement audio processor (2 days)
   - `processor.py` with AudioProcessor class
   - Mel-spectrogram computation
   - Study `/resources/lightning-whisper-mlx/`

2. ✅ Implement audio encoder (2 days)
   - Conv1D feature extraction
   - Transformer encoder
   - Sinusoidal positional embeddings

3. ✅ Implement text decoder (2 days)
   - Transformer decoder with cross-attention
   - Integration with encoder

4. ✅ Implement decoding logic (2 days)
   - Beam search
   - Language detection
   - Timestamp extraction
   - Greedy decoding

5. ✅ Testing and examples (2 days)
   - Test with sample audio files
   - Multilingual examples
   - Transcription benchmarks
   - Streaming transcription

**Week 9: Other Audio Models**

1. ✅ Implement Silero VAD (1 day)
   - Simple voice activity detection
   - Tests with speech samples

2. ✅ Implement YAMNet (2 days)
   - Audio classification
   - MobileNet-based architecture
   - Tests with audio samples

3. ✅ Implement Orpheus-150M (2 days)
   - Audio processing model
   - Tests and examples

**Deliverables**:

- 4 audio models fully implemented
- Shared audio utilities in `smlx/utils/preprocessing.py`
- Audio processing examples
- Transcription and classification guides

---

### Phase 5: Specialized Models (Weeks 10-11) 🎯

**Goal**: Implement OCR and embedding models

**Tasks**:

**Week 10: OCR Models**

1. ✅ Implement TrOCR-small (2 days)
   - Vision encoder + text decoder
   - Reuse vision patterns from VLMs
   - Tests with document images

2. ✅ Implement Donut-base (3 days)
   - Swin Transformer encoder
   - BART decoder
   - Document understanding
   - Tests and examples

**Week 11: Embedding Models**

1. ✅ Implement MiniLM (2 days)
   - BERT-style encoder
   - Sentence embedding output
   - Tests with similarity tasks

2. ✅ Implement all-MiniLM-L6-v2 (1 day)
   - Use MiniLM patterns
   - Pretrained weights

3. ✅ Research and implement Chatterbox (2 days)
   - Determine architecture
   - Implement based on findings
   - Tests and chat examples

**Deliverables**:

- OCR models for document processing
- Embedding models for similarity search
- Chat model implementation
- Complete model coverage (17/17 models)

---

### Phase 6: Polish and Optimization (Week 12) ✨

**Goal**: Finalize, optimize, and document everything

**Tasks**:

1. ✅ Complete quantization implementation (2 days)
   - Full GPTQ integration in convert2mlx.py
   - Quantize all models to 4-bit
   - Performance comparison

2. ✅ Implement model registry (1 day)
   - `smlx/tools/registry.py`
   - Pre-register all 17 models
   - CLI for model management

3. ✅ Comprehensive documentation (2 days)
   - Update CLAUDE.md with all patterns
   - Create model comparison guide
   - Write migration guide (from mlx-lm/mlx-vlm)
   - API documentation

4. ✅ Performance benchmarking (2 days)
   - Benchmark all models on M4 36GB
   - Memory usage profiles
   - Tokens/second metrics
   - Quantization impact analysis

5. ✅ Final testing (1 day)
   - Full test suite run
   - Fix any remaining issues
   - Verify all examples work

**Deliverables**:

- All 17 models implemented and tested
- Complete utils and tools modules
- Comprehensive documentation
- Performance benchmark report
- Ready for production use

---

## Resource Mapping

### Where to Find Patterns

Based on exploration of `/resources/`, here's exactly where to look for each model type:

#### Language Models (Transformer Decoder)

**Primary Resources**:

- `/resources/mlx-lm/mlx_lm/models/` - Core patterns
  - `base.py` (138 lines) - BaseModelArgs, config utilities
  - `gemma.py` (180 lines) - Clean example of transformer
  - `cache.py` (1000 lines) - KV cache implementations
  - `llama.py` - Attention with RoPE
  - `mistral.py` - Sliding window attention

**Key Files to Copy**:

1. BaseModelArgs class → `smlx/utils/config.py`
2. Attention mechanism → `smlx/utils/attention.py`
3. KVCache → `smlx/utils/cache.py`
4. Generation loop → `smlx/utils/generation.py`

---

#### Vision-Language Models

**Primary Resources**:

- `/resources/mlx-vlm/mlx_vlm/models/` - VLM patterns
  - `llava/` - CLIP + LLaMA architecture
  - `smolvlm/` - SmolVLM if available
  - `paligemma/` - PaliGemma architecture
  - `phi3_v/` - Phi-3 with vision

**Key Files to Study**:

1. Vision encoder patterns
2. Image processor (resize, normalize, patchify)
3. Vision-language connector/projector
4. Multimodal generation

**Components**:

- Vision Encoder: Typically ViT or SigLIP
- Projector: 2-3 layer MLP
- Language Model: Standard transformer decoder
- Image Processor: PIL + normalization

---

#### Audio Models (Whisper)

**Primary Resources**:

- `/resources/lightning-whisper-mlx/` - Optimized Whisper
- `/resources/mlx-examples/whisper/` - Canonical implementation

**Key Files**:

1. Audio preprocessing (waveform → mel-spectrogram)
2. Audio encoder (Conv1D + Transformer)
3. Text decoder (Transformer with cross-attention)
4. Beam search decoding
5. Language detection

**Special Considerations**:

- Mel-spectrogram computation (80 channels)
- Multilingual support (99 languages)
- Timestamp alignment
- Streaming support

---

#### OCR Models

**Primary Resources**:

- Combine VLM vision encoder patterns
- Sequence-to-sequence generation

**Architecture**:

- Vision Encoder: ViT or Swin Transformer
- Text Decoder: Standard transformer
- Similar to VLM but image → text only

---

#### Embedding Models

**Primary Resources**:

- `/resources/mlx-examples/` - BERT-style encoders
- Simpler than decoder models (encoder-only)

**Key Differences**:

- No causal masking (bidirectional attention)
- No generation (encode only)
- Output: pooled embeddings (typically [CLS] token)
- Mean pooling or CLS pooling strategies

---

## Testing Strategy

### Test Organization

```
tests/
├── unit/                           # Fast, isolated tests
│   ├── test_config.py             # Config utilities
│   ├── test_loading.py            # Loading utilities
│   ├── test_cache.py              # KV cache
│   ├── test_sampling.py           # Sampling strategies
│   ├── test_generation.py         # Generation utilities
│   └── test_preprocessing.py      # Preprocessing
│
├── integration/                    # Model integration tests
│   ├── test_smollm2_135m.py       # ✅ Existing
│   ├── test_smollm2_360m.py       # New
│   ├── test_smolvlm_256m.py       # New
│   ├── test_whisper_tiny.py       # New
│   └── ...                        # One per model
│
├── quant/                          # ✅ Quantization tests
│   ├── test_gptq.py
│   ├── test_dynamic.py
│   └── ...
│
├── evals/                          # ✅ Evaluation benchmarks
│   ├── test_math_vista.py
│   ├── test_mmmu.py
│   └── ...
│
└── tools/                          # ✅ Tools tests
    ├── test_convert2mlx.py
    └── ...
```

---

### Test Markers (from pytest.ini)

Use these markers for all tests:

```python
@pytest.mark.unit              # Fast, isolated unit tests
@pytest.mark.integration       # Integration tests
@pytest.mark.slow              # Slow-running tests (skip in CI)
@pytest.mark.gpu               # Requires GPU/MLX acceleration
@pytest.mark.requires_model    # Downloads models (skip if offline)
@pytest.mark.benchmark         # Performance benchmarks
@pytest.mark.eval              # Evaluation tests
```

---

### Testing Checklist per Model

For each new model implementation, ensure:

**✅ Unit Tests**:

- [ ] Config loading and validation
- [ ] Model initialization
- [ ] Forward pass with dummy inputs
- [ ] Shape validation for all layers
- [ ] Parameter count matches expected

**✅ Integration Tests** (`@pytest.mark.integration`):

- [ ] Load model from HuggingFace Hub
- [ ] Load model from local cache
- [ ] Basic generation with sample input
- [ ] Streaming generation works
- [ ] Cache update and fetch
- [ ] Quantized model loading (4-bit, 8-bit)

**✅ Slow Tests** (`@pytest.mark.slow`):

- [ ] Generate 100+ tokens
- [ ] Multiple generation rounds
- [ ] Long sequence handling
- [ ] Memory profiling

**✅ Model-Specific Tests**:

- Language: Text completion, chat interface
- Vision-Language: Image + text inputs, vision-only
- Audio: Audio file transcription, multilingual
- OCR: Document image processing
- Embedding: Similarity computation

---

### Continuous Testing

**Pre-commit**:

```bash
# Fast tests only
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m "unit and not slow"
```

**Full Test Suite**:

```bash
# All tests including integration
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest
```

**Nightly/Weekly**:

```bash
# Including slow and GPU tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest -m "slow or gpu"
```

---

## Data Availability

### Model Weights

All models are available via HuggingFace Hub and can be downloaded using existing `smlx/tools/download.py`:

**Language Models**:

- ✅ SmolLM2-135M: `mlx-community/SmolLM2-135M-Instruct`
- ✅ SmolLM2-360M: `HuggingFaceTB/SmolLM2-360M-Instruct`

**Vision-Language Models**:

- ✅ SmolVLM-256M: `HuggingFaceTB/SmolVLM-256M-Instruct`
- ✅ SmolVLM-500M: `HuggingFaceTB/SmolVLM-500M-Instruct`
- ✅ Moondream2: `vikhyatk/moondream2`
- ✅ TinyLLaVA: `bczhou/TinyLLaVA-1.5B`
- ✅ nanoVLM: Available on HF Hub

**Audio Models**:

- ✅ Whisper-tiny: `openai/whisper-tiny`
- ✅ Silero VAD: `snakers4/silero-vad`
- ✅ YAMNet: Available via TensorFlow Hub (needs conversion)
- ✅ Orpheus-150M: Check HF Hub

**OCR Models**:

- ✅ TrOCR-small: `microsoft/trocr-small-printed`
- ✅ Donut-base: `naver-clova-ix/donut-base`

**Embedding Models**:

- ✅ MiniLM: `microsoft/MiniLM-L12-H384-uncased`
- ✅ all-MiniLM-L6-v2: `sentence-transformers/all-MiniLM-L6-v2`

---

### Evaluation Datasets

All datasets available via `smlx/tools/download_data.py`:

**Vision-Language**:

- ✅ MathVista: `AI4Math/MathVista`
- ✅ MMMU: `MMMU/MMMU`
- ✅ MMStar: Available on HF Hub
- ✅ OCRBench: Available on HF Hub

**Language**:

- ✅ General calibration data: Implemented in download.py

---

### Download Commands

**Download all test models**:

```bash
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.tools.download_data --all
```

**Download specific model**:

```bash
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.tools.download_data \
  --model mlx-community/SmolVLM-256M-Instruct
```

**Download evaluation datasets**:

```bash
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.tools.download_data --datasets
```

**Check cache location**:

```bash
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.tools.download_data --cache-dir
# Output: /Users/ryanoboyle/.cache/smlx/
```

---

## Summary and Next Steps

### Current Status

✅ **Fully Functional**:

- 1/17 models (SmolLM2-135M) - Reference implementation
- Complete tools module (download, convert, CLI)
- Quantization framework (AWQ, GPTQ, LoRA, DoRA)
- Testing infrastructure (pytest with markers)
- Evaluation framework (stubs)
- Documentation (READMEs, CLAUDE.md)

❌ **Missing**:

- 16/17 models (stubs only)
- All of smlx/utils/ (empty)
- Full quantization in convert2mlx.py
- Model registry system
- Batch processing utilities

---

### Recommended Starting Point

**Option 1: Build Foundation First** (Recommended)

1. Implement `smlx/utils/` module (Phase 1 - 2 weeks)
2. Refactor SmolLM2-135M to use utils
3. Implement SmolLM2-360M as validation
4. Then proceed with VLM/Audio models

**Option 2: Implement One Model Per Category**

1. SmolLM2-360M (language)
2. SmolVLM-256M (vision-language)
3. Whisper-tiny (audio)
4. Extract common patterns to utils as you go

**Option 3: Focus on Highest Priority Models**

1. SmolVLM-256M (flagship VLM)
2. Whisper-tiny (essential audio)
3. SmolLM2-360M (language completion)

---

### Key Success Factors

1. **Follow SmolLM2-135M patterns exactly**
   - File structure
   - API surface
   - Config management
   - Testing approach

2. **Copy from resources/, don't import**
   - Study reference implementations
   - Adapt for "smol" models
   - Maintain model-specific optimizations

3. **Build utils as you find duplication**
   - Extract common patterns
   - Share across models
   - Keep models DRY

4. **Test incrementally**
   - Unit tests first
   - Integration tests second
   - Slow/GPU tests last

5. **Document as you go**
   - Update READMEs
   - Add examples
   - Note any deviations from patterns

---

### Questions to Consider

1. **Priority**: Which models are most important for your use case?
   - Language, vision-language, audio, OCR, embeddings?

2. **Timeline**: 12-week roadmap or faster/slower pace?

3. **Quantization**: Prioritize 4-bit quantization for all models?

4. **Testing**: How comprehensive? (unit only vs full integration)

5. **Utils**: Build all at once (Phase 1) or extract as needed?

---

## Resources Created

During this analysis, the following documentation files were created:

1. **RESOURCES_INDEX.md** (8 KB) - Master navigation
2. **RESOURCES_QUICK_START.md** (8 KB) - Quick implementation guide
3. **RESOURCES_REFERENCE_MAP.md** (16 KB) - Code patterns with line numbers
4. **RESOURCES_PATTERNS.md** (24 KB) - Comprehensive pattern reference

All files are in `/Users/ryanoboyle/smlx/` and ready for use during implementation.

---

**END OF IMPLEMENTATION PLAN**
