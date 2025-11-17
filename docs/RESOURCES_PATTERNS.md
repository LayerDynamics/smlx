# Comprehensive Report: Reusable Patterns in Resources Directory

## Executive Summary

The resources/ directory contains three primary reference implementations (mlx-lm, mlx-vlm, mlx-examples) with highly reusable patterns for implementing small ML models on MLX. These implementations follow consistent architectural patterns that should be adapted for the SMLX project rather than imported directly.

## Directory Structure Overview

```ascii
resources/
├── mlx-examples/              # Generic example implementations
├── mlx-lm/                    # 100+ language models with unified interface
├── mlx-vlm/                   # 36+ vision-language models
├── lightning-whisper-mlx/      # Audio/speech model
├── mflux/                      # Flux diffusion implementation
├── mlx/                        # Core MLX framework
├── chat-with-mlx/              # Chat interface
├── ml-aim/                     # AIM models
└── transformerlab-app/         # Full application
```

## 1. CORE ARCHITECTURE PATTERNS

### 1.1 Model Structure Pattern (Text Models)

All text models in mlx-lm follow this consistent structure:

**File Organization:**

```python
smlx/models/ModelName/
├── __init__.py          # Export public API
├── model.py            # Core architecture
├── loader.py           # HuggingFace loading
├── generate.py         # Generation logic
├── cache.py            # KV cache (if needed)
├── config.py           # Configuration
└── tokenizer_utils.py  # Optional: Tokenizer helpers
```

**Key Pattern - ModelArgs (Configuration):**

```python
from dataclasses import dataclass
from typing import Optional
import inspect

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        """Load config from dict, filtering unknown keys"""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    # ... other fields
```

**Reference:** mlx-lm/mlx_lm/models/base.py

### 1.2 Model Architecture Pattern

**Core Components:**

1. **Attention Module** - Uses RoPE positional encoding

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        self.rope = nn.RoPE(head_dim, traditional=False, base=10000)
    
    def __call__(self, x, mask=None, cache=None):
        # Query, key, value projections
        # RoPE application
        # Scaled dot-product attention (with cache)
```

2. **MLP Module** - Gated linear units

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
    
    def __call__(self, x):
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))
```

3. **Transformer Block** - Pre-normalization residual

```python
class TransformerBlock(nn.Module):
    def __init__(self, args):
        self.attention = Attention(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(self, x, mask=None, cache=None):
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r, cache
```

4. **Scaled Dot-Product Attention** - Unified function

```python
def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):  # Quantized cache
        return quantized_scaled_dot_product_attention(...)
    else:
        return mx.fast.scaled_dot_product_attention(...)
```

**Reference:** mlx-lm/mlx_lm/models/gemma.py (180 lines, clean structure)

### 1.3 Model Class Pattern

```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        self.vocab_size = args.vocab_size
        self.n_layers = args.num_hidden_layers
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    def __call__(self, x, cache=None):
        x = self.embed_tokens(x)
        caches = []
        
        for i, layer in enumerate(self.layers):
            x, c = layer(x, mask=None, cache=cache[i] if cache else None)
            caches.append(c)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, caches
```

## 2. CACHE PATTERNS

### 2.1 KV Cache Architecture

**Base Cache Interface:**

```python
class _BaseCache:
    @property
    def state(self):
        """Return cache state for serialization"""
        return []
    
    @property
    def meta_state(self):
        """Return metadata"""
        return ""
    
    def is_trimmable(self):
        return False
    
    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj
```

**Standard KVCache:**

```python
class KVCache(_BaseCache):
    def __init__(self):
        self.offset = 0
        self._k = mx.array([])
        self._v = mx.array([])
    
    def update_and_fetch(self, k, v):
        """Append new k,v and return full sequence"""
        self._k = mx.concatenate([self._k, k], axis=2)
        self._v = mx.concatenate([self._v, v], axis=2)
        return self._k, self._v
```

**Rotating KVCache (for long sequences):**

```python
class RotatingKVCache(_BaseCache):
    def __init__(self, max_size=2048, keep=4):
        self.max_size = max_size
        self.keep = keep  # Keep first N tokens
        self.offset = 0
```

**Cache Creation Helper:**

```python
def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    if hasattr(model, "make_cache"):
        return model.make_cache()
    
    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size) for _ in range(num_layers)]
    else:
        return [KVCache() for _ in range(num_layers)]
```

**Reference:** mlx-lm/mlx_lm/models/cache.py (33KB, comprehensive)

## 3. GENERATION & SAMPLING PATTERNS

### 3.1 Sampling Pipeline

```python
def make_sampler(
    temp: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    xtc_probability: float = 0.0,
) -> Callable[[mx.array], mx.array]:
    """Create composable sampler from parameters"""
    
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)  # Greedy
    
    sampling_methods = []
    
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))
    
    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        return categorical_sampling(logprobs, temp)
    
    return sampler
```

**Reference:** mlx-lm/mlx_lm/sample_utils.py (250 lines)

### 3.2 Logits Processors

```python
def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
):
    """Create composable logits processors"""
    processors = []
    
    if logit_bias:
        def processor(tokens, logits):
            logits[:, indices] += values
            return logits
        processors.append(processor)
    
    if repetition_penalty:
        def processor(tokens, logits):
            # Penalize repeated tokens
            return logits
        processors.append(processor)
    
    return processors
```

### 3.3 Generation Loop Pattern

```python
def generate(
    model: nn.Module,
    tokenizer,
    prompt_tokens: List[int],
    max_tokens: int,
    temp: float = 0.8,
    top_p: float = 1.0,
    sampler=None,
):
    """Token-by-token generation with cache"""
    cache = make_prompt_cache(model)
    sampler = sampler or make_sampler(temp=temp, top_p=top_p)
    
    for _ in range(max_tokens):
        # Forward pass
        logits, cache = model(prompt_tokens[-1:], cache=cache)
        
        # Apply processors
        logits = logits[:, -1, :]
        
        # Sample next token
        next_token = sampler(logits[0])
        prompt_tokens.append(next_token.item())
        
        yield next_token
        
        if next_token == tokenizer.eos_token_id:
            break
```

**Reference:** mlx-lm/mlx_lm/generate.py (1000+ lines, comprehensive)

### 3.4 Streaming Generation Pattern

Uses StreamingDetokenizer for incremental text output:

```python
class StreamingDetokenizer:
    """Detokenize one token at a time"""
    
    def reset(self):
        self.tokens = []
        self.offset = 0
    
    def add_token(self, token_id):
        self.tokens.append(token_id)
    
    @property
    def last_segment(self):
        """Return new text since last access"""
        text = self.text
        segment = text[self.offset:]
        self.offset = len(text)
        return segment
```

**Reference:** mlx-lm/mlx_lm/tokenizer_utils.py

## 4. LOADER & TOKENIZER PATTERNS

### 4.1 Model Loading Pattern

```python
def load(model_name: str, adapter_path: Optional[str] = None):
    """Load model and tokenizer from HuggingFace"""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    
    # Download from HF Hub
    model_path = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.safetensors", "config.json", "*.tokenizer.json"]
    )
    
    # Load config
    with open(f"{model_path}/config.json") as f:
        config = json.load(f)
    
    # Create model
    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)
    
    # Load weights
    weights = mx.load(f"{model_path}/model.safetensors")
    model.update_state(weights)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer
```

### 4.2 Tokenizer Wrapper Pattern

```python
class TokenizerWrapper:
    """Unified tokenizer interface"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.detokenizer = StreamingDetokenizer(tokenizer)
    
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
```

**Reference:** mlx-lm/mlx_lm/utils.py (500 lines, comprehensive utilities)

## 5. QUANTIZATION PATTERNS

### 5.1 GPTQ Quantization

**Key Pattern:**

```python
def gptq_quantize(
    model,
    data,
    bits: int = 4,
    group_size: int = 64,
    batch_size: int = 8,
):
    """Post-training quantization"""
    
    # Collect Hessian information
    layers = [l for _, l in tree_flatten(model.leaf_modules())]
    for layer in layers:
        catcher = Catcher(layer)
        # Forward pass to compute H = X^T X
    
    # Quantize layer by layer
    for layer in quantizable_layers:
        # Compute scales and biases to minimize ||W_q - W||
        scales = compute_scales(H, bits, group_size)
        biases = compute_biases(layer.weight, scales)
        
        # Quantize weights
        layer.weight = quantize(layer.weight, bits, scales, biases)
```

**Reference:** mlx-lm/mlx_lm/quant/gptq.py (200 lines)

### 5.2 AWQ (Activation-Aware Weight Quantization)

**Key Pattern:**

```python
def awq_quantize(model, data, bits=4, group_size=64):
    """Activation-aware quantization"""
    # Collect activation statistics
    # Find optimal scaling factors based on activations
    # Quantize with activation-aware scaling
```

**Reference:** mlx-lm/mlx_lm/quant/awq.py (400 lines)

### 5.3 LoRA Fine-tuning

```python
class LoRALinear(nn.Module):
    @staticmethod
    def from_base(linear: nn.Linear, r: int = 8, dropout: float = 0.0):
        """Convert Linear layer to LoRA"""
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(input_dims, output_dims, r, dropout)
        lora_lin.linear = linear  # Store reference
        return lora_lin
    
    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_a.T @ self.lora_b.T) * self.scale
        return base_out + lora_out
    
    def fuse(self):
        """Merge LoRA into base weights"""
        delta = (self.scale * self.lora_b.T @ self.lora_a.T)
        self.linear.weight = self.linear.weight + delta
        return self.linear
```

**Reference:** mlx-lm/mlx_lm/tuner/lora.py (200 lines)

## 6. VISION-LANGUAGE MODEL PATTERNS

### 6.1 SmolVLM Architecture Pattern

**File Structure:**

```python
smlx/models/SmolVLM/
├── __init__.py
├── config.py          # Text + Vision config
├── vision_encoder.py  # Image -> embeddings
├── text_encoder.py    # Standard LLM
├── processor.py       # Image preprocessing
└── model.py           # Unified model
```

**Key Pattern - Multimodal Input Preparation:**

```python
class Model(nn.Module):
    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        """Interleave image and text embeddings"""
        B, T, D_text = inputs_embeds.shape
        N, S, D_img = image_features.shape
        
        # Find <image> token positions
        image_token_index = self.config.image_token_index
        image_positions = np.where(input_ids == image_token_index)[1]
        
        segments = []
        for pos, img_feat in zip(image_positions, image_features):
            # Add text before image
            segments.append(inputs_embeds[:pos])
            # Add image features
            segments.append(img_feat)
        
        return mx.concatenate(segments, axis=0)
    
    def __call__(self, pixel_values, input_ids):
        # Vision encoder
        image_features = self.vision_model(pixel_values)
        
        # Text embedding
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Interleave
        merged = self._prepare_inputs_for_multimodal(...)
        
        # LLM forward pass
        return self.language_model(merged)
```

**Reference:** mlx-vlm/mlx_vlm/models/smolvlm/smolvlm.py

### 6.2 Vision Encoder Patterns

Common patterns:

- ViT (Vision Transformer) style: Patch embedding + transformer
- CNN backbones: ResNet-style feature extraction
- Attention pooling: Global context aggregation

## 7. AUDIO MODEL PATTERNS (Whisper)

### 7.1 Audio Processing Pipeline

```python
class AudioProcessor:
    def __call__(self, audio_waveform: np.ndarray):
        # Convert to mel spectrogram
        mel_spec = self.to_mel(audio_waveform)
        
        # Pad/truncate to max length
        mel_spec = pad_or_trim(mel_spec, self.n_audio_ctx)
        
        # Normalize
        mel_spec = (mel_spec - mean) / std
        
        return mx.array(mel_spec)
```

### 7.2 Encoder-Decoder with Cross-Attention

```python
class WhisperEncoderLayer(nn.Module):
    def __init__(self, n_state, n_head):
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.ffn = FeedForward(n_state)
        self.ffn_ln = nn.LayerNorm(n_state)

class WhisperDecoderLayer(nn.Module):
    def __init__(self, n_state, n_head):
        self.attn = MultiHeadAttention(n_state, n_head)
        self.cross_attn = MultiHeadAttention(n_state, n_head)  # Cross-attention
        self.ffn = FeedForward(n_state)
```

**Reference:** lightning-whisper-mlx/lightning_whisper_mlx/whisper.py

## 8. UTILITIES & HELPERS

### 8.1 Common Utility Functions

```python
# Tree operations
from mlx.utils import tree_flatten, tree_map, tree_unflatten

# Model utilities
def get_total_parameters(model):
    """Count model parameters (handles quantized)"""
    ...

def compute_bits_per_weight(model):
    """Compute effective bits per weight"""
    ...

# Loading utilities
def load(model_name: str):
    """Unified model loading"""
    ...

def save(model, path: str):
    """Save model with safetensors"""
    ...

def dequantize_model(model):
    """Convert quantized to full precision"""
    ...
```

### 8.2 Chat Interface Pattern

```python
def chat(
    model,
    tokenizer,
    system_prompt: str = "",
    messages: List[Dict[str, str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
):
    """Multi-turn chat interface"""
    
    # Build prompt from messages
    prompt = build_chat_template(tokenizer, system_prompt, messages)
    
    # Generate response
    response = stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return response
```

## 9. SERVER/API PATTERNS

### 9.1 Server Infrastructure

mlx-lm provides a FastAPI server implementation with:

- HTTP endpoints for generation
- WebSocket support for streaming
- Request batching
- Model loading/management
- Adapter support

**Reference:** mlx-lm/mlx_lm/server.py (1000+ lines, production-ready)

## 10. UNIQUE PATTERNS BY MODEL TYPE

### 10.1 Text-Only Models

- Standard attention + MLP
- Causal masking
- KV cache for efficiency
- RoPE positional encoding

### 10.2 Vision-Language Models

- Separate vision & text encoders
- Image tokenization (patch embedding)
- Interleaving strategy (image tokens inserted in text)
- Cross-modal attention
- Special image/video tokens

### 10.3 Audio Models

- Mel spectrogram preprocessing
- Encoder-decoder architecture
- Cross-attention for encoder-decoder
- Beam search decoding for sequences
- Language detection

### 10.4 Multimodal (Vision + Audio + Text)

- Multiple input processors
- Unified embedding space
- Sequential processing: audio -> text -> image
- Attention mechanisms for fusion

## 11. CONVERSION PATTERNS

### 11.1 HuggingFace to MLX Conversion

```python
def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
):
    # Load HF model
    # Map parameter names (HF -> MLX naming conventions)
    # Convert data types if needed
    # Apply quantization
    # Save with safetensors
```

**Reference:** mlx-lm/mlx_lm/convert.py

### 11.2 Mixed Quantization

Applies different quantization levels to different layers:

```python
def mixed_quant_predicate_builder(recipe: str, model: nn.Module):
    # Different bits for:
    # - First 1/8 layers: 6 bits
    # - Last 1/8 layers: 6 bits
    # - Middle layers: 2-4 bits
    # - v_proj & down_proj: higher bits
    # - lm_head: 6 bits
```

## 12. DATA & TRAINING PATTERNS

### 12.1 Data Loading for Fine-tuning

```python
class Dataset(mx.data.Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        # Load data (JSONL, text files)
        # Tokenize
        # Create sequences of max_length
    
    def __getitem__(self, idx):
        return {"input_ids": ..., "target_ids": ...}
```

### 12.2 Training Loop Pattern

```python
def train_step(
    model,
    optimizer,
    loss_fn,
    x: mx.array,
    y: mx.array,
):
    def loss_and_grad(model):
        logits = model(x)
        loss = loss_fn(logits, y)
        return loss
    
    loss, grads = mx.value_and_grad(loss_and_grad)(model)
    optimizer.update(model, grads)
    return loss
```

**Reference:** mlx-examples/lora/lora.py

## 13. CRITICAL PATTERNS FOR "SMOL" MODELS

### 13.1 Memory Optimization

- Rotating KV cache for long sequences
- Quantization support (4-bit, 8-bit)
- LoRA for efficient fine-tuning
- Parameter sharing where possible
- Reduced embedding dimensions

### 13.2 Efficiency Patterns

- Lazy evaluation with `mx.eval()` only when needed
- Tree operations for efficient updates
- Compiled generation loops
- Efficient attention masks
- Batch processing

### 13.3 Quantization Integration

- Models should support both full and quantized weights
- LoRA should work with quantized models
- Cache should be quantizable
- Seamless switching between precisions

## 14. FILE FORMAT & SERIALIZATION

### 14.1 Model Serialization

- Use safetensors format (not pickle)
- Store with metadata
- Support loading from HuggingFace Hub
- Version compatibility handling

```python
# Save
mx.save_safetensors(
    "model.safetensors",
    model.state_dict(),
    metadata={"model_type": "gemma", "vocab_size": "vocab_size"}
)

# Load
arrays, metadata = mx.load("model.safetensors", return_metadata=True)
model.update_state(arrays)
```

### 14.2 Config Format

- JSON configuration files
- Dataclass-based config with `from_dict()`
- Validation on load

## 15. KEY REUSABLE UTILITIES

| Utility | Location | Purpose |
|---------|----------|---------|
| `make_sampler()` | sample_utils.py | Create composable samplers |
| `make_logits_processors()` | sample_utils.py | Token filtering/processing |
| `make_prompt_cache()` | models/cache.py | KV cache creation |
| `TokenizerWrapper` | tokenizer_utils.py | Unified tokenizer interface |
| `StreamingDetokenizer` | tokenizer_utils.py | Incremental decoding |
| `tree_flatten/unflatten` | mlx.utils | Recursive tree operations |
| `scaled_dot_product_attention()` | models/base.py | Unified attention computation |
| `create_causal_mask()` | models/base.py | Attention mask creation |
| `load()` | utils.py | Unified model loading |
| `save()` | utils.py | Unified model saving |

## 16. ANTI-PATTERNS TO AVOID

1. **Direct imports from resources/** - Copy and adapt, don't import
2. **PyTorch/TensorFlow in MLX models** - Use only MLX operations
3. **Unbounded sequences** - Use rotating cache for long sequences
4. **Mixed quantization levels in same layer** - Keep consistent within layer
5. **Not supporting quantization** - All models should support 4-bit, 8-bit
6. **Inefficient attention masks** - Use lazy evaluation, causal string when possible
7. **Assuming batch size 1** - Support arbitrary batch dimensions

## 17. RECOMMENDED IMPLEMENTATION APPROACH

For each new model in SMLX:

1. **Copy the architecture pattern** from reference (e.g., gemma.py)
2. **Adapt class names** and parameter names to match model
3. **Implement loader** using mlx-lm pattern with HuggingFace Hub
4. **Add generation** support with sampling pipeline
5. **Integrate cache** using base KVCache pattern
6. **Test with quantization** (AWQ or GPTQ)
7. **Add LoRA support** for fine-tuning
8. **Create examples** demonstrating all features

## Summary of Most Important Patterns

1. **Config Pattern** (BaseModelArgs) - Apply to ALL models
2. **Module Pattern** (Attention, MLP, TransformerBlock) - Reuse across text models
3. **Generation Pattern** (make_sampler, generate loop) - Unified interface
4. **Cache Pattern** (KVCache, RotatingKVCache) - Critical for efficiency
5. **Loading Pattern** (load, save with HF integration) - Standardized
6. **Quantization Pattern** (support bits parameter) - Required for small models
7. **Chat Pattern** (multi-turn with templates) - Improve usability
