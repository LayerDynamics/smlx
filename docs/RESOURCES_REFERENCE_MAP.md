# Resources Reference Map: File Locations and Code Examples

## Quick Reference: Where to Find Each Pattern

### TEXT MODEL PATTERNS

| Pattern | File Location | Size | Key Classes |
|---------|--------------|------|------------|
| Base Model Args | `/mlx-lm/mlx_lm/models/base.py` | 138 lines | `BaseModelArgs`, `scaled_dot_product_attention` |
| Cache Implementation | `/mlx-lm/mlx_lm/models/cache.py` | 1000 lines | `KVCache`, `RotatingKVCache`, `make_prompt_cache` |
| Generation Loop | `/mlx-lm/mlx_lm/generate.py` | 1100+ lines | `generate`, `stream_generate`, `generate_step` |
| Sampling | `/mlx-lm/mlx_lm/sample_utils.py` | 250 lines | `make_sampler`, sampling strategies |
| Tokenizer Utils | `/mlx-lm/mlx_lm/tokenizer_utils.py` | 300 lines | `StreamingDetokenizer`, `TokenizerWrapper` |
| Model Loading | `/mlx-lm/mlx_lm/utils.py` | 500 lines | `load`, `save`, tree utilities |
| Chat Interface | `/mlx-lm/mlx_lm/chat.py` | 200 lines | Chat loop, prompt building |
| Conversion | `/mlx-lm/mlx_lm/convert.py` | 300 lines | HuggingFace -> MLX |

### EXAMPLE MODEL IMPLEMENTATIONS

| Model Type | File Location | Size | Complexity |
|------------|---------------|------|-----------|
| Gemma | `/mlx-lm/mlx_lm/models/gemma.py` | 180 lines | Simple, clean baseline |
| Mixtral | `/mlx-examples/llms/mixtral/mixtral.py` | 250 lines | MoE example |
| Llama | `/mlx-lm/mlx_lm/models/llama.py` | 200 lines | Standard transformer |

### QUANTIZATION PATTERNS

| Technique | File Location | Size | Purpose |
|-----------|--------------|------|---------|
| GPTQ | `/mlx-lm/mlx_lm/quant/gptq.py` | 200 lines | Post-training quantization |
| AWQ | `/mlx-lm/mlx_lm/quant/awq.py` | 400 lines | Activation-aware quantization |
| DWQ | `/mlx-lm/mlx_lm/quant/dwq.py` | 300 lines | Dynamic weight quantization |
| LoRA | `/mlx-lm/mlx_lm/tuner/lora.py` | 200 lines | Parameter-efficient tuning |

### VISION-LANGUAGE PATTERNS

| Pattern | File Location | Size | Complexity |
|---------|--------------|------|-----------|
| SmolVLM | `/mlx-vlm/mlx_vlm/models/smolvlm/smolvlm.py` | 64 lines | Vision-text interleaving |
| VLM Generation | `/mlx-vlm/mlx_vlm/generate.py` | 400 lines | Multimodal generation |
| VLM Prompting | `/mlx-vlm/mlx_vlm/prompt_utils.py` | 500 lines | Image/text prompt handling |

### AUDIO PATTERNS

| Pattern | File Location | Size | Purpose |
|---------|--------------|------|---------|
| Whisper Model | `/lightning-whisper-mlx/lightning_whisper_mlx/whisper.py` | 400 lines | Encoder-decoder |
| Audio Processing | `/lightning-whisper-mlx/lightning_whisper_mlx/audio.py` | 150 lines | Mel spectrogram |
| Decoding | `/lightning-whisper-mlx/lightning_whisper_mlx/decoding.py` | 500 lines | Beam search |

## Most Important Files to Study (In Order)

### Priority 1: Essential Core Patterns

1. `/mlx-lm/mlx_lm/models/base.py` - Foundation for all models
2. `/mlx-lm/mlx_lm/models/gemma.py` - Clean model example
3. `/mlx-lm/mlx_lm/models/cache.py` - KV cache patterns
4. `/mlx-lm/mlx_lm/sample_utils.py` - Sampling pipeline

### Priority 2: Integration Patterns

5. `/mlx-lm/mlx_lm/generate.py` - Full generation loop
6. `/mlx-lm/mlx_lm/utils.py` - Loading/saving utilities
7. `/mlx-lm/mlx_lm/tokenizer_utils.py` - Tokenizer handling
8. `/mlx-lm/mlx_lm/chat.py` - Chat interface

### Priority 3: Advanced Features

9. `/mlx-lm/mlx_lm/quant/gptq.py` - Quantization
10. `/mlx-lm/mlx_lm/tuner/lora.py` - Fine-tuning
11. `/mlx-vlm/mlx_vlm/models/smolvlm/smolvlm.py` - Vision integration

## Exact Code Pattern Examples

### Pattern 1: BaseModelArgs (Copy This Pattern!)

Location: `/mlx-lm/mlx_lm/models/base.py:11-21`

```python
from dataclasses import dataclass
from typing import Optional
import inspect

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
```

**Use for:** All model configs in SMLX

### Pattern 2: Attention Module (Adapt This Pattern!)

Location: `/mlx-lm/mlx_lm/models/gemma.py:37-87`

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        
        self.scale = head_dim**-0.5
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)
```

**Use for:** All attention layers in SMLX

### Pattern 3: MLP Module (Adapt This Pattern!)

Location: `/mlx-lm/mlx_lm/models/gemma.py:90-98`

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
    
    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))
```

**Use for:** Feed-forward layers in all transformers

### Pattern 4: TransformerBlock (Copy This Pattern!)

Location: Infer from examples, generally:

```python
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.attention = Attention(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Any]:
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache
```

**Use for:** All transformer blocks

### Pattern 5: Model Class (Adapt This Pattern!)

Location: `/mlx-lm/mlx_lm/models/gemma.py:145+`

```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_layers = args.num_hidden_layers
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    def __call__(
        self, 
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> Tuple[mx.array, List[Any]]:
        h = self.embed_tokens(input_ids)
        
        if cache is None:
            cache = [None] * len(self.layers)
        
        new_cache = []
        for layer, cache_entry in zip(self.layers, cache):
            h, new_cache_entry = layer(h, cache=cache_entry)
            new_cache.append(new_cache_entry)
        
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, new_cache
```

**Use for:** Core model class for all text models

### Pattern 6: Make Sampler (Copy This Pattern!)

Location: `/mlx-lm/mlx_lm/sample_utils.py:10-69`

```python
def make_sampler(
    temp: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
) -> Callable[[mx.array], mx.array]:
    """Make a sampler function for use with generate_step."""
    
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)
    
    sampling_methods = []
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        sampling_methods.append(
            lambda x: apply_xtc(x, xtc_probability, xtc_threshold, xtc_special_tokens)
        )
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))
    
    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        return categorical_sampling(logprobs, temp)
    
    return sampler
```

**Use for:** Sampling pipeline in all generation functions

### Pattern 7: Load Function (Adapt This Pattern!)

Location: `/mlx-lm/mlx_lm/utils.py:200+`

```python
def load(
    model_path: str,
    adapter_path: Optional[str] = None,
    tokenizer_config: Optional[dict] = None,
    dtype: mx.Dtype = mx.float16,
    quantize: bool = False,
):
    """Load model and tokenizer from directory or HuggingFace repo."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    
    # Download model
    model_path = snapshot_download(
        model_path,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )
    
    # Load config
    with open(f"{model_path}/config.json") as f:
        config = json.load(f)
    
    # Get model classes
    Model, ModelArgs = _get_classes(config)
    
    # Create model
    model = Model(ModelArgs.from_dict(config))
    
    # Load weights
    weights = mx.load(f"{model_path}/model.safetensors")
    if quantize:
        model = quantize_model(model)
    
    model.update_state(weights)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load adapter if provided
    if adapter_path:
        adapter_weights = mx.load(f"{adapter_path}/adapters.safetensors")
        model.update_state(adapter_weights)
    
    return model, tokenizer
```

**Use for:** All model loading in SMLX

### Pattern 8: Generation Function (Adapt This Pattern!)

Location: `/mlx-lm/mlx_lm/generate.py:200+`

```python
def generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, List[int]],
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    stop: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """Generate text with streaming output."""
    
    # Tokenize input
    if isinstance(prompt, str):
        prompt = tokenizer.encode(prompt)
    
    prompt_len = len(prompt)
    
    # Create cache
    cache = make_prompt_cache(model)
    
    # Create sampler
    sampler = make_sampler(
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    
    # Create detokenizer for streaming
    detokenizer = StreamingDetokenizer(tokenizer)
    detokenizer.reset()
    
    for i in range(max_tokens):
        # Forward pass
        logits, cache = model(mx.array(prompt[-1:]), cache=cache)
        
        # Apply processors
        logits = logits[0, -1, :]
        
        # Sample next token
        next_token = sampler(logits)
        prompt.append(next_token.item())
        
        # Detokenize
        detokenizer.add_token(next_token.item())
        
        # Yield incremental text
        new_text = detokenizer.last_segment
        if new_text:
            yield new_text
        
        # Check stopping conditions
        if next_token.item() == tokenizer.eos_token_id:
            break
        if stop and any(stop_word in detokenizer.text for stop_word in stop):
            break
    
    # Finalize
    detokenizer.finalize()
```

**Use for:** Text generation in all models

## Copy-Paste Checklist

When implementing a new model in SMLX:

- [ ] Copy `BaseModelArgs` pattern for config
- [ ] Adapt `Attention` for your model's attention style
- [ ] Copy `MLP` (usually identical)
- [ ] Copy `TransformerBlock` with your attention/mlp
- [ ] Adapt `Model` class for your architecture
- [ ] Implement `load()` function using pattern
- [ ] Use `make_sampler()` for generation
- [ ] Integrate `make_prompt_cache()` for KV cache
- [ ] Add streaming support with `StreamingDetokenizer`
- [ ] Test with quantization support

## File Dependencies

```ascii
generate.py
├── Imports: sample_utils, tokenizer_utils, models.cache, utils
├── Uses: make_sampler, StreamingDetokenizer, make_prompt_cache
└── Requires: Model class with __call__(input_ids, cache)

models/gemma.py (or your model)
├── Imports: models.base (for scaled_dot_product_attention, create_attention_mask)
├── Uses: Attention, MLP, TransformerBlock
└── Defines: Model, ModelArgs

sample_utils.py
├── Standalone (no internal imports)
└── Used by: generate.py, chat.py

tokenizer_utils.py
├── Imports: transformers library
└── Used by: generate.py, utils.py
```

## Testing Patterns

Each new model should have tests for:

1. **Config Loading**: `ModelArgs.from_dict(config_dict)`
2. **Model Instantiation**: `model = Model(args)`
3. **Forward Pass**: `logits, cache = model(input_ids, cache=cache)`
4. **Generation**: `output = generate(model, tokenizer, prompt)`
5. **Quantization**: Model with `bits=4` parameter
6. **Cache**: Multi-token generation with cache reuse
7. **Chat**: Multi-turn conversation interface

See: `/mlx-lm/tests/` for reference test patterns
