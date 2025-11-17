# Quick Start Guide: Using Resources Patterns in SMLX

## TL;DR - The 3 Most Important Files

Start by studying these files in this order:

1. `/resources/mlx-lm/mlx_lm/models/gemma.py` (180 lines) - Complete, clean model example
2. `/resources/mlx-lm/mlx_lm/models/cache.py` (1000 lines) - KV cache patterns
3. `/resources/mlx-lm/mlx_lm/generate.py` (1100 lines) - Full generation pipeline

## Implementation Checklist for New Models

### Step 1: Create Model Config

```bash
File: smlx/models/YourModel/config.py

# Copy pattern from RESOURCES_REFERENCE_MAP.md "Pattern 1: BaseModelArgs"
# Modify field names to match your model's config.json
```

### Step 2: Create Model Architecture

```bash
File: smlx/models/YourModel/model.py

# Copy from RESOURCES_REFERENCE_MAP.md:
# - Pattern 2: Attention (adapt for your model)
# - Pattern 3: MLP (usually exact copy)
# - Pattern 4: TransformerBlock (copy with your Attention/MLP)
# - Pattern 5: Model (adapt for your architecture)
```

### Step 3: Add Loader

```bash
File: smlx/models/YourModel/loader.py

# Copy pattern from RESOURCES_REFERENCE_MAP.md "Pattern 7: Load Function"
# Modify to load your specific model type
```

### Step 4: Add Generation

```bash
File: smlx/models/YourModel/generate.py

# Copy pattern from RESOURCES_REFERENCE_MAP.md "Pattern 8: Generation Function"
# Import model, cache, sampler utilities
```

### Step 5: Add Cache Support

```bash
File: smlx/models/YourModel/cache.py (optional)

# Usually: import from smlx.models.cache
# Only needed if your model has custom cache requirements
```

### Step 6: Create __init__.py

```bash
File: smlx/models/YourModel/__init__.py

# Export: Model, ModelArgs, load, generate
```

## Copy-Paste Code Blocks

### Block 1: Config (from RESOURCES_REFERENCE_MAP Pattern 1)

```python
from dataclasses import dataclass
import inspect

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() 
                     if k in inspect.signature(cls).parameters})

@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    # ... add your fields
```

### Block 2: Attention (adapt from Pattern 2)

Copy from `/resources/mlx-lm/mlx_lm/models/gemma.py` lines 37-87
Modify: parameter names, projection dimensions, positional encoding

### Block 3: MLP (exact copy from Pattern 3)

This is usually identical across models

### Block 4: TransformerBlock (copy Pattern 4)

Combine your Attention + MLP with residual connections

### Block 5: Model Class (adapt Pattern 5)

Main forward pass, embedding, output projection

### Block 6: Load Function (adapt Pattern 7)

Handle HuggingFace repo loading with your specific model

### Block 7: Generate Function (adapt Pattern 8)

Wrap generation loop with your model

## Where Each Component Comes From

| Component | Source File | Notes |
|-----------|------------|-------|
| BaseModelArgs | Pattern 1 | Copy exactly |
| Attention | Pattern 2 | Adapt parameter names |
| MLP | Pattern 3 | Copy exactly |
| TransformerBlock | Pattern 4 | Copy exactly |
| Model | Pattern 5 | Adapt architecture |
| make_sampler | /mlx-lm/sample_utils.py | Import directly |
| make_prompt_cache | /mlx-lm/models/cache.py | Import directly |
| StreamingDetokenizer | /mlx-lm/tokenizer_utils.py | Import directly |
| load | Pattern 7 | Adapt to your model |
| generate | Pattern 8 | Adapt to your model |

## Quickest Path: Copy Gemma Implementation

Gemma (from mlx-lm) is the cleanest example:

1. Copy `/resources/mlx-lm/mlx_lm/models/gemma.py` -> `smlx/models/YourModel/model.py`
2. Find-replace class names and config field names
3. Save to your model directory
4. Copy loader pattern -> `loader.py`
5. Copy generate pattern -> `generate.py`
6. Create `__init__.py` with exports
7. Done!

## Testing Your Implementation

```bash
# 1. Test config loading
python -c "from smlx.models.YourModel import ModelArgs; print(ModelArgs.from_dict({'hidden_size': 512, 'num_hidden_layers': 12}))"

# 2. Test model instantiation
python -c "from smlx.models.YourModel import Model, ModelArgs; m = Model(ModelArgs(...))"

# 3. Test generation
python -c "from smlx.models.YourModel import load, generate; model, tok = load('repo-id'); print(generate(model, tok, 'Hello'))"
```

## Common Pitfalls

1. __Direct imports from resources/__ - Don't do this! Copy and adapt instead
2. __Forgetting cache parameter__ - Model.__call__ must accept cache
3. __Wrong attention masking__ - Always use cached mask for efficiency
4. __Not supporting quantization__ - Add bits parameter to layer creation
5. __Hardcoding batch size__ - Support arbitrary batch dimensions
6. __Missing __init__.py__ - Must export Model, ModelArgs, load, generate

## File Structure Template

```ascii
smlx/models/YourModel/
├── __init__.py              # Exports
├── model.py                 # Architecture (150-200 lines)
├── config.py                # ModelArgs (50-100 lines)
├── loader.py                # load() function (100-150 lines)
├── generate.py              # generate(), chat() (150-200 lines)
├── cache.py                 # Optional: custom KV cache
└── tokenizer_utils.py       # Optional: custom tokenizer handling
```

## Integration with SMLX Utils

```python
# These are already implemented in smlx/utils:
from smlx.utils import load_model, save_model

# These utilities for models:
from smlx.models.cache import make_prompt_cache, KVCache, RotatingKVCache
from smlx.sample_utils import make_sampler
from smlx.tokenizer_utils import StreamingDetokenizer

# Use directly - no need to reimplement!
```

## Resources Documents

Two comprehensive documents have been created:

1. __RESOURCES_PATTERNS.md__ (24KB)
   - Complete architectural patterns
   - Deep dive into each pattern
   - Explanations of design choices
   - Best practices for small models

2. __RESOURCES_REFERENCE_MAP.md__ (15KB)
   - Quick file locations
   - Copy-paste code blocks
   - Exact line numbers
   - Testing patterns
   - Checklists

## Example: SmolLM2-360M Implementation

To add SmolLM2-360M to SMLX:

1. Look at `/resources/mlx-lm/mlx_lm/models/llama.py` (SmolLM uses Llama architecture)
2. Copy structure to `smlx/models/SmolLM2_360M/`
3. Adjust hidden_size, num_layers for 360M variant
4. Test with existing SmolLM2-135M patterns
5. Add to examples/

## Next Steps

1. Read: `/Users/ryanoboyle/smlx/RESOURCES_PATTERNS.md` (full architectural patterns)
2. Reference: `/Users/ryanoboyle/smlx/RESOURCES_REFERENCE_MAP.md` (copy-paste code)
3. Study: `/resources/mlx-lm/mlx_lm/models/gemma.py` (actual implementation)
4. Implement: Follow checklist above
5. Test: Use test patterns from RESOURCES_REFERENCE_MAP.md

## Key Principles to Remember

1. __Copy, don't import__ - Adapt reference code into SMLX
2. __Reuse, don't reinvent__ - Utilities already exist (cache, sampling, generation)
3. __Quantization first__ - All models must support 4-bit, 8-bit
4. __Small models__ - Keep parameter count under 1B
5. __Streaming output__ - Support incremental text generation
6. __Efficient caching__ - Use KV cache for token-by-token generation
7. __Clean interfaces__ - Export load(), generate(), chat()

## Common Reference Queries

__Q: How do I handle different attention mechanisms?__
A: See Attention patterns in RESOURCES_REFERENCE_MAP.md. Modify qkv projection, attention computation, and bias handling.

__Q: How do I add quantization support?__
A: Models work with MLX's QuantizedLinear. No special code needed - passes through automatically.

__Q: How do I implement streaming?__
A: Use StreamingDetokenizer in generate(). See Pattern 8 in RESOURCES_REFERENCE_MAP.md.

__Q: How do I cache effectively?__
A: Use make_prompt_cache() from models.cache. Standard KVCache for short sequences, RotatingKVCache for long.

__Q: How do I support multi-turn chat?__
A: Build prompt with chat template, maintain cache across turns. See chat.py pattern.
