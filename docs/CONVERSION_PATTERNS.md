# MLX Model Conversion Patterns - Comprehensive Summary

## Overview

The resources directory contains multiple conversion scripts and patterns for converting models from Hugging Face format to MLX format. These patterns are organized by model type and provide a blueprint for implementing model-to-MLX conversions.

---

## 1. GENERAL CONVERSION ARCHITECTURE

### Core Pipeline Components

All conversions follow this basic pattern:

```ascii
1. Download/Load Model
   ├── From Hugging Face Hub (snapshot_download)
   └── From local path

2. Load Weights
   ├── Handle PyTorch → MLX conversion
   ├── Apply weight remapping (key renaming)
   └── Handle dtype conversions

3. Optional Processing
   ├── Quantization (4-bit, 8-bit, mixed-bit)
   ├── Dequantization
   └── Dtype casting

4. Save Output
   ├── Save weights as safetensors
   ├── Create sharded files (if >5GB)
   ├── Save config.json
   └── Upload to HuggingFace Hub (optional)
```

### Key Utilities

**File locations:**

- `/resources/mlx-lm/mlx_lm/utils.py` - LLM conversion utilities
- `/resources/mlx-vlm/mlx_vlm/utils.py` - VLM conversion utilities
- `/resources/mlx-examples/*/convert.py` - Model-specific converters

**Common utility functions:**

- `_download()` / `get_model_path()` - Download models from HF Hub
- `load_config()` - Load config.json
- `load_model()` - Initialize MLX model architecture
- `quantize_model()` - Apply quantization
- `dequantize_model()` - Reverse quantization
- `save_model()` - Save weights to safetensors
- `save_config()` - Save config with metadata

---

## 2. LANGUAGE MODEL (LLM) CONVERSIONS

### Files

- **Main converter:** `/resources/mlx-lm/mlx_lm/convert.py`
- **Utilities:** `/resources/mlx-lm/mlx_lm/utils.py`
- **Examples:** `/resources/mlx-examples/llms/{llama,mistral,mixtral}/convert.py`

### Patterns

#### A. Weight Remapping

Different model architectures use different naming conventions. Conversion requires mapping:

```python
# Example: LLaMA weights mapping
def llama(model_path, *, dtype: str):
    # Load sharded PyTorch checkpoints
    torch_files = glob.glob(str(model_path / "consolidated.*.pth"))
    
    # Combine sharded weights
    weights = collections.defaultdict(list)
    for wf in torch_files:
        state = torch.load(wf, map_location=torch.device("cpu"))
        for k, v in state.items():
            v = torch_to_mx(v, dtype=dtype)  # Convert torch tensor to MLX array
            weights[k].append(v)
    
    # Unshard by concatenating along specific axes
    for k, v in weights.items():
        weights[k] = unshard(k, v)
    
    return weights, params
```

#### B. Key Remapping Strategy

```python
# Model-specific key transformations
# Example: HuggingFace → Original format
k = k.replace("model.", "")  # Remove model prefix
k = k.replace(".layers", ".blocks")  # Layer naming
k = k.replace(".self_attn", ".attn")  # Attention naming
k = k.replace(".q_proj", ".query")  # Query projection
k = k.replace(".k_proj", ".key")  # Key projection
k = k.replace(".v_proj", ".value")  # Value projection
k = k.replace("embed_tokens", "tok_embeddings")  # Embedding naming
```

#### C. Data Type Handling

```python
def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    # bfloat16 needs special handling
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))
```

#### D. Weight Saving with Sharding

```python
def save_model(save_path, model):
    weights = dict(tree_flatten(model.parameters()))
    shards = make_shards(weights)  # Split into 5GB chunks
    
    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(save_path / shard_name), shard, 
                          metadata={"format": "mlx"})
```

### Quantization Support

```python
# Mixed-bit quantization predicates
QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]

# Selective quantization by layer
# - First and last layers: higher bits (6-bit)
# - Middle layers: lower bits (2-4 bit)
# - Specific layers (v_proj, down_proj, lm_head): higher precision
```

### Command Line Interface

```bash
# Basic conversion
mlx_lm.convert --hf-path <model> --mlx-path <output>

# With quantization
mlx_lm.convert --hf-path <model> --mlx-path <output> -q --q-bits 4 --q-group-size 64

# Mixed-bit quantization
mlx_lm.convert --hf-path <model> --mlx-path <output> -q --quant-predicate mixed_4_6

# Dtype conversion
mlx_lm.convert --hf-path <model> --mlx-path <output> --dtype float16

# Dequantization
mlx_lm.convert --hf-path <model> --mlx-path <output> -d
```

---

## 3. VISION-LANGUAGE MODEL (VLM) CONVERSIONS

### Files

- **Main converter:** `/resources/mlx-vlm/mlx_vlm/convert.py`
- **Utilities:** `/resources/mlx-vlm/mlx_vlm/utils.py`
- **Models:** `/resources/mlx-vlm/mlx_vlm/models/*/`

### Key Patterns

#### A. Multimodal Module Handling

VLMs have vision and language components - special handling for each:

```python
def skip_multimodal_module(path: str) -> bool:
    """Skip quantizing vision/audio modules (preserve precision)"""
    return (
        "vision_model" in path
        or "vision_tower" in path
        or "sam_model" in path
        or "audio_model" in path
        or "audio_tower" in path
    )
```

**Reason:** Vision encoders are more sensitive to quantization; language models tolerate lower precision better.

#### B. Processor Handling

VLMs include image/audio processors not present in pure LLMs:

```python
# Load processor (handles image preprocessing)
processor = load_processor(
    model_path,
    add_detokenizer=False,
    eos_token_ids=config.get("eos_token_id", None),
)

# Save processor alongside model
processor.save_pretrained(mlx_path)
```

#### C. Multi-Component Loading

```python
def fetch_from_hub(model_path, lazy=False):
    model = load_model(model_path, lazy)  # Model architecture
    config = load_config(model_path)       # Configuration
    processor = load_processor(model_path) # Image/Audio processor
    
    return model, config, processor
```

#### D. Configuration Handling

```python
# Initialize sub-configs for each component
config.setdefault("text_config", {})
config.setdefault("vision_config", {})
config.setdefault("audio_config", {})
config.setdefault("perceiver_config", {})
config.setdefault("projector_config", {})

# Update module-specific configs
model_config = update_module_configs(model_config, model_class, config, modules)
```

### Conversion Command

```bash
# VLM conversion
mlx_vlm.convert --hf-path <vision-model> --mlx-path <output>

# With quantization (skips vision components)
mlx_vlm.convert --hf-path <vision-model> --mlx-path <output> -q --q-bits 4
```

---

## 4. AUDIO MODEL CONVERSIONS

### Files

- **Whisper:** `/resources/mlx-examples/whisper/convert.py`
- **Encodec:** `/resources/mlx-examples/encodec/convert.py`

### Whisper-Specific Patterns

#### A. Dual-Path Architecture

Whisper has encoder-decoder architecture requiring special mapping:

```python
def hf_to_pt(weights, config):
    """Map HuggingFace model names to original Whisper format"""
    config = {
        "n_mels": config["num_mel_bins"],
        "n_audio_ctx": config["max_source_positions"],
        "n_audio_state": config["d_model"],
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_vocab": config["vocab_size"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_state": config["d_model"],
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }
    return config
```

#### B. Key Remapping

```python
def remap(k):
    k = k.replace("model.", "")
    k = k.replace(".layers", ".blocks")
    k = k.replace(".self_attn", ".attn")
    k = k.replace(".encoder_attn", ".cross_attn")
    k = k.replace(".fc1", ".mlp1")
    k = k.replace(".fc2", ".mlp2")
    k = k.replace("embed_positions.weight", "positional_embedding")
    k = k.replace("decoder.embed_tokens", "decoder.token_embedding")
    return k
```

#### C. Convolution Weight Transposition

Audio models use different Conv2D layouts:

```python
def remap(key, value):
    key = key.replace("mlp.0", "mlp1")
    key = key.replace("mlp.2", "mlp2")
    
    # PyTorch Conv2D: (out_channels, in_channels, height, width)
    # MLX Conv2D: (out_channels, in_channels, height, width)
    # But 1D convolutions may need axis swapping
    if "conv" in key and value.ndim == 3:
        value = value.swapaxes(1, 2)
    
    if isinstance(value, torch.Tensor):
        value = mx.array(value.detach())
    return key, value.astype(dtype)
```

#### D. Alignment Heads

Whisper has special alignment heads for audio-text synchronization:

```python
# Alignment heads are base85-encoded and model-specific
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    # ... per model
}

# Applied after model load
if alignment_heads is not None:
    model.set_alignment_heads(alignment_heads)
```

#### E. Quantization

```python
def quantize(weights, config, args):
    model = Whisper(ModelDimensions(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))
    
    # Quantize the model
    nn.quantize(model, args.q_group_size, args.q_bits)
    
    # Update config
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
```

### Whisper Conversion Command

```bash
# Basic Whisper conversion
python convert.py --torch-name-or-path tiny --mlx-path mlx_models

# From HuggingFace
python convert.py --torch-name-or-path openai/whisper-tiny --mlx-path mlx_models

# With quantization
python convert.py --torch-name-or-path tiny --mlx-path mlx_models -q --q-bits 4
```

---

## 5. EMBEDDING MODEL CONVERSIONS

### Files

- **BERT Example:** `/resources/mlx-examples/bert/convert.py`

### Pattern

```python
def replace_key(key: str) -> str:
    """Map BERT layer names"""
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    return key

def convert(bert_model: str, mlx_model: str) -> None:
    model = AutoModel.from_pretrained(bert_model)
    tensors = {
        replace_key(key): tensor.numpy() 
        for key, tensor in model.state_dict().items()
    }
    numpy.savez(mlx_model, **tensors)
```

**Note:** Embeddings often use simpler conversion (direct numpy.savez vs safetensors)

---

## 6. SPECIALIZED MODEL CONVERSIONS

### A. Vision Models (CLIP, SAM)

#### CLIP Pattern

```python
def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))

# In conversion:
model = AutoModel.from_pretrained(model_path)
weights = {k: torch_to_mx(v, dtype=dtype) for k, v in model.state_dict().items()}
```

#### Segment Anything (SAM) Pattern

```python
# Transpose weights for conv dimensions
mlx_weights = dict()
for k, v in weights.items():
    if k in CONV_WEIGHT_KEYS:
        # (C_out, H, W, C_in) - reorder channels last
        v = v.transpose(0, 2, 3, 1)
    mlx_weights[k] = v
```

**Key insight:** Vision models often need axis reordering for convolution compatibility.

### B. Audio/Compression Models (Encodec)

```python
def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    return model_path

# Load and convert
weights = mx.load(str(model_path / "model.safetensors"))
# Apply any model-specific transformations
```

---

## 7. WEIGHT SAVING PATTERNS

### A. Single File (Small Models)

```python
def save_weights(save_path, weights):
    save_path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(save_path / "model.safetensors"), weights)
    
    # Create index
    index_data = {
        "metadata": {"total_size": sum(v.nbytes for v in weights.values())},
        "weight_map": {k: "model.safetensors" for k in weights.keys()}
    }
    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)
```

### B. Sharded Files (Large Models)

```python
def make_shards(weights: dict, max_file_size_gb: int = 5) -> list:
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    
    shards.append(shard)
    return shards

# Save each shard
for i, shard in enumerate(shards):
    shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
    mx.save_safetensors(str(save_path / shard_name), shard, 
                       metadata={"format": "mlx"})
```

---

## 8. CONFIGURATION HANDLING

### A. Config Loading

```python
def load_config(model_path: Path) -> dict:
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    return config
```

### B. Config Saving

```python
def save_config(config: dict, config_path):
    # Clean unused keys
    config.pop("_name_or_path", None)
    config.pop("torch_dtype", None)
    config.pop("vision_config", None)  # VLM-specific
    
    # Add quantization info if present
    if "quantization" in config:
        config["quantization_config"] = config["quantization"]
    
    # Sort for readability
    config = dict(sorted(config.items()))
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
```

### C. Quantization Config

```python
# Updated during quantization
quantized_config["quantization"] = {
    "group_size": 64,
    "bits": 4,
    "mode": "affine"
}
```

---

## 9. COMMON PITFALLS & SOLUTIONS

### Issue: Memory with Large Models

```python
# Solution: Lazy loading
weights = tree_map(mx.array, weights)  # Don't force evaluation
model.update(tree_unflatten(list(weights.items())))

# Solution: Donate pattern
model.update(tree_map(lambda _: mx.array([]), model.parameters()))
```

### Issue: Dtype Conversion

```python
# PyTorch bfloat16 issue
if dtype == "bfloat16":
    a = a.to(torch.float32)  # Upcast temporarily
return mx.array(a.numpy(), getattr(mx, dtype))
```

### Issue: Weight Shape Mismatches

```python
# Solution: Weight remapping function
def remap_weights(weights, mappings):
    for old_key, new_key in mappings.items():
        if old_key in weights:
            weights[new_key] = weights.pop(old_key)
    return weights
```

### Issue: Quantization Sensitivity

```python
# Skip sensitive modules
def skip_multimodal_module(path: str) -> bool:
    return "vision_model" in path or "audio_model" in path

# Use higher precision for critical layers
if "lm_head" in path or "embed_tokens" in path:
    return {"bits": 6, "group_size": 64}
```

---

## 10. QUICK REFERENCE TABLE

| Model Type | Converter | Key Features | Output Format |
|-----------|-----------|--------------|----------------|
| **LLM** | mlx_lm.convert | Weight remapping, Mixed-bit quant | Sharded safetensors + config |
| **VLM** | mlx_vlm.convert | Skip vision quant, Processor save | Sharded safetensors + processor |
| **Whisper** | whisper/convert.py | Alignment heads, Conv transposition | Safetensors + config |
| **CLIP** | clip/convert.py | Torch→MLX conversion | Sharded safetensors |
| **SAM** | segment_anything/convert.py | Axis reordering | Safetensors |
| **BERT** | bert/convert.py | Simple key mapping | NPZ format |
| **Encodec** | encodec/convert.py | Audio codec weights | Safetensors |

---

## 11. IMPLEMENTATION CHECKLIST FOR NEW MODEL CONVERSIONS

When implementing conversions for a new model type:

- [ ] Identify model architecture (LLM, VLM, Audio, etc.)
- [ ] Create key remapping dictionary for weight names
- [ ] Handle dtype conversions (especially bfloat16)
- [ ] Implement weight loading from HuggingFace/local
- [ ] Add quantization support (or skip for sensitive layers)
- [ ] Implement config loading and saving
- [ ] Handle special components (vision encoders, processors, etc.)
- [ ] Add weight sharding for models >5GB
- [ ] Test with small model first (tiny/base variant)
- [ ] Document command-line interface
- [ ] Add metadata to safetensors files
- [ ] Support optional HuggingFace Hub upload
