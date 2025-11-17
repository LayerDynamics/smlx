# Model Conversion Patterns - Quick Reference Index

Quick navigation to conversion patterns and implementations.

## Key Files in Resources

### LLM Conversion (Language Models)

- **Main Converter:** `/resources/mlx-lm/mlx_lm/convert.py` (Lines 84-249)
- **Utilities:** `/resources/mlx-lm/mlx_lm/utils.py` (Lines 503-769)
  - `load_model()` - Line 157
  - `quantize_model()` - Line 563
  - `dequantize_model()` - Line 630
  - `save_model()` - Line 503
  - `save_config()` - Line 676
  - `save()` - Line 702

### VLM Conversion (Vision-Language Models)

- **Main Converter:** `/resources/mlx-vlm/mlx_vlm/convert.py` (Lines 104-256)
- **Utilities:** `/resources/mlx-vlm/mlx_vlm/utils.py`
  - `skip_multimodal_module()` - Line 44
  - `load_model()` - Line 122
  - `fetch_from_hub()` - Line 391
  - `save_weights()` - Line 500
  - `save_config()` - Line 556
  - `make_shards()` - Line 405

### Audio Model Conversions

- **Whisper:** `/resources/mlx-examples/whisper/convert.py`
  - `hf_to_pt()` - Line 112 (config mapping)
  - `convert()` - Line 232 (main conversion)
  - `load_torch_weights_and_config()` - Line 152
  - `quantize()` - Line 298
  - `upload_to_hub()` - Line 256
  
- **Encodec:** `/resources/mlx-examples/encodec/convert.py`
  - `fetch_from_hub()` - Line 13
  - `save_weights()` - Line 31
  - `upload_to_hub()` - Line 31

### Vision Model Conversions

- **CLIP:** `/resources/mlx-examples/clip/convert.py`
  - `torch_to_mx()` - Line 83 (dtype conversion)
  - `save_weights()` - Line 28

- **Segment Anything (SAM):** `/resources/mlx-examples/segment_anything/convert.py`
  - `save_weights()` - Line 11
  - `convert()` - Line 45 (axis reordering)
  - `download()` - Line 35

- **BERT:** `/resources/mlx-examples/bert/convert.py`
  - `replace_key()` - Line 7 (key mapping)
  - `convert()` - Line 22

## Copy-Paste Ready Implementations

### Pattern 1: Generic LLM Conversion Flow

```python
# Location: /resources/mlx-lm/mlx_lm/convert.py:84-169
# Usage pattern for all standard LLMs
from mlx_lm.utils import load, quantize_model, save
```

### Pattern 2: VLM-Specific Conversion (with processor)

```python
# Location: /resources/mlx-vlm/mlx_vlm/convert.py:104-183
# Usage pattern for models with vision components
# Skips quantizing vision modules automatically
```

### Pattern 3: Dtype Conversion Helper

```python
# Location: /resources/mlx-lm/mlx_lm/utils.py:137-137
# Generic torch → MLX array conversion with bfloat16 handling
def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))
```

### Pattern 4: Weight Sharding for Large Models

```python
# Location: /resources/mlx-vlm/mlx_vlm/utils.py:405-426
# Splits weights into 5GB chunks
def make_shards(weights: dict, max_file_size_gb: int = 5) -> list:
    # ... splits weights into safetensors files
```

### Pattern 5: Skip Sensitive Modules

```python
# Location: /resources/mlx-vlm/mlx_vlm/utils.py:44-60
# Prevents quantization of vision/audio encoders
def skip_multimodal_module(path: str) -> bool:
    return "vision_model" in path or "audio_model" in path
```

### Pattern 6: Config Saving with Quantization

```python
# Location: /resources/mlx-vlm/mlx_vlm/utils.py:556-577
# Handles quantization config persistence
config.pop("_name_or_path", None)
if "quantization" in config:
    config["quantization_config"] = config["quantization"]
```

## Model Type Patterns

### How to convert a new LLM

1. Use `/resources/mlx-lm/mlx_lm/convert.py` as base
2. Add weight remapping function if needed (see lines 20-76 for examples)
3. Use `quantize_model()` from utils for quantization support
4. Call `save()` function to output MLX format

### How to convert a new VLM

1. Use `/resources/mlx-vlm/mlx_vlm/convert.py` as base
2. Ensure `skip_multimodal_module()` is used in quantization predicate
3. Load and save processor separately (lines 119-177)
4. Use `fetch_from_hub()` for multi-component loading

### How to convert audio models

1. Start with `/resources/mlx-examples/whisper/convert.py` as reference
2. Implement `hf_to_pt()` for config mapping (Whisper: lines 112-149)
3. Handle tensor transposition for convolutions (Whisper: lines 233-240)
4. Support quantization (Whisper: lines 298-316)

### How to convert embedding models

1. Simple approach: use `/resources/mlx-examples/bert/convert.py`
2. Key remapping with `.replace()` chains (BERT: lines 7-19)
3. Often use numpy.savez instead of safetensors (BERT: line 28)

### How to convert vision models

1. For transformer vision: use `/resources/mlx-examples/clip/convert.py`
2. For CNN-based: transpose conv weights (SAM: lines 56-65)
3. Handle custom weight shapes (SAM: lines 45-65)

## Quantization Recipes

Supported mixed-bit quantization recipes:

- `mixed_2_6` - 2-bit middle layers, 6-bit outer layers
- `mixed_3_4` - 3-bit middle layers, 4-bit outer layers
- `mixed_3_6` - 3-bit middle layers, 6-bit outer layers
- `mixed_4_6` - 4-bit middle layers, 6-bit outer layers

Strategy:

- First 1/8 layers: high bits
- Middle 6/8 layers: low bits (except every 3rd)
- Last 1/8 layers: high bits
- Special handling for v_proj, down_proj, lm_head: high bits

## Saved Output Structure

### Standard safetensors output

```ascii
mlx_model/
├── model.safetensors          # Single file (if <5GB)
├── model-00001-of-00002.safetensors  # Sharded (if >5GB)
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json  # Weight index and metadata
├── config.json                # Model configuration
├── tokenizer.model            # (LLM) Tokenizer
└── preprocessor_config.json   # (VLM) Image processor
```

### Quantization config in config.json

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine"
  },
  "quantization_config": {...}  // duplicate for compatibility
}
```

## Common Issues and Solutions

### Issue: Large model memory overflow

- **Solution:** Use lazy loading (`lazy=True`)
- **Location:** `/resources/mlx-lm/mlx_lm/utils.py:244`

### Issue: bfloat16 dtype conversion fails

- **Solution:** Upcast to float32 temporarily
- **Location:** `/resources/mlx-lm/mlx_lm/utils.py:132`

### Issue: Weight name mismatches

- **Solution:** Implement model-specific key remapping
- **Location:** `/resources/mlx-examples/llms/llama/convert.py:67-125`

### Issue: Vision quantization degrades quality

- **Solution:** Skip quantizing vision modules
- **Location:** `/resources/mlx-vlm/mlx_vlm/utils.py:44-60`

### Issue: Large model saving takes too long

- **Solution:** Shard weights into 5GB chunks
- **Location:** `/resources/mlx-vlm/mlx_vlm/utils.py:405-426`

## Testing Conversion

1. Start with small model variant (tiny, small, base)
2. Verify config.json is created
3. Check safetensors file loads without errors
4. Test with a simple inference script
5. Compare output with original model
6. Scale up to full-size model

## Integration with smlx/

To implement conversions in smlx/:

1. Create `/smlx/tools/convert2mlx.py` with model-specific converters
2. Follow patterns from resources exactly
3. Add quantization support using recipes
4. Test with each model type (LLM, VLM, Audio, etc.)
5. Document in CONVERSION_PATTERNS.md
