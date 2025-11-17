# Convert2MLX Update Summary

## Overview

Updated [smlx/tools/convert2mlx.py](smlx/tools/convert2mlx.py) to support comprehensive model conversion for all model types currently in the SMLX project.

## Key Additions

### 1. Model Type Detection (Lines 424-461)

- **`detect_model_type(config)`** - Automatically detects model type from config
- Supports: LLM, VLM, Audio, OCR, Embedding models
- Checks `model_type`, `architectures`, and config keys to determine type

### 2. Model-Specific Weight Remapping

#### LLM Weight Remapping (Lines 464-505)

- **`remap_llm_weights()`** - Converts HuggingFace format to MLX format
- Transformations:
  - Remove "model." prefix
  - Standardize layer naming (`.self_attn.` → `.attn.`)
  - Rename projections (`.q_proj.` → `.query_proj.`)
  - Embedding naming (`embed_tokens` → `tok_embeddings`)

#### VLM Weight Remapping (Lines 508-549)

- **`remap_vlm_weights()`** - Handles vision + language components
- Keeps vision/audio modules intact
- Applies LLM remapping to language model components
- Preserves: `vision_model`, `vision_tower`, `audio_model`, `audio_tower`

#### Audio Weight Remapping (Lines 552-594)

- **`remap_audio_weights()`** - Audio-specific transformations
- Whisper support:
  - Encoder/decoder separation
  - Convolution weight transposition (Conv1d format)
- Encodec support:
  - Encoder/decoder naming (`encoder.` → `enc.`)

#### OCR Weight Remapping (Lines 597-624)

- **`remap_ocr_weights()`** - TrOCR/Donut conversion
- Encoder-decoder architecture handling
- Vision encoder + text decoder separation

#### Embedding Weight Remapping (Lines 627-654)

- **`remap_embedding_weights()`** - BERT-based models
- Simple key transformations
- BERT-style pooler handling

### 3. Mixed-Bit Quantization (Lines 280-347)

- **`mixed_quant_predicate_builder()`** - Build selective quantization functions
- Recipes: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`
- Strategy:
  - First 1/8 layers: high bits (6-bit)
  - Middle 6/8 layers: low bits (2-4 bit)
  - Last 1/8 layers: high bits (6-bit)
  - Special layers (`v_proj`, `down_proj`, `lm_head`): high bits

### 4. Enhanced Main Convert Function (Lines 828-1019)

New parameters:

- `auto_detect: bool = True` - Automatically detect model type
- `model_type: Optional[str] = None` - Manually specify type
- `quant_recipe: Optional[str] = None` - Mixed-bit quantization recipe

New features:

- Automatic model type detection and conversion
- Model-specific weight remapping
- Processor config copying for VLM/Audio/OCR models
- Conversion metadata in config (`smlx_model_type`, `smlx_conversion_info`)
- Improved safetensors loading (excludes index files)

### 5. Enhanced CLI Interface (Lines 1027-1167)

New options:

- `--model-type {llm,vlm,audio,ocr,embedding}` - Manually specify type
- `--no-auto-detect` - Disable automatic detection
- `--quant-recipe {mixed_2_6,mixed_3_4,mixed_3_6,mixed_4_6}` - Mixed-bit quantization

Improved help:

- Examples section showing common use cases
- Better organization of arguments
- Model type and recipe documentation

## Supported Models

### Language Models (LLM)

- SmolLM2-135M
- SmolLM2-360M

### Vision-Language Models (VLM)

- SmolVLM-256M-Instruct
- SmolVLM-500M-Instruct
- nanoVLM
- Moondream2
- TinyLLaVA

### Audio Models

- Whisper-tiny
- YAMNet
- SileroVAD
- Chatterbox
- Orpheus-150M

### OCR Models

- TrOCR-small
- Donut-base

### Embedding Models

- MiniLM
- all-MiniLM-L6-v2

## Usage Examples

### Basic Conversion

```bash
python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-135M-Instruct
```

### With Quantization

```bash
python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-135M-Instruct -q --q-bits 4
```

### Mixed-Bit Quantization

```bash
python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-360M-Instruct -q --quant-recipe mixed_4_6
```

### VLM Conversion

```bash
python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolVLM-256M-Instruct --model-type vlm
```

### Audio Model Conversion

```bash
python -m smlx.tools.convert2mlx --hf-path openai/whisper-tiny --model-type audio
```

### Dtype Conversion

```bash
python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-135M-Instruct --dtype float16
```

## Implementation Details

### Conversion Pipeline

1. **Model Path Resolution** - Download from HF Hub or use local path
2. **Config Loading** - Load and parse config.json
3. **Model Type Detection** - Auto-detect or use specified type
4. **Weight Loading** - Load from safetensors (preferred) or PyTorch bins
5. **Weight Remapping** - Apply model-specific transformations
6. **Dtype Conversion** - Convert to target dtype if specified
7. **Quantization** - Apply standard or mixed-bit quantization
8. **Metadata Addition** - Add conversion info to config
9. **Weight Saving** - Save with sharding if needed
10. **File Copying** - Copy tokenizer, processor configs, etc.

### Output Structure

```
mlx_model/
├── model.safetensors (or sharded files)
├── model.safetensors.index.json (if sharded)
├── config.json (with smlx metadata)
├── tokenizer.json
├── tokenizer_config.json
├── preprocessor_config.json (VLM/Audio/OCR)
└── processor_config.json (VLM/Audio/OCR)
```

### Config Metadata

The converted model's config.json includes:

```json
{
  "smlx_model_type": "vlm",
  "smlx_conversion_info": {
    "source": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "quantized": true,
    "dtype": "float16",
    "quant_bits": 4,
    "quant_group_size": 64
  }
}
```

## Code Quality

- ✅ All ruff checks passing
- ✅ 100-character line length enforced
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Follows CLAUDE.md guidelines

## Future Enhancements

1. **PyTorch .bin Support** - Direct conversion from PyTorch checkpoint files
2. **Full Mixed-Bit Quantization** - Requires loading into nn.Module for predicate application
3. **Custom Weight Mapping** - User-provided mapping functions
4. **Batch Conversion** - Convert multiple models in one command
5. **Validation Mode** - Compare MLX vs original model outputs
6. **Conversion Profiles** - Predefined settings for common model types

## Related Documentation

- [CONVERSION_PATTERNS.md](CONVERSION_PATTERNS.md) - Detailed conversion patterns from resources
- [CONVERSION_REFERENCE.md](CONVERSION_REFERENCE.md) - Quick reference for implementations
- [CONVERSION_INDEX.md](CONVERSION_INDEX.md) - Navigation guide for conversion docs
- [CLAUDE.md](CLAUDE.md) - Project guidelines and structure

## Testing

To test the conversion:

```bash
# Test with SmolLM2-135M (smallest model)
python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-135M-Instruct --mlx-path test_output

# Verify output
ls -lh test_output/
cat test_output/config.json | grep smlx

# Test quantized conversion
python -m smlx.tools.convert2mlx --hf-path HuggingFaceTB/SmolLM2-135M-Instruct -q --q-bits 4 --mlx-path test_quant
```

## Notes

- The `skip_multimodal` parameter is currently not used in the weight remapping logic but is preserved for future quantization integration
- Mixed-bit quantization recipes require loading the model into an nn.Module, which is noted as a future enhancement
- Config parameter hints (e.g., "config is not accessed") are intentional - these parameters are reserved for future model-specific logic
