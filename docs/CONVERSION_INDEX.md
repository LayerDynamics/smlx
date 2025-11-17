# Model Conversion Patterns - Documentation Index

This documentation provides a comprehensive exploration of MLX model conversion patterns found in the resources directory.

## Documentation Files

### 1. CONVERSION_PATTERNS.md (Main Reference - 605 lines)

**Comprehensive guide to all conversion patterns by model type**

- Sections 1-2: General architecture and LLM conversions
- Section 3: Vision-Language Model (VLM) conversions
- Section 4: Audio model conversions (Whisper, Encodec)
- Section 5: Embedding model conversions
- Section 6: Specialized conversions (CLIP, SAM, BERT)
- Section 7: Weight saving and sharding patterns
- Section 8: Configuration handling
- Section 9: Common pitfalls and solutions
- Section 10-11: Quick reference table and implementation checklist

**Best for:** Understanding complete patterns, learning architecture, solving specific problems

### 2. CONVERSION_REFERENCE.md (Quick Navigation - 214 lines)

**Fast lookup guide with file locations and line numbers**

- Key files organized by model type with exact line references
- Copy-paste ready code snippets for common patterns
- Model-specific conversion step-by-step guides
- Quantization recipes and strategies
- Output file structure examples
- Common issues with solution locations

**Best for:** Quick reference, finding specific code, implementing new conversions

## Quick Navigation by Model Type

### Language Models (LLM)

- **Files:** `/resources/mlx-lm/mlx_lm/convert.py`, `utils.py`
- **Key patterns:** Weight remapping, mixed-bit quantization, weight sharding
- **Documentation:** CONVERSION_PATTERNS.md Section 2, CONVERSION_REFERENCE.md "LLM Conversion"
- **Implementation:** Copy pattern from mlx-lm/convert.py lines 84-169

### Vision-Language Models (VLM)

- **Files:** `/resources/mlx-vlm/mlx_vlm/convert.py`, `utils.py`
- **Key patterns:** Skip multimodal quantization, processor handling, component configs
- **Documentation:** CONVERSION_PATTERNS.md Section 3, CONVERSION_REFERENCE.md "VLM Conversion"
- **Implementation:** Use mlx_vlm.convert patterns, ensure skip_multimodal_module() is active

### Audio Models (Whisper, Encodec)

- **Files:** `/resources/mlx-examples/whisper/convert.py`, `/resources/mlx-examples/encodec/convert.py`
- **Key patterns:** Encoder-decoder mapping, convolution transposition, alignment heads
- **Documentation:** CONVERSION_PATTERNS.md Section 4, CONVERSION_REFERENCE.md "Audio Model Conversions"
- **Implementation:** Start with whisper/convert.py as reference

### Embedding Models (BERT, MiniLM)

- **Files:** `/resources/mlx-examples/bert/convert.py`
- **Key patterns:** Simple key remapping, numpy.savez output
- **Documentation:** CONVERSION_PATTERNS.md Section 5
- **Implementation:** Use BERT pattern with straightforward key replacements

### Vision Models (CLIP, SAM)

- **Files:** `/resources/mlx-examples/clip/convert.py`, `/resources/mlx-examples/segment_anything/convert.py`
- **Key patterns:** Torch→MLX conversion, axis reordering for convolutions
- **Documentation:** CONVERSION_PATTERNS.md Section 6
- **Implementation:** CLIP for transformers, SAM for CNN-based with transpose logic

## Key Concepts Explained

### Weight Remapping

Maps model weight names from one format to another (e.g., HuggingFace to original format)

- **LLM Example:** `.layers` → `.blocks`, `.q_proj` → `.query`
- **Audio Example:** `.fc1` → `.mlp1`, `.fc2` → `.mlp2`
- **See:** CONVERSION_PATTERNS.md Section 2.A, CONVERSION_REFERENCE.md "Weight Remapping"

### Dtype Handling

Converts tensor data types with special handling for bfloat16

- **Pattern:** `torch_to_mx()` function with upcast to float32 intermediate
- **See:** CONVERSION_PATTERNS.md Section 2.C, CONVERSION_REFERENCE.md Pattern 3

### Weight Sharding

Splits large models into 5GB chunks for storage efficiency

- **Pattern:** `make_shards()` function, creates index.json
- **Applies to:** Models >5GB
- **See:** CONVERSION_PATTERNS.md Section 7.B, CONVERSION_REFERENCE.md Pattern 4

### Quantization Support

Reduces model size with optional mixed-bit recipes

- **Recipes:** mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6
- **Strategy:** Higher bits for first/last layers, lower for middle
- **VLM Special:** Skip vision modules, quantize language only
- **See:** CONVERSION_PATTERNS.md Sections 2, 3, 4

### Multimodal Handling

Special treatment for models with multiple components (vision + language, audio + text)

- **Pattern:** `skip_multimodal_module()` predicate
- **Use case:** Preserve precision in vision/audio encoders
- **See:** CONVERSION_PATTERNS.md Section 3.A, CONVERSION_REFERENCE.md Pattern 5

## Code Pattern Summary

### Generic Flow for Any Model

```
1. Download model from HuggingFace Hub
2. Load weights and config
3. Optionally apply weight remapping
4. Handle dtype conversions
5. Optionally quantize
6. Save weights as sharded safetensors
7. Save config.json with metadata
8. Optionally save processor/tokenizer
9. Upload to HuggingFace Hub (optional)
```

### Key Functions Used Across All Models

- `snapshot_download()` - Download from HuggingFace Hub
- `torch_to_mx()` - Convert torch tensors to MLX arrays
- `quantize_model()` - Apply quantization with predicates
- `make_shards()` - Split weights into manageable files
- `mx.save_safetensors()` - Save weights in MLX format
- `save_config()` - Persist model configuration

## File Structure Reference

### Conversion Input

```
HuggingFace Hub or local directory:
├── config.json          (model configuration)
├── model.safetensors    (or pytorch_model.bin)
├── tokenizer.model      (for LLMs)
└── preprocessor_config.json (for VLMs)
```

### Conversion Output

```
MLX Model Directory:
├── model.safetensors              (single file if <5GB)
├── model-00001-of-00002.safetensors (sharded if >5GB)
├── model.safetensors.index.json   (weight index)
├── config.json                    (with quantization metadata)
├── tokenizer.model                (if LLM)
└── preprocessor_config.json       (if VLM)
```

## Common Implementation Scenarios

### Scenario 1: Convert a new LLM variant (e.g., TinyLlama)

1. Reference: CONVERSION_PATTERNS.md Section 2
2. Copy: `/resources/mlx-lm/mlx_lm/convert.py` lines 84-169
3. Key step: Add key remapping if needed (compare weight names)
4. Use: `mlx_lm.convert --hf-path <model> --mlx-path <output>`

### Scenario 2: Convert a VLM with custom components (e.g., LLaVA variant)

1. Reference: CONVERSION_PATTERNS.md Section 3
2. Copy: `/resources/mlx-vlm/mlx_vlm/convert.py` lines 104-183
3. Key step: Ensure `skip_multimodal_module()` is active
4. Use: `mlx_vlm.convert --hf-path <model> --mlx-path <output>`

### Scenario 3: Convert audio model (custom Whisper-like architecture)

1. Reference: CONVERSION_PATTERNS.md Section 4
2. Study: `/resources/mlx-examples/whisper/convert.py` lines 112-240
3. Key steps: hf_to_pt() mapping, convolution transposition, alignment heads
4. Implement: Custom convert() function with tensor transformations

### Scenario 4: Convert embedding model for vector search

1. Reference: CONVERSION_PATTERNS.md Section 5
2. Copy: `/resources/mlx-examples/bert/convert.py`
3. Key step: Simple key remapping with .replace() chains
4. Use: Custom conversion preserving embedding output

### Scenario 5: Fix quantization quality issues

1. Issue: Refer to CONVERSION_REFERENCE.md "Common Issues and Solutions"
2. For LLMs: Use mixed-bit recipes (mixed_4_6 often best)
3. For VLMs: Ensure `skip_multimodal_module()` is active
4. For critical layers: Set higher bits in quantization predicate

## Performance Optimization Tips

- Use `lazy=True` during loading to save memory
- Shard large models into 5GB chunks
- Apply quantization for 4x size reduction (4-bit)
- Skip quantizing vision/audio modules for quality
- Use mixed-bit recipes for better quality at same size
- Test with small variants first (tiny, small, base)

## Resources Directory Structure

### MLX LM Repository

- **Location:** `/resources/mlx-lm/`
- **Key files:** `convert.py`, `utils.py` with load/save/quantize functions
- **Models:** Language models, LLaVA variants
- **Conversions:** Full pipeline with quantization support

### MLX VLM Repository

- **Location:** `/resources/mlx-vlm/`
- **Key files:** `convert.py`, `utils.py` with VLM-specific patterns
- **Models:** Vision-language models with image processors
- **Conversions:** Multi-component handling, processor saving

### MLX Examples Repository

- **Location:** `/resources/mlx-examples/`
- **Subdirectories:** bert/, clip/, encodec/, llava/, llms/, segment_anything/, whisper/
- **Purpose:** Specialized conversion examples for specific model types
- **Key patterns:** Dtype conversion, weight shaping, quantization

### MLX Framework

- **Location:** `/resources/mlx/`
- **Provides:** Core MLX array operations, neural network modules
- **Used by:** All conversion scripts via `import mlx.core as mx`

## Next Steps

1. **Learn patterns:** Read CONVERSION_PATTERNS.md Section 1-3
2. **Understand your model:** Determine model type (LLM, VLM, Audio, etc.)
3. **Find reference:** Locate matching converter in CONVERSION_REFERENCE.md
4. **Copy and adapt:** Use exact line numbers to find and copy code
5. **Test locally:** Start with small model variants
6. **Implement:** Add to smlx/tools/convert2mlx.py
7. **Document:** Follow CLAUDE.md guidelines for code style

## Document Maintenance

These documents were generated by analyzing:

- 12 conversion scripts across resources/
- Over 1500 lines of conversion code
- 8 different model type implementations
- Quantization strategies and weight handling patterns

Last updated: November 13, 2025
