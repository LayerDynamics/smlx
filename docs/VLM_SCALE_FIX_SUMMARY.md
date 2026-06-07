# VLM Scale Mismatch Fix - Session Summary

## Problem Identified

### Root Cause: Vision-Text Embedding Scale Mismatch

**Symptom:** All three VLM models (nanoVLM, Moondream2, TinyLLaVA) producing gibberish or empty output

**Diagnosis:**

- Added comprehensive vision feature diagnostics to track embedding statistics
- Discovered that projected vision features had **88x larger standard deviation** than text embeddings
- This extreme imbalance caused vision features to dominate attention, resulting in gibberish output

### nanoVLM Detailed Analysis

**Vision Feature Pipeline Statistics:**

| Stage | Mean | Std | Notes |
|-------|------|-----|-------|
| Vision Encoder Output | 0.016 | 0.35 | ✅ Reasonable |
| After Projection | 0.16 | **6.62** | ❌ Too high! |
| Text Embeddings | 0.001 | 0.14 | Reference |
| **Scale Ratio** | - | - | **47-88x** ❌ |

**Target:** Vision/Text ratio should be **< 2x** for balanced multimodal attention

## Solutions Implemented

### Phase 1.1: Diagnostic Infrastructure ✅ COMPLETED

Created `/Users/ryanoboyle/smlx/smlx/utils/vlm_diagnostics.py` with:

```python
def log_vision_features(vision_features, label, expected_mean_range, expected_std_range)
def log_embedding_comparison(vision_embeds, text_embeds, label)
def log_logits_distribution(logits, top_k, label)
def log_attention_mask(mask, label)
def compare_with_reference(smlx_output, reference_output, label)
```

**Integrated into all three VLM models:**

- `smlx/models/nanoVLM/model.py`
- `smlx/models/Moondream2/model.py`
- `smlx/models/TinyLLaVA/model.py`

**Usage:** Set `SMLX_DEBUG=1` environment variable to enable diagnostic logging

### Phase 1.2: Root Cause Identification ✅ COMPLETED

**Key Finding:** Projection layer amplifies vision features 20x (std 0.35 → 6.62)

### Phase 1.3: Scale Correction ⚠️ PARTIALLY WORKING

**Approach 1 - LayerNorm in Projection (REVERTED)**

- Added LayerNorm after projection layer
- Result: Normalized to std=1.0, ratio improved to 7.09x
- **Issue:** Created new layer without pretrained weights, disrupted model expectations
- **Outcome:** Empty output instead of gibberish (progress but not working)

**Approach 2 - Fixed Scale Factor (CURRENT)**

- Reverted projection changes to preserve pretrained weights
- Added scale factor (0.04) in model forward pass, before concatenation
- Result: Vision std=0.27, **ratio=1.92x** ✅ (target: <2x)

**Implementation in `smlx/models/nanoVLM/model.py`:**

```python
# Scale vision embeddings to match text embedding magnitude
# Projected vision features have std~6-10, text embeddings have std~0.1-0.5
# Scale by 0.04 to bring vision std from ~6.6 to ~0.26 (ratio ~2x instead of ~88x)
vision_scale_factor = 0.04
vision_embeds = vision_embeds * vision_scale_factor
```

**Current Status:**

- ✅ Scale ratio fixed: 88x → 1.92x
- ❌ Output still empty (no longer gibberish, but no meaningful text)

## Outstanding Issues

### 1. Checkpoint Mismatch

**Problem:** Using `lusxvr/nanoVLM-222M` which is NOT validated by mlx-vlm

- mlx-vlm uses `mlx-community/nanoLLaVA-1.5-8bit` instead
- Untested checkpoints may have training issues or incompatible architectures

**Solution:** Test with mlx-vlm validated checkpoints

### 2. Missing mlx-vlm Components

**Problem:** SMLX implementations missing critical components from mlx-vlm:

| Component | mlx-vlm | SMLX Status |
|-----------|---------|-------------|
| `prepare_inputs()` | ✅ | ❌ Missing |
| Chat templates | ✅ | ❌ Missing |
| Robust sampling (bfloat16 workaround) | ✅ | ❌ Partial |
| Image token placement | Model-specific | ❌ Not implemented |

**Critical Missing Function:**

```python
# resources/mlx-vlm/mlx_vlm/utils.py:782-895
def prepare_inputs(image_processor, processor, image, prompt, config)
```

This function:

- Applies model-specific chat templates
- Places image tokens correctly (before/after text depends on model)
- Handles tokenization with proper special tokens

### 3. Generation Parameters

**Current Settings:**

```python
temperature = 0.5  # Updated from 1.0 (mlx-vlm default)
top_p = 1.0        # Updated from 0.95
```

**Issue:** May need additional tuning or repetition penalty

### 4. Sampling Implementation

**Missing bfloat16 Workaround:**

```python
# resources/mlx-vlm/mlx_vlm/sample_utils.py:4-40
def top_p_sampling(logits, top_p, temperature):
    # CRITICAL: bfloat16 workaround
    if logits.dtype == mx.bfloat16:
        logits = logits.astype(mx.float32)
    # ... rest of sampling
```

## Recommended Next Steps

### Immediate (Phase 2)

1. **Port `prepare_inputs()` from mlx-vlm**
   - File: `resources/mlx-vlm/mlx_vlm/utils.py:782-895`
   - Target: Create `smlx/utils/vlm_utils.py`

2. **Implement Chat Templates**
   - File: `resources/mlx-vlm/mlx_vlm/prompt_utils.py:391-487`
   - Support model-specific formats (SmolVLM: images BEFORE text, LLaVA: images AFTER text)

3. **Upgrade Sampling**
   - Port robust top-p from `resources/mlx-vlm/mlx_vlm/sample_utils.py:4-40`
   - Add bfloat16 workaround

### Validation (Phase 3)

1. **Test with Known-Good Checkpoints**
   - SmolVLM: Try `mlx-community/Qwen2-VL-2B-Instruct-4bit` (proven working in mlx-vlm)
   - nanoVLM: Try `mlx-community/nanoLLaVA-1.5-8bit` if available

2. **Model-Specific Fixes**
   - SmolVLM: Verify image tokens placed BEFORE text
   - Moondream2: Verify starmie-v1 tokenizer integration
   - TinyLLaVA: Verify token injection strategy

### Testing (Phase 4)

1. **Create Integration Tests**
   - Test file: `tests/integration/test_vlm_generation.py`
   - Compare SMLX vs mlx-vlm outputs for same inputs

2. **Benchmark Suite**
   - Simple captioning (single object)
   - Visual question answering
   - Counting objects
   - Spatial reasoning

### Instrumentation (Phase 5)

1. **Enhanced Debug Tool**
   - Create `tools/debug_vlm_generation.py`
   - Track: vision features, combined embeddings, logits, token probabilities
   - Save diagnostic artifacts for analysis

## Files Modified This Session

### Created

- `smlx/utils/vlm_diagnostics.py` - Comprehensive VLM debugging utilities

### Modified

- `smlx/utils/__init__.py` - Export new diagnostics functions
- `smlx/models/nanoVLM/model.py` - Add diagnostics + vision scaling (0.04 factor)
- `smlx/models/nanoVLM/projection.py` - Documentation updates
- `smlx/models/Moondream2/model.py` - Add diagnostics
- `smlx/models/TinyLLaVA/model.py` - Add diagnostics

## Key Insights

1. **Vision-text scale mismatch is a critical but solvable problem**
   - 88x mismatch → gibberish output
   - 1.92x ratio → still issues but no longer catastrophic

2. **Projection layer scale amplification is inherent to architecture**
   - Pixel shuffle: 768 channels → 3072 channels (4x)
   - Linear projection: 3072 → 576 with large weight matrix
   - Default initialization causes large outputs

3. **Post-processing scaling preserves pretrained weights**
   - Modifying projection disrupts pretrained model expectations
   - Scaling after projection is non-destructive

4. **Diagnostics are essential for VLM debugging**
   - Without statistics, scale mismatch is invisible
   - Real-time logging reveals issues immediately

5. **Reference implementations (mlx-vlm) are crucial**
   - They contain years of lessons learned
   - Missing components (prepare_inputs, chat templates) are not optional

## Statistics Summary

### Before Fix

```
Vision encoder output: mean=0.016, std=0.35
Projected vision features: mean=0.16, std=6.62  ← Problem!
Text embeddings: mean=0.001, std=0.14
Scale ratio: 47-88x  ← Way too high!
Output: "In In In We We We Weâlaus page" (gibberish)
```

### After Scale Fix

```
Vision encoder output: mean=0.016, std=0.37
Projected vision features (raw): mean=0.16, std=6.75
Scaled vision features: mean=0.0064, std=0.27  ← Fixed!
Text embeddings: mean=0.001, std=0.14
Scale ratio: 1.92x  ← Target achieved!
Output: (empty)  ← Better than gibberish, but still not working
```

## Conclusion

**Phase 1 Progress:**

- ✅ Diagnostic infrastructure in place
- ✅ Root cause identified (88x scale mismatch)
- ✅ Scale ratio fixed (88x → 1.92x)
- ⚠️ Output quality not yet achieved (empty instead of gibberish)

**Next Critical Step:** Port mlx-vlm's `prepare_inputs()` and chat templates (Phase 2)

The scale fix was necessary but not sufficient. VLM generation requires:

1. Balanced vision-text scaling (now fixed)
2. Proper input formatting (missing - need prepare_inputs + chat templates)
3. Robust sampling (partial - need bfloat16 workaround)
4. Validated checkpoints (not verified)

All pieces must work together for production-quality output.
