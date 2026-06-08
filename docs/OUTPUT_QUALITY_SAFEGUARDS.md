# Output Quality Safeguards Guide

**Comprehensive guide to ensuring high-quality model outputs in SMLX**

---

## Table of Contents

1. [Overview](#overview)
2. [Common Output Quality Issues](#common-output-quality-issues)
3. [Safeguard Components](#safeguard-components)
4. [Quick Start](#quick-start)
5. [Validation Framework](#validation-framework)
6. [Quality Metrics](#quality-metrics)
7. [Model Loading Safety](#model-loading-safety)
8. [Generation with Validation](#generation-with-validation)
9. [Quantization Quality Assurance](#quantization-quality-assurance)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

SMLX provides comprehensive safeguards to detect and prevent common model output quality issues. These safeguards protect against:

- **Gibberish outputs**: Nonsensical character sequences or token combinations
- **Empty responses**: Zero-length or whitespace-only outputs
- **Pathological repetition**: Infinite loops of repeated tokens/phrases
- **Post-quantization degradation**: Quality loss from aggressive quantization
- **Model loading failures**: Incomplete weights, wrong tokenizers, corrupted parameters

### Why Safeguards Matter

Based on analysis in `PossibleCauses.md`, model outputs can fail due to:

1. **Improper model loading** - Random initialization, placeholder tokenizers
2. **Quantization corruption** - Weight rounding errors, calibration mismatch
3. **Wrong inference settings** - Temperature misconfiguration, broken sampling
4. **Vision-specific issues** - Missing visual grounding, connector layer failures

The safeguard system addresses all these issues systematically.

---

## Common Output Quality Issues

### 1. Gibberish Outputs

**Symptoms:**
```
Output: "©©©©©©©©"
Output: "aksdjfh askdjfh"
Output: "-ions."
```

**Causes:**
- Over-quantization (4-bit too aggressive)
- Random/uninitialized weights
- Corrupted connector layers (VLMs)

**Solution:** Use output validation + model integrity checks

### 2. Empty Outputs

**Symptoms:**
```
Output: ""
Output: "   "
```

**Causes:**
- EOS token generated immediately
- Tokenizer decode failure
- Min-length not enforced

**Solution:** Set `min_tokens` parameter + validation

### 3. Repetitive Loops

**Symptoms:**
```
Output: "the the the the the the..."
```

**Causes:**
- Low temperature + low repetition penalty
- Pathological token distribution

**Solution:** Repetition penalty + validation thresholds

### 4. Post-Quantization Degradation

**Symptoms:**
- Perplexity increases >20%
- Outputs become less coherent
- Random nonsensical tokens

**Causes:**
- Aggressive quantization (4-bit Q4_0)
- Poor calibration data

**Solution:** Quality regression testing

---

## Safeguard Components

SMLX's safeguard system consists of three layers:

### Layer 1: Pre-Generation (Model Loading)
```
┌─────────────────────────────────────┐
│  Weight Integrity Checks            │
│  - NaN/Inf detection                │
│  - All-zero weight detection        │
│  - Parameter count verification     │
│  - Tokenizer compatibility          │
└─────────────────────────────────────┘
```

### Layer 2: During Generation (Runtime)
```
┌─────────────────────────────────────┐
│  Generation Parameter Validation    │
│  - Temperature sanity checks        │
│  - Sampling parameter validation    │
│  - Min/max token enforcement        │
└─────────────────────────────────────┘
```

### Layer 3: Post-Generation (Output Validation)
```
┌─────────────────────────────────────┐
│  Output Quality Validation          │
│  - Gibberish detection              │
│  - Repetition analysis              │
│  - Entropy/perplexity checks        │
│  - Retry on failure                 │
└─────────────────────────────────────┘
```

---

## Quick Start

### Basic Output Validation

```python
from smlx.models import load
from smlx.utils.generation import generate
from smlx.utils.validation import validate_text_output

# Load model
bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor

# Generate text
output = generate(model, tokenizer, "Explain AI:", max_tokens=100)

# Validate output
is_valid, reason = validate_text_output(
    output,
    min_length=10,
    max_repetition_ratio=0.5,
    check_gibberish=True
)

if not is_valid:
    print(f"⚠️  Output validation failed: {reason}")
else:
    print(f"✅ Output is valid: {output}")
```

### Generation with Built-in Validation

```python
from smlx.utils import GenerationConfig
from smlx.models import load
from smlx.utils.generation import generate

bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor

# Configure with validation enabled
config = GenerationConfig(
    max_tokens=200,
    temperature=0.7,
    validate_output=True,          # Enable validation
    max_repetition_ratio=0.5,      # Max 50% repetition
    retry_on_failure=True,         # Retry if fails
    max_retries=2                  # Up to 2 retries
)

output = generate(model, tokenizer, "Tell me about space:", **config.__dict__)
```

### Model Loading with Integrity Checks

```python
from smlx.models import load
from smlx.utils import verify_model_integrity, check_tokenizer_compatibility

# Load model
bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor

# Verify model integrity
verify_model_integrity(
    model,
    config={'num_parameters': 135_000_000}  # Expected param count
)

# Check tokenizer compatibility
check_tokenizer_compatibility(
    tokenizer,
    model_config={'vocab_size': 49152}
)
```

---

## Validation Framework

### OutputValidator Class

Configurable validator for comprehensive quality checks:

```python
from smlx.utils import OutputValidator

# Create validator with custom thresholds
validator = OutputValidator(
    min_length=10,                  # Minimum 10 characters
    max_length=5000,                # Maximum 5000 characters
    max_repetition_ratio=0.5,       # Max 50% repetition
    min_unique_ratio=0.3,           # Min 30% unique tokens
    check_gibberish=True,           # Enable gibberish detection
    check_special_chars=True,       # Check special char ratio
    strict=False                    # Strict mode
)

# Validate text
result = validator.validate_text("The cat sat on the mat.")
print(f"Valid: {result.is_valid}")
print(f"Confidence: {result.confidence}")
print(f"Metadata: {result.metadata}")
```

### Audio Validation

```python
from smlx.utils import OutputValidator
import mlx.core as mx

validator = OutputValidator(min_length=16000)  # 1 second at 16kHz

# Validate audio waveform
waveform = mx.random.normal((16000,)) * 0.5
result = validator.validate_audio(waveform, sample_rate=16000)

if result.is_valid:
    print(f"✅ Audio valid: {result.metadata}")
else:
    print(f"⚠️  Audio invalid: {result.reason}")
```

### Token Validation

```python
from smlx.utils import validate_tokens

tokens = [1, 2, 3, 4, 5, 100, 200]
vocab_size = 50000
eos_token_id = 2

is_valid, reason = validate_tokens(tokens, vocab_size, eos_token_id)
```

---

## Quality Metrics

### Perplexity Calculation

Measure how well the model "likes" its own output:

```python
from smlx.utils import calculate_perplexity
from smlx.models import load

bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor

text = "The cat sat on the mat."
perplexity = calculate_perplexity(model, tokenizer, text)

print(f"Perplexity: {perplexity:.1f}")
# Good text: 10-100
# Bad text: >500
# Gibberish: >1000
```

### Repetition Analysis

Detect pathological repetition patterns:

```python
from smlx.utils import analyze_repetition

text = "the cat sat on the mat and the cat sat on the mat"
metrics = analyze_repetition(text, max_n=4)

print(f"1-gram repetition: {metrics['repetition_1gram']:.2%}")
print(f"2-gram repetition: {metrics['repetition_2gram']:.2%}")
print(f"3-gram repetition: {metrics['repetition_3gram']:.2%}")
```

### Comprehensive Quality Assessment

```python
from smlx.utils import assess_quality

# Full quality assessment
metrics = assess_quality(
    model,
    tokenizer,
    text,
    perplexity_threshold=500.0,
    repetition_threshold=0.6,
    min_diversity=0.2
)

print(f"High Quality: {metrics.is_high_quality}")
print(f"Perplexity: {metrics.perplexity:.1f}")
print(f"Repetition (3-gram): {metrics.repetition_3gram:.2%}")
print(f"Unique Token Ratio: {metrics.unique_token_ratio:.2%}")

if not metrics.is_high_quality:
    print(f"Reasons: {metrics.metadata['quality_reasons']}")
```

---

## Model Loading Safety

### Enhanced Weight Verification

```python
from smlx.utils import load_weights, verify_weights

# Load weights
weights = load_weights("path/to/model")

# Verify with enhanced checks
verify_weights(
    weights,
    expected_keys=None,              # Optional: list of expected keys
    model=model,                     # Optional: verify against model
    check_integrity=True,            # NaN/Inf detection
    check_distribution=True          # All-zero, constant weight detection
)
```

### Tokenizer Compatibility

```python
from smlx.utils import check_tokenizer_compatibility

# Verify tokenizer matches model
check_tokenizer_compatibility(
    tokenizer,
    model_config={
        'vocab_size': 49152,
        'max_position_embeddings': 2048
    }
)
```

### Full Model Integrity Check

```python
from smlx.utils import verify_model_integrity

# Comprehensive model verification
verify_model_integrity(
    model,
    weights=weights,
    config={
        'num_parameters': 135_000_000,
        'num_layers': 30,
        'hidden_size': 576
    }
)
```

---

## Generation with Validation

### Using GenerationConfig

```python
from smlx.utils import GenerationConfig

# Create config with safeguards
config = GenerationConfig(
    # Sampling parameters
    max_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,

    # Validation parameters
    validate_output=True,
    max_repetition_ratio=0.5,
    quality_threshold=0.7,          # Min diversity score
    retry_on_failure=True,
    max_retries=2,

    # Safety parameters
    min_tokens=10                   # Prevent empty outputs
)

# Use config
output = generate(model, tokenizer, prompt, **config.__dict__)
```

### Manual Validation Loop

```python
from smlx.utils import validate_text_output, assess_quality

max_attempts = 3
for attempt in range(max_attempts):
    # Generate
    output = generate(model, tokenizer, prompt, max_tokens=100, temperature=0.7)

    # Quick validation
    is_valid, reason = validate_text_output(output)

    if is_valid:
        # Full quality check
        metrics = assess_quality(model, tokenizer, output)

        if metrics.is_high_quality:
            print(f"✅ Success on attempt {attempt + 1}")
            break
        else:
            print(f"⚠️  Low quality (attempt {attempt + 1}): {metrics.metadata['quality_reasons']}")
    else:
        print(f"❌ Validation failed (attempt {attempt + 1}): {reason}")

    # Adjust parameters for retry
    temperature *= 0.8  # Reduce randomness
else:
    print("⚠️  All attempts failed")
```

---

## Quantization Quality Assurance

### Compare Pre/Post Quantization

```python
from smlx.models import load
from smlx.utils.generation import generate
from smlx.quant import quantize_model
from smlx.utils import assess_quality, compare_quality

# Load full precision model
bm = load("smollm2-135m")
model_fp, tokenizer = bm.model, bm.processor

# Quantize to 4-bit
model_4bit = quantize_model(model_fp, bits=4, group_size=64)

# Test prompt
prompt = "Explain quantum computing:"

# Generate with both
output_fp = generate(model_fp, tokenizer, prompt, max_tokens=100, temperature=0.0)
output_4bit = generate(model_4bit, tokenizer, prompt, max_tokens=100, temperature=0.0)

# Assess quality
quality_fp = assess_quality(model_fp, tokenizer, output_fp)
quality_4bit = assess_quality(model_4bit, tokenizer, output_4bit)

# Compare
comparison = compare_quality(quality_fp, quality_4bit, tolerance=0.20)

print(f"Full Precision Quality: {quality_fp.is_high_quality}")
print(f"4-bit Quality: {quality_4bit.is_high_quality}")
print(f"Acceptable Degradation: {comparison['acceptable']}")

if not comparison['acceptable']:
    print(f"⚠️  Quality degraded too much:")
    for degradation in comparison['degradations']:
        print(f"  - {degradation}")
```

### Quantization Quality Test Suite

Run comprehensive quantization tests:

```bash
# Run quantization quality tests
python -m pytest tests/quant/test_output_quality.py -v

# Run specific test
python -m pytest tests/quant/test_output_quality.py::TestQuantizationOutputQuality::test_4bit_quality_vs_full_precision -v

# Run with quality threshold benchmarking
python -m pytest tests/quant/test_output_quality.py::TestQuantizationPerformanceVsQuality -v
```

---

## Troubleshooting

### Issue: Model Outputs Gibberish

**Symptoms:**
```python
output = generate(model, tokenizer, "Hello")
# Output: "©©©©©©"
```

**Diagnosis:**
```python
from smlx.utils import verify_model_integrity, validate_text_output

# Check model loading
verify_model_integrity(model)  # May raise error if weights corrupted

# Check output
is_valid, reason = validate_text_output(output, check_gibberish=True)
print(reason)  # "Gibberish detected: Excessive special characters"
```

**Solutions:**
1. **Reload model with integrity checks**
2. **Use less aggressive quantization** (8-bit instead of 4-bit)
3. **Check tokenizer matches model** - run `check_tokenizer_compatibility()`

### Issue: Empty or Very Short Outputs

**Symptoms:**
```python
output = generate(model, tokenizer, "Explain AI")
# Output: ""
```

**Diagnosis:**
```python
# Check if EOS token is being generated immediately
tokens = tokenizer.encode(prompt)
logits = model(mx.array(tokens)[None, :])
probs = mx.softmax(logits[0, -1, :])
eos_prob = float(probs[tokenizer.eos_token_id])
print(f"EOS probability: {eos_prob:.1%}")  # If >50%, model wants to stop
```

**Solutions:**
1. **Set `min_tokens` parameter**:
   ```python
   config = GenerationConfig(min_tokens=20)
   ```
2. **Ban EOS token** for first N tokens:
   ```python
   config = GenerationConfig(
       logit_bias={tokenizer.eos_token_id: -100}  # Ban EOS
   )
   ```
3. **Check temperature** - too low temperature can cause early stopping

### Issue: Pathological Repetition

**Symptoms:**
```python
output = generate(model, tokenizer, "Once upon a time")
# Output: "the the the the the the the..."
```

**Diagnosis:**
```python
from smlx.utils import analyze_repetition

metrics = analyze_repetition(output, max_n=3)
print(f"3-gram repetition: {metrics['repetition_3gram']:.2%}")
# If >60%, pathological repetition
```

**Solutions:**
1. **Increase repetition penalty**:
   ```python
   config = GenerationConfig(
       repetition_penalty=1.2,
       repetition_context_size=50
   )
   ```
2. **Use validation with retry**:
   ```python
   config = GenerationConfig(
       validate_output=True,
       max_repetition_ratio=0.5,
       retry_on_failure=True
   )
   ```
3. **Increase temperature** to add diversity

### Issue: Quality Degraded After Quantization

**Symptoms:**
- Perplexity increases >20%
- Outputs become nonsensical
- Coherence drops significantly

**Diagnosis:**
```python
from smlx.utils import compare_quality

comparison = compare_quality(quality_fp, quality_quant, tolerance=0.20)
print(comparison['degradations'])
# ["Perplexity increased 35% (50.0 → 67.5)"]
```

**Solutions:**
1. **Use less aggressive quantization**:
   - Try 8-bit instead of 4-bit
   - Increase `group_size` (64 → 128)
2. **Better calibration**:
   ```python
   from smlx.quant import apply_gptq

   # Use representative calibration data
   calibration_data = [...]
   model_gptq = apply_gptq(model, bits=4, calibration_data=calibration_data)
   ```
3. **Consider selective quantization** - quantize only certain layers

---

## Best Practices

### 1. Always Validate Critical Outputs

```python
# For production use
config = GenerationConfig(
    validate_output=True,
    max_repetition_ratio=0.5,
    retry_on_failure=True,
    min_tokens=10
)
```

### 2. Test Quantization Before Deployment

```python
# Run quality tests on representative prompts
test_prompts = [
    "Explain...",
    "List...",
    "Describe..."
]

for prompt in test_prompts:
    output_fp = generate(model_fp, tokenizer, prompt)
    output_q = generate(model_quant, tokenizer, prompt)

    quality_fp = assess_quality(model_fp, tokenizer, output_fp)
    quality_q = assess_quality(model_quant, tokenizer, output_q)

    comparison = compare_quality(quality_fp, quality_q)
    assert comparison['acceptable'], f"Failed on: {prompt}"
```

### 3. Log Quality Metrics in Production

```python
import logging

logger = logging.getLogger(__name__)

output = generate(model, tokenizer, prompt)
metrics = assess_quality(model, tokenizer, output)

logger.info(f"Generated output quality: PPL={metrics.perplexity:.1f}, "
            f"Rep={metrics.repetition_3gram:.2%}, "
            f"Quality={metrics.is_high_quality}")
```

### 4. Use Presets for Common Scenarios

```python
# Conservative (high quality, low risk)
conservative_config = GenerationConfig(
    temperature=0.3,
    repetition_penalty=1.2,
    validate_output=True,
    max_repetition_ratio=0.4,
    retry_on_failure=True
)

# Balanced (good quality, some creativity)
balanced_config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    validate_output=True,
    max_repetition_ratio=0.5
)

# Creative (high diversity, accept more risk)
creative_config = GenerationConfig(
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.05,
    validate_output=True,
    max_repetition_ratio=0.6
)
```

### 5. Monitor Model Health

```python
from smlx.utils import verify_model_integrity

# Periodic health checks
def check_model_health(model, tokenizer):
    try:
        verify_model_integrity(model)
        check_tokenizer_compatibility(tokenizer)

        # Test generation
        test_output = generate(model, tokenizer, "Test", max_tokens=10)
        is_valid, _ = validate_text_output(test_output)

        return is_valid
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        return False

# Run periodically
if not check_model_health(model, tokenizer):
    logger.critical("⚠️  Model health check failed! Reload model.")
```

---

## Summary

SMLX's safeguard system provides comprehensive protection against output quality issues through:

✅ **Pre-generation validation** - Model integrity, weight verification, tokenizer compatibility
✅ **Runtime safeguards** - Parameter validation, min/max enforcement
✅ **Post-generation checks** - Gibberish detection, repetition analysis, quality metrics
✅ **Quantization safety** - Quality regression testing, perplexity monitoring
✅ **Automatic retry** - Configurable retry logic with parameter adjustment

By following the best practices in this guide, you can ensure your SMLX models produce high-quality, reliable outputs in production environments.

For more information:
- See `PossibleCauses.md` for detailed analysis of output failure modes
- Check `tests/quant/test_output_quality.py` for comprehensive test examples
- Review model-specific docs in `docs/ModelImplementations.md`
