# Quantization & Parameter-Efficient Fine-Tuning (smlx.quant)

Quantization techniques and low-rank adaptation methods optimized for "smol" models (< 1B parameters) on Apple M4 chipsets with unified memory architecture.

## Overview

The `smlx.quant` module provides state-of-the-art quantization and parameter-efficient fine-tuning methods that enable:

- **Memory reduction** - 4-bit quantization reduces model size by ~75%
- **Faster inference** - Quantized operations leverage Apple Silicon optimizations
- **Efficient fine-tuning** - LoRA/DoRA enable training with <1% of original parameters
- **Minimal accuracy loss** - Advanced methods preserve model quality

All implementations are built on MLX and optimized for Apple Silicon's unified memory architecture.

## Quick Start - Basic Quantization

The simplest way to quantize a model is using the `apply_quantization()` utility:

```python
from smlx.models.SmolLM2_135M import load
from smlx.utils.quantization import apply_quantization

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Apply 4-bit quantization (reduces size by ~75%)
model = apply_quantization(model, method="4bit")

# Generate with quantized model
from smlx.models.SmolLM2_135M.generate import generate
output = generate(model, tokenizer, "Hello, world!", max_tokens=50)
```

**Supported methods:**
- `fp16` - No quantization (baseline)
- `4bit` - 4-bit quantization (~75% size reduction)
- `8bit` - 8-bit quantization (~50% size reduction)
- `dwq` - Dynamic Weight Quantization

**Advanced methods** (require calibration data, see below):
- `gptq` - GPTQ quantization
- `awq` - AWQ quantization

**Example with custom parameters:**

```python
from smlx.utils.quantization import apply_quantization

# Apply 4-bit quantization with custom group size
model = apply_quantization(
    model,
    method="4bit",
    group_size=128,  # Larger = faster, smaller = more accurate
    verbose=True      # Show progress
)
```

**Checking quantization status:**

```python
from smlx.utils.quantization import get_quantization_info

info = get_quantization_info(model)
print(f"Is quantized: {info['is_quantized']}")
print(f"Quantized layers: {info['num_quantized']}")
print(f"Quantizable layers: {info['num_quantizable']}")
```

**Estimating model size:**

```python
from smlx.utils.quantization import estimate_quantized_size

# Estimate size for different quantization methods
fp16_size = estimate_quantized_size(model, method="fp16")
bit4_size = estimate_quantized_size(model, method="4bit")
bit8_size = estimate_quantized_size(model, method="8bit")

print(f"FP16: {fp16_size:.2f} GB")
print(f"4-bit: {bit4_size:.2f} GB ({(1-bit4_size/fp16_size)*100:.0f}% reduction)")
print(f"8-bit: {bit8_size:.2f} GB ({(1-bit8_size/fp16_size)*100:.0f}% reduction)")
```

**Important Notes:**
- Quantization modifies the model in-place
- FP16 method returns model unchanged (useful for benchmarking)
- Model layers must support `to_quantized()` method (e.g., `nn.Linear`)
- The input dimension must be divisible by `group_size` (default: 64)

## Quick Start - Advanced Quantization (GPTQ/AWQ)

For higher-quality quantization with minimal accuracy loss, use GPTQ or AWQ. These methods require calibration data but provide better results than basic quantization:

**GPTQ Quantization:**

```python
from smlx.models.SmolLM2_135M import load
from smlx.utils.quantization import apply_quantization
from smlx.quant.utils import load_calibration_data

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Load calibration data (cached automatically)
calibration_data = load_calibration_data(tokenizer, num_samples=128)

# Apply GPTQ quantization
model = apply_quantization(
    model,
    method="gptq",
    calibration_data=calibration_data,
    bits=4,
    group_size=128,
    verbose=True
)

# Or let it load calibration data automatically:
model = apply_quantization(
    model,
    method="gptq",
    tokenizer=tokenizer,  # Automatically loads calibration data
    verbose=True
)
```

**AWQ Quantization:**

```python
from smlx.models.SmolLM2_135M import load
from smlx.utils.quantization import apply_quantization
from smlx.quant.awq import llama_awq
from smlx.quant.utils import load_calibration_data

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Load calibration data
calibration_data = load_calibration_data(tokenizer, num_samples=128)

# Apply AWQ quantization with architecture-specific config
model = apply_quantization(
    model,
    method="awq",
    calibration_data=calibration_data,
    awq_config=llama_awq,  # Use llama_awq, mistral_awq, or qwen_awq
    bits=4,
    group_size=128,
    verbose=True
)

# Or with automatic calibration data loading:
model = apply_quantization(
    model,
    method="awq",
    tokenizer=tokenizer,
    awq_config=llama_awq,
    verbose=True
)
```

**Calibration Data:**

The `load_calibration_data()` function downloads and caches calibration text data automatically:

```python
from smlx.quant.utils import load_calibration_data

# Load default calibration dataset (cached in ~/.cache/smlx/calibration/)
calibration_data = load_calibration_data(
    tokenizer,
    num_samples=128,      # Number of calibration samples
    sequence_length=512,  # Length of each sequence
    dataset="default",    # Or "wikitext", or path to custom file
    verbose=True
)
```

**Pre-Quantized Model Detection:**

The model loader automatically detects if a model has pre-quantized weights:

```python
from smlx.models import load_model

# Automatically detects and uses pre-quantized weights
model, tokenizer = load_model(
    "mlx-community/SmolLM2-135M-Instruct-4bit",
    verbose=True
)
# ✓ Detected pre-quantized model:
#   - Quantized layers: 52
#   - Estimated bits: 4

# Disable pre-quantization detection if needed
model, tokenizer = load_model(
    "mlx-community/SmolLM2-135M-Instruct-4bit",
    detect_prequantized=False
)
```

## Supported Methods

### LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning by adding trainable low-rank decomposition matrices to frozen weights.

**Key Features:**

- Only 0.1-1% of parameters are trainable
- Works with both unquantized and quantized models
- Can be merged back into base weights for deployment
- Supports both Linear and Embedding layers

**How It Works:**

For a weight matrix `W  ^(d_out � d_in)`, LoRA adds:

```text
�W = scale * B^T @ A^T
```

where `A  ^(d_in � r)`, `B  ^(r � d_out)`, and `r << min(d_in, d_out)`

**Usage:**

```python
import mlx.nn as nn
from smlx.quant import LoRALinear, LoRAEmbedding

# Convert existing layer to LoRA
linear = nn.Linear(768, 768)
lora_layer = LoRALinear.from_base(linear, r=8, scale=20.0)

# Create from scratch
lora_layer = LoRALinear(
    input_dims=768,
    output_dims=768,
    r=8,              # Rank (4-64 typical)
    dropout=0.1,      # Regularization
    scale=20.0,       # LoRA scaling factor
)

# Training
for batch in train_loader:
    loss = compute_loss(lora_layer(batch))
    loss.backward()
    optimizer.step()

# Merge LoRA weights for deployment
fused_layer = lora_layer.fuse()
```

**Hyperparameters:**

- `r` (rank): 4-8 for small models, 16-32 for larger models
- `scale`: Typically 10-20, controls adaptation magnitude
- `dropout`: 0.0-0.1 for regularization

**Reference:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### DoRA (Weight-Decomposed Low-Rank Adaptation)

Enhanced version of LoRA that decomposes weight updates into magnitude and direction.

**Key Features:**

- Better performance than LoRA on many tasks
- Separates magnitude and direction of weight updates
- Similar parameter efficiency to LoRA
- Supports Linear and Embedding layers

**How It Works:**

DoRA decomposes weights as:

```text
W' = m * (W + �W) / ||W + �W||
```

where `m` is the learned magnitude and `�W` is the low-rank update.

**Usage:**

```python
from smlx.quant import DoRALinear, DoRAEmbedding

# Convert existing layer
linear = nn.Linear(768, 768)
dora_layer = DoRALinear.from_base(linear, r=8)

# Create from scratch
dora_layer = DoRALinear(
    input_dims=768,
    output_dims=768,
    r=8,
    dropout=0.1,
    scale=20.0,
)

# Same training workflow as LoRA
# Fuse when done
fused_layer = dora_layer.fuse()
```

**When to Use DoRA vs LoRA:**

- DoRA: Better for tasks requiring fine-grained control (e.g., instruction tuning)
- LoRA: Faster training, good for most use cases

**Reference:** [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

### GPTQ (GPT Quantization)

Post-training quantization with Hessian-based error compensation for minimal accuracy loss.

**Key Features:**

- 4-bit or 8-bit quantization
- Uses second-order information (Hessian) for optimal weight selection
- Per-channel or per-group quantization
- No training data required (uses calibration set)

**Usage:**

```python
from smlx.quant import gptq_quantize, load_calibration_data
from transformers import AutoTokenizer

# Load model
model = load_your_model()
tokenizer = AutoTokenizer.from_pretrained("model_id")

# Prepare calibration data
calib_data = load_calibration_data(
    tokenizer,
    num_samples=128,
    sequence_length=512
)

# Quantize model
quantized_model = gptq_quantize(
    model=model,
    data=calib_data,
    bits=4,              # 4 or 8 bits
    group_size=128,      # Quantization group size
    percdamp=0.01,       # Damping factor for Hessian
)

# Save quantized model
mx.save("model_4bit.safetensors", quantized_model.parameters())
```

**Hyperparameters:**

- `bits`: 4 (4�compression) or 8 (2�compression)
- `group_size`: 64-128 typical, smaller = more memory but better accuracy
- `percdamp`: 0.01 typical, controls Hessian damping

**Reference:** [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

### AWQ (Activation-aware Weight Quantization)

MLSys 2024 Best Paper - Quantization that preserves salient weights based on activation statistics.

**Key Features:**

- Identifies and protects important weights using activation magnitudes
- 4-bit quantization with near-FP16 accuracy
- Per-channel scaling optimized for hardware
- Architecture-specific implementations (Llama, Mistral, Qwen)

**Usage:**

```python
from smlx.quant import awq_quantize, AWQConfig, llama_awq
from transformers import AutoTokenizer

# Load model and tokenizer
model = load_your_model()
tokenizer = AutoTokenizer.from_pretrained("model_id")

# Configure AWQ
config = AWQConfig(
    bits=4,
    group_size=128,
    symmetric=True,
)

# Prepare calibration data
calib_data = load_calibration_data(tokenizer)

# Quantize (architecture-specific)
quantized_model = llama_awq(
    model=model,
    data=calib_data,
    config=config,
)

# Or use generic quantization
quantized_model = awq_quantize(model, calib_data, config)
```

**Architecture-Specific Functions:**

- `llama_awq()` - Optimized for Llama/SmolLM architectures
- `mistral_awq()` - Optimized for Mistral models
- `qwen_awq()` - Optimized for Qwen models

**Reference:** [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

### Dynamic Quantization

Mixed-precision quantization that selects per-layer bit-widths based on sensitivity analysis.

**Key Features:**

- Automatically determines optimal bit-width for each layer
- Balances accuracy and memory usage
- Sensitivity-based layer selection
- Supports 2, 4, 8-bit quantization

**Usage:**

```python
from smlx.quant import dynamic_quantize, estimate_sensitivities
from transformers import AutoTokenizer

# Load model
model = load_your_model()
tokenizer = AutoTokenizer.from_pretrained("model_id")

# Prepare calibration data
calib_data = load_calibration_data(tokenizer)

# Estimate layer sensitivities
sensitivities = estimate_sensitivities(
    model=model,
    data=calib_data,
)

# Quantize with mixed precision
quantized_model = dynamic_quantize(
    model=model,
    sensitivities=sensitivities,
    target_bits=4.5,     # Average target bits
    available_bits=[2, 4, 8],
)
```

**How It Works:**

1. Analyzes activation statistics for each layer
2. Computes sensitivity scores (impact on output)
3. Assigns higher precision to sensitive layers
4. Quantizes less sensitive layers more aggressively

### DWQ (Distilled Weight Quantization)

Quantization with knowledge distillation from the original model.

**Key Features:**

- Uses teacher model to guide quantization
- Can achieve better accuracy than standard quantization
- Requires some training data
- Supports arbitrary bit-widths

**Usage:**

```python
from smlx.quant import dwq_quantize, dwq_quantize_simple

# Simple version (no training)
quantized_model = dwq_quantize_simple(
    model=model,
    bits=4,
    group_size=128,
)

# Full version with distillation
quantized_model = dwq_quantize(
    model=model,
    data=training_data,
    bits=4,
    epochs=3,
    learning_rate=1e-4,
)
```

### FP4 (4-bit Floating Point)

FP4 quantization uses 4-bit floating point representations instead of integer quantization, providing better dynamic range for weights with wide value distributions. SMLX supports multiple FP4 formats optimized for different use cases.

**Supported FP4 Modes:**

1. **E2M1** - Standard FP4 simulation (1 sign, 2 exponent, 1 mantissa)
2. **MXFP4** - MLX native, hardware accelerated (OCP Microscaling standard)
3. **NVFP4** - NVIDIA GPU optimized
4. **NF4** - Normal Float 4 from QLoRA (information-theoretically optimal)

**Key Features:**

- Wide dynamic range (±0.5 to ±6.0 for E2M1)
- ~4x compression vs FP16
- Better than INT4 for long-tail distributions
- Hardware acceleration (MXFP4/NVFP4)
- QLoRA-compatible (NF4)

**When to Use FP4:**

- ✅ Weights with wide dynamic range
- ✅ Non-uniform distributions
- ✅ QLoRA fine-tuning (use NF4)
- ✅ Apple Silicon inference (use MXFP4)
- ❌ When broad hardware support needed (use INT4/GPTQ/AWQ instead)

#### E2M1 (Standard FP4)

Standard FP4 simulation with flexible group sizes. Best for research and experimentation.

**Format:** 1 sign bit, 2 exponent bits, 1 mantissa bit
**Range:** ±0.5 to ±6.0 (exponential spacing)
**Group Size:** Flexible (default: 64)
**Implementation:** Simulated (lookup table + per-group scaling)

**Usage:**

```python
from smlx.quant import quantize_fp4, dequantize_fp4

# Quantize weights
quantized, scales = quantize_fp4(weights, mode="e2m1", group_size=64)

# Dequantize
restored = dequantize_fp4(quantized, scales, mode="e2m1", group_size=64)

# Model-level quantization
from smlx.quant import quantize_model_fp4
quantized_weights = quantize_model_fp4(model, mode="e2m1", group_size=64)
```

**Advantages:**
- Flexible group sizes (16, 32, 64, 128, 256)
- Easy to understand and debug
- Good for research and analysis

**Disadvantages:**
- Requires dequantization before use
- No hardware acceleration

#### MXFP4 (Microscaling FP4)

MLX native FP4 with hardware acceleration. Best for production inference on Apple Silicon.

**Format:** E2M1 (same as above)
**Range:** ±0.5 to ±6.0
**Group Size:** 32 (fixed, OCP standard requirement)
**Implementation:** MLX native, hardware accelerated

**Usage:**

```python
from smlx.quant import quantize_fp4, dequantize_fp4

# Quantize weights (group_size automatically set to 32)
quantized, scales = quantize_fp4(weights, mode="mxfp4")

# Dequantize
restored = dequantize_fp4(quantized, scales, mode="mxfp4")

# Model quantization
quantized_weights = quantize_model_fp4(model, mode="mxfp4")
```

**Advantages:**
- Hardware accelerated on Apple Silicon
- Direct computation on quantized weights
- OCP Microscaling standard compliant
- Fastest FP4 mode

**Disadvantages:**
- Fixed group_size=32 (not configurable)
- Apple Silicon specific

#### NVFP4 (NVIDIA FP4)

MLX native FP4 optimized for NVIDIA GPUs.

**Format:** E2M1
**Range:** ±0.5 to ±6.0
**Group Size:** 16 (fixed, NVIDIA optimization)
**Implementation:** MLX native, NVIDIA optimized

**Usage:**

```python
from smlx.quant import quantize_fp4, dequantize_fp4

# Quantize weights (group_size automatically set to 16)
quantized, scales = quantize_fp4(weights, mode="nvfp4")

# Dequantize
restored = dequantize_fp4(quantized, scales, mode="nvfp4")
```

**Advantages:**
- Optimized for NVIDIA GPUs
- Smaller group size (higher quality)
- Hardware accelerated

**Disadvantages:**
- Fixed group_size=16
- NVIDIA hardware specific

#### NF4 (Normal Float 4)

Information-theoretically optimal quantization for normally distributed weights. Used in QLoRA for efficient fine-tuning.

**Format:** Custom lookup table (non-uniform spacing)
**Range:** -1.0 to +1.0 (before scaling)
**Group Size:** Flexible (default: 64)
**Distribution:** Optimized for N(0,1)

**Key Difference:** NF4 uses non-uniform quantization with higher precision near zero, matching the distribution of neural network weights.

**Usage:**

```python
from smlx.quant import quantize_fp4, dequantize_fp4

# Quantize normally distributed weights
quantized, scales = quantize_fp4(weights, mode="nf4", group_size=64)

# Dequantize
restored = dequantize_fp4(quantized, scales, mode="nf4", group_size=64)

# Model quantization for QLoRA
quantized_weights = quantize_model_fp4(model, mode="nf4", group_size=64)
```

**NF4 Lookup Table Values:**
```
[-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
  0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0000]
```

**Advantages:**
- Information-theoretically optimal for N(0,1)
- Higher precision near zero (where weights concentrate)
- QLoRA-compatible
- Better quality for normally distributed weights

**Disadvantages:**
- Requires dequantization before use
- Suboptimal for non-normal distributions

#### FP4 Comparison & Best Practices

**Performance Comparison:**

| Mode   | Group Size | Hardware Accel | Best For                    |
|--------|------------|----------------|----------------------------|
| E2M1   | Flexible   | ❌             | Research, experimentation   |
| MXFP4  | 32 (fixed) | ✅             | Production (Apple Silicon)  |
| NVFP4  | 16 (fixed) | ✅             | Production (NVIDIA GPUs)    |
| NF4    | Flexible   | ❌             | QLoRA fine-tuning           |

**Quality Comparison:**

```python
from smlx.quant import compare_fp4_vs_int4

# Compare FP4 vs INT4
comparison = compare_fp4_vs_int4(weights, group_size=64)
print(f"FP4 error: {comparison['fp4_error']:.6f}")
print(f"INT4 error: {comparison['int4_error']:.6f}")
print(f"Recommendation: {comparison['recommendation']}")
```

**Group Size Effects (E2M1/NF4):**

| Group Size | Memory     | Quality | Speed |
|------------|------------|---------|-------|
| 16         | Higher     | Best    | Slower|
| 32         | Medium-High| Good    | Medium|
| 64         | Medium     | Good    | Medium|
| 128        | Medium-Low | Fair    | Faster|
| 256        | Lower      | Lower   | Fastest|

**Recommendation:** Use 64 for balanced quality/memory tradeoff.

**Decision Guide:**

```
Need hardware acceleration on Apple Silicon?
├─ Yes → Use MXFP4 (fastest)
└─ No
    ├─ Need QLoRA fine-tuning?
    │   └─ Yes → Use NF4 (optimal for neural weights)
    └─ No
        ├─ Need custom group sizes?
        │   └─ Yes → Use E2M1 (flexible)
        └─ No → Use INT4/GPTQ/AWQ (broader hardware support)
```

**Utility Functions:**

```python
from smlx.quant import estimate_fp4_size, FP4Mode

# Estimate memory usage
stats = estimate_fp4_size(model, group_size=64)
print(f"FP16: {stats['current_mb']:.2f} MB")
print(f"FP4: {stats['fp4_mb']:.2f} MB")
print(f"Reduction: {stats['reduction_ratio']:.2f}x")

# List available modes
print(f"Available FP4 modes: {[m.value for m in FP4Mode]}")
```

**Example: Complete Workflow**

```python
import mlx.core as mx
from smlx.quant import quantize_fp4, dequantize_fp4

# Generate test weights
weights = mx.random.normal((768, 768))

# Try different modes
for mode in ["e2m1", "mxfp4", "nvfp4", "nf4"]:
    # Quantize
    q, s = quantize_fp4(weights, mode=mode)

    # Dequantize
    restored = dequantize_fp4(q, s, mode=mode)

    # Check error
    error = mx.mean(mx.abs(restored - weights))
    print(f"{mode:8s}: Error = {float(error):.6f}")
```

**For More Details:**
- See `examples/quant/fp4_comparison.py` for comprehensive benchmarks
- See `tests/quant/test_fp4.py` for usage examples

### GGML Quantization Formats

GGML (GPT-Generated Model Language) quantization formats provide compatibility with llama.cpp ecosystem. These formats are optimized for efficient storage and inference.

**Available Formats:**

- **Q4_0**: Simplest 4-bit format with implicit bias (~8x compression, 18 bytes/32 weights)
- **Q4_1**: Improved 4-bit with explicit bias (~7.3x compression, 20 bytes/32 weights)
- **Q4_K**: Advanced hierarchical 4-bit (~7x compression, 4.56 bits/weight)
- **Q6_K**: High-quality hierarchical 6-bit (~5x compression, 6.56 bits/weight)
- **Q4_K_M**: Mixed precision strategy (~6.5x compression, ~4.9 bits/weight average)
- **Q8_0**: High-quality 8-bit format (~2x compression, 34 bytes/32 weights)

#### Q4_0 (GGML 4-bit, Implicit Bias)

The simplest GGML quantization format with symmetric quantization and implicit bias.

**Format Details:**

- **Block size:** 32 weights per block
- **Storage:** 18 bytes per block (2-byte FP16 scale + 16 bytes for weights)
- **Compression:** ~8x from FP32, ~4x from FP16
- **Bias:** Computed as `-8 * scale` (not stored)
- **Quantization range:** [0, 15] (4 bits)
- **Bytes per weight:** 0.5625

**How It Works:**

```text
Quantization:
  scale = max(abs(block)) / 8
  quantized = round((weight / scale) + 8)
  quantized = clip(quantized, 0, 15)

Dequantization:
  weight = scale * (quantized - 8)
```

**Usage:**

```python
from smlx.quant import quantize_to_q4_0, dequantize_from_q4_0, quantize_model_q4_0
import mlx.core as mx

# Quantize individual weight tensor
weights = mx.random.normal((768, 768))
w_quantized, scales = quantize_to_q4_0(weights)

# Dequantize back to float
w_dequantized = dequantize_from_q4_0(w_quantized, scales, weights.shape)

# Quantize entire model
from smlx.models.SmolLM2_135M import load
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
quantize_model_q4_0(model, block_size=32)  # In-place quantization

# Estimate size reduction
from smlx.quant import estimate_q4_0_size
size_info = estimate_q4_0_size(model)
print(f"Original: {size_info['original_mb']:.1f} MB")
print(f"Q4_0: {size_info['q4_0_mb']:.1f} MB ({size_info['reduction_ratio']:.1f}x reduction)")

# Compare with Q4_1
from smlx.quant import compare_q4_0_vs_q4_1
comparison = compare_q4_0_vs_q4_1(weights)
print(f"Q4_0 error: {comparison['q4_0_error']:.6f}")
print(f"Q4_1 error: {comparison['q4_1_error']:.6f}")
print(f"Quality improvement: {comparison['quality_improvement']:.1f}%")
```

**When to Use:**

- Maximum compression for storage/transfer
- llama.cpp compatibility required
- Quality degradation acceptable
- Slightly smaller than Q4_1 (2 bytes less per block)

#### Q4_1 (GGML 4-bit, Explicit Bias)

Improved 4-bit GGML format with explicit bias storage for better quality.

**Format Details:**

- **Block size:** 32 weights per block
- **Storage:** 20 bytes per block (2-byte scale + 2-byte bias + 16 bytes weights)
- **Compression:** ~7.3x from FP32, ~3.6x from FP16
- **Bias:** Stored explicitly as FP16
- **Quantization range:** [0, 15] (4 bits)
- **Bytes per weight:** 0.625

**How It Works:**

```text
Quantization:
  min_val = min(block)
  max_val = max(block)
  scale = (max_val - min_val) / 15
  bias = min_val
  quantized = round((weight - bias) / scale)
  quantized = clip(quantized, 0, 15)

Dequantization:
  weight = scale * quantized + bias
```

**Usage:**

```python
from smlx.quant import quantize_to_q4_1, dequantize_from_q4_1, quantize_model_q4_1

# Quantize individual weight tensor
weights = mx.random.normal((768, 768))
w_quantized, scales, biases = quantize_to_q4_1(weights)

# Dequantize
w_dequantized = dequantize_from_q4_1(w_quantized, scales, biases, weights.shape)

# Quantize entire model
quantize_model_q4_1(model, block_size=32)

# Estimate size
from smlx.quant import estimate_q4_1_size
size_info = estimate_q4_1_size(model)
print(f"Q4_1: {size_info['q4_1_mb']:.1f} MB")
```

**Compared to Q4_0:**

- Better quality (explicit bias allows asymmetric quantization)
- Slightly larger (2 extra bytes per block = +11% overhead)
- Better for weights with non-zero mean
- Recommended over Q4_0 unless size is critical

#### Q8_0 (GGML 8-bit)

High-quality 8-bit GGML quantization with minimal accuracy loss.

**Format Details:**

- **Block size:** 32 weights per block
- **Storage:** 34 bytes per block (2-byte FP16 scale + 32 bytes for 8-bit weights)
- **Compression:** ~2x from FP32, ~1x from FP16
- **Bias:** Computed as `-128 * scale` (implicit)
- **Quantization range:** [0, 255] (8 bits)
- **Bytes per weight:** 1.0625

**How It Works:**

```text
Quantization:
  scale = max(abs(block)) / 128
  quantized = round((weight / scale) + 128)
  quantized = clip(quantized, 0, 255)

Dequantization:
  weight = scale * (quantized - 128)
```

**Usage:**

```python
from smlx.quant import quantize_to_q8_0, dequantize_from_q8_0, quantize_model_q8_0

# Quantize weights
weights = mx.random.normal((768, 768))
w_quantized, scales = quantize_to_q8_0(weights)

# Dequantize
w_dequantized = dequantize_from_q8_0(w_quantized, scales, weights.shape)

# Quantize model
quantize_model_q8_0(model, block_size=32)

# Compare with INT8
from smlx.quant import compare_q8_0_vs_int8
comparison = compare_q8_0_vs_int8(weights)
print(f"Q8_0 uses implicit bias, INT8 uses explicit bias")
print(f"Q8_0 size: {comparison['q8_0_size_bytes']} bytes")
print(f"INT8 size: {comparison['int8_size_bytes']} bytes")
```

**When to Use:**

- Quality-sensitive applications
- Better than 4-bit but still need compression
- ~2x memory reduction with minimal quality loss
- Good intermediate option between FP16 and 4-bit

#### Q4_K (GGML K-Quantization 4-bit)

Advanced 4-bit GGML format with hierarchical two-tier quantization for superior quality.

**Format Details:**

- **Super-block size:** 256 weights (8 sub-blocks of 32 weights each)
- **Storage:** 4.5625 bits per weight (146 bytes per 256 weights)
- **Compression:** ~7x from FP32, ~3.5x from FP16
- **Hierarchical quantization:** Super-block scale (FP16) + per-subblock 6-bit scales and mins
- **Quality:** Superior to Q4_0/Q4_1 at similar compression

**How It Works:**

```text
Q4_K Super-block structure (256 weights):
  - 1 × d_scale (FP16): Super-block scale for dequantizing scales
  - 1 × d_min (FP16): Super-block minimum offset
  - 1 × d_min_scale (FP16): Super-block scale for dequantizing mins
  - 8 × scales (6-bit each): Per-subblock scales (quantized)
  - 8 × mins (6-bit each): Per-subblock minimums (quantized)
  - 256 × weights (4-bit each): Quantized weights

Storage breakdown per 256 weights:
  - 128 bytes: 256 × 4-bit weights (packed)
  - 12 bytes: 8 × 6-bit scales + 8 × 6-bit mins (packed)
  - 2 bytes: d_scale (FP16)
  - 2 bytes: d_min (FP16)
  - 2 bytes: d_min_scale (FP16)
  - Total: 146 bytes / 256 weights = 4.5625 bits/weight

Quantization formula:
  weight = (d_scale * scale_6bit/63) * q_4bit + (d_min + d_min_scale * min_6bit/63)
```

**Usage:**

```python
from smlx.quant.q4_k_m import (
    quantize_to_q4_k,
    dequantize_from_q4_k,
    quantize_model_q4_k,
    estimate_q4_k_size,
)
import mlx.core as mx

# Quantize individual weight tensor
weights = mx.random.normal((768, 768))
packed_w, d_scales, d_mins, d_min_scales, packed_sm = quantize_to_q4_k(weights)

# Dequantize
w_dequant = dequantize_from_q4_k(
    packed_w, d_scales, d_mins, d_min_scales, packed_sm, weights.shape
)

# Quantize entire model
from smlx.models.SmolLM2_135M import load
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
quantize_model_q4_k(model, block_size=256)

# Estimate size
size_info = estimate_q4_k_size(model)
print(f"Q4_K: {size_info['q4_k_mb']:.1f} MB ({size_info['reduction_ratio']:.1f}x)")
print(f"Bits per weight: {size_info['avg_bits_per_weight']:.2f}")
```

**When to Use:**

- Best quality 4-bit quantization
- Superior to Q4_0/Q4_1 at similar compression
- Compatible with llama.cpp Q4_K format
- Recommended over Q4_0/Q4_1 for production

#### Q6_K (GGML K-Quantization 6-bit)

High-quality 6-bit GGML format with hierarchical quantization for critical layers.

**Format Details:**

- **Super-block size:** 256 weights (16 sub-blocks of 16 weights each)
- **Storage:** 6.5625 bits per weight (210 bytes per 256 weights)
- **Compression:** ~4.9x from FP32, ~2.4x from FP16
- **Hierarchical quantization:** Super-block scale (FP16) + per-subblock 8-bit scales
- **Quality:** Better than Q4_K, approaching 8-bit quality

**How It Works:**

```text
Q6_K Super-block structure (256 weights):
  - 1 × d_scale (FP16): Super-block scale for dequantizing scales
  - 16 × scales (int8): Per-subblock scales
  - 256 × weights (6-bit each): Quantized weights in range [0, 63]

6-bit weight packing:
  - Lower 4 bits: Packed 2 per byte (128 bytes total)
  - Upper 2 bits: Packed 4 per byte (64 bytes total)
  - Total weight storage: 192 bytes

Storage breakdown per 256 weights:
  - 128 bytes: Lower 4 bits of weights (2 per byte)
  - 64 bytes: Upper 2 bits of weights (4 per byte)
  - 16 bytes: 16 × int8 scales
  - 2 bytes: d_scale (FP16)
  - Total: 210 bytes / 256 weights = 6.5625 bits/weight

Quantization formula:
  weight = scale * (weight_6bit - 32)
  where scale = d_scale * scale_int8 / 127.0
```

**Usage:**

```python
from smlx.quant.q6_k import (
    quantize_to_q6_k,
    dequantize_from_q6_k,
)
import mlx.core as mx

# Quantize individual weight tensor
weights = mx.random.normal((768, 768))
ql, qh, scales, d_scales = quantize_to_q6_k(weights)

# Dequantize
w_dequant = dequantize_from_q6_k(ql, qh, scales, d_scales, weights.shape)

# Typical usage: as part of Q4_K_M mixed precision (see below)
```

**When to Use:**

- Quality-critical layers (attention projections, output layers)
- As part of Q4_K_M mixed precision strategy
- Better quality needed than Q4_K at ~50% more storage
- Approaching 8-bit quality at lower cost

#### Q4_K_M (GGML K-Quantization Mixed Precision)

Model-level mixed precision strategy using Q6_K for important layers and Q4_K for others.

**Strategy Details:**

- **Average storage:** ~4.8 bits per weight (depends on architecture)
- **Q6_K usage:** Half of attention.v_proj and MLP.down_proj layers (even-indexed)
- **Q4_K usage:** All other layers
- **Quality:** Better than pure Q4_K with minimal size increase
- **Compatible with:** llama.cpp Q4_K_M format concept

**How It Works:**

```text
Layer selection strategy:
  1. Identify transformer layers (layers.0, layers.1, etc.)
  2. For even-indexed layers (0, 2, 4, ...):
     - v_proj: Use Q6_K (attention values - quality critical)
     - down_proj: Use Q6_K (MLP output - quality critical)
  3. All other layers: Use Q4_K

Example for 4-layer model:
  ├─ layers.0.v_proj: Q6_K (6.56 bits/weight)
  ├─ layers.0.down_proj: Q6_K (6.56 bits/weight)
  ├─ layers.0.q_proj, k_proj, o_proj, up_proj: Q4_K (4.56 bits/weight)
  ├─ layers.1.*: All Q4_K (4.56 bits/weight)
  ├─ layers.2.v_proj: Q6_K (6.56 bits/weight)
  ├─ layers.2.down_proj: Q6_K (6.56 bits/weight)
  ├─ layers.2.q_proj, k_proj, o_proj, up_proj: Q4_K (4.56 bits/weight)
  ├─ layers.3.*: All Q4_K (4.56 bits/weight)
  └─ lm_head, embed: Q4_K (4.56 bits/weight)

Typical distribution:
  - ~20% of parameters use Q6_K (important layers)
  - ~80% of parameters use Q4_K (other layers)
  - Average: ~4.9 bits/weight
```

**Usage:**

```python
from smlx.quant.q4_k_m import (
    quantize_model_q4_k_m,
    quantize_model_q4_k_m_ggml,
    estimate_q4_k_size,
)
from smlx.models.SmolLM2_135M import load

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Option 1: MLX Native mode (RECOMMENDED - TRUE memory savings)
quantize_model_q4_k_m(model, use_mlx_native=True)
# Uses MLX's QuantizedLinear with intelligent 4-bit/6-bit layer selection
# Provides actual runtime memory reduction and fast GPU inference

# Option 2: GGML mode (for GGUF export compatibility)
quantize_model_q4_k_m(model, use_mlx_native=False)
# Quantizes to Q4_K/Q6_K formats but dequantizes for MLX
# Useful for GGUF file export, but NO runtime memory savings

# Option 3: Explicit GGML mode
quantize_model_q4_k_m_ggml(model)
# Same as Option 2 but more explicit

# Estimate size
size_info = estimate_q4_k_size(model)
print(f"Q4_K_M: {size_info['q4_k_mb']:.1f} MB")
print(f"Avg bits/weight: {size_info['avg_bits_per_weight']:.2f}")
```

**When to Use:**

- Best quality-to-size ratio for 4-bit quantization
- Production deployments where quality matters
- Alternative to pure Q4_K when you have extra ~10% storage budget
- Compatible with llama.cpp Q4_K_M strategy
- **Recommended:** Use `use_mlx_native=True` for actual memory savings

**GGML Format Comparison:**

| Format | Bytes/Weight | Bits/Weight | Compression | Quality | Use Case |
|--------|--------------|-------------|-------------|---------|----------|
| Q4_0 | 0.5625 | 4.5 | ~8x | Good | Maximum compression |
| Q4_1 | 0.625 | 5.0 | ~7.3x | Better | Better quality 4-bit |
| Q4_K | 0.570 | 4.56 | ~7x | Best 4-bit | Advanced 4-bit |
| Q4_K_M | 0.61* | 4.9* | ~6.5x | Superior | Mixed precision (rec.) |
| Q6_K | 0.820 | 6.56 | ~4.9x | Near Q8 | Quality-critical layers |
| Q8_0 | 1.0625 | 8.5 | ~2x | Excellent | Quality-sensitive |

\* Q4_K_M average depends on architecture (typically ~20% Q6_K, ~80% Q4_K)

### Floating-Point Quantization (FP4/FP8/MXFP)

Floating-point quantization formats provide better dynamic range than integer quantization for certain use cases.

#### MXFP8 (Microscaling FP8) - Recommended

**OCP standard 8-bit floating-point format with true 8-bit storage and Metal GPU acceleration.**

**Format Details:**

- **Element format:** E4M3 (4-bit exponent, 3-bit mantissa)
- **Scale format:** E8M0 (8-bit exponent, shared per block)
- **Block size:** 32 elements (fixed by OCP specification)
- **Storage:** 1 byte per element + 1 byte scale per 32 elements
- **Compression:** ~3.8x from FP32, ~2x from FP16
- **Hardware:** Native Apple Metal GPU acceleration (software emulated on M4)

**How It Works:**

```text
Block structure (32 elements):
  - 32 × E4M3 values (8-bit each) = 256 bits
  - 1 × E8M0 scale (8-bit) = 8 bits
  - Total = 264 bits / 32 elements ≈ 8.25 bits per element

E4M3 range: ±448.0 with 8 quantization levels between powers of 2
E8M0 scale: Shared exponent for entire block
```

**Apple M4 Performance Notes:**

⚠️ **Important**: Apple M4 does NOT have native FP8 hardware support. MXFP8 runs via Metal software emulation.

**M4 Hardware Support:**
- ✅ Supported: FP64, FP32 (CPU); FP32, FP16, BF16, INT8 (GPU/AMX)
- ❌ NOT supported: FP8 execution units
- ⚠️ MXFP8: Software-emulated via Metal shaders

**Performance Characteristics on M4:**
- **Memory savings**: ✅ Real 2x reduction vs FP16 (true 8-bit storage)
- **Inference speed**: ⚠️ May be slower than FP16/INT8 due to emulation overhead
- **Quality**: ✅ Better than INT8 for wide dynamic range / non-uniform distributions
- **Recommendation**: Benchmark MXFP8 vs INT8 for your specific use case

**When to Use MXFP8 on M4:**
- ✅ Memory-constrained scenarios (need 2x savings over FP16)
- ✅ Weights with wide dynamic range or non-uniform distributions
- ✅ Training/fine-tuning with quantized gradients
- ✅ Export/compatibility with OCP MX standard
- ❌ Speed-critical inference (consider INT8 + GPTQ/AWQ instead)

**Usage:**

```python
from smlx.quant import quantize_to_mxfp8, dequantize_from_mxfp8, quantize_model_mxfp8
import mlx.core as mx

# Quantize individual weight tensor
weights = mx.random.normal((768, 768))
w_quantized, scales = quantize_to_mxfp8(weights)
# Returns: (uint8, uint8) - true 8-bit storage!

# Dequantize
w_dequantized = dequantize_from_mxfp8(w_quantized, scales)

# Quantize entire model
from smlx.models.SmolLM2_135M import load
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
quantize_model_mxfp8(model, inplace=True)  # Converts to nn.QuantizedLinear

# Estimate size
from smlx.quant import estimate_mxfp8_size
size_info = estimate_mxfp8_size(model)
print(f"MXFP8: {size_info['mxfp8_mb']:.1f} MB ({size_info['reduction_ratio']:.1f}x reduction)")

# Compare with INT8
from smlx.quant import compare_mxfp8_vs_int8
comparison = compare_mxfp8_vs_int8(weights)
print(f"MXFP8 error: {comparison['mxfp8_error']:.6f}")
print(f"INT8 error: {comparison['int8_error']:.6f}")
```

**MXFP8 vs INT8 Comparison on Apple M4:**

| Metric | MXFP8 | INT8 (GPTQ/AWQ) |
|--------|-------|-----------------|
| **Storage** | uint8 (1 byte/element) ✅ | uint8 (1 byte/element) ✅ |
| **Scale storage** | uint8 (1 byte/32 elements) ✅ | float32 (4 bytes/group) ⚠️ |
| **Memory reduction** | ~3.8x from FP32, ~2x from FP16 | ~3.5x from FP32 |
| **Hardware accel (M4)** | Software emulated ⚠️ | Native AMX/GPU ✅ |
| **Inference speed (M4)** | Medium (emulation overhead) | Fast (native acceleration) ✅ |
| **Quality** | Excellent for wide range | Excellent with GPTQ/AWQ |
| **Group size** | Fixed at 32 (OCP spec) | Flexible (64, 128, 256) |
| **Dynamic range** | ±448.0 (exponential) | ±127 (linear) |
| **Industry standard** | OCP MX v1.0 ✅ | Widely supported ✅ |
| **Best for** | Memory savings, training | Speed, inference |

**Decision Guide for M4:**

```
Is memory your primary constraint?
├─ Yes, need maximum savings
│   ├─ Quality sensitive? → Use MXFP8 (true 2x reduction, better quality)
│   └─ Speed matters? → Use INT8 + GPTQ/AWQ (faster on M4)
└─ No, performance is key
    ├─ Training/fine-tuning? → Use MXFP8 (better gradients)
    ├─ Production inference? → Use INT8 + GPTQ/AWQ (M4 native)
    └─ Export/portability? → Use MXFP8 (OCP standard)
```

**Benchmark Recommendation:**

Always benchmark both methods for your specific model and use case:

```python
from smlx.quant import compare_mxfp8_vs_int8
from smlx.models.SmolLM2_135M import load

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Compare MXFP8 vs INT8
comparison = compare_mxfp8_vs_int8(
    model=model,
    test_prompt="The quick brown fox",
    max_tokens=100
)

print(f"MXFP8 speed: {comparison['mxfp8_tokens_per_sec']:.1f} tok/s")
print(f"INT8 speed: {comparison['int8_tokens_per_sec']:.1f} tok/s")
print(f"MXFP8 memory: {comparison['mxfp8_memory_mb']:.1f} MB")
print(f"INT8 memory: {comparison['int8_memory_mb']:.1f} MB")
print(f"Recommendation: {comparison['recommendation']}")
```

**Advantages of MXFP8:**

- ✅ True 8-bit storage (better than simulated FP8)
- ✅ Lower scale overhead (8-bit vs 32-bit for INT8)
- ✅ Better for non-uniform distributions (exponential spacing)
- ✅ OCP industry standard (portable across frameworks)
- ✅ Better for training/fine-tuning with quantized gradients

**Advantages of INT8 (GPTQ/AWQ):**

- ✅ Native M4 hardware acceleration (AMX + GPU)
- ✅ Faster inference on M4
- ✅ Flexible group sizes (64, 128, 256)
- ✅ Mature ecosystem and tooling
- ✅ Proven quality preservation techniques

**Limitations:**

- MXFP8: Fixed block size of 32 (last dimension must be divisible by 32)
- MXFP8: Only E4M3 format available (no E5M2)
- MXFP8: Requires shape validation/padding
- MXFP8: Software emulated on M4 (slower than INT8)

#### FP8 (Simulated) - DEPRECATED

⚠️  **DEPRECATED**: The `fp8` module provides **simulated** FP8 quantization stored as float16. Use `mxfp8` instead for true 8-bit storage.

**Why it's deprecated:**

- ❌ Stores as float16 (16-bit) → NO memory savings
- ❌ No hardware acceleration
- ❌ Simplified rounding instead of proper FP8 bit layout
- ❌ 2x memory overhead vs true 8-bit

**Migration:** See [smlx/quant/FP8_MIGRATION.md](../smlx/quant/FP8_MIGRATION.md) for complete migration guide.

```python
# OLD (deprecated):
from smlx.quant import quantize_to_fp8_e4m3
values, scales = quantize_to_fp8_e4m3(weights, group_size=64)
# Returns (float16, float16) - NOT true 8-bit!

# NEW (recommended):
from smlx.quant import quantize_to_mxfp8
values, scales = quantize_to_mxfp8(weights)
# Returns (uint8, uint8) - True 8-bit storage with hardware acceleration!
```

#### MXFP4 (Microscaling FP4)

**OCP standard 4-bit floating-point format.**

**Format Details:**

- **Element format:** E2M1 (2-bit exponent, 1-bit mantissa)
- **Scale format:** E8M0 (8-bit exponent, shared per block)
- **Block size:** 32 elements (fixed by OCP specification)
- **Storage:** 0.5 bytes per element + 1 byte scale per 32 elements
- **Compression:** ~7.5x from FP32, ~3.75x from FP16

**Usage:**

```python
from smlx.quant import quantize_to_mxfp4, dequantize_from_mxfp4, quantize_model_mxfp4

# Quantize weights
weights = mx.random.normal((768, 768))
w_quantized, scales = quantize_to_mxfp4(weights)

# Dequantize
w_dequantized = dequantize_from_mxfp4(w_quantized, scales)

# Quantize model
quantize_model_mxfp4(model, inplace=True)

# Estimate size
from smlx.quant import estimate_mxfp4_size
size_info = estimate_mxfp4_size(model)
print(f"MXFP4: {size_info['mxfp4_mb']:.1f} MB")
```

**When to Use:**

- Maximum compression with floating-point benefits
- Better than INT4 for non-uniform distributions
- OCP hardware support
- Quality-sensitive 4-bit quantization

**Format Comparison:**

| Format | Storage | Compression | Hardware Accel | Use Case |
|--------|---------|-------------|----------------|----------|
| MXFP8 (E4M3) | uint8 + scale | ~3.8x | ✅ Metal | **Production (recommended)** |
| FP8 (simulated) | float16 + scale | ~2x | ❌ None | ⚠️  DEPRECATED |
| MXFP4 (E2M1) | 4-bit + scale | ~7.5x | ✅ OCP | Maximum FP compression |
| INT8 | uint8 + fp32 | ~3.5x | ✅ Common | Integer-friendly workloads |
| INT4 | 4-bit + fp32 | ~7.3x | ✅ Common | Integer max compression |

### Automatic Quantization (AutoQuant)

Intelligent quantization strategy selection based on hardware capabilities and use case.

**Key Features:**

- Hardware capability detection (OCP Microscaling, MLX version, chip type)
- Use-case aware selection (inference, training, export)
- Comprehensive format support (integer, floating point, GGML, advanced)
- Automatic fallback when calibration data unavailable
- Model sensitivity analysis for mixed-precision

**Usage:**

```python
from smlx.quant import autoquant, analyze_model, select_strategy, recommend_strategy
from smlx.models.SmolLM2_135M import load

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Simple automatic quantization
quantized_model = autoquant(
    model,
    profile="balanced",  # "conservative", "balanced", or "aggressive"
    use_case="inference", # "inference", "training", or "export"
    verbose=True
)

# With calibration data for better quality
from smlx.quant.utils import load_calibration_data
calib_data = load_calibration_data(tokenizer, num_samples=128)

quantized_model = autoquant(
    model,
    calibration_data=calib_data,
    profile="balanced",
    use_case="inference",
    verbose=True
)

# Analyze model before quantization
analysis = analyze_model(
    model,
    calibration_data=calib_data,
    use_sensitivity=True,
    use_case="inference"
)

print(f"Parameters: {analysis['total_params']:,}")
print(f"FP16 size: {analysis['fp16_size_gb']:.2f} GB")
print(f"Hardware: {analysis['hardware']}")
print(f"Has sensitivities: {analysis['has_sensitivities']}")

# Get strategy recommendation
strategy = select_strategy(
    analysis,
    profile="balanced",
    use_case="inference"
)

print(f"Recommended method: {strategy['method']}")
print(f"Reason: {strategy['reason']}")
print(f"Expected reduction: ~{strategy.get('expected_reduction', 'N/A')}")

# Or just get recommendation without applying
recommendation = recommend_strategy(
    model,
    calibration_data=calib_data,
    profile="balanced",
    use_case="inference"
)
print(recommendation)
```

**Profiles:**

- **conservative**: Prioritizes quality (MXFP8, BFloat16, 8-bit)
- **balanced**: Quality/size tradeoff (AWQ, GPTQ, 4-bit, MXFP4)
- **aggressive**: Maximum compression (4-bit, dynamic quantization)

**Use Cases:**

- **inference**: Optimized for deployment (AWQ, GPTQ, GGML formats)
- **training**: Preserves gradient flow (MXFP8, BFloat16, 8-bit)
- **export**: Compatible formats for sharing (GGML, standard quantization)

**Hardware Detection:**

```python
from smlx.quant import detect_hardware_capabilities

capabilities = detect_hardware_capabilities()
print(f"OCP Microscaling: {capabilities['ocp_microscaling']}")
print(f"MLX version: {capabilities['mlx_version']}")
print(f"Chip type: {capabilities['chip']}")
print(f"Metal support: {capabilities['supports_metal']}")
```

**Selection Logic:**

1. **Training use case** → MXFP8 or BFloat16 (gradient-friendly)
2. **Has calibration + aggressive** → GPTQ or AWQ (best quality)
3. **Has sensitivities** → Dynamic quantization (mixed-precision)
4. **OCP Microscaling + conservative** → MXFP4/MXFP8 (hardware-accelerated)
5. **Balanced** → 4-bit, 6-bit, or 8-bit (standard compression)
6. **No calibration** → MXFP4, MXFP8, or standard bit-width quantization

**Supported Methods in AutoQuant:**

- **Integer:** 4-bit, 6-bit, 8-bit
- **Floating Point (Simulated):** FP4 (E2M1) - for research
- **Floating Point (Hardware):** MXFP4, MXFP8 (OCP standard) - recommended
- **Advanced:** GPTQ, AWQ, DWQ
- **Mixed-Precision:** Dynamic, sensitivity-based
- **Mixed-Bit:** 3-6 bit mixed strategies
- **BFloat16:** Brain Float 16 for training

**Note:** The simulated FP8 module (fp8.py) is deprecated. AutoQuant automatically uses MXFP8 for 8-bit floating-point quantization.

**Example Output:**

```
Hardware capabilities detected:
  OCP Microscaling: True
  MLX version: 0.20.0
  Chip: M4
  Metal support: True

Analyzing model...
  Total parameters: 135M
  FP16 size: 0.27 GB
  Use case: inference
  Profile: balanced
  Has calibration: True
  Has sensitivities: False

Selected strategy: awq
Reason: Balanced profile with calibration data - AWQ provides excellent quality preservation at 4-bit with activation-aware quantization

Applying AWQ quantization...
✓ Quantization complete
  Original: 0.27 GB
  Quantized: 0.07 GB
  Reduction: 75%
```

## Utility Functions

### load_calibration_data()

Load standard calibration dataset for quantization.

```python
from smlx.quant import load_calibration_data
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model_id")

# Load calibration data
calib_data = load_calibration_data(
    tokenizer=tokenizer,
    num_samples=128,      # Number of sequences
    sequence_length=512,  # Tokens per sequence
)
```

Downloads and caches wikitext data, tokenizes into random non-overlapping chunks.

### estimate_model_size()

Estimate memory footprint of a model.

```python
from smlx.quant import estimate_model_size
import mlx.core as mx

# Estimate size
size_info = estimate_model_size(model, dtype=mx.float16)

print(f"Total: {size_info['total_mb']:.2f} MB")
print(f"Parameters: {size_info['parameters']:,}")
print(f"Quantized: {size_info['quantized_mb']:.2f} MB")
print(f"Unquantized: {size_info['unquantized_mb']:.2f} MB")
```

### check_m4_compatibility()

Check if running on Apple M4 chipset.

```python
from smlx.quant import check_m4_compatibility

is_m4 = check_m4_compatibility()
if is_m4:
    print("Running on M4 with unified memory!")
```

### quantize_dequantize()

Helper for testing quantization effects.

```python
from smlx.quant import quantize_dequantize
import mlx.core as mx

weights = mx.random.normal((768, 768))
quantized_weights = quantize_dequantize(weights, bits=4, group_size=128)

# Measure quantization error
error = mx.abs(weights - quantized_weights).mean()
print(f"Quantization error: {error:.6f}")
```

## Best Practices

### Choosing a Quantization Method

**For inference only:**

- Start with **AWQ** (4-bit) - Best accuracy/size tradeoff
- Try **GPTQ** if AWQ doesn't work well
- Use **Dynamic** for automatic mixed-precision

**For fine-tuning:**

- Use **LoRA** for most tasks (fast, simple)
- Try **DoRA** if LoRA accuracy is insufficient
- Combine with 4-bit base model for maximum efficiency

**Memory constraints:**

- 4-bit quantization: ~75% memory reduction
- 8-bit quantization: ~50% memory reduction
- LoRA: ~99% fewer trainable parameters

### Hyperparameter Recommendations

**LoRA/DoRA:**

```python
# Small models (< 500M params)
r=4, scale=10.0, dropout=0.0

# Medium models (500M - 1B params)
r=8, scale=20.0, dropout=0.1
```

**GPTQ/AWQ:**

```python
# Best accuracy
bits=4, group_size=64

# Balanced
bits=4, group_size=128

# Maximum compression
bits=4, group_size=256
```

**Dynamic Quantization:**

```python
# Conservative (better accuracy)
target_bits=5.0, available_bits=[4, 8]

# Aggressive (smaller size)
target_bits=3.5, available_bits=[2, 4, 8]
```

### Workflow Example

Complete quantization + fine-tuning workflow:

```python
from smlx.quant import gptq_quantize, LoRALinear, load_calibration_data
from transformers import AutoTokenizer
import mlx.nn as nn

# 1. Load base model
model = load_your_model()
tokenizer = AutoTokenizer.from_pretrained("model_id")

# 2. Quantize base model to 4-bit
calib_data = load_calibration_data(tokenizer)
model_4bit = gptq_quantize(model, calib_data, bits=4, group_size=128)

# 3. Add LoRA adapters to specific layers
def add_lora_to_linear_layers(model, r=8):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            parent = model
            path = name.split('.')
            for p in path[:-1]:
                parent = getattr(parent, p)
            setattr(parent, path[-1], LoRALinear.from_base(module, r=r))
    return model

model_lora = add_lora_to_linear_layers(model_4bit, r=8)

# 4. Fine-tune LoRA adapters (base weights frozen)
# Only LoRA parameters have gradients
trainable_params = sum(p.size for p in model_lora.trainable_parameters())
print(f"Trainable parameters: {trainable_params:,}")

# ... training loop ...

# 5. Merge LoRA for deployment
for name, module in model_lora.named_modules():
    if isinstance(module, LoRALinear):
        parent = model_lora
        path = name.split('.')
        for p in path[:-1]:
            parent = getattr(parent, p)
        setattr(parent, path[-1], module.fuse())

# 6. Save final model
mx.save("model_4bit_finetuned.safetensors", model_lora.parameters())
```

## Benchmarking Quantization Methods

The `smlx.bench.suites.quantization` module provides comprehensive benchmarking:

```bash
# Compare multiple quantization methods
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m smlx.bench.suites.quantization \
    --model mlx-community/SmolLM2-135M-Instruct \
    --methods fp16 8bit 4bit \
    --output quantization_results.json
```

**Programmatic usage:**

```python
from smlx.bench.suites.quantization import compare_quantization_methods

# Benchmark different quantization methods
results = compare_quantization_methods(
    model_path="mlx-community/SmolLM2-135M-Instruct",
    quantization_methods=["fp16", "4bit", "8bit"],
    test_prompt="The quick brown fox jumps over the lazy dog",
    generation_tokens=100,
    verbose=True,
)

# Access results
for method, result in results.items():
    print(f"\n{method.upper()}:")
    print(f"  Model size: {result.model_size_gb:.2f} GB")
    print(f"  Generation speed: {result.generation_tps:.0f} tokens/sec")
    print(f"  Peak memory: {result.peak_memory_gb:.2f} GB")
    print(f"  Memory reduction: {result.memory_reduction_percent:.1f}%")
    print(f"  Speedup: {result.generation_speedup:.2f}x")
```

**Output includes:**
- Model size (GB) and memory reduction vs FP16
- Generation speed (tokens/sec) and speedup vs FP16
- Peak memory usage during generation
- Comprehensive comparison table
- Recommendations for different use cases

**Example output:**

```
==================================================================
QUANTIZATION COMPARISON BENCHMARK
==================================================================
Model: mlx-community/SmolLM2-135M-Instruct
Methods: fp16, 8bit, 4bit

==================================================================
Testing: FP16
==================================================================
Loading model...
Applying fp16 quantization...
FP16: No quantization applied

Benchmarking fp16...
  Model size: 270 MB
  Generation speed: 45 tokens/sec
  Peak memory: 1.2 GB

==================================================================
COMPARISON SUMMARY
==================================================================

Method       Size (GB)    Reduction    Speed (tok/s)   Speedup    Memory (GB)
----------------------------------------------------------------------
fp16         0.27         +0.0%        45              1.00x      1.20
8bit         0.14         -50.0%       52              1.16x      0.65
4bit         0.07         -75.0%       58              1.29x      0.38

==================================================================
RECOMMENDATIONS
==================================================================
  Fastest generation: 4bit (58 tok/s)
  Lowest memory: 4bit (0.38 GB)
  Smallest size: 4bit (0.07 GB)

  Use cases:
    - Production deployment: 4bit (best size/memory tradeoff)
    - Maximum speed: 4bit
    - Maximum quality: fp16 (baseline)
==================================================================
```

## Testing

Run quantization tests:

```bash
# All quantization tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/ -v

# Quantization utilities tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/utils/test_quantization.py -v

# Benchmark suite tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/bench/suites/test_quantization.py -v

# Specific method tests
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/test_lora.py -v
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/test_gptq.py -v
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/test_awq.py -v
/Users/ryanoboyle/miniforge3/envs/smlx/bin/python -m pytest tests/quant/test_dynamic.py -v
```

## Performance Benchmarks

Typical results on Apple M4 (36GB unified memory):

**SmolLM2-135M:**

- FP16: 270 MB, 45 tokens/sec
- 8-bit: 135 MB, 52 tokens/sec
- 4-bit (GPTQ): 68 MB, 58 tokens/sec
- 4-bit (AWQ): 68 MB, 60 tokens/sec

**SmolLM2-360M:**

- FP16: 720 MB, 38 tokens/sec
- 4-bit (AWQ): 180 MB, 45 tokens/sec
- 4-bit + LoRA (r=8): 182 MB, 43 tokens/sec

*Note: Benchmarks vary based on model architecture and prompt length.*

## Troubleshooting

**Issue:** Poor accuracy after quantization
**Solution:**

- Increase `group_size` for finer-grained quantization
- Try AWQ instead of GPTQ
- Use more calibration samples (256-512)

**Issue:** Out of memory during quantization
**Solution:**

- Reduce `num_samples` in calibration data
- Quantize layer-by-layer instead of full model
- Use 8-bit instead of computing 4-bit on large model

**Issue:** LoRA not learning
**Solution:**

- Increase `scale` parameter (try 20-40)
- Check that base weights are truly frozen
- Increase rank `r` (try 16-32)

**Issue:** Slow quantization speed
**Solution:**

- Reduce calibration samples
- Use `dwq_quantize_simple` instead of full distillation
- Ensure MLX is using GPU (check Activity Monitor)

## References

- **LoRA:** [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **DoRA:** [Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- **GPTQ:** [Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323)
- **AWQ:** [Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- **MLX:** [Apple MLX Framework](https://github.com/ml-explore/mlx)
- **MLX-LM:** [MLX Language Models](https://github.com/ml-explore/mlx-examples/tree/main/llms)
