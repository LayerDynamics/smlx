# Quantization Examples

Comprehensive examples demonstrating all quantization methods available in SMLX.

## Overview

SMLX provides state-of-the-art quantization techniques optimized for Apple Silicon (M1/M2/M3/M4):

- **Q4_K_M** - Mixed-precision 4-bit/6-bit quantization (MLX-native, RECOMMENDED) 🆕
- **GPTQ** - Hessian-based post-training quantization
- **AWQ** - Activation-aware weight quantization
- **DWQ** - Dynamic weight quantization
- **LoRA** - Low-rank adaptation for parameter-efficient fine-tuning
- **QLoRA** - Combines quantization with LoRA for maximum efficiency

## Examples

### 1. Q4_K_M Quantization ([q4_k_m_benchmark.py](q4_k_m_benchmark.py)) 🆕

Demonstrates Q4_K_M mixed-precision quantization with comprehensive benchmarks.

```bash
python q4_k_m_benchmark.py
```

**Features:**
- MLX-native mixed 4-bit/6-bit quantization
- ~4.8 bits/weight average (optimal quality/size)
- 6-7x compression from FP16
- TRUE runtime memory savings
- GGML-compatible mode for GGUF export

**Output:**
- Size comparison (FP16 vs quantized)
- Inference speed (tokens/sec)
- Compression ratios
- Quality metrics

**Quick Usage:**
```python
from smlx.quant import quantize_model_q4_k_m

# Quantize with Q4_K_M (MLX-native, recommended)
quantize_model_q4_k_m(model)  # Uses MLX QuantizedLinear

# Or use explicit mixed-precision API
from smlx.quant import quantize_model_mixed
quantize_model_mixed(model, style="q4_k_m", low_bits=4, high_bits=6)
```

### 2. GPTQ Quantization ([gptq_example.py](gptq_example.py))

Demonstrates GPTQ (GPT Quantization) for accurate 4-bit compression.

```bash
python gptq_example.py
```

**Features:**
- Hessian-based optimization
- Minimal quality degradation
- 4x memory reduction
- Optimized for M4 (group_size=64)

**Output:**
- Performance comparison (FP16 vs 4-bit)
- Quality metrics
- Different bit-width comparison (2-bit, 4-bit, 8-bit)

### 3. AWQ Quantization ([awq_example.py](awq_example.py))

Shows AWQ (Activation-aware Weight Quantization) for high-quality 4-bit models.

```bash
python awq_example.py
```

**Features:**
- Protects salient weight channels
- Activation-aware scaling
- Better quality than standard quantization
- Grid search for optimal scaling factors

**Output:**
- AWQ vs FP16 comparison
- Different group_size configurations
- Quality preservation metrics

### 4. LoRA Example ([lora_example.py](lora_example.py))

Demonstrates parameter-efficient fine-tuning with LoRA.

```bash
python lora_example.py
```

**Features:**
- Only ~0.1-1% parameters trainable
- Base model weights frozen
- Low-rank adaptation matrices
- Merge back into base model

**Output:**
- Trainable parameter count
- Different rank/alpha configurations
- Memory efficiency metrics
- Merge demonstration

### 5. QLoRA Example ([qlora_example.py](qlora_example.py))

Shows QLoRA: combining 4-bit quantization with LoRA for extreme efficiency.

```bash
python qlora_example.py
```

**Features:**
- 4-bit quantized base model
- FP16 LoRA adapters
- ~75% memory reduction vs FP16
- Enables fine-tuning on consumer hardware

**Output:**
- Memory comparison (FP16, 4-bit, QLoRA)
- Speed comparison
- Real-world fine-tuning scenario

### 6. Comparison ([comparison_example.py](comparison_example.py))

Comprehensive comparison of all quantization methods.

```bash
python comparison_example.py
```

**Features:**
- Side-by-side benchmarks
- Quality comparison
- Speed comparison
- Method recommendations

**Output:**
- Performance table
- Quality samples
- Use case recommendations
- M4-specific optimization guide

## Quick Start

```python
from smlx.models.SmolLM2_135M import load
from smlx.quant import quantize_model_q4_k_m, quantize_gptq, quantize_awq, apply_lora

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Option 1: Q4_K_M (RECOMMENDED - MLX-native mixed-precision) 🆕
quantize_model_q4_k_m(model)  # ~4.8 bpw, true memory savings

# Option 2: GPTQ (Hessian-based)
quantized = quantize_gptq(model, bits=4, group_size=64)

# Option 3: AWQ (activation-aware)
quantized = quantize_awq(model, bits=4, group_size=64)

# Option 4: LoRA (fine-tuning)
lora_model = apply_lora(model, rank=8, alpha=16)

# Option 5: QLoRA (quantize + fine-tune)
quantized = quantize_gptq(model, bits=4, group_size=64)
qlora_model = apply_lora(quantized, rank=8, alpha=16)
```

## Method Comparison

| Method | Compression | Quality | Speed | Use Case |
|--------|-------------|---------|-------|----------|
| **Q4_K_M** 🆕 | 6-7x | Excellent | Very Fast | **RECOMMENDED - Best overall** |
| **GPTQ 4-bit** | 4x | Good | Fast | Hessian-based optimization |
| **AWQ 4-bit** | 4x | Excellent | Fast | Activation-aware |
| **GPTQ 8-bit** | 2x | Near-lossless | Fast | Maximum quality |
| **LoRA** | 1x* | Same | Same | Fine-tuning |
| **QLoRA** | 4x* | Good | Fast | Memory-efficient fine-tuning |

*LoRA doesn't compress the base model, only adds small adapters

**Why Q4_K_M is recommended:**
- Uses MLX's native QuantizedLinear (Metal GPU kernels)
- TRUE runtime memory savings (not just storage)
- Mixed 4-bit/6-bit precision for optimal quality/size
- Better compression than GPTQ/AWQ (6-7x vs 4x)
- Quality better than uniform 4-bit (~6% error vs ~7%)

## M4 Optimization Guide

For Apple M4 Macs with unified memory:

### Default: Q4_K_M (RECOMMENDED) 🆕
```python
from smlx.quant import quantize_model_q4_k_m

quantize_model_q4_k_m(model)  # Uses MLX-native by default
```
- **Best overall choice** for M4 Macs
- Mixed 4-bit/6-bit precision (~4.8 bpw avg)
- 6-7x compression with excellent quality
- TRUE runtime memory savings
- Optimized Metal GPU kernels

### Alternative: GPTQ/AWQ 4-bit
```python
# GPTQ 4-bit (Hessian-based)
model = quantize_gptq(model, bits=4, group_size=64)

# AWQ 4-bit (activation-aware)
model = quantize_awq(model, bits=4, group_size=64)
```
- Good for specific use cases requiring Hessian optimization
- 4x compression

### Maximum Quality: GPTQ 8-bit
```python
# GPTQ 8-bit (near-lossless)
model = quantize_gptq(model, bits=8, group_size=64)
```
- Near-lossless quality
- Only 2x compression

### Fine-tuning: QLoRA
```python
# Quantize base
base = quantize_gptq(model, bits=4, group_size=64)

# Add LoRA
qlora = apply_lora(base, rank=8, alpha=16, dropout=0.05)

# Train only LoRA weights
# ... training code ...

# Merge and deploy
final = merge_lora(qlora)
```

## Configuration Guidelines

### GPTQ / AWQ

- **bits**: 4 (default), 8 (higher quality), 2 (experimental)
- **group_size**: 64 (M4 optimized), 32 (finer), 128 (coarser)

### LoRA

- **rank**: 8 (balanced), 4 (fast), 16 (high capacity)
- **alpha**: 2x rank (typical), higher for stronger adaptation
- **dropout**: 0.05 (standard), 0.1 (prevent overfitting)

## Performance Tips

1. **Use 4-bit for production** - Best quality/size tradeoff
2. **group_size=64 for M4** - Optimized for Metal acceleration
3. **Cache quantized models** - Quantize once, use many times
4. **Monitor quality** - Test on representative inputs
5. **Consider QLoRA for fine-tuning** - Enables training on consumer hardware

## Troubleshooting

**Q: Quantization is slow**
- A: Quantization is done once offline. Cache the result.

**Q: Quality degradation too high**
- A: Try AWQ instead of GPTQ, or use 8-bit instead of 4-bit

**Q: Out of memory**
- A: Use 4-bit quantization, or reduce group_size

**Q: LoRA not improving model**
- A: Increase rank, adjust alpha, train longer

## Resources

- [Quantization Documentation](../../docs/Quant.md)
- [SMLX Quant Module](../../smlx/quant/)
- [Model Implementations](../../smlx/models/)

## Citation

```bibtex
@software{smlx_quant,
  title = {SMLX: Small Models for MLX},
  author = {SMLX Contributors},
  year = {2025},
  url = {https://github.com/yourusername/smlx}
}
```
