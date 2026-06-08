# Evaluation Examples

Examples demonstrating how to evaluate models using SMLX's evaluation frameworks.

## Overview

SMLX provides comprehensive evaluation tools for:
- **Text models** (perplexity on WikiText, PTB, etc.)
- **Vision-language models** (Math-Vista, MMMU, MMStar, OCRBench)
- **Audio models** (coming soon)

## Examples

### 1. Text Model Evaluation ([text_eval_example.py](text_eval_example.py))

Evaluate language models on standard benchmarks using perplexity.

```bash
python text_eval_example.py
```

**What it demonstrates:**
- Loading evaluation datasets (WikiText, PTB)
- Calculating perplexity
- Measuring tokens/second
- Understanding evaluation metrics

**Output:**
- Perplexity score
- Throughput metrics
- Sample predictions

### 2. Vision-Language Evaluation ([vlm_eval_example.py](vlm_eval_example.py))

Demonstrates VLM evaluation utilities (requires VLM models when implemented).

```bash
python vlm_eval_example.py
```

**What it demonstrates:**
- Math-Vista: Math reasoning with vision
- MMMU: Multimodal understanding
- MMStar: Multimodal reasoning
- OCRBench: OCR capabilities

**Output:**
- Utility function demos
- Answer normalization examples
- Evaluation workflow

## Quick Start

### Evaluate a Text Model

```python
from smlx.models import load
from smlx.evals.text_eval import evaluate_perplexity

# Load model
bm = load("smollm2-135m")
model, tokenizer = bm.model, bm.processor

# Evaluate
results = evaluate_perplexity(
    model=model,
    tokenizer=tokenizer,
    dataset_name="wikitext",
    split="test",
)

print(f"Perplexity: {results['perplexity']:.2f}")
```

### Evaluate a VLM (When Available)

```python
from smlx.models import load
from smlx.evals.math_vista import evaluate_math_vista

# Load VLM
bm = load("smolvlm-256m")
model, processor = bm.model, bm.processor

# Evaluate
results = evaluate_math_vista(
    model=model,
    processor=processor,
    split="test",
)

print(f"Accuracy: {results['accuracy']:.2f}%")
```

## Available Benchmarks

### Text Benchmarks

| Dataset | Size | Metric | Description |
|---------|------|--------|-------------|
| **WikiText-2** | 2.1M tokens | Perplexity | General text |
| **WikiText-103** | 100M tokens | Perplexity | Large-scale text |
| **PTB** | 1M tokens | Perplexity | Penn Treebank |

### Vision-Language Benchmarks

| Benchmark | Examples | Type | Coverage |
|-----------|----------|------|----------|
| **Math-Vista** | 6,141 | Math reasoning | 28 datasets |
| **MMMU** | - | Understanding | 30+ subjects |
| **MMStar** | - | Reasoning | 6 categories |
| **OCRBench** | - | OCR | Text recognition |

## Evaluation Metrics

### Perplexity

Measures how well a language model predicts text:
- **Lower is better**
- PPL = exp(average negative log likelihood)
- Common ranges:
  - 20-30: Excellent for small models
  - 30-50: Good
  - 50+: Needs improvement

```python
results = evaluate_perplexity(model, tokenizer, "wikitext")
print(f"Perplexity: {results['perplexity']:.2f}")
```

### Accuracy

For VLM benchmarks (multiple choice, exact match):
- **Higher is better**
- Range: 0-100%
- Compare against baselines

```python
results = evaluate_math_vista(model, processor)
print(f"Accuracy: {results['accuracy']:.2f}%")
```

## Best Practices

### 1. Use Consistent Settings

```python
# Always use the same settings for fair comparison
evaluate_perplexity(
    model=model,
    tokenizer=tokenizer,
    dataset_name="wikitext",
    split="test",  # Not "validation"
    batch_size=4,
    seed=42,  # For reproducibility
)
```

### 2. Warm Up the Model

```python
# Run a few warmup iterations
for _ in range(3):
    model(mx.zeros((1, 10)))

# Then evaluate
results = evaluate_perplexity(...)
```

### 3. Track Across Training

```python
# Evaluate at checkpoints
checkpoints = [1000, 5000, 10000]

for step in checkpoints:
    model = load_checkpoint(step)
    ppl = evaluate_perplexity(model, tokenizer, "wikitext")
    print(f"Step {step}: PPL = {ppl['perplexity']:.2f}")
```

### 4. Compare Quantized vs Full Precision

```python
from smlx.quant import quantize_gptq

# Full precision
fp16_results = evaluate_perplexity(model, tokenizer, "wikitext")

# Quantized
q_model = quantize_gptq(model, bits=4)
q_results = evaluate_perplexity(q_model, tokenizer, "wikitext")

print(f"FP16 PPL: {fp16_results['perplexity']:.2f}")
print(f"4-bit PPL: {q_results['perplexity']:.2f}")
print(f"Degradation: {q_results['perplexity'] - fp16_results['perplexity']:.2f}")
```

## Advanced Usage

### Custom Dataset

```python
from smlx.evals.text_eval import load_eval_dataset

# Load your own dataset
dataset = load_eval_dataset(
    "custom",
    split="test",
    data_path="/path/to/data.txt",
)

# Evaluate
results = evaluate_perplexity(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,  # Pass dataset directly
)
```

### Batch Evaluation

```python
# Evaluate multiple models
models = {
    "SmolLM2-135M": load("SmolLM2-135M-Instruct"),
    "SmolLM2-360M": load("SmolLM2-360M-Instruct"),
}

for name, (model, tokenizer) in models.items():
    results = evaluate_perplexity(model, tokenizer, "wikitext")
    print(f"{name}: PPL = {results['perplexity']:.2f}")
```

### Statistical Significance

```python
# Run multiple seeds for confidence intervals
ppls = []

for seed in range(5):
    results = evaluate_perplexity(
        model, tokenizer, "wikitext",
        seed=seed,
    )
    ppls.append(results['perplexity'])

import numpy as np
print(f"Mean PPL: {np.mean(ppls):.2f} ± {np.std(ppls):.2f}")
```

## Troubleshooting

**Q: Evaluation is slow**
- Increase batch_size (but watch memory)
- Use num_samples to evaluate on subset
- Consider using quantized model

**Q: Perplexity is very high**
- Check if model is loaded correctly
- Verify tokenizer matches model
- Ensure model is in eval mode

**Q: VLM evaluation fails**
- VLM models not yet implemented
- Check that image preprocessing is correct
- Verify processor matches model

## Resources

- [Text Evaluation Module](../../smlx/evals/text_eval.py)
- [VLM Evaluation Modules](../../smlx/evals/)
- [Evaluation Documentation](../../docs/EVALUATION.md)

## Citation

```bibtex
@software{smlx_eval,
  title = {SMLX Evaluation Frameworks},
  author = {SMLX Contributors},
  year = {2025},
  url = {https://github.com/yourusername/smlx}
}
```
