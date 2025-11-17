# SMLX Evaluation Guide

This guide covers the evaluation infrastructure in SMLX, including vision-language model benchmarks, usage instructions, and best practices.

## Overview

SMLX provides production-ready implementations of four major vision-language model (VLM) benchmarks:

| Benchmark | Task Type | Samples | Metrics | Reference |
|-----------|-----------|---------|---------|-----------|
| **MathVista** | Math reasoning in visual contexts | 6,141 | Accuracy by category | [ICLR 2024](https://arxiv.org/abs/2310.02255) |
| **MMMU** | Multi-discipline multimodal understanding | 11.5K | Accuracy by subject | [CVPR 2024](https://arxiv.org/abs/2311.16502) |
| **MMStar** | Vision-dependent multimodal reasoning | 1,500 | Accuracy by capability | [NeurIPS 2024](https://github.com/MMStar-Benchmark/MMStar) |
| **OCRBench** | OCR and text understanding | 1,000 | Accuracy by task type | [Paper](https://arxiv.org/abs/2305.07895) |

## Quick Start

### Installation

Install SMLX with evaluation dependencies:

```bash
pip install -e ".[evals]"
```

This installs: `mlx-vlm`, `datasets`, `tqdm`, `Pillow`, and `huggingface_hub`.

### Running Your First Evaluation

```bash
# Download a small model (if not already cached)
python -m smlx.tools.download_data --model mlx-community/SmolVLM-256M-Instruct

# Run MathVista on 10 samples (quick test)
python -m smlx.evals.math_vista \
    --model mlx-community/SmolVLM-256M-Instruct \
    --max-samples 10 \
    --verbose

# Results will be saved to:
# - results/math_vista_predictions.csv (detailed predictions)
# - results/math_vista_summary.json (overall metrics)
```

## Benchmark Guides

### MathVista - Mathematical Reasoning in Visual Contexts

**Dataset:** AI4Math/MathVista (6,141 examples from 28 source datasets)

**Splits:**
- `testmini` (1,000 samples, recommended for development)
- `test` (5,141 samples, ground truth not public)

**Question Types:**
- Multiple choice (options A-E)
- Free-form (integer, float, list)

**Categories:** Evaluated across multiple dimensions including question type, skill required, and visual context.

#### Usage

```bash
# Basic evaluation
python -m smlx.evals.math_vista \
    --model mlx-community/SmolVLM-256M-Instruct \
    --split testmini

# With custom parameters
python -m smlx.evals.math_vista \
    --model mlx-community/SmolVLM-500M-Instruct \
    --split testmini \
    --max-samples 100 \
    --temperature 0.0 \
    --max-tokens 512 \
    --output results/mathvista_500m.csv \
    --verbose

# Quick test on 50 samples
python -m smlx.evals.math_vista \
    --model mlx-community/SmolVLM-256M-Instruct \
    --max-samples 50
```

#### Expected Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| SmolVLM-256M | ~25-30% | Baseline small model |
| SmolVLM-500M | ~30-35% | Improved with more parameters |
| GPT-4V | ~56% | State-of-the-art (reference) |
| Human (HS diploma) | ~60% | Average human performance |

**Runtime:** ~5-10 seconds per sample (M4 Mac, depending on model size)

#### Output Files

- **CSV:** Detailed predictions with columns: `question_id`, `question`, `prediction`, `answer`, `correct`, `category_*`
- **JSON:** Summary with overall accuracy and category-wise breakdown

---

### MMMU - Massive Multi-discipline Multimodal Understanding

**Dataset:** MMMU/MMMU (11,500 questions across 30 subjects)

**Splits:**
- `dev` (150 samples, validation)
- `validation` (900 samples, recommended)
- `test` (10,500 samples, ground truth not public)

**Disciplines:** Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering

**Subjects:** 30 subjects including Accounting, Agriculture, Architecture, Art, Biology, Chemistry, Computer Science, Economics, Electronics, Energy, Finance, Geography, History, Law, Literature, Manage, Marketing, Materials, Math, Mechanical Engineering, Music, Philosophy, Physics, Psychology, Public Health, Sociology, etc.

#### Usage

```bash
# List all available subjects
python -m smlx.evals.mmmu --list-subjects

# Evaluate on a specific subject
python -m smlx.evals.mmmu \
    --model mlx-community/SmolVLM-256M-Instruct \
    --subset Accounting \
    --split validation

# Evaluate on all subjects (validation split)
python -m smlx.evals.mmmu \
    --model mlx-community/SmolVLM-500M-Instruct \
    --split validation

# Evaluate with sample limit
python -m smlx.evals.mmmu \
    --model mlx-community/SmolVLM-256M-Instruct \
    --max-samples 50 \
    --output results/mmmu_256m.csv

# Evaluation-only mode (if you already have predictions)
python -m smlx.evals.mmmu \
    --prediction-file results/mmmu_predictions.csv
```

#### Expected Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| SmolVLM-256M | ~30-35% | Baseline small model |
| SmolVLM-500M | ~35-40% | Better subject knowledge |
| GPT-4V | ~56% | State-of-the-art (reference) |
| Gemini Ultra | ~59% | Best reported performance |

**Runtime:** ~8-15 seconds per sample (may have multiple images)

#### Output Files

- **CSV:** Predictions with `id`, `question`, `prediction`, `answer`, `score`, `subject`
- **JSON:** Summary with overall accuracy and per-subject breakdown

---

### MMStar - Vision-Dependent Multimodal Reasoning

**Dataset:** Lin-Chen/MMStar (1,500 carefully curated samples)

**Split:** `val` (all 1,500 samples)

**Core Capabilities:**
1. Coarse Perception (CP)
2. Fine-grained Perception (FP)
3. Instance Reasoning (IR)
4. Logical Reasoning (LR)
5. Science & Technology (ST)
6. Mathematics (MA)

**Subcategories:** 18 detailed axes balanced across capabilities

#### Usage

```bash
# Full evaluation
python -m smlx.evals.mmstar \
    --model mlx-community/SmolVLM-256M-Instruct

# With custom parameters
python -m smlx.evals.mmstar \
    --model mlx-community/SmolVLM-500M-Instruct \
    --temperature 0.0 \
    --max-tokens 256 \
    --output results/mmstar_500m.csv \
    --verbose

# Quick test
python -m smlx.evals.mmstar \
    --model mlx-community/SmolVLM-256M-Instruct \
    --max-samples 100
```

#### Expected Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| SmolVLM-256M | ~30-35% | Baseline small model |
| SmolVLM-500M | ~35-42% | Better reasoning |
| GPT-4V | ~56-63% | State-of-the-art (reference) |

**Runtime:** ~6-10 seconds per sample

#### Output Files

- **CSV:** Predictions with scores for each capability and subcategory
- **JSON:** Hierarchical breakdown (overall → capability → subcategory)

---

### OCRBench - OCR and Document Understanding

**Dataset:** echo840/OCRBench (1,000 samples from 29 datasets)

**Split:** `test` (all 1,000 samples)

**Task Types:**
1. Text Recognition (scene text, handwritten, artistic, etc.)
2. Scene Text VQA (visual question answering)
3. Document-Oriented VQA
4. Key Information Extraction (KIE)
5. Handwritten Mathematical Expression Recognition (HMER)

#### Usage

```bash
# Full evaluation
python -m smlx.evals.ocrbench \
    --model mlx-community/SmolVLM-256M-Instruct

# With custom parameters
python -m smlx.evals.ocrbench \
    --model mlx-community/SmolVLM-500M-Instruct \
    --temperature 0.0 \
    --output results/ocrbench_500m.csv \
    --verbose

# Evaluation-only mode
python -m smlx.evals.ocrbench \
    --predictions-file results/ocrbench_predictions.csv
```

#### Expected Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| SmolVLM-256M | ~25-32% | Baseline small model |
| SmolVLM-500M | ~32-40% | Better text recognition |
| GPT-4V | ~60-70% | State-of-the-art (reference) |

**Runtime:** ~5-8 seconds per sample

#### Output Files

- **CSV:** Predictions with `sample_id`, `question`, `prediction`, `answer`, `correct`, `type`, `dataset`
- **JSON:** Summary with overall accuracy and task-type breakdown

---

## Programmatic Usage

All evaluation modules can be used programmatically:

```python
from smlx.evals import inference, load_model
from PIL import Image

# Load model once
model, processor = load_model("mlx-community/SmolVLM-256M-Instruct")

# Run inference on a single image
image = Image.open("path/to/image.jpg")
prompt = "What is shown in this image?"

response = inference(
    model=model,
    processor=processor,
    image=image,  # or list of images for multi-image
    prompt=prompt,
    temperature=0.0,
    max_tokens=256,
    verbose=True
)

print(response)
```

For batch evaluation, use the evaluation modules directly:

```python
from smlx.evals.math_vista import main as run_mathvista
from smlx.evals.mmmu import main as run_mmmu
import sys

# Configure arguments
sys.argv = [
    'script',
    '--model', 'mlx-community/SmolVLM-256M-Instruct',
    '--max-samples', '100',
    '--output', 'my_results.csv'
]

# Run evaluation
run_mathvista()
```

---

## Text-Only Model Evaluation

SMLX provides comprehensive evaluation and benchmarking tools for text-only language models like SmolLM2-135M and SmolLM2-360M.

### Overview

| Tool | Purpose | Key Metrics | Reference |
|------|---------|-------------|-----------|
| **Perplexity Evaluation** | Language model quality on standard datasets | Perplexity, std error, tokens/sec | WikiText, PTB, OpenWebText |
| **Text Generation Benchmarks** | Performance characteristics | Context scaling, generation speed, TTFT | Custom prompts |
| **Quantization Comparison** | Memory/speed tradeoffs | Model size, speedup, memory savings | FP16/8-bit/4-bit |
| **Unified Benchmark CLI** | Run all benchmark suites | Comprehensive performance reports | Multiple suites |

### Perplexity Evaluation

**Module:** `smlx/evals/text_eval.py`

Perplexity is the standard metric for evaluating language model quality. Lower perplexity indicates better modeling of the language distribution.

#### Supported Datasets

- **WikiText-2** (wikitext) - 2M tokens, standard benchmark
- **WikiText-103** (wikitext103) - 103M tokens, larger benchmark
- **Penn Treebank** (ptb) - Financial news corpus
- **OpenWebText** (openwebtext) - Web text corpus

#### Basic Usage

```bash
# Evaluate on WikiText-2 (default)
python -m smlx.evals.text_eval \
    --model mlx-community/SmolLM2-135M-Instruct \
    --dataset wikitext \
    --split test

# Evaluate on WikiText-103
python -m smlx.evals.text_eval \
    --model mlx-community/SmolLM2-135M-Instruct \
    --dataset wikitext103 \
    --num-samples 1000

# Quick test with 100 samples
python -m smlx.evals.text_eval \
    --model mlx-community/SmolLM2-135M-Instruct \
    --num-samples 100 \
    --verbose

# Custom configuration
python -m smlx.evals.text_eval \
    --model mlx-community/SmolLM2-135M-Instruct \
    --dataset wikitext \
    --batch-size 16 \
    --sequence-length 512 \
    --output results/perplexity.json
```

#### Expected Performance

| Model | WikiText-2 PPL | WikiText-103 PPL | Notes |
|-------|----------------|------------------|-------|
| SmolLM2-135M | ~15-20 | ~18-25 | Baseline small model |
| SmolLM2-360M | ~12-18 | ~15-22 | Better with more parameters |
| Llama-2-7B | ~5-8 | ~7-10 | Reference (much larger) |

**Runtime:** ~5-15 seconds per 100 tokens (M4 Mac, batch_size=8)

#### Programmatic Usage

```python
from smlx.evals.text_eval import evaluate_perplexity
from smlx.models.SmolLM2_135M import load

# Load model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Evaluate
results = evaluate_perplexity(
    model=model,
    tokenizer=tokenizer,
    dataset="wikitext",
    split="test",
    batch_size=8,
    num_samples=1000,
    verbose=True,
)

print(f"Perplexity: {results['perplexity']:.2f}")
print(f"Std Error: {results['std_error']:.2f}")
print(f"Tokens/sec: {results['tokens_per_second']:.0f}")
print(f"Peak Memory: {results['peak_memory_gb']:.2f} GB")
```

#### Output Metrics

The evaluation returns a dictionary with:
- `perplexity` - exp(mean_loss), lower is better
- `std_error` - Standard error of perplexity
- `mean_loss` - Average cross-entropy loss
- `std_dev` - Standard deviation of loss
- `tokens_evaluated` - Total number of tokens evaluated
- `eval_time` - Total evaluation time (seconds)
- `tokens_per_second` - Throughput metric
- `peak_memory_gb` - Peak memory usage

---

### Text Generation Benchmarks

**Module:** `smlx/bench/suites/text_generation.py`

Comprehensive benchmarking suite for analyzing text generation performance characteristics.

#### Benchmark Types

1. **Context Scaling** - How context length affects performance
2. **Generation Length** - Impact of output length on speed
3. **Temperature Effects** - How sampling temperature affects generation

#### Usage

```bash
# Run comprehensive suite (all benchmarks)
python -m smlx.bench.run text_generation \
    --model mlx-community/SmolLM2-135M-Instruct \
    --verbose

# Save results to JSON
python -m smlx.bench.run text_generation \
    --model mlx-community/SmolLM2-135M-Instruct \
    --output results/text_gen_benchmarks.json

# Run with custom parameters
python -m smlx.bench.run text_generation \
    --model mlx-community/SmolLM2-135M-Instruct \
    --prompt "Once upon a time in a distant galaxy" \
    --generation-tokens 200
```

#### Programmatic Usage

```python
from smlx.bench.suites.text_generation import (
    benchmark_context_scaling,
    benchmark_generation_length,
    benchmark_temperature_effects,
    run_comprehensive_suite,
)
from smlx.models.SmolLM2_135M import load

# Load model once
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Run context scaling benchmark
context_results = benchmark_context_scaling(
    model=model,
    tokenizer=tokenizer,
    context_lengths=[128, 256, 512, 1024, 2048],
    generation_tokens=50,
    verbose=True,
)

# Run generation length benchmark
length_results = benchmark_generation_length(
    model=model,
    tokenizer=tokenizer,
    generation_lengths=[32, 64, 128, 256, 512],
    verbose=True,
)

# Run temperature benchmark
temp_results = benchmark_temperature_effects(
    model=model,
    tokenizer=tokenizer,
    temperatures=[0.0, 0.3, 0.5, 0.7, 1.0],
    verbose=True,
)

# Or run all benchmarks at once
all_results = run_comprehensive_suite(
    model_path="mlx-community/SmolLM2-135M-Instruct",
    verbose=True,
)
```

#### Expected Performance (SmolLM2-135M on M4 Mac)

**Context Scaling:**
| Context Length | Prompt TPS | Generation TPS | Memory (GB) |
|----------------|------------|----------------|-------------|
| 128 tokens | ~800-1200 | ~40-60 | ~1.2 |
| 512 tokens | ~600-900 | ~35-50 | ~1.5 |
| 1024 tokens | ~400-600 | ~30-45 | ~2.0 |
| 2048 tokens | ~200-400 | ~25-40 | ~3.0 |

**Generation Length:**
| Output Tokens | Generation TPS | Total Time | Notes |
|---------------|----------------|------------|-------|
| 32 tokens | ~45-55 | ~0.6s | Quick responses |
| 128 tokens | ~40-50 | ~2.5s | Medium responses |
| 512 tokens | ~35-45 | ~12s | Long-form content |

**Temperature Effects:**
| Temperature | Generation TPS | Output Diversity |
|-------------|----------------|------------------|
| 0.0 | ~45-55 | Deterministic (greedy) |
| 0.3 | ~40-50 | Slightly varied |
| 0.7 | ~38-48 | Creative |
| 1.0 | ~35-45 | Very diverse |

#### Metrics Collected

Each benchmark result includes:
- `prompt_tokens` - Input token count
- `prompt_time` - Time to process prompt (seconds)
- `prompt_tps` - Prompt processing throughput (tokens/sec)
- `generation_tokens` - Output token count
- `generation_time` - Time to generate output (seconds)
- `generation_tps` - Generation throughput (tokens/sec)
- `time_to_first_token` - TTFT latency (seconds)
- `total_time` - End-to-end time (seconds)
- `peak_memory_gb` - Peak memory usage (GB)
- `active_memory_gb` - Active memory usage (GB)

---

### Quantization Comparison

**Module:** `smlx/bench/suites/quantization.py`

Compare different quantization methods to understand memory/speed/quality tradeoffs.

#### Supported Methods

- **FP16** (fp16) - Baseline 16-bit floating point
- **8-bit** (8bit) - 8-bit quantization (~50% memory reduction)
- **4-bit** (4bit) - 4-bit quantization (~75% memory reduction)
- **GPTQ** (gptq) - Advanced 4-bit quantization
- **AWQ** (awq) - Activation-aware quantization
- **DWQ** (dwq) - Dynamic weight quantization

#### Usage

```bash
# Compare FP16 and 4-bit
python -m smlx.bench.run quantization \
    --model mlx-community/SmolLM2-135M-Instruct \
    --methods fp16 4bit

# Compare all methods
python -m smlx.bench.run quantization \
    --model mlx-community/SmolLM2-135M-Instruct \
    --methods fp16 8bit 4bit gptq awq dwq \
    --output results/quantization_comparison.json

# Custom test configuration
python -m smlx.bench.run quantization \
    --model mlx-community/SmolLM2-135M-Instruct \
    --methods fp16 4bit \
    --prompt "The future of AI is" \
    --generation-tokens 200
```

#### Expected Results (SmolLM2-135M)

| Method | Model Size | Memory Reduction | Generation TPS | Speedup | Quality Impact |
|--------|------------|------------------|----------------|---------|----------------|
| FP16 | 0.27 GB | Baseline (0%) | 45-55 tok/s | 1.0x | Best |
| 8-bit | 0.14 GB | 50% | 50-60 tok/s | 1.1x | Minimal |
| 4-bit | 0.07 GB | 75% | 55-65 tok/s | 1.2x | Minor |
| GPTQ | 0.07 GB | 75% | 50-60 tok/s | 1.1x | Minimal |
| AWQ | 0.07 GB | 75% | 55-65 tok/s | 1.2x | Minimal |

**Key Insights:**
- 4-bit quantization provides best memory/speed tradeoff
- Speed improvements due to reduced memory bandwidth
- Quality degradation is minimal for "smol" models
- Larger models benefit more from quantization

#### Programmatic Usage

```python
from smlx.bench.suites.quantization import compare_quantization_methods

results = compare_quantization_methods(
    model_path="mlx-community/SmolLM2-135M-Instruct",
    quantization_methods=["fp16", "8bit", "4bit"],
    test_prompt="The quick brown fox jumps over the lazy dog",
    generation_tokens=100,
    verbose=True,
)

# Access results
for method, result in results.items():
    print(f"\n{method}:")
    print(f"  Model size: {result.model_size_gb:.2f} GB")
    print(f"  Generation speed: {result.generation_tps:.0f} tok/s")
    print(f"  Speedup: {result.generation_speedup:.2f}x")
    print(f"  Memory savings: {result.memory_savings_percent:.1f}%")
```

---

### Unified Benchmark CLI

**Module:** `smlx/bench/run.py`

Unified CLI for running all benchmark suites from a single interface.

#### List Available Suites

```bash
# List all available benchmark suites
python -m smlx.bench.run --list
```

Output:
```
Available Benchmark Suites:
- text_generation: Text generation benchmarks (context scaling, generation length, temperature)
- quantization: Compare FP16, 8-bit, 4-bit quantization methods
- llm: Basic LLM performance benchmarks
- ops: Low-level operation benchmarks (matmul, attention)
```

#### Run Individual Suites

```bash
# Run text generation benchmarks
python -m smlx.bench.run text_generation \
    --model mlx-community/SmolLM2-135M-Instruct

# Run quantization comparison
python -m smlx.bench.run quantization \
    --model mlx-community/SmolLM2-135M-Instruct \
    --methods fp16 8bit 4bit

# Run LLM benchmarks
python -m smlx.bench.run llm \
    --model mlx-community/SmolLM2-135M-Instruct

# Run with output file
python -m smlx.bench.run text_generation \
    --model mlx-community/SmolLM2-135M-Instruct \
    --output results/benchmarks.json
```

#### Run All Suites

```bash
# Run all applicable benchmarks
python -m smlx.bench.run --all \
    --model mlx-community/SmolLM2-135M-Instruct \
    --output-dir results/

# This will create:
# - results/text_generation_results.json
# - results/quantization_results.json
# - results/llm_results.json
```

#### Quiet Mode

```bash
# Run without verbose output
python -m smlx.bench.run text_generation \
    --model mlx-community/SmolLM2-135M-Instruct \
    --quiet
```

---

### Text Evaluation Testing

Run text evaluation tests:

```bash
# All text evaluation tests
pytest tests/evals/test_text_eval.py -v

# Specific test classes
pytest tests/evals/test_text_eval.py::TestPerplexityCalculation -v
pytest tests/evals/test_text_eval.py::TestDatasetLoading -v

# Skip slow tests
pytest tests/evals/test_text_eval.py -m "not slow"

# Run integration tests (requires model download)
SMLX_DOWNLOAD_TEST_MODELS=1 pytest tests/evals/test_text_eval.py::TestFullEvaluation -v
```

---

### Text Evaluation Best Practices

#### Quick Development Testing

```bash
# Start with small sample size
python -m smlx.evals.text_eval \
    --model mlx-community/SmolLM2-135M-Instruct \
    --num-samples 100

# Then run benchmarks with defaults
python -m smlx.bench.run text_generation \
    --model mlx-community/SmolLM2-135M-Instruct
```

#### Production Evaluation

```bash
# Full perplexity evaluation
python -m smlx.evals.text_eval \
    --model mlx-community/SmolLM2-135M-Instruct \
    --dataset wikitext \
    --split test \
    --batch-size 16 \
    --num-samples -1  # Evaluate all samples

# Comprehensive benchmarks with results
python -m smlx.bench.run --all \
    --model mlx-community/SmolLM2-135M-Instruct \
    --output-dir results/$(date +%Y-%m-%d)/
```

#### Comparing Models

```bash
# Create organized comparison
mkdir -p results/model_comparison

# Evaluate SmolLM2-135M
python -m smlx.bench.run --all \
    --model mlx-community/SmolLM2-135M-Instruct \
    --output-dir results/model_comparison/smollm2-135m/

# Evaluate SmolLM2-360M
python -m smlx.bench.run --all \
    --model mlx-community/SmolLM2-360M-Instruct \
    --output-dir results/model_comparison/smollm2-360m/

# Compare results with analysis tools (coming soon)
python -m smlx.tools.compare_results \
    results/model_comparison/smollm2-135m/ \
    results/model_comparison/smollm2-360m/
```

---

### LM Evaluation Harness Integration

**Module:** `smlx/evals/lm_harness.py`

SMLX integrates with EleutherAI's lm-evaluation-harness, providing access to 200+ standardized evaluation tasks for language models.

#### Overview

The lm-evaluation-harness is a comprehensive framework for evaluating language models on a wide range of tasks:

- **Common Sense Reasoning:** HellaSwag, Winogrande, ARC, PIQA
- **Knowledge & QA:** MMLU, TruthfulQA, TriviaQA, NaturalQuestions
- **Math & Coding:** GSM8K, MATH, HumanEval
- **Reasoning:** BoolQ, LogiQA, OpenBookQA
- **Multilingual:** Many tasks available in multiple languages

#### Installation

The lm-evaluation-harness requires additional dependencies:

```bash
pip install lm-eval>=0.4.0
```

#### Popular Tasks

| Task | Category | Description | Samples | Metric |
|------|----------|-------------|---------|--------|
| **hellaswag** | Common Sense | Sentence completion | 10,042 | Accuracy |
| **winogrande** | Common Sense | Pronoun resolution | 1,267 | Accuracy |
| **arc_easy** | Knowledge | Grade-school science | 2,376 | Accuracy |
| **arc_challenge** | Knowledge | Difficult science questions | 1,172 | Accuracy |
| **mmlu** | Knowledge | Massive multitask language understanding | 14,042 | Accuracy |
| **truthfulqa_mc2** | Truthfulness | Truthful question answering | 817 | Accuracy |
| **gsm8k** | Math | Grade school math word problems | 1,319 | Exact Match |

#### List Available Tasks

```bash
# Show popular evaluation tasks
python -m smlx.evals.lm_harness --list-tasks

# This will display:
# - Task names
# - Categories
# - Descriptions
# - Recommended few-shot settings
```

#### Basic Usage

```bash
# Evaluate on HellaSwag
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag

# Evaluate on multiple tasks
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag winogrande arc_easy

# With few-shot examples
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --num-fewshot 10

# Quick test with sample limit
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --limit 100 \
    --verbose

# Save results to file
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag winogrande \
    --output results/lm_harness_results.json
```

#### Programmatic Usage

```python
from smlx.evals.lm_harness import run_evaluation

# Run evaluation programmatically
results = run_evaluation(
    model_path="mlx-community/SmolLM2-135M-Instruct",
    tasks=["hellaswag", "winogrande"],
    num_fewshot=0,  # Zero-shot evaluation
    limit=None,  # Evaluate all samples
    batch_size=8,
    seed=42,
    verbose=True,
)

# Access results
print(f"HellaSwag: {results['results']['hellaswag']['acc']:.3f}")
print(f"Winogrande: {results['results']['winogrande']['acc']:.3f}")
```

#### Expected Performance (SmolLM2-135M)

Zero-shot performance on popular benchmarks:

| Task | SmolLM2-135M | SmolLM2-360M | Llama-2-7B | Notes |
|------|--------------|--------------|------------|-------|
| **HellaSwag** | 0.30-0.35 | 0.35-0.42 | 0.76 | Common sense reasoning |
| **Winogrande** | 0.52-0.55 | 0.55-0.60 | 0.69 | Pronoun resolution |
| **ARC-Easy** | 0.45-0.52 | 0.52-0.60 | 0.78 | Grade-school science |
| **ARC-Challenge** | 0.25-0.30 | 0.30-0.35 | 0.53 | Difficult science |
| **MMLU** | 0.25-0.28 | 0.28-0.32 | 0.46 | Multitask understanding |
| **TruthfulQA** | 0.35-0.40 | 0.40-0.45 | 0.38 | Truthfulness |
| **GSM8K** | 0.05-0.10 | 0.10-0.15 | 0.14 | Math word problems |

**Performance Notes:**
- Small models struggle with complex reasoning tasks
- Few-shot prompting can improve scores by 5-15%
- SmolLM2 models are optimized for efficient inference, not benchmark scores
- Best use cases: simple QA, classification, text completion

#### Advanced Usage

**Custom Batch Size:**
```bash
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --batch-size 16  # Increase for faster evaluation
```

**Seeded Evaluation (Reproducibility):**
```bash
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --seed 42 \
    --limit 100
```

**Multiple Tasks with Different Settings:**
```bash
# Evaluate on a suite of tasks
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag,winogrande,arc_easy,arc_challenge,mmlu \
    --output results/comprehensive_eval.json
```

#### Integration with Custom Models

The `SMLXLM` wrapper can be used with any SMLX model that implements the standard interface:

```python
from smlx.evals.lm_harness import SMLXLM
from lm_eval import simple_evaluate

# Load your custom SMLX model
lm = SMLXLM(
    model_path="path/to/your/model",
    batch_size=8,
    max_tokens=2048,
)

# Run evaluation
results = simple_evaluate(
    model=lm,
    tasks=["hellaswag"],
    num_fewshot=10,
    batch_size=8,
)

print(results["results"])
```

#### Understanding the Results

The evaluation returns a detailed results dictionary:

```python
{
    "results": {
        "hellaswag": {
            "acc": 0.3245,           # Accuracy
            "acc_stderr": 0.0147,    # Standard error
            "acc_norm": 0.3512,      # Normalized accuracy
            "acc_norm_stderr": 0.0150
        }
    },
    "config": {
        "model": "smlx",
        "batch_size": 8,
        "num_fewshot": 0
    }
}
```

**Key Metrics:**
- `acc` - Overall accuracy on the task
- `acc_stderr` - Standard error of the accuracy
- `acc_norm` - Length-normalized accuracy (for multiple choice)
- Exact match metrics for generation tasks (GSM8K, etc.)

#### Testing

Run LM harness integration tests:

```bash
# All LM harness tests
pytest tests/evals/test_lm_harness.py -v

# Specific test classes
pytest tests/evals/test_lm_harness.py::TestLoglikelihood -v
pytest tests/evals/test_lm_harness.py::TestGeneration -v

# Skip slow tests
pytest tests/evals/test_lm_harness.py -m "not slow"

# Run integration tests (requires model download)
SMLX_DOWNLOAD_TEST_MODELS=1 pytest tests/evals/test_lm_harness.py::TestFullEvaluation -v
```

#### Best Practices

**1. Start with Quick Tests:**
```bash
# Test with 10 samples first
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --limit 10
```

**2. Use Zero-Shot for Baselines:**
```bash
# Zero-shot is faster and often sufficient for small models
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --num-fewshot 0
```

**3. Batch Multiple Tasks:**
```bash
# More efficient to run multiple tasks together
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag,winogrande,arc_easy \
    --output results/multi_task_eval.json
```

**4. Save Results:**
```bash
# Always save results for comparison and analysis
mkdir -p results/lm_harness/$(date +%Y-%m-%d)
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --output results/lm_harness/$(date +%Y-%m-%d)/hellaswag.json
```

#### Comparison with Other Benchmarks

| Evaluation Type | Coverage | Speed | Use Case |
|----------------|----------|-------|----------|
| **Perplexity** | Language modeling quality | Fast | Model development |
| **Text Generation** | Generation characteristics | Fast | Performance tuning |
| **LM Harness** | Reasoning & knowledge | Slow | Comprehensive eval |
| **Quantization** | Memory/speed tradeoffs | Medium | Deployment decisions |

**Recommendation:** Use perplexity and generation benchmarks during development, then run LM harness for final comprehensive evaluation.

#### Troubleshooting

**1. lm-eval Not Installed:**
```bash
pip install lm-eval>=0.4.0
```

**2. Model Loading Errors:**
Ensure your model path is correct and the model is downloaded:
```bash
python -m smlx.tools.download_data --model mlx-community/SmolLM2-135M-Instruct
```

**3. Memory Issues:**
Reduce batch size:
```bash
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --batch-size 2
```

**4. Slow Evaluation:**
Use `--limit` to test on fewer samples during development:
```bash
python -m smlx.evals.lm_harness \
    --model mlx-community/SmolLM2-135M-Instruct \
    --tasks hellaswag \
    --limit 100
```

---

## Resource Requirements

### Memory Usage

| Model Size | MLX Memory | Recommended RAM |
|------------|------------|-----------------|
| 256M params | ~1-2 GB | 8 GB |
| 500M params | ~2-3 GB | 16 GB |
| 1B+ params | ~4-8 GB | 32 GB |

**Note:** MLX uses Apple's unified memory efficiently. M4 Macs with 36GB work well for all "smol" models.

### Disk Space

- **Models:** 0.5-2 GB per model (cached in `~/.cache/huggingface/`)
- **Datasets:**
  - MathVista: ~500 MB (with images)
  - MMMU: ~2-3 GB (multi-image, 30 subjects)
  - MMStar: ~300 MB
  - OCRBench: ~600 MB

**Total:** ~5-10 GB for all datasets and a few models

### Evaluation Time Estimates

On M4 Mac with SmolVLM-256M:

| Benchmark | Samples | Estimated Time |
|-----------|---------|----------------|
| MathVista (testmini) | 1,000 | ~1.5-2 hours |
| MMMU (validation) | 900 | ~2-3 hours |
| MMStar (val) | 1,500 | ~2.5-3.5 hours |
| OCRBench (test) | 1,000 | ~1.5-2 hours |

**Tip:** Use `--max-samples` for quick testing during development.

---

## Downloading Data

### Using the CLI Tool

```bash
# Download everything
python -m smlx.tools.download_data --all

# Download specific items
python -m smlx.tools.download_data \
    --model mlx-community/SmolVLM-256M-Instruct \
    --dataset AI4Math/MathVista

# Download all models and datasets
python -m smlx.tools.download_data --models --datasets

# List available shortcuts
python -m smlx.tools.download_data --list

# Custom cache directory
python -m smlx.tools.download_data --cache-dir /path/to/cache --models
```

### Manual Download

```python
from smlx.tools.download import download_model, download_dataset

# Download a model
download_model("mlx-community/SmolVLM-256M-Instruct")

# Download a dataset
download_dataset("AI4Math/MathVista", split="testmini")
```

### Dataset Locations

All datasets are cached via Hugging Face:
- Default cache: `~/.cache/huggingface/hub/`
- SMLX cache: `~/.cache/smlx/` (for custom data)

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found

```
Error: Model mlx-community/SmolVLM-256M-Instruct not found
```

**Solution:** Download the model first:
```bash
python -m smlx.tools.download_data --model mlx-community/SmolVLM-256M-Instruct
```

#### 2. Dataset Download Failures

```
ConnectionError: Failed to download dataset
```

**Solutions:**
- Check internet connection
- Try again (Hugging Face can be slow)
- Manually download from web: https://huggingface.co/datasets/AI4Math/MathVista

#### 3. Out of Memory

```
RuntimeError: Out of memory
```

**Solutions:**
- Use a smaller model (256M instead of 500M)
- Reduce `--max-tokens` (try 256 or 128)
- Close other applications
- Use `--max-samples` to run smaller batches

#### 4. Slow Inference

**Solutions:**
- Ensure you're on Apple Silicon (MLX is optimized for M-series chips)
- Check that MLX is using GPU: `python -c "import mlx.core as mx; print(mx.metal.is_available())"`
- Reduce `--max-tokens` for faster generation
- Use lower temperature (0.0) for faster sampling

#### 5. Import Errors

```
ModuleNotFoundError: No module named 'mlx_vlm'
```

**Solution:** Install evaluation dependencies:
```bash
pip install -e ".[evals]"
```

#### 6. Processor Patch Size Warnings

```
Warning: processor has no attribute patch_size
```

**Solution:** This is handled automatically in `smlx/evals/utils.py`. The warning is informational and can be ignored.

#### 7. Test Markers Not Working

```
pytest: error: unrecognized arguments: -m eval
```

**Solution:** Ensure `pytest.ini` is present and markers are registered. Run from project root.

---

## Best Practices

### Development Workflow

1. **Quick Testing:** Start with `--max-samples 10` to verify everything works
2. **Development:** Use `--max-samples 100` for iterative development
3. **Validation:** Run on full validation sets (testmini, validation split)
4. **Final Evaluation:** Run on test sets for final benchmarking

### Reproducibility

For reproducible results:

```bash
python -m smlx.evals.math_vista \
    --model mlx-community/SmolVLM-256M-Instruct \
    --temperature 0.0 \
    --seed 42 \
    --split testmini
```

**Note:** Even with `temperature=0.0`, minor variations can occur due to MLX's optimizations.

### Efficient Batch Evaluation

Run multiple evaluations in parallel (different terminals/screens):

```bash
# Terminal 1
python -m smlx.evals.math_vista --model model1

# Terminal 2
python -m smlx.evals.mmmu --model model1

# Terminal 3
python -m smlx.evals.mmstar --model model2
```

### Result Management

Organize results by model and date:

```bash
# Create organized output structure
mkdir -p results/2024-01-15/smolvlm-256m

python -m smlx.evals.math_vista \
    --model mlx-community/SmolVLM-256M-Instruct \
    --output results/2024-01-15/smolvlm-256m/mathvista.csv

python -m smlx.evals.mmmu \
    --model mlx-community/SmolVLM-256M-Instruct \
    --output results/2024-01-15/smolvlm-256m/mmmu.csv
```

---

## Testing

Run evaluation tests:

```bash
# All evaluation tests
pytest tests/evals/ -v

# Specific benchmark tests
pytest tests/evals/test_math_vista.py -v
pytest tests/evals/test_mmmu.py -v

# Skip slow integration tests
pytest tests/evals/ -m "not slow"

# Only unit tests (fast)
pytest tests/evals/ -m "unit"

# Run with model downloads enabled
SMLX_DOWNLOAD_TEST_MODELS=1 pytest tests/evals/ -v
```

Test markers:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.eval` - Evaluation tests
- `@pytest.mark.slow` - Slow integration tests
- `@pytest.mark.requires_model` - Requires model download
- `@pytest.mark.gpu` - Requires MLX/GPU

---

## Extending Evaluations

### Adding a New Benchmark

To add a new evaluation benchmark:

1. **Create module:** `smlx/evals/your_benchmark.py`
2. **Implement functions:**
   - `process_question(sample)` - Format the question/prompt
   - `normalize_answer(text)` - Extract and normalize the model's answer
   - `evaluate_answer(prediction, ground_truth)` - Score the answer
   - `main()` - CLI entry point with argparse
3. **Add tests:** `tests/evals/test_your_benchmark.py`
4. **Add fixtures:** Update `tests/evals/conftest.py`
5. **Update documentation:** Add section to this file

### Custom Answer Extraction

For complex answer formats, see `smlx/evals/math_vista.py:normalize_answer()` for examples of:
- Regex pattern matching
- Fuzzy string matching (Levenshtein distance)
- Multi-pattern priority systems
- Numeric extraction with units

---

## Future Enhancements

### Planned Features

1. **Audio Evaluations** - For Whisper, Orpheus, YAMNet
   - Word Error Rate (WER) for speech recognition
   - Character Error Rate (CER) for transcription
   - Audio classification metrics

3. **Advanced Analysis Tools**
   - Multi-model comparison reports
   - Statistical significance testing
   - Interactive visualization dashboards
   - Automated report generation

4. **OCRBench v2** - Upgrade to latest version
   - 10,000 samples (vs current 1,000)
   - 8 abilities across 31 scenarios
   - Bilingual support

### Recently Implemented (v0.2.0)

✅ **Text-Only Evaluations** - Full support for SmolLM2 models
- Perplexity evaluation on WikiText, PTB, OpenWebText
- Text generation benchmarks (context scaling, generation length, temperature)
- Quantization comparison (FP16, 8-bit, 4-bit, GPTQ, AWQ, DWQ)
- Unified benchmark CLI for all suites
- **LM Evaluation Harness Integration** - Access to 200+ standardized tasks
  - Integration with EleutherAI's lm-evaluation-harness
  - Support for HellaSwag, Winogrande, ARC, MMLU, TruthfulQA, GSM8K, and more
  - Zero-shot and few-shot evaluation
  - Programmatic and CLI interfaces
- Comprehensive test coverage

---

## References

- **MathVista:** [arXiv:2310.02255](https://arxiv.org/abs/2310.02255) | [GitHub](https://github.com/lupantech/MathVista) | [Dataset](https://huggingface.co/datasets/AI4Math/MathVista)
- **MMMU:** [arXiv:2311.16502](https://arxiv.org/abs/2311.16502) | [GitHub](https://github.com/MMMU-Benchmark/MMMU) | [Dataset](https://huggingface.co/datasets/MMMU/MMMU)
- **MMStar:** [NeurIPS 2024](https://github.com/MMStar-Benchmark/MMStar) | [Dataset](https://huggingface.co/datasets/Lin-Chen/MMStar)
- **OCRBench:** [arXiv:2305.07895](https://arxiv.org/abs/2305.07895) | [Dataset](https://huggingface.co/datasets/echo840/OCRBench)
- **MLX:** [GitHub](https://github.com/ml-explore/mlx) | [Documentation](https://ml-explore.github.io/mlx/)
- **MLX-VLM:** [GitHub](https://github.com/Blaizzy/mlx-vlm)

---

## Support

For issues or questions:
- **SMLX Issues:** Check [CLAUDE.md](../CLAUDE.md) for project guidelines
- **MLX Issues:** [ml-explore/mlx GitHub](https://github.com/ml-explore/mlx)
- **Benchmark Issues:** Refer to individual benchmark repositories

---

*Last updated: 2025-11-12*
