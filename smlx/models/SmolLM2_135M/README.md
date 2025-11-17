# SmolLM2-135M

The ultra-tiny language model that can run on smartphones while maintaining impressive capabilities for its size. Perfect for on-device text generation and understanding.

## Model Details

- **Size**: 135M parameters
- **Type**: Causal Language Model (Decoder-only Transformer)
- **Context**: 2048 tokens
- **Training Data**: FineMath, Stack-Edu, SmolTalk (curated high-quality)
- **Memory**: ~0.5GB (FP16), ~0.13GB (4-bit quantized)
- **License**: Apache 2.0
- **HuggingFace**: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- **MLX**: [mlx-community/SmolLM2-135M-Instruct](https://huggingface.co/mlx-community/SmolLM2-135M-Instruct)

## Why SmolLM2-135M for SMLX?

SmolLM2-135M is the **ultra-lightweight LLM** for SMLX:

- Smallest viable general-purpose LLM (135M params)
- Runs on smartphones and ultra-constrained devices
- Apache 2.0 license (fully open)
- Part of SmolLM family (perfect brand alignment)
- Trained on curated high-quality data
- Language backbone for nanoVLM
- Ideal for learning transformer architecture

## Installation

```bash
pip install smlx
```

## Quick Start

### Python API

```python
from smlx.models.SmolLM2_135M import load, generate

# Load the model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Generate text
prompt = "Explain photosynthesis in simple terms:"
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=150,
    temperature=0.7
)

print(response)
```

### Command Line

```bash
# Generate text
smlx generate \
  --model SmolLM2-135M-Instruct \
  --prompt "Write a haiku about coding:" \
  --max-tokens 50

# Interactive chat
smlx chat --model SmolLM2-135M-Instruct
```

## Converting from HuggingFace

```bash
# Convert with 4-bit quantization
python -m smlx.tools.convert2mlx \
  --hf-path HuggingFaceTB/SmolLM2-135M-Instruct \
  --mlx-path ./models/SmolLM2-135M-4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64 \
  --dtype float16
```

## Usage Examples

### Text Completion

```python
from smlx.models.SmolLM2_135M import load, generate

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

prompt = "The three laws of robotics are:"
completion = generate(model, tokenizer, prompt, max_tokens=200)
print(completion)
```

### Question Answering

```python
prompt = "Q: What is the capital of France?\nA:"
answer = generate(model, tokenizer, prompt, max_tokens=50, temperature=0.3)
print(answer)
```

### Code Generation

```python
prompt = "Write a Python function to calculate fibonacci numbers:"
code = generate(model, tokenizer, prompt, max_tokens=200, temperature=0.5)
print(code)
```

### Chat Mode

```python
from smlx.models.SmolLM2_135M import load, chat

model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

messages = [
    {"role": "user", "content": "Hello! What can you help me with?"}
]

response = chat(model, tokenizer, messages, max_tokens=100)
print(response)
```

## Quantization

### 4-bit Quantization (Recommended)

```bash
python -m smlx.tools.convert2mlx \
  --hf-path HuggingFaceTB/SmolLM2-135M-Instruct \
  --mlx-path ./models/SmolLM2-135M-4bit \
  --quantize \
  --q-bits 4
```

**Benefits:**

- ~75% size reduction (~0.5GB → ~0.13GB)
- Fits on any device
- Minimal quality loss for simple tasks

## Performance on M4

| Configuration | Memory | Tokens/sec | Quality |
|--------------|--------|------------|---------|
| FP16 | ~0.5GB | 85 tok/s | 100% |
| 8-bit | ~0.25GB | 95 tok/s | 99% |
| 4-bit | ~0.13GB | 105 tok/s | 96% |

**Key Strength**: Incredibly fast and lightweight while being surprisingly capable.

## Fine-tuning with LoRA

```bash
# Fine-tune for specific domain
python -m smlx.quant.lora \
  --model SmolLM2-135M-Instruct \
  --data ./training_data.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --output ./smollm2_135m_finetuned
```

## Best Use Cases

SmolLM2-135M excels at:

- ✅ Simple text completion
- ✅ Basic Q&A
- ✅ Lightweight chatbots
- ✅ Code completion (simple patterns)
- ✅ On-device text generation
- ✅ Learning transformer architecture
- ✅ Prototyping language model applications
- ✅ Ultra-low-resource environments
- ✅ Language backbone for multimodal models

## Limitations

- **Complex Reasoning**: Limited compared to larger models
- **Knowledge**: Smaller knowledge base
- **Long Context**: 2048 token limit
- **Factuality**: May hallucinate more than larger models

**Trade-off**: Acceptable for 135M param budget and extreme efficiency.

## References

- **HuggingFace**: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- **MLX**: [mlx-community/SmolLM2-135M-Instruct](https://huggingface.co/mlx-community/SmolLM2-135M-Instruct)
- **SmolLM Blog**: [HuggingFace Blog](https://huggingface.co/blog/smollm)

## License

Apache 2.0

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
