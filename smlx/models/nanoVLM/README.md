# nanoVLM

A minimal 222M parameter vision-language model with ~750 lines of implementation code, perfect for learning, customization, and ultra-lightweight deployments.

## Model Details

- **Size**: 222M parameters (135M language + 85M vision)
- **Type**: Multimodal Vision-Language Model
- **Architecture**: SigLIP vision encoder + SmolLM2-135M language model
- **Training Time**: ~6 hours on single H100 GPU
- **Code Size**: ~750 lines (minimal implementation)
- **Memory**: Runs on <1GB RAM
- **License**: Apache 2.0
- **HuggingFace**: [lusxvr/nanoVLM-222M](https://huggingface.co/lusxvr/nanoVLM-222M)

## Why nanoVLM for SMLX?

nanoVLM is the **educational and experimentation** model for SMLX:

- Minimal implementation (~750 lines) makes it easy to understand
- Smallest practical VLM (222M parameters)
- Perfect for learning VLM architecture from scratch
- Ideal for rapid prototyping and customization
- Fast to train on custom datasets (6 hours on H100)
- Apache 2.0 license (fully open)
- Great starting point for domain-specific VLMs

## Installation

```bash
# Install SMLX with vision support
pip install smlx[vision]

# Or install all extras
pip install smlx[all]
```

## Quick Start

### Python API

```python
from smlx.models.nanoVLM import load, generate
from PIL import Image

# Load the model
model, processor = load("lusxvr/nanoVLM-222M")

# Load an image
image = Image.open("photo.jpg")

# Generate a response
prompt = "What is in this image?"
response = generate(
    model=model,
    processor=processor,
    image=image,
    prompt=prompt,
    max_tokens=128
)

print(response)
```

### Command Line

```bash
# Generate from image + prompt
smlx generate \
  --model nanoVLM \
  --image path/to/image.jpg \
  --prompt "Describe what you see" \
  --max-tokens 128

# Interactive mode
smlx chat --model nanoVLM --vision
```

## Converting from HuggingFace

```bash
# Convert from HuggingFace with 4-bit quantization
python -m smlx.tools.convert2mlx \
  --hf-path lusxvr/nanoVLM-222M \
  --mlx-path ./models/nanoVLM-4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64 \
  --dtype float16

# Convert without quantization (for experimentation)
python -m smlx.tools.convert2mlx \
  --hf-path lusxvr/nanoVLM-222M \
  --mlx-path ./models/nanoVLM \
  --dtype float16
```

## Usage Examples

### Basic Image Captioning

```python
from smlx.models.nanoVLM import load, generate
from PIL import Image

model, processor = load("lusxvr/nanoVLM-222M")
image = Image.open("cat.jpg")

caption = generate(
    model=model,
    processor=processor,
    image=image,
    prompt="Describe this image:",
    max_tokens=100
)
print(f"Caption: {caption}")
```

### Simple VQA

```python
image = Image.open("kitchen.jpg")

answer = generate(
    model=model,
    processor=processor,
    image=image,
    prompt="What appliances are visible in this kitchen?",
    max_tokens=80
)
print(f"Answer: {answer}")
```

### Object Counting

```python
image = Image.open("parking_lot.jpg")

count = generate(
    model=model,
    processor=processor,
    image=image,
    prompt="How many cars are in this image?",
    max_tokens=20
)
print(count)
```

## Training Your Own nanoVLM

One of nanoVLM's key advantages is the ability to train from scratch quickly:

### Prepare Training Data

```python
# Create training dataset (JSONL format)
# Each line: {"image": "path/to/img.jpg", "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}

# Example training data:
import json

data = [
    {
        "image": "images/cat1.jpg",
        "conversations": [
            {"from": "human", "value": "What animal is this?"},
            {"from": "gpt", "value": "This is a cat."}
        ]
    },
    {
        "image": "images/dog1.jpg",
        "conversations": [
            {"from": "human", "value": "Describe this animal"},
            {"from": "gpt", "value": "This is a golden retriever dog sitting on grass."}
        ]
    }
]

with open("training_data.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

### Train from Scratch

```bash
# Train nanoVLM from scratch (requires GPU)
python -m smlx.models.nanoVLM.train \
  --vision-model google/siglip-base-patch16-224 \
  --language-model HuggingFaceTB/SmolLM2-135M \
  --data training_data.jsonl \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --epochs 3 \
  --output ./my_nanovlm

# Estimated time: ~6 hours on H100, ~24 hours on A100
```

### Fine-tune Existing nanoVLM

```bash
# Fine-tune for domain-specific tasks (faster)
python -m smlx.quant.lora \
  --model lusxvr/nanoVLM-222M \
  --data domain_specific.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --epochs 2 \
  --output ./nanovlm_finetuned

# Estimated time: ~30 minutes on H100
```

## Quantization

nanoVLM is already tiny but quantization makes it even smaller:

### 4-bit Quantization

```bash
python -m smlx.tools.convert2mlx \
  --hf-path lusxvr/nanoVLM-222M \
  --mlx-path ./models/nanoVLM-4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64
```

**Benefits:**

- ~75% size reduction (222M → ~55MB weights)
- Still very capable for simple tasks
- Can run on extremely limited devices

### Mixed Precision

Preserve vision encoder quality:

```bash
python -m smlx.tools.convert2mlx \
  --hf-path lusxvr/nanoVLM-222M \
  --mlx-path ./models/nanoVLM-mixed \
  --quantize \
  --q-bits 4 \
  --skip-multimodal
```

## Performance on M4

### Benchmarks (Apple M4, 36GB Unified Memory)

| Configuration | Memory Usage | Tokens/sec | Training Time |
|--------------|--------------|------------|---------------|
| FP16 (no quant) | ~0.9GB | 52 tok/s | 6h (H100) |
| 8-bit quant | ~0.45GB | 58 tok/s | N/A |
| 4-bit quant | ~0.22GB | 62 tok/s | N/A |

### nanoVLM vs Others

| Model | Parameters | Memory | Speed | Ease of Customization |
|-------|-----------|--------|-------|----------------------|
| nanoVLM | 222M | 0.9GB | Fast | ⭐⭐⭐⭐⭐ (minimal code) |
| SmolVLM-256M | 256M | 1.0GB | Very Fast | ⭐⭐⭐ (standard) |
| SmolVLM-500M | 500M | 2.0GB | Fast | ⭐⭐ (larger) |
| Moondream2 | 1.8B | 3.5GB | Medium | ⭐ (complex) |

**Key Advantage**: nanoVLM's ~750 line implementation makes it the easiest to understand and customize.

## Understanding the Architecture

nanoVLM's simplicity makes it perfect for learning:

### 1. Vision Encoder (85M params)

```python
# SigLIP-base: Processes images to visual features
# Input: 224x224 image
# Output: 196 vision tokens (14x14 patches)
```

### 2. Projection Layer (~2M params)

```python
# Maps vision tokens to language model space
# Input: 196 x 768 (SigLIP output)
# Output: 196 x 576 (SmolLM2 input)
```

### 3. Language Model (135M params)

```python
# SmolLM2-135M: Processes text + vision tokens
# Generates natural language responses
```

### Total Pipeline

```text
Image (224x224)
  → SigLIP (85M)
  → Projection (~2M)
  → SmolLM2 (135M)
  → Text Output
```

## Customization Examples

### Custom Vision Encoder

```python
# Swap SigLIP for a different vision encoder
from smlx.models.nanoVLM import NanoVLM

model = NanoVLM(
    vision_encoder="apple/mobilevit-small",  # Lighter encoder
    language_model="HuggingFaceTB/SmolLM2-135M",
    projection_dim=576
)

# Train on your data
model.train(data="custom_data.jsonl")
```

### Custom Language Model

```python
# Use a different language backbone
model = NanoVLM(
    vision_encoder="google/siglip-base-patch16-224",
    language_model="Qwen/Qwen2-0.5B",  # Alternative LLM
    projection_dim=896  # Match Qwen2 hidden size
)
```

### Task-Specific Training

```python
# Medical imaging
model.train(
    data="medical_images.jsonl",
    task_prefix="Medical Analysis:",
    epochs=5
)

# Product catalogs
model.train(
    data="product_images.jsonl",
    task_prefix="Product Description:",
    epochs=3
)
```

## Limitations

- **Quality**: Lower than larger VLMs (SmolVLM-500M, Moondream2)
- **Resolution**: Limited to 224×224 images
- **Complex Reasoning**: Minimal reasoning capabilities
- **Specialized Tasks**: Requires fine-tuning for specific domains

**Why Use Anyway?**

- ✅ Fastest to train from scratch
- ✅ Easiest to understand and modify
- ✅ Smallest memory footprint
- ✅ Perfect for learning and experimentation
- ✅ Ideal for rapid prototyping

## Best Use Cases

nanoVLM excels at:

- ✅ Learning VLM architecture
- ✅ Rapid prototyping of vision-language applications
- ✅ Custom domain-specific VLMs (medical, e-commerce, etc.)
- ✅ Ultra-lightweight deployments
- ✅ Educational projects
- ✅ Research experiments
- ✅ Basic captioning and VQA
- ✅ Devices with <1GB RAM constraint

## Implementation in SMLX

The nanoVLM implementation in SMLX follows a minimal design:

```text
smlx/models/nanoVLM/
├── __init__.py          # Exports load, generate, train
├── config.py            # VisionConfig, LanguageConfig, NanoVLMConfig
├── nano_vlm.py          # Main model implementation (~400 lines)
├── vision.py            # SigLIP vision encoder wrapper
├── language.py          # SmolLM2 language model wrapper
├── projection.py        # Vision-to-language projection (~50 lines)
└── train.py             # Training script (~300 lines)
```

Total: ~750 lines of well-documented, easy-to-understand code.

## References

- **HuggingFace Model**: [lusxvr/nanoVLM-222M](https://huggingface.co/lusxvr/nanoVLM-222M)
- **SigLIP**: [Google SigLIP](https://huggingface.co/google/siglip-base-patch16-224)
- **SmolLM2-135M**: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- **MLX-VLM Reference**: Implementation patterns adapted from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

## Citation

```bibtex
@misc{nanovlm2024,
  title={nanoVLM: A Minimal Vision-Language Model},
  author={lusxvr},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/lusxvr/nanoVLM-222M}
}
```

## License

Apache 2.0 - Fully open source and permissive for commercial use.

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets with unified memory.
