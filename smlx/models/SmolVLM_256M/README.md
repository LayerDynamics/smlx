# SmolVLM-256M-Instruct

A small vision-language model combining SigLIP vision encoder with SmolLM2 language model.

## Overview

SmolVLM-256M-Instruct is a multimodal model that can:

- Understand and describe images
- Answer questions about visual content
- Engage in multi-turn conversations with image context
- Process multiple images in a single prompt

**Architecture:**

- **Vision**: SigLIP-SO400M encoder (1152 hidden size, 16 heads, 27 layers)
- **Language**: SmolLM2-135M (576 hidden size, 9 heads, 30 layers, GQA with 3 KV heads)
- **Connector**: Idefics3 with pixel shuffle (scale=2)
- **Total Parameters**: ~256M

## Installation

```bash
# Install SMLX with vision dependencies
pip install -e ".[vision]"

# Or install specific requirements
pip install mlx transformers huggingface_hub pillow requests
```

## Quick Start

### Basic Usage

```python
from smlx.models.SmolVLM_256M import load, generate

# Load model and processor
model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")

# Generate from image
output = generate(
    model=model,
    processor=processor,
    prompt="<image>\nDescribe this image:",
    image="https://example.com/photo.jpg",
    max_tokens=100,
    temperature=0.7
)

print(output)
```

### Streaming Generation

```python
from smlx.models.SmolVLM_256M import load, stream_generate

model, processor = load()

for text in stream_generate(
    model=model,
    processor=processor,
    prompt="<image>\nWhat do you see?",
    image="photo.jpg",
    max_tokens=80
):
    print(text, end="", flush=True)
```

### Chat Interface

```python
from smlx.models.SmolVLM_256M import load, chat

model, processor = load()

messages = [
    {"role": "user", "content": "What's in this image?"}
]

response = chat(
    model=model,
    processor=processor,
    messages=messages,
    image="photo.jpg",
    max_tokens=100
)

print(response)
```

### Multiple Images

```python
from smlx.models.SmolVLM_256M import load, generate

model, processor = load()

images = ["image1.jpg", "image2.jpg"]

output = generate(
    model=model,
    processor=processor,
    prompt="<image>\n<image>\nCompare these images.",
    image=images,
    max_tokens=150
)

print(output)
```

## API Reference

### `load(path_or_hf_repo, revision=None, lazy=False, force_download=False)`

Load SmolVLM model and processor.

**Args:**

- `path_or_hf_repo` (str): Local path or HuggingFace repo ID
- `revision` (str, optional): Git revision (branch, tag, or commit)
- `lazy` (bool): If True, lazy load weights. Default: False
- `force_download` (bool): Force re-download from HF Hub

**Returns:**

- `model` (Model): SmolVLM model
- `processor` (Processor): Combined tokenizer + image processor

**Example:**

```python
# Load from HuggingFace Hub
model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")

# Load from local path
model, processor = load("/path/to/model")

# Load specific revision
model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct", revision="main")
```

### `generate(model, processor, prompt, image=None, max_tokens=100, temperature=0.7, top_p=0.9, verbose=False)`

Generate text response from prompt and optional image.

**Args:**

- `model` (Model): SmolVLM model
- `processor` (Processor): Combined tokenizer + image processor
- `prompt` (str): Text prompt (include `<image>` tokens for images)
- `image` (str/Image/List, optional): Image(s) as URL, path, or PIL Image
- `max_tokens` (int): Maximum tokens to generate. Default: 100
- `temperature` (float): Sampling temperature (0=greedy). Default: 0.7
- `top_p` (float): Nucleus sampling threshold. Default: 0.9
- `verbose` (bool): Print generation statistics. Default: False

**Returns:**

- `output` (str): Generated text

**Example:**

```python
output = generate(
    model=model,
    processor=processor,
    prompt="<image>\nWhat is this?",
    image="photo.jpg",
    max_tokens=50,
    temperature=0.5
)
```

### `stream_generate(model, processor, prompt, image=None, max_tokens=100, temperature=0.7, top_p=0.9)`

Generate text with streaming output.

**Args:**

- Same as `generate()` except no `verbose` parameter

**Yields:**

- `text` (str): Generated text tokens as they are produced

**Example:**

```python
for text in stream_generate(
    model=model,
    processor=processor,
    prompt="<image>\nDescribe this:",
    image="photo.jpg",
    max_tokens=100
):
    print(text, end="", flush=True)
```

### `chat(model, processor, messages, image=None, max_tokens=100, temperature=0.7, top_p=0.9)`

Chat-style interaction with conversation history.

**Args:**

- `model` (Model): SmolVLM model
- `processor` (Processor): Combined tokenizer + image processor
- `messages` (List[dict]): List of message dicts with 'role' and 'content'
- `image` (str/Image/List, optional): Image(s)
- `max_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature
- `top_p` (float): Nucleus sampling threshold

**Returns:**

- `response` (str): Generated response text

**Example:**

```python
messages = [
    {"role": "user", "content": "What do you see?"},
    {"role": "assistant", "content": "I see a car."},
    {"role": "user", "content": "What color is it?"}
]

response = chat(
    model=model,
    processor=processor,
    messages=messages,
    image="photo.jpg"
)
```

### `save_model(model, save_path, tokenizer=None)`

Save model weights and configuration.

**Args:**

- `model` (Model): SmolVLM model to save
- `save_path` (str): Directory to save model
- `tokenizer` (optional): Tokenizer to save alongside model

**Example:**

```python
from smlx.models.SmolVLM_256M import load, save_model

model, processor = load()
save_model(model, "./my_model", processor.tokenizer)
```

## Configuration

### Default Configuration

```python
from smlx.models.SmolVLM_256M import DEFAULT_CONFIG

# Vision config (SigLIP-SO400M)
vision_config = DEFAULT_CONFIG.vision_config
# hidden_size=1152, num_heads=16, num_layers=27
# patch_size=14, image_size=384

# Text config (SmolLM2-135M)
text_config = DEFAULT_CONFIG.text_config
# hidden_size=576, num_heads=9, num_layers=30
# num_kv_heads=3 (GQA), vocab_size=49152

# Model config
model_config = DEFAULT_CONFIG
# scale_factor=2 (pixel shuffle)
# image_token_id=49153
```

### Custom Configuration

```python
from smlx.models.SmolVLM_256M import ModelConfig, VisionConfig, TextConfig, Model

# Create custom config
config = ModelConfig(
    vision_config=VisionConfig(
        hidden_size=1152,
        num_attention_heads=16,
        num_hidden_layers=27,
    ),
    text_config=TextConfig(
        hidden_size=576,
        num_attention_heads=9,
        num_hidden_layers=30,
    ),
    scale_factor=2,
)

# Initialize model
model = Model(config)
```

## Image Processing

### Image Formats

SmolVLM accepts images in multiple formats:

```python
# 1. URL
image = "https://example.com/photo.jpg"

# 2. Local file path
image = "/path/to/photo.jpg"

# 3. PIL Image
from PIL import Image
image = Image.open("photo.jpg")

# 4. Multiple images (list)
images = ["photo1.jpg", "photo2.jpg", PIL_image]
```

### Image Preprocessing

Images are automatically preprocessed:

- Resized to 384x384
- Rescaled from [0, 255] to [0, 1]
- Normalized with mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- Converted to MLX format

```python
from smlx.models.SmolVLM_256M import ImageProcessor, load_image

# Manual preprocessing
processor = ImageProcessor()
image = load_image("photo.jpg")
pixel_values = processor(image)
```

## Prompting

### Image Token Placement

Use `<image>` tokens to indicate where images should be inserted:

```python
# Single image before question
prompt = "<image>\nWhat is in this image?"

# Multiple images
prompt = "<image>\n<image>\nCompare these two images."

# Image with context
prompt = "User uploaded an image: <image>\nPlease describe it."
```

If you don't include `<image>` tokens, they are automatically prepended.

### Chat Templates

SmolVLM uses the SmolLM2 chat template:

```python
# Manual formatting
prompt = """<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
<image>
What do you see?<|im_end|>
<|im_start|>assistant
"""

# Or use chat() function for automatic formatting
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What do you see?"}
]
response = chat(model, processor, messages, image="photo.jpg")
```

## Performance

### Memory Usage

- **FP16**: ~512 MB (256M params × 2 bytes)
- **4-bit quantized**: ~128 MB (4x smaller)
- **KV cache**: ~50-100 MB (depends on context length)

### Inference Speed (M4 Max 36GB)

- **First token (prompt processing)**: ~200-400 ms
- **Subsequent tokens**: ~20-40 ms per token
- **Image encoding**: ~100-200 ms

### Optimization Tips

```python
# 1. Use quantization (coming soon)
from smlx.quant import quantize_gptq

quantized_model = quantize_gptq(model, bits=4, group_size=64)

# 2. Use greedy decoding for faster generation
output = generate(
    model=model,
    processor=processor,
    prompt="<image>\nWhat is this?",
    image="photo.jpg",
    temperature=0,  # Greedy
    max_tokens=50
)

# 3. Reduce max_tokens for shorter responses
output = generate(..., max_tokens=20)

# 4. Use lazy loading for faster startup
model, processor = load(lazy=True)
```

## Examples

See [examples/models/smolvlm_256m/](../../../examples/models/smolvlm_256m/) for complete examples:

- `smolvlm_256m_example.py` - Basic usage, Q&A, streaming, chat, multi-image

Run examples:

```bash
python examples/models/smolvlm_256m/smolvlm_256m_example.py
```

## Benchmarks

### Vision-Language Benchmarks

| Benchmark | SmolVLM-256M | Notes |
|-----------|--------------|-------|
| VQAv2 | TBD | Visual question answering |
| GQA | TBD | Compositional visual reasoning |
| TextVQA | TBD | Reading text in images |
| MMMU | TBD | Multimodal understanding (see evals) |
| MMStar | TBD | Multimodal reasoning (see evals) |

## Limitations

- **Image resolution**: Fixed at 384x384 (SigLIP encoder)
- **Context length**: 4096 tokens maximum
- **Single-turn focus**: Best for single-turn or short conversations
- **Text in images**: Limited OCR capabilities
- **Fine details**: Small objects may not be recognized accurately

## Troubleshooting

**Q: Model download is slow**

- Use a faster internet connection
- Download once and use local path: `load("/path/to/model")`

**Q: Out of memory error**

- Reduce `max_tokens` parameter
- Use quantization (4-bit) when available
- Close other applications

**Q: Generation is slow**

- Use `temperature=0` for greedy decoding (faster)
- Reduce `max_tokens`
- Ensure MLX Metal acceleration is working

**Q: Image not recognized**

- Check image format (JPEG, PNG supported)
- Ensure image URL is accessible
- Try local file instead of URL

**Q: Import errors**

- Install vision dependencies: `pip install -e ".[vision]"`
- Ensure transformers, PIL, requests are installed

## Citation

```bibtex
@software{smolvlm_2025,
  title = {SmolVLM-256M-Instruct: Small Vision-Language Model},
  author = {HuggingFace Team},
  year = {2025},
  url = {https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct}
}

@software{smlx_smolvlm_2025,
  title = {SMLX: SmolVLM-256M Implementation},
  author = {SMLX Contributors},
  year = {2025},
  url = {https://github.com/yourusername/smlx}
}
```

## References

- [SmolVLM Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [SmolLM2 Model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Idefics3 Architecture](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)

## License

Same as base models (Apache 2.0 / MIT).
