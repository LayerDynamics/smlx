# Donut-base

An OCR-free document understanding model that processes documents end-to-end without requiring separate OCR, using a Swin Transformer vision encoder and BART text decoder.

## Model Details

- **Size**: ~200M parameters
- **Type**: Document Understanding (Vision-to-Text)
- **Architecture**: Swin Transformer (vision) + BART (text decoder)
- **Approach**: OCR-free - direct image-to-text
- **Tasks**: Document classification, parsing, VQA on documents
- **License**: MIT
- **HuggingFace**: [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base)

## Why Donut for SMLX?

Donut is the **document specialist** for SMLX:

- OCR-free approach (no separate OCR pipeline needed)
- Lightweight at ~200M parameters
- End-to-end trainable for custom document tasks
- MIT license (very permissive)
- Excellent for forms, receipts, invoices, documents
- Fast inference on M4

## Installation

```bash
pip install smlx[vision]
```

## Quick Start

### Python API

```python
from smlx.models.Donut_base import load, generate
from PIL import Image

# Load model
model, processor = load("naver-clova-ix/donut-base-finetuned-docvqa")

# Load document image
image = Image.open("invoice.jpg")

# Parse document
prompt = "What is the invoice number?"
response = generate(
    model=model,
    processor=processor,
    image=image,
    prompt=prompt
)

print(response)
```

### Command Line

```bash
smlx generate \
  --model donut-base \
  --image document.jpg \
  --prompt "Extract all key-value pairs"
```

## Converting from HuggingFace

```bash
# Convert with 4-bit quantization
python -m smlx.tools.convert2mlx \
  --hf-path naver-clova-ix/donut-base \
  --mlx-path ./models/donut-base-4bit \
  --quantize \
  --q-bits 4 \
  --dtype float16
```

## Quantization

### 4-bit Quantization

```bash
python -m smlx.tools.convert2mlx \
  --hf-path naver-clova-ix/donut-base \
  --mlx-path ./models/donut-4bit \
  --quantize \
  --q-bits 4
```

**Benefits:**

- ~75% size reduction
- Fast document processing on M4

## Performance on M4

| Configuration | Memory | Speed | Quality |
|--------------|--------|-------|---------|
| FP16 | ~0.8GB | Fast | 100% |
| 4-bit | ~0.2GB | Faster | 97% |

## Best Use Cases

Donut excels at:

- ✅ Document parsing (invoices, receipts, forms)
- ✅ Document VQA
- ✅ Information extraction from documents
- ✅ OCR-free document understanding
- ✅ Custom document tasks (fine-tunable)

## References

- **HuggingFace**: [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base)
- **Paper**: [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- **GitHub**: [clovaai/donut](https://github.com/clovaai/donut)

## License

MIT

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
