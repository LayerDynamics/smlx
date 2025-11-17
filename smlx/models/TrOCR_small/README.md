# TrOCR-small

Microsoft's smallest Transformer-based OCR model combining a BEiT vision encoder with a RoBERTa text decoder for both printed and handwritten text recognition.

## Model Details

- **Size**: ~60M parameters (estimated)
- **Type**: Optical Character Recognition (OCR)
- **Architecture**: BEiT (vision encoder) + RoBERTa (text decoder)
- **Variants**:
  - `trocr-small-printed` - Printed text
  - `trocr-small-handwritten` - Handwriting
- **Memory**: ~250MB (FP16), ~65MB (4-bit quantized)
- **License**: MIT
- **HuggingFace**: [microsoft/trocr-small-printed](https://huggingface.co/microsoft/trocr-small-printed)

## Why TrOCR-small for SMLX?

TrOCR-small is the **lightweight Transformer OCR** for SMLX:

- Small (~60M params)
- Transformer-based (no external OCR engine needed)
- Both printed and handwritten variants
- MIT license (very permissive)
- Good accuracy for size
- Fast inference on M4

## Installation

```bash
pip install smlx[vision]
```

## Quick Start

### Python API

```python
from smlx.models.TrOCR_small import load, recognize
from PIL import Image

# Load model (choose variant)
model, processor = load("microsoft/trocr-small-printed")

# Load image with text
image = Image.open("document.jpg")

# Recognize text
text = recognize(model=model, processor=processor, image=image)

print(f"Recognized text: {text}")
```

### Command Line

```bash
# OCR on printed text
smlx ocr \
  --model trocr-small-printed \
  --image document.jpg

# OCR on handwritten text
smlx ocr \
  --model trocr-small-handwritten \
  --image handwriting.jpg
```

## Converting from HuggingFace

```bash
# Convert printed variant with 4-bit quantization
python -m smlx.tools.convert2mlx \
  --hf-path microsoft/trocr-small-printed \
  --mlx-path ./models/trocr-small-printed-4bit \
  --quantize \
  --q-bits 4 \
  --skip-multimodal  # Preserve vision encoder quality
```

## Usage Examples

### Printed Text OCR

```python
from smlx.models.TrOCR_small import load, recognize
from PIL import Image

# Load printed text model
model, processor = load("microsoft/trocr-small-printed")

# Recognize printed text
image = Image.open("receipt.jpg")
text = recognize(model, processor, image)
print(text)
```

### Handwritten Text OCR

```python
# Load handwritten text model
model, processor = load("microsoft/trocr-small-handwritten")

# Recognize handwriting
image = Image.open("handwritten_note.jpg")
text = recognize(model, processor, image)
print(text)
```

### Batch OCR

```python
from pathlib import Path

model, processor = load("microsoft/trocr-small-printed")

# Process multiple images
image_paths = Path("documents/").glob("*.jpg")

results = []
for img_path in image_paths:
    image = Image.open(img_path)
    text = recognize(model, processor, image)
    results.append({"file": img_path.name, "text": text})

for result in results:
    print(f"{result['file']}: {result['text']}")
```

### OCR with Preprocessing

```python
from PIL import ImageEnhance

def preprocess_for_ocr(image):
    """Improve OCR accuracy with preprocessing"""
    # Convert to grayscale
    image = image.convert('L')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    return image

# Use preprocessed image
image = Image.open("low_quality.jpg")
image = preprocess_for_ocr(image)
text = recognize(model, processor, image)
print(text)
```

## Quantization

### 4-bit Quantization

```bash
python -m smlx.tools.convert2mlx \
  --hf-path microsoft/trocr-small-printed \
  --mlx-path ./models/trocr-small-4bit \
  --quantize \
  --q-bits 4
```

**Benefits:**

- ~75% size reduction (~250MB → ~65MB)
- Faster inference

## Performance on M4

| Configuration | Memory | Speed | Accuracy |
|--------------|--------|-------|----------|
| FP16 (printed) | ~250MB | Fast | High |
| 4-bit (printed) | ~65MB | Very Fast | Good |
| FP16 (handwritten) | ~250MB | Fast | Medium-High |
| 4-bit (handwritten) | ~65MB | Very Fast | Medium |

**Note**: Handwritten accuracy depends heavily on handwriting quality and consistency.

## Model Variants Comparison

| Variant | Best For | Accuracy | Notes |
|---------|----------|----------|-------|
| trocr-small-printed | Typed/printed text | High | Documents, receipts, forms |
| trocr-small-handwritten | Handwriting | Medium-High | Notes, signatures (if trained on IAM) |

## Best Use Cases

TrOCR-small excels at:

- ✅ Printed text recognition (documents, receipts)
- ✅ Handwritten text recognition (notes, forms)
- ✅ Single-line text OCR
- ✅ Form field extraction
- ✅ Receipt parsing
- ✅ Document digitization
- ✅ On-device OCR
- ✅ Privacy-sensitive document processing

## Limitations

- **Single-line**: Best for single-line text (use line segmentation for multi-line)
- **Language**: Primarily trained on English (may struggle with other languages)
- **Layout**: Doesn't handle complex layouts (needs preprocessing)
- **Accuracy**: Lower than larger models (base/large variants)

**Trade-off**: Acceptable for 60M param budget and on-device constraints.

## Integration with Other Models

Combine with vision models for document understanding:

```python
from smlx.models.TrOCR_small import load as load_ocr, recognize
from smlx.models.SmolVLM_256M_Instruct import load as load_vlm, generate

# Pipeline: TrOCR for text extraction + SmolVLM for understanding
ocr_model, ocr_processor = load_ocr("microsoft/trocr-small-printed")
vlm_model, vlm_processor = load_vlm()

# Step 1: Extract text with OCR
image = Image.open("invoice.jpg")
extracted_text = recognize(ocr_model, ocr_processor, image)

# Step 2: Understand with VLM
prompt = f"Given this extracted text: '{extracted_text}', what is the invoice total?"
answer = generate(vlm_model, vlm_processor, image, prompt)

print(f"Extracted: {extracted_text}")
print(f"Answer: {answer}")
```

## References

- **HuggingFace (Printed)**: [microsoft/trocr-small-printed](https://huggingface.co/microsoft/trocr-small-printed)
- **HuggingFace (Handwritten)**: [microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten)
- **Paper**: [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)
- **GitHub**: [microsoft/unilm/trocr](https://github.com/microsoft/unilm/tree/master/trocr)

## Citation

```bibtex
@article{li2021trocr,
  title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models},
  author={Li, Minghao and Lv, Tengchao and Chen, Jingye and Cui, Lei and Lu, Yijuan and Florencio, Dinei and Zhang, Cha and Li, Zhoujun and Wei, Furu},
  journal={arXiv preprint arXiv:2109.10282},
  year={2021}
}
```

## License

MIT

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
