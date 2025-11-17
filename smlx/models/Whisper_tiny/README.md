# Whisper-tiny

OpenAI's smallest Whisper model for automatic speech recognition (ASR) with 39M parameters, offering multilingual transcription in a tiny package.

## Model Details

- **Size**: 39M parameters (~150MB)
- **Type**: Speech-to-Text (ASR)
- **Languages**: 99 languages (multilingual) + English-only variant
- **Training**: 680k hours of labeled data
- **License**: MIT (OpenAI Whisper)
- **HuggingFace**: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- **MLX**: [mlx-community/whisper-tiny](https://huggingface.co/mlx-community/whisper-tiny)

## Why Whisper-tiny for SMLX?

Whisper-tiny is the **lightweight ASR** for SMLX:

- Smallest Whisper variant (39M params)
- Multilingual support (99 languages!)
- Fast inference on M4
- MIT license (permissive)
- Well-established and reliable
- Perfect for on-device transcription

## Installation

```bash
pip install smlx[audio]
```

## Quick Start

### Python API

```python
from smlx.models.Whisper_tiny import load, transcribe
import soundfile as sf

# Load model
model, processor = load("mlx-community/whisper-tiny")

# Load audio
audio, sr = sf.read("speech.wav")

# Transcribe
transcription = transcribe(
    model=model,
    processor=processor,
    audio=audio,
    sample_rate=sr,
    language="en"  # Optional: auto-detect if not specified
)

print(transcription["text"])
```

### Command Line

```bash
# Transcribe audio file
smlx transcribe \
  --model whisper-tiny \
  --audio speech.mp3 \
  --language en

# Auto-detect language
smlx transcribe \
  --model whisper-tiny \
  --audio speech.mp3
```

## Converting from HuggingFace

```bash
# Convert with 4-bit quantization
python -m smlx.tools.convert2mlx \
  --hf-path openai/whisper-tiny \
  --mlx-path ./models/whisper-tiny-4bit \
  --quantize \
  --q-bits 4 \
  --skip-multimodal  # Preserve audio encoder quality
```

## Quantization

### 4-bit Quantization

```bash
python -m smlx.tools.convert2mlx \
  --hf-path openai/whisper-tiny \
  --mlx-path ./models/whisper-tiny-4bit \
  --quantize \
  --q-bits 4
```

**Benefits:**

- ~75% size reduction
- Faster inference on M4

## Performance on M4

| Configuration | Memory | Speed | WER (Word Error Rate) |
|--------------|--------|-------|---------------------|
| FP16 | ~150MB | Very Fast | Baseline |
| 4-bit | ~40MB | Ultra Fast | +2-3% |

**Key Strength**: Fastest Whisper variant with multilingual support.

## Usage Examples

### Basic Transcription

```python
from smlx.models.Whisper_tiny import load, transcribe

model, processor = load("mlx-community/whisper-tiny")

# Transcribe
result = transcribe(model, processor, "audio.wav")
print(result["text"])
```

### With Timestamps

```python
result = transcribe(
    model, processor, "audio.wav",
    return_timestamps=True
)

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

### Multilingual

```python
# Auto-detect language
result = transcribe(model, processor, "spanish.wav")
print(f"Detected language: {result['language']}")
print(result["text"])

# Or specify language
result = transcribe(model, processor, "spanish.wav", language="es")
```

## Best Use Cases

Whisper-tiny excels at:

- ✅ On-device speech transcription
- ✅ Real-time captioning
- ✅ Meeting transcription
- ✅ Voice notes to text
- ✅ Multilingual transcription (99 languages)
- ✅ Low-resource environments
- ✅ Fast batch processing

## Limitations

- **Accuracy**: Lower than larger Whisper models (base/small/medium)
- **Accents**: May struggle with heavy accents
- **Background Noise**: Less robust than larger variants
- **Technical Terms**: May struggle with domain-specific vocabulary

**Trade-off**: Acceptable for 39M param budget and exceptional speed.

## References

- **HuggingFace**: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- **MLX**: [mlx-community/whisper-tiny](https://huggingface.co/mlx-community/whisper-tiny)
- **Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **GitHub**: [openai/whisper](https://github.com/openai/whisper)

## License

MIT

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
