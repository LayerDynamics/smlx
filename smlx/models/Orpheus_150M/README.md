# Orpheus-150M

A lightweight 150M parameter text-to-speech (TTS) model for natural-sounding voice synthesis, part of the Orpheus family of small TTS models.

## Model Details

- **Size**: 150M parameters
- **Type**: Text-to-Speech (TTS)
- **Output**: Waveform audio synthesis
- **Sample Rate**: 16kHz/24kHz (configurable)
- **Memory**: ~600MB (FP16), ~150MB (4-bit quantized)
- **Variants**: 150M, 400M (use 150M for SMLX)
- **License**: Check HuggingFace model card

## Why Orpheus-150M for SMLX?

Orpheus-150M is the **smallest TTS model** for SMLX:

- Lightest in Orpheus family (150M params)
- Natural-sounding speech synthesis
- Fast inference on M4
- Perfect for on-device TTS
- Good quality-to-size ratio

## Installation

```bash
pip install smlx[audio]
```

## Quick Start

### Python API

```python
from smlx.models.Orpheus_150M import load, synthesize

# Load model
model, processor = load()

# Generate speech
text = "Hello, this is a test of text to speech synthesis."
audio = synthesize(
    model=model,
    processor=processor,
    text=text,
    sample_rate=24000
)

# Save to file
import soundfile as sf
sf.write("output.wav", audio, 24000)
```

### Command Line

```bash
# Synthesize speech
smlx speak \
  --model Orpheus-150M \
  --text "Hello world" \
  --output speech.wav
```

## Usage Examples

### Basic TTS

```python
from smlx.models.Orpheus_150M import load, synthesize

model, processor = load()

text = "The quick brown fox jumps over the lazy dog."
audio = synthesize(model, processor, text)

# Play or save audio
import soundfile as sf
sf.write("speech.wav", audio, 24000)
```

### Batch Synthesis

```python
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

for i, text in enumerate(texts):
    audio = synthesize(model, processor, text)
    sf.write(f"speech_{i}.wav", audio, 24000)
```

### Streaming TTS

```python
def stream_synthesis(model, processor, text, chunk_size=50):
    """Generate speech in chunks for lower latency"""
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        audio_chunk = synthesize(model, processor, chunk)
        yield audio_chunk

# Use in streaming application
for audio_chunk in stream_synthesis(model, processor, long_text):
    # Play audio_chunk immediately
    pass
```

## Quantization

### 4-bit Quantization

```bash
python -m smlx.tools.convert2mlx \
  --hf-path orpheus-150m \
  --mlx-path ./models/orpheus-150m-4bit \
  --quantize \
  --q-bits 4
```

**Benefits:**

- ~75% size reduction (~600MB → ~150MB)
- Faster synthesis on M4

## Performance on M4

| Configuration | Memory | Speed (RTF) | Quality |
|--------------|--------|-------------|---------|
| FP16 | ~600MB | 0.1x (10x faster than real-time) | Excellent |
| 4-bit | ~150MB | 0.08x (12x faster) | Very Good |

**RTF (Real-Time Factor)**: 0.1 means 10 seconds of audio generated per second.

## Best Use Cases

Orpheus-150M excels at:

- ✅ On-device text-to-speech
- ✅ Voice assistants
- ✅ Audiobook generation
- ✅ Accessibility (screen readers)
- ✅ Content narration
- ✅ Voice notifications
- ✅ Educational applications

## References

- **HuggingFace**: Search for "Orpheus-150M" on HuggingFace
- **Related**: Part of small TTS model family

## License

Check model card on HuggingFace

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
