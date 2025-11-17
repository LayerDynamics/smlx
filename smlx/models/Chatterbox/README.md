# Chatterbox

A 500M parameter text-to-speech model built on a 0.5B Llama backbone with AI voice cloning capabilities and configurable expressiveness.

## Model Details

- **Size**: 500M parameters
- **Type**: Text-to-Speech (TTS) with Voice Cloning
- **Architecture**: Built on 0.5B Llama
- **Features**: Voice cloning, expressiveness control, natural speech
- **Memory**: ~2GB (FP16), ~0.5GB (4-bit quantized)
- **License**: Check HuggingFace model card

## Why Chatterbox for SMLX?

Chatterbox is the **voice cloning TTS** for SMLX:

- 500M params - small enough for on-device
- AI voice cloning from short samples
- Configurable expressiveness (emotion, tone)
- Natural-sounding speech
- Low WER (Word Error Rate)
- Good for personalized voice applications

## Installation

```bash
pip install smlx[audio]
```

## Quick Start

### Python API

```python
from smlx.models.Chatterbox import load, synthesize

# Load model
model, processor = load()

# Generate speech
text = "Hello, I can speak with natural expressiveness."
audio = synthesize(
    model=model,
    processor=processor,
    text=text,
    expressiveness=0.7  # 0-1 scale
)

# Save audio
import soundfile as sf
sf.write("output.wav", audio, 24000)
```

### Command Line

```bash
# Synthesize with expressiveness
smlx speak \
  --model Chatterbox \
  --text "Hello world" \
  --expressiveness 0.7 \
  --output speech.wav
```

## Usage Examples

### Basic TTS with Expressiveness

```python
from smlx.models.Chatterbox import load, synthesize

model, processor = load()

# Neutral (low expressiveness)
audio_neutral = synthesize(model, processor, "This is neutral speech.", expressiveness=0.2)

# Expressive (high expressiveness)
audio_expressive = synthesize(model, processor, "This is expressive speech!", expressiveness=0.9)
```

### Voice Cloning

```python
from smlx.models.Chatterbox import load, synthesize, clone_voice

model, processor = load()

# Provide reference audio (3-10 seconds)
import soundfile as sf
reference_audio, sr = sf.read("voice_sample.wav")

# Clone voice
voice_embedding = clone_voice(model, reference_audio, sr)

# Generate speech with cloned voice
audio = synthesize(
    model=model,
    processor=processor,
    text="This is me speaking with a cloned voice.",
    voice_embedding=voice_embedding
)

sf.write("cloned_speech.wav", audio, 24000)
```

### Emotion Control

```python
# Excited/Happy
audio_happy = synthesize(
    model, processor,
    "I'm so excited about this!",
    expressiveness=0.9,
    emotion="happy"
)

# Calm/Neutral
audio_calm = synthesize(
    model, processor,
    "This is a calm statement.",
    expressiveness=0.3,
    emotion="neutral"
)
```

## Quantization

### 4-bit Quantization

```bash
python -m smlx.tools.convert2mlx \
  --hf-path chatterbox-500m \
  --mlx-path ./models/chatterbox-4bit \
  --quantize \
  --q-bits 4
```

**Benefits:**

- ~75% size reduction (~2GB → ~0.5GB)
- Faster synthesis

## Performance on M4

| Configuration | Memory | Speed (RTF) | Quality |
|--------------|--------|-------------|---------|
| FP16 | ~2.0GB | 0.15x | Excellent |
| 4-bit | ~0.5GB | 0.12x | Very Good |

## Best Use Cases

Chatterbox excels at:

- ✅ Personalized voice assistants (voice cloning)
- ✅ Expressive audiobook narration
- ✅ Character voices (games, apps)
- ✅ Emotional speech synthesis
- ✅ Content creation with custom voices
- ✅ Accessibility with personalized voices
- ✅ Voice preservation

## References

- **HuggingFace**: Search for "Chatterbox" on HuggingFace
- **Architecture**: Built on 0.5B Llama

## License

Check model card on HuggingFace

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
