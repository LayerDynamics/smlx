# Silero VAD

An ultra-tiny 5M parameter Voice Activity Detection (VAD) model that detects speech vs silence in audio with enterprise-grade accuracy.

## Model Details

- **Size**: ~5M parameters (~2MB!)
- **Type**: Voice Activity Detection (VAD)
- **Output**: Speech/Silence timestamps
- **Languages**: Language-agnostic
- **Sample Rates**: 8kHz, 16kHz
- **License**: MIT
- **GitHub**: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- **HuggingFace**: [snakers4/silero-vad](https://huggingface.co/snakers4/silero-vad)

## Why Silero VAD for SMLX?

Silero VAD is the **ultra-lightweight audio preprocessor** for SMLX:

- Incredibly tiny (5M params, ~2MB!)
- Enterprise-grade accuracy
- MIT license (very permissive)
- Fast inference
- Language-agnostic
- Perfect for preprocessing audio before ASR
- Saves compute by skipping silent segments

## Installation

```bash
pip install smlx[audio]
```

## Quick Start

### Python API

```python
from smlx.models.Silero_VAD import load, detect_speech
import soundfile as sf

# Load model
model = load()

# Load audio
audio, sr = sf.read("audio.wav")

# Detect speech segments
speech_timestamps = detect_speech(
    model=model,
    audio=audio,
    sample_rate=sr,
    threshold=0.5  # Confidence threshold
)

for segment in speech_timestamps:
    print(f"Speech: {segment['start']:.2f}s - {segment['end']:.2f}s")
```

### Command Line

```bash
# Detect speech in audio file
smlx vad \
  --model silero-vad \
  --audio recording.wav \
  --output segments.json
```

## Usage Examples

### Basic VAD

```python
from smlx.models.Silero_VAD import load, detect_speech

model = load()
timestamps = detect_speech(model, audio, sample_rate=16000)

print(f"Found {len(timestamps)} speech segments")
```

### Preprocessing for ASR

```python
from smlx.models.Silero_VAD import load, detect_speech
from smlx.models.Whisper_tiny import load as load_whisper, transcribe

# Step 1: Detect speech segments
vad_model = load()
speech_segments = detect_speech(vad_model, audio, sample_rate=16000)

# Step 2: Only transcribe speech segments
whisper_model, processor = load_whisper()

transcriptions = []
for segment in speech_segments:
    start_sample = int(segment['start'] * 16000)
    end_sample = int(segment['end'] * 16000)
    audio_segment = audio[start_sample:end_sample]

    result = transcribe(whisper_model, processor, audio_segment)
    transcriptions.append({
        "start": segment['start'],
        "end": segment['end'],
        "text": result["text"]
    })

for t in transcriptions:
    print(f"[{t['start']:.2f}s]: {t['text']}")
```

### Remove Silence from Audio

```python
import numpy as np

def remove_silence(audio, sample_rate, vad_model):
    speech_segments = detect_speech(vad_model, audio, sample_rate)

    speech_audio = []
    for segment in speech_segments:
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)
        speech_audio.append(audio[start_sample:end_sample])

    return np.concatenate(speech_audio) if speech_audio else np.array([])

# Remove silence
model = load()
clean_audio = remove_silence(audio, 16000, model)
print(f"Removed {len(audio) - len(clean_audio)} silent samples")
```

### Real-time VAD (Streaming)

```python
def process_stream(vad_model, audio_stream, sample_rate=16000):
    chunk_size = sample_rate // 2  # 0.5 second chunks

    for i in range(0, len(audio_stream), chunk_size):
        chunk = audio_stream[i:i+chunk_size]
        has_speech = detect_speech(vad_model, chunk, sample_rate)

        if has_speech:
            print(f"Speech detected at {i/sample_rate:.2f}s")
            # Process speech chunk...
        else:
            print(f"Silence at {i/sample_rate:.2f}s")
```

## Performance on M4

| Metric | Value |
|--------|-------|
| Model Size | ~2MB |
| Memory Usage | ~10MB |
| Latency | <1ms per 100ms audio |
| Accuracy | Enterprise-grade |

**Key Strength**: Ultra-tiny with excellent accuracy - perfect preprocessing step.

## Best Use Cases

Silero VAD excels at:

- ✅ Preprocessing audio for ASR (skip silent parts)
- ✅ Voice activity detection in calls/meetings
- ✅ Audio segmentation
- ✅ Silence removal
- ✅ Real-time speech detection
- ✅ Saving compute (only process speech segments)
- ✅ Speaker diarization preprocessing
- ✅ Wake word detection preprocessing

## Integration Example

Combine with other SMLX models:

```python
from smlx.models.Silero_VAD import load as load_vad
from smlx.models.Whisper_tiny import load as load_asr

# Pipeline: VAD → ASR
vad_model = load_vad()
asr_model, processor = load_asr()

# Step 1: Detect speech
speech_segments = detect_speech(vad_model, audio, 16000)

# Step 2: Transcribe only speech (saves ~70% compute!)
for segment in speech_segments:
    audio_chunk = audio[int(segment['start']*16000):int(segment['end']*16000)]
    transcription = transcribe(asr_model, processor, audio_chunk)
    print(f"[{segment['start']:.1f}s]: {transcription['text']}")
```

## Known Issues and Testing

### Integration Test Mac Freeze (RESOLVED)

**Issue**: Running the full integration test suite (22 tests) could freeze the Mac.

**Root Cause**:

- Resource exhaustion from streaming buffer processing
- Metal GPU command buffer accumulation
- Module-scoped fixtures keeping models in memory

**Solution Applied**:

1. **Function-scoped fixtures** - Models load/cleanup for each test
2. **Timeout protection** - MAX_ITERATIONS limit in streaming loop
3. **MLX evaluation checkpoints** - Prevents lazy computation buildup
4. **pytest-timeout** - Global 60-second timeout for all tests

### Safe Testing Workflow

```bash
# Run all tests safely (with function-scoped fixtures)
pytest tests/integration/test_silerovad.py -v

# Skip heavy streaming tests if needed
pytest tests/integration/test_silerovad.py -m "not streaming" -v

# Run individual tests
pytest tests/integration/test_silerovad.py::test_basic_speech_detection -v
```

### Performance Notes

- **Test Execution**: Function-scoped fixtures make tests slower but prevent resource issues
- **Streaming Tests**: Marked with `@pytest.mark.streaming` and `@pytest.mark.heavy_memory`
- **Timeout Protection**: All tests have 60-second timeout to prevent infinite hangs
- **Buffer Safety**: StreamingVAD has max iteration limit (1000) and buffer size checks

For more details, see [TESTING_GUIDE.md](../../../TESTING_GUIDE.md) in the project root.

## References

- **GitHub**: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- **HuggingFace**: [snakers4/silero-vad](https://huggingface.co/snakers4/silero-vad)
- **Blog**: [Silero Models](https://github.com/snakers4/silero-models)

## License

MIT

---

**Part of the SMLX (smol MLX) project** - Small models optimized for Apple M4 chipsets.
