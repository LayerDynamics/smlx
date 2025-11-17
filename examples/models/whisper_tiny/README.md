# Whisper-tiny Examples

This directory contains examples demonstrating how to use the Whisper-tiny model for automatic speech recognition.

## Examples

### 1. Basic Transcription (`basic_transcription.py`)

Simple example showing how to transcribe a single audio file.

```bash
# Basic transcription with auto-detected language
python basic_transcription.py speech.wav

# Transcribe Spanish audio
python basic_transcription.py speech.wav --language es

# Translate Spanish to English
python basic_transcription.py speech.wav --language es --task translate

# Show segments with timestamps
python basic_transcription.py speech.wav --show-segments
```

**Features:**

- Automatic language detection
- Multilingual transcription (99 languages)
- Translation to English
- Segment-level timestamps
- Flexible temperature sampling

### 2. Batch Transcription (`batch_transcription.py`)

Process multiple audio files efficiently.

```bash
# Transcribe multiple files
python batch_transcription.py audio1.wav audio2.wav audio3.wav

# Transcribe all WAV files in directory
python batch_transcription.py audio_dir/*.wav

# Save results to JSON
python batch_transcription.py audio_dir/*.wav --output results.json

# Process with specific language
python batch_transcription.py *.wav --language es --output spanish_results.json
```

**Features:**

- Batch processing with single model load
- Error handling per file
- JSON output for downstream processing
- Progress tracking
- Summary statistics

## Supported Audio Formats

Whisper-tiny uses FFmpeg for audio loading, supporting most common formats:

- WAV, MP3, FLAC
- M4A, OGG, OPUS
- WMA, AAC
- And many more...

## Supported Languages

Whisper-tiny supports 99 languages including:

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Russian (ru)
- Portuguese (pt)
- And many more...

See `smlx/models/Whisper_tiny/tokenizer.py` for the complete list.

## Model Variants

You can use different Whisper models by specifying the `--model` argument:

```bash
# Whisper-tiny (39M params, fastest)
python basic_transcription.py audio.wav --model mlx-community/whisper-tiny

# Whisper-base (74M params)
python basic_transcription.py audio.wav --model mlx-community/whisper-base

# Whisper-small (244M params)
python basic_transcription.py audio.wav --model mlx-community/whisper-small
```

## Programmatic Usage

```python
from smlx.models.Whisper_tiny import load, transcribe

# Load model
model, tokenizer = load("mlx-community/whisper-tiny")

# Transcribe audio
result = transcribe(
    "speech.wav",
    model,
    tokenizer,
    language="en",  # Optional: auto-detected if None
    task="transcribe",  # or "translate" for X->English
    temperature=0.0,  # Greedy sampling
    verbose=True,  # Show progress
)

# Access results
print(result["text"])  # Full transcription
print(result["language"])  # Detected language

# Access segments with timestamps
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

## Advanced Features

### Temperature Fallback

Whisper automatically tries multiple temperatures if transcription quality is low:

```python
result = transcribe(
    "speech.wav",
    model,
    tokenizer,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Try increasingly creative sampling
    compression_ratio_threshold=2.4,  # Fallback if compression ratio too high
    logprob_threshold=-1.0,  # Fallback if log probability too low
)
```

### No-Speech Detection

Automatically skip silent segments:

```python
result = transcribe(
    "speech.wav",
    model,
    tokenizer,
    no_speech_threshold=0.6,  # Skip if no-speech probability > 0.6
)
```

### Conditioning on Previous Text

Use previous transcription as context:

```python
result = transcribe(
    "speech.wav",
    model,
    tokenizer,
    condition_on_previous_text=True,  # Use previous output as prompt
    initial_prompt="Custom vocabulary: quantum, photonics",  # Initial context
)
```

## Requirements

- Python >=3.9
- MLX framework (Apple Silicon)
- FFmpeg (for audio loading)

```bash
# Install dependencies
pip install -e ".[audio]"

# Install FFmpeg (macOS)
brew install ffmpeg
```

## Performance

Whisper-tiny is optimized for Apple Silicon:

| Model | Parameters | Speed (M4) | Accuracy |
|-------|-----------|------------|----------|
| tiny | 39M | ~10x realtime | Good |
| base | 74M | ~5x realtime | Better |
| small | 244M | ~2x realtime | Best |

Processing speed depends on audio duration, batch size, and hardware.

## Troubleshooting

### FFmpeg not found

Install FFmpeg:

```bash
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
```

### Out of memory

Try:

- Using smaller batch size: `batch_size=4`
- Using fp16 precision: `fp16=True` (default)
- Using a smaller model variant

### Poor transcription quality

Try:

- Specifying the correct language: `language="es"`
- Using higher temperature: `temperature=0.5`
- Using a larger model variant (base or small)
- Ensuring audio quality is good (clean, 16kHz+)

### 3. Quantization Example (`quantization_example.py`)

Demonstrates model quantization for reduced memory usage and faster inference.

```bash
# Quantize model to 4-bit
python quantization_example.py --quantize --bits 4

# Compare full precision vs quantized
python quantization_example.py --compare audio.wav

# Transcribe with quantized model
python quantization_example.py --model whisper-tiny-4bit audio.wav
```

**Features:**

- 4-bit and 8-bit quantization
- Performance benchmarking
- WER/CER quality comparison
- Memory footprint reduction (4x-8x smaller)

### 4. Streaming Partial Results (`streaming_partial_results.py`)

Demonstrates real-time streaming transcription with partial and final results.

```bash
# Run all streaming demonstrations
python streaming_partial_results.py

# Shows:
# - Partial result detection strategies (punctuation, buffer, VAD)
# - Result throttling behavior
# - Real-time streaming from file (if audio available)
```

**Features:**

- Partial vs final result detection
- Multiple detection strategies (punctuation, VAD, buffer exhaustion)
- Result throttling demonstration
- Real-time simulation from pre-recorded files
- VAD integration (optional)

## More Advanced Features

### Word-Level Timestamps

Get precise word-level timestamps using Dynamic Time Warping on cross-attention patterns:

```python
from smlx.models.Whisper_tiny import load, transcribe

model, tokenizer = load()
result = transcribe("speech.wav", model, tokenizer, word_timestamps=True)

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"[{word['start']:.2f}s - {word['end']:.2f}s]: {word['word']}")
```

**Features:**

- DTW alignment using cross-attention weights
- Automatic punctuation merging
- Handles both space-separated and non-space-separated languages
- Configurable median filtering for smoothing

### Beam Search Decoding

Improve transcription quality with beam search instead of greedy decoding:

```python
from smlx.models.Whisper_tiny import load, transcribe

model, tokenizer = load()
result = transcribe(
    "speech.wav",
    model,
    tokenizer,
    beam_size=5,  # Use 5 beams (default: greedy)
    patience=1.0,  # Beam search patience
)
```

**Features:**

- **Fully implemented beam search** (exceeds reference implementations!)
- Configurable beam width
- Length-normalized scoring (Google NMT penalty)
- Early stopping with patience parameter
- Better quality than greedy decoding for challenging audio
- Efficient KV cache rearrangement for beam tracking

**Note:** Our beam search implementation is complete and functional, while many reference implementations (mlx-examples, lightning-whisper-mlx) don't have beam search at all!

### Voice Activity Detection (VAD)

Pre-segment audio with VAD to skip silent regions and improve efficiency:

```python
from smlx.models.Whisper_tiny import load
from smlx.models.Whisper_tiny.vad import transcribe_with_vad

model, tokenizer = load()
result = transcribe_with_vad(
    "audio.wav",
    model,
    tokenizer,
    vad_threshold=0.5,  # Detection threshold
    min_gap=0.3,        # Minimum gap between segments
)

# View detected speech segments
for vad_seg in result["vad_segments"]:
    print(f"Speech: {vad_seg['start']:.2f}s - {vad_seg['end']:.2f}s")
```

**Features:**

- Silero VAD integration (requires `pip install silero-vad`)
- Automatic segment merging
- Configurable thresholds and durations
- Skips silent regions for faster processing

### Streaming Transcription

Real-time transcription with sliding window approach:

```python
from smlx.models.Whisper_tiny import load
from smlx.models.Whisper_tiny.streaming import StreamingTranscriber, StreamingConfig

model, tokenizer = load()

# Configure streaming with partial results
config = StreamingConfig(
    chunk_duration=5.0,
    enable_partial_results=True,  # Enable partial results
    partial_result_interval=0.5,   # Emit partials every 0.5s
    enable_vad=False,              # Optional VAD for better detection
)

transcriber = StreamingTranscriber(model, tokenizer, config=config)

# Process audio chunks as they arrive
for audio_chunk in audio_stream:
    result = transcriber.process_chunk(audio_chunk)
    if result:
        # result.is_final indicates if this is a final or partial result
        result_type = "FINAL" if result.is_final else "partial"
        print(f"[{result.start_time:.2f}s] ({result_type}): {result.text}")
```

**Microphone streaming:**

```python
from smlx.models.Whisper_tiny.streaming import MicrophoneStream

with MicrophoneStream() as stream:
    for audio_chunk in stream:
        result = transcriber.process_chunk(audio_chunk)
        if result:
            if result.is_final:
                print(f"FINAL: {result.text}")
            else:
                print(f"partial: {result.text}", end="\r")  # Live update
```

**Features:**

- Sliding window with configurable overlap
- Real-time microphone support (requires `pip install sounddevice`)
- Buffered audio management
- Configurable chunk durations
- **Partial results** - Get intermediate transcriptions before final
- **Multiple detection strategies** - Punctuation, VAD, and buffer-based
- **Result throttling** - Control partial result frequency
- **VAD integration** - Better partial/final detection (optional)

## Installation

```bash
# Base installation
pip install -e ".[audio]"

# Install optional dependencies for advanced features
pip install silero-vad     # For VAD integration
pip install sounddevice    # For microphone streaming
pip install jiwer          # For WER/CER evaluation
```

## References

- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [SMLX Documentation](../../docs/)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Dynamic Time Warping for Word Timestamps](https://github.com/openai/whisper/discussions/1363)
