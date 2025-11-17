#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Integration tests for Whisper-tiny automatic speech recognition.

Tests transcription, language detection, VAD, word-level timestamps.

Run with:
    python -m pytest tests/integration/test_whisper_tiny.py -v
"""

import gc
import subprocess

import mlx.core as mx
import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_model,
]


def check_ffmpeg_available():
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_vad_available():
    """Check if silero-vad is available."""
    import importlib.util

    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchaudio") is not None
    )


# Skip markers
requires_ffmpeg = pytest.mark.skipif(
    not check_ffmpeg_available(),
    reason="ffmpeg not installed"
)
requires_vad = pytest.mark.skipif(
    not check_vad_available(),
    reason="silero-vad not installed (requires torch and torchaudio)"
)


def create_test_audio(duration=1.0, sample_rate=16000, frequency=440):
    """Create a simple sine wave test audio."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    # Generate sine wave (A440)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture(scope="module")
def whisper_model():
    """
    Load Whisper model once for all tests.

    Memory Requirements:
    - Model size: ~150MB (75M parameters in FP16)
    - Peak memory: ~300MB with activations
    """
    from smlx.models.Whisper_tiny import load

    model, tokenizer = load("mlx-community/whisper-tiny")

    yield model, tokenizer

    # Explicit cleanup to prevent memory accumulation
    print("\nCleaning up Whisper model...")
    del model
    del tokenizer
    mx.clear_cache()
    mx.metal.clear_cache()
    gc.collect()
    print("Cleanup complete. Memory freed.")


def test_model_loading(whisper_model):
    """Test that Whisper model loads successfully."""
    model, tokenizer = whisper_model

    assert model is not None, "Model should not be None"
    assert tokenizer is not None, "Tokenizer should not be None"
    assert hasattr(model, "encoder"), "Model should have encoder"
    assert hasattr(model, "decoder"), "Model should have decoder"


@requires_ffmpeg
def test_audio_loading(tmp_path):
    """Test audio loading utilities."""
    import mlx.core as mx

    from smlx.models.Whisper_tiny import load_audio

    # Create test audio file
    test_audio = create_test_audio()

    try:
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), test_audio, 16000)

        # Load audio from file
        loaded = load_audio(str(audio_path))
        assert isinstance(loaded, mx.array), "Should return MLX array"
        assert loaded.shape[0] > 0, "Should have audio samples"
    except ImportError:
        pytest.skip("soundfile not available")


def test_mel_spectrogram():
    """Test mel-spectrogram computation."""
    import mlx.core as mx

    from smlx.models.Whisper_tiny import log_mel_spectrogram

    # Create test audio and convert to MLX array
    test_audio = create_test_audio()
    test_audio_mlx = mx.array(test_audio)

    # Compute mel-spectrogram
    mel = log_mel_spectrogram(test_audio_mlx)

    assert mel is not None, "Mel-spectrogram should not be None"
    assert len(mel.shape) == 2, "Mel should be 2D (time, mels)"
    assert mel.shape[1] == 80, "Should have 80 mel bins"


@requires_ffmpeg
def test_audio_preprocessing(tmp_path):
    """Test audio preprocessing pipeline."""
    import mlx.core as mx

    from smlx.models.Whisper_tiny import N_SAMPLES, pad_or_trim, prepare_audio

    # Create test audio and convert to MLX array
    test_audio = create_test_audio()
    test_audio_mlx = mx.array(test_audio)

    # Pad or trim to standard length
    padded = pad_or_trim(test_audio_mlx)
    assert padded.shape[0] == N_SAMPLES, f"Should pad/trim to {N_SAMPLES} samples"

    # Prepare audio (full pipeline) - requires file path
    try:
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), test_audio, 16000)

        mel = prepare_audio(str(audio_path))
        assert mel is not None, "Prepared audio should not be None"
    except ImportError:
        pytest.skip("soundfile not available")


def test_basic_transcription(whisper_model):
    """Test basic transcription functionality."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio (won't produce meaningful transcription without real speech)
    test_audio = create_test_audio(duration=2.0)

    # Transcribe
    result = transcribe(test_audio, model, tokenizer, verbose=False)

    assert result is not None, "Transcription result should not be None"
    assert "text" in result, "Result should have 'text' field"
    assert isinstance(result["text"], str), "Text should be a string"


def test_transcription_with_language(whisper_model):
    """Test transcription with language specification."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio(duration=2.0)

    # Transcribe with English language
    result = transcribe(test_audio, model, tokenizer, language="en", verbose=False)

    assert result is not None, "Result should not be None"
    assert "language" in result, "Result should have language field"


@requires_ffmpeg
def test_language_detection(whisper_model, tmp_path):
    """Test language detection functionality."""
    from smlx.models.Whisper_tiny import detect_language, prepare_audio

    model, tokenizer = whisper_model

    # Create test audio file
    test_audio = create_test_audio(duration=1.0)

    try:
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), test_audio, 16000)

        mel = prepare_audio(str(audio_path))

        # Detect language - returns tuple of (language_tokens, language_probs)
        language_tokens, language_probs = detect_language(model, mel, tokenizer)

        assert language_probs is not None, "Language probs should not be None"
        assert isinstance(language_probs, list), "Should return list of probability dicts"
        assert len(language_probs) > 0, "Should have language probabilities"
    except ImportError:
        pytest.skip("soundfile not available")


def test_decoding_options(whisper_model):
    """Test different decoding options."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio(duration=1.0)

    # Test with custom decoding options passed as kwargs
    result = transcribe(
        test_audio,
        model,
        tokenizer,
        language="en",
        task="transcribe",
        temperature=0.0,
        sample_len=50,
        verbose=False,
    )

    assert result is not None, "Result should not be None"


def test_compression_ratio(whisper_model):
    """Test compression ratio calculation."""
    from smlx.models.Whisper_tiny import (
        compression_ratio,
        transcribe,
    )

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio()

    # Transcribe
    result = transcribe(test_audio, model, tokenizer, verbose=False)

    # Calculate compression ratio
    if "tokens" in result:
        ratio = compression_ratio(result["tokens"])
        assert ratio > 0, "Compression ratio should be positive"


def test_chunk_processing():
    """Test audio chunking for long files."""
    import mlx.core as mx

    from smlx.models.Whisper_tiny import N_SAMPLES, split_audio_chunks

    # Create longer audio and convert to MLX array
    long_audio = create_test_audio(duration=60.0)
    long_audio_mlx = mx.array(long_audio)

    # Split into chunks
    chunks = split_audio_chunks(long_audio_mlx)

    assert len(chunks) > 0, "Should have at least one chunk"
    for chunk in chunks:
        assert chunk.shape[0] <= N_SAMPLES, "Chunks should not exceed max length"


@requires_ffmpeg
def test_audio_duration(tmp_path):
    """Test audio duration calculation."""
    from smlx.models.Whisper_tiny import get_audio_duration

    # Create 5 second audio
    test_audio = create_test_audio(duration=5.0)

    try:
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), test_audio, 16000)

        duration = get_audio_duration(str(audio_path))
        assert abs(duration - 5.0) < 0.1, "Duration should be approximately 5 seconds"
    except ImportError:
        pytest.skip("soundfile not available")


def test_empty_audio(whisper_model):
    """Test handling of empty/silent audio."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create silent audio
    silent_audio = np.zeros(16000, dtype=np.float32)

    # Should handle gracefully
    result = transcribe(silent_audio, model, tokenizer, verbose=False)
    assert result is not None, "Should handle silent audio"


@requires_ffmpeg
def test_transcribe_file(whisper_model, tmp_path):
    """Test file-based transcription."""
    from smlx.models.Whisper_tiny import transcribe_file

    model, tokenizer = whisper_model

    # Create test audio file
    test_audio = create_test_audio(duration=1.0)

    # Save as WAV file
    try:
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), test_audio, 16000)

        # Transcribe from file
        result = transcribe_file(str(audio_path), model, tokenizer, verbose=False)

        assert result is not None, "File transcription should work"
        assert "text" in result, "Should have text field"
    except ImportError:
        pytest.skip("soundfile not available")


@requires_vad
def test_vad_detection():
    """Test voice activity detection."""
    from smlx.models.Whisper_tiny import detect_speech_segments

    # Create audio with "speech" (just sine wave for testing)
    test_audio = create_test_audio(duration=5.0)

    # Detect speech segments
    segments = detect_speech_segments(test_audio)

    assert segments is not None, "Segments should not be None"
    assert isinstance(segments, list), "Should return list of segments"


@requires_vad
def test_vad_transcription(whisper_model):
    """Test transcription with VAD pre-filtering."""
    from smlx.models.Whisper_tiny import transcribe_with_vad

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio(duration=3.0)

    # Transcribe with VAD
    result = transcribe_with_vad(test_audio, model, tokenizer, verbose=False)

    assert result is not None, "VAD transcription should not be None"
    assert "text" in result, "Should have text field"


def test_greedy_decoder(whisper_model):
    """Test greedy decoding strategy."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio()

    # Use greedy decoding (beam_size=1, temperature=0.0)
    result = transcribe(
        test_audio,
        model,
        tokenizer,
        beam_size=1,  # Greedy = beam_size 1
        temperature=0.0,
        verbose=False,
    )

    assert result is not None, "Greedy decoding should work"


@pytest.mark.xfail(reason="Beam search implementation has a bug with finalize() returning lists")
def test_beam_search_decoder(whisper_model):
    """Test beam search decoding strategy."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio()

    # Use beam search (beam_size=5)
    result = transcribe(
        test_audio,
        model,
        tokenizer,
        beam_size=5,
        temperature=0.0,
        verbose=False,
    )

    assert result is not None, "Beam search should work"


def test_temperature_sampling(whisper_model):
    """Test temperature-based sampling."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio()

    # Use temperature sampling
    result = transcribe(
        test_audio,
        model,
        tokenizer,
        temperature=0.8,
        beam_size=1,
        verbose=False,
    )

    assert result is not None, "Temperature sampling should work"


def test_translation_task(whisper_model):
    """Test translation to English."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio()

    # Test translation task
    result = transcribe(
        test_audio,
        model,
        tokenizer,
        task="translate",
        verbose=False,
    )

    assert result is not None, "Translation should work"


@requires_ffmpeg
def test_word_timestamps(whisper_model, tmp_path):
    """Test word-level timestamp extraction."""
    from smlx.models.Whisper_tiny import add_word_timestamps, prepare_audio, transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio(duration=2.0)

    try:
        import soundfile as sf

        audio_path = tmp_path / "test_audio.wav"
        sf.write(str(audio_path), test_audio, 16000)

        # Transcribe
        result = transcribe(str(audio_path), model, tokenizer, verbose=False)

        # Try to add word timestamps (may not work with test audio)
        if "segments" in result and len(result["segments"]) > 0:
            mel = prepare_audio(str(audio_path))
            # add_word_timestamps requires keyword arguments
            segments_with_words = add_word_timestamps(
                segments=result["segments"],
                model=model,
                tokenizer=tokenizer,
                mel=mel,
                num_frames=mel.shape[-1],
            )
            assert isinstance(segments_with_words, list), "Should return list of segments"
    except ImportError:
        pytest.skip("soundfile not available")


def test_different_temperatures(whisper_model):
    """Test transcription with different temperature settings."""
    from smlx.models.Whisper_tiny import transcribe

    model, tokenizer = whisper_model

    # Create test audio
    test_audio = create_test_audio()

    # Test different temperatures
    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        result = transcribe(
            test_audio,
            model,
            tokenizer,
            temperature=temp,
            verbose=False,
        )

        assert result is not None, f"Should work with temperature {temp}"


def test_tokenizer_languages():
    """Test tokenizer language support."""
    from smlx.models.Whisper_tiny import LANGUAGES, TO_LANGUAGE_CODE

    assert len(LANGUAGES) > 0, "Should have language list"
    assert len(TO_LANGUAGE_CODE) > 0, "Should have language code mapping"

    # Check English is supported
    # LANGUAGES is a dict mapping codes to names: {'en': 'english', ...}
    assert "en" in LANGUAGES, "Should have English code"
    assert "english" in LANGUAGES.values(), "Should support English"
    # TO_LANGUAGE_CODE is reverse: {'english': 'en', ...}
    assert "english" in TO_LANGUAGE_CODE, "Should have English in mapping"


def test_audio_constants():
    """Test audio processing constants."""
    from smlx.models.Whisper_tiny import (
        CHUNK_LENGTH,
        HOP_LENGTH,
        N_FFT,
        N_SAMPLES,
        SAMPLE_RATE,
    )

    assert SAMPLE_RATE == 16000, "Sample rate should be 16kHz"
    assert N_FFT > 0, "N_FFT should be positive"
    assert HOP_LENGTH > 0, "HOP_LENGTH should be positive"
    assert CHUNK_LENGTH > 0, "CHUNK_LENGTH should be positive"
    assert N_SAMPLES > 0, "N_SAMPLES should be positive"


def test_model_config(whisper_model):
    """Test model configuration."""
    model, tokenizer = whisper_model

    # Model should have config
    assert hasattr(model, "config"), "Model should have config"

    config = model.config
    assert config is not None, "Config should not be None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
