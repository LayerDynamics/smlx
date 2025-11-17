"""
Tests for Whisper-tiny model.

Tests model loading, audio processing, tokenization, and basic inference.
"""

import mlx.core as mx
import numpy as np
import pytest

# Skip if Whisper dependencies not available
pytest.importorskip("tiktoken")


@pytest.fixture
def sample_audio():
    """Generate sample audio waveform (1 second at 16kHz)."""
    # Simple sine wave
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return mx.array(audio.astype(np.float32))


@pytest.fixture
def sample_mel():
    """Generate sample mel spectrogram (30 seconds, 80 mel bins)."""
    # Create random mel spectrogram matching MLX Conv1d NLC format
    mel = mx.random.normal((3000, 80))  # (n_frames, n_mels)
    return mel


class TestWhisperConfig:
    """Test Whisper model configuration."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values for Whisper-tiny."""
        from smlx.models.Whisper_tiny.model import ModelConfig

        config = ModelConfig()

        assert config.n_mels == 80
        assert config.n_audio_ctx == 1500
        assert config.n_audio_state == 384
        assert config.n_audio_head == 6
        assert config.n_audio_layer == 4
        assert config.n_vocab == 51865
        assert config.n_text_ctx == 448
        assert config.n_text_state == 384
        assert config.n_text_head == 6
        assert config.n_text_layer == 4
        assert config.dtype == "float16"

    def test_model_config_from_dict(self):
        """Test ModelConfig.from_dict()."""
        from smlx.models.Whisper_tiny.model import ModelConfig

        config_dict = {
            "n_mels": 128,
            "n_audio_layer": 8,
            "dtype": "float32",
            "extra_field": "ignored",  # Should be ignored
        }

        config = ModelConfig.from_dict(config_dict)

        assert config.n_mels == 128
        assert config.n_audio_layer == 8
        assert config.dtype == "float32"
        # Check defaults for unspecified fields
        assert config.n_audio_ctx == 1500

    def test_model_config_to_dict(self):
        """Test ModelConfig.to_dict()."""
        from smlx.models.Whisper_tiny.model import ModelConfig

        config = ModelConfig(n_mels=128, dtype="float32")
        config_dict = config.to_dict()

        assert config_dict["n_mels"] == 128
        assert config_dict["dtype"] == "float32"
        assert config_dict["n_audio_ctx"] == 1500


class TestWhisperTokenizer:
    """Test Whisper tokenizer."""

    def test_get_tokenizer_english(self):
        """Test getting English-only tokenizer."""
        from smlx.models.Whisper_tiny.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(multilingual=False)

        assert tokenizer.language is None
        assert tokenizer.task is None
        assert tokenizer.num_languages == 99

    def test_get_tokenizer_multilingual(self):
        """Test getting multilingual tokenizer."""
        from smlx.models.Whisper_tiny.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(multilingual=True, language="es", task="transcribe")

        assert tokenizer.language == "es"
        assert tokenizer.task == "transcribe"
        assert len(tokenizer.sot_sequence) == 3  # SOT + language + task

    def test_special_tokens(self):
        """Test special token properties."""
        from smlx.models.Whisper_tiny.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(multilingual=True, language="en")

        assert tokenizer.sot > 0
        assert tokenizer.eot > 0
        assert tokenizer.transcribe > 0
        assert tokenizer.translate > 0
        assert tokenizer.no_speech > 0
        assert tokenizer.timestamp_begin > 0

    def test_encode_decode(self):
        """Test encoding and decoding text."""
        from smlx.models.Whisper_tiny.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(multilingual=True, language="en")
        text = "Hello, world!"

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert decoded.lower().replace(",", "").strip() == text.lower().replace(",", "").strip()

    def test_language_codes(self):
        """Test language code support."""
        from smlx.models.Whisper_tiny.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

        assert len(LANGUAGES) == 99
        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "english"

        assert "english" in TO_LANGUAGE_CODE
        assert TO_LANGUAGE_CODE["english"] == "en"
        assert TO_LANGUAGE_CODE["mandarin"] == "zh"  # Alias

    def test_sot_sequence_construction(self):
        """Test SOT sequence construction."""
        from smlx.models.Whisper_tiny.tokenizer import get_tokenizer

        # English transcription
        tokenizer_en = get_tokenizer(multilingual=True, language="en", task="transcribe")
        assert len(tokenizer_en.sot_sequence) == 3  # SOT + lang + task

        # Spanish translation
        tokenizer_es = get_tokenizer(multilingual=True, language="es", task="translate")
        assert len(tokenizer_es.sot_sequence) == 3
        assert tokenizer_es.sot_sequence[-1] == tokenizer_es.translate


class TestWhisperAudio:
    """Test audio processing utilities."""

    def test_audio_constants(self):
        """Test audio processing constants."""
        from smlx.models.Whisper_tiny.audio import (
            CHUNK_LENGTH,
            HOP_LENGTH,
            N_FFT,
            N_FRAMES,
            N_SAMPLES,
            SAMPLE_RATE,
        )

        assert SAMPLE_RATE == 16000
        assert N_FFT == 400
        assert HOP_LENGTH == 160
        assert CHUNK_LENGTH == 30
        assert N_SAMPLES == 480000  # 30s * 16kHz
        assert N_FRAMES == 3000  # 480000 / 160

    def test_pad_or_trim(self, sample_audio):
        """Test audio padding and trimming."""
        from smlx.models.Whisper_tiny.audio import N_SAMPLES, pad_or_trim

        # Test padding
        padded = pad_or_trim(sample_audio, N_SAMPLES)
        assert padded.shape[0] == N_SAMPLES

        # Test trimming
        long_audio = mx.random.normal((N_SAMPLES * 2,))
        trimmed = pad_or_trim(long_audio, N_SAMPLES)
        assert trimmed.shape[0] == N_SAMPLES

    def test_log_mel_spectrogram(self, sample_audio):
        """Test mel spectrogram computation."""
        from smlx.models.Whisper_tiny.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim

        # Pad to 30 seconds
        audio = pad_or_trim(sample_audio, N_SAMPLES)

        # Compute mel spectrogram
        mel = log_mel_spectrogram(audio, n_mels=80)

        # MLX Conv1d expects NLC format: (n_frames, n_mels)
        assert mel.shape[0] == 3000  # n_frames (30 seconds at 16kHz with hop_length=160)
        assert mel.shape[1] == 80  # n_mels
        assert mel.dtype == mx.float32

    def test_split_audio_chunks(self, sample_audio):
        """Test audio chunking."""
        from smlx.models.Whisper_tiny.audio import N_SAMPLES, split_audio_chunks

        # Create 2-minute audio
        long_audio = mx.random.normal((16000 * 120,))

        chunks = split_audio_chunks(long_audio, chunk_length=N_SAMPLES)

        assert len(chunks) == 4  # Four 30-second chunks
        for chunk in chunks:
            assert chunk.shape[0] == N_SAMPLES


@pytest.mark.requires_model
@pytest.mark.slow
class TestWhisperModel:
    """Test Whisper model architecture and inference.

    These tests require model weights and are marked as slow.
    """

    def test_model_creation(self):
        """Test creating Whisper model from config."""
        from smlx.models.Whisper_tiny.model import ModelConfig, Whisper

        config = ModelConfig()
        model = Whisper(config)

        assert model.config == config
        assert model.is_multilingual
        assert model.num_languages == 99

    def test_encoder_forward(self, sample_mel):
        """Test encoder forward pass."""
        from smlx.models.Whisper_tiny.model import ModelConfig, Whisper

        config = ModelConfig()
        model = Whisper(config)

        # Add batch dimension - MLX Conv1d expects (batch, length, channels)
        mel = sample_mel[None, ...]  # (1, 3000, 80) = (batch, n_frames, n_mels)

        # Encoder forward pass
        audio_features = model.encode_audio(mel)

        # Check output shape: (batch, n_ctx//2, n_state)
        # Encoder has stride 2, so 3000 -> 1500
        assert audio_features.shape == (1, 1500, 384)

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        from smlx.models.Whisper_tiny.model import ModelConfig, Whisper

        config = ModelConfig()
        model = Whisper(config)

        # Create dummy inputs
        audio_features = mx.random.normal((1, 1500, 384))
        tokens = mx.array([[50258, 50259, 50359]])  # SOT + lang + task

        # Decoder forward pass
        logits, kv_cache = model.decode_text(tokens, audio_features)

        # Check output shape: (batch, seq_len, vocab_size)
        assert logits.shape == (1, 3, 51865)
        assert kv_cache is not None


@pytest.mark.requires_model
@pytest.mark.slow
class TestWhisperDecoding:
    """Test Whisper decoding logic."""

    def test_decoding_options_defaults(self):
        """Test DecodingOptions defaults."""
        from smlx.models.Whisper_tiny.decoding import DecodingOptions

        options = DecodingOptions()

        assert options.task == "transcribe"
        assert options.language is None
        assert options.temperature == 0.0
        assert options.suppress_blank is True
        assert options.without_timestamps is False

    def test_decoding_result(self):
        """Test DecodingResult dataclass."""
        from smlx.models.Whisper_tiny.decoding import DecodingResult

        result = DecodingResult(
            audio_features=mx.zeros((1, 1500, 384)),
            language="en",
            tokens=[1, 2, 3],
            text="hello world",
            avg_logprob=-0.5,
            no_speech_prob=0.1,
            temperature=0.0,
            compression_ratio=2.5,
        )

        assert result.language == "en"
        assert result.text == "hello world"
        assert len(result.tokens) == 3

    def test_compression_ratio(self):
        """Test compression ratio computation."""
        from smlx.models.Whisper_tiny.decoding import compression_ratio

        # Repetitive text has high compression ratio
        repetitive = "a" * 100
        ratio_repetitive = compression_ratio(repetitive)
        assert ratio_repetitive > 5.0

        # Normal text has lower compression ratio
        normal = "The quick brown fox jumps over the lazy dog."
        ratio_normal = compression_ratio(normal)
        assert ratio_normal < 5.0


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestWhisperIntegration:
    """Integration tests for full Whisper pipeline.

    These tests require downloading model weights from HuggingFace.
    """

    def test_load_model(self):
        """Test loading model from stub (without actual download)."""
        from smlx.models.Whisper_tiny.model import ModelConfig, Whisper

        # Just test we can create a model
        config = ModelConfig()
        model = Whisper(config)

        assert isinstance(model, Whisper)

    def test_tokenizer_integration(self):
        """Test tokenizer with model vocab size."""
        from smlx.models.Whisper_tiny.model import ModelConfig
        from smlx.models.Whisper_tiny.tokenizer import get_tokenizer

        config = ModelConfig()
        tokenizer = get_tokenizer(multilingual=True, language="en")

        # Check tokenizer vocab matches model vocab
        assert tokenizer.encoding.n_vocab <= config.n_vocab


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
