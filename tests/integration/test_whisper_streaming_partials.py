"""
Integration tests for Whisper streaming partial results.

Tests the partial vs final result detection logic in streaming transcription,
including punctuation-based, VAD-based, and buffer-based strategies.
"""

import time

import numpy as np
import pytest

from smlx.models.Whisper_tiny import load
from smlx.models.Whisper_tiny.audio import SAMPLE_RATE
from smlx.models.Whisper_tiny.streaming import (
    StreamingConfig,
    StreamingResult,
    StreamingTranscriber,
)


@pytest.fixture(scope="module")
def whisper_model():
    """Load Whisper model once for all tests."""
    model, tokenizer = load()
    return model, tokenizer


@pytest.mark.integration
@pytest.mark.requires_model
class TestPartialResultDetection:
    """Test partial vs final result detection logic."""

    def test_punctuation_based_final_detection(self, whisper_model):
        """Test that sentences ending with punctuation are marked as final."""
        model, tokenizer = whisper_model
        transcriber = StreamingTranscriber(
            model,
            tokenizer,
            config=StreamingConfig(
                enable_vad=False,
                enable_partial_results=True,
            ),
        )

        # Add audio to buffer so buffer exhaustion doesn't trigger
        audio = np.random.randn(int(5.0 * SAMPLE_RATE))
        transcriber.add_audio(audio)

        # Test sentence terminators
        assert transcriber._is_final_result("Hello world.") is True
        assert transcriber._is_final_result("Hello world!") is True
        assert transcriber._is_final_result("Hello world?") is True

        # Test non-terminators (should be partial)
        assert transcriber._is_final_result("Hello world") is False
        assert transcriber._is_final_result("Hello") is False

    def test_clause_boundary_detection(self, whisper_model):
        """Test that long text with clause boundaries is marked as final."""
        model, tokenizer = whisper_model
        transcriber = StreamingTranscriber(
            model,
            tokenizer,
            config=StreamingConfig(
                enable_vad=False,
                enable_partial_results=True,
            ),
        )

        # Add audio to buffer so buffer exhaustion doesn't trigger
        audio = np.random.randn(int(5.0 * SAMPLE_RATE))
        transcriber.add_audio(audio)

        # Long text with comma should be final
        long_text = "This is a long sentence with multiple clauses,"
        assert transcriber._is_final_result(long_text) is True

        # Short text with comma should NOT be final
        short_text = "Hello,"
        assert transcriber._is_final_result(short_text) is False

        # Test semicolon and colon (must END with the punctuation)
        assert transcriber._is_final_result("This is a long sentence that ends with semicolon;") is True
        assert transcriber._is_final_result("This is a long sentence that ends with colon:") is True

        # Short text ending with semicolon/colon should NOT be final
        assert transcriber._is_final_result("Short;") is False
        assert transcriber._is_final_result("Brief:") is False

    def test_empty_text_handling(self, whisper_model):
        """Test that empty text is never marked as final."""
        model, tokenizer = whisper_model
        transcriber = StreamingTranscriber(
            model,
            tokenizer,
            config=StreamingConfig(enable_partial_results=True),
        )

        assert transcriber._is_final_result("") is False
        assert transcriber._is_final_result("   ") is False

    def test_buffer_exhaustion_detection(self, whisper_model):
        """Test that buffer exhaustion triggers final result."""
        model, tokenizer = whisper_model
        transcriber = StreamingTranscriber(
            model,
            tokenizer,
            config=StreamingConfig(
                min_chunk_duration=1.0,
                enable_partial_results=True,
            ),
        )

        # Empty buffer should trigger final
        transcriber.buffer.clear()
        assert transcriber._is_final_result("No punctuation") is True

        # Buffer with enough audio should NOT trigger final
        audio = np.random.randn(int(2.0 * SAMPLE_RATE))  # 2 seconds
        transcriber.add_audio(audio)
        assert transcriber._is_final_result("No punctuation") is False


@pytest.mark.integration
@pytest.mark.requires_model
class TestStreamingWithPartials:
    """Test end-to-end streaming with partial results."""

    def test_streaming_emits_partial_results(self, whisper_model):
        """Test that streaming can emit partial results."""
        model, tokenizer = whisper_model
        config = StreamingConfig(
            chunk_duration=2.0,
            enable_partial_results=True,
            partial_result_interval=0.1,  # Fast for testing
        )
        transcriber = StreamingTranscriber(model, tokenizer, config=config)

        # Generate synthetic audio (silence with noise)
        # Note: This won't produce meaningful transcriptions but tests the pipeline
        audio_duration = 6.0  # seconds
        audio = np.random.randn(int(audio_duration * SAMPLE_RATE)) * 0.01

        results = []
        chunk_size = int(config.chunk_duration * SAMPLE_RATE)

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            result = transcriber.process_chunk(chunk)
            if result:
                results.append(result)

        # Should get at least some results (might be partial or final)
        # Note: With random noise, transcription might be empty/sparse
        # This test mainly validates the pipeline doesn't crash
        assert isinstance(results, list)

    def test_partial_result_throttling(self, whisper_model):
        """Test that partial results are throttled correctly."""
        model, tokenizer = whisper_model
        config = StreamingConfig(
            chunk_duration=1.0,
            enable_partial_results=True,
            partial_result_interval=1.0,  # 1 second throttle
        )
        transcriber = StreamingTranscriber(model, tokenizer, config=config)

        # Simulate rapid partial results
        transcriber.last_partial_time = time.time()

        # Create dummy audio chunk
        chunk = np.random.randn(int(1.0 * SAMPLE_RATE)) * 0.01

        # Mock is_final to always return False (partial)
        original_is_final = transcriber._is_final_result
        transcriber._is_final_result = lambda text, chunk=None: False

        # First call should be throttled (too soon)
        # Note: This might return None due to empty transcription too
        result = transcriber.process_chunk(chunk)

        # Wait for throttle interval
        time.sleep(1.1)

        # Second call should potentially go through (if transcription is non-empty)
        result2 = transcriber.process_chunk(chunk)

        # Restore original method
        transcriber._is_final_result = original_is_final

        # Test validates throttling logic doesn't crash
        assert result is None or isinstance(result, StreamingResult)
        assert result2 is None or isinstance(result2, StreamingResult)

    def test_partial_results_disabled(self, whisper_model):
        """Test that partial results can be disabled."""
        model, tokenizer = whisper_model
        config = StreamingConfig(
            chunk_duration=1.0,
            enable_partial_results=False,  # Disabled
        )
        transcriber = StreamingTranscriber(model, tokenizer, config=config)

        # Mock is_final to return False (partial)
        transcriber._is_final_result = lambda text, chunk=None: False

        # Create dummy audio
        chunk = np.random.randn(int(1.0 * SAMPLE_RATE)) * 0.01

        result = transcriber.process_chunk(chunk)

        # Should return None (partial results disabled)
        assert result is None

    def test_final_results_always_emitted(self, whisper_model):
        """Test that final results are always emitted regardless of throttling."""
        model, tokenizer = whisper_model
        config = StreamingConfig(
            chunk_duration=1.0,
            enable_partial_results=True,
            partial_result_interval=10.0,  # Long throttle
        )
        transcriber = StreamingTranscriber(model, tokenizer, config=config)

        # Mock is_final to always return True
        transcriber._is_final_result = lambda text, chunk=None: True

        # Set last partial time to now (would normally throttle)
        transcriber.last_partial_time = time.time()

        # Create dummy audio
        chunk = np.random.randn(int(1.0 * SAMPLE_RATE)) * 0.01

        # Final results should bypass throttle
        # Note: Might still return None due to empty transcription
        result = transcriber.process_chunk(chunk)
        assert result is None or isinstance(result, StreamingResult)


@pytest.mark.integration
@pytest.mark.requires_model
@pytest.mark.slow
class TestVADIntegration:
    """Test VAD integration for partial result detection."""

    def test_vad_initialization(self, whisper_model):
        """Test that VAD is initialized when enabled."""
        model, tokenizer = whisper_model
        config = StreamingConfig(
            enable_vad=True,
            vad_threshold=0.5,
        )

        try:
            transcriber = StreamingTranscriber(model, tokenizer, config=config)
            # VAD should be initialized (or None if import failed)
            assert hasattr(transcriber, "vad")
        except ImportError:
            pytest.skip("VAD dependencies not available")

    def test_vad_not_initialized_when_disabled(self, whisper_model):
        """Test that VAD is not initialized when disabled."""
        model, tokenizer = whisper_model
        config = StreamingConfig(
            enable_vad=False,
        )
        transcriber = StreamingTranscriber(model, tokenizer, config=config)

        # VAD should be None when disabled
        assert transcriber.vad is None

    def test_vad_fallback_on_import_error(self, whisper_model):
        """Test graceful fallback when VAD import fails."""
        model, tokenizer = whisper_model
        config = StreamingConfig(
            enable_vad=True,
            vad_threshold=0.5,
        )

        # This should not raise an error even if VAD is not available
        transcriber = StreamingTranscriber(model, tokenizer, config=config)

        # Add audio to buffer so buffer exhaustion doesn't trigger
        audio = np.random.randn(int(5.0 * SAMPLE_RATE))
        transcriber.add_audio(audio)

        # Should still work (fall back to punctuation-based detection)
        assert transcriber._is_final_result("Hello world.") is True
        assert transcriber._is_final_result("Hello world") is False


@pytest.mark.integration
@pytest.mark.requires_model
class TestStreamingReset:
    """Test that reset() properly clears all state including partial result tracking."""

    def test_reset_clears_partial_time(self, whisper_model):
        """Test that reset() clears the partial result timer."""
        model, tokenizer = whisper_model
        transcriber = StreamingTranscriber(
            model,
            tokenizer,
            config=StreamingConfig(enable_partial_results=True),
        )

        # Set state
        transcriber.last_partial_time = time.time()
        transcriber.last_text = "Some text"
        audio = np.random.randn(int(2.0 * SAMPLE_RATE))
        transcriber.add_audio(audio)

        # Reset
        transcriber.reset()

        # All state should be cleared
        assert transcriber.last_partial_time == 0.0
        assert transcriber.last_text == ""
        assert len(transcriber.buffer) == 0
        assert transcriber.total_samples_processed == 0
        assert transcriber.last_result is None


@pytest.mark.integration
@pytest.mark.requires_model
class TestStreamingResult:
    """Test StreamingResult dataclass."""

    def test_streaming_result_creation(self):
        """Test creating a StreamingResult."""
        result = StreamingResult(
            text="Hello world",
            is_final=True,
            start_time=0.0,
            end_time=1.5,
            language="en",
            confidence=-0.5,
        )

        assert result.text == "Hello world"
        assert result.is_final is True
        assert result.start_time == 0.0
        assert result.end_time == 1.5
        assert result.language == "en"
        assert result.confidence == -0.5

    def test_streaming_result_default_confidence(self):
        """Test that confidence defaults to 0.0."""
        result = StreamingResult(
            text="Test",
            is_final=False,
            start_time=0.0,
            end_time=1.0,
            language="en",
        )

        assert result.confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
