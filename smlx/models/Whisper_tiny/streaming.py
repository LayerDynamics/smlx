"""
Streaming inference for Whisper - real-time transcription.

Provides utilities for transcribing audio streams in real-time using a sliding
window approach with overlap to ensure continuous transcription.

This is useful for:
- Live microphone input
- Real-time video captioning
- Live translation
- Voice assistants

Usage:
    from smlx.models.Whisper_tiny import load
    from smlx.models.Whisper_tiny.streaming import StreamingTranscriber

    model, tokenizer = load()
    transcriber = StreamingTranscriber(model, tokenizer)

    # Process audio chunks as they arrive
    for audio_chunk in audio_stream:
        result = transcriber.process_chunk(audio_chunk)
        if result:
            print(result["text"])

Example with microphone:
    import sounddevice as sd

    model, tokenizer = load()
    transcriber = StreamingTranscriber(model, tokenizer, chunk_duration=5.0)

    def audio_callback(indata, frames, time, status):
        audio = indata[:, 0]  # Mono
        result = transcriber.process_chunk(audio)
        if result:
            print(f"Transcribed: {result['text']}")

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        input("Press Enter to stop...")
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, load_audio
from .model import Whisper
from .tokenizer import WhisperTokenizer
from .transcribe import transcribe


@dataclass
class StreamingConfig:
    """Configuration for streaming transcription.

    Attributes:
        chunk_duration: Duration of each processing chunk (seconds)
        overlap_duration: Overlap between consecutive chunks (seconds)
        min_chunk_duration: Minimum chunk duration to process (seconds)
        buffer_duration: Maximum audio buffer duration (seconds)
        vad_threshold: VAD threshold for silence detection (0-1, None = disabled)
        language: Language code (None = auto-detect)
        task: "transcribe" or "translate"
        temperature: Sampling temperature
        beam_size: Beam search size (None = greedy)
        enable_vad: Enable Voice Activity Detection for better partial/final detection
        enable_partial_results: Enable emission of partial (non-final) results
        partial_result_interval: Minimum interval between partial results (seconds)
    """

    chunk_duration: float = 5.0
    overlap_duration: float = 1.0
    min_chunk_duration: float = 1.0
    buffer_duration: float = 30.0
    vad_threshold: Optional[float] = None
    language: Optional[str] = None
    task: str = "transcribe"
    temperature: float = 0.0
    beam_size: Optional[int] = None
    enable_vad: bool = False
    enable_partial_results: bool = True
    partial_result_interval: float = 0.5


@dataclass
class StreamingResult:
    """Result from streaming transcription.

    Attributes:
        text: Transcribed text
        is_final: Whether this is a final result (not partial)
        start_time: Start time relative to stream start (seconds)
        end_time: End time relative to stream start (seconds)
        language: Detected or specified language
        confidence: Average log probability (quality metric)
    """

    text: str
    is_final: bool
    start_time: float
    end_time: float
    language: str
    confidence: float = 0.0


class StreamingTranscriber:
    """Real-time streaming transcriber for Whisper.

    Processes audio in chunks with overlapping windows to provide continuous
    transcription of audio streams.

    Attributes:
        model: Whisper model
        tokenizer: Whisper tokenizer
        config: Streaming configuration
        buffer: Audio buffer (deque of samples)
        total_samples_processed: Total samples processed so far
        last_result: Last transcription result
    """

    def __init__(
        self,
        model: Whisper,
        tokenizer: WhisperTokenizer,
        config: Optional[StreamingConfig] = None,
    ):
        """Initialize streaming transcriber.

        Args:
            model: Whisper model
            tokenizer: Whisper tokenizer
            config: Streaming configuration (default config if None)

        Example:
            >>> from smlx.models.Whisper_tiny import load
            >>> from smlx.models.Whisper_tiny.streaming import StreamingTranscriber
            >>> model, tokenizer = load()
            >>> transcriber = StreamingTranscriber(model, tokenizer)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or StreamingConfig()

        # Audio buffer
        max_buffer_samples = int(self.config.buffer_duration * SAMPLE_RATE)
        self.buffer: deque[float] = deque(maxlen=max_buffer_samples)

        # State tracking
        self.total_samples_processed = 0
        self.last_result: Optional[StreamingResult] = None
        self.last_text = ""
        self.last_partial_time = 0.0  # For throttling partial results

        # Calculate chunk sizes in samples
        self.chunk_samples = int(self.config.chunk_duration * SAMPLE_RATE)
        self.overlap_samples = int(self.config.overlap_duration * SAMPLE_RATE)
        self.min_chunk_samples = int(self.config.min_chunk_duration * SAMPLE_RATE)

        # Initialize VAD if enabled
        self.vad = None
        if self.config.enable_vad:
            try:
                from .vad import SileroVAD

                vad_threshold = self.config.vad_threshold if self.config.vad_threshold is not None else 0.5
                self.vad = SileroVAD(threshold=vad_threshold)
            except ImportError:
                # VAD not available, fall back to punctuation-based detection
                pass

    def reset(self):
        """Reset transcriber state.

        Clears buffer and resets all counters.
        """
        self.buffer.clear()
        self.total_samples_processed = 0
        self.last_result = None
        self.last_text = ""
        self.last_partial_time = 0.0

    def add_audio(self, audio: Union[np.ndarray, mx.array]):
        """Add audio samples to buffer.

        Args:
            audio: Audio samples (mono, 16kHz)
        """
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Flatten if needed
        if audio.ndim > 1:
            audio = audio.flatten()

        # Add to buffer
        self.buffer.extend(audio)

    def has_enough_audio(self) -> bool:
        """Check if buffer has enough audio to process.

        Returns:
            True if buffer has at least min_chunk_duration worth of audio
        """
        return len(self.buffer) >= self.min_chunk_samples

    def _extract_chunk(self) -> Optional[np.ndarray]:
        """Extract next audio chunk from buffer.

        Returns:
            Audio chunk as numpy array, or None if not enough audio
        """
        if not self.has_enough_audio():
            return None

        # Extract chunk
        chunk_size = min(self.chunk_samples, len(self.buffer))
        chunk = np.array(list(self.buffer)[:chunk_size])

        # Remove processed audio (keeping overlap)
        advance_samples = max(chunk_size - self.overlap_samples, 0)
        for _ in range(advance_samples):
            if self.buffer:
                self.buffer.popleft()

        return chunk

    def _is_final_result(self, text: str, audio_chunk: Optional[np.ndarray] = None) -> bool:
        """Determine if transcription result is final or partial.

        A result is considered final if:
        1. Ends with sentence terminator (. ! ?)
        2. Ends with clause boundary and is reasonably long (, ; :)
        3. Buffer is exhausted (no more audio to process)
        4. VAD detects silence (if VAD is enabled)

        Args:
            text: Transcribed text to analyze
            audio_chunk: Audio chunk that was transcribed (for VAD analysis)

        Returns:
            True if this is a final result, False if partial

        Example:
            >>> is_final = transcriber._is_final_result("Hello world.")
            >>> assert is_final == True  # Ends with period
            >>> is_partial = transcriber._is_final_result("Hello world")
            >>> assert is_partial == False  # No terminator
        """
        text = text.rstrip()
        if not text:
            return False

        # Strategy 1: Punctuation-based detection
        # Final if ends with sentence terminators
        if text[-1] in ".!?":
            return True

        # Soft finals: clause boundaries for longer text
        # This allows streaming of long sentences in chunks
        if text[-1] in ",;:" and len(text) > 20:
            return True

        # Strategy 2: VAD-based detection (if enabled)
        if hasattr(self, "vad") and self.vad is not None and audio_chunk is not None:
            try:
                segments = self.vad.detect_segments(audio_chunk)
                # No speech detected = final result (silence)
                if not segments:
                    return True
                # Multiple segments suggest natural pause points
                if len(segments) > 1:
                    return True
            except Exception:
                # Fall through to other strategies if VAD fails
                pass

        # Strategy 3: Buffer exhaustion check
        # If we don't have enough audio for another chunk, mark as final
        if not self.has_enough_audio():
            return True

        # Otherwise, it's a partial result
        return False

    def process_chunk(
        self, audio: Optional[Union[np.ndarray, mx.array]] = None
    ) -> Optional[StreamingResult]:
        """Process audio chunk and return transcription.

        Args:
            audio: New audio to add (None = process buffered audio)

        Returns:
            StreamingResult if transcription available, None otherwise

        Example:
            >>> result = transcriber.process_chunk(audio_chunk)
            >>> if result:
            ...     print(result.text)
        """
        # Add new audio if provided
        if audio is not None:
            self.add_audio(audio)

        # Check if we have enough audio
        if not self.has_enough_audio():
            return None

        # Extract chunk
        chunk = self._extract_chunk()
        if chunk is None:
            return None

        # Transcribe chunk
        start_time = self.total_samples_processed / SAMPLE_RATE
        self.total_samples_processed += len(chunk) - self.overlap_samples

        try:
            result = transcribe(
                chunk,
                self.model,
                self.tokenizer,
                language=self.config.language,
                task=self.config.task,
                temperature=self.config.temperature,
                beam_size=self.config.beam_size,
                verbose=False,
            )

            text = result["text"].strip()
            end_time = start_time + len(chunk) / SAMPLE_RATE

            # Skip if text is empty or unchanged
            if not text or text == self.last_text:
                return None

            # Determine if this is a final or partial result
            is_final = self._is_final_result(text, chunk)

            # Throttle partial results if enabled
            if not is_final and self.config.enable_partial_results:
                current_time = time.time()
                time_since_last_partial = current_time - self.last_partial_time

                # Skip partial results that are too frequent
                if time_since_last_partial < self.config.partial_result_interval:
                    return None

                # Update throttle timer
                self.last_partial_time = current_time
            elif not self.config.enable_partial_results and not is_final:
                # Partial results disabled - skip non-final results
                return None

            # Create result
            streaming_result = StreamingResult(
                text=text,
                is_final=is_final,
                start_time=start_time,
                end_time=end_time,
                language=result["language"],
                confidence=result.get("segments", [{}])[0].get("avg_logprob", 0.0)
                if result.get("segments")
                else 0.0,
            )

            self.last_result = streaming_result
            self.last_text = text

            return streaming_result

        except Exception:
            # Return None on error (could be empty chunk, etc.)
            return None

    def process_file(
        self, audio_path: str, chunk_duration: Optional[float] = None
    ) -> list[StreamingResult]:
        """Process entire audio file in streaming fashion.

        Useful for testing streaming behavior on pre-recorded audio.

        Args:
            audio_path: Path to audio file
            chunk_duration: Override chunk duration (seconds)

        Returns:
            List of streaming results

        Example:
            >>> results = transcriber.process_file("audio.wav")
            >>> for result in results:
            ...     print(f"[{result.start_time:.2f}s]: {result.text}")
        """
        # Load audio
        audio = load_audio(audio_path)

        # Override chunk duration if specified
        if chunk_duration is not None:
            old_chunk_duration = self.config.chunk_duration
            self.config.chunk_duration = chunk_duration
            self.chunk_samples = int(chunk_duration * SAMPLE_RATE)

        # Reset state
        self.reset()

        # Process in chunks
        results = []
        chunk_size = int(self.config.chunk_duration * SAMPLE_RATE)

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            result = self.process_chunk(chunk)
            if result:
                results.append(result)

        # Process remaining buffered audio
        while self.has_enough_audio():
            result = self.process_chunk()
            if result:
                results.append(result)

        # Restore config if modified
        if chunk_duration is not None:
            self.config.chunk_duration = old_chunk_duration
            self.chunk_samples = int(old_chunk_duration * SAMPLE_RATE)

        return results


class MicrophoneStream:
    """Helper for streaming from microphone using sounddevice.

    Requires: pip install sounddevice

    Example:
        >>> from smlx.models.Whisper_tiny import load
        >>> from smlx.models.Whisper_tiny.streaming import MicrophoneStream, StreamingTranscriber
        >>>
        >>> model, tokenizer = load()
        >>> transcriber = StreamingTranscriber(model, tokenizer)
        >>>
        >>> with MicrophoneStream() as stream:
        ...     for audio_chunk in stream:
        ...         result = transcriber.process_chunk(audio_chunk)
        ...         if result:
        ...             print(result.text)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 1,
        chunk_duration: float = 1.0,
        device: Optional[int] = None,
    ):
        """Initialize microphone stream.

        Args:
            sample_rate: Audio sampling rate
            channels: Number of audio channels (1 = mono)
            chunk_duration: Duration of each chunk (seconds)
            device: Audio device index (None = default)

        Raises:
            ImportError: If sounddevice is not installed
        """
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "sounddevice is required for MicrophoneStream. Install with: pip install sounddevice"
            ) from err

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.device = device
        self.chunk_samples = int(chunk_duration * sample_rate)
        self._stream = None
        self._queue: deque[np.ndarray] = deque()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            print(f"Audio stream status: {status}")
        # Extract mono channel if needed
        audio = indata[:, 0] if self.channels == 1 and indata.ndim > 1 else indata
        self._queue.append(audio.copy())

    def __enter__(self):
        """Start microphone stream."""
        import sounddevice as sd  # type: ignore[import-not-found]

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop microphone stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def __iter__(self):
        """Iterate over audio chunks."""
        return self

    def __next__(self) -> np.ndarray:
        """Get next audio chunk."""
        while not self._queue:
            time.sleep(0.01)  # Wait for audio
        return self._queue.popleft()

    def read(self) -> Optional[np.ndarray]:
        """Read next audio chunk (non-blocking).

        Returns:
            Audio chunk if available, None otherwise
        """
        if self._queue:
            return self._queue.popleft()
        return None
