"""
End-to-end transcription pipeline for Whisper.

Provides high-level API for transcribing audio files with automatic
language detection, temperature fallback, and batched processing.

Based on OpenAI's Whisper transcription implementation.
"""

import sys
from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np

from .audio import (
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult, decode
from .model import Whisper
from .tokenizer import LANGUAGES, WhisperTokenizer, get_tokenizer


def _format_timestamp(seconds: float) -> str:
    """Format timestamp as HH:MM:SS.mmm.

    Args:
        seconds: Timestamp in seconds

    Returns:
        Formatted timestamp string

    Example:
        >>> _format_timestamp(90.5)
        '01:30.500'
        >>> _format_timestamp(3661.123)
        '01:01:01.123'
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def transcribe(
    audio: Union[str, np.ndarray, mx.array],
    model: Whisper,
    tokenizer: Optional[WhisperTokenizer] = None,
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    batch_size: int = 6,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'¿([{-",
    append_punctuations: str = ".!?,;:%)]}\"",
    **decode_options,
) -> dict[str, Any]:
    """Transcribe audio file using Whisper.

    Main entry point for end-to-end transcription. Handles long audio files
    by processing them in 30-second chunks with optional batching.

    Args:
        audio: Audio file path, numpy array, or MLX array
        model: Whisper model
        tokenizer: Tokenizer (created if not provided)
        verbose: Whether to print progress (True=detailed, False=minimal, None=silent)
        temperature: Temperature(s) for sampling. Can be tuple for fallback.
        compression_ratio_threshold: If compression ratio exceeds this, try higher temperature
        logprob_threshold: If average log probability below this, try higher temperature
        no_speech_threshold: If no-speech probability above this, treat as silent
        condition_on_previous_text: Use previous text as prompt for next window
        initial_prompt: Initial prompt text for first window
        batch_size: Number of 30s segments to process in parallel
        word_timestamps: Add word-level timestamps using DTW alignment (default: False)
        prepend_punctuations: Punctuation marks to merge with following word
        append_punctuations: Punctuation marks to merge with preceding word
        **decode_options: Additional options passed to DecodingOptions

    Returns:
        Dictionary with:
        - text: Full transcription
        - segments: List of segment dicts with timestamps and text
        - language: Detected or specified language code
        - (if word_timestamps=True) each segment has 'words' field with word-level timestamps

    Example:
        >>> from smlx.models.Whisper_tiny import load, transcribe
        >>> model, tokenizer = load()
        >>>
        >>> # Basic transcription
        >>> result = transcribe("speech.wav", model, tokenizer)
        >>> print(result["text"])
        >>>
        >>> # With language specification and custom parameters
        >>> result = transcribe(
        ...     "speech.wav",
        ...     model,
        ...     tokenizer,
        ...     language="es",
        ...     task="translate",
        ...     temperature=0.5,
        ...     verbose=True
        ... )
        >>> for seg in result["segments"]:
        ...     print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")

    Notes:
        - Audio longer than 30s is automatically split into chunks
        - Temperature fallback tries increasing temperatures if quality is low
        - Batching processes multiple chunks in parallel for efficiency
        - Silent segments (high no_speech_prob) are automatically skipped
    """
    # Ensure temperature is a tuple
    if not isinstance(temperature, (list, tuple)):
        temperature = (temperature,)

    # Get dtype from decode options
    dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32

    # Create tokenizer if not provided
    if tokenizer is None:
        language = decode_options.get("language", None)
        task = decode_options.get("task", "transcribe")
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=task,
        )

    # Compute mel spectrogram with padding for slicing
    mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-2] - N_FRAMES

    # Set up safe printing for non-UTF-8 systems
    if verbose:
        system_encoding = sys.getdefaultencoding()
        if system_encoding != "utf-8":

            def make_safe(x):
                return x.encode(system_encoding, errors="replace").decode(system_encoding)

        else:

            def make_safe(x):
                return x

    # Detect language if not specified
    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. "
                    "Use the `language` option to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)
            from .decoding import detect_language

            _, probs = detect_language(model, mel_segment, tokenizer)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")

    # Update tokenizer if language was detected
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    # Initialize tracking variables
    input_stride = N_FRAMES // model.config.n_audio_ctx  # mel frames per token: 2
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE  # time per token: 0.02s
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    # Add initial prompt if provided
    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def decode_with_fallback(
        segment_batch: mx.array, time_offsets: list[float]
    ) -> list[DecodingResult]:
        """Decode batch with temperature fallback.

        Tries temperatures in sequence if quality metrics fail thresholds.

        Args:
            segment_batch: Batch of mel spectrograms
            time_offsets: Time offsets for each segment

        Returns:
            List of decoding results
        """
        # Try each temperature
        for temp_idx, temp in enumerate(temperature):
            kwargs = {**decode_options}
            options = DecodingOptions(**kwargs, temperature=temp)
            results = []

            # Process batch
            segment_batch_list = [segment_batch[i : i + 1] for i in range(segment_batch.shape[0])]
            for segment in segment_batch_list:
                result = decode(model, segment[0], options)
                results.append(result)

            # Check quality for each result
            needs_fallback = [False] * len(results)
            for i, result in enumerate(results):
                # Check compression ratio
                if (
                    compression_ratio_threshold is not None
                    and result.compression_ratio > compression_ratio_threshold
                ):
                    needs_fallback[i] = True

                # Check average log probability
                if (
                    logprob_threshold is not None
                    and result.avg_logprob < logprob_threshold
                ):
                    needs_fallback[i] = True

                # Override if no speech detected
                if (
                    no_speech_threshold is not None
                    and result.no_speech_prob > no_speech_threshold
                ):
                    needs_fallback[i] = False

            # If this is not the last temperature and any need fallback, try next temp
            if temp_idx < len(temperature) - 1 and any(needs_fallback):
                continue

            return results

        return results

    def new_segment(
        start: float, end: float, tokens: list[int], result: DecodingResult
    ) -> dict:
        """Create segment dictionary.

        Args:
            start: Start time in seconds
            end: End time in seconds
            tokens: Token IDs
            result: Decoding result

        Returns:
            Segment dictionary
        """
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # Process audio in batches
    seek = 0
    while seek < content_frames:
        time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)

        # Collect batch of segments
        mel_segments = []
        time_offsets = []
        segment_sizes = []

        for _ in range(batch_size):
            if seek >= content_frames:
                break

            segment_size = min(N_FRAMES, content_frames - seek)
            mel_segment = mel[seek : seek + segment_size]
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE

            # Pad to 30 seconds
            mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(dtype)
            mel_segments.append(mel_segment)
            time_offsets.append(seek * HOP_LENGTH / SAMPLE_RATE)
            segment_sizes.append(segment_size)

            seek += N_FRAMES

        if not mel_segments:
            break

        # Stack into batch
        segment_batch = mx.stack(mel_segments, axis=0)

        # Set prompt for context
        decode_options["prompt"] = all_tokens[prompt_reset_since:]

        # Decode batch with fallback
        results = decode_with_fallback(segment_batch, time_offsets)

        # Process results
        for i, result in enumerate(results):
            time_offset = time_offsets[i]
            segment_size = segment_sizes[i]
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE

            # Skip if no speech
            if (
                no_speech_threshold is not None
                and result.no_speech_prob > no_speech_threshold
            ):
                if verbose:
                    print(f"[{_format_timestamp(time_offset)}] No speech detected")
                continue

            # Extract tokens and create segment
            tokens = result.tokens

            # Parse timestamps from tokens
            timestamp_tokens = mx.array([t >= tokenizer.timestamp_begin for t in tokens])
            consecutive = np.where(
                np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
            )[0]

            if len(consecutive) > 0:
                # Multiple timestamp segments within 30s chunk
                consecutive = consecutive + 1
                consecutive = consecutive.tolist()

                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    if len(sliced_tokens) == 0:
                        continue

                    start_timestamp_pos = sliced_tokens[0] - tokenizer.timestamp_begin
                    end_timestamp_pos = sliced_tokens[-1] - tokenizer.timestamp_begin

                    segment = new_segment(
                        start=time_offset + start_timestamp_pos * time_precision,
                        end=time_offset + end_timestamp_pos * time_precision,
                        tokens=sliced_tokens,
                        result=result,
                    )
                    all_segments.append(segment)
                    last_slice = current_slice
            else:
                # Single segment for entire chunk
                duration = segment_duration
                timestamps = [t for t in tokens if t >= tokenizer.timestamp_begin]
                if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                    last_timestamp_pos = timestamps[-1] - tokenizer.timestamp_begin
                    duration = last_timestamp_pos * time_precision

                segment = new_segment(
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                    result=result,
                )
                all_segments.append(segment)

            # Add tokens to history
            all_tokens.extend(tokens)

            # Reset prompt if temperature is high (less reliable context)
            if not condition_on_previous_text or result.temperature > 0.5:
                prompt_reset_since = len(all_tokens)

            # Print segment if verbose
            if verbose:
                segment_text = tokenizer.decode(tokens)
                if verbose is True:
                    # Detailed output
                    print(
                        f"[{_format_timestamp(time_offset)} --> "
                        f"{_format_timestamp(time_offset + duration)}] "
                        f"{make_safe(segment_text)}"
                    )
                else:
                    # Minimal output
                    print(make_safe(segment_text), end=" ", flush=True)

    if verbose is False:
        print()  # Newline after minimal output

    # Clean up empty segments
    all_segments = [
        seg for seg in all_segments if seg["text"].strip() != ""
    ]

    # Add word-level timestamps if requested
    if word_timestamps and all_segments:
        from .timing import add_word_timestamps

        # Calculate number of frames after encoder stride
        num_frames = mel.shape[-2] // input_stride

        all_segments = add_word_timestamps(
            segments=all_segments,
            model=model,
            tokenizer=tokenizer,
            mel=mel,
            num_frames=num_frames,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
        )

    # Decode full text
    full_text = tokenizer.decode(all_tokens[len(initial_prompt_tokens) :])

    return {
        "text": full_text,
        "segments": all_segments,
        "language": language,
    }


def transcribe_file(
    audio_path: str,
    model: Whisper,
    tokenizer: Optional[WhisperTokenizer] = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function for transcribing audio file.

    Args:
        audio_path: Path to audio file
        model: Whisper model
        tokenizer: Tokenizer (created if not provided)
        **kwargs: Additional arguments for transcribe()

    Returns:
        Transcription result dictionary

    Example:
        >>> from smlx.models.Whisper_tiny import load, transcribe_file
        >>> model, tokenizer = load()
        >>> result = transcribe_file("speech.wav", model, tokenizer, verbose=True)
        >>> print(result["text"])
    """
    return transcribe(audio_path, model, tokenizer, **kwargs)
