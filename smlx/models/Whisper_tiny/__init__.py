"""
Whisper-tiny model for automatic speech recognition.

A lightweight Whisper model (39M parameters) for speech-to-text transcription
supporting 99 languages and English translation.

Example:
    >>> from smlx.models.Whisper_tiny import load, transcribe
    >>>
    >>> # Load model
    >>> model, tokenizer = load("mlx-community/whisper-tiny")
    >>>
    >>> # Transcribe audio
    >>> result = transcribe("speech.wav", model, tokenizer)
    >>> print(result["text"])
    >>>
    >>> # Transcribe with language specification
    >>> result = transcribe(
    ...     "speech.wav",
    ...     model,
    ...     tokenizer,
    ...     language="es",
    ...     task="translate",  # Translate to English
    ...     verbose=True
    ... )
"""

from .audio import (
    CHUNK_LENGTH,
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_SAMPLES,
    N_SAMPLES_PER_TOKEN,
    SAMPLE_RATE,
    TOKENS_PER_SECOND,
    get_audio_duration,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    prepare_audio,
    split_audio_chunks,
)
from .decoding import (
    BeamSearchDecoder,
    DecodingOptions,
    DecodingResult,
    GreedyDecoder,
    compression_ratio,
    decode,
    detect_language,
)
from .loader import load, load_model, load_tokenizer, save_model
from .model import AudioEncoder, ModelConfig, TextDecoder, Whisper
from .timing import add_word_timestamps, find_alignment, merge_punctuations
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, WhisperTokenizer, get_tokenizer
from .transcribe import transcribe, transcribe_file
from .vad import (
    SileroVAD,
    SpeechSegment,
    detect_speech_segments,
    filter_segments,
    merge_segments,
    transcribe_with_vad,
)

__all__ = [
    # Main API
    "load",
    "transcribe",
    "transcribe_file",
    # Model components
    "Whisper",
    "AudioEncoder",
    "TextDecoder",
    "ModelConfig",
    # Tokenizer
    "WhisperTokenizer",
    "get_tokenizer",
    "LANGUAGES",
    "TO_LANGUAGE_CODE",
    # Audio processing
    "load_audio",
    "log_mel_spectrogram",
    "prepare_audio",
    "pad_or_trim",
    "split_audio_chunks",
    "get_audio_duration",
    # Audio constants
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "CHUNK_LENGTH",
    "N_SAMPLES",
    "N_FRAMES",
    "N_SAMPLES_PER_TOKEN",
    "FRAMES_PER_SECOND",
    "TOKENS_PER_SECOND",
    # Decoding
    "decode",
    "detect_language",
    "DecodingOptions",
    "DecodingResult",
    "GreedyDecoder",
    "BeamSearchDecoder",
    "compression_ratio",
    # Word-level timestamps
    "add_word_timestamps",
    "find_alignment",
    "merge_punctuations",
    # Voice Activity Detection (VAD)
    "transcribe_with_vad",
    "detect_speech_segments",
    "SileroVAD",
    "SpeechSegment",
    "merge_segments",
    "filter_segments",
    # Model loading/saving
    "load_model",
    "load_tokenizer",
    "save_model",
]
