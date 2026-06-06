#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Speech synthesis functions for Chatterbox.

Provides voice cloning, expressiveness control, and emotion-based synthesis.
Includes audio output validation to detect silent or clipped audio.
"""

from typing import Optional

import mlx.core as mx
import numpy as np

from .config import AVAILABLE_EMOTIONS
from .model import Chatterbox
from .processor import ChatterboxProcessor
from ...utils.validation import validate_audio_output


def clone_voice(
    model: Chatterbox,
    processor: ChatterboxProcessor,
    reference_audio: np.ndarray,
    sample_rate: int = 24000,
) -> mx.array:
    """
    Clone voice from reference audio.

    Args:
        model: Chatterbox model
        processor: Chatterbox processor
        reference_audio: Reference audio (3-10 seconds recommended)
        sample_rate: Sample rate of reference audio

    Returns:
        Voice embedding for synthesis

    Example:
        >>> model, processor = load()
        >>> import soundfile as sf
        >>> ref_audio, sr = sf.read("my_voice.wav")
        >>> voice_emb = clone_voice(model, processor, ref_audio, sr)
        >>> audio = synthesize(model, processor, "Hello", voice_embedding=voice_emb)
    """
    # Process reference audio to mel-spectrogram
    mel = processor.process_audio(reference_audio, sr=sample_rate)

    # Add batch dimension
    mel = mx.expand_dims(mel, axis=0)

    # Encode to voice embedding
    voice_embedding = model.encode_voice(mel)

    print(f"✓ Voice cloned from {len(reference_audio)/sample_rate:.1f}s reference")

    return voice_embedding


def synthesize(
    model: Chatterbox,
    processor: ChatterboxProcessor,
    text: str,
    voice_embedding: Optional[mx.array] = None,
    emotion: str = "neutral",
    expressiveness: float = 0.5,
    sample_rate: int = 24000,
    validate_output: bool = False,
    check_clipping: bool = True,
    check_silence: bool = True,
    retry_on_failure: bool = False,
    max_retries: int = 2,
) -> np.ndarray:
    """
    Synthesize speech with voice cloning and expressiveness.

    Args:
        model: Chatterbox model
        processor: Chatterbox processor
        text: Input text
        voice_embedding: Optional voice embedding from clone_voice()
        emotion: Emotion ("neutral", "happy", "sad", "angry", etc.)
        expressiveness: Expressiveness scale [0, 1]
        sample_rate: Output sample rate
        validate_output: Enable audio validation (silence/clipping detection)
        check_clipping: Check for audio clipping (values > 1.0 or < -1.0)
        check_silence: Check for silent audio (all zeros or near-zero)
        retry_on_failure: Retry synthesis if validation fails
        max_retries: Maximum number of retries on validation failure

    Returns:
        Audio waveform as numpy array
        Shape: (samples,)
        Range: [-1.0, 1.0]

    Example:
        >>> model, processor = load()
        >>>
        >>> # Basic synthesis
        >>> audio = synthesize(model, processor, "Hello world")
        >>>
        >>> # With voice cloning
        >>> voice_emb = clone_voice(model, processor, reference_audio)
        >>> audio = synthesize(model, processor, "Hello", voice_embedding=voice_emb)
        >>>
        >>> # With emotion and expressiveness
        >>> audio = synthesize(
        ...     model, processor,
        ...     "I'm so excited!",
        ...     emotion="happy",
        ...     expressiveness=0.9
        ... )
        >>>
        >>> # With validation
        >>> audio = synthesize(
        ...     model, processor,
        ...     "Hello world",
        ...     validate_output=True,
        ...     retry_on_failure=True
        ... )
    """
    # Internal synthesis function for retry logic
    def _synthesize_internal(current_expressiveness: float) -> np.ndarray:
        # Process text
        input_ids = processor(text)

        # Add batch dimension
        input_ids = mx.expand_dims(input_ids, axis=0)

        # Get emotion ID
        emotion_to_use = emotion
        if emotion_to_use not in AVAILABLE_EMOTIONS:
            print(f"Warning: Unknown emotion '{emotion_to_use}', using 'neutral'")
            emotion_to_use = "neutral"

        emotion_id = mx.array([AVAILABLE_EMOTIONS.index(emotion_to_use)])

        # Clamp expressiveness
        exp_clamped = max(0.0, min(1.0, current_expressiveness))

        # Generate audio
        mel, waveform = model(
            input_ids=input_ids,
            voice_embedding=voice_embedding,
            emotion_id=emotion_id,
            expressiveness=exp_clamped,
        )

        # Remove batch dimension
        waveform = waveform[0]

        # Convert to numpy
        audio = np.array(waveform)

        # Normalize
        if audio.max() > 0:
            audio = audio / np.abs(audio).max()

        return audio

    # Retry loop with validation
    current_expressiveness = expressiveness
    for attempt in range(max(1, max_retries + 1 if retry_on_failure else 1)):
        audio = _synthesize_internal(current_expressiveness)

        # Validate output if enabled
        if validate_output:
            # Convert to MLX array for validation
            audio_mx = mx.array(audio)

            is_valid, reason = validate_audio_output(
                audio_mx,
                sample_rate=sample_rate,
                check_clipping=check_clipping,
                check_silence=check_silence,
            )

            if not is_valid and retry_on_failure and attempt < max_retries:
                print(f"Audio validation failed: {reason}. Retrying (attempt {attempt + 2}/{max_retries + 1})...")
                # Adjust expressiveness for retry
                current_expressiveness = min(1.0, current_expressiveness * 1.2)
                continue

            if not is_valid and not retry_on_failure:
                print(f"Warning: Audio validation failed: {reason}")

        # Success or max retries reached
        break

    duration_s = len(audio) / sample_rate
    if getattr(model, "weights_loaded", False):
        print(f"\n✓ Generated {duration_s:.2f}s of audio")
        print(f"Emotion: {emotion}, Expressiveness: {current_expressiveness:.1f}")
    else:
        # Honesty: without pre-trained model+vocoder weights this is noise.
        print(
            f"\n⚠ Produced {duration_s:.2f}s of audio from UNINITIALIZED (random) "
            "weights — this is placeholder noise, not real synthesis."
        )
        print(f"Emotion: {emotion}, Expressiveness: {current_expressiveness:.1f}")
        print("Load pre-trained weights from HuggingFace for actual synthesis")

    return audio


def synthesize_batch(
    model: Chatterbox,
    processor: ChatterboxProcessor,
    texts: list[str],
    voice_embedding: Optional[mx.array] = None,
    emotion: str = "neutral",
    expressiveness: float = 0.5,
) -> list[np.ndarray]:
    """
    Synthesize multiple texts with same voice and style.

    Args:
        model: Chatterbox model
        processor: Chatterbox processor
        texts: List of input texts
        voice_embedding: Optional voice embedding
        emotion: Emotion for all texts
        expressiveness: Expressiveness for all texts

    Returns:
        List of audio waveforms

    Example:
        >>> texts = ["Hello", "How are you?", "Goodbye"]
        >>> audios = synthesize_batch(model, processor, texts, emotion="happy")
    """
    audios = []

    for text in texts:
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            voice_embedding=voice_embedding,
            emotion=emotion,
            expressiveness=expressiveness,
        )
        audios.append(audio)

    return audios


def synthesize_with_emotions(
    model: Chatterbox,
    processor: ChatterboxProcessor,
    texts: list[str],
    emotions: list[str],
    expressiveness: float = 0.5,
) -> list[np.ndarray]:
    """
    Synthesize texts with different emotions.

    Args:
        model: Chatterbox model
        processor: Chatterbox processor
        texts: List of input texts
        emotions: List of emotions (one per text)
        expressiveness: Expressiveness for all texts

    Returns:
        List of audio waveforms

    Example:
        >>> texts = ["I'm happy!", "I'm sad.", "I'm angry!"]
        >>> emotions = ["happy", "sad", "angry"]
        >>> audios = synthesize_with_emotions(model, processor, texts, emotions)
    """
    if len(texts) != len(emotions):
        raise ValueError("Number of texts must match number of emotions")

    audios = []

    for text, emotion in zip(texts, emotions):
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            emotion=emotion,
            expressiveness=expressiveness,
        )
        audios.append(audio)

    return audios


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
):
    """
    Save audio to file.

    Args:
        audio: Audio waveform
        output_path: Output file path (.wav)
        sample_rate: Sample rate

    Example:
        >>> audio = synthesize(model, processor, "Hello")
        >>> save_audio(audio, "output.wav", sample_rate=24000)
    """
    try:
        import soundfile as sf

        sf.write(output_path, audio, sample_rate)
        print(f"✓ Audio saved to {output_path}")
    except ImportError:
        print("soundfile not installed. Install with: pip install soundfile")
        print("Falling back to scipy...")

        try:
            from scipy.io import wavfile

            # scipy expects int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio_int16)
            print(f"✓ Audio saved to {output_path}")
        except ImportError:
            print("Neither soundfile nor scipy available")
            print("Install soundfile: pip install soundfile")


def get_available_emotions() -> list[str]:
    """
    Get list of available emotions.

    Returns:
        List of emotion names

    Example:
        >>> emotions = get_available_emotions()
        >>> print(emotions)
        ['neutral', 'happy', 'sad', 'angry', ...]
    """
    return AVAILABLE_EMOTIONS.copy()


def demo_emotions(
    model: Chatterbox,
    processor: ChatterboxProcessor,
    text: str = "This is a test of emotion control.",
) -> dict[str, np.ndarray]:
    """
    Generate audio with all available emotions.

    Args:
        model: Chatterbox model
        processor: Chatterbox processor
        text: Text to synthesize

    Returns:
        Dictionary mapping emotion names to audio

    Example:
        >>> emotion_audios = demo_emotions(model, processor, "Hello world")
        >>> for emotion, audio in emotion_audios.items():
        ...     save_audio(audio, f"emotion_{emotion}.wav")
    """
    emotion_audios = {}

    print(f"\nGenerating '{text}' with all emotions:")

    for emotion in AVAILABLE_EMOTIONS:
        print(f"  {emotion}...")
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            emotion=emotion,
            expressiveness=0.7,
        )
        emotion_audios[emotion] = audio

    return emotion_audios


def demo_expressiveness_range(
    model: Chatterbox,
    processor: ChatterboxProcessor,
    text: str = "This is a test of expressiveness control.",
    steps: int = 5,
) -> dict[float, np.ndarray]:
    """
    Generate audio with different expressiveness levels.

    Args:
        model: Chatterbox model
        processor: Chatterbox processor
        text: Text to synthesize
        steps: Number of expressiveness levels to try

    Returns:
        Dictionary mapping expressiveness values to audio

    Example:
        >>> exp_audios = demo_expressiveness_range(model, processor, "Hello")
        >>> for exp_level, audio in exp_audios.items():
        ...     save_audio(audio, f"exp_{exp_level:.1f}.wav")
    """
    expressiveness_audios = {}

    print(f"\nGenerating '{text}' with different expressiveness:")

    for i in range(steps):
        exp_level = i / (steps - 1)  # 0.0 to 1.0
        print(f"  Expressiveness {exp_level:.2f}...")
        audio = synthesize(
            model=model,
            processor=processor,
            text=text,
            emotion="neutral",
            expressiveness=exp_level,
        )
        expressiveness_audios[exp_level] = audio

    return expressiveness_audios
