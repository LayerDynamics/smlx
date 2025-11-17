"""
Data loaders for SMLX - Loading images, audio, text, and other modalities.

This module provides centralized data loading utilities for all modalities,
supporting various input sources (files, URLs, BytesIO, base64, etc.).

Adapted from:
- resources/mlx-vlm/mlx_vlm/utils.py (image/audio loading patterns)
- smlx/utils/vision.py (existing SMLX image loading)
- smlx/models/Whisper_tiny/audio.py (audio loading patterns)
"""

import subprocess
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np
import requests
from PIL import Image, ImageOps


def load_image(
    image_source: Union[str, Path, BytesIO, Image.Image], timeout: int = 10
) -> Image.Image:
    """
    Load an image from a file path, URL, BytesIO, base64 data URI, or PIL Image.

    Supports multiple input sources:
    - File path (str or Path)
    - URL (str starting with http:// or https://)
    - BytesIO object
    - Base64 data URI (str starting with data:image/)
    - PIL Image object (returned as-is after conversion to RGB)

    Args:
        image_source: Source of the image
        timeout: Timeout in seconds for HTTP requests (default: 10)

    Returns:
        PIL Image in RGB format with EXIF orientation applied

    Raises:
        ValueError: If the image source is invalid or cannot be loaded

    Example:
        >>> # Load from file
        >>> img = load_image("photo.jpg")
        >>>
        >>> # Load from URL
        >>> img = load_image("https://example.com/image.jpg")
        >>>
        >>> # Load from base64 data URI
        >>> img = load_image("data:image/jpeg;base64,/9j/4AAQ...")
    """
    # If already a PIL Image, just convert to RGB and return
    if isinstance(image_source, Image.Image):
        image = ImageOps.exif_transpose(image_source)
        return image.convert("RGB")

    # Handle BytesIO, base64 data URIs, or file paths
    if (
        isinstance(image_source, BytesIO)
        or (isinstance(image_source, str) and image_source.startswith("data:image/"))
        or Path(image_source).is_file()
    ):
        try:
            # Handle base64 encoded data URIs
            if isinstance(image_source, str) and image_source.startswith("data:image/"):
                import base64

                if "," not in image_source:
                    raise ValueError("Invalid data URI format - missing comma separator")

                _, data = image_source.split(",", 1)
                image_source = BytesIO(base64.b64decode(data))

            image = Image.open(image_source)
        except OSError as e:
            raise ValueError(
                f"Failed to load image from {image_source} with error: {e}"
            ) from e

    # Handle HTTP(S) URLs
    elif isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            ) from e
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL, file path, BytesIO, "
            f"base64 data URI, or PIL Image."
        )

    # Apply EXIF orientation and convert to RGB
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using linear interpolation.

    Args:
        audio: Audio array to resample
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array

    Example:
        >>> audio_48k = np.random.randn(48000)
        >>> audio_16k = resample_audio(audio_48k, orig_sr=48000, target_sr=16000)
        >>> audio_16k.shape
        (16000,)
    """
    if orig_sr == target_sr:
        return audio

    # Calculate the resampling ratio
    ratio = target_sr / orig_sr

    # Handle different audio shapes
    if audio.ndim == 1:
        # Mono audio - simple case
        new_length = int(len(audio) * ratio)
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, audio)

    elif audio.ndim == 2:
        # Multi-channel audio - transpose to (samples, channels) if needed
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T

        # Resample each channel
        n_samples, n_channels = audio.shape
        new_length = int(n_samples * ratio)
        old_indices = np.arange(n_samples)
        new_indices = np.linspace(0, n_samples - 1, new_length)

        resampled = np.zeros((new_length, n_channels))
        for i in range(n_channels):
            resampled[:, i] = np.interp(new_indices, old_indices, audio[:, i])
    else:
        raise ValueError(f"Audio array has unsupported shape: {audio.shape}")

    return resampled


def load_audio(
    file: Union[str, Path],
    sr: int = 16000,
    mono: bool = True,
    timeout: int = 10,
) -> mx.array:
    """
    Load audio from a file path or URL using ffmpeg.

    Supports various audio formats via ffmpeg, with automatic resampling
    and conversion to mono if requested.

    Args:
        file: Audio file path or URL
        sr: Target sample rate (default: 16000)
        mono: Convert to mono by averaging channels (default: True)
        timeout: Timeout in seconds for HTTP requests (default: 10)

    Returns:
        MLX array containing audio samples as float32 [-1, 1]

    Raises:
        ValueError: If audio cannot be loaded
        RuntimeError: If ffmpeg is not available

    Example:
        >>> # Load from file
        >>> audio = load_audio("speech.wav", sr=16000)
        >>>
        >>> # Load from URL
        >>> audio = load_audio("https://example.com/audio.mp3", sr=16000)
    """
    try:
        import soundfile as sf

        # Handle URLs
        if isinstance(file, str) and file.startswith(("http://", "https://")):
            try:
                response = requests.get(file, stream=True, timeout=timeout)
                response.raise_for_status()
                audio, sample_rate = sf.read(BytesIO(response.content), always_2d=True)
            except Exception as e:
                raise ValueError(f"Failed to load audio from URL: {file} with error {e}") from e
        else:
            # Local file
            audio, sample_rate = sf.read(file, always_2d=True)

        # Resample if needed
        if sample_rate != sr:
            audio = resample_audio(audio, sample_rate, sr)

        # Convert to mono if requested
        if mono and audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Convert to MLX array
        return mx.array(audio.astype(np.float32))

    except ImportError:
        # Fallback to ffmpeg if soundfile not available
        return _load_audio_ffmpeg(file, sr=sr, mono=mono)


def _load_audio_ffmpeg(
    file: Union[str, Path], sr: int = 16000, mono: bool = True
) -> mx.array:
    """
    Load audio using ffmpeg (fallback method).

    Args:
        file: Audio file path
        sr: Target sample rate
        mono: Convert to mono

    Returns:
        MLX array containing audio samples

    Raises:
        RuntimeError: If ffmpeg is not available or fails
    """
    try:
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            str(file),
            "-f",
            "s16le",
            "-ac",
            "1" if mono else "2",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]

        # Run ffmpeg
        out = subprocess.run(cmd, capture_output=True, check=True).stdout

    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg or install soundfile: "
            "pip install soundfile"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to load audio: {e.stderr.decode()}") from e

    # Convert to numpy array
    audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0

    return mx.array(audio)


def load_text(file: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Load text from a file.

    Args:
        file: Text file path
        encoding: Text encoding (default: "utf-8")

    Returns:
        Text content as string

    Raises:
        ValueError: If file cannot be read

    Example:
        >>> text = load_text("document.txt")
        >>> print(text[:100])
    """
    try:
        with open(file, "r", encoding=encoding) as f:
            return f.read()
    except OSError as e:
        raise ValueError(f"Failed to load text from {file} with error: {e}") from e


def load_video(
    file: Union[str, Path],
    fps: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> list[Image.Image]:
    """
    Load video frames as a list of PIL Images.

    Uses ffmpeg to extract frames from video files.

    Args:
        file: Video file path
        fps: Target frames per second (default: use original fps)
        max_frames: Maximum number of frames to load (default: load all)

    Returns:
        List of PIL Images (one per frame)

    Raises:
        RuntimeError: If ffmpeg is not available or fails
        ValueError: If video cannot be loaded

    Example:
        >>> frames = load_video("video.mp4", fps=1, max_frames=10)
        >>> print(f"Loaded {len(frames)} frames")
    """
    try:
        # Build ffmpeg command to extract frames
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(file),
        ]

        # Add FPS filter if specified
        if fps is not None:
            cmd.extend(["-vf", f"fps={fps}"])

        # Add max frames limit
        if max_frames is not None:
            cmd.extend(["-frames:v", str(max_frames)])

        # Output as raw RGB24 frames
        cmd.extend(["-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"])

        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, check=True)

    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to load video: {e.stderr.decode()}") from e

    # Get video dimensions (need to parse ffmpeg output)
    # For now, assume we need to get width/height from ffprobe
    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(file),
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, check=True, text=True)
        width, height = map(int, probe_result.stdout.strip().split("x"))
    except Exception:
        raise ValueError("Failed to get video dimensions. Is ffprobe installed?")

    # Parse raw frames
    raw_frames = result.stdout
    frame_size = width * height * 3  # RGB24

    frames = []
    for i in range(0, len(raw_frames), frame_size):
        frame_data = raw_frames[i : i + frame_size]
        if len(frame_data) < frame_size:
            break

        # Convert to PIL Image
        frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 3)
        frames.append(Image.fromarray(frame_array))

    return frames
