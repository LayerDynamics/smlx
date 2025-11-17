"""
Preprocessing pipelines for SMLX - Image, audio, and text preprocessing.

This module provides standard preprocessing classes for different modalities,
consolidating logic currently scattered across the codebase.

Adapted from:
- smlx/utils/vision.py (image preprocessing)
- resources/mlx-examples/clip/image_processor.py (CLIP-style processing)
- smlx/models/Whisper_tiny/audio.py (audio preprocessing)
"""

from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizer


class ImagePreprocessor:
    """
    Standard image preprocessing pipeline for vision models.

    Applies common transformations: resize, normalize, convert to tensor.
    Follows patterns from CLIP, SigLIP, and other vision encoders.

    Args:
        size: Target size (width, height) or single int for square
        resize_mode: Resize mode - "shortest_edge", "longest_edge", or "exact"
        crop_size: Optional crop size after resize
        mean: Mean values for normalization per channel (default: ImageNet mean)
        std: Standard deviation for normalization per channel (default: ImageNet std)
        rescale_factor: Factor to rescale pixel values (default: 1/255)
        do_normalize: Whether to apply normalization (default: True)
        do_center_crop: Whether to apply center crop (default: False)

    Example:
        >>> processor = ImagePreprocessor(
        ...     size=224,
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
        >>> image = Image.open("photo.jpg")
        >>> pixel_values = processor(image)  # Returns MLX array [C, H, W]
    """

    def __init__(
        self,
        size: Union[int, tuple[int, int]] = 224,
        resize_mode: str = "shortest_edge",
        crop_size: Optional[Union[int, tuple[int, int]]] = None,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        do_center_crop: bool = False,
    ):
        # Convert size to tuple
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.resize_mode = resize_mode
        self.do_center_crop = do_center_crop

        # Convert crop_size to tuple
        if crop_size is not None:
            if isinstance(crop_size, int):
                self.crop_size = (crop_size, crop_size)
            else:
                self.crop_size = crop_size
        else:
            self.crop_size = None

        # Default to ImageNet normalization
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

    def resize(self, image: Image.Image) -> Image.Image:
        """Resize image according to resize_mode."""
        if self.resize_mode == "shortest_edge":
            # Resize so shortest edge matches target, preserving aspect ratio
            scale = max(self.size[0] / image.width, self.size[1] / image.height)
            new_size = (int(image.width * scale), int(image.height * scale))
            return image.resize(new_size, Image.Resampling.BICUBIC)

        elif self.resize_mode == "longest_edge":
            # Resize so longest edge matches target, preserving aspect ratio
            scale = min(self.size[0] / image.width, self.size[1] / image.height)
            new_size = (int(image.width * scale), int(image.height * scale))
            return image.resize(new_size, Image.Resampling.BICUBIC)

        elif self.resize_mode == "exact":
            # Resize to exact dimensions, ignoring aspect ratio
            return image.resize(self.size, Image.Resampling.BICUBIC)

        else:
            raise ValueError(
                f"Invalid resize_mode '{self.resize_mode}'. "
                f"Must be 'shortest_edge', 'longest_edge', or 'exact'"
            )

    def center_crop(self, image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
        """Apply center crop to image."""
        width, height = image.size
        crop_w, crop_h = crop_size

        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        return image.crop((left, top, right, bottom))

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization with mean and std."""
        if self.do_normalize:
            mean_array = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
            std_array = np.array(self.std, dtype=np.float32).reshape(-1, 1, 1)
            image = (image - mean_array) / std_array
        return image

    def __call__(self, image: Image.Image) -> mx.array:
        """
        Preprocess image and return MLX array.

        Args:
            image: PIL Image to preprocess

        Returns:
            MLX array of shape [C, H, W]
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        image = self.resize(image)

        # Center crop if requested
        if self.do_center_crop and self.crop_size is not None:
            image = self.center_crop(image, self.crop_size)

        # Convert to numpy array [H, W, C]
        img_array = np.array(image)

        # Transpose to [C, H, W]
        img_array = img_array.transpose(2, 0, 1)

        # Rescale pixel values (typically [0, 255] → [0, 1])
        if self.rescale_factor != 1.0:
            img_array = img_array.astype(np.float32) * self.rescale_factor
        else:
            img_array = img_array.astype(np.float32)

        # Normalize
        img_array = self.normalize(img_array)

        # Convert to MLX array
        return mx.array(img_array)


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for speech and audio models.

    Handles mel-spectrogram computation and normalization.

    Args:
        sample_rate: Target sample rate (default: 16000)
        n_fft: FFT size (default: 400)
        hop_length: Hop length for STFT (default: 160)
        n_mels: Number of mel filterbanks (default: 80)
        fmin: Minimum frequency (default: 0)
        fmax: Maximum frequency (default: 8000)
        normalize: Whether to normalize features (default: True)

    Example:
        >>> processor = AudioPreprocessor(sample_rate=16000, n_mels=80)
        >>> audio = mx.random.randn((16000,))  # 1 second of audio
        >>> features = processor(audio)  # Returns mel-spectrogram
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        normalize: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize

    def compute_mel_spectrogram(self, audio: mx.array) -> mx.array:
        """
        Compute mel-spectrogram from audio.

        Args:
            audio: Audio array of shape (num_samples,)

        Returns:
            Mel-spectrogram of shape (n_mels, time)
        """
        # Convert to numpy for processing (MLX doesn't have full audio processing yet)
        audio_np = np.array(audio)

        try:
            import librosa

            # Compute mel-spectrogram using librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )

            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        except ImportError:
            # Fallback: Simple STFT-based approach if librosa not available
            from scipy import signal

            # Compute STFT
            f, t, Zxx = signal.stft(
                audio_np,
                fs=self.sample_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
            )

            # Take magnitude
            mel_spec = np.abs(Zxx)

            # Convert to log scale
            mel_spec = np.log(mel_spec + 1e-9)

        # Normalize if requested
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        return mx.array(mel_spec.astype(np.float32))

    def __call__(self, audio: mx.array) -> mx.array:
        """
        Preprocess audio and return mel-spectrogram.

        Args:
            audio: Audio array

        Returns:
            Mel-spectrogram features
        """
        return self.compute_mel_spectrogram(audio)


class TextPreprocessor:
    """
    Text preprocessing and tokenization pipeline.

    Wraps tokenizer with additional preprocessing options.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default: 512)
        padding: Padding strategy - "max_length", "longest", or False (default: False)
        truncation: Whether to truncate to max_length (default: True)
        add_special_tokens: Whether to add special tokens (default: True)

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> processor = TextPreprocessor(tokenizer, max_length=128)
        >>> result = processor("Hello world!")
        >>> result.keys()
        dict_keys(['input_ids', 'attention_mask'])
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: Union[bool, str] = False,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens

    def __call__(
        self, text: Union[str, list[str]], return_mlx: bool = True
    ) -> dict[str, Union[mx.array, list[int]]]:
        """
        Tokenize text and optionally convert to MLX arrays.

        Args:
            text: Text string or list of strings
            return_mlx: Whether to return MLX arrays (default: True)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_tokens,
            return_tensors=None,  # Get lists first
        )

        # Convert to MLX arrays if requested
        if return_mlx:
            result = {}
            for key, val in encoding.items():
                try:
                    # Try to convert to array
                    result[key] = mx.array(val)
                except (ValueError, TypeError):
                    # Keep as list if conversion fails (e.g., non-uniform lengths)
                    result[key] = val
            return result
        else:
            return dict(encoding)


class MultimodalPreprocessor:
    """
    Combined preprocessor for multimodal (vision + language) inputs.

    Args:
        image_processor: ImagePreprocessor instance
        text_processor: TextPreprocessor instance

    Example:
        >>> image_proc = ImagePreprocessor(size=224)
        >>> text_proc = TextPreprocessor(tokenizer, max_length=128)
        >>> processor = MultimodalPreprocessor(image_proc, text_proc)
        >>>
        >>> result = processor(
        ...     image="photo.jpg",
        ...     text="What is in this image?"
        ... )
        >>> result.keys()
        dict_keys(['pixel_values', 'input_ids', 'attention_mask'])
    """

    def __init__(
        self,
        image_processor: ImagePreprocessor,
        text_processor: TextPreprocessor,
    ):
        self.image_processor = image_processor
        self.text_processor = text_processor

    def __call__(
        self,
        image: Optional[Union[str, Image.Image]] = None,
        text: Optional[Union[str, list[str]]] = None,
    ) -> dict[str, mx.array]:
        """
        Process multimodal inputs.

        Args:
            image: Image source (optional)
            text: Text string or list (optional)

        Returns:
            Dictionary with processed inputs
        """
        from .loaders import load_image

        result = {}

        if image is not None:
            # Load and process image
            if isinstance(image, str):
                image = load_image(image)
            result["pixel_values"] = self.image_processor(image)

        if text is not None:
            # Process text
            text_result = self.text_processor(text)
            result.update(text_result)

        return result
