#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TrOCR Image Processor and Tokenizer.

Handles image preprocessing and text tokenization for TrOCR.
"""

from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .config import TrOCRConfig


class TrOCRImageProcessor:
    """Image processor for TrOCR vision encoder.

    Resizes and normalizes images for BEiT encoder.
    """

    def __init__(
        self,
        image_size: int = 384,
        mean: tuple = (0.5, 0.5, 0.5),
        std: tuple = (0.5, 0.5, 0.5),
    ):
        """Initialize image processor.

        Args:
            image_size: Target image size
            mean: Normalization mean
            std: Normalization std
        """
        self.image_size = image_size
        self.mean = mx.array(mean).reshape(1, 1, 3)
        self.std = mx.array(std).reshape(1, 1, 3)

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load image from file.

        Args:
            image_path: Path to image

        Returns:
            PIL Image
        """
        return Image.open(image_path).convert("RGB")

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target size.

        Args:
            image: PIL Image

        Returns:
            Resized image
        """
        # Handle both old and new Pillow versions
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore
        return image.resize((self.image_size, self.image_size), resample)

    def normalize(self, image_array: mx.array) -> mx.array:
        """Normalize image.

        Args:
            image_array: Image array (H, W, C) in [0, 1]

        Returns:
            Normalized image
        """
        return (image_array - self.mean) / self.std

    def __call__(
        self, image: Union[str, Path, Image.Image, np.ndarray, mx.array]
    ) -> mx.array:
        """Process image for model input.

        Args:
            image: Image source

        Returns:
            Processed image (1, H, W, C) in MLX NHWC format
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = self.load_image(image)

        # Convert to PIL if numpy/mlx
        if isinstance(image, (np.ndarray, mx.array)):
            if isinstance(image, mx.array):
                image = np.array(image)
            image = Image.fromarray(image.astype(np.uint8))

        # Resize
        image = self.resize_image(image)

        # Convert to array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = mx.array(image_array)

        # Normalize
        image_array = self.normalize(image_array)

        # Add batch dimension (keep NHWC format for MLX)
        image_array = image_array[None, :]  # (1, H, W, C)

        return image_array


class TrOCRTokenizer:
    """Tokenizer for TrOCR text decoder.

    Uses XLMRoberta tokenizer with SentencePiece BPE.
    AutoTokenizer automatically selects the correct tokenizer class.
    """

    def __init__(
        self,
        vocab_size: int = 64044,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        model_name: Optional[str] = None,
    ):
        """Initialize tokenizer.

        Args:
            vocab_size: Vocabulary size
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            model_name: HuggingFace model name to load tokenizer from
        """
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # Load actual tokenizer if available
        self._tokenizer = None
        if model_name:
            try:
                from transformers import AutoTokenizer

                # Use AutoTokenizer to automatically select XLMRobertaTokenizer
                tokenizer_name = model_name if model_name else "microsoft/trocr-small-printed"
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

                # Verify and update vocab size from loaded tokenizer
                if hasattr(self._tokenizer, 'vocab_size'):
                    if self._tokenizer.vocab_size != self.vocab_size:
                        print(f"Info: Updating vocab size from config ({self.vocab_size}) "
                              f"to tokenizer ({self._tokenizer.vocab_size})")
                        self.vocab_size = self._tokenizer.vocab_size

            except Exception as e:
                # Fail loudly: a model_name was requested, so the caller expects a
                # real tokenizer. Silently dropping to char-level tokenization would
                # produce garbage OCR output that looks like success.
                raise RuntimeError(
                    f"Failed to load TrOCR tokenizer from '{model_name}': {e}. "
                    "A real tokenizer is required for correct OCR; refusing to fall "
                    "back to char-level tokenization."
                ) from e

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS

        Returns:
            Token IDs
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "TrOCRTokenizer has no real tokenizer backend. Construct it with a "
                "valid model_name (e.g. 'microsoft/trocr-small-printed') — char-level "
                "encoding would produce incorrect tokens for TrOCR's subword vocab."
            )
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self, token_ids: Union[list[int], mx.array], skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        # Convert to list of ints
        if isinstance(token_ids, mx.array):
            # Handle both 1D and 2D arrays
            token_ids_raw = token_ids.tolist()
            # Flatten if needed and ensure we have a list of ints
            if isinstance(token_ids_raw, list) and token_ids_raw:
                # If first element is a list, flatten
                if isinstance(token_ids_raw[0], list):
                    token_ids_list = [int(tid) for sublist in token_ids_raw for tid in sublist]
                else:
                    token_ids_list = [int(tid) for tid in token_ids_raw]
            else:
                # Single value
                token_ids_list = [int(token_ids_raw)] if isinstance(token_ids_raw, (int, float)) else []
        else:
            token_ids_list = [int(tid) for tid in token_ids]

        if self._tokenizer is None:
            raise RuntimeError(
                "TrOCRTokenizer has no real tokenizer backend. Construct it with a "
                "valid model_name (e.g. 'microsoft/trocr-small-printed') — char-level "
                "decoding would produce incorrect text for TrOCR's subword vocab."
            )

        # Filter out any invalid token IDs that are outside vocab
        # This prevents errors when decoding token IDs that don't exist in the vocabulary
        valid_token_ids = [
            tid for tid in token_ids_list
            if 0 <= tid < self._tokenizer.vocab_size
        ]

        if valid_token_ids:
            return self._tokenizer.decode(
                valid_token_ids, skip_special_tokens=skip_special_tokens
            )
        return ""

    def batch_decode(
        self, token_ids_batch: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        """Batch decode token IDs.

        Args:
            token_ids_batch: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return [
            self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_batch
        ]


class TrOCRProcessor:
    """Combined processor for TrOCR.

    Handles both image processing and tokenization.
    """

    def __init__(self, config: TrOCRConfig, model_name: Optional[str] = None):
        """Initialize processor.

        Args:
            config: TrOCR configuration
            model_name: Optional HuggingFace model name for loading tokenizer
        """
        self.config = config
        self.image_processor = TrOCRImageProcessor(
            image_size=config.vision_config.image_size
        )
        self.tokenizer = TrOCRTokenizer(
            vocab_size=config.decoder_config.vocab_size,
            bos_token_id=config.decoder_config.bos_token_id,
            eos_token_id=config.decoder_config.eos_token_id,
            pad_token_id=config.decoder_config.pad_token_id,
            model_name=model_name,
        )

    def process_image(
        self, image: Union[str, Path, Image.Image, np.ndarray, mx.array]
    ) -> mx.array:
        """Process image for model.

        Args:
            image: Image source

        Returns:
            Processed image tensor
        """
        return self.image_processor(image)

    def encode_text(self, text: str) -> list[int]:
        """Encode text to tokens.

        Args:
            text: Input text

        Returns:
            Token IDs
        """
        return self.tokenizer.encode(text)

    def decode_text(
        self, token_ids: Union[list[int], mx.array], skip_special_tokens: bool = True
    ) -> str:
        """Decode tokens to text.

        Args:
            token_ids: Token IDs
            skip_special_tokens: Skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens)

    def __call__(self, image, text=None):
        """Process image and optionally text.

        Args:
            image: Image source
            text: Optional text for encoding

        Returns:
            Dictionary with processed inputs
        """
        pixel_values = self.process_image(image)

        result = {"pixel_values": pixel_values}

        if text is not None:
            input_ids = self.encode_text(text)
            result["input_ids"] = mx.array(input_ids)[None, :]  # Add batch dim

        return result


__all__ = [
    "TrOCRProcessor",
    "TrOCRImageProcessor",
    "TrOCRTokenizer",
]
