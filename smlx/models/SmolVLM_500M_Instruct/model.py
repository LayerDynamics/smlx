#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SmolVLM-500M-Instruct Model Implementation.

A vision-language model combining:
- SigLIP 93M vision encoder (768 hidden size, 12 layers)
- SmolLM2-360M language model (960 hidden size, 32 layers)
- Idefics3 connector with pixel shuffle

Total parameters: ~500M
"""

import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .connector import Idefics3Connector
from .language import LanguageModel, LanguageModelOutput
from .vision import VisionModel


class Model(nn.Module):
    """SmolVLM-256M-Instruct multimodal model.

    Processes images through vision encoder, merges with text embeddings,
    and generates text with language model.

    Args:
        config: ModelConfig with vision, text, and model settings

    Example:
        >>> from smlx.models.SmolVLM_256M import load, generate
        >>> model, processor = load("HuggingFaceTB/SmolVLM-256M-Instruct")
        >>> image = load_image("photo.jpg")
        >>> output = generate(
        ...     model=model,
        ...     processor=processor,
        ...     prompt="Describe this image:",
        ...     image=image,
        ...     max_tokens=100
        ... )
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        # Three main components
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.connector = Idefics3Connector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Get merged text and image embeddings.

        Args:
            input_ids: Token IDs [B, seq_len] with <image> tokens
            pixel_values: Preprocessed images [B, C, H, W] in channel-first format
            pixel_attention_mask: Attention mask for images (unused in SigLIP)

        Returns:
            Merged embeddings [B, seq_len, hidden_size]
        """
        # Text-only case
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        # Get text embeddings
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Process image through vision encoder. Preprocessing gives one of:
        # - HF AutoProcessor: [B, num_sub_images, C, H, W] (large images split into
        #   sub-images, each expanded to the right number of <image> tokens)
        # - custom processor: [B, C, H, W] (batch) or [C, H, W] (single)
        # MLX vision model expects channel-last [..., H, W, C].
        if len(pixel_values.shape) == 5:
            # HF processor format: [B, num_sub_images, C, H, W]. Take the first
            # batch element -> [num_sub_images, H, W, C]; the connector pixel-shuffles
            # each sub-image into its token block.
            pixel_values_mlx = pixel_values[0].transpose(0, 2, 3, 1)
        elif len(pixel_values.shape) == 4:
            # Custom processor batch [B, C, H, W] -> [B, H, W, C]
            pixel_values_mlx = pixel_values.transpose(0, 2, 3, 1)
        else:
            # Single image [C, H, W] -> [1, H, W, C]
            pixel_values_mlx = pixel_values.transpose(1, 2, 0)
            pixel_values_mlx = mx.expand_dims(pixel_values_mlx, axis=0)

        pooler_output, embeddings, hidden_state = self.vision_model(
            pixel_values_mlx, output_hidden_states=True
        )

        # Project vision features to language space (connector does pixel shuffle)
        image_features = pooler_output.astype(inputs_embeds.dtype)
        image_features = self.connector(image_features)

        # Merge text and image embeddings
        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )

        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(
        self, image_features: mx.array, inputs_embeds: mx.array, input_ids: mx.array
    ) -> mx.array:
        """Replace <image> tokens with actual image embeddings.

        Args:
            image_features: Vision features [num_images, num_patches, hidden_size]
            inputs_embeds: Text embeddings [B, seq_len, hidden_size]
            input_ids: Token IDs [B, seq_len] with <image> tokens

        Returns:
            Merged embeddings [B, seq_len', hidden_size]
        """
        image_token_index = self.config.image_token_index

        # Find positions of <image> tokens (assuming batch size 1)
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        if not image_positions:
            # No images in input
            return inputs_embeds

        num_images, num_patches, vision_hidden_size = image_features.shape

        # Cast to match input embeddings dtype
        image_features = image_features.astype(inputs_embeds.dtype)

        # The HF processor expands each <image> placeholder into exactly the number
        # of vision tokens the encoder produces, so this is a 1:1 replacement (no
        # change in sequence length). Reshape features to [1, total_tokens, hidden].
        reshaped_image_features = image_features.reshape(1, -1, vision_hidden_size)

        num_image_tokens = len(image_positions)
        num_image_features = reshaped_image_features.shape[1]
        if num_image_tokens != num_image_features:
            raise ValueError(
                f"Token count mismatch: found {num_image_tokens} <image> token positions "
                f"but have {num_image_features} vision features "
                f"(image_features {image_features.shape}). This indicates a mismatch "
                f"between processor token expansion and vision encoder output."
            )

        # MLX can't do in-place positional assignment; use put_along_axis to place
        # each vision feature at its <image> token position (1:1, seq_len unchanged).
        position_indices = mx.array(image_positions).reshape(1, -1, 1)
        position_indices_broadcast = mx.broadcast_to(
            position_indices, reshaped_image_features.shape
        )
        final_embeds = mx.put_along_axis(
            inputs_embeds,
            position_indices_broadcast,
            reshaped_image_features,
            axis=1,
        )
        return final_embeds

    @property
    def layers(self):
        """Access language model layers."""
        return self.language_model.layers

    @property
    def vision(self):
        """Alias for vision_model."""
        return self.vision_model

    @property
    def language(self):
        """Alias for language_model."""
        return self.language_model

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        """Forward pass through VLM.

        Args:
            input_ids: Token IDs [B, seq_len]
            pixel_values: Images [B, C, H, W] or None for text-only
            cache: KV cache for generation

        Returns:
            LanguageModelOutput with logits
        """
        # Get merged embeddings
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)

        # Forward through language model
        output = self.language_model(inputs=input_ids, cache=cache, inputs_embeds=input_embeddings)

        return output

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights during loading.

        Handles PyTorch -> MLX conversion:
        - Remove "model." prefix
        - Rename "text_model" to "language_model"
        - Move "lm_head" to language_model
        """
        # Remove "model." prefix
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.", k)
                else (f"language_model.{k}" if re.match(r"^lm_head\.", k) else k)
            ): v
            for k, v in weights.items()
        }

        # Rename "text_model" to "language_model"
        weights = {
            (f"language_model.{k.split('.', 1)[1]}" if re.match(r"^text_model\.", k) else k): v
            for k, v in weights.items()
        }

        return weights


__all__ = ["Model"]
