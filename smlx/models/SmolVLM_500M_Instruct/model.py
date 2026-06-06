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

        # Process image through vision encoder
        # Note: MLX expects [B, H, W, C] but preprocessing gives [B, C, H, W]
        # Handle both single image and batch
        if len(pixel_values.shape) == 4:
            # Batch of images [B, C, H, W] -> [B, H, W, C]
            pixel_values_mlx = pixel_values.transpose(0, 2, 3, 1)
        else:
            # Single image [C, H, W] -> [H, W, C]
            pixel_values_mlx = pixel_values.transpose(1, 2, 0)
            pixel_values_mlx = mx.expand_dims(pixel_values_mlx, axis=0)

        pooler_output, embeddings, hidden_state = self.vision_model(
            pixel_values_mlx, output_hidden_states=True
        )

        # Project vision features to language space
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

        # Find positions of <image> tokens (assuming batch size 1 for now)
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        if not image_positions:
            # No images in input
            return inputs_embeds

        num_images, num_patches, vision_hidden_size = image_features.shape

        # Reshape image features for insertion: [num_images * num_patches, hidden_size]
        reshaped_image_hidden_states = image_features.reshape(-1, vision_hidden_size)

        # Cast to match input embeddings dtype
        reshaped_image_hidden_states = reshaped_image_hidden_states.astype(inputs_embeds.dtype)

        # Replace <image> tokens with image features by building a new array
        # For SmolVLM: Each image replaces num_patches consecutive <image> tokens
        # Build list of embedding chunks
        batch_size = inputs_embeds.shape[0]
        result_parts = []

        for b in range(batch_size):
            parts = []
            last_pos = 0
            image_idx = 0

            for pos in image_positions:
                # Add text embeddings before this image position
                if pos > last_pos:
                    parts.append(inputs_embeds[b:b+1, last_pos:pos, :])

                # Add image embeddings
                img_start = image_idx * num_patches
                img_end = (image_idx + 1) * num_patches
                image_embeds = reshaped_image_hidden_states[img_start:img_end, :]
                # Add batch dimension: [num_patches, hidden_size] -> [1, num_patches, hidden_size]
                parts.append(mx.expand_dims(image_embeds, axis=0))

                last_pos = pos + 1
                image_idx += 1

            # Add remaining text embeddings after last image
            if last_pos < inputs_embeds.shape[1]:
                parts.append(inputs_embeds[b:b+1, last_pos:, :])

            # Concatenate all parts for this batch
            result_parts.append(mx.concatenate(parts, axis=1))

        # Concatenate all batches
        return mx.concatenate(result_parts, axis=0)

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
