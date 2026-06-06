#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
SmolVLM-256M-Instruct Model Implementation.

A vision-language model combining:
- SigLIP-SO400M vision encoder (1152 hidden size)
- SmolLM2-135M language model (576 hidden size)
- Idefics3 connector with pixel shuffle

Total parameters: ~256M
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

        This method handles three image input formats:
        1. HF processor multi-image: [B, num_sub_images, C, H, W] - for large images
           split into patches (e.g., 17 sub-images of 512×512 each)
        2. Custom processor batch: [B, C, H, W] - batch of single images
        3. Custom processor single: [C, H, W] - single image

        Multi-image scenario (HF processor):
        - Large images are split into overlapping sub-images (e.g., 17 patches)
        - Each sub-image processed through vision encoder → [num_sub_images, num_patches, 768]
        - Connector projects to text space → [num_sub_images, 64, 576]
        - Total tokens: 17 × 64 = 1088 vision tokens for one large image
        - These 1088 tokens replace 1088 <image> tokens in the input sequence

        Args:
            input_ids: Token IDs [B, seq_len] with <image> tokens
            pixel_values: Preprocessed images in one of three formats:
                - [B, num_sub_images, C, H, W] (HF processor, multi-patch)
                - [B, C, H, W] (custom processor, batch)
                - [C, H, W] (custom processor, single)
            pixel_attention_mask: Attention mask for images (unused in SigLIP)

        Returns:
            Merged embeddings [B, seq_len, hidden_size] where seq_len includes
            all vision tokens (64-1088 per image depending on splitting)
        """
        # Text-only case
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        # Get text embeddings
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Process image through vision encoder
        # Note: MLX expects [B, H, W, C] but preprocessing gives different formats:
        # - Custom processor: [B, C, H, W] or [C, H, W]
        # - HF processor: [B, num_sub_images, C, H, W]

        if len(pixel_values.shape) == 5:
            # HF processor format: [B, num_sub_images, C, H, W]
            # Multi-image scenario: Large images split into sub-patches for better detail
            #
            # Example for 1536×1536 image:
            # - Split into 17 overlapping 512×512 patches (3×3 grid + 8 half-overlaps)
            # - Each patch: 512×512 → SigLIP → 1024 patches × 768 dim
            # - Connector: 1024 patches → 64 tokens × 576 dim (via pixel shuffle)
            # - Total: 17 patches × 64 tokens = 1088 vision tokens
            B, num_sub_images, C, H, W = pixel_values.shape

            # Take first batch element (mlx-vlm approach)
            # pixel_values[0] is [num_sub_images, C, H, W]
            # Convert to MLX format [num_sub_images, H, W, C]
            pixel_values_mlx = pixel_values[0].transpose(0, 2, 3, 1)

            # Process through vision encoder
            # Output: [num_sub_images, num_patches, hidden_size]
            # e.g., [17, 1024, 768] for 17 sub-images
            pooler_output, embeddings, hidden_state = self.vision_model(
                pixel_values_mlx, output_hidden_states=True
            )

            # Connector will process each sub-image separately via pixel_shuffle
            # Output: [num_sub_images, 64, text_hidden_size]
            # e.g., [17, 64, 576] which becomes [1, 1088, 576] after reshaping
            # These 1088 vision embeddings will replace 1088 <image> tokens

        elif len(pixel_values.shape) == 4:
            # Custom processor: Batch of images [B, C, H, W] -> [B, H, W, C]
            pixel_values_mlx = pixel_values.transpose(0, 2, 3, 1)

            pooler_output, embeddings, hidden_state = self.vision_model(
                pixel_values_mlx, output_hidden_states=True
            )
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

        This method handles the injection-based approach where <image> tokens in the
        input sequence are replaced with actual vision embeddings. The HuggingFace
        processor expands each <image> token into multiple tokens (64-1088 depending
        on image splitting), and this method assigns the corresponding vision features.

        Args:
            image_features: Vision features [num_images, num_patches, hidden_size]
                           e.g., [17, 64, 576] for multi-patch images
            inputs_embeds: Text embeddings [B, seq_len, hidden_size]
            input_ids: Token IDs [B, seq_len] with <image> tokens

        Returns:
            Merged embeddings [B, seq_len, hidden_size] with image tokens replaced

        Raises:
            ValueError: If number of image token positions doesn't match image features
        """
        image_token_index = self.config.image_token_index

        # Find positions of <image> tokens (assuming batch size 1 for now)
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        if not image_positions:
            # No images in input
            return inputs_embeds

        num_images, num_patches, vision_hidden_size = image_features.shape

        # Cast to match input embeddings dtype
        image_features = image_features.astype(inputs_embeds.dtype)

        # With HF processor, image_positions contains ALL image token positions
        # Examples:
        # - Single patch: 64 tokens (for 512x512 image)
        # - Multi-patch: 1088 tokens (17 sub-images × 64 tokens each)
        # Reshape image features to match: [1, num_images * num_patches, hidden_size]
        # For HF: [17, 64, 576] -> [1, 1088, 576]
        reshaped_image_features = image_features.reshape(1, -1, vision_hidden_size)

        # Validation: Ensure token count matches feature count
        num_image_tokens = len(image_positions)
        num_image_features = reshaped_image_features.shape[1]

        if num_image_tokens != num_image_features:
            raise ValueError(
                f"Token count mismatch: Found {num_image_tokens} <image> token positions "
                f"but have {num_image_features} vision features. "
                f"Image features shape: {image_features.shape}, "
                f"Reshaped: {reshaped_image_features.shape}. "
                f"This usually indicates a mismatch between processor token expansion "
                f"and vision encoder output."
            )

        # MLX workaround: Can't use direct item assignment (inputs_embeds[:, positions, :] = features)
        # Instead, use mx.put_along_axis() to insert image features at specified positions
        # Reference: Similar to nanoVLM implementation but adapted for multiple positions

        batch_size, seq_len, embed_dim = inputs_embeds.shape

        # Reshape position indices for put_along_axis: [1, num_positions, 1]
        position_indices = mx.array(image_positions).reshape(1, -1, 1)

        # Broadcast position indices to match embedding dimension
        # [1, num_positions, embed_dim]
        position_indices_broadcast = mx.broadcast_to(position_indices, reshaped_image_features.shape)

        # Use put_along_axis to insert image features at the specified positions
        # This creates a new array with image features at image token positions
        final_embeds = mx.put_along_axis(
            inputs_embeds,
            position_indices_broadcast,
            reshaped_image_features,
            axis=1
        )

        return final_embeds

    @property
    def layers(self):
        """Access language model layers."""
        return self.language_model.layers

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
        output = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )

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
            (
                f"language_model.{k.split('.', 1)[1]}"
                if re.match(r"^text_model\.", k)
                else k
            ): v
            for k, v in weights.items()
        }

        return weights


__all__ = ["Model"]
