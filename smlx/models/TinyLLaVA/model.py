#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
TinyLLaVA Vision-Language Model.

A 1.5B parameter multimodal model combining:
- SigLIP vision encoder
- TinyLlama language model
- Simple MLP projector
"""

import logging
import os
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .vision import VisionModel
from .language import TinyLlamaModel
from .connector import build_projector
from smlx.utils.cache import KVCache
from smlx.utils.vlm_diagnostics import (
    log_embedding_comparison,
    log_logits_distribution,
    log_vision_features,
)

logger = logging.getLogger(__name__)

# Enable debug logging if SMLX_DEBUG is set
DEBUG = os.getenv("SMLX_DEBUG", "0") == "1"
if DEBUG:
    logger.setLevel(logging.DEBUG)


class TinyLLaVA(nn.Module):
    """TinyLLaVA multimodal model.

    Architecture:
    - Vision Encoder: SigLIP (1152 hidden, 27 layers)
    - Vision-Language Connector: 2-layer MLP
    - Language Model: TinyLlama (2048 hidden, 22 layers)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_tower = VisionModel(config.vision_config)

        # Language model
        self.language_model = TinyLlamaModel(config.text_config)

        # Vision-language projector
        self.multi_modal_projector = build_projector(
            config.projector_config,
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
        )

        # Language model head
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

        # Vision feature layer selection
        self.vision_feature_layer = config.vision_feature_layer

    def get_vision_tower(self):
        """Get the vision encoder."""
        return self.vision_tower

    def get_language_model(self):
        """Get the language model."""
        return self.language_model

    def encode_images(self, pixel_values: mx.array) -> mx.array:
        """Encode images into visual embeddings.

        Args:
            pixel_values: Image tensor [B, H, W, C] (MLX format)

        Returns:
            Visual embeddings [B, num_patches, text_hidden_size]
        """
        # Extract vision features
        # VisionModel returns: (pooler_output, initial_embeddings, hidden_states)
        vision_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=True,
        )

        # Select features from specified layer
        if self.vision_feature_layer == -1:
            vision_features = vision_outputs[0]  # Last layer (pooler_output)
        else:
            # Get hidden states from specific layer
            all_hidden_states = vision_outputs[2]  # Hidden states tuple
            vision_features = all_hidden_states[self.vision_feature_layer]

        if DEBUG:
            log_vision_features(vision_features, label="tinyllava_vision_features_pre_proj")

        # Project to language space
        image_features = self.multi_modal_projector(vision_features)

        if DEBUG:
            log_vision_features(image_features, label="tinyllava_projected_vision_features")

        # No additional normalization or scaling - let the learned projector handle it
        # Dtype casting happens later in prepare_inputs_for_generation() to match text embeddings

        return image_features

    def prepare_inputs_for_generation(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_features: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Prepare inputs using token injection to replace <image> tokens.

        Args:
            input_ids: Text token IDs [B, seq_len] containing <image> token(s)
            pixel_values: Optional image tensor [B, H, W, C]
            image_features: Pre-computed image features [B, num_patches, hidden_size]

        Returns:
            Tuple of (combined_embeddings, image_features)
        """
        # Encode images if provided
        if pixel_values is not None and image_features is None:
            image_features = self.encode_images(pixel_values)

        # Get text embeddings
        text_embeddings = self.language_model.embed_tokens(input_ids)

        # If no images, return text embeddings only
        if image_features is None:
            return text_embeddings, None

        # Token injection: replace <image> token embeddings with vision features
        # LLaVA approach: find positions of IMAGE_TOKEN_INDEX and inject features
        import numpy as np

        # Find image token positions (assuming batch size is 1)
        image_token_index = self.config.image_token_index
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        if not image_positions:
            # No image tokens found, return text only
            return text_embeddings, image_features

        # One <image> placeholder token is replaced by that image's patch
        # features, so the number of images must match the number of tokens.
        num_images = image_features.shape[0]

        if DEBUG:
            log_embedding_comparison(
                image_features, text_embeddings, label="tinyllava_vision_vs_text_embeds"
            )

        # Inject image features at token positions
        # This replaces the single <image> token with num_patches features
        # Strategy: split text embeddings and insert image features

        # Replace each <image> token (in order) with its image's patch features.
        # Handles one or many images; each <image> placeholder maps to one image.
        if len(image_positions) != num_images:
            raise ValueError(
                f"Number of <image> tokens ({len(image_positions)}) does not match "
                f"the number of images ({num_images}); each <image> placeholder must "
                f"correspond to exactly one image."
            )

        feats = image_features.astype(text_embeddings.dtype)
        segments = []
        prev = 0
        for img_idx, pos in enumerate(sorted(image_positions)):
            # Text tokens up to (not including) this <image> placeholder.
            segments.append(text_embeddings[:, prev:pos, :])
            # This image's patch embeddings: (1, num_patches, hidden_size).
            segments.append(feats[img_idx : img_idx + 1])
            prev = pos + 1  # Skip the <image> token itself.
        # Trailing text after the last <image> token.
        segments.append(text_embeddings[:, prev:, :])
        combined_embeddings = mx.concatenate(segments, axis=1)

        return combined_embeddings, image_features

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_features: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass.

        Args:
            input_ids: Text token IDs [B, seq_len]
            pixel_values: Optional image tensor [B, C, H, W]
            image_features: Pre-computed image features
            mask: Optional attention mask
            cache: Optional KV cache for generation

        Returns:
            Tuple of (logits, image_features)
        """
        # Prepare combined inputs
        hidden_states, image_features = self.prepare_inputs_for_generation(
            input_ids, pixel_values, image_features
        )

        # Forward through language model using forward_embeddings
        # This properly handles the embedding input instead of token IDs
        hidden_states = self.language_model.forward_embeddings(
            hidden_states, mask=mask, cache=cache
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        if DEBUG:
            # Log logits distribution for last token (used for generation)
            log_logits_distribution(logits[:, -1, :], label="tinyllava_output_logits")

        return logits, image_features

    @staticmethod
    def sanitize(weights):
        """Sanitize weights from HuggingFace format to MLX format.

        Handles layer name mapping and parameter reshaping.
        """
        from .vision import VisionModel

        sanitized_weights = {}

        for k, v in weights.items():
            # Remove model prefix if present
            if k.startswith("model."):
                k = k.replace("model.", "", 1)  # Only replace first occurrence

            # Fix vision tower duplicate prefix and naming
            # HF weights have: vision_tower.vision_tower.vision_model.encoder...
            # Model expects: vision_tower.encoder...
            if k.startswith("vision_tower.vision_tower."):
                k = k.replace("vision_tower.vision_tower.", "vision_tower.", 1)
                k = k.replace("vision_model.", "", 1)  # Remove vision_model prefix
                k = k.replace("vision_embeddings.", "embeddings.", 1)
                k = k.replace("vision_post_layernorm.", "post_layernorm.", 1)

            # Add language_model prefix if it's missing
            # HF weights have: embed_tokens.weight, layers.X..., norm.weight
            # Model expects: language_model.embed_tokens.weight, language_model.layers.X...
            if (
                k.startswith("embed_tokens.")
                or k.startswith("layers.")
                or k.startswith("norm.weight")
            ):
                k = f"language_model.{k}"

            # Map layer names
            k = k.replace(".layer.", ".layers.")
            k = k.replace(".transformer.", ".")

            # Map projector keys: mm_projector -> multi_modal_projector
            k = k.replace("mm_projector.", "multi_modal_projector.")

            # Map projector Sequential indices to named layers
            # HF uses Sequential with indices 0, 1 (activation), 2
            # Our model uses named layers: linear_1, activation, linear_2
            k = k.replace("multi_modal_projector.0.", "multi_modal_projector.linear_1.")
            k = k.replace("multi_modal_projector.2.", "multi_modal_projector.linear_2.")

            sanitized_weights[k] = v

        # Apply vision model-specific sanitization (conv2d weight transposition)
        vision_weights = {
            k: v for k, v in sanitized_weights.items() if k.startswith("vision_tower.")
        }
        vision_weights_sanitized = VisionModel.sanitize(vision_weights)

        # Update with sanitized vision weights
        for k in vision_weights:
            if k in vision_weights_sanitized:
                sanitized_weights[k] = vision_weights_sanitized[k]

        return sanitized_weights


__all__ = ["TinyLLaVA"]
