#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
nanoVLM Main Model.

Combines SigLIP vision encoder, MLP projection, and SmolLM2 language model
for a minimal 222M parameter vision-language model.

Architecture:
    Image (224x224)
      → SigLIP Vision Encoder (85M params)
      → MLP Projection (~2M params)
      → SmolLM2 Language Model (135M params)
      → Text Output
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import NanoVLMConfig
from .projection import MLPProjection
from .vision import VisionModel

# Import SmolLM2 components
try:
    from ..SmolLM2_135M.model import Model as SmolLM2Model
except ImportError:
    # Fallback if SmolLM2 not available
    SmolLM2Model = None


class NanoVLM(nn.Module):
    """
    nanoVLM: Minimal Vision-Language Model.

    A lightweight 222M parameter multimodal model combining:
    - SigLIP-base vision encoder (85M)
    - MLP projection layer (~2M)
    - SmolLM2-135M language model (135M)

    Args:
        config: NanoVLMConfig instance

    Example:
        >>> config = NanoVLMConfig()
        >>> model = NanoVLM(config)
        >>>
        >>> # Process image + text
        >>> image = mx.random.normal((1, 3, 224, 224))
        >>> input_ids = mx.array([[1, 2, 3, 4, 5]])
        >>> output = model(image, input_ids)
    """

    def __init__(self, config: NanoVLMConfig):
        super().__init__()
        self.config = config

        # Vision encoder (SigLIP-base)
        self.vision_model = VisionModel(config.vision_config)

        # Vision-to-language projection
        self.projection = MLPProjection(config.projection_config)

        # Language model (SmolLM2-135M)
        if SmolLM2Model is None:
            raise ImportError(
                "SmolLM2_135M model not found. "
                "Please ensure smlx.models.SmolLM2_135M is available."
            )
        self.language_model = SmolLM2Model(config.language_config)

        # Cache for generation
        self._kv_cache = None

    def encode_image(self, pixel_values: mx.array) -> mx.array:
        """
        Encode image to vision features.

        Args:
            pixel_values: Image tensor
                Shape: (batch_size, channels, height, width)
                Example: (1, 3, 224, 224)

        Returns:
            Projected vision features in language space
                Shape: (batch_size, num_patches, language_hidden_size)
                Example: (1, 196, 576)
        """
        # Vision encoder
        # Input: (1, 3, 224, 224)
        # Output: (1, 196, 768)
        vision_features = self.vision_model(pixel_values)

        # Project to language space
        # Input: (1, 196, 768)
        # Output: (1, 196, 576)
        projected_features = self.projection(vision_features)

        return projected_features

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_token_mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        """
        Forward pass of nanoVLM.

        Args:
            input_ids: Text token IDs
                Shape: (batch_size, sequence_length)
            pixel_values: Image tensor (optional)
                Shape: (batch_size, channels, height, width)
            image_token_mask: Mask indicating image token positions
                Shape: (batch_size, sequence_length)
                Values: 1 for image tokens, 0 for text tokens
            cache: KV cache for generation

        Returns:
            Logits for next token prediction
                Shape: (batch_size, sequence_length, vocab_size)

        Note:
            When pixel_values is provided, image features are injected at
            positions indicated by image_token_mask (where value = 1).
        """
        # Get text embeddings from language model
        # Shape: (batch_size, seq_len, hidden_size)
        text_embeds = self.language_model.model.embed_tokens(input_ids)

        # If image is provided, inject vision features
        if pixel_values is not None and image_token_mask is not None:
            # Encode image
            # Shape: (batch_size, num_patches, hidden_size)
            vision_embeds = self.encode_image(pixel_values)

            # Inject vision embeddings at image token positions
            # This replaces text embeddings at positions where mask = 1
            batch_size, seq_len, hidden_size = text_embeds.shape
            num_patches = vision_embeds.shape[1]

            # Create combined embeddings
            for b in range(batch_size):
                # Find image token positions using boolean mask
                mask_b = image_token_mask[b] == 1
                # Count how many image tokens
                num_image_tokens = int(mx.sum(mask_b))

                if num_image_tokens > 0:
                    # Find first image token position
                    # Convert to numpy to use argmax for finding first True
                    import numpy as np
                    mask_np = np.array(mask_b)
                    start_pos = int(np.argmax(mask_np))  # First position where mask is True

                    # Replace text embeddings with vision embeddings
                    # Note: This assumes image tokens are contiguous
                    end_pos = start_pos + num_patches

                    # Ensure we don't exceed sequence length
                    if end_pos <= seq_len:
                        text_embeds[b, start_pos:end_pos] = vision_embeds[b]

            # Use combined embeddings
            inputs_embeds = text_embeds
        else:
            # Text-only mode
            inputs_embeds = text_embeds

        # Pass through language model
        # Note: We need to call the model's forward pass directly with embeddings
        # This requires accessing the transformer layers
        hidden_states = inputs_embeds

        # Apply transformer layers
        for i, layer in enumerate(self.language_model.model.layers):
            if cache is not None:
                hidden_states = layer(hidden_states, cache=cache[i])
            else:
                hidden_states = layer(hidden_states)

        # Apply final norm
        hidden_states = self.language_model.model.norm(hidden_states)

        # Get logits (handle both tied and untied embeddings)
        if self.language_model.args.tie_word_embeddings:
            # Use embedding weights as output projection
            logits = self.language_model.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.language_model.lm_head(hidden_states)

        return logits

    def reset_cache(self):
        """Reset KV cache for generation."""
        self._kv_cache = None

    def get_cache(self):
        """Get current KV cache."""
        return self._kv_cache

    def set_cache(self, cache):
        """Set KV cache."""
        self._kv_cache = cache

    @staticmethod
    def sanitize(weights):
        """
        Sanitize HuggingFace weights to match model parameter names.

        Maps from HuggingFace nanoVLM naming to our naming convention:
        - vision.* or vision_encoder.* -> vision_model.*
        - patch_emb or patch_embedding -> embeddings.patch_embedding
        - pos_emb or position_embedding -> embeddings.position_embedding
        - blocks.* or layers.* -> encoder.layers.*
        - proj_mlp -> projection.*
        """
        sanitized_weights = {}

        for k, v in weights.items():
            new_key = k

            # Skip position embeddings from vision model (nanoVLM doesn't use them in weights)
            if (k.startswith("vision.") or k.startswith("vision_encoder.") or k.startswith("vision_model.")):
                if ("position_embedding" in k or "pos_emb" in k) and "patch_embedding" not in k:
                    continue  # Skip - will be randomly initialized

            # Vision model mappings (handle both vision. and vision_encoder. prefixes)
            if k.startswith("vision.") or k.startswith("vision_encoder."):
                # Normalize to vision_model. prefix
                if k.startswith("vision_encoder."):
                    new_key = k.replace("vision_encoder.", "vision_model.", 1)
                elif k.startswith("vision."):
                    new_key = k.replace("vision.", "vision_model.", 1)

                # Map patch embedding (handle both patch_emb and patch_embedding)
                # Handle patch_embedding.position_embedding -> should NOT be changed to embeddings.patch_embedding.position_embedding
                # The structure should be: embeddings.patch_embedding (Conv2d) and embeddings.position_embedding (Embedding)
                # So patch_embedding.position_embedding is incorrect and should be fixed first
                if "patch_embedding.position_embedding" in new_key:
                    # This is wrong - position_embedding should be at same level as patch_embedding
                    new_key = new_key.replace("patch_embedding.position_embedding", "embeddings.position_embedding")
                elif "embeddings.patch_embedding" not in new_key:
                    # Only add "embeddings." if not already present
                    new_key = new_key.replace("patch_emb.", "embeddings.patch_embedding.")
                    if "position_embedding" not in new_key:  # Don't replace if it contains position_embedding
                        new_key = new_key.replace("patch_embedding.", "embeddings.patch_embedding.")

                # Handle .embedding -> .weight for Conv2d patch embedding
                new_key = new_key.replace("patch_embedding.embedding", "patch_embedding.weight")

                # Map transformer blocks (handle both blocks.* and layers.*)
                new_key = new_key.replace("blocks.", "encoder.layers.")
                if ".layers." in new_key and "encoder.layers." not in new_key:
                    new_key = new_key.replace(".layers.", ".encoder.layers.")

                # Map attention layers
                new_key = new_key.replace("attn.qkv.", "self_attn.qkv.")
                new_key = new_key.replace("attn.proj.", "self_attn.proj.")

                # Map layer norms
                new_key = new_key.replace("ln1.", "layer_norm1.")
                new_key = new_key.replace("ln2.", "layer_norm2.")
                new_key = new_key.replace("post_ln.", "post_layernorm.")

                # Map MLP layers
                new_key = new_key.replace("mlp.fc1.", "mlp.fc1.")
                new_key = new_key.replace("mlp.fc2.", "mlp.fc2.")

            # Projection mappings (handle both vision.proj_mlp and proj_mlp)
            if k.startswith("vision.proj_mlp.") or k.startswith("proj_mlp."):
                new_key = new_key.replace("vision.proj_mlp.", "projection.")
                new_key = new_key.replace("proj_mlp.", "projection.")

            sanitized_weights[new_key] = v

        return sanitized_weights


def create_model(config: NanoVLMConfig) -> NanoVLM:
    """
    Create nanoVLM model.

    Args:
        config: NanoVLMConfig instance

    Returns:
        NanoVLM model

    Example:
        >>> from smlx.models.nanoVLM.config import DEFAULT_CONFIG
        >>> model = create_model(DEFAULT_CONFIG)
        >>> print(f"Parameters: {sum(p.size for p in model.parameters())}M")
    """
    model = NanoVLM(config)
    model.eval()  # Set to evaluation mode by default
    return model
