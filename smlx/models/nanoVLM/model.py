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

import logging
import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import diagnostics for debugging
from smlx.utils.vlm_diagnostics import (
    log_logits_distribution,
    log_vision_features,
)

from .config import NanoVLMConfig
from .projection import MLPProjection
from .vision import VisionModel

logger = logging.getLogger(__name__)

# Enable debug logging if SMLX_DEBUG is set
DEBUG = os.getenv("SMLX_DEBUG", "0") == "1"
if DEBUG:
    logger.setLevel(logging.DEBUG)

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

        if DEBUG:
            log_vision_features(vision_features, label="vision_encoder_output")

        # Project to language space
        # Input: (1, 196, 768)
        # Output: (1, 196, 576)
        projected_features = self.projection(vision_features)

        if DEBUG:
            log_vision_features(projected_features, label="projected_vision_features")

        # No additional normalization - let the learned projector handle scaling
        # Dtype will be cast when concatenating with text embeddings

        return projected_features

    def _prepare_inputs_for_multimodal(
        self,
        vision_embeds: mx.array,
        text_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        """
        Prepare multimodal inputs by replacing image tokens with vision embeddings.

        Follows PaliGemma/mlx-vlm pattern:
        1. Create masks to identify image tokens vs text tokens
        2. Insert vision embeddings where image tokens are
        3. Insert text embeddings where text tokens are

        Args:
            vision_embeds: Vision embeddings from encoder
                Shape: (batch_size, num_vision_tokens, hidden_size)
            text_embeds: Text embeddings from language model
                Shape: (batch_size, sequence_length, hidden_size)
            input_ids: Input token IDs (contains image_token_id markers)
                Shape: (batch_size, sequence_length)

        Returns:
            Combined embeddings with image tokens replaced by vision features
                Shape: (batch_size, sequence_length, hidden_size)
        """
        batch_size, sequence_length, embed_dim = text_embeds.shape
        num_vision_tokens = vision_embeds.shape[1]

        # No scaling: the trained modality projector already maps vision features
        # into the language embedding space. huggingface/nanoVLM applies NO scaling
        # at merge time — an extra factor (a previous 0.15 hack) corrupts the
        # vision/text balance and produces generic/hallucinated descriptions.
        image_token_id = self.config.image_token_id
        vision_embeds = vision_embeds.astype(text_embeds.dtype)
        ids = np.array(input_ids)

        # Reference merge (updated_token_embd[mask] = image_embd.view(-1, dim)):
        # place the j-th vision embedding at the j-th <image> token position, in
        # order — works wherever the image tokens sit (not just a leading block).
        rows = []
        for b in range(batch_size):
            row = text_embeds[b]  # (seq, dim) — text positions keep their embeddings
            positions = np.where(ids[b] == image_token_id)[0].tolist()
            if positions:
                if len(positions) != num_vision_tokens:
                    raise ValueError(
                        f"nanoVLM merge: {len(positions)} <image> token positions != "
                        f"{num_vision_tokens} vision embeddings. prepare_inputs must insert "
                        f"exactly num_vision_tokens (={num_vision_tokens}) image tokens."
                    )
                idx = mx.broadcast_to(
                    mx.array(positions).reshape(-1, 1), (len(positions), embed_dim)
                )
                row = mx.put_along_axis(row, idx, vision_embeds[b], axis=0)
            rows.append(row[None])

        return mx.concatenate(rows, axis=0)

    def create_attention_mask(self, input_ids: mx.array, has_image: bool = True) -> mx.array:
        """
        Create the multimodal attention mask for a vision+text sequence.

        Text tokens use causal attention (attend only to earlier positions). Image
        tokens additionally attend to every other image token bidirectionally,
        since image patches have no left-to-right ordering. Image-token positions
        are located from ``input_ids`` (image_token_id), so they may appear
        anywhere in the sequence (e.g. text + image + text) — they are NOT assumed
        to be a contiguous prefix.

        Args:
            input_ids: Token IDs [batch, seq_len]; image_token_id marks image patches.
            has_image: Whether image tokens are present. When False, a plain causal
                mask is returned.

        Returns:
            Additive attention mask [batch, 1, seq_len, seq_len] with 0.0 where a
            query may attend and -inf where it may not.
        """
        batch_size, seq_len = input_ids.shape

        # Causal base: query position i may attend key position j iff j <= i.
        idx = mx.arange(seq_len)
        attend = idx[:, None] >= idx[None, :]  # (seq_len, seq_len) bool
        attend = mx.broadcast_to(attend[None], (batch_size, seq_len, seq_len))

        if has_image:
            # Allow image<->image bidirectional attention within each sequence.
            is_image = input_ids == self.config.image_token_id  # (batch, seq_len)
            image_pair = is_image[:, :, None] & is_image[:, None, :]
            attend = attend | image_pair

        # Convert the boolean "may attend" matrix into an additive mask.
        additive = mx.where(
            attend,
            mx.array(0.0, dtype=mx.float32),
            mx.array(-float("inf"), dtype=mx.float32),
        )
        return additive[:, None, :, :]  # (batch, 1, seq_len, seq_len)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_token_mask: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        """
        Forward pass of nanoVLM.

        Args:
            input_ids: Text token IDs (contains image_token_id markers for vision)
                Shape: (batch_size, sequence_length)
            pixel_values: Image tensor (optional)
                Shape: (batch_size, channels, height, width)
            image_token_mask: Deprecated - not used with new image token replacement
            mask: Attention mask (optional)
                Shape: (batch_size, total_seq_len, total_seq_len)
                Causal mask for multimodal sequence
            cache: KV cache for generation

        Returns:
            Logits for next token prediction
                Shape: (batch_size, sequence_length, vocab_size)

        Note:
            Follows mlx-vlm pattern:
            - input_ids contains image_token_id (49150) markers where images should be
            - Model replaces these markers with actual vision embeddings
            - Uses _prepare_inputs_for_multimodal() for replacement logic
        """
        # Get text embeddings for all tokens (including image token placeholders)
        # Shape: (batch_size, seq_len, hidden_size)
        text_embeds = self.language_model.model.embed_tokens(input_ids)

        # If image is provided, replace image tokens with vision features
        if pixel_values is not None:
            # Encode image
            # Shape: (batch_size, num_patches, hidden_size)
            # After pixel shuffle: (batch_size, 49, 576)
            vision_embeds = self.encode_image(pixel_values)

            # Cast vision features to match text embedding dtype (critical for quantized models)
            vision_embeds = vision_embeds.astype(text_embeds.dtype)

            # Replace image tokens with vision embeddings
            # This handles scaling internally
            inputs_embeds = self._prepare_inputs_for_multimodal(
                vision_embeds, text_embeds, input_ids
            )
        else:
            # Text-only mode
            inputs_embeds = text_embeds

        # Build the attention mask if the caller did not supply one. This forward
        # drives the language-model layers directly, bypassing the backbone's own
        # causal-mask creation — so without this the attention would be fully
        # bidirectional (a correctness bug: text tokens would attend to future
        # tokens). Text is masked causally; image-patch tokens additionally
        # attend to each other bidirectionally (patches have no left-to-right
        # order). Generation re-runs the full sequence each step (no KV cache),
        # so a full (N, N) mask is always appropriate here.
        if mask is None and inputs_embeds.shape[1] > 1:
            mask = self.create_attention_mask(input_ids, has_image=pixel_values is not None)

        # Pass through language model
        # Note: We need to call the model's forward pass directly with embeddings
        # This requires accessing the transformer layers
        hidden_states = inputs_embeds

        # Apply transformer layers with attention mask
        for i, layer in enumerate(self.language_model.model.layers):
            if cache is not None:
                hidden_states = layer(hidden_states, mask=mask, cache=cache[i])
            else:
                hidden_states = layer(hidden_states, mask=mask)

        # Apply final norm
        hidden_states = self.language_model.model.norm(hidden_states)

        # Get logits (handle both tied and untied embeddings)
        if self.language_model.args.tie_word_embeddings:
            # Use embedding weights as output projection
            logits = self.language_model.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.language_model.lm_head(hidden_states)

        if DEBUG:
            # Log logits distribution for last token (used for generation)
            log_logits_distribution(logits[:, -1, :], label="output_logits")

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

            # Skip weights that don't exist in model
            # Note: decoder.head.weight is the tied embedding weight, don't skip it

            # 2. Rotary embedding inv_freq (computed on-the-fly in MLX)
            if "rotary_embd.inv_freq" in k or "rotary.inv_freq" in k:
                continue  # Skip - computed dynamically

            # 3. Standalone position embeddings (not under patch_embedding)
            if (
                k.startswith("vision.")
                or k.startswith("vision_encoder.")
                or k.startswith("vision_model.")
            ):
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
                # HF has: patch_embedding.conv.weight/bias and patch_embedding.position_embedding
                # Model expects: embeddings.patch_embedding.weight/bias and embeddings.position_embedding.weight

                # Handle position embedding first (needs to move up one level)
                if "patch_embedding.position_embedding" in new_key:
                    # Move position_embedding up to embeddings level and add .weight suffix
                    new_key = new_key.replace(
                        "patch_embedding.position_embedding", "embeddings.position_embedding.weight"
                    )
                # Handle patch_embedding.conv (remove .conv layer)
                elif ".patch_embedding.conv." in new_key:
                    new_key = new_key.replace(
                        ".patch_embedding.conv.", ".embeddings.patch_embedding."
                    )
                # Handle generic patch_embedding
                elif ".patch_emb." in new_key:
                    new_key = new_key.replace(".patch_emb.", ".embeddings.patch_embedding.")
                elif (
                    ".patch_embedding." in new_key and ".embeddings.patch_embedding." not in new_key
                ):
                    new_key = new_key.replace(".patch_embedding.", ".embeddings.patch_embedding.")

                # Map transformer blocks (handle both blocks.* and layers.*)
                new_key = new_key.replace("blocks.", "encoder.layers.")
                if ".layers." in new_key and "encoder.layers." not in new_key:
                    new_key = new_key.replace(".layers.", ".encoder.layers.")

                # Map attention layers
                # First handle HF-specific naming (qkv_proj, out_proj)
                if ".attn.qkv_proj." in new_key:
                    new_key = new_key.replace(".attn.qkv_proj.", ".self_attn.qkv.")
                elif ".attn.qkv." in new_key:
                    new_key = new_key.replace(".attn.qkv.", ".self_attn.qkv.")

                if ".attn.out_proj." in new_key:
                    new_key = new_key.replace(".attn.out_proj.", ".self_attn.proj.")
                elif ".attn.proj." in new_key:
                    new_key = new_key.replace(".attn.proj.", ".self_attn.proj.")

                # Map layer norms
                new_key = new_key.replace("ln1.", "layer_norm1.")
                new_key = new_key.replace("ln2.", "layer_norm2.")
                new_key = new_key.replace("post_ln.", "post_layernorm.")

                # Map final layer norm (HF uses "layer_norm", we use "post_layernorm")
                # This must come after encoder.layers check to avoid affecting layer_norm1/2
                if ".layer_norm." in new_key and "encoder.layers" not in new_key:
                    new_key = new_key.replace(".layer_norm.", ".post_layernorm.")

                # Map MLP layers
                new_key = new_key.replace("mlp.fc1.", "mlp.fc1.")
                new_key = new_key.replace("mlp.fc2.", "mlp.fc2.")

            # Projection mappings (handle both vision.proj_mlp, proj_mlp, and MP)
            if k.startswith("vision.proj_mlp.") or k.startswith("proj_mlp.") or k.startswith("MP."):
                new_key = new_key.replace("vision.proj_mlp.", "projection.")
                new_key = new_key.replace("proj_mlp.", "projection.")
                new_key = new_key.replace("MP.proj.", "projection.proj.")

            # Language model / decoder mappings
            # Map decoder.* to language_model.model.*
            if k.startswith("decoder."):
                # Special case: decoder.head.weight -> language_model.model.embed_tokens.weight
                # (nanoVLM uses tied embeddings, so head weight is the embedding weight)
                if k == "decoder.head.weight":
                    new_key = "language_model.model.embed_tokens.weight"
                else:
                    # decoder.blocks.* -> language_model.model.layers.*
                    new_key = new_key.replace("decoder.", "language_model.model.")
                    new_key = new_key.replace(
                        "language_model.model.blocks.", "language_model.model.layers."
                    )

                    # Map attention layer naming
                    # decoder uses: .attn.* but SmolLM2 uses .self_attn.*
                    new_key = new_key.replace(".attn.", ".self_attn.")

                    # Map attention projection naming
                    # HF uses out_proj but SmolLM2 uses o_proj
                    new_key = new_key.replace(".self_attn.out_proj.", ".self_attn.o_proj.")

                    # Map norm naming
                    new_key = new_key.replace(".norm1.", ".input_layernorm.")
                    new_key = new_key.replace(".norm2.", ".post_attention_layernorm.")

                    # MLP and projection naming should match (gate_proj, up_proj, down_proj)

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
