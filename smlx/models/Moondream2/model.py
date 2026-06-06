#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Moondream2 Vision-Language Model.

Combines a vision encoder, Phi language model, and region modules
for captioning, VQA, object detection, and pointing tasks.
"""

import logging
import os
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .vision import VisionEncoder, prepare_crops, reconstruct_from_crops
from .language import PhiModel
from .region import DetectionHead, CoordinateDecoder
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


class VisionProjector(nn.Module):
    """Projects vision features to language model dimension.

    Uses a 2-layer MLP to map vision features to text embedding space.
    This matches the HuggingFace Moondream2 architecture (vision.proj_mlp).

    The HF model concatenates features from multiple vision layers/sources,
    resulting in input dimension of 2*vision_hidden_size.
    """

    def __init__(self, vision_hidden_size: int, text_hidden_size: int, text_intermediate_size: int = None):
        super().__init__()
        # The HF model concatenates features, so input is 2*vision_hidden_size
        # fc1: Linear(2*vision_hidden, intermediate) = Linear(2304, 8192)
        # fc2: Linear(intermediate, text_hidden) = Linear(8192, 2048)
        projector_input_size = 2 * vision_hidden_size
        intermediate_size = text_intermediate_size or (text_hidden_size * 4)

        self.fc1 = nn.Linear(projector_input_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, text_hidden_size)
        self.activation = nn.GELU()

    def __call__(self, vision_features: mx.array) -> mx.array:
        x = self.fc1(vision_features)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class Moondream2(nn.Module):
    """Moondream2 multimodal model.

    Architecture:
    - Vision Encoder: Custom encoder with crop-based tiling
    - Vision Projector: Maps vision features to text space
    - Language Model: Phi transformer for text generation
    - Region Modules: For object detection and pointing
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(config.vision_config)

        # Vision-to-text projection
        self.vision_projection = VisionProjector(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            config.text_config.intermediate_size,
        )

        # Language model
        self.language_model = PhiModel(config.text_config)

        # Language model head
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

        # Region modules (optional, for detection/pointing)
        self.detection_head = DetectionHead(
            config.text_config.hidden_size,
            config.region_config.max_detections,
        )
        self.point_decoder = CoordinateDecoder(config.text_config.hidden_size)

    # Attribute aliases for API compatibility
    @property
    def vision(self):
        """Alias for vision_encoder for API compatibility."""
        return self.vision_encoder

    @property
    def language(self):
        """Alias for language_model for API compatibility."""
        return self.language_model

    @property
    def vision_proj(self):
        """Alias for vision_projection for API compatibility."""
        return self.vision_projection

    def encode_image(
        self,
        pixel_values: mx.array,
        use_tiling: bool = True,
    ) -> mx.array:
        """Encode image into visual embeddings.

        Args:
            pixel_values: Image tensor [B, C, H, W]
            use_tiling: Whether to use crop-based tiling

        Returns:
            Visual embeddings [B, num_patches, text_hidden_size]
        """
        if use_tiling and self.config.vision_config.use_tiling:
            # Prepare crops
            B = pixel_values.shape[0]
            all_vision_features = []

            for i in range(B):
                crops, crop_coords = prepare_crops(
                    pixel_values[i],
                    self.config.vision_config.image_size,
                    self.config.vision_config.max_crops,
                )

                # Encode each crop
                crop_features = []
                for crop in crops:
                    # Add batch dimension
                    crop_input = crop[None, :, :, :]

                    # Extract multi-layer features
                    final_output, all_hidden_states = self.vision_encoder(
                        crop_input, output_hidden_states=True
                    )

                    # Remove batch dim: [1, N, hidden_size] -> [N, hidden_size]
                    final_output = final_output[0]

                    # Extract features from specified layers for multi-scale representation
                    # vision_feature_layers: [3, 7, 15, 23, 27]
                    selected_features = []
                    for layer_idx in self.config.vision_feature_layers:
                        if layer_idx < len(all_hidden_states):
                            # Remove batch dim: [1, N, hidden_size] -> [N, hidden_size]
                            layer_features = all_hidden_states[layer_idx][0]
                            selected_features.append(layer_features)

                    # Global features: average pool across selected layers
                    # Shape: [num_selected_layers, N, hidden_size] -> [N, hidden_size]
                    if selected_features:
                        global_features = mx.stack(selected_features, axis=0).mean(axis=0)
                    else:
                        # Fallback: use final output as global
                        global_features = final_output

                    # Local features: use final layer output
                    local_features = final_output

                    # Concatenate global + local: [N, hidden_size] + [N, hidden_size] -> [N, 2*hidden_size]
                    concat_features = mx.concatenate([global_features, local_features], axis=-1)
                    crop_features.append(concat_features)

                # Reconstruct from crops
                vision_features = reconstruct_from_crops(
                    crop_features,
                    crop_coords,
                    (pixel_values.shape[2], pixel_values.shape[3]),
                    self.config.vision_config.crop_overlap,
                )

                all_vision_features.append(vision_features)

            # Stack batch
            vision_features = mx.stack(all_vision_features, axis=0)

        else:
            # Simple encoding without tiling
            final_output, all_hidden_states = self.vision_encoder(
                pixel_values, output_hidden_states=True
            )

            # Extract features from specified layers for multi-scale representation
            selected_features = []
            for layer_idx in self.config.vision_feature_layers:
                if layer_idx < len(all_hidden_states):
                    layer_features = all_hidden_states[layer_idx]
                    selected_features.append(layer_features)

            # Global features: average pool across selected layers
            # Shape: [num_selected_layers, B, N, hidden_size] -> [B, N, hidden_size]
            if selected_features:
                global_features = mx.stack(selected_features, axis=0).mean(axis=0)
            else:
                # Fallback: use final output as global
                global_features = final_output

            # Local features: use final layer output
            local_features = final_output

            # Concatenate global + local: [B, N, hidden_size] + [B, N, hidden_size] -> [B, N, 2*hidden_size]
            vision_features = mx.concatenate([global_features, local_features], axis=-1)

        # Project to text space: [B, N, 2*hidden_size] -> [B, N, text_hidden_size]
        if DEBUG:
            log_vision_features(vision_features, label="moondream2_vision_features_pre_proj")

        vision_embeddings = self.vision_projection(vision_features)

        if DEBUG:
            log_vision_features(vision_embeddings, label="moondream2_projected_vision_embeds")

        return vision_embeddings

    def create_attention_mask(
        self,
        vision_seq_len: int,
        text_seq_len: int,
    ) -> mx.array:
        """Create attention mask for vision+text tokens.

        Following HuggingFace Moondream2 implementation:
        - Vision tokens (positions 0 to vision_seq_len-1) use bidirectional attention
        - Text tokens (positions vision_seq_len onwards) use causal attention
        - Text tokens can attend to all vision tokens

        Args:
            vision_seq_len: Number of vision tokens (729 for Moondream2)
            text_seq_len: Number of text tokens

        Returns:
            Additive attention mask [1, 1, total_len, total_len]
            where 0.0 = allowed, -inf = masked
        """
        total_len = vision_seq_len + text_seq_len

        # Create full causal mask (lower triangular)
        # Start with all positions masked (-inf)
        # Use bfloat16 to match model dtype
        mask = mx.full((total_len, total_len), float('-inf'), dtype=mx.bfloat16)

        # Vision tokens can attend to all vision tokens (bidirectional)
        mask[:vision_seq_len, :vision_seq_len] = 0.0

        # Text tokens can attend to all vision tokens
        mask[vision_seq_len:, :vision_seq_len] = 0.0

        # Text tokens use causal attention (can only attend to previous text tokens)
        for i in range(text_seq_len):
            for j in range(i + 1):
                mask[vision_seq_len + i, vision_seq_len + j] = 0.0

        # Add batch and head dimensions: [total_len, total_len] -> [1, 1, total_len, total_len]
        mask = mask.reshape(1, 1, total_len, total_len)

        return mask

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        vision_embeddings: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        position_ids: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass.

        Args:
            input_ids: Text token IDs [B, seq_len]
            pixel_values: Optional image tensor [B, C, H, W]
            vision_embeddings: Pre-computed vision embeddings [B, num_patches, hidden_size]
            mask: Optional attention mask
            cache: Optional KV cache for generation

        Returns:
            logits: Output logits [B, seq_len, vocab_size]
            vision_embeddings: Vision embeddings if pixel_values provided
        """
        # Encode image if provided
        if pixel_values is not None:
            vision_embeddings = self.encode_image(pixel_values)

        # Get text embeddings
        text_embeddings = self.language_model.embed_tokens(input_ids)

        # Combine vision and text embeddings
        if vision_embeddings is not None:
            # Prepend vision embeddings to text
            # Vision acts as a prefix to the text sequence
            B = text_embeddings.shape[0]

            if DEBUG:
                log_embedding_comparison(
                    vision_embeddings, text_embeddings, label="moondream2_vision_vs_text_embeds"
                )

            combined_embeddings = mx.concatenate(
                [vision_embeddings, text_embeddings], axis=1
            )

            # Create proper attention mask if not provided
            # Vision tokens use bidirectional attention, text tokens use causal
            if mask is None:
                vision_seq_len = vision_embeddings.shape[1]
                text_seq_len = text_embeddings.shape[1]
                mask = self.create_attention_mask(vision_seq_len, text_seq_len)
        else:
            combined_embeddings = text_embeddings
            # For text-only, create standard causal mask if not provided
            if mask is None and cache is None:
                # Only create causal mask for prefill (when cache is None)
                text_seq_len = text_embeddings.shape[1]
                mask = nn.MultiHeadAttention.create_additive_causal_mask(text_seq_len)
                mask = mask.reshape(1, 1, text_seq_len, text_seq_len)
                # Cast to bfloat16 to match model dtype
                mask = mask.astype(mx.bfloat16)

        # Forward through language model layers
        hidden_states = combined_embeddings

        if cache is None:
            cache = [None] * len(self.language_model.layers)

        for layer, layer_cache in zip(self.language_model.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache, position_ids=position_ids)

        # Final layer norm
        hidden_states = self.language_model.final_layernorm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        if DEBUG:
            # Log logits distribution for last token (used for generation)
            log_logits_distribution(logits[:, -1, :], label="moondream2_output_logits")

        return logits, vision_embeddings

    def cache_vision_embeddings(
        self,
        vision_embeddings: mx.array,
        cache: list,
    ) -> None:
        """Cache vision embeddings by running them through the language model.

        This matches the HuggingFace implementation's load_encoded_image() behavior.
        Vision embeddings are run through all transformer layers to populate the KV cache,
        allowing subsequent text-only forward passes to attend to the cached vision.

        Args:
            vision_embeddings: Pre-computed vision embeddings [B, vision_seq_len, hidden_size]
                              (should include BOS embedding prepended)
            cache: KV cache list to populate

        Example:
            >>> # Encode and cache vision
            >>> vision_emb = model.encode_image(pixel_values)
            >>> bos_emb = model.language_model.embed_tokens(mx.array([[0]]))
            >>> vision_with_bos = mx.concatenate([bos_emb, vision_emb], axis=1)
            >>> cache = make_kv_caches(model.config.text_config)
            >>> model.cache_vision_embeddings(vision_with_bos, cache)
            >>> # Now pass text tokens with vision already cached
            >>> logits, _ = model(text_ids, vision_embeddings=None, cache=cache)
        """
        vision_seq_len = vision_embeddings.shape[1]

        # Create bidirectional attention mask for vision tokens only
        # All vision tokens can attend to all other vision tokens
        mask = mx.zeros((1, 1, vision_seq_len, vision_seq_len), dtype=mx.bfloat16)

        # Create position_ids for vision tokens: [0, 1, 2, ..., vision_seq_len-1]
        # This ensures proper RoPE positioning for vision embeddings
        position_ids = mx.arange(vision_seq_len)[None, :]  # [1, vision_seq_len]

        # Run vision embeddings through language model to populate cache
        hidden_states = vision_embeddings

        for layer, layer_cache in zip(self.language_model.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache, position_ids=position_ids)

        # We don't need the output, just the cache population
        # Evaluate both cache and hidden_states to prevent graph accumulation
        mx.eval([hidden_states] + [c.keys for c in cache] + [c.values for c in cache])

    def detect_objects(
        self,
        input_ids: mx.array,
        vision_embeddings: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Detect objects in the image.

        Args:
            input_ids: Text token IDs (detection prompt)
            vision_embeddings: Vision embeddings from image
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            boxes: Predicted boxes [B, max_detections, 4]
            confidences: Confidence scores [B, max_detections]
        """
        # Forward pass
        logits, _ = self(
            input_ids,
            vision_embeddings=vision_embeddings,
            mask=mask,
            cache=cache,
        )

        # Get hidden states (we need the pre-logits)
        # Re-run just to get hidden states (inefficient, but simple)
        text_embeddings = self.language_model.embed_tokens(input_ids)
        combined_embeddings = mx.concatenate(
            [vision_embeddings, text_embeddings], axis=1
        )

        hidden_states = combined_embeddings
        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states, mask=mask, cache=None)
        hidden_states = self.language_model.final_layernorm(hidden_states)

        # Apply detection head
        boxes, confidences = self.detection_head(hidden_states)

        return boxes, confidences

    def point_to_object(
        self,
        input_ids: mx.array,
        vision_embeddings: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        """Point to an object in the image.

        Args:
            input_ids: Text token IDs (pointing query)
            vision_embeddings: Vision embeddings from image
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            coordinates: Predicted (x, y) coordinates [B, 2] normalized to [0, 1]
        """
        # Forward pass
        logits, _ = self(
            input_ids,
            vision_embeddings=vision_embeddings,
            mask=mask,
            cache=cache,
        )

        # Get hidden states
        text_embeddings = self.language_model.embed_tokens(input_ids)
        combined_embeddings = mx.concatenate(
            [vision_embeddings, text_embeddings], axis=1
        )

        hidden_states = combined_embeddings
        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states, mask=mask, cache=None)
        hidden_states = self.language_model.final_layernorm(hidden_states)

        # Use last token's hidden state for coordinate prediction
        last_hidden = hidden_states[:, -1, :]

        # Decode coordinates
        coordinates = self.point_decoder(last_hidden)

        return coordinates

    @staticmethod
    def sanitize(weights):
        """Sanitize weights from HuggingFace format to MLX format.

        Maps HuggingFace Moondream2 weight keys to SMLX's expected format.

        Key transformations:
        Vision:
        - vision.blocks.X → vision_encoder.layers.X
        - vision.attn → vision_encoder.attention
        - vision.ln1 → vision_encoder.layer_norm1
        - vision.ln2 → vision_encoder.layer_norm2
        - vision.post_ln → vision_encoder.post_layernorm
        - vision.patch_emb → vision_encoder.embeddings.patch_embedding
        - vision.pos_emb → vision_encoder.embeddings.position_embeddings
        - vision.proj_mlp → vision_projection

        Text/Language:
        - text.blocks.X → language_model.layers.X
        - text.blocks.X.attn.qkv → split into q_proj, k_proj, v_proj
        - text.blocks.X.attn.proj → self_attn.o_proj
        - text.blocks.X.ln → input_layernorm (and duplicated to post_attention_layernorm)
        - text.post_ln → language_model.final_layernorm
        - text.wte → language_model.embed_tokens.weight
        - text.lm_head → lm_head

        Region:
        - region.coord_decoder → point_decoder
        - region.size_decoder → detection_head.size_decoder
        """
        print(f"[SANITIZE] Starting weight sanitization for {len(weights)} weights")
        sanitized_weights = {}
        qkv_weights_to_split = {}  # Store QKV weights to split later
        skipped_weights = []
        transformed_count = 0

        for k, v in weights.items():
            original_key = k

            # Remove "model." prefix if present
            if k.startswith("model."):
                k = k.replace("model.", "")

            # Vision encoder transformations
            if k.startswith("vision."):
                # Handle specific vision components
                if k.startswith("vision.blocks."):
                    k = k.replace("vision.blocks.", "vision_encoder.layers.")
                elif k.startswith("vision.patch_emb."):
                    k = k.replace("vision.patch_emb.", "vision_encoder.embeddings.patch_embedding.")
                elif k == "vision.pos_emb":
                    # Note: position_embeddings (with 's') to match the VisionEmbeddings attribute
                    k = "vision_encoder.embeddings.position_embeddings"
                elif k.startswith("vision.post_ln."):
                    k = k.replace("vision.post_ln.", "vision_encoder.post_layernorm.")
                elif k.startswith("vision.proj_mlp."):
                    # vision.proj_mlp.fc1 → vision_projection.fc1
                    k = k.replace("vision.proj_mlp.", "vision_projection.")

            # Map vision encoder layer component names
            if "vision_encoder.layers." in k:
                # Map attention module names (vision uses combined QKV, keep it)
                k = k.replace(".attn.qkv.", ".attention.qkv.")
                k = k.replace(".attn.proj.", ".attention.proj.")
                # Map layer norm names
                k = k.replace(".ln1.", ".layer_norm1.")
                k = k.replace(".ln2.", ".layer_norm2.")
                # MLP stays the same

            # Language model transformations
            if k.startswith("text."):
                # Handle text top-level components
                if k == "text.wte":
                    k = "language_model.embed_tokens.weight"
                elif k.startswith("text.lm_head."):
                    k = k.replace("text.lm_head.", "lm_head.")
                elif k.startswith("text.post_ln."):
                    k = k.replace("text.post_ln.", "language_model.final_layernorm.")
                elif k.startswith("text.blocks."):
                    k = k.replace("text.blocks.", "language_model.layers.")

            # Handle language model QKV splitting
            if "language_model.layers." in k and ".attn.qkv." in k:
                # Store for splitting later (need both weight and bias)
                base_key = k.replace(".attn.qkv.", ".attn.qkv_")
                qkv_weights_to_split[base_key] = (k, v)
                if original_key != k:
                    transformed_count += 1
                continue  # Don't add to sanitized_weights yet

            # Map language model layer component names
            if "language_model.layers." in k:
                # Map attention output projection
                k = k.replace(".attn.proj.", ".self_attn.o_proj.")
                # Map layer norm - Phi has separate input and post-attention norms
                # HF has single ln, so we duplicate it
                if ".ln." in k:
                    input_key = k.replace(".ln.", ".input_layernorm.")
                    post_key = k.replace(".ln.", ".post_attention_layernorm.")
                    sanitized_weights[input_key] = v
                    sanitized_weights[post_key] = v
                    continue  # Already added, skip the normal flow

            # Region module transformations
            if k.startswith("region."):
                # Map region components to detection/pointing modules
                if k.startswith("region.coord_decoder."):
                    # Map to point_decoder (CoordinateDecoder)
                    # coord_decoder.fc1/fc2 structure matches CoordinateDecoder
                    k = k.replace("region.coord_decoder.fc1.", "point_decoder.fc1.")
                    k = k.replace("region.coord_decoder.fc2.", "point_decoder.fc2.")
                elif k.startswith("region.coord_encoder."):
                    # Skip coord_encoder for now (not in CoordinateDecoder)
                    skipped_weights.append((original_key, "region.coord_encoder not in model"))
                    continue
                elif k == "region.coord_features":
                    # Skip coord_features (not in simple CoordinateDecoder)
                    skipped_weights.append((original_key, "region.coord_features not in model"))
                    continue
                elif k.startswith("region.size_"):
                    # Skip size_decoder, size_encoder, size_features
                    # The HF model's region structure doesn't match SMLX DetectionHead
                    # These weights are for a different region architecture
                    skipped_weights.append((original_key, "region.size_* not in model"))
                    continue

            # Skip lm_head bias (model defined with bias=False)
            if k == "lm_head.bias":
                skipped_weights.append((original_key, "lm_head.bias not needed (bias=False)"))
                continue

            # Patch embedding weight handling
            # Moondream2 uses Linear projection (NOT Conv2d)
            # HF format: [out_features, in_features] = [1152, 588]
            # MLX Linear expects same format - NO transpose needed
            if "patch_embedding.weight" in k:
                # Expected shape: [hidden_size, patch_dim] = [1152, 588]
                # where patch_dim = patch_size^2 * num_channels = 14*14*3 = 588
                if len(v.shape) == 2:
                    # Linear weight - keep as-is
                    pass
                elif len(v.shape) == 4:
                    # If somehow still Conv2d format, warn and skip
                    print(f"[SANITIZE] WARNING: patch_embedding has Conv2d shape {v.shape}, expected Linear [1152, 588]")
                    print("[SANITIZE] Skipping this weight - may cause issues")
                    skipped_weights.append((original_key, f"unexpected shape {v.shape}"))
                    continue

            if original_key != k:
                transformed_count += 1
            sanitized_weights[k] = v

        # Split QKV weights for language model (Phi uses separate Q/K/V projections)
        # Group by layer and weight type
        qkv_groups = {}
        for base_key, (full_key, tensor) in qkv_weights_to_split.items():
            # Extract layer number and weight/bias
            # base_key format: language_model.layers.X.attn.qkv_weight or qkv_bias
            parts = base_key.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_idx = part
                    break

            if layer_idx is None:
                continue

            is_bias = base_key.endswith("_bias")
            is_weight = base_key.endswith("_weight")

            group_key = f"layer_{layer_idx}"
            if group_key not in qkv_groups:
                qkv_groups[group_key] = {}

            if is_weight:
                qkv_groups[group_key]["weight"] = (base_key, tensor)
            elif is_bias:
                qkv_groups[group_key]["bias"] = (base_key, tensor)

        # Now split each group's QKV into separate Q, K, V
        for group_key, group_data in qkv_groups.items():
            layer_idx = group_key.split("_")[1]

            if "weight" in group_data:
                base_key, qkv_weight = group_data["weight"]
                # Split along dimension 0: [3*hidden, hidden] → 3x [hidden, hidden]
                # Convert to MLX array first if needed
                if not isinstance(qkv_weight, mx.array):
                    import numpy as np
                    # Convert via numpy to avoid buffer format issues
                    if hasattr(qkv_weight, "numpy"):
                        qkv_weight = mx.array(qkv_weight.numpy())
                    elif isinstance(qkv_weight, np.ndarray):
                        qkv_weight = mx.array(qkv_weight.astype(np.float32))
                    else:
                        qkv_weight = mx.array(np.array(qkv_weight, dtype=np.float32))

                hidden_size = qkv_weight.shape[0] // 3
                q_weight = qkv_weight[:hidden_size, :]
                k_weight = qkv_weight[hidden_size:2*hidden_size, :]
                v_weight = qkv_weight[2*hidden_size:, :]

                sanitized_weights[f"language_model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_weight
                sanitized_weights[f"language_model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_weight
                sanitized_weights[f"language_model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_weight

            if "bias" in group_data:
                base_key, qkv_bias = group_data["bias"]
                # Split along dimension 0: [3*hidden] → 3x [hidden]
                # Convert to MLX array first if needed
                if not isinstance(qkv_bias, mx.array):
                    import numpy as np
                    # Convert via numpy to avoid buffer format issues
                    if hasattr(qkv_bias, "numpy"):
                        qkv_bias = mx.array(qkv_bias.numpy())
                    elif isinstance(qkv_bias, np.ndarray):
                        qkv_bias = mx.array(qkv_bias.astype(np.float32))
                    else:
                        qkv_bias = mx.array(np.array(qkv_bias, dtype=np.float32))

                hidden_size = qkv_bias.shape[0] // 3
                q_bias = qkv_bias[:hidden_size]
                k_bias = qkv_bias[hidden_size:2*hidden_size]
                v_bias = qkv_bias[2*hidden_size:]

                sanitized_weights[f"language_model.layers.{layer_idx}.self_attn.q_proj.bias"] = q_bias
                sanitized_weights[f"language_model.layers.{layer_idx}.self_attn.k_proj.bias"] = k_bias
                sanitized_weights[f"language_model.layers.{layer_idx}.self_attn.v_proj.bias"] = v_bias

        # Print summary
        qkv_split_count = len(qkv_groups) * 3  # Each QKV becomes 3 weights (Q, K, V)
        print(f"[SANITIZE] Summary:")
        print(f"[SANITIZE]   Input weights: {len(weights)}")
        print(f"[SANITIZE]   Output weights: {len(sanitized_weights)}")
        print(f"[SANITIZE]   Transformed: {transformed_count}")
        print(f"[SANITIZE]   QKV splits: {len(qkv_groups)} groups → {qkv_split_count} weights")
        print(f"[SANITIZE]   Skipped: {len(skipped_weights)}")

        if skipped_weights:
            print(f"[SANITIZE] Skipped weights ({len(skipped_weights)} total):")
            for i, (key, reason) in enumerate(skipped_weights):
                if i < 10:
                    print(f"[SANITIZE]   - {key}: {reason}")
                elif i == 10:
                    print(f"[SANITIZE]   ... ({len(skipped_weights) - 10} more skipped)")
                    break

        return sanitized_weights


__all__ = ["Moondream2", "VisionProjector"]
