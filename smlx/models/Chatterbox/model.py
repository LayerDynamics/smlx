#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Chatterbox TTS model architecture.

Built on 0.5B Llama backbone with voice cloning and expressiveness control.

NOTE: This is a reference implementation showing the API structure.
For production use, search for voice cloning TTS models on HuggingFace.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import ChatterboxConfig
from .vocoder import HiFiGANGenerator, HiFiGANConfig


# ============================================================================
# Utilities (from SmolLM2_135M)
# ============================================================================


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: int | None = None,
):
    """Create a causal attention mask."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def create_attention_mask(h, cache=None, window_size: int | None = None):
    """Create attention mask for the given input."""
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, window_size=window_size)
    if N == 1:
        return None
    if window_size and N > window_size:
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: mx.array | str | None,
) -> mx.array:
    """Scaled dot-product attention with optional KV cache quantization."""
    # Use MLX's fast scaled_dot_product_attention
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )


# ============================================================================
# RoPE (Rotary Position Embedding) Utilities
# ============================================================================


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: dict | None = None,
    max_position_embeddings: int | None = None,
):
    """
    Initialize RoPE (Rotary Position Embedding) with optional scaling.

    Args:
        dims: Dimension of the embeddings
        base: Base for the exponential scaling
        traditional: Whether to use traditional RoPE
        scaling_config: Optional scaling configuration
        max_position_embeddings: Maximum sequence length

    Returns:
        RoPE module
    """
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    if rope_type in ["default", "linear"]:
        scale = (
            1 / scaling_config["factor"]
            if rope_type == "linear" and scaling_config is not None
            else 1.0
        )
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)
    else:
        # For now, we only support default and linear scaling
        raise ValueError(f"Unsupported RoPE type {rope_type}")


# ============================================================================
# Llama Components (from SmolLM2_135M)
# ============================================================================


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) support.

    Uses RoPE for positional encoding and supports KV caching for efficient generation.
    """

    def __init__(self, config):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = (
            config.num_key_value_heads
            if hasattr(config, "num_key_value_heads") and config.num_key_value_heads is not None
            else config.num_attention_heads
        )

        self.head_dim = head_dim = (
            config.head_dim
            if hasattr(config, "head_dim") and config.head_dim is not None
            else config.hidden_size // n_heads
        )
        self.scale = head_dim**-0.5

        attention_bias = config.attention_bias if hasattr(config, "attention_bias") else False

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        # Rotary Position Embedding
        rope_theta = config.rope_theta if hasattr(config, "rope_theta") else 10000.0
        rope_traditional = config.rope_traditional if hasattr(config, "rope_traditional") else False
        rope_scaling = config.rope_scaling if hasattr(config, "rope_scaling") else None
        max_pos = (
            config.max_position_embeddings if hasattr(config, "max_position_embeddings") else None
        )

        self.rope: nn.RoPE | NoPE = initialize_rope(
            self.head_dim,
            rope_theta,
            rope_traditional,
            rope_scaling,
            max_pos,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Project to Q, K, V
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Scaled dot-product attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """
    MLP block with SiLU activation (SwiGLU-style).

    Uses the gated linear unit pattern: down(silu(gate(x)) * up(x))
    """

    def __init__(self, config):
        super().__init__()

        dim = config.hidden_size
        hidden_dim = config.intermediate_size
        mlp_bias = config.mlp_bias if hasattr(config, "mlp_bias") else False

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with pre-normalization.

    Architecture:
        x -> LayerNorm -> Attention -> Add -> LayerNorm -> MLP -> Add
    """

    def __init__(self, config, use_sliding: bool = False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(config)
        self.mlp = MLP(config)

        rms_eps = config.rms_norm_eps if hasattr(config, "rms_norm_eps") else 1e-5
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=rms_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=rms_eps)
        self.config = config

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        # Pre-norm attention with residual
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        # Pre-norm MLP with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class NoPE(nn.Module):
    """
    No-op module used to disable rotary embeddings in selected layers.

    This is a key feature of SmolLM3 architecture - selective RoPE disabling.
    """

    def __call__(self, x, offset: int = 0):
        return x


# ============================================================================
# Llama Backbone for TTS
# ============================================================================


class LlamaBackbone(nn.Module):
    """
    Full Llama backbone for TTS (from SmolLM2_135M architecture).

    Uses proper Llama components:
    - RoPE positional encoding
    - Grouped Query Attention (GQA)
    - SwiGLU MLP
    - RMSNorm
    - Optional NoPE (selective RoPE disabling)

    Args:
        config: Llama backbone configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = [
            TransformerBlock(config, use_sliding=False) for _ in range(config.num_hidden_layers)
        ]

        # Final layer norm
        rms_eps = config.rms_norm_eps if hasattr(config, "rms_norm_eps") else 1e-5
        self.norm = nn.RMSNorm(config.hidden_size, eps=rms_eps)

        # Optional: Disable RoPE for specified layers (NoPE feature)
        if hasattr(config, "no_rope_layers") and config.no_rope_layers is not None:
            for idx, use_rope in enumerate(config.no_rope_layers):
                if not use_rope:
                    self.layers[idx].self_attn.rope = NoPE()

    def __call__(
        self,
        input_ids: mx.array = None,
        input_embeddings: mx.array = None,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        """
        Forward pass through Llama backbone.

        Args:
            input_ids: Token IDs (batch, seq_len) - optional if input_embeddings provided
            input_embeddings: Pre-computed embeddings (batch, seq_len, hidden_size) - optional
            mask: Attention mask
            cache: KV cache for generation

        Returns:
            Hidden states (batch, seq_len, hidden_size)
        """
        # Get embeddings
        if input_embeddings is not None:
            h = input_embeddings
        elif input_ids is not None:
            h = self.embedding(input_ids)
        else:
            raise ValueError("Must provide either input_ids or input_embeddings")

        # Initialize cache if not provided
        if cache is None:
            cache = [None] * len(self.layers)

        # Create attention mask if not provided
        if mask is None:
            mask = create_attention_mask(h, cache[0] if cache else None)

        # Forward through transformer layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        # Final layer norm
        return self.norm(h)


class VoiceEncoder(nn.Module):
    """
    Voice encoder for voice cloning.

    Encodes reference audio mel-spectrogram into voice embedding.

    Args:
        config: Voice encoder configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Convolutional layers for mel-spectrogram
        self.conv_layers = [
            nn.Conv1d(config.num_mels, config.hidden_size, kernel_size=5, padding=2),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2),
        ]

        # Transformer layers
        self.encoder_layers = [
            nn.TransformerEncoderLayer(config.hidden_size, config.num_heads, config.hidden_size * 4)
            for _ in range(config.num_layers)
        ]

        # Voice embedding projection
        self.voice_proj = nn.Linear(config.hidden_size, config.embedding_dim)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Encode mel-spectrogram to voice embedding.

        Args:
            mel: Mel-spectrogram (batch, time, num_mels)
                MLX Conv1d expects NLC format: (batch, length, channels)

        Returns:
            Voice embedding (batch, embedding_dim)
        """
        # MLX Conv1d expects (batch, length, channels) format
        # Input is already in correct format: (batch, time, num_mels)
        x = mel

        # Convolutional layers
        for conv in self.conv_layers:
            x = nn.relu(conv(x))

        # Transformer encoding (no mask needed)
        for layer in self.encoder_layers:
            x = layer(x, mask=None)

        # Global average pooling over time
        x = mx.mean(x, axis=1)

        # Project to voice embedding
        voice_embedding = self.voice_proj(x)

        return voice_embedding


class ExpressivenessModule(nn.Module):
    """
    Expressiveness and emotion control module.

    Args:
        config: Expressiveness configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Emotion embeddings
        self.emotion_embeddings = nn.Embedding(config.num_emotions, config.emotion_embedding_dim)

        # Expressiveness projection
        self.expressiveness_proj = nn.Linear(1, config.emotion_embedding_dim)

    def __call__(self, emotion_id: mx.array = None, expressiveness: float = 0.5) -> mx.array:
        """
        Get expressiveness embedding.

        Args:
            emotion_id: Emotion ID (batch,)
            expressiveness: Expressiveness scale [0, 1]

        Returns:
            Combined expressiveness embedding (batch, emotion_embedding_dim)
        """
        batch_size = emotion_id.shape[0] if emotion_id is not None else 1

        # Get emotion embedding
        if emotion_id is not None:
            emotion_emb = self.emotion_embeddings(emotion_id)
        else:
            # Default to neutral (ID 0)
            emotion_emb = self.emotion_embeddings(mx.zeros((batch_size,), dtype=mx.int32))

        # Get expressiveness embedding
        expressiveness_value = mx.full((batch_size, 1), expressiveness)
        expressiveness_emb = self.expressiveness_proj(expressiveness_value)

        # Combine
        combined = emotion_emb + expressiveness_emb

        return combined


class Chatterbox(nn.Module):
    """
    Chatterbox text-to-speech model.

    Built on 0.5B Llama backbone with:
    - Voice cloning via voice encoder
    - Expressiveness control
    - Emotion embeddings
    - Acoustic head for mel-spectrogram generation

    Total parameters: ~500M

    Args:
        config: Model configuration

    NOTE:
        This is a reference implementation. For production use:
        1. Search HuggingFace for voice cloning TTS models
        2. Load pre-trained weights
        3. Implement full neural vocoder
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        # Set True by the loader once real pre-trained model + vocoder weights
        # are applied. While False, synthesis output is noise, not speech.
        self.weights_loaded = False

        print("Note: Using reference Chatterbox architecture with HiFi-GAN vocoder")
        print("Production deployment requires:")
        print("  1. Pre-trained voice cloning TTS weights")
        print("  2. Pre-trained HiFi-GAN vocoder weights")
        print("  3. Search HuggingFace for similar models")

        # Text tokenizer would be initialized externally

        # Llama backbone (simplified - full implementation needed)
        self.llama_backbone = LlamaBackbone(config.llama_config)

        # Voice encoder
        if config.use_voice_cloning:
            self.voice_encoder = VoiceEncoder(config.voice_encoder_config)
            # Project voice embedding to match hidden size
            self.voice_embedding_proj = nn.Linear(
                config.voice_encoder_config.embedding_dim, config.llama_config.hidden_size
            )

        # Expressiveness module
        if config.use_expressiveness:
            self.expressiveness_module = ExpressivenessModule(config.expressiveness_config)
            # Project expressiveness embedding to match hidden size
            self.expressiveness_proj = nn.Linear(
                config.expressiveness_config.emotion_embedding_dim, config.llama_config.hidden_size
            )

        # Acoustic head (generate mel-spectrogram)
        self.acoustic_head = nn.Linear(
            config.llama_config.hidden_size, config.acoustic_config.num_mels
        )

        # HiFi-GAN Vocoder
        vocoder_config = HiFiGANConfig(
            n_mels=config.acoustic_config.num_mels,
            sample_rate=config.acoustic_config.sample_rate,
            hop_length=config.acoustic_config.hop_length,
        )
        self.vocoder = HiFiGANGenerator(
            n_mels=vocoder_config.n_mels,
            upsample_rates=vocoder_config.upsample_rates,
            upsample_kernel_sizes=vocoder_config.upsample_kernel_sizes,
            upsample_initial_channel=vocoder_config.upsample_initial_channel,
            mrf_kernel_sizes=vocoder_config.mrf_kernel_sizes,
            mrf_dilation_rates=vocoder_config.mrf_dilation_rates,
        )
        print("✓ HiFi-GAN vocoder initialized")

    def encode_voice(self, reference_mel: mx.array) -> mx.array:
        """
        Encode reference audio to voice embedding.

        Args:
            reference_mel: Reference mel-spectrogram (batch, time, n_mels)

        Returns:
            Voice embedding (batch, embedding_dim)
        """
        if not self.config.use_voice_cloning:
            raise ValueError("Voice cloning not enabled in config")

        return self.voice_encoder(reference_mel)

    def __call__(
        self,
        input_ids: mx.array,
        voice_embedding: mx.array = None,
        emotion_id: mx.array = None,
        expressiveness: float = 0.5,
        cache: Any = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass for TTS.

        Args:
            input_ids: Text token IDs (batch, seq_len)
            voice_embedding: Voice embedding for cloning (batch, embedding_dim)
            emotion_id: Emotion ID (batch,)
            expressiveness: Expressiveness scale [0, 1]
            cache: Optional KV cache for generation

        Returns:
            Tuple of (mel_spectrogram, waveform)
            - mel: (batch, time, n_mels)
            - waveform: (batch, samples)
        """
        batch_size, seq_len = input_ids.shape

        # Embed text using Llama backbone
        hidden_states = self.llama_backbone.embedding(input_ids)

        # Add voice embedding if provided
        if voice_embedding is not None and self.config.use_voice_cloning:
            # Project to hidden size
            voice_emb_projected = self.voice_embedding_proj(voice_embedding)
            # Broadcast voice embedding to all positions
            voice_emb_expanded = mx.expand_dims(voice_emb_projected, axis=1)
            voice_emb_broadcast = mx.broadcast_to(
                voice_emb_expanded, (batch_size, seq_len, voice_emb_projected.shape[-1])
            )
            # Add to hidden states (simplified - full implementation would use cross-attention)
            hidden_states = hidden_states + voice_emb_broadcast

        # Add expressiveness/emotion if enabled
        if self.config.use_expressiveness:
            exp_embedding = self.expressiveness_module(emotion_id, expressiveness)
            # Project to hidden size
            exp_projected = self.expressiveness_proj(exp_embedding)
            exp_expanded = mx.expand_dims(exp_projected, axis=1)
            exp_broadcast = mx.broadcast_to(
                exp_expanded, (batch_size, seq_len, exp_projected.shape[-1])
            )
            hidden_states = hidden_states + exp_broadcast

        # Apply transformer layers from Llama backbone with proper attention masks and cache
        hidden_states = self.llama_backbone(
            input_embeddings=hidden_states,
            mask=None,  # Causal mask will be auto-created in LlamaBackbone
            cache=cache,
        )

        # Generate mel-spectrogram
        mel = self.acoustic_head(hidden_states)

        # The HiFi-GAN vocoder's stacked dilated convs have a minimum receptive
        # field (~20 mel frames); a very short prompt yields fewer frames and the
        # conv stack underflows. Vocode a copy padded on the time axis to a safe
        # minimum so any prompt length works, then trim the waveform back to the
        # audio that corresponds to the real (unpadded) frames — no fabricated
        # tail, and the returned mel stays unpadded.
        min_mel_frames = 32
        orig_frames = mel.shape[1]
        vocoder_mel = mel
        if orig_frames < min_mel_frames:
            vocoder_mel = mx.pad(mel, [(0, 0), (0, min_mel_frames - orig_frames), (0, 0)])

        # Generate waveform using HiFi-GAN vocoder
        waveform = self.vocoder(vocoder_mel)

        if orig_frames < min_mel_frames:
            hop = waveform.shape[1] // vocoder_mel.shape[1]  # samples per mel frame
            waveform = waveform[:, : orig_frames * hop]

        return mel, waveform


def create_model(config: ChatterboxConfig = None) -> Chatterbox:
    """
    Create Chatterbox model.

    Args:
        config: Optional configuration

    Returns:
        Chatterbox instance
    """
    if config is None:
        from .config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG

    return Chatterbox(config)
