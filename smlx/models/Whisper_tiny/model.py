"""
Whisper-tiny model architecture for MLX.

Whisper is an encoder-decoder model for automatic speech recognition (ASR).
The encoder processes audio spectrograms, and the decoder generates text transcriptions.

Architecture:
- AudioEncoder: Processes mel-spectrogram features
- TextDecoder: Generates text tokens autoregressively
- Supports multiple languages and translation tasks

Based on OpenAI's Whisper: https://arxiv.org/abs/2212.04356
"""

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .tokenizer import WhisperTokenizer


@dataclass
class ModelConfig:
    """Whisper model configuration.

    Default values are for Whisper-tiny (39M parameters).
    """

    n_mels: int = 80
    """Number of mel-frequency bins"""

    n_audio_ctx: int = 1500
    """Audio context length (30s * 50 frames/sec)"""

    n_audio_state: int = 384
    """Audio encoder hidden dimension"""

    n_audio_head: int = 6
    """Number of audio encoder attention heads"""

    n_audio_layer: int = 4
    """Number of audio encoder layers"""

    n_vocab: int = 51865
    """Vocabulary size (multilingual)"""

    n_text_ctx: int = 448
    """Text context length"""

    n_text_state: int = 384
    """Text decoder hidden dimension"""

    n_text_head: int = 6
    """Number of text decoder attention heads"""

    n_text_layer: int = 4
    """Number of text decoder layers"""

    dtype: str = "float16"
    """Model dtype"""

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "n_mels": self.n_mels,
            "n_audio_ctx": self.n_audio_ctx,
            "n_audio_state": self.n_audio_state,
            "n_audio_head": self.n_audio_head,
            "n_audio_layer": self.n_audio_layer,
            "n_vocab": self.n_vocab,
            "n_text_ctx": self.n_text_ctx,
            "n_text_state": self.n_text_state,
            "n_text_head": self.n_text_head,
            "n_text_layer": self.n_text_layer,
            "dtype": self.dtype,
        }


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> mx.array:
    """Generate sinusoidal positional embeddings.

    Args:
        length: Sequence length
        channels: Number of channels (must be even)
        max_timescale: Maximum timescale for positional encoding

    Returns:
        Positional embeddings of shape (length, channels)
    """
    assert channels % 2 == 0, "channels must be even"

    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]

    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for Whisper."""

    def __init__(self, n_state: int, n_head: int):
        """Initialize multi-head attention.

        Args:
            n_state: Hidden dimension
            n_head: Number of attention heads
        """
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x: mx.array,
        xa: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[mx.array, tuple, mx.array]:
        """Forward pass.

        Args:
            x: Query input of shape (batch, seq_len, n_state)
            xa: Key/value input for cross-attention (optional)
            mask: Attention mask
            kv_cache: Cached key/value tensors

        Returns:
            Tuple of (output, kv_cache, attention_weights)
        """
        q = self.query(x)

        # Self-attention or cross-attention
        if xa is None:
            # Self-attention
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            # Cross-attention without cache
            k = self.key(xa)
            v = self.value(xa)
        else:
            # Cross-attention with cache
            k, v = kv_cache

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), (k, v), qk

    def qkv_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Attention mask

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        # Reshape for multi-head attention
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        # Compute attention scores
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.astype(mx.float32)

        # Apply softmax and compute weighted values
        w = mx.softmax(qk, axis=-1).astype(q.dtype)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)

        return out, qk


class ResidualAttentionBlock(nn.Module):
    """Transformer block with self-attention, optional cross-attention, and MLP."""

    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        """Initialize residual attention block.

        Args:
            n_state: Hidden dimension
            n_head: Number of attention heads
            cross_attention: Whether to include cross-attention layer
        """
        super().__init__()

        # Self-attention
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        # Cross-attention (for decoder only)
        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        # MLP
        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        xa: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[mx.array, tuple, Optional[mx.array]]:
        """Forward pass.

        Args:
            x: Input tensor
            xa: Cross-attention input (encoder output)
            mask: Attention mask
            kv_cache: Cached key/value tensors

        Returns:
            Tuple of (output, kv_cache, cross_attention_weights)
        """
        kv, cross_kv = kv_cache if kv_cache else (None, None)

        # Self-attention
        y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
        x = x + y

        # Cross-attention (if applicable)
        cross_qk = None
        if self.cross_attn:
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=cross_kv
            )
            x = x + y

        # MLP
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))

        return x, (kv, cross_kv), cross_qk


class AudioEncoder(nn.Module):
    """Whisper audio encoder.

    Processes mel-spectrogram features through convolutional layers
    and transformer blocks to produce audio embeddings.
    """

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize audio encoder.

        Args:
            n_mels: Number of mel-frequency bins
            n_ctx: Audio context length
            n_state: Hidden dimension
            n_head: Number of attention heads
            n_layer: Number of transformer layers
            dtype: Model dtype
        """
        super().__init__()

        # Convolutional front-end
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        # Positional embeddings
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)

        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)
        ]
        self.ln_post = nn.LayerNorm(n_state)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode audio features.

        Args:
            x: Mel-spectrogram of shape (batch, n_ctx, n_mels)
                MLX Conv1d expects NLC format: (batch, length, channels)

        Returns:
            Audio embeddings of shape (batch, n_ctx//2, n_state)
        """
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        # Add positional embeddings
        assert x.shape[1:] == self._positional_embedding.shape, (
            f"Incorrect audio shape: got {x.shape}, "
            f"expected (*, {self._positional_embedding.shape[0]}, {self._positional_embedding.shape[1]})"
        )
        x = x + self._positional_embedding

        # Apply transformer blocks
        for block in self.blocks:
            x, _, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    """Whisper text decoder.

    Generates text tokens autoregressively, attending to encoder outputs
    via cross-attention.
    """

    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize text decoder.

        Args:
            n_vocab: Vocabulary size
            n_ctx: Text context length
            n_state: Hidden dimension
            n_head: Number of attention heads
            n_layer: Number of transformer layers
            dtype: Model dtype
        """
        super().__init__()

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        # Transformer blocks with cross-attention
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        self.ln = nn.LayerNorm(n_state)

        # Causal mask
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(
            dtype
        )

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        kv_cache: Optional[list] = None,
    ) -> tuple[mx.array, list, list]:
        """Decode text tokens.

        Args:
            x: Text tokens of shape (batch, seq_len)
            xa: Encoder output of shape (batch, audio_ctx, n_state)
            kv_cache: Cached key/value tensors for each layer

        Returns:
            Tuple of (logits, kv_cache, cross_attention_weights)
        """
        # Compute offset for positional embeddings
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0

        # Embed tokens and add positional embeddings
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        # Initialize cache if needed
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        # Apply transformer blocks
        cross_qk = [None] * len(self.blocks)
        for e, block in enumerate(self.blocks):
            x, kv_cache[e], cross_qk[e] = block(
                x, xa, mask=self._mask, kv_cache=kv_cache[e]
            )

        # Final layer norm and project to vocabulary
        x = self.ln(x)
        logits = x @ self.token_embedding.weight.T

        return logits, kv_cache, cross_qk


class Whisper(nn.Module):
    """Whisper model for automatic speech recognition.

    Combines AudioEncoder and TextDecoder for end-to-end speech recognition.
    Supports multilingual transcription and translation.
    """

    def __init__(self, config: ModelConfig):
        """Initialize Whisper model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Get dtype
        dtype_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        dtype = dtype_map.get(config.dtype, mx.float16)

        # Initialize encoder and decoder
        self.encoder = AudioEncoder(
            config.n_mels,
            config.n_audio_ctx,
            config.n_audio_state,
            config.n_audio_head,
            config.n_audio_layer,
            dtype,
        )

        self.decoder = TextDecoder(
            config.n_vocab,
            config.n_text_ctx,
            config.n_text_state,
            config.n_text_head,
            config.n_text_layer,
            dtype,
        )

    def encode_audio(self, mel: mx.array) -> mx.array:
        """Encode mel-spectrogram to audio embeddings.

        Args:
            mel: Mel-spectrogram of shape (batch, n_mels, n_frames)

        Returns:
            Audio embeddings of shape (batch, n_frames//2, n_state)
        """
        return self.encoder(mel)

    def decode_text(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: Optional[list] = None,
    ) -> tuple[mx.array, list]:
        """Decode text tokens given audio features.

        Args:
            tokens: Text tokens of shape (batch, seq_len)
            audio_features: Encoded audio of shape (batch, audio_ctx, n_state)
            kv_cache: Cached key/value tensors

        Returns:
            Tuple of (logits, kv_cache)
        """
        logits, kv_cache, _ = self.decoder(tokens, audio_features, kv_cache)
        return logits, kv_cache

    def __call__(self, mel: mx.array, tokens: mx.array) -> mx.array:
        """Forward pass.

        Args:
            mel: Mel-spectrogram of shape (batch, n_mels, n_frames)
            tokens: Text tokens of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        audio_features = self.encode_audio(mel)
        logits, _, _ = self.decoder(tokens, audio_features)
        return logits

    @property
    def is_multilingual(self) -> bool:
        """Check if model is multilingual."""
        return self.config.n_vocab >= 51865

    @property
    def num_languages(self) -> int:
        """Get number of supported languages."""
        return self.config.n_vocab - 51765 - int(self.is_multilingual)

    @property
    def alignment_heads(self) -> list[tuple[int, int]]:
        """Get alignment heads for word-level timestamp alignment.

        Returns layer/head index pairs that are used for computing word-level
        timestamps via Dynamic Time Warping on cross-attention patterns.

        For Whisper-tiny, we use all heads from the last decoder layer.

        Returns:
            List of (layer_index, head_index) tuples
        """
        last_layer = self.config.n_text_layer - 1
        return [(last_layer, head) for head in range(self.config.n_text_head)]

    def detect_language(self, mel: mx.array, tokenizer: Optional["WhisperTokenizer"] = None):
        """Detect spoken language in audio.

        Convenience method that calls the detect_language function from decoding module.

        Args:
            mel: Mel spectrogram of shape (n_mels, n_frames) or (batch, n_mels, n_frames)
            tokenizer: Tokenizer (created if not provided)

        Returns:
            Tuple of (language_tokens, language_probs)

        Example:
            >>> model, tokenizer = load()
            >>> mel = prepare_audio("speech.wav")
            >>> lang_tokens, lang_probs = model.detect_language(mel, tokenizer)
            >>> max(lang_probs[0], key=lang_probs[0].get)
            'en'
        """
        from .decoding import detect_language

        return detect_language(self, mel, tokenizer)

    def forward_with_cross_qk(
        self, mel: mx.array, tokens: mx.array
    ) -> tuple[mx.array, list]:
        """Forward pass that returns cross-attention QK scores for word-level timestamp alignment.

        This method is used for computing word-level timestamps by exposing the cross-attention
        patterns between the decoder and encoder. These patterns are used with Dynamic Time
        Warping to align words with their corresponding audio frames.

        Args:
            mel: Mel-spectrogram of shape (batch, n_mels, n_frames)
            tokens: Text tokens of shape (batch, seq_len)

        Returns:
            Tuple of (logits, cross_qk) where:
            - logits: Shape (batch, seq_len, vocab_size)
            - cross_qk: List of cross-attention QK scores for each layer,
                       each with shape (batch, n_heads, seq_len, audio_ctx)

        Example:
            >>> model, tokenizer = load()
            >>> mel = prepare_audio("speech.wav")
            >>> tokens = tokenizer.encode("hello world")
            >>> logits, cross_qk = model.forward_with_cross_qk(mel, tokens)
            >>> # Use cross_qk with DTW for word-level timestamps
        """
        audio_features = self.encode_audio(mel)
        logits, _, cross_qk = self.decoder(tokens, audio_features)
        return logits, cross_qk
