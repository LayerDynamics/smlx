#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Configuration for Chatterbox TTS model.

Chatterbox is built on a 0.5B Llama backbone with added voice cloning
and expressiveness control capabilities.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LlamaBackboneConfig:
    """
    Llama backbone configuration for TTS (based on SmolLM2_135M architecture).

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads for GQA
        intermediate_size: FFN intermediate dimension
        max_position_embeddings: Maximum sequence length
        rope_theta: RoPE theta parameter
        head_dim: Dimension of each attention head (computed if None)
        attention_bias: Whether to use bias in attention projections
        mlp_bias: Whether to use bias in MLP layers
        rope_traditional: Whether to use traditional RoPE
        rope_scaling: Optional RoPE scaling configuration
        tie_word_embeddings: Whether to tie input/output embeddings
        layer_types: Layer type for each layer (e.g., ["full_attention"] * num_layers)
        sliding_window: Optional sliding window size
        no_rope_layer_interval: Interval for disabling RoPE (NoPE feature)
        no_rope_layers: Binary list indicating which layers have RoPE disabled
        rms_norm_eps: Epsilon for RMS normalization
        attention_dropout: Attention dropout rate
        hidden_dropout: Hidden state dropout rate
    """

    vocab_size: int = 49152  # SmolLM2 vocab
    hidden_size: int = 1024  # 0.5B model
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA
    intermediate_size: int = 2752
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Additional fields for full Llama support (from SmolLM2_135M)
    head_dim: Optional[int] = None  # Computed as hidden_size // num_attention_heads if None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = True
    layer_types: Optional[list] = None  # Set to ["full_attention"] * num_hidden_layers in __post_init__
    sliding_window: Optional[int] = None
    no_rope_layer_interval: int = 4  # Every 4th layer has RoPE disabled (SmolLM3/NoPE feature)
    no_rope_layers: Optional[list] = None  # Binary list set in __post_init__

    def __post_init__(self):
        """Post-initialization to set default values."""
        # Set default layer types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        # Configure NoPE layers (SmolLM3 feature)
        # Every 4th layer has RoPE disabled by default
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0)
                for i in range(self.num_hidden_layers)
            ]
        elif len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError(
                f"`no_rope_layers` length ({len(self.no_rope_layers)}) "
                f"must match num_hidden_layers ({self.num_hidden_layers})"
            )


@dataclass
class VoiceEncoderConfig:
    """
    Voice encoder for voice cloning.

    Encodes reference audio into voice embedding.

    Args:
        num_mels: Number of mel-frequency bins
        hidden_size: Hidden dimension
        num_layers: Number of encoder layers
        num_heads: Number of attention heads
        embedding_dim: Output voice embedding dimension
    """

    num_mels: int = 80
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    embedding_dim: int = 256  # Voice embedding size


@dataclass
class ExpressivenessConfig:
    """
    Expressiveness control configuration.

    Args:
        num_emotions: Number of emotion categories
        emotion_embedding_dim: Emotion embedding dimension
        expressiveness_range: Min/max expressiveness scale
    """

    num_emotions: int = 8  # neutral, happy, sad, angry, excited, calm, etc.
    emotion_embedding_dim: int = 128
    expressiveness_range: tuple = (0.0, 1.0)


@dataclass
class AcousticConfig:
    """
    Acoustic feature generation configuration.

    Args:
        num_mels: Number of mel-frequency bins
        sample_rate: Audio sample rate
        hop_length: STFT hop length
        win_length: STFT window length
        n_fft: FFT size
    """

    num_mels: int = 80
    sample_rate: int = 24000
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024


@dataclass
class ChatterboxConfig:
    """
    Chatterbox TTS model configuration.

    Total parameters: ~500M
    - Llama backbone: ~400M
    - Voice encoder: ~40M
    - Acoustic head: ~40M
    - Expressiveness/emotion: ~20M

    Args:
        llama_config: Llama backbone configuration
        voice_encoder_config: Voice encoder configuration
        expressiveness_config: Expressiveness control configuration
        acoustic_config: Acoustic feature configuration
    """

    model_type: str = "chatterbox"
    llama_config: LlamaBackboneConfig = None
    voice_encoder_config: VoiceEncoderConfig = None
    expressiveness_config: ExpressivenessConfig = None
    acoustic_config: AcousticConfig = None

    # Model behavior
    use_voice_cloning: bool = True
    use_expressiveness: bool = True
    use_emotion_control: bool = True

    def __post_init__(self):
        if self.llama_config is None:
            self.llama_config = LlamaBackboneConfig()
        if self.voice_encoder_config is None:
            self.voice_encoder_config = VoiceEncoderConfig()
        if self.expressiveness_config is None:
            self.expressiveness_config = ExpressivenessConfig()
        if self.acoustic_config is None:
            self.acoustic_config = AcousticConfig()

    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)


# Default configuration
DEFAULT_CONFIG = ChatterboxConfig()

# Available emotions
AVAILABLE_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "excited",
    "calm",
    "surprised",
    "fearful",
]


def load_config(model_path: str) -> ChatterboxConfig:
    """
    Load configuration from model directory.

    Args:
        model_path: Path to model directory

    Returns:
        ChatterboxConfig instance
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        print(f"Config not found at {config_path}, using default config")
        return DEFAULT_CONFIG

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Parse nested configs
    llama_config = LlamaBackboneConfig(**config_dict.get("llama_config", {}))
    voice_encoder_config = VoiceEncoderConfig(
        **config_dict.get("voice_encoder_config", config_dict.get("voice_config", {}))
    )
    expressiveness_config = ExpressivenessConfig(
        **config_dict.get("expressiveness_config", {})
    )
    acoustic_config = AcousticConfig(**config_dict.get("acoustic_config", {}))

    # Create main config
    config = ChatterboxConfig(
        llama_config=llama_config,
        voice_encoder_config=voice_encoder_config,
        expressiveness_config=expressiveness_config,
        acoustic_config=acoustic_config,
        use_voice_cloning=config_dict.get("use_voice_cloning", True),
        use_expressiveness=config_dict.get("use_expressiveness", True),
        use_emotion_control=config_dict.get("use_emotion_control", True),
    )

    return config


def save_config(config: ChatterboxConfig, output_path: str):
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        output_path: Output directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"✓ Config saved to {config_path}")
