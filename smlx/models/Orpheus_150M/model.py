#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Orpheus-150M TTS model architecture.

NOTE: This is a reference implementation showing the API structure.
For production use, load pre-trained weights from HuggingFace Hub.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import Orpheus150MConfig
from .vocoder import HiFiGANVocoder, HiFiGANConfig


class TextEncoder(nn.Module):
    """
    Text encoder for TTS.

    Encodes text/phonemes into hidden representations.

    Args:
        config: Text encoder configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Positional encoding
        self.pos_encoding = nn.Embedding(config.max_seq_len, config.embedding_dim)

        # Transformer layers
        self.layers = [
            nn.TransformerEncoderLayer(
                config.embedding_dim,
                config.num_heads,
                config.hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ]

        self.norm = nn.LayerNorm(config.embedding_dim)

    def __call__(self, input_ids: mx.array, mask: mx.array = None) -> mx.array:
        """
        Encode text.

        Args:
            input_ids: Token IDs (batch, seq_len)
            mask: Attention mask (batch, seq_len)

        Returns:
            Encoded features (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.embedding(input_ids)

        # Add positional encoding
        positions = mx.arange(seq_len)
        pos_emb = self.pos_encoding(positions)
        x = x + pos_emb

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        # Final norm
        x = self.norm(x)

        return x


class DurationPredictor(nn.Module):
    """
    Predicts duration for each phoneme.

    Args:
        config: Duration predictor configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Convolutional layers
        self.convs = []
        for i in range(config.num_layers):
            in_dim = config.input_dim if i == 0 else config.hidden_dim
            self.convs.append(
                nn.Conv1d(
                    in_dim,
                    config.hidden_dim,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2,
                )
            )

        # Output layer (predict log duration)
        self.output = nn.Linear(config.hidden_dim, 1)

    def __call__(self, encoder_output: mx.array) -> mx.array:
        """
        Predict durations.

        Args:
            encoder_output: Encoder features (batch, seq_len, input_dim)

        Returns:
            Predicted durations (batch, seq_len)
        """
        # MLX Conv1d expects (batch, length, in_channels)
        x = encoder_output

        # Apply convolutions
        for conv in self.convs:
            x = nn.relu(conv(x))

        # Predict duration (log scale)
        log_durations = self.output(x).squeeze(-1)

        # Convert to actual durations
        durations = mx.exp(log_durations)

        return durations


class AcousticDecoder(nn.Module):
    """
    Acoustic decoder for generating mel-spectrograms.

    Args:
        config: Decoder configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer decoder layers
        self.layers = [
            nn.TransformerEncoderLayer(
                config.input_dim, config.num_heads, config.hidden_dim, dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ]

        self.norm = nn.LayerNorm(config.input_dim)

        # Output projection to mel-spectrogram
        self.mel_proj = nn.Linear(config.input_dim, config.num_mels)

    def __call__(self, encoder_output: mx.array) -> mx.array:
        """
        Generate mel-spectrogram.

        Args:
            encoder_output: Encoder features (batch, seq_len, input_dim)

        Returns:
            Mel-spectrogram (batch, seq_len, num_mels)
        """
        x = encoder_output

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, mask=None)

        # Normalize
        x = self.norm(x)

        # Project to mel-spectrogram
        mel = self.mel_proj(x)

        return mel


# Vocoder is now imported from vocoder.py
# See vocoder.py for HiFi-GAN V3 implementation


class Orpheus150M(nn.Module):
    """
    Orpheus-150M text-to-speech model.

    Architecture:
    1. Text Encoder: Encodes text/phonemes -> hidden representations
    2. Duration Predictor: Predicts how long each phoneme lasts
    3. Acoustic Decoder: Generates mel-spectrogram
    4. Vocoder: Converts mel-spectrogram to waveform

    Total parameters: ~150M

    Args:
        config: Model configuration

    NOTE:
        This is a reference implementation. For production use:
        1. Load pre-trained weights from HuggingFace Hub
        2. Implement full neural vocoder (HiFi-GAN, etc.)
        3. Add pitch/energy prediction for naturalness
    """

    def __init__(self, config: Orpheus150MConfig):
        super().__init__()
        self.config = config

        print("Note: Orpheus-150M TTS Model initialized")
        print("Vocoder: HiFi-GAN V3 (lightweight, 0.92M params)")
        print("For best quality, load pre-trained weights:")
        print("  - HuggingFace Hub: canopylabs/orpheus-150m-* (when available)")
        print("  - Vocoder weights: nvidia/tts_hifigan")

        # Create components
        self.text_encoder = TextEncoder(config.text_encoder_config)
        self.duration_predictor = DurationPredictor(config.duration_config)
        self.decoder = AcousticDecoder(config.decoder_config)

        # Initialize HiFi-GAN vocoder
        # Use vocoder_config if provided, otherwise create default V3 config
        if hasattr(config, 'vocoder_config') and isinstance(config.vocoder_config, HiFiGANConfig):
            vocoder_config = config.vocoder_config
        else:
            vocoder_config = HiFiGANConfig(mel_channels=config.decoder_config.num_mels)
        self.vocoder = HiFiGANVocoder(vocoder_config)

    def __call__(
        self, input_ids: mx.array, durations: mx.array = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass for TTS.

        Args:
            input_ids: Text token IDs (batch, seq_len)
            durations: Optional target durations for training (batch, seq_len)

        Returns:
            Tuple of (waveform, mel, predicted_durations)
            - waveform: (batch, samples)
            - mel: (batch, time, num_mels)
            - predicted_durations: (batch, seq_len)
        """
        # Encode text
        encoder_output = self.text_encoder(input_ids)

        # Predict durations
        predicted_durations = self.duration_predictor(encoder_output)

        # Use predicted or provided durations
        if durations is None:
            durations = predicted_durations

        # Expand encoder output according to durations
        # This creates the time-aligned sequence
        expanded_output = self._expand_by_duration(encoder_output, durations)

        # Generate mel-spectrogram
        mel = self.decoder(expanded_output)

        # Generate waveform
        waveform = self.vocoder(mel)

        return waveform, mel, predicted_durations

    def _expand_by_duration(self, encoder_output: mx.array, durations: mx.array) -> mx.array:
        """
        Expand encoder output according to predicted durations (Length Regulator).

        This implements the length regulator from FastSpeech, which expands each encoder
        frame to match its predicted duration, creating time-aligned representations.

        Args:
            encoder_output: Encoder outputs (batch, seq_len, dim)
            durations: Predicted durations for each frame (batch, seq_len)

        Returns:
            Expanded output (batch, total_time, dim)
            where total_time = sum of all durations

        Example:
            >>> encoder_output.shape  # (1, 4, 384)
            >>> durations = mx.array([[2, 3, 1, 2]])  # Each frame repeated N times
            >>> expanded = _expand_by_duration(encoder_output, durations)
            >>> expanded.shape  # (1, 8, 384) - sum([2,3,1,2]) = 8
        """
        batch_size, seq_len, dim = encoder_output.shape

        # Round and clamp durations to valid range
        durations_int = mx.round(durations).astype(mx.int32)
        durations_int = mx.clip(
            durations_int,
            self.config.min_duration,
            self.config.max_duration,
        )

        # Expand each sequence in batch
        expanded_batch = []

        for b in range(batch_size):
            # Process each sequence independently
            seq_frames = []

            for i in range(seq_len):
                # Get duration for this frame
                duration = int(durations_int[b, i])

                if duration > 0:
                    # Extract frame and repeat it 'duration' times
                    frame = encoder_output[b, i : i + 1, :]  # (1, dim)
                    repeated = mx.repeat(frame, duration, axis=0)  # (duration, dim)
                    seq_frames.append(repeated)

            # Concatenate all expanded frames for this sequence
            if seq_frames:
                expanded_seq = mx.concatenate(seq_frames, axis=0)
            else:
                # Handle edge case: all durations are 0
                expanded_seq = mx.zeros((1, dim))

            expanded_batch.append(expanded_seq)

        # Find maximum length in batch for padding
        max_len = max(seq.shape[0] for seq in expanded_batch)

        # Pad all sequences to same length
        padded_batch = []
        for seq in expanded_batch:
            if seq.shape[0] < max_len:
                # Pad with zeros
                pad_len = max_len - seq.shape[0]
                padding = mx.zeros((pad_len, dim))
                seq = mx.concatenate([seq, padding], axis=0)
            padded_batch.append(seq)

        # Stack into batch
        return mx.stack(padded_batch, axis=0)  # (batch, max_len, dim)


def create_model(config: Orpheus150MConfig = None) -> Orpheus150M:
    """
    Create Orpheus-150M model.

    Args:
        config: Optional configuration

    Returns:
        Orpheus150M instance
    """
    if config is None:
        from .config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG

    return Orpheus150M(config)
