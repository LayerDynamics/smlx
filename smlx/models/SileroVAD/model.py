#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Silero VAD Model Architecture.

Implements the Silero Voice Activity Detection model using LSTM.
The model is lightweight (~1MB) and optimized for real-time inference.

Architecture:
    - Input: Raw audio waveform
    - LSTM layers for temporal modeling
    - Output: Speech probability (0-1)
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VADConfig


class SileroVAD(nn.Module):
    """Silero Voice Activity Detection model.

    A compact LSTM-based model for detecting speech activity in audio.
    Processes audio in chunks and outputs speech probability.

    Args:
        config: VAD configuration
    """

    def __init__(self, config: VADConfig):
        super().__init__()
        self.config = config

        # LSTM layers (MLX requires manual stacking)
        self.lstm_layers = []
        for i in range(config.num_layers):
            input_size = config.context_size if i == 0 else config.hidden_size
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=config.hidden_size,
                    bias=True,
                )
            )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, 1)

        # Initialize hidden state
        self._hidden_state: Optional[list] = None

    def forward(
        self,
        x: mx.array,
        hidden: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """Forward pass through the VAD model.

        Args:
            x: Input audio tensor [batch, seq_len, context_size]
            hidden: Optional list of hidden states (one per layer) from previous forward pass
                   Each element is a tuple (hidden_state, cell_state) or None

        Returns:
            Tuple of (speech_probs, hidden_states)
            - speech_probs: Speech probabilities [batch, seq_len, 1]
            - hidden_states: Updated LSTM hidden states (list of tuples (h, c))
        """
        # Initialize hidden states if needed
        if hidden is None:
            hidden = [None] * len(self.lstm_layers)

        # Pass through LSTM layers sequentially
        new_hidden = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            # Extract hidden and cell states for this layer (if they exist)
            if hidden[i] is not None:
                h_prev, c_prev = hidden[i]
            else:
                h_prev, c_prev = None, None

            # MLX LSTM returns (hidden_sequence, cell_sequence)
            # and expects separate hidden and cell arguments
            hidden_seq, cell_seq = lstm_layer(x, hidden=h_prev, cell=c_prev)

            # CRITICAL: Evaluate after each LSTM layer to prevent exponential graph growth
            # Without this, graphs accumulate across layers and iterations causing Metal GPU overflow
            mx.eval(hidden_seq)
            mx.eval(cell_seq)

            # Use hidden sequence as input to next layer
            x = hidden_seq

            # Extract final timestep states for next iteration
            # Shape: [batch, seq_len, hidden_size] -> [batch, hidden_size]
            final_hidden = hidden_seq[..., -1, :]
            final_cell = cell_seq[..., -1, :]

            # Evaluate final states to prevent graph buildup
            mx.eval(final_hidden)
            mx.eval(final_cell)

            # Store as tuple for next iteration
            new_hidden.append((final_hidden, final_cell))

        # Project to speech probability
        logits = self.output_proj(x)

        # Apply sigmoid for probability
        probs = mx.sigmoid(logits)

        return probs, new_hidden

    def __call__(
        self,
        x: mx.array,
        hidden: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """Make model callable."""
        return self.forward(x, hidden)

    def reset_state(self) -> None:
        """Reset the hidden state for streaming inference."""
        self._hidden_state = None

    def predict(
        self,
        audio: mx.array,
        reset_state: bool = False,
    ) -> mx.array:
        """Predict speech probability for audio chunk.

        Args:
            audio: Audio waveform [seq_len] or [batch, seq_len]
            reset_state: Whether to reset hidden state

        Returns:
            Speech probabilities [seq_len] or [batch, seq_len]
        """
        # Defensive type check
        import numpy as np
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio.astype(np.float32))
        if not isinstance(audio, mx.array):
            raise TypeError(f"audio must be MLX array, got {type(audio)}")

        if reset_state:
            self.reset_state()

        # Ensure batch dimension
        if audio.ndim == 1:
            audio = audio[None, :]
            squeeze_output = True
        else:
            squeeze_output = False

        # Reshape to [batch, seq_len, context_size]
        batch_size = audio.shape[0]
        seq_len = audio.shape[1]

        # Pad if needed
        if seq_len < self.config.context_size:
            padding = self.config.context_size - seq_len
            audio = mx.pad(audio, [(0, 0), (0, padding)])
            seq_len = self.config.context_size

        # Reshape to chunks
        num_chunks = seq_len // self.config.context_size
        if seq_len % self.config.context_size != 0:
            # Pad to multiple of context_size
            padding = self.config.context_size - (seq_len % self.config.context_size)
            audio = mx.pad(audio, [(0, 0), (0, padding)])
            num_chunks += 1

        # Reshape to [batch, num_chunks, context_size]
        audio = audio.reshape(batch_size, num_chunks, self.config.context_size)

        # Forward pass
        probs, self._hidden_state = self.forward(audio, self._hidden_state)

        # Force evaluation of both probabilities and hidden state to prevent graph accumulation
        # Critical fix: Without this, LSTM states build exponential computation graphs
        # across iterations, leading to Metal GPU buffer overflow and kernel panic
        # Evaluate probs first
        mx.eval(probs)

        # Then evaluate hidden states (list of (h, c) tuples)
        if self._hidden_state is not None:
            # Flatten list of tuples and evaluate all hidden and cell states at once
            all_states = [state for layer_states in self._hidden_state
                          for state in layer_states]
            mx.eval(all_states)

        # Flatten to [batch, num_chunks]
        probs = probs.squeeze(-1)

        if squeeze_output:
            probs = probs.squeeze(0)

        return probs

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Sanitize weights from different source formats.

        Args:
            weights: Model weights dictionary

        Returns:
            Sanitized weights compatible with MLX
        """
        sanitized = {}

        for key, value in weights.items():
            # Remove common prefixes
            new_key = key
            for prefix in ["model.", "encoder.", "vad."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]

            # Map PyTorch LSTM parameter names to MLX
            new_key = new_key.replace("weight_ih_l", "Wii_")
            new_key = new_key.replace("weight_hh_l", "Whi_")
            new_key = new_key.replace("bias_ih_l", "bii_")
            new_key = new_key.replace("bias_hh_l", "bhi_")

            # Map layer indices
            new_key = new_key.replace("_0", "_layer_0")
            new_key = new_key.replace("_1", "_layer_1")

            sanitized[new_key] = value

        return sanitized


class StreamingVAD:
    """Streaming wrapper for Silero VAD.

    Provides convenient interface for processing audio streams
    with automatic chunking and state management.
    """

    def __init__(self, model: SileroVAD, config: VADConfig):
        """Initialize streaming VAD.

        Args:
            model: Silero VAD model
            config: VAD configuration
        """
        self.model = model
        self.config = config

        # Streaming state
        self.buffer = mx.array([])
        self.model.reset_state()

    def process_chunk(self, audio_chunk: mx.array) -> mx.array:
        """Process an audio chunk and return speech probabilities.

        Args:
            audio_chunk: Audio samples [num_samples]

        Returns:
            Speech probability for this chunk (scalar)
        """
        # Convert to MLX array if needed (handle numpy arrays from tests)
        import numpy as np
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = mx.array(audio_chunk.astype(np.float32))

        # Ensure it's definitely an MLX array (defensive check)
        if not isinstance(audio_chunk, mx.array):
            raise TypeError(f"audio_chunk must be MLX array or numpy array, got {type(audio_chunk)}")

        # Add to buffer
        if self.buffer.size > 0:
            self.buffer = mx.concatenate([self.buffer, audio_chunk])
        else:
            self.buffer = audio_chunk

        # Process complete windows
        window_size = self.config.window_size_samples
        results = []

        # Timeout protection: prevent infinite loops
        MAX_ITERATIONS = 1000  # Safety limit
        iteration = 0

        while self.buffer.size >= window_size:
            iteration += 1
            if iteration > MAX_ITERATIONS:
                raise RuntimeError(
                    f"StreamingVAD exceeded max iterations ({MAX_ITERATIONS}). "
                    f"Buffer size: {self.buffer.size}, window_size: {window_size}. "
                    "This may indicate a buffer slicing issue or corrupted state."
                )

            # Extract window
            window = self.buffer[:window_size]
            new_buffer = self.buffer[window_size:]

            # Defensive check: ensure buffer is actually shrinking
            if new_buffer.size >= self.buffer.size:
                raise RuntimeError(
                    f"Buffer size not decreasing: {self.buffer.size} -> {new_buffer.size}. "
                    "MLX array slicing may have failed. This is a critical error."
                )

            self.buffer = new_buffer

            # Force evaluation to prevent lazy computation buildup
            mx.eval(self.buffer)
            mx.eval(window)

            # Predict
            prob = self.model.predict(window, reset_state=False)

            # Force evaluation of probability to prevent graph accumulation
            # The predict() method already evaluates hidden states internally
            mx.eval(prob)

            # Take mean if multiple values
            if prob.size > 1:
                prob = mx.mean(prob)
                mx.eval(prob)

            results.append(prob.item())

        if results:
            return mx.array(results)
        else:
            return mx.array([])

    def reset(self) -> None:
        """Reset streaming state."""
        self.buffer = mx.array([])
        self.model.reset_state()


__all__ = ["SileroVAD", "StreamingVAD"]
