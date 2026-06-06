#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
CAD sequence tokenizer.

This module provides tokenization for CAD command sequences, converting
between CAD operations and token IDs for model processing.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .commands import CADCommandType, get_command_parameters, validate_parameters
from .config import CADVocabularyConfig

# Authoritative CAD vocabulary size. The decoder embedding and the generation
# head MUST be sized to this so every token the tokenizer can emit (max id 1103)
# is representable. Breakdown: 10 special + 70 commands + 4 * 256 parameter bins
# (parameter offset base 80) = 1104. Import this constant instead of hardcoding
# the number elsewhere; CADTokenizer asserts its computed size matches it.
CAD_VOCAB_SIZE = 1104


class CADTokenizer:
    """
    Tokenizer for CAD command sequences.

    Converts between CAD operations (command + parameters) and token sequences
    for neural network processing.

    Example:
        >>> tokenizer = CADTokenizer()
        >>> sequence = [
        ...     (CADCommandType.SKETCH_START, {"plane": "XY"}),
        ...     (CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50}),
        ...     (CADCommandType.SKETCH_END, {}),
        ...     (CADCommandType.EXTRUDE, {"distance": 100}),
        ... ]
        >>> tokens = tokenizer.encode(sequence)
        >>> decoded = tokenizer.decode(tokens)
    """

    def __init__(self, config: CADVocabularyConfig | None = None):
        """
        Initialize CAD tokenizer.

        Args:
            config: Vocabulary configuration (uses default if None)
        """
        self.config = config or CADVocabularyConfig()

        # Special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 3  # Separator between command and parameters

        # Command token offset (after special tokens)
        self.command_offset = 10

        # Build vocabulary
        self._build_vocabulary()

    def _build_vocabulary(self):
        """
        Build command and parameter vocabularies.

        Uses 8-bit quantization (256 bins) following Text2CAD (NeurIPS 2024) approach.
        This reduces vocabulary size while maintaining reconstruction precision.

        Vocabulary structure:
            - Tokens 0-9: Special tokens (PAD, BOS, EOS, SEP, etc.)
            - Tokens 10-79: CAD commands (70 commands reserved)
            - Tokens 80-335: X coordinates (256 bins, 8-bit quantization)
            - Tokens 336-591: Y coordinates (256 bins, 8-bit quantization)
            - Tokens 592-847: Distances/radii (256 bins, 8-bit quantization)
            - Tokens 848-1103: Angles (256 bins for 0-360°, 8-bit quantization)

        Total vocabulary size: ~1104 tokens (reduced from 1100+ with better organization)
        """
        # Command to token ID mapping
        self.command_to_token = {cmd: self.command_offset + cmd.value for cmd in CADCommandType}
        self.token_to_command = {v: k for k, v in self.command_to_token.items()}

        # Parameter quantization bins (8-bit = 256 bins per parameter type)
        self.param_bins = 256  # 8-bit quantization

        # Parameter token offsets (organized by type)
        self.param_offset_base = 80  # Start after commands
        self.x_coord_offset = self.param_offset_base  # 80-335
        self.y_coord_offset = self.x_coord_offset + self.param_bins  # 336-591
        self.distance_offset = self.y_coord_offset + self.param_bins  # 592-847
        self.angle_offset = self.distance_offset + self.param_bins  # 848-1103

        # Vocabulary size: special tokens + commands + parameter bins
        # 10 special + 70 commands + (4 * 256 parameter bins) = 10 + 70 + 1024 = 1104
        self.vocab_size = self.angle_offset + self.param_bins
        # Guard against the scheme drifting away from the shared constant that
        # the decoder/head are sized to.
        assert self.vocab_size == CAD_VOCAB_SIZE, (
            f"CAD vocab scheme drifted: computed {self.vocab_size} != "
            f"CAD_VOCAB_SIZE={CAD_VOCAB_SIZE}"
        )

    def encode(
        self,
        sequence: list[tuple[CADCommandType, dict[str, Any]]],
        add_special_tokens: bool = True,
    ) -> mx.array:
        """
        Encode CAD sequence to token IDs.

        Args:
            sequence: List of (command, parameters) tuples
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            Token IDs as MLX array [seq_len]

        Example:
            >>> tokenizer = CADTokenizer()
            >>> seq = [(CADCommandType.CIRCLE, {"cx": 0, "cy": 0, "r": 50})]
            >>> tokens = tokenizer.encode(seq)
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        for command, parameters in sequence:
            # Encode command
            command_token = self.command_to_token[command]
            tokens.append(command_token)

            # Encode parameters
            param_tokens = self._encode_parameters(command, parameters)
            tokens.extend(param_tokens)

            # Add separator
            tokens.append(self.sep_token_id)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return mx.array(tokens, dtype=mx.int32)

    def decode(
        self, tokens: mx.array | list[int], skip_special_tokens: bool = True
    ) -> list[tuple[CADCommandType, dict[str, Any]]]:
        """
        Decode token IDs to CAD sequence.

        Args:
            tokens: Token IDs as MLX array or list
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of (command, parameters) tuples

        Example:
            >>> tokenizer = CADTokenizer()
            >>> tokens = mx.array([1, 21, 100, 100, 150, 3, 2])  # Example tokens
            >>> sequence = tokenizer.decode(tokens)
        """
        if isinstance(tokens, mx.array):
            tokens = tokens.tolist()

        sequence = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Skip special tokens
            if skip_special_tokens and token in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ):
                i += 1
                continue

            # Check if this is a command token
            if token in self.token_to_command:
                command = self.token_to_command[token]
                i += 1

                # Collect parameter tokens until separator
                param_tokens = []
                while i < len(tokens) and tokens[i] != self.sep_token_id:
                    if tokens[i] not in (
                        self.pad_token_id,
                        self.bos_token_id,
                        self.eos_token_id,
                    ):
                        param_tokens.append(tokens[i])
                    i += 1

                # Decode parameters
                parameters = self._decode_parameters(command, param_tokens)

                sequence.append((command, parameters))

            i += 1

        return sequence

    def _encode_parameters(self, command: CADCommandType, parameters: dict[str, Any]) -> list[int]:
        """
        Encode command parameters to token IDs using 8-bit quantization.

        Uses 256 bins per parameter type (8-bit quantization) following
        Text2CAD (NeurIPS 2024) for efficient encoding while maintaining
        reconstruction precision.

        Parameter types map to different token ranges:
            - X coordinates (cx, x1, x2, etc.): tokens 80-335
            - Y coordinates (cy, y1, y2, etc.): tokens 336-591
            - Distances/radii (r, distance, width, height): tokens 592-847
            - Angles (angle, start_angle, end_angle): tokens 848-1103

        Args:
            command: CAD command type
            parameters: Parameter dictionary

        Returns:
            List of parameter token IDs
        """
        param_specs = get_command_parameters(command)
        tokens = []

        for spec in param_specs:
            if spec.name not in parameters:
                continue

            value = parameters[spec.name]

            if spec.type is float:
                # Determine parameter type by name
                param_name = spec.name.lower()

                # Determine offset and range based on parameter type
                if any(x in param_name for x in ["cx", "x1", "x2", "x", "start_x", "end_x"]):
                    # X coordinate
                    offset = self.x_coord_offset
                    min_val = spec.min_value or self.config.min_coordinate
                    max_val = spec.max_value or self.config.max_coordinate

                elif any(y in param_name for y in ["cy", "y1", "y2", "y", "start_y", "end_y"]):
                    # Y coordinate
                    offset = self.y_coord_offset
                    min_val = spec.min_value or self.config.min_coordinate
                    max_val = spec.max_value or self.config.max_coordinate

                elif any(a in param_name for a in ["angle", "rotation", "theta"]):
                    # Angle (typically 0-360 degrees or 0-2π radians)
                    offset = self.angle_offset
                    min_val = spec.min_value or 0.0
                    max_val = spec.max_value or 360.0

                else:
                    # Distance/radius/size
                    offset = self.distance_offset
                    min_val = spec.min_value or 0.0
                    max_val = spec.max_value or self.config.max_coordinate

                # 8-bit quantization: map [min_val, max_val] to [0, 255]
                normalized = (value - min_val) / (max_val - min_val)
                quantized = int(normalized * 255)
                quantized = max(0, min(255, quantized))  # Clamp to [0, 255]

                token = offset + quantized
                tokens.append(token)

            elif spec.type is int:
                # Direct encoding for small integers (use distance offset)
                # Clamp to valid range
                quantized = max(0, min(255, value))
                token = self.distance_offset + quantized
                tokens.append(token)

            elif spec.type is str:
                # Encode string as hash (simplified)
                # Use distance offset for string hashes
                hash_val = hash(value) % 256
                token = self.distance_offset + hash_val
                tokens.append(token)

        return tokens

    def _decode_parameters(
        self, command: CADCommandType, param_tokens: list[int]
    ) -> dict[str, Any]:
        """
        Decode parameter tokens to dictionary using 8-bit dequantization.

        Reverses the 8-bit quantization process to reconstruct parameter values.
        Precision: ~0.4% error for typical CAD coordinates (256-level quantization).

        Args:
            command: CAD command type
            param_tokens: Parameter token IDs

        Returns:
            Parameter dictionary with decoded values
        """
        param_specs = get_command_parameters(command)
        parameters = {}

        if len(param_tokens) != len(param_specs):
            # Handle mismatch gracefully
            return {}

        for spec, token in zip(param_specs, param_tokens):
            if spec.type is float:
                # Determine parameter type by name
                param_name = spec.name.lower()

                # Determine offset and range based on parameter type
                if any(x in param_name for x in ["cx", "x1", "x2", "x", "start_x", "end_x"]):
                    # X coordinate
                    offset = self.x_coord_offset
                    min_val = spec.min_value or self.config.min_coordinate
                    max_val = spec.max_value or self.config.max_coordinate

                elif any(y in param_name for y in ["cy", "y1", "y2", "y", "start_y", "end_y"]):
                    # Y coordinate
                    offset = self.y_coord_offset
                    min_val = spec.min_value or self.config.min_coordinate
                    max_val = spec.max_value or self.config.max_coordinate

                elif any(a in param_name for a in ["angle", "rotation", "theta"]):
                    # Angle
                    offset = self.angle_offset
                    min_val = spec.min_value or 0.0
                    max_val = spec.max_value or 360.0

                else:
                    # Distance/radius/size
                    offset = self.distance_offset
                    min_val = spec.min_value or 0.0
                    max_val = spec.max_value or self.config.max_coordinate

                # Extract quantized value (0-255)
                quantized = token - offset
                quantized = max(0, min(255, quantized))  # Safety clamp

                # Dequantize: [0, 255] -> [min_val, max_val]
                normalized = quantized / 255.0
                value = min_val + normalized * (max_val - min_val)
                parameters[spec.name] = float(value)

            elif spec.type is int:
                # Decode integer from distance offset
                quantized = token - self.distance_offset
                parameters[spec.name] = max(0, min(255, int(quantized)))

            elif spec.type is str:
                # For strings, use default values (hash decoding is lossy)
                parameters[spec.name] = spec.default or ""

        return parameters

    def batch_encode(
        self,
        sequences: list[list[tuple[CADCommandType, dict[str, Any]]]],
        padding: bool = True,
        max_length: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Batch encode multiple CAD sequences.

        Args:
            sequences: List of CAD sequences
            padding: Whether to pad sequences
            max_length: Maximum sequence length (uses config default if None)

        Returns:
            Tuple of (token_ids, attention_mask) as MLX arrays
            - token_ids: [batch, seq_len]
            - attention_mask: [batch, seq_len]
        """
        max_length = max_length or self.config.max_sequence_length

        # Encode each sequence
        encoded = [self.encode(seq) for seq in sequences]

        if not padding:
            # Return as list (variable length)
            return encoded

        # Pad to max length
        batch_size = len(encoded)
        token_ids = mx.full((batch_size, max_length), self.pad_token_id, dtype=mx.int32)
        attention_mask = mx.zeros((batch_size, max_length), dtype=mx.int32)

        for i, tokens in enumerate(encoded):
            length = min(len(tokens), max_length)
            token_ids[i, :length] = tokens[:length]
            attention_mask[i, :length] = 1

        return token_ids, attention_mask

    def validate_sequence(
        self, sequence: list[tuple[CADCommandType, dict[str, Any]]]
    ) -> tuple[bool, list[str]]:
        """
        Validate CAD sequence.

        Args:
            sequence: CAD sequence to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for i, (command, parameters) in enumerate(sequence):
            # Validate parameters
            is_valid, error = validate_parameters(command, parameters)
            if not is_valid:
                errors.append(f"Step {i} ({command.name}): {error}")

        return len(errors) == 0, errors
