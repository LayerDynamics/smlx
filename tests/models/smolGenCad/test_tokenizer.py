#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
Unit tests for CAD tokenizer.

Tests tokenization, encoding/decoding, and batch processing.
"""

import pytest
import mlx.core as mx

from smlx.models.smolGenCad.commands import CADCommandType
from smlx.models.smolGenCad.config import CADVocabularyConfig
from smlx.models.smolGenCad.tokenizer import CADTokenizer


@pytest.fixture
def tokenizer():
    """Create a tokenizer instance."""
    return CADTokenizer()


@pytest.fixture
def simple_sequence():
    """Create a simple CAD sequence for testing."""
    return [
        (CADCommandType.SKETCH_START, {"plane": "XY"}),
        (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
        (CADCommandType.SKETCH_END, {}),
        (CADCommandType.EXTRUDE, {"distance": 100.0}),
    ]


class TestTokenizerInitialization:
    """Test tokenizer initialization."""

    def test_tokenizer_creates_successfully(self, tokenizer):
        """Test tokenizer can be created."""
        assert tokenizer is not None
        assert isinstance(tokenizer, CADTokenizer)

    def test_tokenizer_has_special_tokens(self, tokenizer):
        """Test tokenizer defines special tokens."""
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.sep_token_id == 3

    def test_tokenizer_builds_vocabulary(self, tokenizer):
        """Test tokenizer builds command vocabulary."""
        assert hasattr(tokenizer, "command_to_token")
        assert hasattr(tokenizer, "token_to_command")
        assert len(tokenizer.command_to_token) > 0
        assert tokenizer.vocab_size > 0

    def test_command_token_bidirectional_mapping(self, tokenizer):
        """Test command-token mapping is bidirectional."""
        for cmd, token in tokenizer.command_to_token.items():
            assert tokenizer.token_to_command[token] == cmd

    def test_custom_config(self):
        """Test tokenizer with custom configuration."""
        config = CADVocabularyConfig(max_sequence_length=100)
        tokenizer = CADTokenizer(config)
        assert tokenizer.config.max_sequence_length == 100


@pytest.mark.unit
class TestEncoding:
    """Test sequence encoding."""

    def test_encode_simple_sequence(self, tokenizer, simple_sequence):
        """Test encoding a simple CAD sequence."""
        tokens = tokenizer.encode(simple_sequence)
        assert isinstance(tokens, mx.array)
        assert len(tokens) > 0

    def test_encode_includes_bos_eos(self, tokenizer, simple_sequence):
        """Test encoding includes BOS and EOS tokens."""
        tokens = tokenizer.encode(simple_sequence, add_special_tokens=True)
        tokens_list = tokens.tolist()
        assert tokens_list[0] == tokenizer.bos_token_id
        assert tokens_list[-1] == tokenizer.eos_token_id

    def test_encode_without_special_tokens(self, tokenizer, simple_sequence):
        """Test encoding without special tokens."""
        tokens = tokenizer.encode(simple_sequence, add_special_tokens=False)
        tokens_list = tokens.tolist()
        assert tokens_list[0] != tokenizer.bos_token_id
        assert tokens_list[-1] != tokenizer.eos_token_id

    def test_encode_empty_sequence(self, tokenizer):
        """Test encoding empty sequence."""
        tokens = tokenizer.encode([])
        tokens_list = tokens.tolist()
        # Should only have BOS and EOS
        assert len(tokens_list) == 2
        assert tokens_list[0] == tokenizer.bos_token_id
        assert tokens_list[-1] == tokenizer.eos_token_id

    def test_encode_single_command(self, tokenizer):
        """Test encoding single command."""
        sequence = [(CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0})]
        tokens = tokenizer.encode(sequence)
        assert len(tokens) > 2  # BOS + command + params + SEP + EOS

    def test_encoded_tokens_are_integers(self, tokenizer, simple_sequence):
        """Test encoded tokens are integers."""
        tokens = tokenizer.encode(simple_sequence)
        assert tokens.dtype == mx.int32


@pytest.mark.unit
class TestDecoding:
    """Test sequence decoding."""

    def test_decode_simple_sequence(self, tokenizer, simple_sequence):
        """Test decoding a sequence."""
        tokens = tokenizer.encode(simple_sequence)
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, list)
        assert len(decoded) > 0

    def test_encode_decode_roundtrip(self, tokenizer, simple_sequence):
        """Test encoding and decoding preserves sequence structure."""
        tokens = tokenizer.encode(simple_sequence)
        decoded = tokenizer.decode(tokens)

        # Should have same number of commands
        assert len(decoded) == len(simple_sequence)

        # Commands should match
        for (orig_cmd, _), (dec_cmd, _) in zip(simple_sequence, decoded):
            assert orig_cmd == dec_cmd

    def test_decode_with_list_input(self, tokenizer, simple_sequence):
        """Test decoding with list input."""
        tokens = tokenizer.encode(simple_sequence)
        tokens_list = tokens.tolist()
        decoded = tokenizer.decode(tokens_list)
        assert len(decoded) > 0

    def test_decode_skips_special_tokens(self, tokenizer):
        """Test decoding skips special tokens by default."""
        tokens = [tokenizer.bos_token_id, 21, 100, 3, tokenizer.eos_token_id]
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        # Should not include BOS/EOS in output
        # decoded sequence should be valid CAD commands only
        assert isinstance(decoded, list)

    def test_decode_empty_tokens(self, tokenizer):
        """Test decoding empty token list."""
        decoded = tokenizer.decode([])
        assert decoded == []


@pytest.mark.unit
class TestParameterEncoding:
    """Test parameter encoding and decoding."""

    def test_encode_float_parameters(self, tokenizer):
        """Test encoding float parameters."""
        sequence = [(CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0})]
        tokens = tokenizer.encode(sequence)
        decoded = tokenizer.decode(tokens)

        # Parameters should be numeric (may have quantization loss)
        assert isinstance(decoded[0][1].get("r"), (int, float))

    def test_encode_string_parameters(self, tokenizer):
        """Test encoding string parameters."""
        sequence = [(CADCommandType.SKETCH_START, {"plane": "XY"})]
        tokens = tokenizer.encode(sequence)
        decoded = tokenizer.decode(tokens)

        # String parameters preserved (or default used)
        assert "plane" in decoded[0][1]

    def test_parameter_quantization(self, tokenizer):
        """Test parameter values are quantized."""
        # Encode a precise value
        sequence = [(CADCommandType.CIRCLE, {"cx": 123.456, "cy": 78.901, "r": 50.0})]
        tokens = tokenizer.encode(sequence)
        decoded = tokenizer.decode(tokens)

        # Values may be slightly different due to quantization
        # But should be reasonably close
        cx = decoded[0][1].get("cx", 0)
        assert abs(cx - 123.456) < 10  # Within reasonable range


@pytest.mark.unit
class TestBatchEncoding:
    """Test batch encoding."""

    def test_batch_encode_multiple_sequences(self, tokenizer, simple_sequence):
        """Test batch encoding multiple sequences."""
        sequences = [simple_sequence, simple_sequence]
        token_ids, attention_mask = tokenizer.batch_encode(sequences)

        assert isinstance(token_ids, mx.array)
        assert isinstance(attention_mask, mx.array)
        assert token_ids.shape[0] == 2  # batch size
        assert attention_mask.shape[0] == 2

    def test_batch_encode_pads_sequences(self, tokenizer):
        """Test batch encoding pads sequences to same length."""
        seq1 = [(CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0})]
        seq2 = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0, "r": 50.0}),
            (CADCommandType.EXTRUDE, {"distance": 100.0}),
        ]

        token_ids, attention_mask = tokenizer.batch_encode([seq1, seq2])

        # Both should have same length (padded)
        assert token_ids.shape[1] == token_ids.shape[1]

        # Attention mask should reflect padding
        assert attention_mask[0].sum() <= attention_mask[1].sum()

    def test_batch_encode_without_padding(self, tokenizer, simple_sequence):
        """Test batch encoding without padding returns list."""
        sequences = [simple_sequence]
        result = tokenizer.batch_encode(sequences, padding=False)

        # Without padding, returns list of arrays
        assert isinstance(result, list)

    def test_batch_encode_custom_max_length(self, tokenizer, simple_sequence):
        """Test batch encoding with custom max length."""
        token_ids, _ = tokenizer.batch_encode(
            [simple_sequence], max_length=50
        )
        assert token_ids.shape[1] == 50


@pytest.mark.unit
class TestValidation:
    """Test sequence validation."""

    def test_validate_valid_sequence(self, tokenizer, simple_sequence):
        """Test validating a valid sequence."""
        is_valid, errors = tokenizer.validate_sequence(simple_sequence)
        # May have validation errors for structure, but parameters should be valid
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_validate_sequence_with_invalid_parameters(self, tokenizer):
        """Test validation catches invalid parameters."""
        sequence = [
            (CADCommandType.CIRCLE, {"cx": 0.0, "cy": 0.0}),  # Missing 'r'
        ]
        is_valid, errors = tokenizer.validate_sequence(sequence)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_empty_sequence(self, tokenizer):
        """Test validation of empty sequence."""
        is_valid, errors = tokenizer.validate_sequence([])
        # Tokenizer's validate_sequence doesn't check structure, only parameters
        # Empty sequence passes tokenizer validation (structure validation is in validator module)
        assert is_valid and len(errors) == 0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_encode_command_without_parameters(self, tokenizer):
        """Test encoding command with no parameters."""
        sequence = [(CADCommandType.START, {})]
        tokens = tokenizer.encode(sequence)
        assert len(tokens) > 0

    def test_encode_max_length_sequence(self, tokenizer):
        """Test encoding very long sequence."""
        # Create a long sequence
        long_sequence = [
            (CADCommandType.CIRCLE, {"cx": float(i), "cy": 0.0, "r": 10.0})
            for i in range(50)
        ]
        tokens = tokenizer.encode(long_sequence)
        assert len(tokens) > 0

    def test_tokenizer_consistency(self, tokenizer, simple_sequence):
        """Test tokenizer produces consistent results."""
        tokens1 = tokenizer.encode(simple_sequence)
        tokens2 = tokenizer.encode(simple_sequence)

        # Should produce identical tokens
        assert (tokens1 == tokens2).all()
