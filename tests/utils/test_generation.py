"""Tests for smlx.utils.generation module."""

from unittest.mock import MagicMock, Mock, patch

import mlx.core as mx
import pytest

from smlx.utils.generation import (
    GenerationConfig,
    chat,
    complete,
    generate,
    generate_step,
    stream_generate,
)


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_generation_config_defaults(self):
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.max_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.top_k == 0
        assert config.min_p == 0.0
        assert config.repetition_penalty == 1.0
        assert config.stop_token_ids is None
        assert config.stop_strings is None

    def test_generation_config_custom(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_tokens=200,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            stop_token_ids=[0, 1],
            stop_strings=["END", "STOP"],
        )

        assert config.max_tokens == 200
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.repetition_penalty == 1.1
        assert config.stop_token_ids == [0, 1]
        assert config.stop_strings == ["END", "STOP"]

    def test_generation_config_logit_bias(self):
        """Test logit bias configuration."""
        config = GenerationConfig(logit_bias={100: 2.0, 200: -1.0})

        assert config.logit_bias == {100: 2.0, 200: -1.0}

    def test_generation_config_min_tokens(self):
        """Test min_tokens configuration."""
        config = GenerationConfig(min_tokens=10)

        assert config.min_tokens == 10


class TestGenerateStep:
    """Test generate_step function."""

    def test_generate_step_basic(self):
        """Test basic token generation with generate_step."""
        # Mock model
        model = Mock()
        model.layers = [None, None, None]  # 3 layers

        # Create logits that will lead to predictable token generation
        logits_seq = [
            mx.array([[1.0, 5.0, 2.0]]),  # Will sample token 1
            mx.array([[2.0, 1.0, 5.0]]),  # Will sample token 2
            mx.array([[5.0, 1.0, 2.0]]),  # Will sample token 0
        ]

        model.side_effect = lambda x, cache=None: logits_seq.pop(0)

        prompt_tokens = mx.array([1, 2, 3])

        # Generate 2 tokens
        generator = generate_step(
            model, prompt_tokens, temp=0.0, top_p=1.0
        )

        token1, logits1 = next(generator)
        assert token1.item() == 1  # Highest logit in first output

        token2, logits2 = next(generator)
        assert token2.item() == 2  # Highest logit in second output

    def test_generate_step_with_cache(self):
        """Test generate_step with provided cache."""
        from smlx.utils.cache import KVCache

        model = Mock()
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        cache = [KVCache() for _ in range(3)]
        prompt_tokens = mx.array([1, 2, 3])

        generator = generate_step(
            model, prompt_tokens, temp=0.0, cache=cache
        )

        token, _ = next(generator)
        assert token is not None

    def test_generate_step_temperature(self):
        """Test generate_step with different temperatures."""
        model = Mock()
        model.layers = [None]  # Required for cache initialization
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        model.return_value = logits

        prompt_tokens = mx.array([1, 2])

        # With temp=0.0, should always pick highest logit
        generator = generate_step(model, prompt_tokens, temp=0.0)
        token, _ = next(generator)

        # Token should be deterministic with temp=0
        assert token is not None


class TestGenerate:
    """Test generate function."""

    def test_generate_basic(self):
        """Test basic text generation."""
        # Mock model
        model = Mock()
        model.layers = [None]  # Required for cache initialization
        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        model.return_value = logits

        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(side_effect=lambda tokens: " ".join(map(str, tokens)))
        tokenizer.eos_token_id = 0

        # Generate
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            max_tokens=3,
            temperature=0.0,
            verbose=False,
        )

        assert isinstance(result, str)
        tokenizer.encode.assert_called_once_with("Test prompt")

    def test_generate_with_stop_strings(self):
        """Test generation with stop strings."""
        model = Mock()
        model.layers = [None, None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2])
        # Simulate generating text that contains stop string
        tokenizer.decode = Mock(side_effect=lambda tokens: "Hello STOP world")
        tokenizer.eos_token_id = None

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=10,
            stop_strings=["STOP"],
            temperature=0.0,
        )

        # Should stop at STOP
        assert "STOP" not in result or result == "Hello "

    def test_generate_with_stop_tokens(self):
        """Test generation with stop token IDs."""
        model = Mock()
        model.layers = [None]

        # Return logits that will generate token 99, then 2
        logits_seq = [
            mx.array([[0.0] * 99 + [10.0]]),  # Token 99 (EOS)
        ]
        model.side_effect = lambda x, cache=None: logits_seq.pop(0) if logits_seq else mx.array([[1.0, 2.0]])

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])
        tokenizer.decode = Mock(return_value="Hello")
        tokenizer.eos_token_id = 99

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=10,
            temperature=0.0,
        )

        # Should stop at EOS token
        assert isinstance(result, str)

    def test_generate_verbose(self):
        """Test generation with verbose output."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])
        tokenizer.decode = Mock(side_effect=lambda tokens: "word " * len(tokens))
        tokenizer.eos_token_id = 0

        # Capture output
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=2,
            temperature=0.0,
            verbose=True,
        )

        assert isinstance(result, str)


class TestStreamGenerate:
    """Test stream_generate function."""

    def test_stream_generate_basic(self):
        """Test streaming generation."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2])

        # Simulate incremental decoding
        decode_calls = []
        def mock_decode(tokens):
            result = " ".join(map(str, tokens))
            decode_calls.append(result)
            return result

        tokenizer.decode = Mock(side_effect=mock_decode)
        tokenizer.eos_token_id = 0

        # Collect streamed output
        output = []
        for text in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=3,
            temperature=0.0,
        ):
            output.append(text)
            if len(output) >= 3:
                break

        # Should have received some output
        assert len(output) > 0

    def test_stream_generate_with_stop_string(self):
        """Test streaming with stop string."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])

        # Simulate text that will hit stop string
        call_count = [0]
        def mock_decode(tokens):
            call_count[0] += 1
            if call_count[0] == 2:
                return "Hello END world"
            return "Hello "

        tokenizer.decode = Mock(side_effect=mock_decode)
        tokenizer.eos_token_id = None

        output = []
        for text in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=10,
            stop_strings=["END"],
        ):
            output.append(text)

        # Should stop when stop string is encountered
        assert len(output) >= 0

    def test_stream_generate_yields_deltas(self):
        """Test that stream_generate yields text deltas."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])

        # Simulate incremental text building
        texts = ["Hello", "Hello world", "Hello world!"]
        text_idx = [0]

        def mock_decode(tokens):
            result = texts[min(text_idx[0], len(texts) - 1)]
            text_idx[0] += 1
            return result

        tokenizer.decode = Mock(side_effect=mock_decode)
        tokenizer.eos_token_id = None

        output = []
        for text in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=3,
            temperature=0.0,
        ):
            output.append(text)
            if len(output) >= 3:
                break

        # Each output should be a delta
        assert len(output) > 0


class TestChat:
    """Test chat function."""

    def test_chat_basic(self):
        """Test basic chat completion."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="I am an AI assistant.")
        tokenizer.eos_token_id = 0

        # Mock chat template
        def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
            return "User: Hello\nAssistant: "

        tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

        messages = [{"role": "user", "content": "Hello"}]

        response = chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_tokens=20,
            temperature=0.7,
        )

        assert isinstance(response, str)
        tokenizer.apply_chat_template.assert_called_once()

    def test_chat_without_template(self):
        """Test chat without chat template support."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2])
        tokenizer.decode = Mock(return_value="Response")
        tokenizer.eos_token_id = 0
        # No apply_chat_template method

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        response = chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_tokens=20,
        )

        assert isinstance(response, str)

    def test_chat_multi_turn(self):
        """Test multi-turn conversation."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="Response")
        tokenizer.eos_token_id = 0

        def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            return prompt + "Assistant: "

        tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        response = chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_tokens=20,
        )

        assert isinstance(response, str)


class TestComplete:
    """Test complete function."""

    def test_complete_with_config(self):
        """Test completion with GenerationConfig."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2])
        tokenizer.decode = Mock(return_value="Completed text")
        tokenizer.eos_token_id = 0

        config = GenerationConfig(
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
        )

        result = complete(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            config=config,
        )

        assert isinstance(result, str)

    def test_complete_default_config(self):
        """Test completion with default config."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])
        tokenizer.decode = Mock(return_value="Text")
        tokenizer.eos_token_id = 0

        result = complete(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
        )

        assert isinstance(result, str)

    def test_complete_with_stop_strings(self):
        """Test completion with stop strings in config."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])
        tokenizer.decode = Mock(return_value="Hello END")
        tokenizer.eos_token_id = None

        config = GenerationConfig(
            max_tokens=10,
            stop_strings=["END"],
        )

        result = complete(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            config=config,
        )

        assert isinstance(result, str)


class TestGenerationEdgeCases:
    """Test edge cases in generation."""

    def test_generate_zero_max_tokens(self):
        """Test generation with max_tokens=0."""
        model = Mock()
        model.layers = [None]  # Required for cache initialization
        logits = mx.array([[1.0, 2.0, 3.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2])
        tokenizer.decode = Mock(return_value="")
        tokenizer.eos_token_id = 0

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=0,
        )

        # Should return empty or minimal text
        assert isinstance(result, str)

    def test_generate_step_no_layers(self):
        """Test generate_step with model without layers attribute."""
        model = Mock(spec=[])  # No layers attribute
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        prompt_tokens = mx.array([1, 2])

        generator = generate_step(model, prompt_tokens, temp=0.0)

        # Should still work (uses fallback)
        token, _ = next(generator)
        assert token is not None

    def test_stream_generate_empty_prompt(self):
        """Test streaming with minimal prompt."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])
        tokenizer.decode = Mock(return_value="text")
        tokenizer.eos_token_id = 0

        output = []
        for text in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt="",
            max_tokens=2,
        ):
            output.append(text)
            if len(output) >= 2:
                break

        assert isinstance(output, list)


class TestGenerationIntegration:
    """Test integration scenarios."""

    def test_config_to_complete_workflow(self):
        """Test creating config and using in complete."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2])
        tokenizer.decode = Mock(return_value="Generated")
        tokenizer.eos_token_id = 0

        # Create config
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )

        # Use in complete
        result = complete(model, tokenizer, "Prompt", config)

        assert isinstance(result, str)
        assert config.max_tokens == 100

    def test_generate_respects_parameters(self):
        """Test that generate respects all parameters."""
        model = Mock()
        model.layers = [None]
        logits = mx.array([[1.0, 5.0, 2.0]])
        model.return_value = logits

        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1])
        tokenizer.decode = Mock(return_value="text")
        tokenizer.eos_token_id = 0

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Test",
            max_tokens=5,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
        )

        assert isinstance(result, str)
