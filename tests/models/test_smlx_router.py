"""
Tests for Model Inference Router.

This module tests the ModelRouter, ModelCapabilities, and capability detection.
"""

import pytest
from unittest.mock import MagicMock, patch

from smlx.models.smlx_router import (
    CAPABILITY_MAP,
    ModelCapabilities,
    ModelRouter,
    get_router,
)


# ============================================================================
# ModelCapabilities Tests
# ============================================================================


@pytest.mark.unit
def test_model_capabilities_defaults():
    """Test default ModelCapabilities values."""
    caps = ModelCapabilities()

    assert caps.can_chat is False
    assert caps.can_complete is True
    assert caps.can_stream is True
    assert caps.can_transcribe is False
    assert caps.can_caption is False
    assert caps.can_detect is False
    assert caps.requires_image is False
    assert caps.requires_audio is False
    assert caps.max_context_length == 2048
    assert caps.supports_batch is False
    assert caps.modality == "text"
    assert caps.category == "language"
    assert caps.extra_capabilities == {}


@pytest.mark.unit
def test_model_capabilities_custom():
    """Test custom ModelCapabilities."""
    caps = ModelCapabilities(
        can_chat=True,
        can_caption=True,
        requires_image=True,
        max_context_length=8192,
        modality="vision-language",
        category="vision-language",
    )

    assert caps.can_chat is True
    assert caps.can_caption is True
    assert caps.requires_image is True
    assert caps.max_context_length == 8192
    assert caps.modality == "vision-language"
    assert caps.category == "vision-language"


@pytest.mark.unit
def test_model_capabilities_repr():
    """Test ModelCapabilities string representation."""
    caps = ModelCapabilities(
        can_chat=True,
        can_complete=True,
        can_stream=True,
        modality="text",
    )

    repr_str = repr(caps)
    assert "chat" in repr_str
    assert "complete" in repr_str
    assert "stream" in repr_str
    assert "modality=text" in repr_str


@pytest.mark.unit
def test_model_capabilities_extra():
    """Test extra_capabilities field."""
    caps = ModelCapabilities(
        extra_capabilities={
            "can_translate": True,
            "num_languages": 99,
        }
    )

    assert caps.extra_capabilities["can_translate"] is True
    assert caps.extra_capabilities["num_languages"] == 99


# ============================================================================
# CAPABILITY_MAP Tests
# ============================================================================


@pytest.mark.unit
def test_capability_map_exists():
    """Test that CAPABILITY_MAP is defined and not empty."""
    assert CAPABILITY_MAP is not None
    assert len(CAPABILITY_MAP) > 0


@pytest.mark.unit
def test_capability_map_language_models():
    """Test capabilities for language models."""
    # SmolLM2-135M
    caps = CAPABILITY_MAP["smollm2-135m"]
    assert caps.can_chat is True
    assert caps.can_complete is True
    assert caps.can_stream is True
    assert caps.modality == "text"
    assert caps.category == "language"
    assert caps.max_context_length == 8192

    # SmolLM2-360M
    caps = CAPABILITY_MAP["smollm2-360m"]
    assert caps.can_chat is True
    assert caps.can_complete is True
    assert caps.can_stream is True


@pytest.mark.unit
def test_capability_map_vlm_models():
    """Test capabilities for vision-language models."""
    # SmolVLM-256M
    caps = CAPABILITY_MAP["smolvlm-256m"]
    assert caps.can_chat is True
    assert caps.can_caption is True
    assert caps.requires_image is True
    assert caps.modality == "vision-language"
    assert caps.category == "vision-language"

    # Moondream2
    caps = CAPABILITY_MAP["moondream2"]
    assert caps.can_detect is True
    assert caps.extra_capabilities.get("can_point") is True
    assert caps.extra_capabilities.get("can_detect_regions") is True


@pytest.mark.unit
def test_capability_map_audio_models():
    """Test capabilities for audio models."""
    # Whisper-tiny
    caps = CAPABILITY_MAP["whisper-tiny"]
    assert caps.can_transcribe is True
    assert caps.requires_audio is True
    assert caps.modality == "audio"
    assert caps.category == "audio"
    assert caps.extra_capabilities.get("can_translate") is True
    assert caps.extra_capabilities.get("num_languages") == 99

    # YAMNet
    caps = CAPABILITY_MAP["yamnet"]
    assert caps.requires_audio is True
    assert caps.extra_capabilities.get("can_classify_audio") is True

    # Silero VAD
    caps = CAPABILITY_MAP["silero-vad"]
    assert caps.requires_audio is True
    assert caps.extra_capabilities.get("can_detect_voice") is True


@pytest.mark.unit
def test_capability_map_document_models():
    """Test capabilities for document/OCR models."""
    # TrOCR-small
    caps = CAPABILITY_MAP["trocr-small"]
    assert caps.requires_image is True
    assert caps.modality == "vision"
    assert caps.category == "document"
    assert caps.extra_capabilities.get("can_ocr") is True
    assert caps.extra_capabilities.get("supports_handwriting") is True

    # Donut
    caps = CAPABILITY_MAP["donut-base"]
    assert caps.requires_image is True
    assert caps.extra_capabilities.get("can_understand_document") is True
    assert caps.extra_capabilities.get("ocr_free") is True


@pytest.mark.unit
def test_capability_map_embedding_models():
    """Test capabilities for embedding models."""
    # MiniLM
    caps = CAPABILITY_MAP["minilm"]
    assert caps.modality == "text"
    assert caps.category == "embedding"
    assert caps.extra_capabilities.get("can_embed") is True
    assert caps.extra_capabilities.get("embedding_dim") == 384

    # all-MiniLM-L6-v2
    caps = CAPABILITY_MAP["all-minilm-l6-v2"]
    assert caps.extra_capabilities.get("can_embed") is True


@pytest.mark.unit
def test_capability_map_aliases():
    """Test that model aliases work."""
    # SmolLM2 alias
    assert "smollm2" in CAPABILITY_MAP
    assert CAPABILITY_MAP["smollm2"].can_chat is True

    # SmolVLM alias
    assert "smolvlm" in CAPABILITY_MAP
    assert CAPABILITY_MAP["smolvlm"].can_caption is True

    # Whisper alias
    assert "whisper" in CAPABILITY_MAP
    assert CAPABILITY_MAP["whisper"].can_transcribe is True


# ============================================================================
# ModelRouter Tests
# ============================================================================


@pytest.mark.unit
def test_router_initialization():
    """Test ModelRouter initialization."""
    router = ModelRouter()
    assert router is not None
    assert router._function_cache == {}


@pytest.mark.unit
def test_router_get_capabilities():
    """Test getting capabilities by model type."""
    router = ModelRouter()

    # Language model
    caps = router.get_capabilities("smollm2-135m")
    assert caps.can_chat is True
    assert caps.modality == "text"

    # Vision-language model
    caps = router.get_capabilities("smolvlm-256m")
    assert caps.can_caption is True
    assert caps.requires_image is True

    # Audio model
    caps = router.get_capabilities("whisper-tiny")
    assert caps.can_transcribe is True
    assert caps.requires_audio is True


@pytest.mark.unit
def test_router_get_capabilities_with_hf_id():
    """Test getting capabilities with HuggingFace model ID."""
    router = ModelRouter()

    # This should use infer_model_type to extract "smollm2-135m"
    with patch("smlx.models.smlx_router.infer_model_type") as mock_infer:
        mock_infer.return_value = "smollm2-135m"
        caps = router.get_capabilities("mlx-community/SmolLM2-135M-Instruct")
        assert caps.can_chat is True


@pytest.mark.unit
def test_router_get_capabilities_unknown_model():
    """Test getting capabilities for unknown model."""
    router = ModelRouter()

    with pytest.raises(ValueError, match="Unknown model type"):
        router.get_capabilities("unknown-model-xyz")


@pytest.mark.unit
def test_router_can_handle():
    """Test checking if model has specific capability."""
    router = ModelRouter()

    # SmolLM2 can chat
    assert router.can_handle("smollm2-135m", "can_chat") is True

    # SmolLM2 cannot transcribe
    assert router.can_handle("smollm2-135m", "can_transcribe") is False

    # Whisper can transcribe
    assert router.can_handle("whisper-tiny", "can_transcribe") is True

    # SmolVLM can caption
    assert router.can_handle("smolvlm-256m", "can_caption") is True

    # Unknown model
    assert router.can_handle("unknown-model", "can_chat") is False


@pytest.mark.unit
def test_router_can_handle_invalid_capability():
    """Test checking for non-existent capability."""
    router = ModelRouter()

    # Non-existent capability should return False
    assert router.can_handle("smollm2-135m", "can_fly") is False


@pytest.mark.unit
def test_router_route_text_generation():
    """Test routing text generation request."""
    router = ModelRouter()

    # Mock the model module and generate function
    mock_module = MagicMock()
    mock_module.generate = MagicMock(return_value="Generated text")

    with patch.object(router, "_get_generation_module", return_value=mock_module):
        result = router.route_text_generation(
            model_type="smollm2-135m",
            model=MagicMock(),
            tokenizer=MagicMock(),
            prompt="Hello",
            max_tokens=50,
        )

        assert result == "Generated text"
        mock_module.generate.assert_called_once()


@pytest.mark.unit
def test_router_route_text_generation_unsupported():
    """Test routing text generation for unsupported model."""
    router = ModelRouter()

    # TrOCR doesn't support text completion (it's a vision model)
    # This should fail at the capability check, not at module import
    with pytest.raises(ValueError, match="does not support text completion"):
        router.route_text_generation(
            model_type="trocr-small",
            model=MagicMock(),
            tokenizer=MagicMock(),
            prompt="Hello",
        )


@pytest.mark.unit
def test_router_route_chat():
    """Test routing chat request."""
    router = ModelRouter()

    mock_module = MagicMock()
    mock_module.chat = MagicMock(return_value="Chat response")

    with patch.object(router, "_get_generation_module", return_value=mock_module):
        result = router.route_chat(
            model_type="smollm2-135m",
            model=MagicMock(),
            tokenizer=MagicMock(),
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Chat response"
        mock_module.chat.assert_called_once()


@pytest.mark.unit
def test_router_route_chat_unsupported():
    """Test routing chat for unsupported model."""
    router = ModelRouter()

    # Whisper doesn't support chat
    with pytest.raises(ValueError, match="does not support chat"):
        router.route_chat(
            model_type="whisper-tiny",
            model=MagicMock(),
            tokenizer=MagicMock(),
            messages=[{"role": "user", "content": "Hello"}],
        )


@pytest.mark.unit
def test_router_route_streaming_generation():
    """Test routing streaming generation request."""
    router = ModelRouter()

    def mock_stream():
        yield "Hello"
        yield " "
        yield "world"

    mock_module = MagicMock()
    mock_module.stream_generate = MagicMock(return_value=mock_stream())

    with patch.object(router, "_get_generation_module", return_value=mock_module):
        result = list(
            router.route_streaming_generation(
                model_type="smollm2-135m",
                model=MagicMock(),
                tokenizer=MagicMock(),
                prompt="Hello",
            )
        )

        assert result == ["Hello", " ", "world"]


@pytest.mark.unit
def test_router_route_streaming_unsupported():
    """Test routing streaming for unsupported model."""
    router = ModelRouter()

    # Whisper doesn't support streaming (audio transcription model)
    # This should fail at the capability check
    with pytest.raises(ValueError, match="does not support streaming"):
        list(
            router.route_streaming_generation(
                model_type="whisper-tiny",
                model=MagicMock(),
                tokenizer=MagicMock(),
                prompt="Hello",
            )
        )


@pytest.mark.unit
def test_router_route_multimodal():
    """Test routing multimodal generation request."""
    router = ModelRouter()

    mock_module = MagicMock()
    mock_module.generate = MagicMock(return_value="Image description")

    with patch.object(router, "_get_generation_module", return_value=mock_module):
        result = router.route_multimodal(
            model_type="smolvlm-256m",
            model=MagicMock(),
            processor=MagicMock(),
            prompt="Describe:",
            image="image.jpg",
        )

        assert result == "Image description"
        mock_module.generate.assert_called_once()


@pytest.mark.unit
def test_router_route_multimodal_missing_image():
    """Test routing multimodal without required image."""
    router = ModelRouter()

    with pytest.raises(ValueError, match="requires image input"):
        router.route_multimodal(
            model_type="smolvlm-256m",
            model=MagicMock(),
            processor=MagicMock(),
            prompt="Describe:",
            image=None,  # Missing required image
        )


@pytest.mark.unit
def test_router_route_multimodal_unsupported():
    """Test routing multimodal for unsupported model."""
    router = ModelRouter()

    # Whisper is audio-only, doesn't support multimodal generation
    with pytest.raises(ValueError, match="does not support multimodal generation"):
        router.route_multimodal(
            model_type="whisper-tiny",  # Audio-only model
            model=MagicMock(),
            processor=MagicMock(),
            prompt="Describe:",
            image="image.jpg",
        )


@pytest.mark.unit
def test_router_route_transcription():
    """Test routing audio transcription request."""
    router = ModelRouter()

    mock_module = MagicMock()
    mock_module.transcribe = MagicMock(
        return_value={"text": "Transcribed text", "language": "en"}
    )

    with patch.object(router, "_get_generation_module", return_value=mock_module):
        result = router.route_transcription(
            model_type="whisper-tiny",
            model=MagicMock(),
            tokenizer=MagicMock(),
            audio="audio.wav",
            language="en",
        )

        assert result["text"] == "Transcribed text"
        mock_module.transcribe.assert_called_once()


@pytest.mark.unit
def test_router_route_transcription_unsupported():
    """Test routing transcription for unsupported model."""
    router = ModelRouter()

    with pytest.raises(ValueError, match="does not support transcription"):
        router.route_transcription(
            model_type="smollm2-135m",  # Language model, not audio
            model=MagicMock(),
            tokenizer=MagicMock(),
            audio="audio.wav",
        )


@pytest.mark.unit
def test_router_get_generation_module_invalid():
    """Test getting generation module for invalid model type."""
    router = ModelRouter()

    with pytest.raises(ValueError, match="not in registry"):
        router._get_generation_module("unknown-model-xyz")


# ============================================================================
# Singleton Tests
# ============================================================================


@pytest.mark.unit
def test_get_router_singleton():
    """Test that get_router returns singleton instance."""
    router1 = get_router()
    router2 = get_router()

    assert router1 is router2
    assert isinstance(router1, ModelRouter)


@pytest.mark.unit
def test_get_router_multiple_calls():
    """Test multiple calls to get_router return same instance."""
    routers = [get_router() for _ in range(5)]

    # All should be the same instance
    for router in routers[1:]:
        assert router is routers[0]
