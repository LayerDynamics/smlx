# Copyright © 2025 SMLX Project

"""
Tests for audio transcription endpoints.
"""

from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.dependencies import get_model_manager
from smlx.server.routes.audio import router


@pytest.fixture
def app():
    """Create test app with audio router."""
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return app


@pytest.fixture
def mock_manager():
    """Create mock model manager."""
    manager = Mock()
    mock_model = Mock()
    mock_tokenizer = Mock()
    manager.load_model = AsyncMock(return_value=(mock_model, mock_tokenizer))
    return manager


@pytest.fixture
def client(app, mock_manager):
    """Create test client with dependency override."""
    app.dependency_overrides[get_model_manager] = lambda: mock_manager
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.mark.integration
class TestCreateTranscription:
    """Tests for create transcription endpoint."""

    @patch("smlx.server.routes.audio.transcribe_audio")
    def test_simple_transcription(self, mock_transcribe, client):
        """Test simple audio transcription."""
        mock_transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "duration": 2.5,
            "segments": None,
        }

        # Create fake audio file
        audio_data = b"fake audio data"

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", BytesIO(audio_data), "audio/wav")},
            data={"model": "whisper-tiny"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world"
        assert data["language"] == "en"
        assert data["duration"] == 2.5

    def test_transcription_invalid_model(self, client):
        """Test transcription with non-Whisper model."""
        audio_data = b"fake audio data"

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", BytesIO(audio_data), "audio/wav")},
            data={"model": "not-a-whisper-model"},
        )

        # Should fail validation or raise error
        assert response.status_code in [400, 500]

    def test_transcription_missing_file(self, client):
        """Test transcription without audio file."""
        response = client.post("/v1/audio/transcriptions", data={"model": "whisper-tiny"})

        assert response.status_code == 422  # Validation error
