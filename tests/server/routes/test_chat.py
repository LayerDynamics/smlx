# Copyright © 2025 SMLX Project

"""
Tests for chat completion endpoints.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.dependencies import get_model_manager
from smlx.server.routes.chat import router
from smlx.server.schemas import Message, Role


@pytest.fixture
def app():
    """Create test app with chat router."""
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return app


@pytest.fixture
def mock_manager():
    """Create mock model manager."""
    manager = Mock()
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(side_effect=lambda x: [1] * len(x.split()))
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
class TestCreateChatCompletion:
    """Tests for create chat completion endpoint."""

    @patch("smlx.server.routes.chat.generate_chat_response")
    def test_simple_chat_completion(self, mock_generate, client):
        """Test simple chat completion."""
        mock_generate.return_value = "Hello! How can I help you?"

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "SmolLM2-135M",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data

    @patch("smlx.server.routes.chat.generate_chat_response")
    def test_chat_with_system_message(self, mock_generate, client):
        """Test chat with system message."""
        mock_generate.return_value = "I'm a helpful assistant!"

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "SmolLM2-135M",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Who are you?"},
                ],
                "max_tokens": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"

    def test_chat_validation_error(self, client):
        """Test chat with invalid parameters."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "SmolLM2-135M"},  # Missing required 'messages'
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.unit
class TestFormatMessages:
    """Tests for message formatting."""

    def test_format_messages_as_prompt(self):
        """Test formatting messages as prompt."""
        from smlx.server.routes.chat import format_messages_as_prompt

        messages = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hello"),
        ]

        prompt = format_messages_as_prompt(messages)

        assert "System: You are helpful" in prompt
        assert "User: Hello" in prompt
        assert "Assistant:" in prompt
