# Copyright © 2025 SMLX Project

"""
Tests for text completion endpoints.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.dependencies import get_model_manager
from smlx.server.routes.completions import router


@pytest.fixture
def app():
    """Create test app with completions router."""
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
class TestCreateCompletion:
    """Tests for create completion endpoint."""

    @patch("smlx.server.routes.completions.generate_text")
    def test_simple_completion(self, mock_generate, client):
        """Test simple text completion."""
        mock_generate.return_value = "Generated text response"

        response = client.post(
            "/v1/completions",
            json={"model": "SmolLM2-135M", "prompt": "Hello", "max_tokens": 50},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "usage" in data

    @patch("smlx.server.routes.completions.generate_text")
    def test_completion_with_echo(self, mock_generate, client):
        """Test completion with echo parameter."""
        mock_generate.return_value = "response"

        response = client.post(
            "/v1/completions",
            json={
                "model": "SmolLM2-135M",
                "prompt": "Hello",
                "max_tokens": 50,
                "echo": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # When echo=True, prompt should be included in response
        assert "choices" in data

    def test_completion_validation_error(self, client):
        """Test completion with invalid parameters."""
        response = client.post(
            "/v1/completions",
            json={"model": "SmolLM2-135M"},  # Missing required 'prompt'
        )

        assert response.status_code == 422  # Validation error
