# Copyright © 2025 SMLX Project

"""
Tests for models listing endpoints.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.dependencies import get_model_manager
from smlx.server.routes.models import router


@pytest.fixture
def app():
    """Create test app with models router."""
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return app


@pytest.fixture
def mock_manager():
    """Create mock model manager."""
    manager = Mock()
    manager.list_supported_models.return_value = {
        "mlx-community/SmolLM2-135M-Instruct": "smollm",
        "whisper-tiny": "whisper",
    }
    manager.unload_model = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def client(app, mock_manager):
    """Create test client with dependency override."""
    # Override the get_model_manager dependency
    app.dependency_overrides[get_model_manager] = lambda: mock_manager
    client = TestClient(app)
    yield client
    # Clean up after test
    app.dependency_overrides.clear()


@pytest.mark.unit
class TestListModels:
    """Tests for list models endpoint."""

    def test_list_models(self, client, mock_manager):
        """Test listing all supported models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 2
        assert all(m["object"] == "model" for m in data["data"])
        assert all(m["owned_by"] == "smlx" for m in data["data"])


@pytest.mark.unit
class TestGetModel:
    """Tests for get model endpoint."""

    def test_get_existing_model(self, client, mock_manager):
        """Test getting info about an existing model."""
        response = client.get("/v1/models/mlx-community/SmolLM2-135M-Instruct")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mlx-community/SmolLM2-135M-Instruct"
        assert data["object"] == "model"
        assert data["owned_by"] == "smlx"

    def test_get_nonexistent_model(self, app):
        """Test getting info about a nonexistent model."""
        # Create a separate mock manager with no models
        empty_manager = Mock()
        empty_manager.list_supported_models.return_value = {}

        # Create a new client with the empty manager override
        app.dependency_overrides[get_model_manager] = lambda: empty_manager
        client = TestClient(app)

        response = client.get("/v1/models/nonexistent-model")

        assert response.status_code == 404

        # Clean up
        app.dependency_overrides.clear()


@pytest.mark.unit
class TestUnloadModel:
    """Tests for unload model endpoint."""

    def test_unload_loaded_model(self, client, mock_manager):
        """Test unloading a loaded model."""
        mock_manager.unload_model = AsyncMock(return_value=True)

        response = client.delete("/v1/models/test-model")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_unload_not_loaded_model(self, client, mock_manager):
        """Test unloading a model that isn't loaded."""
        mock_manager.unload_model = AsyncMock(return_value=False)

        response = client.delete("/v1/models/nonexistent-model")

        assert response.status_code == 404
