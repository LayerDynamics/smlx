# Copyright © 2025 SMLX Project

"""
Tests for embeddings endpoints.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.dependencies import get_model_manager
from smlx.server.routes.embeddings import router


@pytest.fixture
def app():
    """Create test app with embeddings router."""
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return app


@pytest.fixture
def mock_manager():
    """Create mock model manager."""
    manager = Mock()
    manager.load_model = AsyncMock(
        side_effect=NotImplementedError("Embedding models not yet implemented")
    )
    return manager


@pytest.fixture
def client(app, mock_manager):
    """Create test client with dependency override."""
    app.dependency_overrides[get_model_manager] = lambda: mock_manager
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.mark.integration
class TestCreateEmbeddings:
    """Tests for create embeddings endpoint."""

    def test_embeddings_not_implemented(self, app, mock_manager):
        """Test that embeddings return not implemented error."""
        # Override dependency
        app.dependency_overrides[get_model_manager] = lambda: mock_manager
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings", json={"model": "all-MiniLM-L6-v2", "input": "Hello world"}
        )

        assert response.status_code == 501
        data = response.json()
        assert "not yet implemented" in data["detail"].lower()

        app.dependency_overrides.clear()

    def test_embeddings_validation_error(self, app, mock_manager):
        """Test embeddings with invalid parameters."""
        # Override dependency even though we won't reach it
        app.dependency_overrides[get_model_manager] = lambda: mock_manager
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"model": "all-MiniLM-L6-v2"},  # Missing required 'input'
        )

        assert response.status_code == 422  # Validation error

        app.dependency_overrides.clear()
