# Copyright © 2025 SMLX Project

"""
Tests for FastAPI application.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from smlx.server.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.unit
class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns server info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "SMLX Server"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"
        assert data["docs"] == "/docs"


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @patch("smlx.server.app.model_manager")
    def test_health_no_models(self, mock_manager, client):
        """Test health endpoint with no models loaded."""
        mock_manager.loaded_models = {}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["models_loaded"] == 0

    @patch("smlx.server.app.model_manager")
    def test_health_with_models(self, mock_manager, client):
        """Test health endpoint with models loaded."""
        mock_manager.loaded_models = {"model1": Mock(), "model2": Mock()}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] == 2


@pytest.mark.unit
class TestGetModelManager:
    """Tests for get_model_manager helper."""

    @patch("smlx.server.app.model_manager", None)
    def test_get_model_manager_not_initialized(self):
        """Test get_model_manager raises error when not initialized."""
        from fastapi import HTTPException

        from smlx.server.app import get_model_manager

        with pytest.raises(HTTPException) as exc_info:
            get_model_manager()

        assert exc_info.value.status_code == 500
        assert "not initialized" in exc_info.value.detail

    @patch("smlx.server.app.model_manager")
    def test_get_model_manager_success(self, mock_manager):
        """Test get_model_manager returns manager when initialized."""
        from smlx.server.app import get_model_manager

        result = get_model_manager()

        assert result == mock_manager


@pytest.mark.integration
class TestLifespan:
    """Tests for application lifespan management."""

    @pytest.mark.asyncio
    @patch("smlx.server.app.ModelManager")
    async def test_lifespan_startup_shutdown(self, mock_manager_class):
        """Test application lifespan startup and shutdown."""
        mock_manager = Mock()
        mock_manager.cleanup = AsyncMock()
        mock_manager_class.return_value = mock_manager

        # Simulate lifespan
        from smlx.server.app import lifespan

        async with lifespan(app):
            # During startup
            mock_manager_class.assert_called_once()

            # Manager should be available
            from smlx.server.app import model_manager
            assert model_manager is not None

        # After shutdown
        mock_manager.cleanup.assert_called_once()


@pytest.mark.integration
class TestMiddleware:
    """Tests for middleware configuration."""

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is configured."""
        # Check middleware stack - middleware is wrapped, so check the cls attribute
        middleware_classes = [
            getattr(m.cls, "__name__", "") for m in app.user_middleware  # type: ignore
        ]

        assert "CORSMiddleware" in middleware_classes

    def test_custom_middleware_configured(self):
        """Test that custom middleware is configured."""
        # Check middleware stack - middleware is wrapped, so check the cls attribute
        middleware_classes = [
            getattr(m.cls, "__name__", "") for m in app.user_middleware  # type: ignore
        ]

        # Our custom middleware should be present
        assert "ErrorHandlingMiddleware" in middleware_classes
        assert "LoggingMiddleware" in middleware_classes
        assert "RateLimitMiddleware" in middleware_classes


@pytest.mark.integration
class TestRouterInclusion:
    """Tests for router inclusion."""

    def test_all_routes_included(self):
        """Test that all expected routes are included."""
        routes = [getattr(route, "path", "") for route in app.routes]  # type: ignore

        # Check OpenAI-compatible routes
        assert "/v1/completions" in routes
        assert "/v1/chat/completions" in routes
        assert "/v1/audio/transcriptions" in routes
        assert "/v1/embeddings" in routes
        assert "/v1/models" in routes
        assert "/v1/models/{model_id:path}" in routes

        # Check utility routes
        assert "/" in routes
        assert "/health" in routes
