# Copyright © 2025 SMLX Project

"""
Tests for error handling middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.middleware.error_handling import ErrorHandlingMiddleware


@pytest.fixture
def app_with_middleware():
    """Create test app with error handling middleware."""
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware)

    @app.get("/test/ok")
    async def test_ok():
        return {"status": "ok"}

    @app.get("/test/value_error")
    async def test_value_error():
        raise ValueError("Invalid value")

    @app.get("/test/not_implemented")
    async def test_not_implemented():
        raise NotImplementedError("Feature not implemented")

    @app.get("/test/generic_error")
    async def test_generic_error():
        raise Exception("Something went wrong")

    return app


@pytest.fixture
def client(app_with_middleware):
    """Create test client."""
    return TestClient(app_with_middleware)


@pytest.mark.unit
class TestErrorHandlingMiddleware:
    """Tests for ErrorHandlingMiddleware."""

    def test_normal_request(self, client):
        """Test that normal requests pass through."""
        response = client.get("/test/ok")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_value_error_handling(self, client):
        """Test that ValueError is converted to 400 error."""
        response = client.get("/test/value_error")

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["message"] == "Invalid value"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "invalid_request"

    def test_not_implemented_error_handling(self, client):
        """Test that NotImplementedError is converted to 501 error."""
        response = client.get("/test/not_implemented")

        assert response.status_code == 501
        data = response.json()
        assert "error" in data
        assert data["error"]["message"] == "Feature not implemented"
        assert data["error"]["type"] == "not_implemented_error"
        assert data["error"]["code"] == "not_implemented"

    def test_generic_error_handling(self, client):
        """Test that generic Exception is converted to 500 error."""
        response = client.get("/test/generic_error")

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["message"] == "Internal server error"
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "internal_error"
