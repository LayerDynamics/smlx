# Copyright © 2025 SMLX Project

"""
Tests for logging middleware.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.middleware.logging import LoggingMiddleware


@pytest.fixture
def app_with_middleware():
    """Create test app with logging middleware."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/test/slow")
    async def slow_endpoint():
        import time
        time.sleep(0.1)
        return {"status": "slow"}

    return app


@pytest.fixture
def client(app_with_middleware):
    """Create test client."""
    return TestClient(app_with_middleware)


@pytest.mark.unit
class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @patch("builtins.print")
    def test_request_logging(self, mock_print, client):
        """Test that requests are logged."""
        response = client.get("/test")

        assert response.status_code == 200

        # Check that print was called for request and response
        assert mock_print.call_count >= 2

        # Check request log
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("GET /test" in str(call) for call in calls)

    @patch("builtins.print")
    def test_response_logging_includes_duration(self, mock_print, client):
        """Test that response logs include duration."""
        response = client.get("/test")

        assert response.status_code == 200

        # Check response log includes status code and duration
        calls = [str(call) for call in mock_print.call_args_list]
        response_logs = [call for call in calls if "200" in call and "s)" in call]
        assert len(response_logs) > 0

    @patch("builtins.print")
    def test_different_status_codes_logged(self, mock_print, client):
        """Test that different status codes are logged correctly."""
        # Test successful request
        client.get("/test")

        # Test 404
        client.get("/nonexistent")

        # Check both status codes are logged
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("200" in str(call) for call in calls)
        assert any("404" in str(call) for call in calls)
