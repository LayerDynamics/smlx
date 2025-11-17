# Copyright © 2025 SMLX Project

"""
Tests for rate limiting middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from smlx.server.middleware.rate_limit import RateLimitMiddleware


@pytest.fixture
def app_with_rate_limit():
    """Create test app with rate limiting (low limit for testing)."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=5)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app_with_rate_limit):
    """Create test client."""
    return TestClient(app_with_rate_limit)


@pytest.mark.unit
class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    def test_requests_within_limit(self, client):
        """Test that requests within limit are allowed."""
        # Make 3 requests (under limit of 5)
        for _ in range(3):
            response = client.get("/test")
            assert response.status_code == 200

    def test_rate_limit_exceeded(self, client):
        """Test that excessive requests are blocked."""
        # Make requests up to limit (5)
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429

        data = response.json()
        assert "error" in data
        assert data["error"]["message"] == "Rate limit exceeded"
        assert data["error"]["type"] == "rate_limit_error"
        assert data["error"]["code"] == "rate_limit_exceeded"

    def test_retry_after_header(self, client):
        """Test that rate limit response includes Retry-After header."""
        # Exceed rate limit
        for _ in range(6):
            response = client.get("/test")

        # Check last response has Retry-After header
        if response.status_code == 429:
            assert "retry-after" in response.headers
            assert int(response.headers["retry-after"]) == 60

    def test_rate_limit_per_client(self):
        """Test that rate limits are per client IP."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app, requests_per_minute=5)

        # Check that client_requests dict exists and is initialized
        assert hasattr(middleware, "client_requests")
        assert isinstance(middleware.client_requests, dict)
        assert hasattr(middleware, "requests_per_minute")
        assert middleware.requests_per_minute == 5
        assert middleware.window_size == 60


@pytest.mark.unit
class TestRateLimitConfiguration:
    """Tests for rate limit configuration."""

    def test_custom_rate_limit(self):
        """Test configuring custom rate limit."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app, requests_per_minute=100)

        assert middleware.requests_per_minute == 100
        assert middleware.window_size == 60

    def test_default_rate_limit(self):
        """Test default rate limit."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)

        assert middleware.requests_per_minute == 60  # Default
        assert middleware.window_size == 60
