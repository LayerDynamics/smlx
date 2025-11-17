# Copyright © 2025 SMLX Project

"""
Rate limiting middleware for SMLX Server.
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limiting.

    Uses a simple sliding window algorithm per client IP.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            app: ASGI application
            requests_per_minute: Maximum requests per minute per client
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        self.client_requests: defaultdict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Clean old requests
        now = time.time()
        self.client_requests[client_ip] = [
            req_time
            for req_time in self.client_requests[client_ip]
            if now - req_time < self.window_size
        ]

        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "param": None,
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(self.window_size)},
            )

        # Record this request
        self.client_requests[client_ip].append(now)

        # Process request
        response = await call_next(request)
        return response
