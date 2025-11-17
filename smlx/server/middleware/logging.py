# Copyright © 2025 SMLX Project

"""
Logging middleware for SMLX Server.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log HTTP requests and responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        print(f"→ {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        print(f"← {request.method} {request.url.path} - {response.status_code} ({duration:.2f}s)")

        return response
