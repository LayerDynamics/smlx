# Copyright © 2025 SMLX Project

"""
Error handling middleware for SMLX Server.
"""

import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle errors and return consistent error responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except ValueError as e:
            # Client errors (400)
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_request",
                    }
                },
            )
        except NotImplementedError as e:
            # Not implemented (501)
            return JSONResponse(
                status_code=501,
                content={
                    "error": {
                        "message": str(e),
                        "type": "not_implemented_error",
                        "param": None,
                        "code": "not_implemented",
                    }
                },
            )
        except Exception as e:
            # Internal server errors (500)
            print(f"❌ Internal server error: {e}")
            traceback.print_exc()

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "type": "server_error",
                        "param": None,
                        "code": "internal_error",
                    }
                },
            )
