#!/usr/bin/env python3
"""
Flask API Response Utilities
=============================
Provides standardized response formats for Flask API endpoints.
"""

from datetime import datetime
from typing import Any

from flask import jsonify


def error_response(
    message: str, code: int = 400, details: Any = None
) -> tuple[Any, int]:
    """Create a standardized error response.

    Args:
        message: Human-readable error message
        code: HTTP status code (default: 400 Bad Request)
        details: Optional additional error details

    Returns:
        Tuple of (Flask Response, status code)

    Examples:
        >>> error_response("Project not found", 404)
        ({'error': {'message': 'Project not found', 'code': 404, ...}}, 404)

        >>> error_response("Invalid input", 400, {"field": "port", "expected": "1024-65535"})
        ({'error': {'message': 'Invalid input', 'code': 400, 'details': {...}, ...}}, 400)
    """
    error_obj = {
        "message": message,
        "code": code,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if details is not None:
        error_obj["details"] = details

    return jsonify({"error": error_obj}), code


def success_response(data: Any, code: int = 200) -> tuple[Any, int]:
    """Create a standardized success response.

    Args:
        data: Response data (will be JSON-serialized)
        code: HTTP status code (default: 200 OK)

    Returns:
        Tuple of (Flask Response, status code)
    """
    return jsonify(data), code
