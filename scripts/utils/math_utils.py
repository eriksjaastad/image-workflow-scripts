#!/usr/bin/env python3
"""
Math Utility Functions
======================
Provides safe mathematical operations for dashboard calculations.

Key functions:
- safe_rate: Compute rate with zero-division guards
- safe_divide: General-purpose safe division
"""


def safe_rate(
    numerator: int | float,
    denominator: int | float,
    default: float = 0.0,
    precision: int = 1,
) -> float:
    """Safely compute rate with guards against division by zero or negative values.

    Args:
        numerator: The dividend (e.g., images processed)
        denominator: The divisor (e.g., hours worked)
        default: Value to return if calculation is invalid (default: 0.0)
        precision: Number of decimal places to round to (default: 1)

    Returns:
        Rounded rate, or default if inputs are invalid

    Examples:
        >>> safe_rate(100, 5)
        20.0
        >>> safe_rate(0, 5)
        0.0
        >>> safe_rate(100, 0)
        0.0
        >>> safe_rate(-10, 5)
        0.0
    """
    if numerator <= 0 or denominator <= 0:
        return default
    return round(numerator / denominator, precision)


def safe_divide(
    numerator: int | float,
    denominator: int | float,
    default: float = 0.0,
    epsilon: float = 1e-9,
) -> float:
    """Safely divide two numbers with guard against division by zero.

    Uses epsilon to handle floating-point precision issues.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if denominator is effectively zero
        epsilon: Minimum threshold for denominator (default: 1e-9)

    Returns:
        Division result, or default if denominator is too small

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 1e-10)
        0.0
    """
    if abs(denominator) < epsilon:
        return default
    return numerator / denominator
