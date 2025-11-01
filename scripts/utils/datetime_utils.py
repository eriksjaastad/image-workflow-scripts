#!/usr/bin/env python3
"""
Datetime Utility Functions
==========================
Provides consistent timezone handling across the dashboard system.

Key functions:
- normalize_to_naive_utc: Convert any datetime to naive UTC (strips tzinfo after conversion)
"""

from datetime import UTC, datetime


def normalize_to_naive_utc(dt: datetime) -> datetime:
    """Convert datetime to naive UTC (drop tzinfo after conversion to UTC).

    This ensures consistent handling of timezone-aware and naive datetimes.

    Behavior:
    - If dt is naive (no tzinfo): Assume it's already UTC, return as-is
    - If dt is timezone-aware: Convert to UTC, then strip tzinfo

    Examples:
        >>> naive = datetime(2024, 11, 1, 14, 0, 0)  # No timezone
        >>> normalize_to_naive_utc(naive)
        datetime(2024, 11, 1, 14, 0, 0)  # Unchanged, assumed UTC

        >>> from zoneinfo import ZoneInfo
        >>> aware = datetime(2024, 11, 1, 14, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # +05:30
        >>> normalize_to_naive_utc(aware)
        datetime(2024, 11, 1, 8, 30, 0)  # Converted to UTC, tzinfo stripped

    Args:
        dt: Datetime to normalize (may be naive or timezone-aware)

    Returns:
        Naive datetime in UTC
    """
    if dt.tzinfo is None:
        # Already naive, assume UTC
        return dt
    # Convert to UTC, then strip timezone info
    return dt.astimezone(UTC).replace(tzinfo=None)


def safe_parse_iso(ts: str | None) -> datetime | None:
    """Parse ISO timestamp and normalize to naive UTC.

    Handles common ISO format variations:
    - With 'Z' suffix (converts to +00:00)
    - With explicit timezone offset
    - Naive timestamp (assumes UTC)

    Args:
        ts: ISO 8601 timestamp string or None

    Returns:
        Naive UTC datetime, or None if parsing fails
    """
    if not ts:
        return None
    try:
        # Convert 'Z' suffix to +00:00 for fromisoformat compatibility
        normalized = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(normalized)
        return normalize_to_naive_utc(dt)
    except (ValueError, AttributeError):
        return None
