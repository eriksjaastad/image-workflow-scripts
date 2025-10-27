#!/usr/bin/env python3
"""
Centralized Error Bus
=====================

Lightweight, process-safe (append-only) error recorder with JSONL persistence
under data/log_archives/errors.jsonl. Provides a simple API for scripts to
record errors and load recent entries for UI banners.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path("data") / "log_archives"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ERRORS_FILE = LOG_DIR / "errors.jsonl"


@dataclass
class ErrorEvent:
    ts: str
    tool: str  # e.g., ai_reviewer, web_sorter, desktop_multi_crop
    level: str  # error | warning
    message: str
    context: Dict[str, Any]


def record_error(tool: str, message: str, level: str = "error", context: Optional[Dict[str, Any]] = None) -> None:
    """Append a single error/warning to JSONL file (best-effort)."""
    try:
        evt = ErrorEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            tool=tool,
            level=level,
            message=message,
            context=context or {},
        )
        with ERRORS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(evt)) + "\n")
    except Exception:
        # fail-open: don't crash callers
        pass


def load_recent_errors(limit: int = 50, tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load up to 'limit' most recent errors, optionally filtered by tools."""
    results: List[Dict[str, Any]] = []
    try:
        if not ERRORS_FILE.exists():
            return []
        # Read tail-ish: simple read-all then slice due to file size being small
        lines = ERRORS_FILE.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                obj = json.loads(line)
                if tools and obj.get("tool") not in tools:
                    continue
                results.append(obj)
                if len(results) >= limit:
                    break
            except Exception:
                continue
    except Exception:
        return []
    return list(reversed(results))
