#!/usr/bin/env python3
"""
scan_dir_state
--------------
Report directory state (EMPTY/PARTIAL/FULL) with counts and recency as JSON.

Usage:
  python scripts/tools/scan_dir_state.py --path /abs/path/to/content [--recent-mins 10] [--json /abs/out/state.json]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def scan(root: Path, recent_mins: int) -> dict:
    total_files = 0
    total_bytes = 0
    by_ext: dict[str, dict[str, int]] = {}
    hidden_files = 0
    latest_mtime = None

    now = datetime.now(UTC)
    recent_delta = timedelta(minutes=recent_mins)
    recent_files = 0

    if not root.exists() or not root.is_dir():
        return {
            "path": str(root),
            "scannedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": "path_not_found",
            "state": "EMPTY",
            "totalFiles": 0,
            "totalBytes": 0,
        }

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not is_hidden(d)]
        for name in filenames:
            if is_hidden(name):
                hidden_files += 1
                continue
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            total_files += 1
            total_bytes += st.st_size
            mtime = datetime.fromtimestamp(st.st_mtime, tz=UTC)
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
            if now - mtime <= recent_delta:
                recent_files += 1
            ext = p.suffix.lower().lstrip(".")
            if ext:
                slot = by_ext.setdefault(ext, {"files": 0, "bytes": 0})
                slot["files"] += 1
                slot["bytes"] += st.st_size

    latest_mtime_str = (
        latest_mtime.strftime("%Y-%m-%dT%H:%M:%SZ") if latest_mtime else None
    )

    if total_files == 0:
        state = "EMPTY"
    elif recent_files == 0:
        state = "FULL"
    else:
        state = "PARTIAL"

    return {
        "path": str(root),
        "scannedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totalFiles": total_files,
        "totalBytes": total_bytes,
        "byExtension": by_ext,
        "hiddenFiles": hidden_files,
        "latestMtimeUtc": latest_mtime_str,
        "recentWindowMins": recent_mins,
        "recentFiles": recent_files,
        "state": state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report directory state with counts and recency"
    )
    parser.add_argument("--path", required=True, help="Directory to scan")
    parser.add_argument(
        "--recent-mins",
        type=int,
        default=10,
        help="Recency window in minutes (default 10)",
    )
    parser.add_argument("--json", help="Optional output JSON path")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    data = scan(root, args.recent_mins)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[*] Wrote {out}")
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
