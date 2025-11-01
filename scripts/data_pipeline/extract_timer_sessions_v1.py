#!/usr/bin/env python3
"""
Extract Timer Sessions v1
==========================
Extracts and normalizes legacy ActivityTimer session data to snapshot format.

Reads from:
- data/timer_data/session_*.json (per-session files)
- data/timer_data/daily_*.json (daily aggregates)

Outputs to:
- snapshot/timer_sessions_v1/day=YYYYMMDD/sessions.jsonl

Note: This is for archival/comparison. Primary session source is derived_sessions_v1.
"""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Load config
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "metrics_config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMER_DIR = PROJECT_ROOT / "data" / "timer_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot" / "timer_sessions_v1"


def parse_epoch_timestamp(epoch: float) -> datetime:
    """Convert epoch timestamp to UTC datetime."""
    return datetime.fromtimestamp(epoch, tz=UTC)


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp to UTC."""
    if ts_str.endswith("Z"):
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def normalize_session(
    raw_session: dict[str, Any], source_file: str
) -> dict[str, Any] | None:
    """Normalize a legacy timer session to canonical schema."""
    try:
        # Extract script_id
        script_id = raw_session.get("script_name") or raw_session.get(
            "script", "unknown"
        )
        session_id = raw_session.get("session_id", "unknown")

        # Parse timestamps
        start_time = raw_session.get("start_time")
        end_time = raw_session.get("end_time")

        if start_time is None:
            return None

        # Handle epoch timestamps
        if isinstance(start_time, (int, float)):
            start_dt = parse_epoch_timestamp(start_time)
            end_dt = (
                parse_epoch_timestamp(end_time) if end_time and end_time > 0 else None
            )
        else:
            # String timestamp
            start_dt = parse_iso_timestamp(start_time)
            end_dt = parse_iso_timestamp(end_time) if end_time else None

        day_str = start_dt.strftime("%Y%m%d")

        # Extract metrics
        active_seconds = raw_session.get("active_time", 0.0)
        idle_seconds = raw_session.get("idle_time", 0.0)
        files_processed = raw_session.get("files_processed", 0)

        # Build normalized session
        normalized = {
            "source": "legacy_activity_timer",
            "script_id": script_id,
            "session_id": session_id,
            "start_ts_utc": start_dt.isoformat(),
            "end_ts_utc": end_dt.isoformat() if end_dt else None,
            "active_seconds": float(active_seconds),
            "idle_seconds": float(idle_seconds),
            "files_processed": int(files_processed),
            "day": day_str,
            "extra": {"source_file": source_file},
        }

        # Capture batches and operations if present
        if "batches" in raw_session:
            normalized["extra"]["batches"] = raw_session["batches"]

        if "operations" in raw_session:
            normalized["extra"]["operations"] = raw_session["operations"]

        if "efficiency" in raw_session:
            normalized["extra"]["efficiency"] = raw_session["efficiency"]

        return normalized

    except Exception:
        return None


def extract_from_daily_file(daily_path: Path) -> list[dict[str, Any]]:
    """Extract sessions from a daily aggregate file."""
    sessions = []

    try:
        with open(daily_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array format and dict format
        if isinstance(data, list):
            # Array of sessions
            for raw_session in data:
                normalized = normalize_session(raw_session, daily_path.name)
                if normalized:
                    sessions.append(normalized)
        elif isinstance(data, dict):
            # Dict with scripts -> sessions structure
            scripts_data = data.get("scripts", {})
            for script_name, script_data in scripts_data.items():
                if isinstance(script_data, dict):
                    for raw_session in script_data.get("sessions", []):
                        normalized = normalize_session(raw_session, daily_path.name)
                        if normalized:
                            sessions.append(normalized)

    except Exception as e:
        print(f"  ⚠️  Error reading {daily_path}: {e}")

    return sessions


def extract_from_session_file(session_path: Path) -> dict[str, Any] | None:
    """Extract a single session from a per-session file."""
    try:
        with open(session_path, encoding="utf-8") as f:
            raw_session = json.load(f)

        return normalize_session(raw_session, session_path.name)

    except Exception:
        return None


def main():
    """Main entry point."""
    print("Extracting legacy timer sessions...")

    if not TIMER_DIR.exists():
        print(f"Timer data directory not found: {TIMER_DIR}")
        return

    # Collect sessions from daily files
    by_day = defaultdict(list)
    seen_session_keys = set()  # (script_id, session_id, start_ts)
    duplicate_count = 0

    # Process daily files first (most complete data)
    daily_files = list(TIMER_DIR.glob("daily_*.json"))
    print(f"Found {len(daily_files)} daily files")

    for daily_file in sorted(daily_files):
        print(f"  Processing {daily_file.name}...")
        sessions = extract_from_daily_file(daily_file)

        for session in sessions:
            session_key = (
                session["script_id"],
                session["session_id"],
                session["start_ts_utc"],
            )

            # Dedupe
            if session_key in seen_session_keys:
                duplicate_count += 1
                continue

            seen_session_keys.add(session_key)
            day = session["day"]
            by_day[day].append(session)

    # Process individual session files (supplement)
    session_files = list(TIMER_DIR.glob("session_*.json"))
    print(f"\nFound {len(session_files)} session files")

    for session_file in sorted(session_files):
        session = extract_from_session_file(session_file)
        if not session:
            continue

        session_key = (
            session["script_id"],
            session["session_id"],
            session["start_ts_utc"],
        )

        # Dedupe
        if session_key in seen_session_keys:
            duplicate_count += 1
            continue

        seen_session_keys.add(session_key)
        day = session["day"]
        by_day[day].append(session)

    print(
        f"\nExtracted {len(seen_session_keys)} unique sessions ({duplicate_count} duplicates skipped)"
    )
    print(f"Days: {len(by_day)}")

    # Write partitioned output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for day_str in sorted(by_day.keys()):
        day_sessions = by_day[day_str]

        # Create partition directory
        day_dir = OUTPUT_DIR / f"day={day_str}"
        day_dir.mkdir(parents=True, exist_ok=True)
        output_file = day_dir / "sessions.jsonl"

        # Write sessions
        with open(output_file, "w") as f:
            for session in sorted(day_sessions, key=lambda s: s["start_ts_utc"]):
                f.write(json.dumps(session) + "\n")

        total_written += len(day_sessions)
        print(f"  {day_str}: {len(day_sessions)} sessions")

    print(f"\n✅ Done! {total_written} sessions written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
