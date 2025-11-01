#!/usr/bin/env python3
"""
Extract Progress Snapshots v1
==============================
Extracts and normalizes progress data from crop and sorter progress files.

Reads from:
- data/crop_progress/*.json
- data/sorter_progress/*.json

Outputs to:
- snapshot/progress_snapshots_v1/day=YYYYMMDD/snapshots.jsonl
"""

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CROP_PROGRESS_DIR = PROJECT_ROOT / "data" / "crop_progress"
SORTER_PROGRESS_DIR = PROJECT_ROOT / "data" / "sorter_progress"
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot" / "progress_snapshots_v1"


def parse_timestamp(ts_str: str) -> datetime | None:
    """Parse various timestamp formats to UTC datetime."""
    if not ts_str:
        return None

    try:
        # Try ISO with Z
        if ts_str.endswith("Z"):
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.astimezone(UTC)

        # Try ISO with timezone
        if "+" in ts_str or ts_str.count("-") > 2:
            dt = datetime.fromisoformat(ts_str)
            return dt.astimezone(UTC)

        # Try ISO naive (assume UTC)
        dt = datetime.fromisoformat(ts_str)
        return dt.replace(tzinfo=UTC)
    except ValueError:
        pass

    # Try space-separated format: "2025-10-02 01:20:23"
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=UTC)
    except ValueError:
        pass

    return None


def generate_directory_key(directory_path: str) -> str:
    """Generate stable hash for directory path."""
    return hashlib.sha1(directory_path.encode()).hexdigest()[:12]


def generate_progress_id(
    tool_id: str, directory_key: str, snapshot_ts: datetime
) -> str:
    """Generate stable progress snapshot ID."""
    canonical = f"{tool_id}|{directory_key}|{snapshot_ts.isoformat()}"
    return f"prog:{hashlib.md5(canonical.encode()).hexdigest()[:12]}"


def normalize_crop_progress(
    raw_progress: dict[str, Any], source_file: str
) -> list[dict[str, Any]]:
    """Normalize crop progress to canonical schema."""
    snapshots = []

    base_directory = raw_progress.get("base_directory", "unknown")
    session_start = raw_progress.get("session_start")

    snapshot_ts = parse_timestamp(session_start) or datetime.now(UTC)
    day_str = snapshot_ts.strftime("%Y%m%d")

    # Get file modification time as fallback
    try:
        source_path = CROP_PROGRESS_DIR / source_file
        if source_path.exists():
            mtime = source_path.stat().st_mtime
            fallback_ts = datetime.fromtimestamp(mtime, tz=UTC)
            if not session_start:
                snapshot_ts = fallback_ts
                day_str = snapshot_ts.strftime("%Y%m%d")
    except Exception:
        pass

    # Extract per-directory progress
    directories = raw_progress.get("directories", {})

    for dir_name, dir_data in directories.items():
        directory_key = generate_directory_key(f"{base_directory}/{dir_name}")

        snapshot = {
            "source": "crop_progress",
            "tool_id": "multi_crop_tool",
            "directory_key": directory_key,
            "snapshot_ts_utc": snapshot_ts.isoformat(),
            "day": day_str,
            "status": dir_data.get("status", "unknown"),
            "files_processed": dir_data.get("files_processed", 0),
            "total_files": dir_data.get("total_files", 0),
            "schema_version": "progress_v1",
            "extra": {
                "base_directory": base_directory,
                "directory_name": dir_name,
                "source_file": source_file,
                "session_start": session_start,
                "current_directory_index": raw_progress.get("current_directory_index"),
                "current_file_index": raw_progress.get("current_file_index"),
            },
        }

        # Add completed files sample if present
        if "completed_files" in dir_data:
            completed = dir_data["completed_files"]
            snapshot["completed_files_sample"] = (
                completed[:50] if len(completed) > 50 else completed
            )

        snapshot["progress_id"] = generate_progress_id(
            snapshot["tool_id"], directory_key, snapshot_ts
        )

        snapshots.append(snapshot)

    return snapshots


def normalize_sorter_progress(
    raw_progress: dict[str, Any], source_file: str
) -> dict[str, Any] | None:
    """Normalize sorter progress to canonical schema."""
    directory = raw_progress.get("directory", "unknown")
    started_at = raw_progress.get("started_at")
    updated_at = raw_progress.get("updated_at")

    snapshot_ts = (
        parse_timestamp(updated_at) or parse_timestamp(started_at) or datetime.now(UTC)
    )
    day_str = snapshot_ts.strftime("%Y%m%d")

    directory_key = generate_directory_key(directory)

    # Count completed files
    completed_files = raw_progress.get("completed_files", [])
    completed_count = len(completed_files)
    total_files = raw_progress.get("total_files", completed_count)

    snapshot = {
        "source": "sorter_progress",
        "tool_id": "character_sorter",
        "directory_key": directory_key,
        "snapshot_ts_utc": snapshot_ts.isoformat(),
        "day": day_str,
        "status": raw_progress.get("status", "in_progress"),
        "files_processed": completed_count,
        "total_files": total_files,
        "completed_files_sample": completed_files[:50]
        if len(completed_files) > 50
        else completed_files,
        "schema_version": "progress_v1",
        "extra": {
            "directory": directory,
            "source_file": source_file,
            "started_at": started_at,
            "updated_at": updated_at,
            "current_index": raw_progress.get("current_index", 0),
        },
    }

    snapshot["progress_id"] = generate_progress_id(
        snapshot["tool_id"], directory_key, snapshot_ts
    )

    return snapshot


def extract_from_progress_file(
    progress_path: Path, tool_type: str
) -> list[dict[str, Any]]:
    """Extract progress snapshots from a file."""
    snapshots = []

    try:
        with open(progress_path, encoding="utf-8") as f:
            raw_progress = json.load(f)

        if tool_type == "crop":
            snapshots = normalize_crop_progress(raw_progress, progress_path.name)
        elif tool_type == "sorter":
            snapshot = normalize_sorter_progress(raw_progress, progress_path.name)
            if snapshot:
                snapshots.append(snapshot)

    except Exception as e:
        print(f"  ⚠️  Error reading {progress_path}: {e}")

    return snapshots


def main():
    """Main entry point."""
    print("Extracting progress snapshots...")

    # Collect all progress files
    progress_files = []

    if CROP_PROGRESS_DIR.exists():
        for p in CROP_PROGRESS_DIR.glob("*.json"):
            progress_files.append((p, "crop"))

    if SORTER_PROGRESS_DIR.exists():
        for p in SORTER_PROGRESS_DIR.glob("*.json"):
            progress_files.append((p, "sorter"))

    print(f"Found {len(progress_files)} progress files")

    # Extract snapshots
    by_day = defaultdict(list)
    seen_progress_ids = set()
    duplicate_count = 0

    for progress_file, tool_type in progress_files:
        print(f"  Processing {progress_file.name} ({tool_type})...")
        snapshots = extract_from_progress_file(progress_file, tool_type)

        for snapshot in snapshots:
            progress_id = snapshot["progress_id"]

            # Dedupe
            if progress_id in seen_progress_ids:
                duplicate_count += 1
                continue

            seen_progress_ids.add(progress_id)
            day = snapshot["day"]
            by_day[day].append(snapshot)

    print(
        f"\nExtracted {len(seen_progress_ids)} unique snapshots ({duplicate_count} duplicates skipped)"
    )
    print(f"Days: {len(by_day)}")

    # Write partitioned output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for day_str in sorted(by_day.keys()):
        day_snapshots = by_day[day_str]

        # Create partition directory
        day_dir = OUTPUT_DIR / f"day={day_str}"
        day_dir.mkdir(parents=True, exist_ok=True)
        output_file = day_dir / "snapshots.jsonl"

        # Write snapshots
        with open(output_file, "w") as f:
            for snapshot in sorted(day_snapshots, key=lambda s: s["snapshot_ts_utc"]):
                f.write(json.dumps(snapshot) + "\n")

        total_written += len(day_snapshots)
        print(f"  {day_str}: {len(day_snapshots)} snapshots")

    print(f"\n✅ Done! {total_written} snapshots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
