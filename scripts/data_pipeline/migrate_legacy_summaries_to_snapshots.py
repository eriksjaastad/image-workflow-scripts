#!/usr/bin/env python3
"""
Migrate Legacy Daily Summaries to Snapshot Format
==================================================
Converts data/daily_summaries/*.json (legacy format)
to data/snapshot/daily_aggregates_v1/ (new format).

This preserves all historical data that doesn't exist in raw logs.

Usage:
    python scripts/data_pipeline/migrate_legacy_summaries_to_snapshots.py [--dry-run]
"""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = PROJECT_ROOT / "data" / "daily_summaries"
OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot" / "daily_aggregates_v1"


def parse_legacy_summary(legacy_file: Path) -> dict[str, Any]:
    """
    Parse a legacy daily_summary_YYYYMMDD.json file.

    Legacy format varies, but typically contains:
    - operations (list or dict)
    - scripts (dict or list)
    - File counts, timestamps, etc.
    """
    with open(legacy_file) as f:
        data = json.load(f)

    # Extract date from filename
    filename = legacy_file.stem  # e.g., "daily_summary_20250901"
    day_str = filename.replace("daily_summary_", "")

    return {"day": day_str, "legacy_data": data}


def convert_to_snapshot_format(legacy_parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Convert legacy summary to snapshot daily_aggregate_v1 format.

    Snapshot format:
    {
      "by_script": {
        "script_name": {
          "operations": {"op_type": count},
          "files_processed": int,
          "event_count": int,
          "first_op_ts": "ISO",
          "last_op_ts": "ISO"
        }
      },
      "by_operation": {"op_type": count},
      "projects_touched": [str],
      "total_files_processed": int,
      "total_events": int,
      "first_op_ts": "ISO",
      "last_op_ts": "ISO"
    }
    """
    legacy = legacy_parsed["legacy_data"]
    legacy_parsed["day"]

    # Initialize snapshot structure
    by_script = defaultdict(
        lambda: {
            "operations": defaultdict(int),
            "files_processed": 0,
            "event_count": 0,
            "timestamps": [],
        }
    )
    by_operation = defaultdict(int)
    projects_touched = set()

    # Parse legacy format (handle various structures)
    operations = []

    # Try different legacy formats
    if isinstance(legacy.get("operations"), list):
        operations = legacy["operations"]
    elif isinstance(legacy.get("operations"), dict):
        # Sometimes operations are grouped by script
        for script_ops in legacy["operations"].values():
            if isinstance(script_ops, list):
                operations.extend(script_ops)
    elif isinstance(legacy.get("data"), list):
        operations = legacy["data"]

    # Process each operation
    for op in operations:
        if not isinstance(op, dict):
            continue

        script = op.get("script", "unknown")
        operation = op.get("operation", "unknown")
        file_count = op.get("file_count", 1)
        timestamp = op.get("timestamp")

        # Update per-script stats
        by_script[script]["operations"][operation] += 1
        by_script[script]["files_processed"] += file_count
        by_script[script]["event_count"] += 1

        if timestamp:
            by_script[script]["timestamps"].append(timestamp)

        # Update global stats
        by_operation[operation] += 1

        # Track projects
        if "source_dir" in op:
            projects_touched.add(op["source_dir"])
        if "dest_dir" in op:
            projects_touched.add(op["dest_dir"])

    # Build final snapshot format
    script_stats = {}
    all_timestamps = []

    for script, data in by_script.items():
        if data["timestamps"]:
            # Normalize timestamps to ISO format
            normalized_ts = []
            for ts in data["timestamps"]:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromtimestamp(ts, tz=UTC)
                    normalized_ts.append(dt)
                except Exception:
                    continue

            if normalized_ts:
                script_stats[script] = {
                    "operations": dict(data["operations"]),
                    "files_processed": data["files_processed"],
                    "event_count": data["event_count"],
                    "first_op_ts": min(normalized_ts).isoformat(),
                    "last_op_ts": max(normalized_ts).isoformat(),
                }
                all_timestamps.extend(normalized_ts)
        else:
            # No timestamps, create minimal record
            script_stats[script] = {
                "operations": dict(data["operations"]),
                "files_processed": data["files_processed"],
                "event_count": data["event_count"],
                "first_op_ts": None,
                "last_op_ts": None,
            }

    return {
        "by_script": script_stats,
        "by_operation": dict(by_operation),
        "projects_touched": sorted(list(projects_touched)),
        "total_files_processed": sum(
            s["files_processed"] for s in script_stats.values()
        ),
        "total_events": sum(s["event_count"] for s in script_stats.values()),
        "first_op_ts": min(all_timestamps).isoformat() if all_timestamps else None,
        "last_op_ts": max(all_timestamps).isoformat() if all_timestamps else None,
        "_migrated_from": "legacy_daily_summary",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy daily summaries to snapshot format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    args = parser.parse_args()

    if not LEGACY_DIR.exists():
        print(f"❌ Legacy directory not found: {LEGACY_DIR}")
        return

    # Find all legacy summary files
    legacy_files = sorted(LEGACY_DIR.glob("daily_summary_*.json"))

    if not legacy_files:
        print(f"❌ No legacy summary files found in {LEGACY_DIR}")
        return

    print(f"{'=' * 70}")
    print("Legacy Daily Summaries → Snapshot Migration")
    print(f"{'=' * 70}")
    print(f"Found: {len(legacy_files)} legacy summary files")
    print(f"Source: {LEGACY_DIR}")
    print(f"Target: {OUTPUT_DIR}")
    print(
        f"Mode: {'DRY RUN (no files written)' if args.dry_run else 'LIVE (will write files)'}"
    )
    print(f"{'=' * 70}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    migrated_count = 0
    skipped_count = 0
    error_count = 0

    for legacy_file in legacy_files:
        try:
            # Parse legacy file
            legacy_parsed = parse_legacy_summary(legacy_file)
            day = legacy_parsed["day"]

            # Check if snapshot already exists
            snapshot_dir = OUTPUT_DIR / f"day={day}"
            snapshot_file = snapshot_dir / "aggregate.json"

            if snapshot_file.exists():
                print(f"  ⏭️  {day}: Already exists in snapshots (skipping)")
                skipped_count += 1
                continue

            # Convert to snapshot format
            snapshot = convert_to_snapshot_format(legacy_parsed)

            if not args.dry_run:
                # Write snapshot
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                with open(snapshot_file, "w") as f:
                    json.dump(snapshot, f, indent=2)

            print(
                f"  ✅ {day}: Migrated ({snapshot['total_events']} events, {snapshot['total_files_processed']} files, {len(snapshot['by_script'])} scripts)"
            )
            migrated_count += 1

        except Exception as e:
            print(f"  ❌ {legacy_file.name}: Error - {e}")
            error_count += 1

    print(f"\n{'=' * 70}")
    print("Migration Summary")
    print(f"{'=' * 70}")
    print(f"✅ Migrated: {migrated_count}")
    print(f"⏭️  Skipped (already exists): {skipped_count}")
    print(f"❌ Errors: {error_count}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        print(
            "🔍 DRY RUN: No files were written. Run without --dry-run to perform migration."
        )
    else:
        print(
            "✅ Migration complete! All legacy daily summaries converted to snapshot format."
        )
        print("\nNext steps:")
        print("  1. Verify dashboard loads all data correctly")
        print("  2. Run tests to ensure data integrity")
        print(
            "  3. Once verified, legacy data/daily_summaries/ can be archived/deleted"
        )


if __name__ == "__main__":
    main()
