#!/usr/bin/env python3
"""
Update system heartbeat file.

This script updates ops/heartbeat.json with the current timestamp to signal
that the image workflow system is alive and functioning. The deadman switch
workflow (.github/workflows/deadman.yml) monitors this file and alerts if
it becomes stale.

Usage:
    # Update heartbeat (typically called from cron or at script completion)
    python scripts/tools/update_heartbeat.py

    # Update with custom notes
    python scripts/tools/update_heartbeat.py --notes "Completed batch 1101"

    # Dry run (show what would be written)
    python scripts/tools/update_heartbeat.py --dry-run

Cron example (update every 30 minutes):
    */30 * * * * cd /path/to/image-workflow-scripts && python scripts/tools/update_heartbeat.py

Pre-commit hook example (update on every commit):
    #!/bin/bash
    python scripts/tools/update_heartbeat.py --notes "Commit: $(git log -1 --oneline)"
"""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def update_heartbeat(notes: str = "", dry_run: bool = False) -> None:
    """
    Update the system heartbeat file with current UTC timestamp.

    Args:
        notes: Optional notes to include in heartbeat
        dry_run: If True, print what would be written without actually writing

    Returns:
        None
    """
    # Find project root (2 levels up from this script)
    project_root = Path(__file__).resolve().parents[2]
    heartbeat_file = project_root / "ops" / "heartbeat.json"

    # Generate heartbeat data
    heartbeat_data = {
        "last_ok": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": "image-workflow-scripts",
        "version": "1.0",
        "notes": notes if notes else "System operational",
    }

    if dry_run:
        print("DRY RUN - Would write to:", heartbeat_file)
        print(json.dumps(heartbeat_data, indent=2))
        return

    # Ensure ops directory exists
    heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

    # Write heartbeat file
    try:
        with heartbeat_file.open("w") as f:
            json.dump(heartbeat_data, f, indent=2)
            f.write("\n")  # Add trailing newline
        print(f"✓ Updated heartbeat: {heartbeat_file}")
        print(f"  Timestamp: {heartbeat_data['last_ok']}")
        if notes:
            print(f"  Notes: {notes}")
    except IOError as e:
        print(f"Error writing heartbeat file: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """
    Main entry point for heartbeat updater.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Update system heartbeat file for deadman switch monitoring"
    )
    parser.add_argument(
        "--notes",
        "-n",
        default="",
        help="Optional notes to include in heartbeat"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without actually writing"
    )
    args = parser.parse_args()

    update_heartbeat(notes=args.notes, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
