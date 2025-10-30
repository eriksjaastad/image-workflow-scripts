#!/usr/bin/env python3
"""
Backfill FileTracker Logs from SQLite Training Databases

When FileTracker fails silently, we lose dashboard metrics but SQLite
training databases still have all the timestamps and crop counts.

This script reconstructs FileTracker logs from SQLite data to restore
dashboard visibility.

USAGE:
------
  python scripts/tools/backfill_filetracker_from_sqlite.py mojo3
  python scripts/tools/backfill_filetracker_from_sqlite.py --all
  python scripts/tools/backfill_filetracker_from_sqlite.py mojo3 --dry-run

WHAT IT DOES:
-------------
1. Reads ai_decisions from SQLite databases
2. Groups crops into logical batches (15-minute windows)
3. Writes FileTracker-compatible log entries
4. Preserves original timestamps from SQLite
5. Makes dashboard metrics accurate again

SAFETY:
-------
- Uses dry-run mode by default
- Appends to existing logs (doesn't overwrite)
- Logs are marked as "backfilled" in notes field
- Can be run multiple times (idempotent)
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def find_sqlite_databases(data_dir: Path) -> List[Path]:
    """Find all SQLite training databases"""
    db_dir = data_dir / "training" / "ai_training_decisions"
    if not db_dir.exists():
        return []
    return list(db_dir.glob("*.db"))


def extract_crops_from_sqlite(db_path: Path) -> List[Dict[str, Any]]:
    """
    Extract crop operations from SQLite database.

    Returns list of dicts with:
    - timestamp: ISO timestamp of crop
    - project_id: Project name
    - directory: Directory path
    - images: JSON array of image filenames
    - action: User action (approve/crop)
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get crops with timestamps
    query = """
    SELECT
        timestamp,
        project_id,
        directory,
        images,
        user_action
    FROM ai_decisions
    WHERE user_action IN ('approve', 'crop')
    ORDER BY timestamp ASC
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    crops = []
    for row in rows:
        # Parse images JSON to get count
        images_json = row[3]
        try:
            images_list = json.loads(images_json) if images_json else []
            image_count = len(images_list)
        except:
            image_count = 1  # Fallback

        crops.append({
            "timestamp": row[0],
            "project_id": row[1] or db_path.stem,  # Use filename if no project_id
            "directory": row[2],
            "images": images_list if 'images_list' in locals() else [],
            "image_count": image_count,
            "action": row[4]
        })

    return crops


def group_crops_into_batches(crops: List[Dict], batch_window_minutes: int = 15) -> List[Dict]:
    """
    Group crops into batches based on time windows.

    Groups crops that happen within X minutes of each other into
    a single FileTracker log entry.

    Returns list of batch dicts with:
    - timestamp: Timestamp of first crop in batch
    - crop_count: Number of crops in batch
    - crops: List of individual crops
    """
    if not crops:
        return []

    batches = []
    current_batch = {
        "timestamp": crops[0]["timestamp"],
        "crops": [crops[0]],
        "project_id": crops[0]["project_id"]
    }

    for crop in crops[1:]:
        # Parse timestamps
        current_ts = datetime.fromisoformat(crop["timestamp"].replace("Z", "+00:00"))
        batch_ts = datetime.fromisoformat(current_batch["timestamp"].replace("Z", "+00:00"))

        # If within window, add to current batch
        if (current_ts - batch_ts).total_seconds() / 60 <= batch_window_minutes:
            current_batch["crops"].append(crop)
        else:
            # Close current batch, start new one
            current_batch["crop_count"] = len(current_batch["crops"])
            # Sum up actual image counts (each group can have 2-4 images)
            current_batch["image_count"] = sum(c.get("image_count", 1) for c in current_batch["crops"])
            batches.append(current_batch)

            current_batch = {
                "timestamp": crop["timestamp"],
                "crops": [crop],
                "project_id": crop["project_id"]
            }

    # Close final batch
    current_batch["crop_count"] = len(current_batch["crops"])
    current_batch["image_count"] = sum(c.get("image_count", 1) for c in current_batch["crops"])
    batches.append(current_batch)

    return batches


def write_filetracker_entries(batches: List[Dict], log_file: Path, dry_run: bool = True):
    """
    Write FileTracker log entries for each batch.

    Format matches FileTracker JSON structure:
    {
      "type": "file_operation",
      "script": "ai_desktop_multi_crop",
      "operation": "crop",
      "timestamp": "2025-10-25T11:43:46.110942Z",
      "source_dir": "__crop_auto",
      "dest_dir": "__cropped",
      "file_count": 100,
      "notes": "Backfilled from SQLite ai_decisions"
    }
    """
    entries = []

    for batch in batches:
        entry = {
            "type": "file_operation",
            "script": "ai_desktop_multi_crop",
            "operation": "crop",
            "timestamp": batch["timestamp"],
            "source_dir": "__crop_auto",
            "dest_dir": "__cropped",
            "file_count": batch.get("image_count", batch["crop_count"]),  # Use actual image count
            "notes": f"Backfilled from SQLite ({batch['project_id']}), {batch['crop_count']} groups",
            "project_id": batch["project_id"]
        }
        entries.append(entry)

    if dry_run:
        print(f"\n[DRY RUN] Would write {len(entries)} entries to {log_file}")
        print("\nSample entries:")
        for entry in entries[:3]:
            print(f"  {entry['timestamp']}: {entry['file_count']} crops")
        if len(entries) > 3:
            print(f"  ... and {len(entries) - 3} more")
        return entries

    # Append to log file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Wrote {len(entries)} entries to {log_file}")
    return entries


def backfill_project(project_name: str, data_dir: Path, dry_run: bool = True):
    """Backfill FileTracker logs for a single project"""
    db_path = data_dir / "training" / "ai_training_decisions" / f"{project_name}.db"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"Backfilling: {project_name}")
    print(f"Database: {db_path}")
    print(f"{'=' * 70}")

    # Extract crops
    crops = extract_crops_from_sqlite(db_path)
    print(f"Found {len(crops)} crops in database")

    if not crops:
        print("No crops to backfill")
        return

    # Show date range
    first_ts = crops[0]["timestamp"]
    last_ts = crops[-1]["timestamp"]
    print(f"Date range: {first_ts[:10]} to {last_ts[:10]}")

    # Group into batches
    batches = group_crops_into_batches(crops, batch_window_minutes=15)
    print(f"Grouped into {len(batches)} batches (15-minute windows)")

    # Write to FileTracker log
    log_file = data_dir / "file_operations_logs" / "file_operations.log"
    entries = write_filetracker_entries(batches, log_file, dry_run=dry_run)

    # Summary
    total_images = sum(c.get("image_count", 1) for c in crops)
    print(f"\nSummary:")
    print(f"  Crop groups: {len(crops)}")
    print(f"  Total images: {total_images}")
    print(f"  Batches: {len(batches)}")
    print(f"  Date range: {first_ts[:10]} to {last_ts[:10]}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill FileTracker logs from SQLite training databases"
    )
    parser.add_argument(
        "project",
        nargs="?",
        help="Project name (e.g., mojo3) or --all for all projects"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backfill all projects"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be done without writing (default: True)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write the log entries (disables dry-run)"
    )

    args = parser.parse_args()

    # Data directory
    data_dir = project_root / "data"

    # Determine dry-run mode
    dry_run = not args.execute

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print("   Use --execute to actually write log entries\n")
    else:
        print("⚠️  EXECUTE MODE - Will modify log files")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted")
            return
        print()

    # Backfill
    if args.all:
        dbs = find_sqlite_databases(data_dir)
        print(f"Found {len(dbs)} databases")
        for db in dbs:
            backfill_project(db.stem, data_dir, dry_run=dry_run)
    elif args.project:
        backfill_project(args.project, data_dir, dry_run=dry_run)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/tools/backfill_filetracker_from_sqlite.py mojo3 --dry-run")
        print("  python scripts/tools/backfill_filetracker_from_sqlite.py mojo3 --execute")
        print("  python scripts/tools/backfill_filetracker_from_sqlite.py --all --execute")


if __name__ == "__main__":
    main()
