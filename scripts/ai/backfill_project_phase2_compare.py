#!/usr/bin/env python3
"""
Backfill Phase 2: Compare and Merge Databases

This script compares the temporary backfill database with the real database
and merges data intelligently:
- Adds records that exist in temp but not in real database
- Fills NULL fields in real database from temp database
- NEVER overwrites existing data in real database
- Preserves all existing AI predictions and timestamps

Usage:
    python scripts/ai/backfill_project_phase2_compare.py \\
        --temp-db data/training/ai_training_decisions/mojo3_backfill_temp.db \\
        --real-db data/training/ai_training_decisions/mojo3.db \\
        --dry-run
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_all_records(db_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all records from database, keyed by group_id."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM ai_decisions")

    records = {}
    for row in cursor:
        record = dict(row)
        records[record["group_id"]] = record

    conn.close()
    return records


def build_image_to_record_map(records: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Build a map from image filename to group_id for finding matches."""
    image_map = {}
    for group_id, record in records.items():
        try:
            images = json.loads(record["images"])
            for img_path in images:
                # Extract just the filename (no directory)
                filename = Path(img_path).name
                image_map[filename] = group_id
        except (json.JSONDecodeError, TypeError):
            continue
    return image_map


def coords_are_close(
    coords1_json: Optional[str], coords2_json: Optional[str], tolerance: float = 0.01
) -> bool:
    """
    Check if two crop coordinate sets are within tolerance (1% default).

    Args:
        coords1_json: JSON string like "[0.1, 0.2, 0.9, 0.8]"
        coords2_json: JSON string like "[0.1, 0.2, 0.9, 0.8]"
        tolerance: Allowed deviation (0.01 = 1%)

    Returns:
        True if coordinates are within tolerance
    """
    if coords1_json is None or coords2_json is None:
        return False

    try:
        coords1 = json.loads(coords1_json)
        coords2 = json.loads(coords2_json)

        if len(coords1) != 4 or len(coords2) != 4:
            return False

        # Check each coordinate
        for c1, c2 in zip(coords1, coords2):
            if abs(c1 - c2) > tolerance:
                return False

        return True
    except Exception:
        return False


def compare_records(
    temp_record: Dict[str, Any], real_record: Optional[Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]:
    """
    Compare temp and real records and determine merge action.

    CRITICAL RULES:
    - NEVER overwrite timestamps (timestamp, crop_timestamp)
    - Temp database is source of truth for: final_crop_coords, user_selected_index, user_action
    - Use 1% tolerance when comparing coordinates
    - Only update if coordinates differ beyond tolerance

    Returns:
        (action, data) where action is:
        - 'add': Record doesn't exist in real DB, add it
        - 'fill_nulls': Record exists but has NULL values to fill
        - 'update_coords': Crop coordinates differ beyond tolerance
        - 'skip': Record exists and is complete, no changes needed
    """
    if real_record is None:
        return ("add", temp_record)

    # Check which fields need to be filled or updated
    fields_to_update = {}

    # CRITICAL: Check if final_crop_coords differ (with tolerance)
    temp_coords = temp_record.get("final_crop_coords")
    real_coords = real_record.get("final_crop_coords")

    # If both have coords, check if they're different (beyond tolerance)
    if temp_coords and real_coords:
        if not coords_are_close(temp_coords, real_coords, tolerance=0.01):
            # Coordinates differ beyond 1% tolerance - temp is source of truth
            fields_to_update["final_crop_coords"] = temp_coords
            # Recalculate crop_match with new coordinates
            if temp_record.get("crop_match") is not None:
                fields_to_update["crop_match"] = temp_record["crop_match"]

    # If real DB has NULL coords but temp has coords, fill them
    if real_coords is None and temp_coords is not None:
        fields_to_update["final_crop_coords"] = temp_coords

    # AI prediction fields - only fill if NULL in real DB (never overwrite existing AI data)
    if (
        real_record["ai_selected_index"] is None
        and temp_record["ai_selected_index"] is not None
    ):
        fields_to_update["ai_selected_index"] = temp_record["ai_selected_index"]

    if (
        real_record["ai_crop_coords"] is None
        and temp_record["ai_crop_coords"] is not None
    ):
        fields_to_update["ai_crop_coords"] = temp_record["ai_crop_coords"]

    if (
        real_record["ai_confidence"] is None
        and temp_record["ai_confidence"] is not None
    ):
        fields_to_update["ai_confidence"] = temp_record["ai_confidence"]

    # Match fields - only fill if NULL in real DB
    if (
        real_record["selection_match"] is None
        and temp_record["selection_match"] is not None
    ):
        fields_to_update["selection_match"] = temp_record["selection_match"]

    if real_record["crop_match"] is None and temp_record["crop_match"] is not None:
        fields_to_update["crop_match"] = temp_record["crop_match"]

    # User action - only fill if NULL or empty in real DB
    if (
        real_record["user_action"] is None or real_record["user_action"] == ""
    ) and temp_record["user_action"]:
        fields_to_update["user_action"] = temp_record["user_action"]

    # User selected index - only fill if NULL in real DB
    if (
        real_record.get("user_selected_index") is None
        and temp_record.get("user_selected_index") is not None
    ):
        fields_to_update["user_selected_index"] = temp_record["user_selected_index"]

    if fields_to_update:
        # Determine if this is a coordinate correction or null filling
        if "final_crop_coords" in fields_to_update and real_coords is not None:
            return (
                "update_coords",
                {
                    "group_id": real_record["group_id"],
                    "updates": fields_to_update,
                    "old_coords": real_coords,
                },
            )
        else:
            return (
                "fill_nulls",
                {"group_id": real_record["group_id"], "updates": fields_to_update},
            )

    return ("skip", {})


def add_record(cursor: sqlite3.Cursor, record: Dict[str, Any], dry_run: bool = True):
    """Add a new record to the real database."""
    if dry_run:
        return

    cursor.execute(
        """
        INSERT INTO ai_decisions (
            group_id, timestamp, project_id, directory, batch_number,
            images, ai_selected_index, ai_crop_coords, ai_confidence,
            user_selected_index, user_action, final_crop_coords,
            crop_timestamp, image_width, image_height,
            selection_match, crop_match
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            record["group_id"],
            record["timestamp"],
            record["project_id"],
            record["directory"],
            record["batch_number"],
            record["images"],
            record["ai_selected_index"],
            record["ai_crop_coords"],
            record["ai_confidence"],
            record["user_selected_index"],
            record["user_action"],
            record["final_crop_coords"],
            record["crop_timestamp"],
            record["image_width"],
            record["image_height"],
            record["selection_match"],
            record["crop_match"],
        ),
    )


def fill_null_fields(
    cursor: sqlite3.Cursor, group_id: str, updates: Dict[str, Any], dry_run: bool = True
):
    """
    Update NULL fields in existing record.

    CRITICAL: NEVER update timestamp or crop_timestamp fields.
    """
    if dry_run:
        return

    # Remove any timestamp fields if they somehow got included
    updates_safe = {
        k: v for k, v in updates.items() if k not in ["timestamp", "crop_timestamp"]
    }

    if not updates_safe:
        return  # Nothing to update

    set_clauses = []
    values = []

    for field, value in updates_safe.items():
        set_clauses.append(f"{field} = ?")
        values.append(value)

    values.append(group_id)  # For WHERE clause

    sql = f"UPDATE ai_decisions SET {', '.join(set_clauses)} WHERE group_id = ?"
    cursor.execute(sql, values)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Compare and merge backfill data"
    )
    parser.add_argument(
        "--temp-db", required=True, help="Path to temporary backfill database"
    )
    parser.add_argument(
        "--real-db", required=True, help="Path to real production database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying database",
    )
    args = parser.parse_args()

    temp_db_path = Path(args.temp_db)
    real_db_path = Path(args.real_db)

    print("=" * 80)
    print("PHASE 2: DATABASE COMPARISON & MERGE")
    print("=" * 80)
    print(f"Temp DB: {temp_db_path}")
    print(f"Real DB: {real_db_path}")
    print(
        f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will modify database)'}"
    )
    print()

    # Verify files exist
    if not temp_db_path.exists():
        print(f"❌ ERROR: Temp database not found: {temp_db_path}")
        return

    if not real_db_path.exists():
        print(f"❌ ERROR: Real database not found: {real_db_path}")
        return

    print("Loading databases...")
    temp_records = get_all_records(temp_db_path)
    real_records = get_all_records(real_db_path)

    print(f"  Temp DB: {len(temp_records):,} records")
    print(f"  Real DB: {len(real_records):,} records")
    print()

    # Build image filename to group_id maps for matching
    print("Building image filename maps...")
    temp_image_map = build_image_to_record_map(temp_records)
    real_image_map = build_image_to_record_map(real_records)

    print(f"  Temp DB: {len(temp_image_map):,} unique images")
    print(f"  Real DB: {len(real_image_map):,} unique images")
    print()

    # Find overlapping images (same filename in both databases)
    overlapping_images = set(temp_image_map.keys()) & set(real_image_map.keys())
    print(f"  Overlapping images: {len(overlapping_images):,}")
    print()

    # Compare all records
    print("Comparing records...")
    actions = {"add": [], "fill_nulls": [], "update_coords": [], "skip": []}

    # Track which temp records matched with real records
    matched_temp_group_ids = set()

    # First, check for overlapping images and compare their records
    for img_filename in overlapping_images:
        temp_group_id = temp_image_map[img_filename]
        real_group_id = real_image_map[img_filename]

        temp_record = temp_records[temp_group_id]
        real_record = real_records[real_group_id]

        action, data = compare_records(temp_record, real_record)
        actions[action].append((real_group_id, data))  # Use real group_id for updates
        matched_temp_group_ids.add(temp_group_id)

    # Then, add any temp records that didn't match (new groups to add)
    for temp_group_id, temp_record in temp_records.items():
        if temp_group_id not in matched_temp_group_ids:
            actions["add"].append((temp_group_id, temp_record))

    # Display summary
    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"Records to ADD:       {len(actions['add']):,}")
    print(f"Records to UPDATE:    {len(actions['fill_nulls']):,}")
    print(
        f"Coords to CORRECT:    {len(actions['update_coords']):,} ⚠️  (temp is source of truth)"
    )
    print(f"Records to SKIP:      {len(actions['skip']):,}")
    print()

    # Show sample actions
    if actions["add"]:
        print("Sample records to ADD (first 5):")
        for i, (group_id, record) in enumerate(actions["add"][:5]):
            print(f"  {i+1}. {group_id}")
            print(
                f"     - AI: index={record['ai_selected_index']}, confidence={record['ai_confidence']:.2f}"
            )
            print(
                f"     - User: action={record['user_action']}, coords={record['final_crop_coords'][:50] if record['final_crop_coords'] else None}"
            )
        if len(actions["add"]) > 5:
            print(f"  ... and {len(actions['add']) - 5:,} more")
        print()

    if actions["update_coords"]:
        print("⚠️  Sample COORDINATE CORRECTIONS (first 5):")
        print(
            "    (Temp coordinates extracted from physical images are source of truth)"
        )
        for i, (group_id, data) in enumerate(actions["update_coords"][:5]):
            print(f"  {i+1}. {group_id}")
            print(f"     - OLD: {data['old_coords'][:60]}...")
            new_coords = data["updates"]["final_crop_coords"]
            print(f"     - NEW: {new_coords[:60]}...")
        if len(actions["update_coords"]) > 5:
            print(f"  ... and {len(actions['update_coords']) - 5:,} more")
        print()

    if actions["fill_nulls"]:
        print("Sample records to UPDATE (first 5):")
        for i, (group_id, data) in enumerate(actions["fill_nulls"][:5]):
            print(f"  {i+1}. {group_id}")
            for field, value in data["updates"].items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"     - Fill {field}: {value}")
        if len(actions["fill_nulls"]) > 5:
            print(f"  ... and {len(actions['fill_nulls']) - 5:,} more")
        print()

    # Check for records in real DB but not in temp (shouldn't happen, just informational)
    real_only = set(real_records.keys()) - set(temp_records.keys())
    if real_only:
        print(f"Records in real DB but NOT in temp: {len(real_only):,}")
        print("  (These will be left unchanged - this is expected)")
        print()

    # Execute changes
    if args.dry_run:
        print("=" * 80)
        print("🔍 DRY RUN COMPLETE - No changes made")
        print("=" * 80)
        print()
        print("To actually merge the data, run without --dry-run:")
        print("  python scripts/ai/backfill_project_phase2_compare.py \\")
        print(f"    --temp-db {temp_db_path} \\")
        print(f"    --real-db {real_db_path}")
        print()
        print("⚠️  RECOMMENDATION: Review the samples above before proceeding!")
    else:
        print("=" * 80)
        print("⚠️  APPLYING CHANGES TO REAL DATABASE")
        print("=" * 80)

        conn = sqlite3.connect(str(real_db_path))
        cursor = conn.cursor()

        try:
            # Add new records
            print(f"Adding {len(actions['add']):,} new records...")
            for group_id, record in actions["add"]:
                add_record(cursor, record, dry_run=False)

            # Update existing records (fill NULLs)
            print(
                f"Updating {len(actions['fill_nulls']):,} existing records (filling NULLs)..."
            )
            for group_id, data in actions["fill_nulls"]:
                fill_null_fields(
                    cursor, data["group_id"], data["updates"], dry_run=False
                )

            # Correct coordinates
            print(
                f"Correcting {len(actions['update_coords']):,} crop coordinates (temp is source of truth)..."
            )
            for group_id, data in actions["update_coords"]:
                fill_null_fields(
                    cursor, data["group_id"], data["updates"], dry_run=False
                )

            conn.commit()
            print()
            print("✅ MERGE COMPLETE!")
            print()
            print("Summary:")
            print(f"  - Added: {len(actions['add']):,} records")
            print(f"  - Updated: {len(actions['fill_nulls']):,} records (filled NULLs)")
            print(
                f"  - Corrected: {len(actions['update_coords']):,} records (fixed coordinates)"
            )
            print(f"  - Skipped: {len(actions['skip']):,} records (already complete)")
            print(
                f"  - Total in real DB: {len(real_records) + len(actions['add']):,} records"
            )

        except Exception as e:
            print(f"❌ ERROR during merge: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

        print()
        print("=" * 80)
        print("🎉 BACKFILL COMPLETE!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Validate the merged data:")
        print("     python scripts/ai/validate_training_data.py <project_id>")
        print("  2. Check AI performance stats:")
        print(f"     sqlite3 {real_db_path}")
        print(
            "     SELECT COUNT(*), AVG(selection_match), AVG(crop_match) FROM ai_decisions;"
        )


if __name__ == "__main__":
    main()
