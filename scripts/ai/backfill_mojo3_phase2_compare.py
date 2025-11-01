#!/usr/bin/env python3
"""
Backfill Mojo3 - Phase 2: Compare and Merge Databases

This script compares the temporary backfill database with the real mojo3.db
and merges data intelligently:
- Adds records that exist in temp but not in real database
- Fills NULL fields in real database from temp database
- NEVER overwrites existing data in real database
- Preserves all existing AI predictions

Usage:
    python scripts/ai/backfill_mojo3_phase2_compare.py --dry-run
    python scripts/ai/backfill_mojo3_phase2_compare.py  # Actually merge
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_DB_PATH = (
    PROJECT_ROOT / "data/training/ai_training_decisions/mojo3_backfill_temp.db"
)
REAL_DB_PATH = PROJECT_ROOT / "data/training/ai_training_decisions/mojo3.db"


def get_all_records(db_path: Path) -> dict[str, dict[str, Any]]:
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


def build_image_to_record_map(records: dict[str, dict[str, Any]]) -> dict[str, str]:
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
    coords1_json: str | None, coords2_json: str | None, tolerance: float = 0.01
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
        return all(
            abs(c1 - c2) <= tolerance for c1, c2 in zip(coords1, coords2, strict=False)
        )
    except Exception:
        return False


def compare_records(
    temp_record: dict[str, Any], real_record: dict[str, Any] | None
) -> tuple[str, dict[str, Any]]:
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
        return (
            "fill_nulls",
            {"group_id": real_record["group_id"], "updates": fields_to_update},
        )

    return ("skip", {})


def add_record(cursor: sqlite3.Cursor, record: dict[str, Any], dry_run: bool = True):
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
    cursor: sqlite3.Cursor, group_id: str, updates: dict[str, Any], dry_run: bool = True
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
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying database",
    )
    args = parser.parse_args()

    # Verify files exist
    if not TEMP_DB_PATH.exists():
        return

    if not REAL_DB_PATH.exists():
        return

    temp_records = get_all_records(TEMP_DB_PATH)
    real_records = get_all_records(REAL_DB_PATH)

    # Build image filename to group_id maps for matching
    temp_image_map = build_image_to_record_map(temp_records)
    real_image_map = build_image_to_record_map(real_records)

    # Find overlapping images (same filename in both databases)
    overlapping_images = set(temp_image_map.keys()) & set(real_image_map.keys())

    # Compare all records
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

    # Show sample actions
    if actions["add"]:
        for _i, (_group_id, record) in enumerate(actions["add"][:5]):
            pass
        if len(actions["add"]) > 5:
            pass

    if actions["update_coords"]:
        for _i, (_group_id, data) in enumerate(actions["update_coords"][:5]):
            data["updates"]["final_crop_coords"]
        if len(actions["update_coords"]) > 5:
            pass

    if actions["fill_nulls"]:
        for _i, (_group_id, data) in enumerate(actions["fill_nulls"][:5]):
            for _field, value in data["updates"].items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
        if len(actions["fill_nulls"]) > 5:
            pass

    # Check for records in real DB but not in temp (shouldn't happen, just informational)
    real_only = set(real_records.keys()) - set(temp_records.keys())
    if real_only:
        pass

    # Execute changes
    if args.dry_run:
        pass
    else:
        conn = sqlite3.connect(str(REAL_DB_PATH))
        cursor = conn.cursor()

        try:
            # Add new records
            for _group_id, record in actions["add"]:
                add_record(cursor, record, dry_run=False)

            # Update existing records (fill NULLs)
            for _group_id, data in actions["fill_nulls"]:
                fill_null_fields(
                    cursor, data["group_id"], data["updates"], dry_run=False
                )

            # Correct coordinates
            for _group_id, data in actions["update_coords"]:
                fill_null_fields(
                    cursor, data["group_id"], data["updates"], dry_run=False
                )

            conn.commit()

        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


if __name__ == "__main__":
    main()
