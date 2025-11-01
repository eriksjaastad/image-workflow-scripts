#!/usr/bin/env python3
r"""
Phase 1B: Backfill User Ground Truth into Temp Database

This script:
1. Reads all records from temp database (which has AI predictions)
2. For each group, finds which image exists in final directory
3. Extracts crop coordinates via template matching
4. Updates database with user ground truth
5. Calculates selection_match and crop_match metrics

Input: temp database (with AI predictions, user fields NULL)
Output: temp database (complete with both AI and user data)

Usage:
    python scripts/ai/backfill_project_phase1b_user_data.py \\
        --temp-db data/training/ai_training_decisions/mojo3_backfill_temp.db \\
        --final-dir mojo3/
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm

# Add project root to path
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))


def extract_crop_coordinates(
    original_path: Path, cropped_path: Path, confidence_threshold: float = 0.8
) -> tuple[list[float], float] | None:
    """Extract crop coordinates using template matching."""
    try:
        original = cv2.imread(str(original_path))
        cropped = cv2.imread(str(cropped_path))

        if original is None or cropped is None:
            return None

        orig_height, orig_width = original.shape[:2]
        crop_height, crop_width = cropped.shape[:2]

        if crop_width > orig_width or crop_height > orig_height:
            return None

        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(original_gray, cropped_gray, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)

        confidence = max_val
        if confidence < confidence_threshold:
            return None

        x1, y1 = max_loc
        x2, y2 = x1 + crop_width, y1 + crop_height

        # Normalize coordinates
        coords = [x1 / orig_width, y1 / orig_height, x2 / orig_width, y2 / orig_height]

        return coords, confidence

    except Exception:
        return None


def coords_match(
    coords1: list[float], coords2: list[float], tolerance: float = 0.05
) -> bool:
    """Check if two coordinate sets match within tolerance."""
    if not coords1 or not coords2:
        return False

    if len(coords1) != 4 or len(coords2) != 4:
        return False

    return all(
        abs(c1 - c2) <= tolerance for c1, c2 in zip(coords1, coords2, strict=False)
    )


def find_user_selected_image(
    group_images: list[str], group_directory: str, final_dir: Path
) -> tuple[int, Path] | None:
    """Find which image from the group exists in final directory."""
    # Try each image in the group
    for idx, filename in enumerate(group_images):
        # Search recursively in final directory
        matches = list(final_dir.glob(f"**/{filename}"))
        if matches:
            return idx, matches[0]

    return None


def get_all_records_from_db(db_path: Path) -> list[dict]:
    """Read all records from database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT group_id, images, directory, ai_selected_index, ai_crop_coords
        FROM ai_decisions
        ORDER BY timestamp
    """
    )

    results = cursor.fetchall()
    conn.close()

    records = []
    for (
        group_id,
        images_json,
        directory,
        ai_selected_index,
        ai_crop_coords_json,
    ) in results:
        try:
            images = json.loads(images_json)
            ai_crop_coords = (
                json.loads(ai_crop_coords_json) if ai_crop_coords_json else None
            )

            records.append(
                {
                    "group_id": group_id,
                    "images": images,
                    "directory": directory,
                    "ai_selected_index": ai_selected_index,
                    "ai_crop_coords": ai_crop_coords,
                }
            )
        except Exception:
            continue

    return records


def update_record_with_user_data(
    db_path: Path,
    group_id: str,
    user_selected_index: int,
    user_action: str,
    final_crop_coords: list[float] | None,
    selection_match: bool,
    crop_match: bool | None,
    dry_run: bool = False,
):
    """Update database record with user ground truth."""
    if dry_run:
        return  # Don't actually update in dry-run mode

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    crop_coords_json = json.dumps(final_crop_coords) if final_crop_coords else None
    crop_timestamp = datetime.now().isoformat() + "Z" if final_crop_coords else None

    cursor.execute(
        """
        UPDATE ai_decisions
        SET user_selected_index = ?,
            user_action = ?,
            final_crop_coords = ?,
            crop_timestamp = ?,
            selection_match = ?,
            crop_match = ?
        WHERE group_id = ?
    """,
        (
            user_selected_index,
            user_action,
            crop_coords_json,
            crop_timestamp,
            selection_match,
            crop_match,
            group_id,
        ),
    )

    conn.commit()
    conn.close()


def process_records(
    records: list[dict], final_dir: Path, db_path: Path, dry_run: bool = False
):
    """Process all records and backfill user data."""
    stats = {
        "total_records": len(records),
        "found_in_final": 0,
        "not_in_final": 0,
        "crop_coords_extracted": 0,
        "crop_coords_failed": 0,
        "selection_matches": 0,
        "crop_matches": 0,
        "errors": 0,
    }

    samples = []  # Collect sample updates for dry-run

    for record in tqdm(records, desc="Backfilling" if not dry_run else "Analyzing"):
        try:
            group_id = record["group_id"]
            images = record["images"]
            directory = record["directory"]
            ai_selected_index = record["ai_selected_index"]
            ai_crop_coords = record["ai_crop_coords"]

            # Find which image exists in final directory
            result = find_user_selected_image(images, directory, final_dir)

            if not result:
                # User rejected/skipped this entire group
                stats["not_in_final"] += 1
                update_record_with_user_data(
                    db_path,
                    group_id,
                    user_selected_index=None,
                    user_action="skip",
                    final_crop_coords=None,
                    selection_match=None,
                    crop_match=None,
                    dry_run=dry_run,
                )
                continue

            user_selected_index, final_image_path = result
            stats["found_in_final"] += 1

            # Calculate selection match
            selection_match = (
                (ai_selected_index == user_selected_index)
                if ai_selected_index is not None
                else None
            )
            if selection_match:
                stats["selection_matches"] += 1

            # Extract crop coordinates from physical files
            original_path = Path(directory) / images[user_selected_index]

            crop_result = extract_crop_coordinates(original_path, final_image_path)

            if crop_result:
                final_crop_coords, _confidence = crop_result
                stats["crop_coords_extracted"] += 1

                # Calculate crop match
                crop_match = (
                    coords_match(ai_crop_coords, final_crop_coords)
                    if ai_crop_coords
                    else None
                )
                if crop_match:
                    stats["crop_matches"] += 1
            else:
                final_crop_coords = None
                crop_match = None
                stats["crop_coords_failed"] += 1

            # Collect sample for dry-run
            if dry_run and len(samples) < 10:
                samples.append(
                    {
                        "group_id": group_id,
                        "ai_selected": ai_selected_index,
                        "user_selected": user_selected_index,
                        "selection_match": selection_match,
                        "crop_match": crop_match,
                    }
                )

            # Update database
            update_record_with_user_data(
                db_path,
                group_id,
                user_selected_index=user_selected_index,
                user_action="crop",
                final_crop_coords=final_crop_coords,
                selection_match=selection_match,
                crop_match=crop_match,
                dry_run=dry_run,
            )

        except Exception:
            stats["errors"] += 1

    return stats, samples


def main():
    parser = argparse.ArgumentParser(
        description="Backfill user ground truth into temp database"
    )
    parser.add_argument(
        "--temp-db",
        required=True,
        help="Path to temp database (e.g., data/training/ai_training_decisions/mojo3_backfill_temp.db)",
    )
    parser.add_argument(
        "--final-dir",
        required=True,
        help="Directory containing final cropped images (e.g., mojo3/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying database",
    )
    args = parser.parse_args()

    temp_db = Path(args.temp_db)
    final_dir = Path(args.final_dir)

    # Check database exists
    if not temp_db.exists():
        return

    # Check final directory exists
    if not final_dir.exists():
        return

    # Read all records
    records = get_all_records_from_db(temp_db)

    # Process records
    stats, samples = process_records(records, final_dir, temp_db, dry_run=args.dry_run)

    # Print statistics

    if stats["found_in_final"] > 0:
        stats["selection_matches"] / stats["found_in_final"] * 100

    if stats["crop_coords_extracted"] > 0:
        stats["crop_matches"] / stats["crop_coords_extracted"] * 100

    if args.dry_run and samples:
        for _i, sample in enumerate(samples, 1):
            "✅ MATCH" if sample["selection_match"] else "❌ MISMATCH"

    if args.dry_run or stats["found_in_final"] > 0:
        pass


if __name__ == "__main__":
    main()
