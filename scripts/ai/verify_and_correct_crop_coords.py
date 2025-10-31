#!/usr/bin/env python3
"""
Verify and Correct Crop Coordinates from Physical Images

SOURCE OF TRUTH: Physical cropped images on disk
VERIFICATION TARGET: Database final_crop_coords

This script:
1. Scans all cropped images in the project directory
2. Extracts ACTUAL crop coordinates via template matching with originals
3. Compares against database coordinates
4. Reports mismatches
5. Optionally corrects database to match physical reality

CRITICAL: This treats the physical files as ground truth, not the database.

Usage:
    # Generate inspection report (dry-run)
    python scripts/ai/verify_and_correct_crop_coords.py mojo3 \\
        --cropped-dir mojo3/ \\
        --original-dir /Volumes/T7Shield/Eros/original/mojo3 \\
        --database data/training/ai_training_decisions/mojo3.db \\
        --dry-run
    
    # Correct database mismatches
    python scripts/ai/verify_and_correct_crop_coords.py mojo3 \\
        --cropped-dir mojo3/ \\
        --original-dir /Volumes/T7Shield/Eros/original/mojo3 \\
        --database data/training/ai_training_decisions/mojo3.db \\
        --execute
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Paths
WORKSPACE = Path(__file__).resolve().parents[2]


def extract_crop_coordinates(
    original_path: Path, cropped_path: Path, confidence_threshold: float = 0.8
) -> Optional[Tuple[List[float], float]]:
    """
    Extract crop coordinates using template matching.

    Returns: (normalized_coords, confidence) or None
    """
    try:
        # Load images
        original = cv2.imread(str(original_path))
        cropped = cv2.imread(str(cropped_path))

        if original is None or cropped is None:
            return None

        # Get dimensions
        orig_height, orig_width = original.shape[:2]
        crop_height, crop_width = cropped.shape[:2]

        # Skip if cropped image is larger than original
        if crop_width > orig_width or crop_height > orig_height:
            return None

        # Convert to grayscale for matching
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(original_gray, cropped_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        confidence = max_val
        if confidence < confidence_threshold:
            return None

        # Extract coordinates
        x1, y1 = max_loc
        x2, y2 = x1 + crop_width, y1 + crop_height

        # Normalize coordinates [0, 1]
        x1_norm = x1 / orig_width
        y1_norm = y1 / orig_height
        x2_norm = x2 / orig_width
        y2_norm = y2 / orig_height

        coords = [x1_norm, y1_norm, x2_norm, y2_norm]

        return coords, confidence

    except Exception as e:
        print(f"    [!] Error extracting coordinates: {e}")
        return None


def coords_match(
    coords1: List[float], coords2: List[float], tolerance: float = 0.001
) -> bool:
    """Check if two coordinate sets match within tolerance."""
    if not coords1 or not coords2:
        return False

    if len(coords1) != 4 or len(coords2) != 4:
        return False

    for c1, c2 in zip(coords1, coords2):
        if abs(c1 - c2) > tolerance:
            return False

    return True


def format_coords(coords: List[float]) -> str:
    """Format coordinates for display."""
    return "[" + ", ".join(f"{x:.4f}" for x in coords) + "]"


class CropCoordinateValidator:
    """Validates and corrects crop coordinates against physical images."""

    def __init__(
        self,
        project_id: str,
        cropped_dir: Path,
        original_dir: Path,
        database_path: Path,
        dry_run: bool = True,
    ):
        self.project_id = project_id
        self.cropped_dir = Path(cropped_dir)
        self.original_dir = Path(original_dir)
        self.database_path = Path(database_path)
        self.dry_run = dry_run

        # Validation
        if not self.cropped_dir.exists():
            raise ValueError(f"Cropped directory not found: {self.cropped_dir}")
        if not self.original_dir.exists():
            raise ValueError(f"Original directory not found: {self.original_dir}")
        if not self.database_path.exists():
            raise ValueError(f"Database not found: {self.database_path}")

    def scan_cropped_images(self) -> List[Path]:
        """Find all cropped images recursively."""
        patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
        images = []
        for pattern in patterns:
            images.extend(self.cropped_dir.glob(pattern))
        return sorted(images)

    def get_crop_records_from_database(self) -> List[Dict]:
        """Get all crop action records from database."""
        conn = sqlite3.connect(str(self.database_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT group_id, images, user_selected_index, final_crop_coords, user_action
            FROM ai_decisions
            WHERE user_action = 'crop'
            ORDER BY timestamp
        """
        )

        results = cursor.fetchall()
        conn.close()

        records = []
        for group_id, images_json, user_idx, final_coords, user_action in results:
            try:
                images = json.loads(images_json)
                selected_filename = images[user_idx] if user_idx < len(images) else None
                coords = json.loads(final_coords) if final_coords else None

                if selected_filename:
                    records.append(
                        {
                            "group_id": group_id,
                            "images": images,
                            "user_selected_index": user_idx,
                            "selected_filename": selected_filename,
                            "final_crop_coords": coords,
                            "user_action": user_action,
                        }
                    )
            except Exception as e:
                print(f"Error parsing record {group_id}: {e}")
                continue

        return records
        """Get database record for an image filename."""
        conn = sqlite3.connect(str(self.database_path))
        cursor = conn.cursor()

        # Search for record containing this filename in images JSON array
        cursor.execute(
            """
            SELECT group_id, images, user_selected_index, final_crop_coords, user_action
            FROM ai_decisions
            WHERE images LIKE ?
        """,
            (f"%{filename}%",),
        )

        results = cursor.fetchall()
        conn.close()

        # Find exact match in images array
        for group_id, images_json, user_idx, final_coords, user_action in results:
            try:
                images = json.loads(images_json)
                if filename in images:
                    selected_filename = (
                        images[user_idx] if user_idx < len(images) else None
                    )
                    coords = json.loads(final_coords) if final_coords else None

                    return {
                        "group_id": group_id,
                        "images": images,
                        "user_selected_index": user_idx,
                        "selected_filename": selected_filename,
                        "final_crop_coords": coords,
                        "user_action": user_action,
                        "is_selected": (selected_filename == filename),
                    }
            except Exception:
                continue

        return None

    def validate_and_correct(self) -> Dict:
        """Validate all cropped images and report/correct mismatches."""
        print(f"\n{'=' * 80}")
        print(f"CROP COORDINATE VALIDATION - {self.project_id}")
        print(f"{'=' * 80}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE CORRECTION'}")
        print(f"Cropped images: {self.cropped_dir}")
        print(f"Original images: {self.original_dir}")
        print(f"Database: {self.database_path}")

        # Get all crop records from database
        print("\nQuerying database for crop actions...")
        crop_records = self.get_crop_records_from_database()
        print(f"Found {len(crop_records)} crop actions in database")

        results = {
            "total_crop_records": len(crop_records),
            "cropped_file_found": 0,
            "cropped_file_not_found": 0,
            "original_not_found": 0,
            "coordinates_match": 0,
            "coordinates_mismatch": 0,
            "template_match_failed": 0,
            "mismatches": [],
            "corrections_made": 0,
        }

        for i, record in enumerate(crop_records, 1):
            filename = record["selected_filename"]

            # Progress indicator every 100 images
            if i % 100 == 0:
                print(
                    f"\n[{i}/{len(crop_records)}] Progress: {i/len(crop_records)*100:.1f}% complete"
                )

            print(f"\n[{i}/{len(crop_records)}] Validating: {filename}")

            # Find cropped file (recursive search)
            cropped_matches = list(self.cropped_dir.glob(f"**/{filename}"))
            if not cropped_matches:
                print(f"    ⚠️  Cropped file not found in {self.cropped_dir}")
                results["cropped_file_not_found"] += 1
                continue

            cropped_path = cropped_matches[0]
            results["cropped_file_found"] += 1

            # Find original image (recursive search in original dir)
            original_matches = list(self.original_dir.glob(f"**/{filename}"))
            if not original_matches:
                print(f"    ⚠️  Original not found: {filename}")
                results["original_not_found"] += 1
                continue

            original_path = original_matches[0]

            # Extract coordinates from physical images
            match_result = extract_crop_coordinates(original_path, cropped_path)
            if not match_result:
                print("    ❌ Template matching failed")
                results["template_match_failed"] += 1
                continue

            physical_coords, confidence = match_result
            db_coords = record["final_crop_coords"]

            print(
                f"    Physical coords: {format_coords(physical_coords)} (confidence: {confidence:.4f})"
            )
            print(
                f"    Database coords: {format_coords(db_coords) if db_coords else 'NULL'}"
            )

            # Compare coordinates
            if coords_match(physical_coords, db_coords):
                print("    ✅ Match!")
                results["coordinates_match"] += 1
            else:
                print("    ⚠️  MISMATCH!")
                results["coordinates_mismatch"] += 1
                results["mismatches"].append(
                    {
                        "filename": filename,
                        "group_id": record["group_id"],
                        "physical_coords": physical_coords,
                        "database_coords": db_coords,
                        "confidence": confidence,
                    }
                )

                # Correct if not dry run
                if not self.dry_run:
                    if self.update_database_coords(record["group_id"], physical_coords):
                        print("    ✅ Corrected in database")
                        results["corrections_made"] += 1
                    else:
                        print("    ❌ Failed to update database")

        return results

    def update_database_coords(self, group_id: str, coords: List[float]) -> bool:
        """Update database with corrected coordinates."""
        try:
            conn = sqlite3.connect(str(self.database_path))
            cursor = conn.cursor()

            coords_json = json.dumps(coords)
            timestamp = datetime.now().isoformat() + "Z"

            cursor.execute(
                """
                UPDATE ai_decisions
                SET final_crop_coords = ?,
                    crop_timestamp = ?
                WHERE group_id = ?
            """,
                (coords_json, timestamp, group_id),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"    [!] Database update error: {e}")
            return False

    def print_summary(self, results: Dict):
        """Print validation summary."""
        print(f"\n{'=' * 80}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total crop records in DB:   {results['total_crop_records']}")
        print(f"Cropped file found:         {results['cropped_file_found']}")
        print(f"Cropped file not found:     {results['cropped_file_not_found']}")
        print(f"Original not found:         {results['original_not_found']}")
        print(f"Template matching failed:   {results['template_match_failed']}")
        print("")
        print(f"Coordinates match:          {results['coordinates_match']} ✅")
        print(f"Coordinates MISMATCH:       {results['coordinates_mismatch']} ⚠️")

        if results["coordinates_mismatch"] > 0:
            print("\n⚠️  MISMATCHES DETECTED:")
            for mismatch in results["mismatches"][:10]:  # Show first 10
                print(f"   {mismatch['filename']}")
                print(f"     Physical: {format_coords(mismatch['physical_coords'])}")
                print(
                    f"     Database: {format_coords(mismatch['database_coords']) if mismatch['database_coords'] else 'NULL'}"
                )

            if len(results["mismatches"]) > 10:
                print(f"   ... and {len(results['mismatches']) - 10} more")

        if not self.dry_run:
            print(f"\nCorrections made:           {results['corrections_made']}")
        else:
            print("\n💡 Run with --execute to apply corrections")

        print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and correct crop coordinates from physical images"
    )
    parser.add_argument("project_id", help="Project ID (e.g., mojo3)")
    parser.add_argument(
        "--cropped-dir", required=True, help="Directory with cropped images"
    )
    parser.add_argument(
        "--original-dir", required=True, help="Directory with original images"
    )
    parser.add_argument("--database", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report only, don't modify database"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Apply corrections to database"
    )

    args = parser.parse_args()

    # Default to dry-run unless --execute specified
    dry_run = not args.execute

    validator = CropCoordinateValidator(
        project_id=args.project_id,
        cropped_dir=args.cropped_dir,
        original_dir=args.original_dir,
        database_path=args.database,
        dry_run=dry_run,
    )

    results = validator.validate_and_correct()
    validator.print_summary(results)


if __name__ == "__main__":
    main()
