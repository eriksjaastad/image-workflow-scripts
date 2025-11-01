#!/usr/bin/env python3
"""
Phase 1: Backfill mojo3 Database with AI Predictions

This script:
1. Groups original images from mojo3
2. Runs trained AI models to get predictions (selection + crop)
3. Matches against final directory to find user choices
4. Extracts crop coordinates via template matching
5. Creates temporary database with BOTH AI predictions and user ground truth

Output: mojo3_backfill_temp.db

Usage:
    python scripts/ai/backfill_mojo3_with_ai_phase1.py
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

# Add project root to path
WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

# Import grouping utilities
from scripts.utils.companion_file_utils import (
    find_consecutive_stage_groups,
    sort_image_files_by_timestamp_and_stage,
)

# Paths
ORIGINAL_DIR = Path("/Volumes/T7Shield/Eros/original/mojo3")
FINAL_DIR = WORKSPACE / "mojo3"
TEMP_DB = (
    WORKSPACE / "data" / "training" / "ai_training_decisions" / "mojo3_backfill_temp.db"
)
PROJECT_ID = "mojo3"

# AI Model paths (will check if they exist)
RANKER_MODEL_PATH = WORKSPACE / "data" / "ai_data" / "models" / "ranker_v2.pth"
CROP_MODEL_PATH = WORKSPACE / "data" / "ai_data" / "models" / "crop_proposer.pth"


def load_ai_models():
    """Load trained AI models if available."""
    try:
        import torch

        # Check if models exist
        if not RANKER_MODEL_PATH.exists() or not CROP_MODEL_PATH.exists():
            return None, None

        # TODO: Load actual models - this needs the model architecture classes
        # For now, return None to indicate models aren't loaded
        # You'll need to implement actual model loading based on your training code

        return None, None

    except ImportError:
        return None, None


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


def create_temp_database(db_path: Path):
    """Create temporary database with schema."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_decisions (
            group_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            project_id TEXT NOT NULL,
            directory TEXT,
            batch_number INTEGER,
            images TEXT NOT NULL,
            ai_selected_index INTEGER,
            ai_crop_coords TEXT,
            ai_confidence REAL,
            user_selected_index INTEGER NOT NULL,
            user_action TEXT NOT NULL,
            final_crop_coords TEXT,
            crop_timestamp TEXT,
            image_width INTEGER NOT NULL,
            image_height INTEGER NOT NULL,
            selection_match BOOLEAN,
            crop_match BOOLEAN,
            ai_crop_accepted BOOLEAN
        )
    """
    )

    conn.commit()
    conn.close()


def group_original_images(original_dir: Path) -> list[tuple[str, list[Path]]]:
    """Group original images using the grouping logic."""
    # Get all image files
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg"]
    all_images = []
    for pattern in patterns:
        all_images.extend(original_dir.glob(pattern))

    # Sort by timestamp and stage
    sorted_images = sort_image_files_by_timestamp_and_stage(all_images)

    # Group consecutive images
    groups = find_consecutive_stage_groups(sorted_images)

    return groups


def find_selected_image(group: list[Path], final_dir: Path) -> tuple[int, Path] | None:
    """Find which image from the group exists in final directory."""
    for idx, img_path in enumerate(group):
        filename = img_path.name

        # Search recursively in final directory
        matches = list(final_dir.glob(f"**/{filename}"))
        if matches:
            return idx, matches[0]

    return None


def process_groups(
    groups: list[tuple[str, list[Path]]], final_dir: Path, ranker=None, cropper=None
):
    """Process all groups and create database records."""
    records = []

    stats = {
        "total_groups": len(groups),
        "found_in_final": 0,
        "not_in_final": 0,
        "coord_extracted": 0,
        "coord_failed": 0,
    }

    for i, (group_id, group_images) in enumerate(groups, 1):
        if i % 100 == 0:
            pass

        # Find which image was selected
        selection_result = find_selected_image(group_images, final_dir)

        if not selection_result:
            stats["not_in_final"] += 1
            continue

        user_selected_index, final_image_path = selection_result
        stats["found_in_final"] += 1

        # Get image dimensions from selected image
        try:
            with Image.open(group_images[user_selected_index]) as img:
                image_width, image_height = img.size
        except Exception:
            continue

        # Extract crop coordinates
        crop_result = extract_crop_coordinates(
            group_images[user_selected_index], final_image_path
        )

        if crop_result:
            final_crop_coords, _confidence = crop_result
            stats["coord_extracted"] += 1
        else:
            final_crop_coords = None
            stats["coord_failed"] += 1

        # TODO: Run AI models here to get predictions
        # For now, AI fields are NULL
        ai_selected_index = None
        ai_crop_coords = None
        ai_confidence = None
        selection_match = None
        crop_match = None
        ai_crop_accepted = None

        # Create record
        record = {
            "group_id": group_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "project_id": PROJECT_ID,
            "directory": str(group_images[0].parent),
            "batch_number": None,
            "images": json.dumps([img.name for img in group_images]),
            "ai_selected_index": ai_selected_index,
            "ai_crop_coords": json.dumps(ai_crop_coords) if ai_crop_coords else None,
            "ai_confidence": ai_confidence,
            "user_selected_index": user_selected_index,
            "user_action": "crop",
            "final_crop_coords": (
                json.dumps(final_crop_coords) if final_crop_coords else None
            ),
            "crop_timestamp": (
                datetime.now().isoformat() + "Z" if final_crop_coords else None
            ),
            "image_width": image_width,
            "image_height": image_height,
            "selection_match": selection_match,
            "crop_match": crop_match,
            "ai_crop_accepted": ai_crop_accepted,
        }

        records.append(record)

    return records, stats


def write_records_to_database(records: list[dict], db_path: Path):
    """Write records to temporary database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    for record in records:
        cursor.execute(
            """
            INSERT INTO ai_decisions (
                group_id, timestamp, project_id, directory, batch_number,
                images, ai_selected_index, ai_crop_coords, ai_confidence,
                user_selected_index, user_action, final_crop_coords, crop_timestamp,
                image_width, image_height, selection_match, crop_match, ai_crop_accepted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                record["ai_crop_accepted"],
            ),
        )

    conn.commit()
    conn.close()


def main():
    # Check directories exist
    if not ORIGINAL_DIR.exists():
        return

    if not FINAL_DIR.exists():
        return

    # Load AI models
    ranker, cropper = load_ai_models()

    # Create temporary database
    if TEMP_DB.exists():
        response = input("Delete and recreate? (yes/no): ")
        if response.lower() != "yes":
            return
        TEMP_DB.unlink()

    TEMP_DB.parent.mkdir(parents=True, exist_ok=True)
    create_temp_database(TEMP_DB)

    # Group original images
    groups = group_original_images(ORIGINAL_DIR)

    # Process groups
    records, _stats = process_groups(groups, FINAL_DIR, ranker, cropper)

    # Write to database
    write_records_to_database(records, TEMP_DB)

    # Print statistics


if __name__ == "__main__":
    main()
