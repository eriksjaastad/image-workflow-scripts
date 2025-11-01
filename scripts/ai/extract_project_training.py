#!/usr/bin/env python3
"""
Extract training data from any completed project.

Uses the SAME grouping logic as Web Image Selector to find image groups,
then extracts selections and crops based on what appears in the final directory.

Usage:
    python scripts/ai/extract_project_training.py <project_id>
    python scripts/ai/extract_project_training.py mixed-0919 --dry-run
    python scripts/ai/extract_project_training.py tattersail-0918

Output:
    Appends to data/training/selection_only_log.csv
    Appends to data/training/select_crop_log.csv
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.companion_file_utils import (
    detect_stage,
    extract_datetime_from_filename,
    find_consecutive_stage_groups,
    sort_image_files_by_timestamp_and_stage,
)


def load_project_manifest(project_id: str) -> dict | None:
    """Load project manifest to get metadata."""
    manifest_path = Path(f"data/projects/{project_id}.project.json")

    if not manifest_path.exists():
        return None

    with manifest_path.open("r") as f:
        return json.load(f)


def scan_images(directory: Path) -> list[Path]:
    """Recursively find all PNG images."""
    return list(directory.rglob("*.png"))


def find_groups(images: list[Path]) -> list[list[Path]]:
    """
    Group images using EXACT same logic as Web Image Selector.

    Returns:
        List of groups, each group is a list of Paths
    """
    # Sort first (required by grouping logic)
    sorted_images = sort_image_files_by_timestamp_and_stage(images)

    # Group by timestamp and stage progression
    grouped = find_consecutive_stage_groups(sorted_images, min_group_size=2)

    return grouped


def extract_training_data(
    raw_dir: Path, final_dir: Path, project_id: str
) -> tuple[list[dict], list[dict]]:
    """
    Extract selections and crops from a project.

    Returns:
        (selections, crops)
    """
    # Scan raw images
    raw_images = scan_images(raw_dir)

    # Scan final images
    final_images = scan_images(final_dir)

    # Build lookup of final image filenames
    final_filenames = {img.name for img in final_images}

    # Group raw images using Web Image Selector logic
    groups = find_groups(raw_images)

    if not groups:
        return [], []

    # Extract selections and crops
    selections = []
    crops = []
    selections_count = 0
    crops_count = 0

    for group in groups:
        # Find which image(s) from this group appear in final
        winners = [img for img in group if img.name in final_filenames]

        if not winners:
            # Whole group was rejected - no training data
            continue

        if len(winners) > 1:
            # Multiple winners? Shouldn't happen, but skip to be safe
            continue

        winner = winners[0]
        losers = [img for img in group if img.name not in final_filenames]

        if not losers:
            # No losers means this was the only option - no selection made
            continue

        # Create timestamp for this group
        dt = extract_datetime_from_filename(winner.name)
        if dt:
            group_id = dt.strftime("%Y%m%d_%H%M%S")
            timestamp_iso = dt.replace(tzinfo=UTC).isoformat()
        else:
            # Fallback: use filename stem
            group_id = (
                winner.stem.split("_stage")[0]
                if "_stage" in winner.stem
                else winner.stem
            )
            timestamp_iso = datetime.now(UTC).isoformat()

        # Record selection
        selections.append(
            {
                "session_id": f"{project_id}_extraction",
                "set_id": f"group_{group_id}",
                "chosen_path": str(winner),
                "neg_paths": json.dumps([str(img) for img in losers]),
                "timestamp": timestamp_iso,
            }
        )
        selections_count += 1

        # Check if image was cropped (compare mtimes)
        # Find the raw version of the winner
        raw_winner = None
        for img in group:
            if img.name == winner.name:
                raw_winner = img
                break

        if raw_winner and raw_winner.exists():
            # Find corresponding final image
            final_winner = None
            for img in final_images:
                if img.name == winner.name:
                    final_winner = img
                    break

            if final_winner and final_winner.exists():
                raw_mtime = datetime.fromtimestamp(raw_winner.stat().st_mtime, tz=UTC)
                final_mtime = datetime.fromtimestamp(
                    final_winner.stat().st_mtime, tz=UTC
                )

                # If mtimes differ by more than 1 second, it was cropped
                was_cropped = abs((final_mtime - raw_mtime).total_seconds()) > 1.0

                if was_cropped:
                    # Get stage if available
                    stage_str = detect_stage(winner.name)
                    stage = None
                    if stage_str:
                        try:
                            # Extract number from stage string (e.g., "stage2" -> 2)
                            import re

                            match = re.search(r"stage([\d.]+)", stage_str)
                            if match:
                                stage = float(match.group(1))
                        except Exception:
                            pass

                    crops.append(
                        {
                            "session_id": f"{project_id}_extraction",
                            "directory": project_id,
                            "chosen_path": str(final_winner),
                            "chosen_stage": stage if stage else "",
                            "crop_x1": 0.0,  # Placeholder - actual coords unknown
                            "crop_y1": 0.0,
                            "crop_x2": 1.0,
                            "crop_y2": 1.0,
                            "timestamp": final_mtime.isoformat(),
                        }
                    )
                    crops_count += 1

    return selections, crops


def write_to_csv(selections: list[dict], crops: list[dict], project_root: Path) -> None:
    """Append selections and crops to training logs."""
    # Write selections
    if selections:
        selection_log = project_root / "data/training/selection_only_log.csv"
        file_exists = selection_log.exists()

        fieldnames = ["session_id", "set_id", "chosen_path", "neg_paths", "timestamp"]

        with selection_log.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(selections)

    # Write crops
    if crops:
        crop_log = project_root / "data/training/select_crop_log.csv"
        file_exists = crop_log.exists()

        fieldnames = [
            "session_id",
            "directory",
            "chosen_path",
            "chosen_stage",
            "crop_x1",
            "crop_y1",
            "crop_x2",
            "crop_y2",
            "timestamp",
        ]

        with crop_log.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(crops)


def main():
    parser = argparse.ArgumentParser(
        description="Extract training data from a completed project"
    )
    parser.add_argument(
        "project_id", help="Project ID (e.g., mixed-0919, tattersail-0918)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without writing",
    )

    args = parser.parse_args()
    project_id = args.project_id

    # Load project manifest
    manifest = load_project_manifest(project_id)
    if manifest:
        pass

    # Define paths
    project_root = Path("/Users/eriksjaastad/projects/Eros Mate")
    raw_dir = project_root / "training data" / project_id
    final_dir = project_root / "training data" / f"{project_id}_final"

    # Validate directories exist
    if not raw_dir.exists():
        return 1

    if not final_dir.exists():
        return 1

    # Extract training data
    selections, crops = extract_training_data(raw_dir, final_dir, project_id)

    if not selections:
        return 1

    if args.dry_run:
        return 0

    # Write to CSV logs
    write_to_csv(selections, crops, project_root)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
