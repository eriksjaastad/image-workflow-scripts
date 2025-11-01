#!/usr/bin/env python3
"""
Extract training data from Mojo 1 project (historical data extraction)

This script:
1. Groups all raw images in mojo1/ by timestamp (finds image groups)
2. Identifies which images Erik selected (they're in mojo1_final/)
3. Creates selection training data: (chosen_image, rejected_images)
4. Identifies which selected images Erik cropped (modified in Oct 2025)
5. Outputs training data compatible with existing logs

Usage:
    python scripts/ai/extract_mojo1_training.py

Output:
    - data/training/mojo1_selection_log.csv (selection decisions)
    - data/training/mojo1_crop_log.csv (crop decisions)
    - data/training/mojo1_extraction_report.json (stats and metadata)
"""

import csv
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

# Project configuration
RAW_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1")
FINAL_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/training data/mojo1_final")
OUTPUT_DIR = Path("/Users/eriksjaastad/projects/Eros Mate/data/training")

# Project dates (from timesheet.csv)
PROJECT_START = datetime(2025, 10, 1, tzinfo=UTC)
PROJECT_END = datetime(2025, 10, 11, 23, 59, 59, tzinfo=UTC)


def parse_filename(filename: str) -> dict | None:
    """
    Extract timestamp and stage from filename.

    Patterns:
        20250708_150059_stage2_upscaled.png
        20250725_183405_stage1.5_face_swapped.png
        20250820_063708_stage1_generated.png
        20250706_023542_stage3_enhanced.png
    """
    # Match: YYYYMMDD_HHMMSS_stageX.X_suffix.png
    pattern = r"(\d{8}_\d{6})_stage(\d+(?:\.\d+)?)"
    match = re.match(pattern, filename)

    if match:
        return {"timestamp": match.group(1), "stage": float(match.group(2))}
    return None


def group_raw_images() -> dict[str, list[dict]]:
    """
    Group all raw images by timestamp.

    Returns:
        Dict mapping timestamp -> list of image info dicts
    """
    groups = defaultdict(list)

    raw_files = list(RAW_DIR.rglob("*.png"))

    for img_path in raw_files:
        parsed = parse_filename(img_path.name)
        if parsed:
            groups[parsed["timestamp"]].append(
                {"path": img_path, "filename": img_path.name, "stage": parsed["stage"]}
            )

    return groups


def find_winners(raw_groups: dict) -> dict[str, dict]:
    """
    For each group, find which image (if any) Erik selected.
    Selected images are in mojo1_final/.

    Returns:
        Dict mapping timestamp -> winner info
    """
    final_files = list(FINAL_DIR.rglob("*.png"))

    winners = {}
    matched_count = 0

    for final_img in final_files:
        parsed = parse_filename(final_img.name)
        if parsed:
            timestamp = parsed["timestamp"]

            if timestamp in raw_groups:
                # Found a match!
                matched_count += 1

                # Get file modification time (when Erik cropped it)
                mtime = datetime.fromtimestamp(final_img.stat().st_mtime, tz=UTC)

                # Extract original date from filename
                # Format: YYYYMMDD_HHMMSS
                filename_date = datetime.strptime(timestamp[:8], "%Y%m%d").replace(
                    tzinfo=UTC
                )

                # If file was modified significantly after its filename date, it was cropped
                # Most raw files are from July/August, crops happened in October
                days_after = (mtime - filename_date).days
                was_cropped = (
                    days_after > 7
                )  # Modified more than a week after original date

                winners[timestamp] = {
                    "winner_filename": final_img.name,
                    "winner_path": final_img,
                    "winner_stage": parsed["stage"],
                    "raw_group": raw_groups[timestamp],
                    "mtime": mtime,
                    "filename_date": filename_date,
                    "days_after": days_after,
                    "was_cropped": was_cropped,
                }

    return winners


def create_selection_training_data(winners: dict) -> list[dict]:
    """
    Create selection training entries: (winner, losers) pairs.

    Format matches existing selection_only_log.csv:
        session_id, set_id, chosen_path, neg_paths, timestamp
    """
    entries = []
    session_id = "mojo1_historical_extraction"

    for timestamp, data in winners.items():
        winner_filename = data["winner_filename"]
        winner_stage = data["winner_stage"]

        # Find losers (images in same group that weren't selected)
        losers = []
        loser_stages = []

        for img_info in data["raw_group"]:
            if img_info["filename"] != winner_filename:
                losers.append(str(img_info["path"]))
                loser_stages.append(img_info["stage"])

        # Only create entry if there were alternatives to choose from
        if losers:
            entries.append(
                {
                    "session_id": session_id,
                    "set_id": timestamp,
                    "chosen_path": str(data["winner_path"]),
                    "chosen_stage": winner_stage,
                    "neg_paths": json.dumps(losers),  # JSON array as string
                    "neg_stages": loser_stages,
                    "timestamp": timestamp,
                    "is_anomaly": winner_stage
                    < max([img["stage"] for img in data["raw_group"]]),
                }
            )

    return entries


def create_crop_training_data(winners: dict) -> list[dict]:
    """
    Create crop training entries for images Erik cropped.

    We can't extract crop coordinates (they're baked into the cropped image),
    but we can flag which images were cropped for future analysis.
    """
    entries = []
    session_id = "mojo1_historical_extraction"

    for timestamp, data in winners.items():
        if data["was_cropped"]:
            entries.append(
                {
                    "session_id": session_id,
                    "set_id": timestamp,
                    "directory": "mojo1",
                    "chosen_path": str(data["winner_path"]),
                    "chosen_stage": data["winner_stage"],
                    "timestamp": timestamp,
                    "note": "Crop coordinates not available (image already cropped)",
                    "mtime": data["mtime"].isoformat(),
                }
            )

    return entries


def write_selection_log(entries: list[dict], output_path: Path):
    """Write selection training data to CSV."""
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "set_id",
                "chosen_path",
                "neg_paths",
                "timestamp",
            ],
        )
        writer.writeheader()

        for entry in entries:
            writer.writerow(
                {
                    "session_id": entry["session_id"],
                    "set_id": entry["set_id"],
                    "chosen_path": entry["chosen_path"],
                    "neg_paths": entry["neg_paths"],
                    "timestamp": entry["timestamp"],
                }
            )


def write_crop_log(entries: list[dict], output_path: Path):
    """Write crop training data to CSV."""
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "set_id",
                "directory",
                "chosen_path",
                "chosen_stage",
                "timestamp",
                "note",
                "mtime",
            ],
        )
        writer.writeheader()

        for entry in entries:
            writer.writerow(entry)


def generate_report(
    raw_groups: dict, winners: dict, selection_entries: list, crop_entries: list
) -> dict:
    """Generate extraction statistics report."""
    # Analyze anomalies (when Erik chose lower stage)
    anomalies = [e for e in selection_entries if e["is_anomaly"]]

    # Count stage preferences
    stage_counts = defaultdict(int)
    for entry in selection_entries:
        stage_counts[entry["chosen_stage"]] += 1

    report = {
        "extraction_date": datetime.now(UTC).isoformat(),
        "project": "mojo-1",
        "project_dates": {
            "start": PROJECT_START.isoformat(),
            "end": PROJECT_END.isoformat(),
        },
        "raw_data": {
            "total_images": sum(len(group) for group in raw_groups.values()),
            "total_groups": len(raw_groups),
            "directory": str(RAW_DIR),
        },
        "final_data": {
            "total_selections": len(winners),
            "total_cropped": sum(1 for w in winners.values() if w["was_cropped"]),
            "directory": str(FINAL_DIR),
        },
        "training_data": {
            "selection_entries": len(selection_entries),
            "crop_entries": len(crop_entries),
            "anomaly_cases": len(anomalies),
            "anomaly_rate": len(anomalies) / len(selection_entries)
            if selection_entries
            else 0,
        },
        "stage_distribution": dict(stage_counts),
        "sample_anomalies": [
            {
                "timestamp": a["set_id"],
                "chosen_stage": a["chosen_stage"],
                "available_stages": [
                    img["stage"] for img in winners[a["set_id"]]["raw_group"]
                ],
                "max_stage": max(
                    img["stage"] for img in winners[a["set_id"]]["raw_group"]
                ),
            }
            for a in anomalies[:10]  # First 10 examples
        ],
    }

    return report


def main():
    # Step 1: Group raw images by timestamp
    raw_groups = group_raw_images()

    # Step 2: Find which images Erik selected
    winners = find_winners(raw_groups)

    # Step 3: Create selection training data
    selection_entries = create_selection_training_data(winners)

    # Step 4: Create crop training data
    crop_entries = create_crop_training_data(winners)

    # Step 5: Write output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection_path = OUTPUT_DIR / "mojo1_selection_log.csv"
    crop_path = OUTPUT_DIR / "mojo1_crop_log.csv"
    report_path = OUTPUT_DIR / "mojo1_extraction_report.json"

    write_selection_log(selection_entries, selection_path)
    write_crop_log(crop_entries, crop_path)

    # Step 6: Generate report
    report = generate_report(raw_groups, winners, selection_entries, crop_entries)

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Print summary


if __name__ == "__main__":
    main()
