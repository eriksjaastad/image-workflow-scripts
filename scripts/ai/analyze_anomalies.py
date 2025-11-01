#!/usr/bin/env python3
"""
Analyze and extract anomaly cases from training data.

Anomaly = User chose a LOWER stage number over a higher one.
These cases teach the AI nuanced quality judgment beyond "highest stage = best".

Usage:
    python scripts/ai/analyze_anomalies.py                    # Show stats
    python scripts/ai/analyze_anomalies.py --export           # Export to CSV
    python scripts/ai/analyze_anomalies.py --by-project       # Group by project
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_stage(filename: str) -> int | None:
    """
    Extract stage number from filename.

    Examples:
        20250708_060711_stage2_upscaled.png → 2
        20250708_060558_stage1_generated.png → 1
        20250708_060558_stage1.5_upscaled.png → None (not a simple int)

    Returns:
        Stage number if found and is integer, else None
    """
    match = re.search(r"stage(\d+(?:\.\d+)?)_", filename)
    if match:
        stage_str = match.group(1)
        try:
            # Only return if it's a whole number
            stage = float(stage_str)
            if stage == int(stage):
                return int(stage)
        except ValueError:
            pass
    return None


def get_project_id(path: str) -> str:
    """Extract project ID from path."""
    parts = Path(path).parts
    for part in parts:
        if part.startswith("mojo"):
            return part
        if part in [
            "eleni",
            "aiko",
            "dalia",
            "kiara",
            "jmlimages-random",
            "tattersail-0918",
            "1100",
            "1101_hailey",
            "1011",
            "1012",
            "1013",
            "agent-1001",
            "agent-1002",
            "agent-1003",
            "Kiara_Slender",
            "Kiara_Average",
            "Aiko_raw",
            "Eleni_raw",
        ]:
            return part
        if part.endswith(("_raw", "_final")):
            return part.replace("_raw", "").replace("_final", "")
    return "unknown"


def analyze_selection(chosen_path: str, neg_paths_json: str) -> tuple[bool, dict]:
    """
    Determine if this is an anomaly case.

    Returns:
        (is_anomaly, info_dict)

        is_anomaly = True if user chose lower stage over higher stage
        info_dict contains chosen_stage, max_rejected_stage, stages, etc.
    """
    chosen_stage = parse_stage(chosen_path)

    # Parse negative paths from JSON string
    try:
        neg_paths = json.loads(neg_paths_json.replace('""', '"'))
    except json.JSONDecodeError:
        return False, {}

    # Get all rejected stages
    rejected_stages = [parse_stage(p) for p in neg_paths]
    rejected_stages = [s for s in rejected_stages if s is not None]

    # Can only determine anomaly if chosen has a stage and there are rejected stages
    if chosen_stage is None or not rejected_stages:
        return False, {}

    max_rejected_stage = max(rejected_stages)

    info = {
        "chosen_stage": chosen_stage,
        "rejected_stages": sorted(rejected_stages),
        "max_rejected_stage": max_rejected_stage,
        "chosen_path": chosen_path,
        "rejected_paths": neg_paths,
        "project": get_project_id(chosen_path),
    }

    # Anomaly = chose lower stage than highest available
    is_anomaly = chosen_stage < max_rejected_stage

    return is_anomaly, info


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze anomaly cases in training data"
    )
    parser.add_argument("--export", action="store_true", help="Export anomalies to CSV")
    parser.add_argument(
        "--by-project", action="store_true", help="Show breakdown by project"
    )
    args = parser.parse_args()

    csv_path = Path("data/training/selection_only_log.csv")

    if not csv_path.exists():
        return

    total_selections = 0
    total_anomalies = 0
    anomalies_by_project = defaultdict(list)
    normal_by_project = defaultdict(int)
    unparseable = 0

    all_anomalies = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_selections += 1

            is_anomaly, info = analyze_selection(row["chosen_path"], row["neg_paths"])

            if not info:  # Couldn't parse stages
                unparseable += 1
                continue

            project = info["project"]

            if is_anomaly:
                total_anomalies += 1
                anomalies_by_project[project].append(info)
                all_anomalies.append(
                    {
                        "session_id": row["session_id"],
                        "set_id": row["set_id"],
                        "project": project,
                        "chosen_stage": info["chosen_stage"],
                        "max_rejected_stage": info["max_rejected_stage"],
                        "chosen_path": info["chosen_path"],
                        "rejected_paths": info["rejected_paths"],
                        "timestamp": row["timestamp"],
                    }
                )
            else:
                normal_by_project[project] += 1

    # Print summary

    if args.by_project or args.export:
        # Sort projects by anomaly count
        sorted_projects = sorted(
            anomalies_by_project.items(), key=lambda x: len(x[1]), reverse=True
        )

        for project, anomaly_list in sorted_projects:
            anomaly_count = len(anomaly_list)
            normal_count = normal_by_project[project]
            total = anomaly_count + normal_count
            100 * anomaly_count / total if total > 0 else 0

    if args.export:
        output_path = Path("data/training/anomaly_cases.csv")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "session_id",
                    "set_id",
                    "project",
                    "chosen_stage",
                    "max_rejected_stage",
                    "chosen_path",
                    "rejected_paths",
                    "timestamp",
                ],
            )
            writer.writeheader()
            writer.writerows(all_anomalies)

        # Also create a summary JSON
        summary = {
            "total_selections": total_selections,
            "total_anomalies": total_anomalies,
            "anomaly_rate": total_anomalies / total_selections,
            "by_project": {
                project: {
                    "anomalies": len(anomaly_list),
                    "normal": normal_by_project[project],
                    "total": len(anomaly_list) + normal_by_project[project],
                    "anomaly_rate": len(anomaly_list)
                    / (len(anomaly_list) + normal_by_project[project]),
                }
                for project, anomaly_list in anomalies_by_project.items()
            },
        }

        summary_path = Path("data/training/anomaly_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Show some example anomalies
        for _i, _anomaly in enumerate(all_anomalies[:5], 1):
            pass


if __name__ == "__main__":
    main()
