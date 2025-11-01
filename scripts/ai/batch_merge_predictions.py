#!/usr/bin/env python3
"""
Batch Merge AI Predictions - All Projects

Merges temp databases with real databases for all projects that have both.
Supports dry-run mode to preview changes before applying.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DB_DIR = WORKSPACE / "data" / "training" / "ai_training_decisions"
PHASE2_SCRIPT = WORKSPACE / "scripts" / "ai" / "backfill_project_phase2_compare.py"

# Projects with existing databases (from mapping)
PROJECTS = [
    "agent-1003",
    "agent-1002",
    "agent-1001",
    "1013",
    "1011",
    "1012",
    "Aiko",
    "Eleni",
    "Kiara_Slender",
    "1100",
    "1101_Hailey",
    "tattersail-0918",
    "jmlimages-random",
    "mojo2",
    "mojo1",
]

# Projects without existing databases (skip these)
# 1010, 1102, Patricia, dalia, mixed-0919


def main():
    parser = argparse.ArgumentParser(
        description="Batch merge AI predictions for all projects"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying them"
    )
    args = parser.parse_args()

    # Track results
    results = {"successful": [], "failed": [], "skipped": []}

    start_time = datetime.now()

    for _idx, project_id in enumerate(PROJECTS, 1):
        temp_db = DB_DIR / f"{project_id}_ai_predictions_temp.db"
        real_db = DB_DIR / f"{project_id}.db"

        # Check if both databases exist
        if not temp_db.exists():
            results["skipped"].append(project_id)
            continue

        if not real_db.exists():
            results["skipped"].append(project_id)
            continue

        # Build command
        cmd = [
            sys.executable,
            str(PHASE2_SCRIPT),
            "--temp-db",
            str(temp_db),
            "--real-db",
            str(real_db),
        ]

        if args.dry_run:
            cmd.append("--dry-run")

        # Run Phase 2 merge
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Let output stream to terminal
                text=True,
            )

            results["successful"].append(project_id)

        except subprocess.CalledProcessError:
            results["failed"].append(project_id)
        except KeyboardInterrupt:
            sys.exit(1)

    # Final summary
    end_time = datetime.now()
    end_time - start_time

    if results["successful"]:
        for _p in results["successful"]:
            pass

    if results["failed"]:
        for _p in results["failed"]:
            pass

    if results["skipped"]:
        for _p in results["skipped"]:
            pass

    if args.dry_run:
        pass
    else:
        pass


if __name__ == "__main__":
    main()
