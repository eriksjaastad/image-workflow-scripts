#!/usr/bin/env python3
"""
Import Historical Projects from Timesheet CSV

This script reads a timesheet CSV and creates project manifests for all historical projects.
It handles multi-day projects, special cases, and validates data before creating files.

Usage:
    python scripts/import_historical_projects.py <csv_file> [--dry-run]

Example:
    python scripts/import_historical_projects.py ~/Downloads/timesheet.csv --dry-run
    python scripts/import_historical_projects.py ~/Downloads/timesheet.csv --commit
"""

import argparse
import csv
import json
import sys
from datetime import datetime, time
from pathlib import Path


def parse_time(time_str: str) -> time | None:
    """Parse time string like '4:00 PM' to time object."""
    if not time_str or not time_str.strip():
        return None

    try:
        # Handle various time formats
        time_str = time_str.strip()
        dt = datetime.strptime(time_str, "%I:%M %p")
        return dt.time()
    except ValueError:
        try:
            # Try 24-hour format
            dt = datetime.strptime(time_str, "%H:%M")
            return dt.time()
        except ValueError:
            return None


def parse_date(date_str: str) -> datetime | None:
    """Parse date string like '8/29/2025' to datetime."""
    if not date_str or not date_str.strip():
        return None

    try:
        return datetime.strptime(date_str.strip(), "%m/%d/%Y")
    except ValueError:
        return None


def combine_datetime(date: datetime, time_obj: time | None) -> datetime:
    """Combine date and time into a single datetime."""
    if time_obj:
        return datetime.combine(date.date(), time_obj)
    # Default to start of day if no time
    return datetime.combine(date.date(), datetime.min.time())


def parse_csv_to_projects(csv_path: Path) -> list[dict]:
    """
    Parse timesheet CSV into project data structures.

    Returns list of projects, each with:
    - project_id: str
    - start_date: datetime
    - end_date: datetime
    - initial_images: int
    - final_images: int
    - total_hours: float
    - notes: str
    - rows: List[Dict] (all CSV rows for this project)
    """
    projects = []
    current_project = None

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader, start=1):
            # Skip empty rows
            if not any(row):
                continue

            # Parse columns
            date_str = row[0] if len(row) > 0 else ""
            start_time_str = row[1] if len(row) > 1 else ""
            end_time_str = row[2] if len(row) > 2 else ""
            row[3] if len(row) > 3 else ""
            project_name = row[4] if len(row) > 4 else ""
            hours_num_str = row[5] if len(row) > 5 else ""
            initial_images_str = row[6] if len(row) > 6 else ""
            final_images_str = row[7] if len(row) > 7 else ""
            notes = row[8] if len(row) > 8 else ""

            # Parse date
            date = parse_date(date_str)
            if not date:
                continue  # Skip rows without valid dates

            # Parse times
            start_time = parse_time(start_time_str)
            end_time = parse_time(end_time_str)

            # Parse numbers
            hours_num = float(hours_num_str) if hours_num_str.strip() else 0.0
            initial_images = (
                int(initial_images_str) if initial_images_str.strip() else None
            )
            final_images = int(final_images_str) if final_images_str.strip() else None

            # Row data
            row_data = {
                "row_idx": row_idx,
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "hours": hours_num,
                "project_name": project_name.strip(),
                "initial_images": initial_images,
                "final_images": final_images,
                "notes": notes.strip(),
            }

            # Determine if this is a new project or continuation
            if project_name.strip():
                # New project starts
                if current_project:
                    projects.append(current_project)

                current_project = {
                    "project_id": project_name.strip(),
                    "start_date": combine_datetime(date, start_time),
                    "end_date": combine_datetime(date, end_time)
                    if end_time
                    else combine_datetime(date, start_time),
                    "initial_images": initial_images,
                    "final_images": final_images,
                    "total_hours": hours_num,
                    "notes": notes.strip(),
                    "rows": [row_data],
                }
            # Continuation of current project
            # Only count as continuation if there were actual hours worked (not a day off)
            elif current_project and hours_num > 0:
                current_project["end_date"] = (
                    combine_datetime(date, end_time)
                    if end_time
                    else combine_datetime(date, start_time)
                )
                current_project["total_hours"] += hours_num

                # Update final images if provided
                if final_images is not None:
                    current_project["final_images"] = final_images

                # Append notes
                if notes.strip():
                    if current_project["notes"]:
                        current_project["notes"] += "; " + notes.strip()
                    else:
                        current_project["notes"] = notes.strip()

                current_project["rows"].append(row_data)

        # Don't forget last project
        if current_project:
            projects.append(current_project)

    return projects


def sanitize_project_id(project_id: str) -> str:
    """
    Convert project name to valid project ID.

    Examples:
        'mojo1-4/mojo-1' → 'mojo1'
        'mojo1-4/mojo-2' → 'mojo2'
        'Aiko_raw' → 'aiko_raw'
        'agent-1001' → 'agent-1001'
    """
    # Handle mojo special cases
    if "mojo-1" in project_id.lower():
        return "mojo1"
    if "mojo-2" in project_id.lower():
        return "mojo2"
    if "mojo-3" in project_id.lower():
        return "mojo3"

    # Default: lowercase and clean
    return project_id.lower().replace(" ", "_")


def create_project_manifest(
    project: dict, workspace_root: Path, dry_run: bool = True
) -> tuple[bool, str]:
    """
    Create a project manifest file from parsed project data.

    Returns: (success: bool, message: str)
    """
    # Sanitize project ID
    project_id = sanitize_project_id(project["project_id"])

    # Skip mojo1 and mojo2 (already exist)
    if project_id in ["mojo1", "mojo2"]:
        return (True, f"SKIPPED: {project_id} (already exists)")

    # Construct paths
    content_dir = workspace_root / "content" / project["project_id"]
    manifest_path = workspace_root / "data" / "projects" / f"{project_id}.project.json"

    # Check if manifest already exists
    if manifest_path.exists():
        return (False, f"SKIP: {project_id} manifest already exists at {manifest_path}")

    # Format dates to ISO-8601 UTC
    started_at = project["start_date"].strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    finished_at = project["end_date"].strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    created_at = (
        project["start_date"].strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    )  # Use start as created

    # Build manifest
    manifest = {
        "projectId": project_id,
        "title": project["project_id"],  # Use original name as title
        "status": "archived",
        "createdAt": created_at,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "paths": {
            "root": str(content_dir),
            "selectedDir": str(content_dir / "__selected"),
            "cropDir": str(content_dir / "__crop"),
        },
        "counts": {
            "initialImages": project.get("initial_images", 0) or 0,
            "finalImages": project.get("final_images", 0) or 0,
        },
        "metrics": {"totalHours": project["total_hours"], "notes": project["notes"]},
        "steps": [
            {"name": "selection", "status": "completed"},
            {"name": "sorting", "status": "completed"},
            {"name": "cropping", "status": "completed"},
            {"name": "delivery", "status": "completed"},
        ],
        "source": "imported_from_timesheet",
    }

    # Write manifest
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return (True, f"✅ Created: {manifest_path}")
    return (True, f"[DRY RUN] Would create: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Import historical projects from timesheet CSV"
    )
    parser.add_argument("csv_file", type=str, help="Path to timesheet CSV file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without creating files"
    )
    parser.add_argument(
        "--commit", action="store_true", help="Actually create manifest files"
    )

    args = parser.parse_args()

    # Default to dry-run if neither flag specified
    dry_run = not args.commit

    # Validate CSV file
    csv_path = Path(args.csv_file).expanduser()
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return 1

    # Get workspace root
    workspace_root = Path(__file__).parent.parent

    print(f"{'='*70}")
    print("Import Historical Projects from Timesheet")
    print(f"{'='*70}")
    print(f"CSV file: {csv_path}")
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'COMMIT (creating files)'}")
    print(f"{'='*70}\n")

    # Parse CSV
    print("📋 Parsing CSV...")
    try:
        projects = parse_csv_to_projects(csv_path)
        print(f"   Found {len(projects)} projects\n")
    except Exception as e:
        print(f"❌ Error parsing CSV: {e}")
        return 1

    # Apply manual fixes for known data issues
    print("🔧 Applying data fixes...")
    for project in projects:
        project_id = sanitize_project_id(project["project_id"])

        # Slender Kiara: missing initial images (ballpark ~2100)
        if project_id == "slender_kiara" and not project.get("initial_images"):
            project["initial_images"] = 2100
            print(f"   Fixed: {project_id} initial images set to 2100 (estimated)")

        # Dalia: 10 minutes rounded to 0, bump to 0.5h
        if project_id == "dalia" and project["total_hours"] < 0.5:
            project["total_hours"] = 0.5
            print(f"   Fixed: {project_id} hours rounded up to 0.5h (10 min → 0.5h)")

    # Filter out future projects (no final images, no hours, or explicitly mojo3)
    filtered_projects = []
    for p in projects:
        project_id = sanitize_project_id(p["project_id"])
        # Skip mojo3 (future project) and any project with no data
        if project_id == "mojo3":
            print(f"   Skipped: {project_id} (future project, not started yet)")
            continue
        if not p.get("final_images") and p["total_hours"] == 0:
            print(f"   Skipped: {project_id} (no data)")
            continue
        filtered_projects.append(p)

    projects = filtered_projects
    print(f"   → {len(projects)} projects ready to import\n")

    # Show project summary
    print("Projects Found:")
    print(f"{'-'*70}")
    for i, project in enumerate(projects, 1):
        project_id = sanitize_project_id(project["project_id"])
        start = project["start_date"].strftime("%Y-%m-%d")
        end = project["end_date"].strftime("%Y-%m-%d")
        days = len(project["rows"])
        hours = project["total_hours"]
        initial = project.get("initial_images") or 0
        final = project.get("final_images") or 0

        print(
            f"{i:2d}. {project_id:20s} | {start} → {end} | {days}d | {hours:4.1f}h | {initial:5d} → {final:5d} imgs"
        )

    print(f"{'-'*70}\n")

    # Create manifests
    print("Creating Manifests:")
    print(f"{'-'*70}")

    success_count = 0
    skip_count = 0
    error_count = 0

    for project in projects:
        success, message = create_project_manifest(project, workspace_root, dry_run)
        print(f"  {message}")

        if success:
            if "SKIP" in message:
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1

    print(f"{'-'*70}")
    print("\n📊 Summary:")
    print(f"   Total projects: {len(projects)}")
    print(f"   Created: {success_count}")
    print(f"   Skipped: {skip_count}")
    print(f"   Errors: {error_count}")

    if dry_run:
        print("\n💡 This was a DRY RUN. To create manifests, run with --commit")
    else:
        print("\n✅ Import complete!")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
