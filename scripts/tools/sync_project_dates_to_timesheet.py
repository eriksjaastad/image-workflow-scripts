#!/usr/bin/env python3
"""
Match project manifest dates to timesheet.

This script updates project manifest dates (createdAt, startedAt, finishedAt)
to match the dates recorded in the timesheet.

Usage:
    python scripts/tools/sync_project_dates_to_timesheet.py --dry-run
    python scripts/tools/sync_project_dates_to_timesheet.py --execute
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
TIMESHEET_PATH = WORKSPACE / "data/timesheet.csv"
PROJECTS_DIR = WORKSPACE / "data/projects"
BACKUP_LOG = WORKSPACE / "data/daily_summaries/project_dates_backup.json"


def parse_timesheet():
    """Parse timesheet and extract project dates."""
    projects = {}
    current_project = None  # Track ongoing project

    with open(TIMESHEET_PATH) as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 5:
                continue

            date_str = row[0].strip()
            hours_str = row[3].strip() if len(row) > 3 else ""
            project_name = row[4].strip() if len(row) > 4 else ""

            # Skip summary rows
            if "/" not in date_str:
                continue

            # Parse date: "8/29/2025" -> "2025-08-29"
            try:
                month, day, year = date_str.split("/")
                normalized_date = f"{year}-{int(month):02d}-{int(day):02d}"
            except ValueError:
                continue

            # If project_name is empty but hours are logged (and > 0), continue previous project
            # If no hours logged OR hours are "0:00:00", this is a day off - don't continue project
            if (
                not project_name
                and hours_str
                and hours_str != "0:00:00"
                and current_project
            ):
                project_name = current_project
            elif project_name:
                current_project = project_name
            else:
                continue

            # Handle multiple projects on same day (e.g., "agent-1001/agent-1002")
            project_ids = [p.strip() for p in project_name.split("/")]

            for project_id in project_ids:
                if not project_id:
                    continue

                # Normalize project ID
                normalized_id = (
                    project_id.lower().replace(" ", "_").replace("mojo-", "mojo")
                )

                if normalized_id not in projects:
                    projects[normalized_id] = {
                        "original_name": project_id,
                        "start_date": normalized_date,
                        "end_date": normalized_date,
                        "dates": [normalized_date],
                    }
                else:
                    # Update end date and add to dates list
                    projects[normalized_id]["end_date"] = normalized_date
                    if normalized_date not in projects[normalized_id]["dates"]:
                        projects[normalized_id]["dates"].append(normalized_date)

    return projects


def find_matching_manifest(project_key, manifests):
    """Find matching manifest file for a project key."""
    # Try exact match first
    for manifest_file in manifests:
        manifest_id = manifest_file.stem.replace(".project", "")
        if manifest_id.lower() == project_key:
            return manifest_file

    # Try partial matches
    for manifest_file in manifests:
        manifest_id = manifest_file.stem.replace(".project", "")
        if project_key in manifest_id.lower() or manifest_id.lower() in project_key:
            return manifest_file

    return None


def format_utc_timestamp(date_str):
    """Convert YYYY-MM-DD to UTC ISO format with Z suffix."""
    # Start of day in UTC
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(hour=0, minute=0, second=0, tzinfo=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    parser = argparse.ArgumentParser(description="Sync project dates to timesheet")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute the date updates"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("❌ Must specify either --dry-run or --execute")
        return 1

    # Parse timesheet
    print("📋 Parsing timesheet...")
    timesheet_projects = parse_timesheet()
    print(f"   Found {len(timesheet_projects)} projects in timesheet\n")

    # Find all project manifests
    manifests = list(PROJECTS_DIR.glob("*.project.json"))
    print(f"📁 Found {len(manifests)} project manifests\n")

    # Compare and plan updates
    updates = []
    unmatched_timesheet = []
    unmatched_manifests = []

    for project_key, timesheet_data in timesheet_projects.items():
        manifest_file = find_matching_manifest(project_key, manifests)

        if not manifest_file:
            unmatched_timesheet.append((project_key, timesheet_data))
            continue

        # Read current manifest
        with open(manifest_file) as f:
            manifest = json.load(f)

        # Calculate new dates
        new_started_at = format_utc_timestamp(timesheet_data["start_date"])
        new_finished_at = (
            format_utc_timestamp(timesheet_data["end_date"])
            if timesheet_data["start_date"] != timesheet_data["end_date"]
            else None
        )

        # Check if update needed
        old_started_at = manifest.get("startedAt")
        old_finished_at = manifest.get("finishedAt")

        if old_started_at != new_started_at or old_finished_at != new_finished_at:
            updates.append(
                {
                    "manifest_file": manifest_file,
                    "project_id": manifest.get("projectId"),
                    "timesheet_name": timesheet_data["original_name"],
                    "old_started_at": old_started_at,
                    "new_started_at": new_started_at,
                    "old_finished_at": old_finished_at,
                    "new_finished_at": new_finished_at,
                    "manifest": manifest,
                }
            )

    # Find manifests without timesheet entries
    for manifest_file in manifests:
        manifest_id = manifest_file.stem.replace(".project", "")
        matched = False

        for project_key in timesheet_projects:
            if find_matching_manifest(project_key, [manifest_file]):
                matched = True
                break

        if not matched and not manifest_id.startswith("TEST-"):
            with open(manifest_file) as f:
                manifest = json.load(f)
            unmatched_manifests.append((manifest_file, manifest.get("projectId")))

    # Display report
    print("=" * 80)
    print("📊 SYNC REPORT")
    print("=" * 80)

    if updates:
        print(f"\n✏️  {len(updates)} projects need date updates:\n")
        for update in updates:
            print(f"📁 {update['project_id']} (timesheet: {update['timesheet_name']})")
            print(
                f"   startedAt:  {update['old_started_at']} → {update['new_started_at']}"
            )
            if update["new_finished_at"]:
                print(
                    f"   finishedAt: {update['old_finished_at']} → {update['new_finished_at']}"
                )
            else:
                print(
                    f"   finishedAt: {update['old_finished_at']} → (same day as start)"
                )
            print()
    else:
        print("\n✅ All projects already in sync!\n")

    if unmatched_timesheet:
        print(f"⚠️  {len(unmatched_timesheet)} timesheet entries without manifests:\n")
        for project_key, data in unmatched_timesheet:
            print(f"   • {data['original_name']} ({data['start_date']})")
        print()

    if unmatched_manifests:
        print(f"ℹ️  {len(unmatched_manifests)} manifests not in timesheet:\n")
        for manifest_file, project_id in unmatched_manifests:
            print(f"   • {project_id}")
        print()

    # Execute updates
    if args.execute and updates:
        print("=" * 80)
        print("💾 EXECUTING UPDATES")
        print("=" * 80)

        # Create backup log
        backup_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "updates": [
                {
                    "project_id": u["project_id"],
                    "old_started_at": u["old_started_at"],
                    "old_finished_at": u["old_finished_at"],
                    "new_started_at": u["new_started_at"],
                    "new_finished_at": u["new_finished_at"],
                }
                for u in updates
            ],
        }

        BACKUP_LOG.write_text(json.dumps(backup_data, indent=2))
        print(f"\n📝 Backup log created: {BACKUP_LOG}\n")

        # Update manifests
        for update in updates:
            manifest = update["manifest"]
            manifest["startedAt"] = update["new_started_at"]
            manifest["createdAt"] = update["new_started_at"]  # Match created to started

            if update["new_finished_at"]:
                manifest["finishedAt"] = update["new_finished_at"]

            # Write updated manifest
            with open(update["manifest_file"], "w") as f:
                json.dump(manifest, f, indent=2)

            print(f"✅ Updated: {update['project_id']}")

        print(f"\n✅ Successfully updated {len(updates)} project manifests!")

    elif args.dry_run:
        print("🧪 DRY RUN - No files were modified")
        print("   Run with --execute to apply these changes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
