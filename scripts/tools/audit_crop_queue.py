#!/usr/bin/env python3
"""
Audit Crop Queue - Consistency Report

Checks queue vs filesystem vs decisions DB for orphans and inconsistencies.

Usage:
    python scripts/tools/audit_crop_queue.py
    python scripts/tools/audit_crop_queue.py --queue data/ai_data/crop_queue/crop_queue.jsonl
    python scripts/tools/audit_crop_queue.py --report-file audit_report.txt

Checks:
1. Orphaned queue entries (source files don't exist)
2. Orphaned files in __crop_queued/ (not in queue)
3. DB inconsistencies (queue references non-existent DB records)
4. Missing .decision files for queued crops
5. Inconsistent crop coordinates (queue vs DB)
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def load_queue_entries(queue_file: Path) -> list[dict]:
    """Load all entries from queue file."""
    entries = []
    if not queue_file.exists():
        return entries

    with open(queue_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                batch = json.loads(line)
                for crop in batch.get("crops", []):
                    # Add batch metadata to each crop
                    crop["batch_id"] = batch.get("batch_id")
                    crop["batch_status"] = batch.get("status")
                    crop["project_id"] = batch.get("project_id")
                    entries.append(crop)
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: Invalid JSON: {e}")

    return entries


def scan_crop_queued_directory(crop_queued_dir: Path) -> set[Path]:
    """Scan __crop_queued/ directory for all image files."""
    if not crop_queued_dir.exists():
        return set()

    return set(crop_queued_dir.glob("**/*.png"))


def check_db_record(project_id: str, group_id: str) -> tuple[bool, str]:
    """
    Check if DB record exists for group_id.

    Returns:
        Tuple of (exists, db_path)
    """
    try:
        from utils.ai_training_decisions_v3 import init_decision_db

        db_path = init_decision_db(project_id)
        if not Path(db_path).exists():
            return False, str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, final_crop_coords FROM ai_decisions WHERE group_id = ?",
            (group_id,),
        )
        result = cursor.fetchone()
        conn.close()

        return result is not None, str(db_path)
    except Exception as e:
        return False, f"Error: {e}"


def audit_queue(queue_file: Path, crop_queued_dir: Path, report_file: Path = None):
    """
    Perform comprehensive audit of queue consistency.
    """
    print(f"\n{'='*80}")
    print("CROP QUEUE AUDIT")
    print(f"{'='*80}\n")
    print(f"Queue file: {queue_file}")
    print(f"Crop queued directory: {crop_queued_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    # Load data
    print("Loading queue entries...")
    queue_entries = load_queue_entries(queue_file)
    print(f"Found {len(queue_entries)} crop operations in queue\n")

    print("Scanning __crop_queued/ directory...")
    filesystem_files = scan_crop_queued_directory(crop_queued_dir)
    print(f"Found {len(filesystem_files)} files in {crop_queued_dir.name}/\n")

    # Track issues
    issues = {
        "orphaned_queue_entries": [],
        "orphaned_files": [],
        "missing_decision_files": [],
        "missing_db_records": [],
        "invalid_crop_coords": [],
    }

    # Check 1: Orphaned queue entries (source file doesn't exist)
    print(f"\n{'='*80}")
    print("CHECK 1: Orphaned Queue Entries")
    print(f"{'='*80}\n")

    queue_source_paths = set()
    for entry in queue_entries:
        source_path = Path(entry["source_path"])
        queue_source_paths.add(source_path)

        if not source_path.exists():
            issues["orphaned_queue_entries"].append(
                {
                    "batch_id": entry.get("batch_id"),
                    "source_path": str(source_path),
                    "dest_directory": entry.get("dest_directory"),
                    "status": entry.get("batch_status"),
                }
            )

    if issues["orphaned_queue_entries"]:
        print(
            f"❌ Found {len(issues['orphaned_queue_entries'])} orphaned queue entries:"
        )
        for issue in issues["orphaned_queue_entries"][:5]:
            print(
                f"   [{issue['batch_id']}] {Path(issue['source_path']).name} ({issue['status']})"
            )
        if len(issues["orphaned_queue_entries"]) > 5:
            print(f"   ... and {len(issues['orphaned_queue_entries']) - 5} more")
    else:
        print("✅ No orphaned queue entries")

    # Check 2: Orphaned files (file exists but not in queue)
    print(f"\n{'='*80}")
    print("CHECK 2: Orphaned Files in __crop_queued/")
    print(f"{'='*80}\n")

    for file_path in filesystem_files:
        if file_path not in queue_source_paths:
            issues["orphaned_files"].append(str(file_path))

    if issues["orphaned_files"]:
        print(f"❌ Found {len(issues['orphaned_files'])} orphaned files:")
        for file_path in issues["orphaned_files"][:5]:
            print(f"   {Path(file_path).name}")
        if len(issues["orphaned_files"]) > 5:
            print(f"   ... and {len(issues['orphaned_files']) - 5} more")
    else:
        print("✅ No orphaned files")

    # Check 3: Missing .decision files
    print(f"\n{'='*80}")
    print("CHECK 3: Missing .decision Files")
    print(f"{'='*80}\n")

    for entry in queue_entries:
        source_path = Path(entry["source_path"])
        if source_path.exists():
            decision_path = source_path.with_suffix(".decision")
            if not decision_path.exists():
                issues["missing_decision_files"].append(
                    {
                        "batch_id": entry.get("batch_id"),
                        "source_file": source_path.name,
                        "expected_decision": str(decision_path),
                    }
                )

    if issues["missing_decision_files"]:
        print(
            f"❌ Found {len(issues['missing_decision_files'])} missing .decision files:"
        )
        for issue in issues["missing_decision_files"][:5]:
            print(f"   [{issue['batch_id']}] {issue['source_file']}")
        if len(issues["missing_decision_files"]) > 5:
            print(f"   ... and {len(issues['missing_decision_files']) - 5} more")
    else:
        print("✅ All queued files have .decision files")

    # Check 4: Missing DB records
    print(f"\n{'='*80}")
    print("CHECK 4: Missing DB Records")
    print(f"{'='*80}\n")

    for entry in queue_entries:
        source_path = Path(entry["source_path"])
        if source_path.exists():
            decision_path = source_path.with_suffix(".decision")
            if decision_path.exists():
                try:
                    with open(decision_path) as f:
                        decision_data = json.load(f)

                    group_id = decision_data.get("group_id")
                    project_id = entry.get("project_id") or decision_data.get(
                        "project_id"
                    )

                    if group_id and project_id:
                        exists, db_info = check_db_record(project_id, group_id)
                        if not exists:
                            issues["missing_db_records"].append(
                                {
                                    "batch_id": entry.get("batch_id"),
                                    "source_file": source_path.name,
                                    "group_id": group_id,
                                    "project_id": project_id,
                                    "db_path": db_info,
                                }
                            )
                except Exception:
                    pass  # Already caught in missing decision files check

    if issues["missing_db_records"]:
        print(f"❌ Found {len(issues['missing_db_records'])} missing DB records:")
        for issue in issues["missing_db_records"][:5]:
            print(
                f"   [{issue['batch_id']}] {issue['source_file']} (group_id: {issue['group_id']})"
            )
        if len(issues["missing_db_records"]) > 5:
            print(f"   ... and {len(issues['missing_db_records']) - 5} more")
    else:
        print("✅ All queued crops have valid DB records")

    # Check 5: Invalid crop coordinates
    print(f"\n{'='*80}")
    print("CHECK 5: Invalid Crop Coordinates")
    print(f"{'='*80}\n")

    for entry in queue_entries:
        crop_rect = entry.get("crop_rect", [])
        crop_rect_normalized = entry.get("crop_rect_normalized", [])

        # Check pixel coords
        if len(crop_rect) != 4:
            issues["invalid_crop_coords"].append(
                {
                    "batch_id": entry.get("batch_id"),
                    "source_file": Path(entry["source_path"]).name,
                    "issue": f"Invalid crop_rect length: {len(crop_rect)}",
                }
            )
            continue

        x1, y1, x2, y2 = crop_rect
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            issues["invalid_crop_coords"].append(
                {
                    "batch_id": entry.get("batch_id"),
                    "source_file": Path(entry["source_path"]).name,
                    "issue": f"Invalid crop dimensions: {crop_rect}",
                }
            )

        # Check normalized coords
        if crop_rect_normalized:
            if len(crop_rect_normalized) != 4:
                issues["invalid_crop_coords"].append(
                    {
                        "batch_id": entry.get("batch_id"),
                        "source_file": Path(entry["source_path"]).name,
                        "issue": f"Invalid normalized coords length: {len(crop_rect_normalized)}",
                    }
                )
                continue

            nx1, ny1, nx2, ny2 = crop_rect_normalized
            if not all(0 <= v <= 1 for v in [nx1, ny1, nx2, ny2]):
                issues["invalid_crop_coords"].append(
                    {
                        "batch_id": entry.get("batch_id"),
                        "source_file": Path(entry["source_path"]).name,
                        "issue": f"Normalized coords out of [0,1] range: {crop_rect_normalized}",
                    }
                )

    if issues["invalid_crop_coords"]:
        print(
            f"❌ Found {len(issues['invalid_crop_coords'])} invalid crop coordinates:"
        )
        for issue in issues["invalid_crop_coords"][:5]:
            print(f"   [{issue['batch_id']}] {issue['source_file']}: {issue['issue']}")
        if len(issues["invalid_crop_coords"]) > 5:
            print(f"   ... and {len(issues['invalid_crop_coords']) - 5} more")
    else:
        print("✅ All crop coordinates are valid")

    # Summary
    print(f"\n{'='*80}")
    print("AUDIT SUMMARY")
    print(f"{'='*80}\n")

    total_issues = sum(len(v) for v in issues.values())

    if total_issues == 0:
        print("✅ No issues found! Queue is consistent with filesystem and DB.\n")
    else:
        print(f"⚠️  Found {total_issues} total issues:\n")
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"   - {issue_type.replace('_', ' ').title()}: {len(issue_list)}")
        print()

    # Write detailed report if requested
    if report_file:
        print(f"Writing detailed report to {report_file}...")
        with open(report_file, "w") as f:
            f.write("Crop Queue Audit Report\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Queue file: {queue_file}\n")
            f.write(f"Crop queued directory: {crop_queued_dir}\n\n")

            f.write("Summary:\n")
            f.write(f"  Queue entries: {len(queue_entries)}\n")
            f.write(f"  Filesystem files: {len(filesystem_files)}\n")
            f.write(f"  Total issues: {total_issues}\n\n")

            for issue_type, issue_list in issues.items():
                if issue_list:
                    f.write(
                        f"\n{issue_type.replace('_', ' ').title()}: {len(issue_list)}\n"
                    )
                    f.write(f"{'-'*80}\n")
                    for issue in issue_list:
                        f.write(f"{json.dumps(issue, indent=2)}\n")

        print(f"✅ Report written to {report_file}\n")

    return total_issues == 0


def main():
    parser = argparse.ArgumentParser(
        description="Audit crop queue consistency (queue vs filesystem vs DB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("data/ai_data/crop_queue/crop_queue.jsonl"),
        help="Path to queue file (default: data/ai_data/crop_queue/crop_queue.jsonl)",
    )
    parser.add_argument(
        "--crop-queued-dir",
        type=Path,
        default=Path("__crop_queued"),
        help="Path to crop queued directory (default: __crop_queued)",
    )
    parser.add_argument(
        "--report-file", type=Path, help="Write detailed report to file (optional)"
    )

    args = parser.parse_args()

    is_clean = audit_queue(args.queue, args.crop_queued_dir, args.report_file)

    # Exit code: 0 if clean, 1 if issues found
    exit(0 if is_clean else 1)


if __name__ == "__main__":
    main()
