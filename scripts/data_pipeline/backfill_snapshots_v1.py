#!/usr/bin/env python3
"""
Backfill Snapshots v1
=====================
Backfills historical snapshot data in configurable date ranges.

Usage:
    python scripts/backfill_snapshots_v1.py --start-date 2025-09-01 --end-date 2025-09-30
    python scripts/backfill_snapshots_v1.py --days 90  # Last 90 days
"""

import argparse
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Extraction scripts
EXTRACTORS = [
    "extract_operation_events_v1.py",
    "extract_timer_sessions_v1.py",
    "extract_progress_snapshots_v1.py",
    "extract_projects_v1.py",
    "derive_sessions_from_ops_v1.py",
    "build_daily_aggregates_v1.py",
]


def parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD to datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)


def run_extractor(script_name: str) -> bool:
    """Run an extraction script."""
    script_path = PROJECT_ROOT / "scripts" / "data_pipeline" / script_name

    if not script_path.exists():
        print(f"  ⚠️  Script not found: {script_name}")
        return False

    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"  ❌ {script_name} failed with exit code {result.returncode}")
            return False

        print(f"  ✅ {script_name} completed successfully")
        return True

    except Exception as e:
        print(f"  ❌ Error running {script_name}: {e}")
        return False


def backfill(start_date: datetime, end_date: datetime) -> None:
    """
    Backfill snapshots for a date range.

    Note: The current extractors process all available data, not just the
    specified date range. This function serves as a framework for chunked
    processing if needed in the future.
    """
    days_count = (end_date - start_date).days + 1

    print("=" * 70)
    print("Snapshot Backfill v1")
    print("=" * 70)
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"End date:   {end_date.strftime('%Y-%m-%d')}")
    print(f"Days:       {days_count}")
    print("=" * 70)
    print()
    print("Note: Extractors process all available data.")
    print("Future enhancement: Add date-range filtering to extractors.")
    print()

    # Run all extractors
    failed = []

    for script_name in EXTRACTORS:
        success = run_extractor(script_name)
        if not success:
            failed.append(script_name)

    # Summary
    print("\n" + "=" * 70)
    print("Backfill Summary")
    print("=" * 70)

    if failed:
        print(f"❌ {len(failed)} extractors failed:")
        for script_name in failed:
            print(f"  - {script_name}")
        print("\nRun failed extractors individually to debug.")
    else:
        print("✅ All extractors completed successfully!")

    print("=" * 70)

    # Run validation
    print("\nRunning validation...")
    validation_script = (
        PROJECT_ROOT / "scripts" / "data_pipeline" / "validate_snapshots_v1.py"
    )
    if validation_script.exists():
        subprocess.run([sys.executable, str(validation_script)], check=False)

    return len(failed) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill historical snapshot data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill specific date range
  python scripts/backfill_snapshots_v1.py --start-date 2025-09-01 --end-date 2025-09-30
  
  # Backfill last 90 days
  python scripts/backfill_snapshots_v1.py --days 90
  
  # Backfill last 7 days
  python scripts/backfill_snapshots_v1.py --days 7
        """,
    )

    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--days", type=int, help="Number of days to backfill (from today)"
    )

    args = parser.parse_args()

    # Determine date range
    if args.days:
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=args.days)
    elif args.start_date and args.end_date:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    else:
        print("Error: Specify either --days or both --start-date and --end-date")
        sys.exit(1)

    # Run backfill
    success = backfill(start_date, end_date)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
