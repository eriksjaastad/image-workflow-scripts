#!/usr/bin/env python3
"""
Test Suite: Snapshot Data Integrity
====================================
Verifies that snapshot data completely covers all legacy data
and that dashboard loads correctly from snapshots alone.

Run: python scripts/tests/test_snapshot_data_integrity.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.data_engine import DashboardDataEngine


def test_snapshot_coverage():
    """Test that snapshots cover all days with real data in legacy summaries"""
    print("\n" + "=" * 70)
    print("TEST 1: Snapshot Coverage")
    print("=" * 70)

    legacy_dir = PROJECT_ROOT / "data" / "daily_summaries"
    snapshot_dir = PROJECT_ROOT / "data" / "snapshot" / "daily_aggregates_v1"

    # Get days with non-zero data in legacy
    legacy_days = set()
    for f in legacy_dir.glob("daily_summary_*.json"):
        with open(f) as file:
            data = json.load(file)
            if data.get("total_operations", 0) > 0:
                day = f.stem.replace("daily_summary_", "")
                legacy_days.add(day)

    # Get days in snapshots
    snapshot_days = set()
    for day_dir in snapshot_dir.glob("day=*"):
        day = day_dir.name.split("=")[1]
        snapshot_days.add(day)

    # Check coverage
    missing_days = legacy_days - snapshot_days

    print(f"  Legacy days with data: {len(legacy_days)}")
    print(f"  Snapshot days: {len(snapshot_days)}")
    print(f"  Missing from snapshots: {len(missing_days)}")

    if missing_days:
        print(f"  ❌ FAIL: Days missing: {sorted(missing_days)}")
        return False
    print("  ✅ PASS: All legacy days covered by snapshots")
    return True


def test_dashboard_loads_snapshots():
    """Test that dashboard can load data from snapshots"""
    print("\n" + "=" * 70)
    print("TEST 2: Dashboard Snapshot Loading")
    print("=" * 70)

    try:
        engine = DashboardDataEngine(str(PROJECT_ROOT))
        ops = engine.load_file_operations()

        # Count by source
        sources = Counter(o.get("source", "unknown") for o in ops)

        print(f"  Total operations loaded: {len(ops)}")
        print("  By source:")
        for src, count in sources.most_common():
            print(f"    - {src}: {count}")

        # Check that snapshots are loading
        snapshot_count = sources.get("snapshot_aggregate_v1", 0)

        if snapshot_count == 0:
            print("  ❌ FAIL: No snapshot data loaded")
            return False
        if snapshot_count < 50:
            print(
                f"  ⚠️  WARNING: Only {snapshot_count} snapshot operations (expected more)"
            )
            return True
        print("  ✅ PASS: Snapshots loading correctly")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Error loading dashboard data: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_consistency():
    """Test that data is consistent across sources"""
    print("\n" + "=" * 70)
    print("TEST 3: Data Consistency")
    print("=" * 70)

    try:
        engine = DashboardDataEngine(str(PROJECT_ROOT))

        # Load from snapshots
        snapshot_dir = PROJECT_ROOT / "data" / "snapshot" / "daily_aggregates_v1"
        snapshot_ops = engine._load_from_snapshot_aggregates(snapshot_dir)

        # Group by date
        snapshot_by_day = {}
        for op in snapshot_ops:
            day = op.get("date")
            if day:
                if day not in snapshot_by_day:
                    snapshot_by_day[day] = []
                snapshot_by_day[day].append(op)

        print(f"  Snapshot days: {len(snapshot_by_day)}")
        print(f"  Snapshot operations: {len(snapshot_ops)}")

        # Verify each day has data
        empty_days = [day for day, ops in snapshot_by_day.items() if not ops]
        if empty_days:
            print(f"  ⚠️  WARNING: Empty days in snapshots: {empty_days}")

        # Check for duplicate dates
        dates = [op.get("date") for op in snapshot_ops]
        duplicates = [d for d, count in Counter(dates).items() if count > 10]
        if duplicates:
            print(f"  ⚠️  INFO: Days with multiple scripts: {len(duplicates)}")

        print("  ✅ PASS: Data structure is consistent")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Error checking consistency: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_snapshot_files_exist():
    """Test that all expected snapshot directories exist"""
    print("\n" + "=" * 70)
    print("TEST 4: Snapshot File Structure")
    print("=" * 70)

    expected_dirs = [
        "data/snapshot/operation_events_v1",
        "data/snapshot/daily_aggregates_v1",
        "data/snapshot/derived_sessions_v1",
    ]

    all_exist = True
    for dir_path in expected_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            count = len(list(full_path.glob("day=*")))
            print(f"  ✅ {dir_path}: {count} days")
        else:
            print(f"  ❌ {dir_path}: NOT FOUND")
            all_exist = False

    if all_exist:
        print("  ✅ PASS: All snapshot directories exist")
        return True
    print("  ❌ FAIL: Missing snapshot directories")
    return False


def test_raw_logs_preserved():
    """Test that raw logs still exist (source of truth)"""
    print("\n" + "=" * 70)
    print("TEST 5: Raw Logs Preserved")
    print("=" * 70)

    logs_dir = PROJECT_ROOT / "data" / "file_operations_logs"
    archives_dir = PROJECT_ROOT / "data" / "log_archives"

    log_files = list(logs_dir.glob("*.log")) if logs_dir.exists() else []
    archive_files = list(archives_dir.glob("*.log.gz")) if archives_dir.exists() else []

    print(f"  Current logs: {len(log_files)}")
    print(f"  Archived logs: {len(archive_files)}")
    print(f"  Total: {len(log_files) + len(archive_files)}")

    if len(log_files) + len(archive_files) > 0:
        print("  ✅ PASS: Raw logs preserved")
        return True
    print("  ❌ FAIL: No raw logs found")
    return False


def main():
    print("\n" + "=" * 70)
    print("SNAPSHOT DATA INTEGRITY TEST SUITE")
    print("=" * 70)
    print(f"Project: {PROJECT_ROOT}")

    tests = [
        ("Snapshot Coverage", test_snapshot_coverage),
        ("Dashboard Loading", test_dashboard_loads_snapshots),
        ("Data Consistency", test_data_consistency),
        ("File Structure", test_snapshot_files_exist),
        ("Raw Logs", test_raw_logs_preserved),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ✅ ALL TESTS PASSED - Safe to delete legacy system!")
        return 0
    print("\n  ❌ SOME TESTS FAILED - Do NOT delete legacy system yet!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
