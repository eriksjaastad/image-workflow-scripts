#!/usr/bin/env python3
"""
Dashboard Smoke Test
====================
Comprehensive smoke test that validates the entire data layer pipeline.

Tests:
1. Label generation for all time slices
2. Data aggregation and alignment
3. API response contract compliance
4. Edge cases (empty data, sparse data, etc.)
5. Performance (basic timing)

Run with: python3 smoke_test.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.dashboard.analytics import DashboardAnalytics
from scripts.dashboard.data_engine import DashboardDataEngine


def colored(text: str, color: str) -> str:
    """Add terminal color to text."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def test_label_generation():
    """Test 1: Label generation for all time slices."""
    print("\n" + "=" * 60)
    print(colored("TEST 1: Label Generation", "blue"))
    print("=" * 60)

    engine = DashboardDataEngine(str(project_root))
    slices = {
        "15min": (
            2,
            90,
            200,
        ),  # 2 days lookback, expect 90-200 labels (accounting for partial days)
        "1H": (
            2,
            24,
            50,
        ),  # 2 days lookback, expect 24-50 labels (roughly 24 hours + partial)
        "D": (
            7,
            7,
            8,
        ),  # 7 days lookback, expect 7-8 labels (lookback_days includes today)
        "W": (30, 4, 6),  # 30 days lookback, expect 4-6 labels
        "M": (90, 3, 5),  # 90 days lookback, expect 3-5 labels
    }

    passed = 0
    failed = 0

    for slice_type, (lookback, min_labels, max_labels) in slices.items():
        labels = engine.build_time_labels(slice_type, lookback)
        count = len(labels)

        if min_labels <= count <= max_labels:
            print(
                f"  ✅ {slice_type:6s}: {count:3d} labels (expected {min_labels}-{max_labels})"
            )
            passed += 1
        else:
            print(
                f"  ❌ {slice_type:6s}: {count:3d} labels (expected {min_labels}-{max_labels})"
            )
            failed += 1

        # Check sorted order
        timestamps = [datetime.fromisoformat(label) for label in labels]
        if timestamps == sorted(timestamps):
            print("     ✓ Chronologically sorted")
        else:
            print("     ✗ NOT sorted")
            failed += 1

    print(
        f"\n  Summary: {colored(f'{passed} passed', 'green')}, {colored(f'{failed} failed', 'red')}"
    )
    return failed == 0


def test_alignment():
    """Test 2: Data alignment to baseline labels."""
    print("\n" + "=" * 60)
    print(colored("TEST 2: Alignment & Gap Filling", "blue"))
    print("=" * 60)

    analytics = DashboardAnalytics(project_root)

    # Test case: sparse data with gaps
    baseline = ["2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05"]
    records = [
        {"time_slice": "2025-10-01", "operation": "crop", "file_count": 100},
        {"time_slice": "2025-10-03", "operation": "crop", "file_count": 75},
        {"time_slice": "2025-10-05", "operation": "crop", "file_count": 50},
    ]

    result = analytics._aggregate_to_baseline(
        records=records,
        baseline_labels=baseline,
        group_field="operation",  # Use operation instead of script to avoid display name transformation
        value_field="file_count",
    )

    passed = 0
    failed = 0

    # Check alignment
    if "crop" in result:
        print("  ✅ Operation found in result")
        passed += 1
    else:
        print("  ❌ Operation not found in result")
        failed += 1
        return False

    # Check counts
    expected = [100, 0, 75, 0, 50]
    if result["crop"]["counts"] == expected:
        print(f"  ✅ Counts aligned correctly: {result['crop']['counts']}")
        passed += 1
    else:
        print(
            f"  ❌ Counts misaligned: got {result['crop']['counts']}, expected {expected}"
        )
        failed += 1

    # Check no nulls
    if None not in result["crop"]["counts"]:
        print("  ✅ No null values (gaps filled with zeros)")
        passed += 1
    else:
        print("  ❌ Found null values in counts")
        failed += 1

    # Check length
    if len(result["crop"]["counts"]) == len(baseline):
        print(f"  ✅ Counts length matches baseline: {len(result['crop']['counts'])}")
        passed += 1
    else:
        print(
            f"  ❌ Length mismatch: {len(result['crop']['counts'])} vs {len(baseline)}"
        )
        failed += 1

    print(
        f"\n  Summary: {colored(f'{passed} passed', 'green')}, {colored(f'{failed} failed', 'red')}"
    )
    return failed == 0


def test_contract_compliance():
    """Test 3: API response contract compliance."""
    print("\n" + "=" * 60)
    print(colored("TEST 3: Contract Compliance", "blue"))
    print("=" * 60)

    analytics = DashboardAnalytics(project_root)

    # Generate full response
    response = analytics.generate_dashboard_response(
        time_slice="D", lookback_days=7, project_id=None
    )

    passed = 0
    failed = 0

    # Check required top-level keys
    required_keys = [
        "metadata",
        "projects",
        "charts",
        "timing_data",
        "project_comparisons",
        "project_kpi",
        "project_metrics",
        "project_markers",
    ]

    for key in required_keys:
        if key in response:
            print(f"  ✅ Has key: {key}")
            passed += 1
        else:
            print(f"  ❌ Missing key: {key}")
            failed += 1

    # Check metadata structure
    if "baseline_labels" in response.get("metadata", {}):
        baseline = response["metadata"]["baseline_labels"]
        all_slices_present = all(s in baseline for s in ["15min", "1H", "D", "W", "M"])
        if all_slices_present:
            print("  ✅ Metadata has all time slice baseline labels")
            passed += 1
        else:
            print("  ❌ Missing some time slice baseline labels")
            failed += 1

    # Check charts structure
    charts = response.get("charts", {})
    if "by_script" in charts and "by_operation" in charts:
        print("  ✅ Charts has by_script and by_operation")
        passed += 1

        # Validate each tool has dates and counts
        all_valid = True
        for tool, data in charts["by_script"].items():
            if "dates" not in data or "counts" not in data:
                all_valid = False
                break
            if len(data["dates"]) != len(data["counts"]):
                all_valid = False
                break

        if all_valid:
            print("  ✅ All tools have dates/counts with matching lengths")
            passed += 1
        else:
            print("  ❌ Some tools have mismatched dates/counts")
            failed += 1
    else:
        print("  ❌ Charts missing by_script or by_operation")
        failed += 1

    # Check timing_data structure
    timing = response.get("timing_data", {})
    if timing:
        all_valid = all(
            "work_time_minutes" in stats and "timing_method" in stats
            for stats in timing.values()
        )
        if all_valid:
            print("  ✅ Timing data has correct structure")
            passed += 1
        else:
            print("  ❌ Some timing data missing required fields")
            failed += 1

    print(
        f"\n  Summary: {colored(f'{passed} passed', 'green')}, {colored(f'{failed} failed', 'red')}"
    )
    return failed == 0


def test_toy_examples():
    """Test 4: Generate toy examples for documentation."""
    print("\n" + "=" * 60)
    print(colored("TEST 4: Toy Examples (3 labels each)", "blue"))
    print("=" * 60)

    analytics = DashboardAnalytics(project_root)

    # Daily example
    baseline_d = ["2025-10-13", "2025-10-14", "2025-10-15"]
    records_d = [
        {
            "time_slice": "2025-10-13",
            "script": "01_web_image_selector",
            "file_count": 120,
        },
        {
            "time_slice": "2025-10-14",
            "script": "01_web_image_selector",
            "file_count": 95,
        },
        {
            "time_slice": "2025-10-15",
            "script": "01_web_image_selector",
            "file_count": 143,
        },
    ]

    result_d = analytics._aggregate_to_baseline(
        records=records_d,
        baseline_labels=baseline_d,
        group_field="script",
        value_field="file_count",
    )

    print("\n  📊 Daily Example (D slice):")
    print(f"     Baseline: {baseline_d}")
    for tool, data in result_d.items():
        print(f"     {tool}:")
        print(f"       dates:  {data['dates']}")
        print(f"       counts: {data['counts']}")

    # Hourly example
    baseline_h = ["2025-10-15T14:00:00", "2025-10-15T15:00:00", "2025-10-15T16:00:00"]
    records_h = [
        {"time_slice": "2025-10-15T14:00:00", "operation": "crop", "file_count": 25},
        {"time_slice": "2025-10-15T15:00:00", "operation": "crop", "file_count": 30},
        {"time_slice": "2025-10-15T16:00:00", "operation": "crop", "file_count": 22},
    ]

    result_h = analytics._aggregate_to_baseline(
        records=records_h,
        baseline_labels=baseline_h,
        group_field="operation",
        value_field="file_count",
    )

    print("\n  📊 Hourly Example (1H slice):")
    print(f"     Baseline: {baseline_h}")
    for op, data in result_h.items():
        print(f"     {op}:")
        print(f"       dates:  {data['dates']}")
        print(f"       counts: {data['counts']}")

    print("\n  ✅ Toy examples generated successfully")
    return True


def test_performance():
    """Test 5: Basic performance check."""
    print("\n" + "=" * 60)
    print(colored("TEST 5: Performance", "blue"))
    print("=" * 60)

    import time

    analytics = DashboardAnalytics(project_root)

    # Test each time slice
    slices = ["15min", "1H", "D", "W", "M"]

    for slice_type in slices:
        start = time.time()
        response = analytics.generate_dashboard_response(
            time_slice=slice_type, lookback_days=30, project_id=None
        )
        elapsed = time.time() - start

        # Check response size
        json_size = len(json.dumps(response, default=str))

        print(f"  ⏱️  {slice_type:6s}: {elapsed:.3f}s | JSON size: {json_size:,} bytes")

        # Warn if too slow
        if elapsed > 5.0:
            print("     ⚠️  Warning: Slow response time")

    print("\n  ✅ Performance test complete")
    return True


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print(colored("🚀 PRODUCTIVITY DASHBOARD SMOKE TEST", "yellow"))
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = []

    # Run all tests
    results.append(("Label Generation", test_label_generation()))
    results.append(("Alignment", test_alignment()))
    results.append(("Contract Compliance", test_contract_compliance()))
    results.append(("Toy Examples", test_toy_examples()))
    results.append(("Performance", test_performance()))

    # Summary
    print("\n" + "=" * 60)
    print(colored("📊 FINAL SUMMARY", "yellow"))
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    for name, result in results:
        status = colored("✅ PASS", "green") if result else colored("❌ FAIL", "red")
        print(f"  {status} - {name}")

    print(
        f"\n  Total: {colored(f'{passed}/{len(results)} tests passed', 'green' if failed == 0 else 'yellow')}"
    )

    if failed == 0:
        print(f"\n{colored('🎉 ALL TESTS PASSED!', 'green')}")
        print("\nNext steps:")
        print("  1. Start the API server: python3 scripts/dashboard/api.py")
        print("  2. Open dashboard: scripts/dashboard/dashboard_template.html")
        print("  3. Test endpoints: curl http://localhost:8000/api/data/D")
        return 0
    print(f"\n{colored('⚠️  SOME TESTS FAILED', 'red')}")
    print("\nPlease review the failures above and fix issues.")
    return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
