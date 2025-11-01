#!/usr/bin/env python3
"""
Quick Dashboard Test
====================
Run this before deleting legacy system to verify dashboard works.

Usage:
    python scripts/tests/test_dashboard_manual.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard.analytics import DashboardAnalytics
from scripts.dashboard.data_engine import DashboardDataEngine


def main():
    print("\n" + "=" * 70)
    print("DASHBOARD MANUAL TEST")
    print("=" * 70)
    print()

    try:
        # Test data engine
        print("1. Testing Data Engine...")
        engine = DashboardDataEngine(str(PROJECT_ROOT))
        ops = engine.load_file_operations()
        print(f"   ✅ Loaded {len(ops)} operations")

        # Test analytics
        print("\n2. Testing Analytics Engine...")
        analytics = DashboardAnalytics(PROJECT_ROOT)
        response = analytics.generate_dashboard_response(
            time_slice="D", lookback_days=30, project_id=None
        )
        print("   ✅ Generated dashboard response")
        print(
            f"   Metadata: {response.get('metadata', {}).get('session_source', 'unknown')}"
        )

        # Check data sources
        print("\n3. Data Sources:")
        from collections import Counter

        sources = Counter(o.get("source", "unknown") for o in ops)
        for src, count in sources.most_common():
            print(f"   - {src}: {count} operations")

        # Show date range
        print("\n4. Date Range:")
        dates_raw = [o.get("date") for o in ops if o.get("date")]
        # Normalize all to strings
        dates_str = []
        for d in dates_raw:
            if hasattr(d, "isoformat"):
                dates_str.append(d.isoformat())
            else:
                dates_str.append(str(d))
        dates = sorted(set(dates_str))
        if dates:
            print(f"   First: {dates[0]}")
            print(f"   Last: {dates[-1]}")
            print(f"   Total days: {len(dates)}")

        print("\n" + "=" * 70)
        print("✅ DASHBOARD TEST PASSED")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Open dashboard in browser: http://localhost:5001")
        print("  2. Verify all your data shows up")
        print("  3. Check charts load correctly")
        print("  4. If everything looks good, run:")
        print("     bash scripts/data_pipeline/delete_legacy_system.sh")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("Dashboard test failed. Do NOT delete legacy system.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
