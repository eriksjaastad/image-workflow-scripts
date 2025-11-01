#!/usr/bin/env python3
"""
Complete Dashboard Test Suite
============================
Runs all dashboard-related tests to ensure the system is working perfectly
and locked in. This prevents regression issues like the historical data loss.
"""

import sys
import unittest
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test modules
from test_dashboard_data_engine import TestDashboardDataEngine, TestDashboardIntegration
from test_data_consolidation import TestConsolidationIntegration, TestDataConsolidation


def run_all_tests():
    """Run all dashboard tests with comprehensive reporting"""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDashboardDataEngine,
        TestDashboardIntegration,
        TestDataConsolidation,
        TestConsolidationIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, descriptions=True, failfast=False)

    print("🧪 Running Complete Dashboard Test Suite")
    print("=" * 50)
    print("Testing all critical dashboard functionality:")
    print("• Data loading from daily summaries")
    print("• Data loading from detailed logs")
    print("• Data loading from archived files")
    print("• Chart data transformation")
    print("• Data consolidation")
    print("• Dashboard verification")
    print("• Error handling")
    print("=" * 50)

    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('Exception:')[-1].strip()}")

    if not result.failures and not result.errors:
        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Dashboard system is locked in and working perfectly!")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
