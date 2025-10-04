#!/usr/bin/env python3
"""
Dashboard System Test Runner
===========================
Runs all critical dashboard tests to verify the system is locked in and working perfectly.
This prevents regression issues like the historical data loss we just fixed.
"""

import unittest
import sys
from pathlib import Path
import time

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_core_functionality_tests():
    """Run core dashboard functionality tests"""
    print("🧪 Running Core Dashboard Functionality Tests")
    print("=" * 50)
    
    # Import and run core functionality tests
    from test_dashboard_core_functionality import TestDashboardCoreFunctionality, TestDashboardDataIntegrity
    
    test_suite = unittest.TestSuite()
    
    # Add core functionality tests
    core_tests = unittest.TestLoader().loadTestsFromTestCase(TestDashboardCoreFunctionality)
    integrity_tests = unittest.TestLoader().loadTestsFromTestCase(TestDashboardDataIntegrity)
    
    test_suite.addTests(core_tests)
    test_suite.addTests(integrity_tests)
    
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def run_cron_job_tests():
    """Run cron job system tests"""
    print("\n🕐 Running Cron Job System Tests")
    print("=" * 50)
    
    # Import and run cron job tests
    from test_cron_job_system import TestCronJobSystem, TestCronJobIntegration
    
    test_suite = unittest.TestSuite()
    
    # Add cron job tests
    cron_tests = unittest.TestLoader().loadTestsFromTestCase(TestCronJobSystem)
    integration_tests = unittest.TestLoader().loadTestsFromTestCase(TestCronJobIntegration)
    
    test_suite.addTests(cron_tests)
    test_suite.addTests(integration_tests)
    
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


def main():
    """Run all dashboard tests"""
    print("🚀 Dashboard System Test Suite")
    print("=" * 60)
    print("Verifying that the dashboard system is working perfectly")
    print("and locked in to prevent regression issues.")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run core functionality tests
    core_success = run_core_functionality_tests()
    
    # Run cron job tests
    cron_success = run_cron_job_tests()
    
    total_time = time.time() - start_time
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    if core_success and cron_success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Dashboard system is LOCKED IN and working perfectly!")
        print("🛡️  Historical data loss issue is PREVENTED by these tests")
        print("🚀 Cron job automation is READY and SAFE")
        
        print("\n📋 What's Protected:")
        print("• Historical data access (152,963+ records)")
        print("• Daily summaries and archived logs")
        print("• Chart data transformation")
        print("• Data consolidation with verification")
        print("• Cron job automation with 2-day buffer")
        print("• Error handling and graceful failures")
        
        print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
        
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Dashboard system needs attention before it's locked in")
        
        if not core_success:
            print("• Core dashboard functionality has issues")
        if not cron_success:
            print("• Cron job system has issues")
        
        print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
        
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
