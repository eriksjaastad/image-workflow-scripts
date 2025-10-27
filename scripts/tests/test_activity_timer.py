#!/usr/bin/env python3
"""
Comprehensive Test Suite for Activity Timer System
==================================================

Tests the utils/activity_timer.py module for:
- Basic timer functionality
- Idle detection
- Activity tracking
- Session management
- Cross-script reporting
- Data persistence
"""

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.activity_timer import ActivityTimer, TimerReporter


class TestActivityTimer(unittest.TestCase):
    """Test the ActivityTimer class"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_data_dir = None
        
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def _create_timer_with_test_dir(self, script_name="test_script", idle_threshold=60):
        """Create timer with custom data directory for testing"""
        timer = ActivityTimer(script_name, idle_threshold)
        # Override data directory for testing
        timer.data_dir = self.test_dir / "timer_data"
        timer.data_dir.mkdir(exist_ok=True)
        timer.session_file = timer.data_dir / f"session_{timer.session_id}.json"
        timer.daily_file = timer.data_dir / f"daily_{time.strftime('%Y%m%d')}.json"
        return timer
        
    def test_timer_initialization(self):
        """Test timer initializes correctly"""
        timer = self._create_timer_with_test_dir("test_script")
        
        self.assertEqual(timer.script_name, "test_script")
        self.assertEqual(timer.idle_threshold, 60)
        self.assertIsNone(timer.session_start)
        self.assertFalse(timer.is_active)
        self.assertEqual(timer.active_time, 0.0)
        self.assertEqual(timer.idle_time, 0.0)
        
    def test_session_start_and_end(self):
        """Test session lifecycle"""
        timer = self._create_timer_with_test_dir("test_script")
        
        # Start session
        start_time = time.time()
        timer.start_session()
        
        self.assertIsNotNone(timer.session_start)
        self.assertTrue(timer.is_active)
        self.assertGreaterEqual(timer.session_start, start_time)
        
        # Verify session file is created
        self.assertTrue(timer.session_file.exists())
        
        # End session
        time.sleep(0.1)  # Small delay to ensure measurable time
        timer.end_session()
        
        self.assertIsNotNone(timer.current_session.end_time)
        self.assertGreater(timer.active_time, 0)
        
    def test_activity_marking(self):
        """Test activity marking updates timer correctly"""
        timer = self._create_timer_with_test_dir("test_script")
        timer.start_session()
        
        initial_activity = timer.last_activity
        time.sleep(0.1)
        
        timer.mark_activity()
        
        self.assertGreater(timer.last_activity, initial_activity)
        self.assertTrue(timer.is_active)
        self.assertGreater(timer.active_time, 0)
        
    def test_batch_tracking(self):
        """Test batch marking and tracking"""
        timer = self._create_timer_with_test_dir("test_script")
        timer.start_session()
        
        # Mark batch start
        timer.mark_batch("Test Batch 1", "Testing batch functionality")
        
        self.assertEqual(timer.current_batch, "Test Batch 1")
        self.assertEqual(len(timer.current_session.batches), 1)
        
        batch = timer.current_session.batches[0]
        self.assertEqual(batch["name"], "Test Batch 1")
        self.assertEqual(batch["description"], "Testing batch functionality")
        self.assertIn("start_time", batch)
        
        # End batch
        timer.end_batch("Batch completed successfully")
        
        self.assertIsNone(timer.current_batch)
        self.assertIn("end_time", batch)
        self.assertIn("duration", batch)
        self.assertEqual(batch["summary"], "Batch completed successfully")
        
    def test_operation_logging(self):
        """Test operation logging"""
        timer = self._create_timer_with_test_dir("test_script")
        timer.start_session()
        
        # Log various operations
        timer.log_operation("crop", file_count=3, details="Cropped 3 images")
        timer.log_operation("delete", file_count=1, details="Deleted 1 image")
        timer.log_operation("move", file_count=2)
        
        operations = timer.current_session.operations
        self.assertEqual(len(operations), 3)
        
        # Check first operation
        crop_op = operations[0]
        self.assertEqual(crop_op["type"], "crop")
        self.assertEqual(crop_op["file_count"], 3)
        self.assertEqual(crop_op["details"], "Cropped 3 images")
        
        # Check operation without details
        move_op = operations[2]
        self.assertEqual(move_op["type"], "move")
        self.assertEqual(move_op["file_count"], 2)
        self.assertEqual(move_op["details"], "")
        
    def test_current_stats(self):
        """Test current statistics calculation"""
        timer = self._create_timer_with_test_dir("test_script")
        timer.start_session()
        
        # Add some activity
        timer.mark_activity()
        time.sleep(0.1)
        timer.log_operation("test", file_count=5)
        timer.mark_batch("Test Batch")
        timer.end_batch()
        
        stats = timer.get_current_stats()
        
        self.assertEqual(stats["script"], "test_script")
        self.assertGreater(stats["total_time"], 0)
        self.assertGreater(stats["active_time"], 0)
        self.assertGreaterEqual(stats["efficiency"], 0)
        self.assertLessEqual(stats["efficiency"], 100)
        self.assertEqual(stats["files_processed"], 5)
        self.assertEqual(stats["total_operations"], 1)
        self.assertEqual(stats["batches_completed"], 1)
        
    def test_idle_detection(self):
        """Test idle detection with short threshold"""
        # Use very short idle threshold for testing
        timer = self._create_timer_with_test_dir("test_script", idle_threshold=1)
        timer.start_session()
        
        # Mark activity and wait for idle
        timer.mark_activity()
        self.assertTrue(timer.is_active)
        
        # Simulate time passing beyond idle threshold
        timer.last_activity = time.time() - 2  # 2 seconds ago
        
        # Get stats should detect idle
        stats = timer.get_current_stats()
        self.assertFalse(stats["is_active"])
        
    def test_data_persistence(self):
        """Test session data is saved and can be loaded"""
        timer = self._create_timer_with_test_dir("test_script")
        timer.start_session()
        
        # Add some data
        timer.mark_batch("Persistent Batch")
        timer.log_operation("test", file_count=3)
        timer.mark_activity()
        
        # Verify session file exists and contains data
        self.assertTrue(timer.session_file.exists())
        
        with open(timer.session_file, 'r') as f:
            session_data = json.load(f)
            
        self.assertEqual(session_data["script_name"], "test_script")
        self.assertEqual(len(session_data["batches"]), 1)
        self.assertEqual(len(session_data["operations"]), 1)
        self.assertEqual(session_data["batches"][0]["name"], "Persistent Batch")
        
    def test_live_stats_printing(self):
        """Test live stats printing doesn't crash"""
        timer = self._create_timer_with_test_dir("test_script")
        timer.start_session()
        timer.mark_activity()
        
        # This should not raise an exception
        try:
            timer.print_live_stats()
        except Exception as e:
            self.fail(f"print_live_stats raised an exception: {e}")


class TestTimerReporter(unittest.TestCase):
    """Test the TimerReporter class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def _create_test_daily_data(self, date_str="20250924"):
        """Create test daily data file"""
        data_dir = self.test_dir / "timer_data"
        data_dir.mkdir(exist_ok=True)
        
        daily_file = data_dir / f"daily_{date_str}.json"
        
        test_sessions = [
            {
                "script_name": "test_script_1",
                "session_id": "20250924_100000",
                "start_time": 1695542400.0,
                "end_time": 1695546000.0,
                "active_time": 3000.0,  # 50 minutes
                "idle_time": 600.0,     # 10 minutes
                "operations": [
                    {"type": "crop", "file_count": 10},
                    {"type": "delete", "file_count": 2}
                ],
                "batches": [
                    {"name": "Batch 1", "start_time": 1695542400.0, "end_time": 1695544200.0}
                ]
            },
            {
                "script_name": "test_script_2", 
                "session_id": "20250924_110000",
                "start_time": 1695546000.0,
                "end_time": 1695548400.0,
                "active_time": 2000.0,  # 33 minutes
                "idle_time": 400.0,     # 7 minutes
                "operations": [
                    {"type": "move", "file_count": 5}
                ],
                "batches": []
            }
        ]
        
        with open(daily_file, 'w') as f:
            json.dump(test_sessions, f)
            
        return daily_file
        
    def test_daily_summary_calculation(self):
        """Test daily summary calculations"""
        # Create test data
        self._create_test_daily_data()
        
        # Create reporter with test directory
        reporter = TimerReporter()
        reporter.data_dir = self.test_dir / "timer_data"
        
        summary = reporter.daily_summary("20250924")
        
        # Check totals
        self.assertEqual(summary['total_active_time'], 5000.0)  # 50 + 33 minutes
        self.assertEqual(summary['total_session_time'], 6000.0)  # 60 + 40 minutes
        self.assertAlmostEqual(summary['efficiency'], 83.33, places=1)  # 5000/6000 * 100
        self.assertEqual(summary['total_files_processed'], 17)  # 10 + 2 + 5
        self.assertEqual(summary['session_count'], 2)
        
        # Check script breakdown
        script_stats = summary['script_breakdown']
        self.assertIn('test_script_1', script_stats)
        self.assertIn('test_script_2', script_stats)
        
        script1_stats = script_stats['test_script_1']
        self.assertEqual(script1_stats['active_time'], 3000.0)
        self.assertEqual(script1_stats['files_processed'], 12)  # 10 + 2
        self.assertEqual(script1_stats['sessions'], 1)
        
    def test_cross_script_totals(self):
        """Test cross-script totals calculation"""
        # Create test data for today and yesterday using actual current date
        from datetime import datetime, timedelta
        today = datetime.now().strftime('%Y%m%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        self._create_test_daily_data(today)
        self._create_test_daily_data(yesterday)
        
        reporter = TimerReporter()
        reporter.data_dir = self.test_dir / "timer_data"
        
        # Get totals for last 2 days
        totals = reporter.cross_script_totals(2)
        
        # Should have data from both days
        self.assertEqual(totals['total_active_time'], 10000.0)  # 2 * 5000
        self.assertEqual(totals['total_files_processed'], 34)   # 2 * 17
        
    def test_empty_data_handling(self):
        """Test handling of missing data files"""
        reporter = TimerReporter()
        reporter.data_dir = self.test_dir / "timer_data"
        
        # Try to get summary for non-existent date
        summary = reporter.daily_summary("20250101")
        self.assertEqual(summary, {})
        
        # Cross-script totals with no data
        totals = reporter.cross_script_totals(7)
        self.assertEqual(totals['total_active_time'], 0)
        self.assertEqual(totals['script_totals'], {})
        
    def test_reporting_functions_dont_crash(self):
        """Test that reporting functions handle edge cases gracefully"""
        self._create_test_daily_data()
        
        reporter = TimerReporter()
        reporter.data_dir = self.test_dir / "timer_data"
        
        # These should not raise exceptions
        try:
            reporter.print_daily_summary("20250924")
            reporter.print_cross_script_summary(7)
        except Exception as e:
            self.fail(f"Reporting functions raised an exception: {e}")


class TestActivityTimerIntegration(unittest.TestCase):
    """Integration tests for the complete activity timer system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_full_workflow_simulation(self):
        """Test a complete workflow simulation"""
        # Create timer with test directory
        timer = ActivityTimer("integration_test", idle_threshold=60)
        timer.data_dir = self.test_dir / "timer_data"
        timer.data_dir.mkdir(exist_ok=True)
        timer.session_file = timer.data_dir / f"session_{timer.session_id}.json"
        timer.daily_file = timer.data_dir / f"daily_{time.strftime('%Y%m%d')}.json"
        
        # Simulate workflow
        timer.start_session()
        
        # Batch 1
        timer.mark_batch("Image Selection Batch 1")
        timer.mark_activity()
        timer.log_operation("select", file_count=50)
        timer.log_operation("delete", file_count=10)
        timer.end_batch("Completed 50 selections")
        
        # Batch 2
        timer.mark_batch("Image Selection Batch 2")
        timer.mark_activity()
        timer.log_operation("select", file_count=30)
        timer.log_operation("move", file_count=30)
        timer.end_batch("Completed 30 selections")
        
        # End session
        timer.end_session()
        
        # Verify data was saved
        self.assertTrue(timer.daily_file.exists())
        
        # Test reporter
        reporter = TimerReporter()
        reporter.data_dir = timer.data_dir
        
        summary = reporter.daily_summary()
        
        self.assertGreater(summary['total_active_time'], 0)
        self.assertEqual(summary['total_files_processed'], 120)  # 50+10+30+30
        self.assertEqual(len(summary['script_breakdown']), 1)
        self.assertIn('integration_test', summary['script_breakdown'])
        
        script_stats = summary['script_breakdown']['integration_test']
        self.assertEqual(script_stats['operations'], 4)
        self.assertEqual(script_stats['files_processed'], 120)


def run_activity_timer_tests():
    """Run all activity timer tests"""
    print("🧪 Starting Activity Timer Test Suite...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestActivityTimer))
    suite.addTests(loader.loadTestsFromTestCase(TestTimerReporter))
    suite.addTests(loader.loadTestsFromTestCase(TestActivityTimerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n🎉 All Activity Timer tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False


if __name__ == "__main__":
    success = run_activity_timer_tests()
    sys.exit(0 if success else 1)
