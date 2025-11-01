#!/usr/bin/env python3
"""
Comprehensive tests for Data Consolidation System
===============================================
Tests the consolidation script, cron job functionality, and data integrity
to ensure the system we just built works correctly and prevents data loss.
"""

import gzip
import json
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataConsolidation(unittest.TestCase):
    """Test the data consolidation functionality"""

    def setUp(self):
        """Set up test environment with temporary data directories"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.file_ops_dir = self.data_dir / "file_operations_logs"
        self.summaries_dir = self.data_dir / "daily_summaries"
        self.archives_dir = self.data_dir / "log_archives"

        # Create directories
        self.file_ops_dir.mkdir(parents=True)
        self.summaries_dir.mkdir(parents=True)
        self.archives_dir.mkdir(parents=True)

        # Sample file operations data
        self.sample_operations = [
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T10:00:00Z",
                "script": "desktop_image_selector_crop",
                "session_id": "session_001",
                "operation": "crop",
                "file_count": 5,
                "files": ["image1.png", "image2.png"],
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T11:00:00Z",
                "script": "desktop_image_selector_crop",
                "session_id": "session_001",
                "operation": "delete",
                "file_count": 2,
                "files": ["bad1.png"],
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T12:00:00Z",
                "script": "character_sorter",
                "session_id": "session_002",
                "operation": "move",
                "file_count": 3,
                "files": ["char1.png"],
            },
        ]

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)

    def test_consolidate_daily_data_basic(self):
        """Test basic daily data consolidation"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create test log file
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            for op in self.sample_operations:
                f.write(json.dumps(op) + "\n")

        # Set test environment variable
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            # Test consolidation
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            # Clean up environment variable
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Check that summary file was created
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        self.assertTrue(summary_file.exists())

        # Verify summary content
        with open(summary_file) as f:
            summary = json.load(f)

        self.assertEqual(summary["date"], "20251003")
        self.assertEqual(summary["total_operations"], 3)
        self.assertIn("desktop_image_selector_crop", summary["scripts"])
        self.assertIn("character_sorter", summary["scripts"])

        # Check script data
        script_data = summary["scripts"]["desktop_image_selector_crop"]
        self.assertEqual(script_data["total_files"], 7)  # 5 + 2
        self.assertEqual(script_data["operations"]["crop"], 5)
        self.assertEqual(script_data["operations"]["delete"], 2)

    def test_consolidate_daily_data_with_timing(self):
        """Test consolidation includes work time calculation"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create test log file with timing data
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            for op in self.sample_operations:
                f.write(json.dumps(op) + "\n")

        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Check timing data
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        with open(summary_file) as f:
            summary = json.load(f)

        script_data = summary["scripts"]["desktop_image_selector_crop"]
        self.assertIn("work_time_seconds", script_data)
        self.assertIn("work_time_minutes", script_data)
        self.assertGreater(script_data["work_time_seconds"], 0)

    def test_consolidate_daily_data_session_counting(self):
        """Test session counting in consolidation"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create test log with multiple sessions
        operations_with_sessions = self.sample_operations + [
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T15:00:00Z",
                "script": "desktop_image_selector_crop",
                "session_id": "session_003",  # Different session
                "operation": "crop",
                "file_count": 1,
            }
        ]

        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            for op in operations_with_sessions:
                f.write(json.dumps(op) + "\n")

        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Check session counting
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        with open(summary_file) as f:
            summary = json.load(f)

        script_data = summary["scripts"]["desktop_image_selector_crop"]
        self.assertEqual(script_data["session_count"], 2)  # session_001 and session_003

    def test_consolidate_daily_data_file_counting(self):
        """Test accurate file counting in consolidation"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create test log with various file counts
        operations = [
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T10:00:00Z",
                "script": "test_script",
                "session_id": "session_001",
                "operation": "move",
                "file_count": 5,
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T11:00:00Z",
                "script": "test_script",
                "session_id": "session_001",
                "operation": "move",
                "file_count": 3,
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T12:00:00Z",
                "script": "test_script",
                "session_id": "session_001",
                "operation": "delete",
                "file_count": 2,
            },
        ]

        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            for op in operations:
                f.write(json.dumps(op) + "\n")

        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Check file counting
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        with open(summary_file) as f:
            summary = json.load(f)

        script_data = summary["scripts"]["test_script"]
        self.assertEqual(script_data["total_files"], 10)  # 5 + 3 + 2
        self.assertEqual(script_data["operations"]["move"], 8)  # 5 + 3
        self.assertEqual(script_data["operations"]["delete"], 2)

    def test_consolidate_daily_data_handles_null_file_count(self):
        """Test consolidation handles null/None file_count gracefully"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create test log with null file_count
        operations_with_null = [
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T10:00:00Z",
                "script": "test_script",
                "session_id": "session_001",
                "operation": "move",
                "file_count": None,  # This should not crash
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T11:00:00Z",
                "script": "test_script",
                "session_id": "session_001",
                "operation": "move",
                "file_count": 5,
            },
        ]

        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            for op in operations_with_null:
                f.write(json.dumps(op) + "\n")

        # Should not crash
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Check that it handled the null gracefully
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        with open(summary_file) as f:
            summary = json.load(f)

        script_data = summary["scripts"]["test_script"]
        self.assertEqual(script_data["total_files"], 5)  # Only counted the non-null

    def test_consolidate_daily_data_archives_old_logs(self):
        """Test that old logs are archived after consolidation"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create old log file (3 days ago)
        old_date = "20250930"
        old_log_file = self.file_ops_dir / f"file_operations_{old_date}.log"
        with open(old_log_file, "w") as f:
            f.write(json.dumps(self.sample_operations[0]) + "\n")

        # Create current log file
        current_date = "20251003"
        current_log_file = self.file_ops_dir / f"file_operations_{current_date}.log"
        with open(current_log_file, "w") as f:
            f.write(json.dumps(self.sample_operations[0]) + "\n")

        # Consolidate current date (this should archive the old log)
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data(current_date, dry_run=False)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Check that old log was archived
        archived_file = self.archives_dir / f"file_operations_{old_date}.log.gz"
        self.assertTrue(archived_file.exists())

        # Check that old log was removed
        self.assertFalse(old_log_file.exists())

        # Check that current log still exists
        self.assertTrue(current_log_file.exists())

    def test_consolidate_daily_data_dry_run_mode(self):
        """Test that dry run mode doesn't modify files"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create test log file
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            f.write(json.dumps(self.sample_operations[0]) + "\n")

        # Create old log file
        old_log_file = self.file_ops_dir / "file_operations_20250930.log"
        with open(old_log_file, "w") as f:
            f.write(json.dumps(self.sample_operations[0]) + "\n")

        # Run in dry run mode
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Check that summary was created (dry run still creates summaries)
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        self.assertTrue(summary_file.exists())

        # Check that old log was NOT archived (dry run doesn't modify files)
        archived_file = self.archives_dir / "file_operations_20250930.log.gz"
        self.assertFalse(archived_file.exists())

        # Check that old log still exists
        self.assertTrue(old_log_file.exists())

    def test_consolidate_daily_data_handles_missing_files(self):
        """Test consolidation handles missing log files gracefully"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Set test environment variable
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            # Try to consolidate a date with no data
            consolidate_daily_data("20250999", dry_run=True)

            # Should not crash and should create empty summary
            summary_file = self.summaries_dir / "daily_summary_20250999.json"
            self.assertTrue(summary_file.exists())
        finally:
            # Clean up environment variable
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        with open(summary_file) as f:
            summary = json.load(f)

        self.assertEqual(summary["total_operations"], 0)
        self.assertEqual(summary["scripts"], {})

    def test_consolidate_daily_data_handles_malformed_json(self):
        """Test consolidation handles malformed JSON gracefully"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create log file with malformed JSON with proper file operation format
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            f.write(
                '{"type": "file_operation", "timestamp": "2025-10-03T10:00:00Z", "script": "test", "operation": "test", "file_count": 1}\n'
            )
            f.write("invalid json line\n")
            f.write(
                '{"type": "file_operation", "timestamp": "2025-10-03T11:00:00Z", "script": "test", "operation": "test", "file_count": 1}\n'
            )

        # Should not crash
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Should process valid records
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        self.assertTrue(summary_file.exists())

    @patch("cleanup_logs.DashboardDataEngine")
    def test_consolidate_daily_data_dashboard_verification(self, mock_dashboard_class):
        """Test dashboard verification in consolidation"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Mock the dashboard data engine
        mock_engine = MagicMock()
        mock_dashboard_class.return_value = mock_engine
        mock_engine.load_file_operations.return_value = [
            {"date": date(2025, 10, 3), "script": "test", "file_count": 5}
        ]

        # Create test log file
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            f.write(json.dumps(self.sample_operations[0]) + "\n")

        # Test consolidation with verification
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Verify that dashboard verification was called
        mock_engine.load_file_operations.assert_called_once()

        # Check that summary file was created
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        self.assertTrue(summary_file.exists())

    @patch("cleanup_logs.DashboardDataEngine")
    def test_consolidate_daily_data_verification_failure(self, mock_dashboard_class):
        """Test that verification failure prevents consolidation"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Mock the dashboard data engine to fail verification
        mock_engine = MagicMock()
        mock_dashboard_class.return_value = mock_engine
        mock_engine.load_file_operations.return_value = []  # No records found

        # Create test log file
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            f.write(json.dumps(self.sample_operations[0]) + "\n")

        # Test that verification failure raises exception
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            with self.assertRaises(Exception) as context:
                consolidate_daily_data("20251003", dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        self.assertIn("Dashboard verification failed", str(context.exception))

        # Check that summary file was removed due to verification failure
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        self.assertFalse(summary_file.exists())


class TestConsolidationIntegration(unittest.TestCase):
    """Integration tests for the complete consolidation system"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.file_ops_dir = self.data_dir / "file_operations_logs"
        self.summaries_dir = self.data_dir / "daily_summaries"
        self.archives_dir = self.data_dir / "log_archives"

        # Create directories
        self.file_ops_dir.mkdir(parents=True)
        self.summaries_dir.mkdir(parents=True)
        self.archives_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_consolidation_workflow(self):
        """Test complete consolidation workflow from logs to dashboard"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create comprehensive test data
        operations = [
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T09:00:00Z",
                "script": "desktop_image_selector_crop",
                "session_id": "session_001",
                "operation": "crop",
                "file_count": 10,
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T09:30:00Z",
                "script": "desktop_image_selector_crop",
                "session_id": "session_001",
                "operation": "delete",
                "file_count": 5,
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-03T10:00:00Z",
                "script": "character_sorter",
                "session_id": "session_002",
                "operation": "move",
                "file_count": 8,
            },
        ]

        # Create log file
        log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(log_file, "w") as f:
            for op in operations:
                f.write(json.dumps(op) + "\n")

        # Run consolidation
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=False)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Verify summary was created
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        self.assertTrue(summary_file.exists())

        # Verify summary content
        with open(summary_file) as f:
            summary = json.load(f)

        self.assertEqual(summary["date"], "20251003")
        self.assertEqual(summary["total_operations"], 3)

        # Verify script data
        desktop_data = summary["scripts"]["desktop_image_selector_crop"]
        self.assertEqual(desktop_data["total_files"], 15)  # 10 + 5
        self.assertEqual(desktop_data["operations"]["crop"], 10)
        self.assertEqual(desktop_data["operations"]["delete"], 5)
        self.assertEqual(desktop_data["session_count"], 1)

        character_data = summary["scripts"]["character_sorter"]
        self.assertEqual(character_data["total_files"], 8)
        self.assertEqual(character_data["operations"]["move"], 8)
        self.assertEqual(character_data["session_count"], 1)

        # Verify timing data
        self.assertIn("work_time_seconds", desktop_data)
        self.assertIn("work_time_minutes", desktop_data)
        self.assertGreater(desktop_data["work_time_seconds"], 0)

    def test_multiple_days_consolidation(self):
        """Test consolidating multiple days of data"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create data for multiple days
        for day in ["20251001", "20251002", "20251003"]:
            log_file = self.file_ops_dir / f"file_operations_{day}.log"
            with open(log_file, "w") as f:
                op = {
                    "type": "file_operation",
                    "timestamp": f"2025-10-{day[-2:]}T10:00:00Z",
                    "script": "test_script",
                    "session_id": f"session_{day}",
                    "operation": "move",
                    "file_count": int(day[-1]),  # 1, 2, 3
                }
                f.write(json.dumps(op) + "\n")

        # Consolidate each day
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            for day in ["20251001", "20251002", "20251003"]:
                consolidate_daily_data(day, dry_run=True)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Verify all summaries were created
        for day in ["20251001", "20251002", "20251003"]:
            summary_file = self.summaries_dir / f"daily_summary_{day}.json"
            self.assertTrue(summary_file.exists())

            with open(summary_file) as f:
                summary = json.load(f)

            self.assertEqual(summary["date"], day)
            self.assertEqual(
                summary["scripts"]["test_script"]["total_files"], int(day[-1])
            )

    def test_consolidation_with_archived_data(self):
        """Test consolidation with existing archived data"""
        import os

        from cleanup_logs import consolidate_daily_data

        # Create existing archived file
        archived_file = self.archives_dir / "file_operations_20250930.log.gz"
        with gzip.open(archived_file, "wt") as f:
            f.write(
                json.dumps(
                    {
                        "type": "file_operation",
                        "timestamp": "2025-09-30T10:00:00Z",
                        "script": "old_script",
                        "operation": "move",
                        "file_count": 5,
                    }
                )
                + "\n"
            )

        # Create current log file
        current_log_file = self.file_ops_dir / "file_operations_20251003.log"
        with open(current_log_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "type": "file_operation",
                        "timestamp": "2025-10-03T10:00:00Z",
                        "script": "current_script",
                        "operation": "move",
                        "file_count": 3,
                    }
                )
                + "\n"
            )

        # Consolidate current date
        os.environ["EM_TEST_DATA_ROOT"] = str(self.temp_dir / "data")
        try:
            consolidate_daily_data("20251003", dry_run=False)
        finally:
            if "EM_TEST_DATA_ROOT" in os.environ:
                del os.environ["EM_TEST_DATA_ROOT"]

        # Verify archived file still exists
        self.assertTrue(archived_file.exists())

        # Verify current summary was created
        summary_file = self.summaries_dir / "daily_summary_20251003.json"
        self.assertTrue(summary_file.exists())

        with open(summary_file) as f:
            summary = json.load(f)

        self.assertEqual(summary["scripts"]["current_script"]["total_files"], 3)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
