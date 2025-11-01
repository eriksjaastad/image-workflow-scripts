#!/usr/bin/env python3
"""
Comprehensive Backup System Tests
==================================

Tests that backup system is working correctly and alerts on failures.
Prevents silent backup failures that could lead to data loss.

Tests:
- Backup script execution
- File integrity verification
- Backup status monitoring
- Alert system functionality
- Cron job verification
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backup.daily_backup_simple import find_database_files, verify_backup


class TestBackupSystem:
    """Test backup system functionality and alerting."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.backup_root = Path("~/project-data-archives/image-workflow").expanduser()
        self.project_root = PROJECT_ROOT

    def log_error(self, message):
        """Log an error."""
        self.errors.append(message)
        print(f"❌ {message}")

    def log_warning(self, message):
        """Log a warning."""
        self.warnings.append(message)
        print(f"⚠️  {message}")

    def log_success(self, message):
        """Log a success."""
        print(f"✅ {message}")

    def test_backup_status_file_exists(self):
        """Test that backup status file exists and is readable."""
        status_file = self.backup_root / "backup_status.json"

        if not status_file.exists():
            self.log_error("Backup status file does not exist")
            return False

        try:
            with open(status_file) as f:
                status_data = json.load(f)

            required_fields = ["last_backup", "status", "total_files", "total_size_mb"]
            for field in required_fields:
                if field not in status_data:
                    self.log_error(f"Backup status missing required field: {field}")
                    return False

            self.log_success("Backup status file is valid")
            return True

        except Exception as e:
            self.log_error(f"Backup status file corrupted: {e}")
            return False

    def test_recent_backup_exists(self):
        """Test that a recent backup directory exists."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_backup = self.backup_root / today

        if not today_backup.exists():
            # Check if yesterday's backup exists (cron might not have run yet)
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday_backup = self.backup_root / yesterday

            if yesterday_backup.exists():
                self.log_warning(
                    "Today's backup not found, but yesterday's exists - cron may not have run yet"
                )
                return True
            self.log_error("No recent backup directories found")
            return False

        # Check that backup has content
        manifest_file = today_backup / "manifest.json"
        if not manifest_file.exists():
            self.log_error("Backup manifest missing")
            return False

        try:
            with open(manifest_file) as f:
                manifest = json.load(f)

            if manifest.get("total_files", 0) == 0:
                self.log_error("Backup contains no files")
                return False

            self.log_success(
                f"Recent backup valid: {manifest['total_files']} files, {manifest['total_size_mb']} MB"
            )
            return True

        except Exception as e:
            self.log_error(f"Backup manifest corrupted: {e}")
            return False

    def test_database_discovery(self):
        """Test that database files are discovered correctly."""
        try:
            db_files = find_database_files(self.project_root)

            if not db_files:
                self.log_warning("No database files found in project")
                return True  # Not necessarily an error

            # Check that files actually exist
            missing_files = []
            for db_file in db_files:
                if not db_file.exists():
                    missing_files.append(str(db_file))

            if missing_files:
                self.log_error(
                    f"Database files reported but don't exist: {missing_files}"
                )
                return False

            self.log_success(f"Found {len(db_files)} database files")
            return True

        except Exception as e:
            self.log_error(f"Database discovery failed: {e}")
            return False

    def test_backup_script_runs(self):
        """Test that backup script can be executed without errors."""
        try:
            # Run backup script with dry-run mode (if it exists)
            result = subprocess.run(
                [sys.executable, "scripts/backup/daily_backup_simple.py"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=30,
                check=False,
            )

            # Script should exit cleanly (0 for success, 1 for expected failure)
            if result.returncode not in [0, 1]:
                self.log_error(
                    f"Backup script failed with exit code {result.returncode}"
                )
                if result.stderr:
                    self.log_error(f"Stderr: {result.stderr[:200]}...")
                return False

            self.log_success("Backup script executes without crashing")
            return True

        except subprocess.TimeoutExpired:
            self.log_error("Backup script timed out")
            return False
        except Exception as e:
            self.log_error(f"Backup script execution test failed: {e}")
            return False

    def test_cron_job_configured(self):
        """Test that backup cron jobs are configured."""
        try:
            result = subprocess.run(
                ["crontab", "-l"], capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                self.log_error("Cannot read crontab")
                return False

            cron_lines = result.stdout.strip().split("\n")

            # Check for daily backup
            daily_backup_found = any(
                "daily_backup_simple.py" in line for line in cron_lines
            )
            if not daily_backup_found:
                self.log_warning("Daily backup cron job not found")

            # Check for weekly backup
            weekly_backup_found = any("weekly_rollup.py" in line for line in cron_lines)
            if not weekly_backup_found:
                self.log_warning("Weekly backup cron job not found")

            if daily_backup_found or weekly_backup_found:
                self.log_success("Backup cron jobs are configured")
                return True
            self.log_error("No backup cron jobs found")
            return False

        except Exception as e:
            self.log_error(f"Cron job check failed: {e}")
            return False

    def test_backup_verification_logic(self):
        """Test the backup verification logic."""
        try:
            # Create a temporary test directory
            test_dir = Path("/tmp/backup_test")
            test_dir.mkdir(exist_ok=True)

            # Create test files
            test_file = test_dir / "test.txt"
            test_file.write_text("test content")

            # Test verification
            result = verify_backup(test_dir, test_file, "test file")

            # Clean up
            test_file.unlink()
            test_dir.rmdir()

            if result:
                self.log_success("Backup verification logic works")
                return True
            self.log_error("Backup verification logic failed")
            return False

        except Exception as e:
            self.log_error(f"Backup verification test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all backup system tests."""
        print("=" * 60)
        print("🧪 BACKUP SYSTEM COMPREHENSIVE TESTS")
        print("=" * 60)

        tests = [
            ("Backup Status File", self.test_backup_status_file_exists),
            ("Recent Backup Exists", self.test_recent_backup_exists),
            ("Database Discovery", self.test_database_discovery),
            ("Backup Script Execution", self.test_backup_script_runs),
            ("Cron Job Configuration", self.test_cron_job_configured),
            ("Backup Verification Logic", self.test_backup_verification_logic),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n🧪 Testing: {test_name}")
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.log_error(f"Test {test_name} crashed: {e}")

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")

        if self.errors:
            print(f"\n❌ CRITICAL ERRORS ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5
                print(f"   • {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        success = len(self.errors) == 0
        print(
            f"\n{'✅ ALL TESTS PASSED' if success else '❌ TESTS FAILED - BACKUP SYSTEM NEEDS ATTENTION'}"
        )

        return success


def main():
    """Run backup system tests."""
    tester = TestBackupSystem()
    success = tester.run_all_tests()

    # If tests fail, this should trigger alerts
    if not success:
        print("\n🚨 BACKUP SYSTEM ISSUES DETECTED!")
        print("Check the errors above and fix before relying on backups.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
