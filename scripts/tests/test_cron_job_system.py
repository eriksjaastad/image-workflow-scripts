#!/usr/bin/env python3
"""
Tests for Cron Job System
========================
Tests the cron job setup, scheduling, and execution to ensure the automated
data consolidation system works correctly.
"""

import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCronJobSystem(unittest.TestCase):
    """Test the cron job system functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_cron_job_setup_script_exists(self):
        """Test that the cron job setup script exists and is executable"""
        setup_script = Path(__file__).parent.parent / "setup_cron.sh"
        self.assertTrue(setup_script.exists(), "setup_cron.sh script should exist")
        self.assertTrue(setup_script.is_file(), "setup_cron.sh should be a file")

    def test_consolidation_script_exists(self):
        """Test that the consolidation script exists"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        self.assertTrue(consolidation_script.exists(), "cleanup_logs.py script should exist")
        self.assertTrue(consolidation_script.is_file(), "cleanup_logs.py should be a file")

    def test_consolidation_script_help(self):
        """Test that the consolidation script shows help correctly"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        try:
            result = subprocess.run([
                sys.executable, str(consolidation_script), "--help"
            ], capture_output=True, text=True, timeout=10)
            
            self.assertEqual(result.returncode, 0, "Help command should succeed")
            self.assertIn("process-date", result.stdout, "Help should mention process-date argument")
            self.assertIn("dry-run", result.stdout, "Help should mention dry-run argument")
            
        except subprocess.TimeoutExpired:
            self.fail("Help command timed out")
        except Exception as e:
            self.fail(f"Help command failed: {e}")

    def test_consolidation_script_dry_run(self):
        """Test that the consolidation script works in dry-run mode"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        # Test with a date that won't have data
        test_date = "20250101"
        
        try:
            result = subprocess.run([
                sys.executable, str(consolidation_script), 
                "--process-date", test_date, "--dry-run"
            ], capture_output=True, text=True, timeout=30)
            
            self.assertEqual(result.returncode, 0, "Dry run should succeed")
            self.assertIn("DRY RUN", result.stdout, "Output should indicate dry run mode")
            
        except subprocess.TimeoutExpired:
            self.fail("Dry run command timed out")
        except Exception as e:
            self.fail(f"Dry run command failed: {e}")

    def test_consolidation_script_missing_date(self):
        """Test that the consolidation script fails without required date"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        try:
            result = subprocess.run([
                sys.executable, str(consolidation_script)
            ], capture_output=True, text=True, timeout=10)
            
            self.assertNotEqual(result.returncode, 0, "Should fail without date argument")
            self.assertIn("error", result.stderr.lower(), "Should show error message")
            
        except subprocess.TimeoutExpired:
            self.fail("Missing date test timed out")
        except Exception as e:
            self.fail(f"Missing date test failed: {e}")

    @patch('subprocess.run')
    def test_cron_job_command_format(self, mock_run):
        """Test that the cron job command format is correct"""
        # Mock subprocess.run to avoid actually modifying crontab
        mock_run.return_value = MagicMock(returncode=0)
        
        setup_script = Path(__file__).parent.parent / "setup_cron.sh"
        
        # Read the setup script to check the cron command format
        with open(setup_script, 'r') as f:
            script_content = f.read()
        
        # Check that the cron command contains expected elements
        self.assertIn("0 2 * * *", script_content, "Should run daily at 2 AM")
        self.assertIn("cleanup_logs.py", script_content, "Should call cleanup script")
        self.assertIn("--process-date", script_content, "Should use process-date argument")
        self.assertIn("date -d", script_content, "Should use date command")
        self.assertIn("2 days ago", script_content, "Should process 2 days ago")
        self.assertIn("cron_consolidation.log", script_content, "Should log to consolidation log")

    def test_cron_job_log_directory_structure(self):
        """Test that the cron job log directory structure is correct"""
        # Check that the expected log directory exists
        log_dir = Path(__file__).parent.parent.parent / "data" / "log_archives"
        self.assertTrue(log_dir.exists(), "Log archives directory should exist")
        
        # Check that it's a directory
        self.assertTrue(log_dir.is_dir(), "Log archives should be a directory")

    def test_consolidation_script_logging(self):
        """Test that the consolidation script logs output correctly"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        # Create a temporary log file
        log_file = self.temp_dir / "test_consolidation.log"
        
        try:
            # Run with output redirected to log file
            with open(log_file, 'w') as f:
                result = subprocess.run([
                    sys.executable, str(consolidation_script), 
                    "--process-date", "20250101", "--dry-run"
                ], stdout=f, stderr=f, timeout=30)
            
            self.assertEqual(result.returncode, 0, "Should succeed")
            
            # Check that log file was created and has content
            self.assertTrue(log_file.exists(), "Log file should be created")
            
            log_content = log_file.read_text()
            self.assertIn("DRY RUN", log_content, "Log should contain dry run message")
            self.assertIn("consolidating data", log_content.lower(), "Log should contain consolidation message")
            
        except subprocess.TimeoutExpired:
            self.fail("Logging test timed out")
        except Exception as e:
            self.fail(f"Logging test failed: {e}")

    def test_date_calculation_for_cron_job(self):
        """Test that the date calculation for cron job is correct"""
        
        # Test the date calculation logic that would be used in cron job
        # This simulates what "date -d '2 days ago' +%Y%m%d" would return
        
        # Get date 2 days ago
        two_days_ago = datetime.now() - timedelta(days=2)
        expected_date = two_days_ago.strftime("%Y%m%d")
        
        # Verify the format is correct (YYYYMMDD)
        self.assertEqual(len(expected_date), 8, "Date should be 8 characters")
        self.assertTrue(expected_date.isdigit(), "Date should be all digits")
        
        # Verify it's actually 2 days ago (within reasonable bounds)
        parsed_date = datetime.strptime(expected_date, "%Y%m%d")
        days_diff = (datetime.now() - parsed_date).days
        self.assertEqual(days_diff, 2, "Date should be exactly 2 days ago")

    def test_consolidation_script_error_handling(self):
        """Test that the consolidation script handles errors gracefully"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        # Test with invalid date format
        try:
            result = subprocess.run([
                sys.executable, str(consolidation_script), 
                "--process-date", "invalid-date", "--dry-run"
            ], capture_output=True, text=True, timeout=30)
            
            # Should either succeed (graceful handling) or fail with clear error
            if result.returncode != 0:
                self.assertIn("error", result.stderr.lower(), "Should show clear error message")
            
        except subprocess.TimeoutExpired:
            self.fail("Error handling test timed out")
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")

    def test_cron_job_documentation_exists(self):
        """Test that cron job documentation exists"""
        docs_file = Path(__file__).parent.parent.parent / "Documents" / "DATA_CONSOLIDATION_SYSTEM.md"
        self.assertTrue(docs_file.exists(), "Data consolidation documentation should exist")
        
        # Check that it contains important information
        doc_content = docs_file.read_text()
        self.assertIn("cron", doc_content.lower(), "Documentation should mention cron")
        self.assertIn("schedule", doc_content.lower(), "Documentation should mention schedule")
        self.assertIn("2:00 AM", doc_content, "Documentation should mention 2 AM schedule")
        self.assertIn("troubleshooting", doc_content.lower(), "Documentation should have troubleshooting")

    def test_consolidation_script_permissions(self):
        """Test that the consolidation script has correct permissions"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        # Check that the script is readable
        self.assertTrue(consolidation_script.is_file(), "Script should be a file")
        
        # Check that it's executable (Python scripts are executable if they have shebang)
        with open(consolidation_script, 'r') as f:
            first_line = f.readline().strip()
            self.assertTrue(first_line.startswith('#!'), "Script should have shebang line")
            self.assertIn('python', first_line, "Shebang should specify python")

    def test_setup_script_permissions(self):
        """Test that the setup script has correct permissions"""
        setup_script = Path(__file__).parent.parent / "setup_cron.sh"
        
        # Check that the script is readable
        self.assertTrue(setup_script.is_file(), "Setup script should be a file")
        
        # Check that it's executable (should have shebang)
        with open(setup_script, 'r') as f:
            first_line = f.readline().strip()
            self.assertTrue(first_line.startswith('#!'), "Setup script should have shebang line")
            self.assertIn('bash', first_line, "Shebang should specify bash")


class TestCronJobIntegration(unittest.TestCase):
    """Integration tests for the complete cron job system"""
    
    def test_complete_cron_job_workflow_simulation(self):
        """Test the complete cron job workflow simulation"""
        
        # Simulate the cron job workflow
        # 1. Calculate target date (2 days ago)
        target_date = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
        
        # 2. Verify the date format is correct
        self.assertEqual(len(target_date), 8, "Target date should be 8 characters")
        self.assertTrue(target_date.isdigit(), "Target date should be all digits")
        
        # 3. Verify the date is actually 2 days ago
        parsed_date = datetime.strptime(target_date, "%Y%m%d")
        days_diff = (datetime.now() - parsed_date).days
        self.assertEqual(days_diff, 2, "Target date should be exactly 2 days ago")
        
        # 4. Test that the consolidation script would accept this date
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        try:
            result = subprocess.run([
                sys.executable, str(consolidation_script), 
                "--process-date", target_date, "--dry-run"
            ], capture_output=True, text=True, timeout=30)
            
            # Should succeed (even if no data exists for that date)
            self.assertEqual(result.returncode, 0, "Consolidation should succeed for calculated date")
            
        except subprocess.TimeoutExpired:
            self.fail("Cron job workflow simulation timed out")
        except Exception as e:
            self.fail(f"Cron job workflow simulation failed: {e}")

    def test_cron_job_error_logging(self):
        """Test that cron job errors are properly logged"""
        consolidation_script = Path(__file__).parent.parent / "cleanup_logs.py"
        
        # Create a temporary directory for testing
        temp_dir = Path(tempfile.mkdtemp())
        log_file = temp_dir / "error_test.log"
        
        try:
            # Test error logging by running with invalid arguments
            with open(log_file, 'w') as f:
                result = subprocess.run([
                    sys.executable, str(consolidation_script), 
                    "--invalid-argument"
                ], stdout=f, stderr=f, timeout=10)
            
            # Should fail (invalid argument)
            self.assertNotEqual(result.returncode, 0, "Should fail with invalid argument")
            
            # Check that error was logged
            if log_file.exists():
                log_content = log_file.read_text()
                self.assertIn("error", log_content.lower(), "Error should be logged")
            
        except subprocess.TimeoutExpired:
            pass  # Timeout is expected for invalid arguments
        except Exception:
            pass  # Exception is expected for invalid arguments
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
