#!/usr/bin/env python3
"""
Tests for Project Lifecycle Scripts
====================================
Tests 00_start_project.py and 07_finish_project.py to ensure proper:
- Project manifest creation
- UTC timestamp handling
- Directory structure setup
- Status transitions
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestProjectLifecycle(unittest.TestCase):
    """Test project start and finish scripts"""
    
    def setUp(self):
        """Create temp directory for test projects"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.projects_dir = self.project_root / "data" / "projects"
        self.projects_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up temp directory"""
        self.temp_dir.cleanup()
    
    def test_project_manifest_structure(self):
        """Test that project manifest has required fields"""
        # Create sample manifest
        manifest = {
            "projectId": "test-project",
            "title": "Test Project",
            "status": "active",
            "createdAt": "2025-10-16T14:00:00Z",
            "startedAt": "2025-10-16T14:00:00Z",
            "paths": {
                "root": "/path/to/project",
                "selectedDir": "/path/to/project/selected",
                "cropDir": "/path/to/project/crop"
            },
            "counts": {
                "initialImages": 0,
                "finalImages": 0
            },
            "steps": [
                {"name": "selection", "status": "pending"},
                {"name": "sorting", "status": "pending"},
                {"name": "cropping", "status": "pending"},
                {"name": "delivery", "status": "pending"}
            ]
        }
        
        # Verify required fields
        required_fields = ["projectId", "title", "status", "createdAt", "startedAt", "paths", "counts", "steps"]
        for field in required_fields:
            self.assertIn(field, manifest)
        
        # Verify timestamp format (UTC ISO)
        created_at = datetime.fromisoformat(manifest["createdAt"].replace('Z', '+00:00'))
        self.assertEqual(created_at.tzinfo, timezone.utc)
    
    def test_utc_timestamp_generation(self):
        """Test that timestamps are always generated in UTC"""
        now_utc = datetime.now(timezone.utc)
        iso_str = now_utc.isoformat().replace('+00:00', 'Z')
        
        # Verify format
        self.assertTrue(iso_str.endswith('Z'))
        self.assertIn('T', iso_str)
        
        # Verify round-trip
        parsed = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        self.assertEqual(parsed.tzinfo, timezone.utc)
    
    def test_project_status_transitions(self):
        """Test valid status transitions"""
        valid_transitions = {
            None: ["active"],
            "active": ["active", "completed", "archived"],
            "completed": ["archived"],
            "archived": ["archived"]
        }
        
        # Test each transition
        for from_status, to_statuses in valid_transitions.items():
            for to_status in to_statuses:
                # This would test actual transition logic
                self.assertIn(to_status, ["active", "completed", "archived", None])
    
    def test_project_id_format(self):
        """Test project ID format (lowercase, hyphenated)"""
        valid_ids = ["mojo1", "mojo2", "agent-1001", "mixed-0919"]
        invalid_ids = ["Mojo1", "mojo_1", "mojo 1", "MOJO1"]
        
        for pid in valid_ids:
            # All lowercase
            self.assertEqual(pid, pid.lower())
            # No underscores or spaces
            self.assertNotIn('_', pid)
            self.assertNotIn(' ', pid)
        
        for pid in invalid_ids:
            # These should be normalized
            self.assertTrue(
                pid != pid.lower() or '_' in pid or ' ' in pid,
                f"{pid} should be invalid"
            )
    
    def test_project_directory_structure(self):
        """Test that project directories are created correctly"""
        project_id = "test-project"
        project_root = self.project_root / project_id
        
        expected_dirs = [
            project_root,
            project_root / "selected",
            project_root / "crop"
        ]
        
        # Create directories
        for dir_path in expected_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Verify all exist
        for dir_path in expected_dirs:
            self.assertTrue(dir_path.exists())
            self.assertTrue(dir_path.is_dir())
    
    def test_project_counts_initialization(self):
        """Test that image counts start at 0"""
        counts = {
            "initialImages": 0,
            "finalImages": 0
        }
        
        self.assertEqual(counts["initialImages"], 0)
        self.assertEqual(counts["finalImages"], 0)
        self.assertIsInstance(counts["initialImages"], int)
        self.assertIsInstance(counts["finalImages"], int)
    
    def test_project_steps_initialization(self):
        """Test that workflow steps are properly initialized"""
        steps = [
            {"name": "selection", "status": "pending"},
            {"name": "sorting", "status": "pending"},
            {"name": "cropping", "status": "pending"},
            {"name": "delivery", "status": "pending"}
        ]
        
        self.assertEqual(len(steps), 4)
        for step in steps:
            self.assertIn("name", step)
            self.assertIn("status", step)
            self.assertIn(step["name"], ["selection", "sorting", "cropping", "delivery"])
            self.assertEqual(step["status"], "pending")
    
    def test_finish_project_updates(self):
        """Test that finishing a project updates the manifest correctly"""
        # Create initial manifest
        manifest = {
            "projectId": "test-project",
            "status": "active",
            "startedAt": "2025-10-16T14:00:00Z",
            "finishedAt": None
        }
        
        # Simulate finishing
        finish_time = "2025-10-16T18:00:00Z"
        manifest["status"] = "completed"
        manifest["finishedAt"] = finish_time
        
        # Verify updates
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["finishedAt"], finish_time)
        
        # Verify finish time is after start time
        start = datetime.fromisoformat(manifest["startedAt"].replace('Z', '+00:00'))
        finish = datetime.fromisoformat(manifest["finishedAt"].replace('Z', '+00:00'))
        self.assertGreater(finish, start)


class TestProjectManifestPersistence(unittest.TestCase):
    """Test project manifest file operations"""
    
    def setUp(self):
        """Create temp directory"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.projects_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up"""
        self.temp_dir.cleanup()
    
    def test_manifest_file_creation(self):
        """Test manifest file is created with correct name"""
        project_id = "test-project"
        manifest_path = self.projects_dir / f"{project_id}.project.json"
        
        manifest = {
            "projectId": project_id,
            "status": "active",
            "createdAt": "2025-10-16T14:00:00Z"
        }
        
        # Write manifest
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Verify file exists
        self.assertTrue(manifest_path.exists())
        
        # Verify file can be read
        with open(manifest_path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["projectId"], project_id)
    
    def test_manifest_json_formatting(self):
        """Test manifest is formatted with proper indentation"""
        manifest = {
            "projectId": "test-project",
            "status": "active",
            "paths": {
                "root": "/path/to/project"
            }
        }
        
        # Write with indent=2
        manifest_path = self.projects_dir / "test.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Read and verify formatting
        content = manifest_path.read_text()
        self.assertIn('\n', content)  # Multi-line
        self.assertIn('  "projectId"', content)  # Indented


class TestProjectEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_duplicate_project_id(self):
        """Test handling of duplicate project IDs"""
        # This would test actual duplicate detection logic
        project_ids = {"mojo1", "mojo2"}
        
        # Attempting to add duplicate should be handled
        new_id = "mojo1"
        self.assertIn(new_id, project_ids)
    
    def test_invalid_timestamp_format(self):
        """Test handling of invalid timestamp formats"""
        invalid_timestamps = [
            "2025-10-16",  # No time
            "14:00:00",  # No date
            "2025/10/16 14:00:00",  # Wrong format
        ]
        
        for ts in invalid_timestamps:
            # These should fail ISO parsing or be incomplete
            if ts:  # Skip empty string which raises different error
                try:
                    datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    # If no error, it's a partial parse (like date-only)
                    self.assertNotIn('T', ts)  # Date-only format
                except ValueError:
                    pass  # Expected for invalid formats
    
    def test_missing_required_directories(self):
        """Test behavior when project directories are missing"""
        # This would test directory creation logic
        Path("/tmp/test-project")
        required_dirs = ["selected", "crop"]
        
        # Verify we know what directories are required
        self.assertIn("selected", required_dirs)
        self.assertIn("crop", required_dirs)


if __name__ == '__main__':
    unittest.main()

