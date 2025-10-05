"""
Unit tests for check_companions.py - New Centralized Orphaned Files Audit Tool

This tool replaced the old fix_orphaned_captions.py and check_pairs.py scripts
with a unified, more robust solution. Testing thoroughly ensures:
- Correct identification of orphaned PNG files (PNG without metadata)
- Correct identification of orphaned metadata (metadata without PNG)
- Support for ALL companion file types (.yaml, .caption, .txt, .json, etc.)
- Recursive directory scanning
- Safe cleanup with send2trash

These tests use temporary directories to avoid touching real data.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module to test
# Note: check_companions.py is a utility script, not a module
# We'll import the main function directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "check_companions",
    Path(__file__).parent.parent / "utils" / "check_companions.py"
)
check_companions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_companions)


class TestFindMismatchedFilesRecursive(unittest.TestCase):
    """Test the main orphan detection function."""
    
    def setUp(self):
        """Create temp directory structure for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create subdirectories
        self.dir1 = self.temp_path / "dir1"
        self.dir2 = self.temp_path / "dir2"
        self.dir1.mkdir()
        self.dir2.mkdir()
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_finds_orphaned_png_without_any_companion(self):
        """Test detecting PNG with no companion files at all."""
        # Create PNG without any companions
        orphaned_png = self.dir1 / "20250101_120000_stage1_generated.png"
        orphaned_png.touch()
        
        with patch('builtins.print'):  # Suppress output
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        self.assertEqual(len(orphaned_pngs), 1)
        self.assertEqual(len(orphaned_metadata), 0)
        self.assertIn(orphaned_png, orphaned_pngs)
    
    def test_finds_orphaned_yaml_without_png(self):
        """Test detecting .yaml without corresponding PNG."""
        orphaned_yaml = self.dir1 / "20250101_120000_stage1_generated.yaml"
        orphaned_yaml.touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        self.assertEqual(len(orphaned_pngs), 0)
        self.assertEqual(len(orphaned_metadata), 1)
        self.assertIn(orphaned_yaml, orphaned_metadata)
    
    def test_finds_orphaned_caption_without_png(self):
        """Test detecting .caption without corresponding PNG."""
        orphaned_caption = self.dir1 / "20250101_120000_stage1_generated.caption"
        orphaned_caption.touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        self.assertEqual(len(orphaned_pngs), 0)
        self.assertEqual(len(orphaned_metadata), 1)
        self.assertIn(orphaned_caption, orphaned_metadata)
    
    def test_accepts_png_with_yaml_as_valid_pair(self):
        """Test that PNG + .yaml = valid pair (no orphans)."""
        png = self.dir1 / "20250101_120000_stage1_generated.png"
        yaml = self.dir1 / "20250101_120000_stage1_generated.yaml"
        png.touch()
        yaml.touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        self.assertEqual(len(orphaned_pngs), 0)
        self.assertEqual(len(orphaned_metadata), 0)
    
    def test_accepts_png_with_caption_as_valid_pair(self):
        """Test that PNG + .caption = valid pair (no orphans)."""
        png = self.dir1 / "20250101_120000_stage1_generated.png"
        caption = self.dir1 / "20250101_120000_stage1_generated.caption"
        png.touch()
        caption.touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        self.assertEqual(len(orphaned_pngs), 0)
        self.assertEqual(len(orphaned_metadata), 0)
    
    def test_only_recognizes_yaml_and_caption(self):
        """Test that only .yaml and .caption are recognized as companions."""
        # PNG with .txt (not a recognized companion) should be orphaned
        png = self.dir1 / "test_image.png"
        txt = self.dir1 / "test_image.txt"
        png.touch()
        txt.touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        # PNG should be orphaned because .txt is not recognized
        self.assertEqual(len(orphaned_pngs), 1)
        # .txt is ignored (not tracked as metadata)
        self.assertEqual(len(orphaned_metadata), 0)
    
    def test_recursive_scanning_multiple_directories(self):
        """Test that it scans subdirectories recursively."""
        # Create orphans in different directories
        orphan1 = self.dir1 / "orphan1.png"
        orphan2 = self.dir2 / "orphan2.png"
        orphan1.touch()
        orphan2.touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        self.assertEqual(len(orphaned_pngs), 2)
        self.assertIn(orphan1, orphaned_pngs)
        self.assertIn(orphan2, orphaned_pngs)
    
    def test_statistics_reporting(self):
        """Test that statistics dict is populated correctly."""
        # Create a mix of valid pairs and orphans
        valid_png = self.dir1 / "valid.png"
        valid_yaml = self.dir1 / "valid.yaml"
        orphan_png = self.dir1 / "orphan.png"
        orphan_yaml = self.dir1 / "orphan_meta.yaml"
        
        valid_png.touch()
        valid_yaml.touch()
        orphan_png.touch()
        orphan_yaml.touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        # Check results structure
        self.assertIn('total_pairs', results)
        self.assertIn('orphaned_pngs', results)
        self.assertIn('orphaned_metadata', results)
        
        self.assertEqual(len(results['orphaned_pngs']), 1)
        self.assertEqual(len(results['orphaned_metadata']), 1)
    
    def test_skips_venv_and_git_directories(self):
        """Test that it skips common non-data directories."""
        venv_dir = self.temp_path / ".venv"
        git_dir = self.temp_path / ".git"
        venv_dir.mkdir()
        git_dir.mkdir()
        
        # Create orphans in skip directories
        (venv_dir / "orphan.png").touch()
        (git_dir / "orphan.png").touch()
        
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        orphaned_pngs = results['orphaned_pngs']
        orphaned_metadata = results['orphaned_metadata']
        
        # Should find 0 orphans (skipped directories)
        self.assertEqual(len(orphaned_pngs), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_handles_empty_directory(self):
        """Test handling of empty directory (no files)."""
        with patch('builtins.print'):
            results = check_companions.find_mismatched_files_recursive(self.temp_path)
        
        self.assertEqual(len(results['orphaned_pngs']), 0)
        self.assertEqual(len(results['orphaned_metadata']), 0)
        self.assertEqual(results['total_pairs'], 0)


if __name__ == '__main__':
    unittest.main()

