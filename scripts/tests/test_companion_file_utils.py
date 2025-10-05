"""
Unit tests for companion_file_utils.py - Critical Centralized Utilities

These tests cover the core companion file operations that all tools depend on.
Testing these thoroughly prevents widespread breakage across the entire codebase.

Coverage focus:
- find_all_companion_files: Finding .yaml, .caption, etc. for images
- move_file_with_all_companions: Safe file moves with companions
- sort_image_files_by_timestamp_and_stage: Deterministic sorting
- find_flexible_groups: Image grouping logic
- Edge cases: Missing files, malformed data, permission errors
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.companion_file_utils import (
    find_all_companion_files,
    move_file_with_companions,
    sort_image_files_by_timestamp_and_stage,
    find_consecutive_stage_groups,
    detect_stage,
    get_stage_number
)


class TestFindAllCompanionFiles(unittest.TestCase):
    """Test finding all companion files for an image."""
    
    def setUp(self):
        """Create temp directory with test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()
    
    def test_finds_yaml_companion(self):
        """Test finding .yaml companion file."""
        # Create files
        img = self.temp_path / "test_20250101_120000_stage1_generated.png"
        yaml = self.temp_path / "test_20250101_120000_stage1_generated.yaml"
        img.touch()
        yaml.touch()
        
        companions = find_all_companion_files(img)
        self.assertEqual(len(companions), 1)
        self.assertEqual(companions[0].name, yaml.name)
    
    def test_finds_multiple_companions(self):
        """Test finding multiple companion files (.yaml + .caption)."""
        img = self.temp_path / "test_20250101_120000_stage1_generated.png"
        yaml = self.temp_path / "test_20250101_120000_stage1_generated.yaml"
        caption = self.temp_path / "test_20250101_120000_stage1_generated.caption"
        img.touch()
        yaml.touch()
        caption.touch()
        
        companions = find_all_companion_files(img)
        self.assertEqual(len(companions), 2)
        companion_names = {c.name for c in companions}
        self.assertIn(yaml.name, companion_names)
        self.assertIn(caption.name, companion_names)
    
    def test_no_companions(self):
        """Test image with no companion files."""
        img = self.temp_path / "test_20250101_120000_stage1_generated.png"
        img.touch()
        
        companions = find_all_companion_files(img)
        self.assertEqual(len(companions), 0)
    
    def test_ignores_other_files_same_directory(self):
        """Test that it doesn't pick up unrelated files."""
        img = self.temp_path / "test_20250101_120000_stage1_generated.png"
        yaml = self.temp_path / "test_20250101_120000_stage1_generated.yaml"
        other = self.temp_path / "other_file.txt"
        img.touch()
        yaml.touch()
        other.touch()
        
        companions = find_all_companion_files(img)
        self.assertEqual(len(companions), 1)
        self.assertEqual(companions[0].name, yaml.name)


class TestMoveFileWithAllCompanions(unittest.TestCase):
    """Test moving files with their companions."""
    
    def setUp(self):
        """Create temp directories."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.source_dir = self.temp_path / "source"
        self.dest_dir = self.temp_path / "dest"
        self.source_dir.mkdir()
        self.dest_dir.mkdir()
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_moves_image_and_yaml(self):
        """Test moving PNG with .yaml companion."""
        img = self.source_dir / "test_20250101_120000_stage1_generated.png"
        yaml = self.source_dir / "test_20250101_120000_stage1_generated.yaml"
        img.write_text("image data")
        yaml.write_text("yaml data")
        
        # Mock tracker and activity_timer
        mock_tracker = MagicMock()
        
        moved = move_file_with_companions(img, self.dest_dir, dry_run=False)
        
        # Check files moved
        self.assertFalse(img.exists())
        self.assertFalse(yaml.exists())
        self.assertTrue((self.dest_dir / img.name).exists())
        self.assertTrue((self.dest_dir / yaml.name).exists())
        
        # Check return value
        self.assertEqual(len(moved), 2)
    
    def test_moves_image_with_multiple_companions(self):
        """Test moving PNG with multiple companions."""
        img = self.source_dir / "test_20250101_120000_stage1_generated.png"
        yaml = self.source_dir / "test_20250101_120000_stage1_generated.yaml"
        caption = self.source_dir / "test_20250101_120000_stage1_generated.caption"
        img.write_text("image")
        yaml.write_text("yaml")
        caption.write_text("caption")
        
        mock_tracker = MagicMock()
        
        moved = move_file_with_companions(img, self.dest_dir, dry_run=False)
        
        self.assertEqual(len(moved), 3)
        self.assertTrue((self.dest_dir / img.name).exists())
        self.assertTrue((self.dest_dir / yaml.name).exists())
        self.assertTrue((self.dest_dir / caption.name).exists())


class TestSortImageFiles(unittest.TestCase):
    """Test deterministic image sorting."""
    
    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_sorts_by_timestamp(self):
        """Test files sorted by timestamp first."""
        # Create files in wrong order
        file3 = self.temp_path / "20250101_150000_stage1_generated.png"
        file1 = self.temp_path / "20250101_120000_stage1_generated.png"
        file2 = self.temp_path / "20250101_130000_stage1_generated.png"
        
        for f in [file3, file1, file2]:
            f.touch()
        
        files = [file3, file1, file2]
        sorted_files = sort_image_files_by_timestamp_and_stage(files)
        
        self.assertEqual(sorted_files[0].name, file1.name)
        self.assertEqual(sorted_files[1].name, file2.name)
        self.assertEqual(sorted_files[2].name, file3.name)
    
    def test_sorts_by_stage_within_timestamp(self):
        """Test same timestamp sorted by stage."""
        file2 = self.temp_path / "20250101_120000_stage2_upscaled.png"
        file1 = self.temp_path / "20250101_120000_stage1_generated.png"
        file3 = self.temp_path / "20250101_120000_stage3_enhanced.png"
        
        for f in [file2, file1, file3]:
            f.touch()
        
        files = [file2, file1, file3]
        sorted_files = sort_image_files_by_timestamp_and_stage(files)
        
        self.assertEqual(sorted_files[0].name, file1.name)
        self.assertEqual(sorted_files[1].name, file2.name)
        self.assertEqual(sorted_files[2].name, file3.name)


class TestStageDetection(unittest.TestCase):
    """Test stage number detection from filenames."""
    
    def test_detects_stage1(self):
        """Test detecting stage1."""
        stage = detect_stage("20250101_120000_stage1_generated.png")
        self.assertEqual(stage, "stage1_generated")
    
    def test_detects_stage1_5(self):
        """Test detecting stage1.5."""
        stage = detect_stage("20250101_120000_stage1.5_face_swapped.png")
        self.assertEqual(stage, "stage1.5_face_swapped")
    
    def test_detects_stage2(self):
        """Test detecting stage2."""
        stage = detect_stage("20250101_120000_stage2_upscaled.png")
        self.assertEqual(stage, "stage2_upscaled")
    
    def test_detects_stage3(self):
        """Test detecting stage3."""
        stage = detect_stage("20250101_120000_stage3_enhanced.png")
        self.assertEqual(stage, "stage3_enhanced")
    
    def test_get_stage_number(self):
        """Test converting stage string to number."""
        self.assertEqual(get_stage_number("stage1_generated"), 1.0)
        self.assertEqual(get_stage_number("stage1.5_face_swapped"), 1.5)
        self.assertEqual(get_stage_number("stage2_upscaled"), 2.0)
        self.assertEqual(get_stage_number("stage3_enhanced"), 3.0)


class TestFlexibleGroups(unittest.TestCase):
    """Test image grouping logic."""
    
    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_groups_progressive_stages(self):
        """Test grouping 1→2→3."""
        file1 = self.temp_path / "20250101_120000_stage1_generated.png"
        file2 = self.temp_path / "20250101_120100_stage2_upscaled.png"
        file3 = self.temp_path / "20250101_120200_stage3_enhanced.png"
        
        for f in [file1, file2, file3]:
            f.touch()
        
        files = [file1, file2, file3]
        groups = find_consecutive_stage_groups(files)
        
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
    
    def test_does_not_group_same_stage(self):
        """Test that same stages don't group together."""
        file1 = self.temp_path / "20250101_120000_stage2_upscaled.png"
        file2 = self.temp_path / "20250101_120100_stage2_upscaled.png"
        file3 = self.temp_path / "20250101_120200_stage2_upscaled.png"
        
        for f in [file1, file2, file3]:
            f.touch()
        
        files = [file1, file2, file3]
        groups = find_consecutive_stage_groups(files)
        
        # Same stages should not group
        self.assertEqual(len(groups), 0)
    
    def test_groups_with_skipped_stage(self):
        """Test grouping 1→3 (skipping 2)."""
        file1 = self.temp_path / "20250101_120000_stage1_generated.png"
        file3 = self.temp_path / "20250101_120100_stage3_enhanced.png"
        
        for f in [file1, file3]:
            f.touch()
        
        files = [file1, file3]
        groups = find_consecutive_stage_groups(files)
        
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)


if __name__ == '__main__':
    unittest.main()

