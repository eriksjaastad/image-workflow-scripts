"""
Unit tests for triplet_deduplicator.py - Companion File Deletion

These tests verify that triplet_deduplicator correctly deletes ALL companion files
(.yaml, .caption, etc.) when removing duplicate triplets, not just PNG + YAML.

Coverage focus:
- remove_triplet_files: Deletes PNG + ALL companions using centralized utility
- Dry-run mode: Shows ALL companions that would be deleted
- find_all_companion_files integration: Finds .caption and other companions
- Edge cases: Missing companions, permission errors
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.triplet_deduplicator import remove_triplet_files


class TestRemoveTripletFiles(unittest.TestCase):
    """Test removing triplet files with ALL companions."""
    
    def setUp(self):
        """Create temp directory with test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()
    
    @patch('utils.triplet_deduplicator.safe_delete_paths')
    @patch('utils.triplet_deduplicator.find_all_companion_files')
    def test_removes_png_and_yaml(self, mock_find_companions, mock_safe_delete):
        """Test removing PNG with .yaml companion."""
        # Create test files
        stage1 = self.temp_path / "20250101_120000_stage1_generated.png"
        yaml1 = self.temp_path / "20250101_120000_stage1_generated.yaml"
        stage1.touch()
        yaml1.touch()
        
        # Mock find_all_companion_files to return yaml
        mock_find_companions.return_value = [yaml1]
        mock_safe_delete.return_value = [stage1, yaml1]
        
        # Call remove_triplet_files
        triplet = {'stage1': stage1}
        removed = remove_triplet_files(triplet)
        
        # Verify find_all_companion_files was called
        mock_find_companions.assert_called_once_with(stage1)
        
        # Verify safe_delete_paths was called with PNG + companions
        mock_safe_delete.assert_called_once()
        call_args = mock_safe_delete.call_args[0][0]  # First positional arg
        self.assertIn(stage1, call_args)
        self.assertIn(yaml1, call_args)
        
        # Verify return value
        self.assertEqual(len(removed), 2)
    
    @patch('utils.triplet_deduplicator.safe_delete_paths')
    @patch('utils.triplet_deduplicator.find_all_companion_files')
    def test_removes_png_with_all_companions(self, mock_find_companions, mock_safe_delete):
        """Test removing PNG with multiple companions (.yaml + .caption)."""
        # Create test files
        stage1 = self.temp_path / "20250101_120000_stage1_generated.png"
        yaml1 = self.temp_path / "20250101_120000_stage1_generated.yaml"
        caption1 = self.temp_path / "20250101_120000_stage1_generated.caption"
        stage1.touch()
        yaml1.touch()
        caption1.touch()
        
        # Mock find_all_companion_files to return both companions
        mock_find_companions.return_value = [yaml1, caption1]
        mock_safe_delete.return_value = [stage1, yaml1, caption1]
        
        # Call remove_triplet_files
        triplet = {'stage1': stage1}
        removed = remove_triplet_files(triplet)
        
        # Verify all files were passed to safe_delete_paths
        call_args = mock_safe_delete.call_args[0][0]
        self.assertIn(stage1, call_args)
        self.assertIn(yaml1, call_args)
        self.assertIn(caption1, call_args)
        
        # Verify return value includes all files
        self.assertEqual(len(removed), 3)
    
    @patch('utils.triplet_deduplicator.safe_delete_paths')
    @patch('utils.triplet_deduplicator.find_all_companion_files')
    def test_removes_complete_triplet_with_companions(self, mock_find_companions, mock_safe_delete):
        """Test removing complete triplet (stage1/1.5/2) with all companions."""
        # Create full triplet
        stage1 = self.temp_path / "20250101_120000_stage1_generated.png"
        yaml1 = self.temp_path / "20250101_120000_stage1_generated.yaml"
        stage1_5 = self.temp_path / "20250101_120000_stage1.5_face_swapped.png"
        caption1_5 = self.temp_path / "20250101_120000_stage1.5_face_swapped.caption"
        stage2 = self.temp_path / "20250101_120000_stage2_upscaled.png"
        yaml2 = self.temp_path / "20250101_120000_stage2_upscaled.yaml"
        
        for f in [stage1, yaml1, stage1_5, caption1_5, stage2, yaml2]:
            f.touch()
        
        # Mock find_all_companion_files for each stage
        def mock_find(png):
            if png == stage1:
                return [yaml1]
            elif png == stage1_5:
                return [caption1_5]
            elif png == stage2:
                return [yaml2]
            return []
        
        mock_find_companions.side_effect = mock_find
        
        # Mock safe_delete_paths to return appropriate files for each call
        def mock_delete(files, **kwargs):
            return files
        
        mock_safe_delete.side_effect = mock_delete
        
        # Call remove_triplet_files with full triplet
        triplet = {
            'stage1': stage1,
            'stage1_5': stage1_5,
            'stage2': stage2
        }
        removed = remove_triplet_files(triplet)
        
        # Verify find_all_companion_files was called for each stage
        self.assertEqual(mock_find_companions.call_count, 3)
        
        # Verify safe_delete_paths was called 3 times (once per stage)
        self.assertEqual(mock_safe_delete.call_count, 3)
        
        # Verify all 6 files were included in removed list
        self.assertEqual(len(removed), 6)
    
    @patch('utils.triplet_deduplicator.safe_delete_paths')
    @patch('utils.triplet_deduplicator.find_all_companion_files')
    def test_handles_missing_companions_gracefully(self, mock_find_companions, mock_safe_delete):
        """Test removing image when no companion files exist."""
        stage1 = self.temp_path / "20250101_120000_stage1_generated.png"
        stage1.touch()
        # No yaml or caption
        
        # Mock find_all_companion_files to return empty list
        mock_find_companions.return_value = []
        mock_safe_delete.return_value = [stage1]
        
        # Should still work
        triplet = {'stage1': stage1}
        removed = remove_triplet_files(triplet)
        
        # Verify safe_delete_paths was called with just the PNG
        call_args = mock_safe_delete.call_args[0][0]
        self.assertEqual(call_args, [stage1])
        self.assertEqual(len(removed), 1)
    
    @patch('utils.triplet_deduplicator.safe_delete_paths')
    @patch('utils.triplet_deduplicator.find_all_companion_files')
    def test_continues_on_error(self, mock_find_companions, mock_safe_delete):
        """Test that errors don't stop processing of other files."""
        stage1 = self.temp_path / "20250101_120000_stage1_generated.png"
        stage2 = self.temp_path / "20250101_120000_stage2_upscaled.png"
        stage1.touch()
        stage2.touch()
        
        # Mock find_all_companion_files
        mock_find_companions.return_value = []
        
        # Mock safe_delete_paths to raise error on first call, succeed on second
        mock_safe_delete.side_effect = [
            Exception("Permission denied"),
            [stage2]
        ]
        
        # Call with both stages
        triplet = {'stage1': stage1, 'stage2': stage2}
        removed = remove_triplet_files(triplet)
        
        # Should process stage2 even though stage1 failed
        self.assertEqual(mock_safe_delete.call_count, 2)
        # Only stage2 should be in removed (stage1 errored)
        self.assertEqual(len(removed), 1)


if __name__ == '__main__':
    unittest.main()

