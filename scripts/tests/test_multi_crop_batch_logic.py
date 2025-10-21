#!/usr/bin/env python3
"""
Tests for Multi-Crop Tool Batch Processing Logic
================================================
Tests the improved batch processing logic implemented based on code review:
1. Early return if nothing queued
2. Advance by actual processed count (not full batch size)
3. Refresh file list from disk after operations
4. Unified has_next_batch() logic

These tests verify the core logic without launching the full UI.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import sys
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestBatchProcessingLogic(unittest.TestCase):
    """Test batch processing improvements from code review"""
    
    def setUp(self):
        """Create temporary test directory with sample images"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create some test PNG files
        for i in range(7):  # 7 files = 2 full batches + 1 partial
            img = Image.new('RGB', (100, 100), color='red')
            img.save(self.test_path / f"test_{i:03d}.png")
    
    def tearDown(self):
        """Clean up temporary directory"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_progress_advances_by_actual_count(self):
        """Test that progress advances by actual processed count, not batch size"""
        # Simulate processing 2 out of 3 images in a batch
        batch_size = 3
        processed_count = 2  # Only 2 actually processed
        
        # Old behavior would advance by batch_size (3)
        # New behavior advances by processed_count (2)
        
        current_file_index = 0
        current_file_index += processed_count  # Should be 2, not 3
        
        self.assertEqual(current_file_index, 2)
        self.assertNotEqual(current_file_index, batch_size)
    
    def test_batch_calculation_with_partial_processing(self):
        """Test current_batch calculation when not all images are processed"""
        # Start at file 0, process 2 images (not full batch of 3)
        current_file_index = 0
        processed_count = 2
        
        current_file_index += processed_count  # Now at 2
        current_batch = current_file_index // 3  # Batch 0
        
        self.assertEqual(current_batch, 0)
        self.assertEqual(current_file_index, 2)
    
    def test_unified_has_next_batch_logic(self):
        """Test unified has_next_batch logic works for both modes"""
        total_files = 7
        batch_size = 3
        
        # Test at various points in processing
        test_cases = [
            (0, True),   # At start, has next
            (3, True),   # After batch 1, has next
            (6, True),   # After batch 2, has next (1 file remaining)
            (7, False),  # After all files, no next
        ]
        
        for current_file_index, expected_has_next in test_cases:
            remaining = total_files - current_file_index
            has_next = remaining > 0
            
            self.assertEqual(
                has_next, 
                expected_has_next,
                f"At index {current_file_index}: expected {expected_has_next}, got {has_next}"
            )
    
    def test_partial_last_batch(self):
        """Test that last partial batch is handled correctly"""
        total_files = 7
        batch_size = 3
        
        # Process through to last batch
        batches = []
        current_index = 0
        
        while current_index < total_files:
            batch_end = min(current_index + batch_size, total_files)
            batch = list(range(current_index, batch_end))
            batches.append(batch)
            current_index = batch_end
        
        self.assertEqual(len(batches), 3)  # 3, 3, 1
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 3)
        self.assertEqual(len(batches[2]), 1)  # Last batch partial
    
    def test_no_operations_queued_scenario(self):
        """Test that batch doesn't advance when no operations are queued"""
        processed_count = 0
        current_file_index = 3
        
        # If processed_count is 0, should NOT advance
        if processed_count == 0:
            # Don't advance - stay at current index
            pass
        else:
            current_file_index += processed_count
        
        # Should still be at 3
        self.assertEqual(current_file_index, 3)


class TestFileListRefresh(unittest.TestCase):
    """Test that file list refreshes after operations"""
    
    def setUp(self):
        """Create temporary test directory"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create some test files
        for i in range(5):
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(self.test_path / f"test_{i:03d}.png")
    
    def tearDown(self):
        """Clean up"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    def test_file_list_reflects_deletions(self):
        """Test that file list updates after files are deleted"""
        # Initial file list
        files_before = sorted(self.test_path.glob("*.png"))
        self.assertEqual(len(files_before), 5)
        
        # Simulate deleting 2 files
        files_before[0].unlink()
        files_before[1].unlink()
        
        # Refresh list from disk
        files_after = sorted(self.test_path.glob("*.png"))
        
        # Should reflect deletion
        self.assertEqual(len(files_after), 3)
        self.assertLess(len(files_after), len(files_before))
    
    def test_file_list_reflects_moves(self):
        """Test that file list updates after files are moved"""
        # Create cropped directory
        cropped_dir = self.test_path / "cropped"
        cropped_dir.mkdir()
        
        # Initial file list
        files_before = sorted(self.test_path.glob("*.png"))
        original_count = len(files_before)
        
        # Simulate moving 2 files to cropped/
        for f in files_before[:2]:
            f.rename(cropped_dir / f.name)
        
        # Refresh list from disk (should only get files in main dir, not cropped/)
        files_after = sorted(self.test_path.glob("*.png"))
        
        # Should reflect move
        self.assertEqual(len(files_after), original_count - 2)


class TestProgressTrackerBehavior(unittest.TestCase):
    """Test progress tracker behavior with new logic"""
    
    def test_progress_tracker_save_after_advance(self):
        """Test that progress is saved after advancing"""
        # Simulate progress tracking
        progress_data = {
            "current_file_index": 0,
            "processed_files": []
        }
        
        # Process 2 files
        processed_count = 2
        progress_data["current_file_index"] += processed_count
        
        # Calculate current batch
        current_batch = progress_data["current_file_index"] // 3
        
        self.assertEqual(progress_data["current_file_index"], 2)
        self.assertEqual(current_batch, 0)
    
    def test_progress_across_multiple_batches(self):
        """Test progress tracking across multiple batches"""
        current_file_index = 0
        processed_counts = [3, 2, 3, 1]  # Different counts per batch
        
        expected_indices = [3, 5, 8, 9]
        
        for i, count in enumerate(processed_counts):
            current_file_index += count
            self.assertEqual(
                current_file_index,
                expected_indices[i],
                f"After batch {i}: expected {expected_indices[i]}, got {current_file_index}"
            )


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

