#!/usr/bin/env python3
"""
Comprehensive Tests for Nearest-Up Grouping Logic
Tests every imaginable edge case to prevent future breakage
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.companion_file_utils import (
    find_consecutive_stage_groups, 
    sort_image_files_by_timestamp_and_stage, 
    get_stage_number, 
    detect_stage, 
    extract_timestamp_from_filename
)


class TestNearestUpGrouping(unittest.TestCase):
    """Comprehensive tests for nearest-up grouping logic."""

    def test_basic_functionality(self):
        """Test basic nearest-up functionality (1,3,2,3 → 1,2,3)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test case: 1, 3, 2, 3 - should pick 1→2→3 (nearest-up)
            test_files = [
                "20250705_214626_stage1_generated.png",
                "20250705_214953_stage3_enhanced.png", 
                "20250705_215137_stage2_upscaled.png",
                "20250705_215319_stage3_enhanced.png"
            ]
            
            file_paths = []
            for filename in test_files:
                file_path = temp_path / filename
                file_path.write_text("dummy content")
                file_paths.append(file_path)
            
            groups = find_consecutive_stage_groups(file_paths)
            
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0]), 3)
            
            stages = [get_stage_number(detect_stage(f.name)) for f in groups[0]]
            self.assertEqual(stages, [1.0, 2.0, 3.0])
    
    def test_erik_problematic_case(self):
        """Test Erik's specific problematic case (3,3,2,3 → 2,3)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Erik's files: stage3, stage3, stage2, stage3
            test_files = [
                "20250706_150145_stage3_enhanced.png",
                "20250706_150520_stage3_enhanced.png", 
                "20250706_150702_stage2_upscaled.png",
                "20250706_150907_stage3_enhanced.png"
            ]
            
            file_paths = []
            for filename in test_files:
                file_path = temp_path / filename
                file_path.write_text("dummy content")
                file_paths.append(file_path)
            
            groups = find_consecutive_stage_groups(file_paths)
            
            # Should create one group: stage2 → stage3
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0]), 2)
            
            stages = [get_stage_number(detect_stage(f.name)) for f in groups[0]]
            self.assertEqual(stages, [2.0, 3.0])
    
    def test_perfect_sequence(self):
        """Test perfect sequence (1 → 1.5 → 2 → 3)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_files = [
                "20250705_214626_stage1_generated.png",
                "20250705_214953_stage1.5_face_swapped.png",
                "20250705_215137_stage2_upscaled.png",
                "20250705_215319_stage3_enhanced.png"
            ]
            file_paths = [temp_path / f for f in test_files]
            for fp in file_paths:
                fp.write_text("test")
            
            groups = find_consecutive_stage_groups(file_paths)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0]), 4)
    
    def test_jump_sequence(self):
        """Test jump sequence (1 → 3, no 2)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_files = [
                "20250705_214626_stage1_generated.png",
                "20250705_214953_stage3_enhanced.png"
            ]
            file_paths = [temp_path / f for f in test_files]
            for fp in file_paths:
                fp.write_text("test")
            
            groups = find_consecutive_stage_groups(file_paths)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0]), 2)
    
    def test_same_stages_no_grouping(self):
        """Test that same stages don't group."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_files = [
                "20250705_214626_stage2_upscaled.png",
                "20250705_214953_stage2_upscaled.png",
                "20250705_215137_stage2_upscaled.png"
            ]
            file_paths = [temp_path / f for f in test_files]
            for fp in file_paths:
                fp.write_text("test")
            
            groups = find_consecutive_stage_groups(file_paths)
            self.assertEqual(len(groups), 0)
    
    def test_backwards_sequence_no_grouping(self):
        """Test that backwards sequences don't group."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_files = [
                "20250705_214626_stage3_enhanced.png",
                "20250705_214953_stage2_upscaled.png",
                "20250705_215137_stage1_generated.png"
            ]
            file_paths = [temp_path / f for f in test_files]
            for fp in file_paths:
                fp.write_text("test")
            
            groups = find_consecutive_stage_groups(file_paths)
            self.assertEqual(len(groups), 0)
    
    def test_multiple_groups(self):
        """Test multiple separate groups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_files = [
                "20250705_214626_stage1_generated.png",
                "20250705_214953_stage2_upscaled.png",
                "20250705_215137_stage1_generated.png",
                "20250705_215319_stage2_upscaled.png"
            ]
            file_paths = [temp_path / f for f in test_files]
            for fp in file_paths:
                fp.write_text("test")
            
            groups = find_consecutive_stage_groups(file_paths)
            self.assertEqual(len(groups), 2)
            self.assertEqual(len(groups[0]), 2)
            self.assertEqual(len(groups[1]), 2)
    
    def test_time_gap_functionality(self):
        """Test time gap parameter breaks groups on large time gaps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Files with large time gaps
            test_files = [
                "20250705_214626_stage1_generated.png",  # 21:46:26
                "20250705_214953_stage2_upscaled.png",   # 21:49:53 (3+ min gap)
                "20250705_215137_stage3_enhanced.png",   # 21:51:37
            ]
            
            file_paths = [temp_path / f for f in test_files]
            for fp in file_paths:
                fp.write_text("test")
            
            # Without time gap - should group all
            groups_no_gap = find_consecutive_stage_groups(file_paths)
            self.assertEqual(len(groups_no_gap), 1)
            
            # With 2 minute gap - should break at stage2
            groups_with_gap = find_consecutive_stage_groups(file_paths, time_gap_minutes=2)
            self.assertEqual(len(groups_with_gap), 1)
            self.assertEqual(len(groups_with_gap[0]), 2)  # stage2 and stage3
    
    def test_lookahead_functionality(self):
        """Test lookahead parameter affects search range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # stage1, then many stage3s, then stage2 far away
            test_files = ["20250705_214626_stage1_generated.png"]
            for i in range(20):
                test_files.append(f"20250705_214{700+i:03d}_stage3_enhanced.png")
            test_files.append("20250705_214950_stage2_upscaled.png")
            
            file_paths = [temp_path / f for f in test_files]
            for fp in file_paths:
                fp.write_text("test")
            
            # With small lookahead - should not find distant stage2
            groups_small = find_consecutive_stage_groups(file_paths, lookahead=10)
            stages_small = [get_stage_number(detect_stage(f.name)) for f in groups_small[0]]
            self.assertEqual(stages_small, [1.0, 3.0])
            
            # With large lookahead - should find distant stage2
            groups_large = find_consecutive_stage_groups(file_paths, lookahead=25)
            stages_large = [get_stage_number(detect_stage(f.name)) for f in groups_large[0]]
            self.assertEqual(stages_large, [1.0, 2.0])


if __name__ == '__main__':
    unittest.main()