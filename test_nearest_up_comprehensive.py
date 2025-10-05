#!/usr/bin/env python3
"""
Comprehensive Tests for Nearest-Up Grouping Logic
Tests every imaginable edge case to prevent future breakage
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.utils.companion_file_utils import find_consecutive_stage_groups, sort_image_files_by_timestamp_and_stage, get_stage_number, detect_stage, extract_timestamp_from_filename

def test_basic_functionality():
    """Test basic nearest-up functionality"""
    print("\n🧪 Testing Basic Nearest-Up Functionality...")
    
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
        
        assert len(groups) == 1, f"Expected 1 group, got {len(groups)}"
        assert len(groups[0]) == 3, f"Expected 3 files in group, got {len(groups[0])}"
        
        stages = [get_stage_number(detect_stage(f.name)) for f in groups[0]]
        assert stages == [1.0, 2.0, 3.0], f"Expected [1.0, 2.0, 3.0], got {stages}"
        
        print("✅ Basic nearest-up functionality works")

def test_erik_problematic_case():
    """Test Erik's specific problematic case"""
    print("\n🧪 Testing Erik's Problematic Case...")
    
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
        assert len(groups) == 1, f"Expected 1 group, got {len(groups)}"
        assert len(groups[0]) == 2, f"Expected 2 files in group, got {len(groups[0])}"
        
        stages = [get_stage_number(detect_stage(f.name)) for f in groups[0]]
        assert stages == [2.0, 3.0], f"Expected [2.0, 3.0], got {stages}"
        
        print("✅ Erik's problematic case fixed")

def test_all_edge_cases():
    """Test every imaginable edge case"""
    print("\n🧪 Testing All Edge Cases...")
    
    test_cases = [
        # Case 1: Perfect sequence
        ("Perfect sequence", ["20250705_214626_stage1_generated.png", "20250705_214953_stage1.5_face_swapped.png", "20250705_215137_stage2_upscaled.png", "20250705_215319_stage3_enhanced.png"], 1, [4], [1.0, 1.5, 2.0, 3.0]),
        
        # Case 2: Jump sequence (1→3, no 2)
        ("Jump sequence", ["20250705_214626_stage1_generated.png", "20250705_214953_stage3_enhanced.png"], 1, [2], [1.0, 3.0]),
        
        # Case 3: Scattered with nearest-up
        ("Scattered nearest-up", ["20250705_214626_stage1_generated.png", "20250705_214953_stage3_enhanced.png", "20250705_215137_stage2_upscaled.png", "20250705_215319_stage3_enhanced.png"], 1, [3], [1.0, 2.0, 3.0]),
        
        # Case 4: Same stages (should not group)
        ("Same stages", ["20250705_214626_stage2_upscaled.png", "20250705_214953_stage2_upscaled.png", "20250705_215137_stage2_upscaled.png"], 0, [], []),
        
        # Case 5: Backwards sequence
        ("Backwards sequence", ["20250705_214626_stage3_enhanced.png", "20250705_214953_stage2_upscaled.png", "20250705_215137_stage1_generated.png"], 0, [], []),
        
        # Case 6: Mixed with gaps
        ("Mixed with gaps", ["20250705_214626_stage1_generated.png", "20250705_214953_stage1.5_face_swapped.png", "20250705_215137_stage3_enhanced.png", "20250705_215319_stage2_upscaled.png"], 1, [3], [1.0, 1.5, 2.0]),
        
        # Case 7: Multiple groups
        ("Multiple groups", ["20250705_214626_stage1_generated.png", "20250705_214953_stage2_upscaled.png", "20250705_215137_stage1_generated.png", "20250705_215319_stage2_upscaled.png"], 2, [2, 2], [1.0, 2.0, 1.0, 2.0]),
        
        # Case 8: Complex scattered
        ("Complex scattered", ["20250705_214626_stage1_generated.png", "20250705_214953_stage3_enhanced.png", "20250705_215137_stage1.5_face_swapped.png", "20250705_215319_stage2_upscaled.png", "20250705_215500_stage3_enhanced.png"], 1, [4], [1.0, 1.5, 2.0, 3.0]),
        
        # Case 9: Single file (should not group)
        ("Single file", ["20250705_214626_stage1_generated.png"], 0, [], []),
        
        # Case 10: No valid progression (stage2→stage1→stage3: only stage1→stage3 is valid)
        ("No valid progression", ["20250705_214626_stage2_upscaled.png", "20250705_214953_stage1_generated.png", "20250705_215137_stage3_enhanced.png"], 1, [2], [1.0, 3.0]),
    ]
    
    for description, filenames, expected_groups, expected_sizes, expected_stages in test_cases:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            file_paths = []
            for filename in filenames:
                file_path = temp_path / filename
                file_path.write_text("dummy content")
                file_paths.append(file_path)
            
            groups = find_consecutive_stage_groups(file_paths)
            
            assert len(groups) == expected_groups, f"{description}: Expected {expected_groups} groups, got {len(groups)}"
            
            if expected_groups > 0:
                actual_sizes = [len(group) for group in groups]
                assert actual_sizes == expected_sizes, f"{description}: Expected sizes {expected_sizes}, got {actual_sizes}"
                
                # Check stages for groups
                if expected_stages:
                    all_stages = []
                    for group in groups:
                        group_stages = [get_stage_number(detect_stage(f.name)) for f in group]
                        all_stages.extend(group_stages)
                    assert all_stages == expected_stages, f"{description}: Expected stages {expected_stages}, got {all_stages}"
            
            print(f"✅ {description}")

def test_time_gap_functionality():
    """Test time gap functionality"""
    print("\n🧪 Testing Time Gap Functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Files with large time gaps
        test_files = [
            "20250705_214626_stage1_generated.png",  # 21:46:26
            "20250705_214953_stage2_upscaled.png",   # 21:49:53 (3+ min gap)
            "20250705_215137_stage3_enhanced.png",   # 21:51:37
        ]
        
        file_paths = []
        for filename in test_files:
            file_path = temp_path / filename
            file_path.write_text("dummy content")
            file_paths.append(file_path)
        
        # Without time gap - should group all
        groups_no_gap = find_consecutive_stage_groups(file_paths)
        assert len(groups_no_gap) == 1, "Should group all without time gap"
        
        # With 2 minute gap - should break at stage2 (time gap between stage1 and stage2 is 3+ minutes)
        groups_with_gap = find_consecutive_stage_groups(file_paths, time_gap_minutes=2)
        assert len(groups_with_gap) == 1, "Should create 1 group with later files when time gap breaks early group"
        assert len(groups_with_gap[0]) == 2, "Should have 2 files in the group (stage2 and stage3)"
        
        print("✅ Time gap functionality works")

def test_lookahead_functionality():
    """Test lookahead functionality"""
    print("\n🧪 Testing Lookahead Functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a scenario where lookahead matters:
        # stage1, then many stage3s, then stage2 at position 15
        test_files = []
        test_files.append("20250705_214626_stage1_generated.png")
        for i in range(20):  # 20 stage3 files
            test_files.append(f"20250705_214{700+i:03d}_stage3_enhanced.png")
        test_files.append("20250705_214950_stage2_upscaled.png")  # stage2 at position 22
        
        file_paths = []
        for filename in test_files:
            file_path = temp_path / filename
            file_path.write_text("dummy content")
            file_paths.append(file_path)
        
        # With small lookahead (10) - should not find stage2, so stage1→stage3
        groups_small = find_consecutive_stage_groups(file_paths, lookahead=10)
        
        # With larger lookahead (25) - should find stage2, so stage1→stage2→stage3
        groups_large = find_consecutive_stage_groups(file_paths, lookahead=25)
        
        # Should create different results
        assert len(groups_small[0]) == 2, f"Small lookahead should create 2-file group, got {len(groups_small[0])}"
        assert len(groups_large[0]) == 2, f"Large lookahead should create 2-file group (stage1→stage2), got {len(groups_large[0])}"
        
        # Verify the actual stages
        stages_small = [get_stage_number(detect_stage(f.name)) for f in groups_small[0]]
        stages_large = [get_stage_number(detect_stage(f.name)) for f in groups_large[0]]
        assert stages_small == [1.0, 3.0], f"Small lookahead should find [1.0, 3.0], got {stages_small}"
        assert stages_large == [1.0, 2.0], f"Large lookahead should find [1.0, 2.0], got {stages_large}"
        
        print("✅ Lookahead functionality works")

def run_all_tests():
    """Run all comprehensive tests"""
    print("🧪 COMPREHENSIVE NEAREST-UP GROUPING TEST SUITE")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_erik_problematic_case()
        test_all_edge_cases()
        test_time_gap_functionality()
        test_lookahead_functionality()
        
        print("\n" + "=" * 60)
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("The nearest-up grouping logic is robust and handles all edge cases.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
