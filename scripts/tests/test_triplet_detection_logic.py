#!/usr/bin/env python3
"""
Comprehensive Tests for Triplet Detection Logic
Tests the actual grouping behavior to catch bugs like same-stage grouping
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_all_consecutive_combinations():
    """Test ALL valid consecutive stage combinations"""
    print("\n🧪 Testing All Consecutive Stage Combinations...")

    try:
        from scripts.utils.companion_file_utils import (
            detect_stage,
            find_consecutive_stage_groups,
            get_stage_number,
            sort_image_files_by_timestamp_and_stage,
        )

        # Test all valid consecutive combinations
        test_cases = [
            # (description, files, expected_groups, expected_group_sizes)
            (
                "1→1.5",
                [
                    "20250705_214626_stage1_generated.png",
                    "20250705_214953_stage1.5_face_swapped.png",
                ],
                1,
                [2],
            ),
            (
                "1→2",
                [
                    "20250705_214626_stage1_generated.png",
                    "20250705_214953_stage2_upscaled.png",
                ],
                1,
                [2],
            ),
            (
                "1→3",
                [
                    "20250705_214626_stage1_generated.png",
                    "20250705_214953_stage3_enhanced.png",
                ],
                1,
                [2],
            ),
            (
                "1.5→2",
                [
                    "20250705_214626_stage1.5_face_swapped.png",
                    "20250705_214953_stage2_upscaled.png",
                ],
                1,
                [2],
            ),
            (
                "1.5→3",
                [
                    "20250705_214626_stage1.5_face_swapped.png",
                    "20250705_214953_stage3_enhanced.png",
                ],
                1,
                [2],
            ),
            (
                "2→3",
                [
                    "20250705_214626_stage2_upscaled.png",
                    "20250705_214953_stage3_enhanced.png",
                ],
                1,
                [2],
            ),
            (
                "1→1.5→2",
                [
                    "20250705_214626_stage1_generated.png",
                    "20250705_214953_stage1.5_face_swapped.png",
                    "20250705_215137_stage2_upscaled.png",
                ],
                1,
                [3],
            ),
            (
                "1→1.5→3",
                [
                    "20250705_214626_stage1_generated.png",
                    "20250705_214953_stage1.5_face_swapped.png",
                    "20250705_215137_stage3_enhanced.png",
                ],
                1,
                [3],
            ),
            (
                "1→2→3",
                [
                    "20250705_214626_stage1_generated.png",
                    "20250705_214953_stage2_upscaled.png",
                    "20250705_215137_stage3_enhanced.png",
                ],
                1,
                [3],
            ),
            (
                "1.5→2→3",
                [
                    "20250705_214626_stage1.5_face_swapped.png",
                    "20250705_214953_stage2_upscaled.png",
                    "20250705_215137_stage3_enhanced.png",
                ],
                1,
                [3],
            ),
            (
                "1→1.5→2→3",
                [
                    "20250705_214626_stage1_generated.png",
                    "20250705_214953_stage1.5_face_swapped.png",
                    "20250705_215137_stage2_upscaled.png",
                    "20250705_215319_stage3_enhanced.png",
                ],
                1,
                [4],
            ),
        ]

        for description, test_files, expected_groups, expected_sizes in test_cases:
            print(f"  Testing {description}: {test_files}")

            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                file_paths = []

                for filename in test_files:
                    file_path = temp_path / filename
                    file_path.write_text("dummy content")
                    file_paths.append(file_path)

                # Pre-sort by timestamp and stage as required by the algorithm
                sorted_paths = sort_image_files_by_timestamp_and_stage(file_paths)
                groups = find_consecutive_stage_groups(sorted_paths)

                # Verify expected number of groups
                assert (
                    len(groups) == expected_groups
                ), f"{description}: Expected {expected_groups} groups, got {len(groups)}"

                # Verify expected group sizes
                actual_sizes = [len(group) for group in groups]
                assert (
                    actual_sizes == expected_sizes
                ), f"{description}: Expected group sizes {expected_sizes}, got {actual_sizes}"

                # Verify stage progression is consecutive
                for group in groups:
                    stages = [get_stage_number(detect_stage(f.name)) for f in group]
                    print(f"    Stages: {stages}")

                    # Check that stages are strictly increasing (each stage should be > previous)
                    for i in range(len(stages) - 1):
                        assert (
                            stages[i + 1] > stages[i]
                        ), f"{description}: Stage {stages[i+1]} should be > {stages[i]}"

                        # Stages should be in the ordered_stages list and progressing forward
                        # This allows 1→2, 1→3, 1.5→3, etc. as long as they're in order
                        if stages[i + 1] != stages[i]:
                            # Just verify they're both valid stages and progressing forward
                            valid_stages = [1.0, 1.5, 2.0, 3.0]
                            assert (
                                stages[i] in valid_stages
                            ), f"{description}: Stage {stages[i]} not in valid stages"
                            assert (
                                stages[i + 1] in valid_stages
                            ), f"{description}: Stage {stages[i+1]} not in valid stages"

        print("✅ All consecutive combinations test PASSED")

    except Exception as e:
        print(f"❌ All consecutive combinations test FAILED: {e}")
        assert False


def test_same_stage_not_grouped():
    """Test that same stages are NOT grouped together (this would catch the bug)"""
    print("\n🧪 Testing Same Stage NOT Grouped...")

    try:
        from scripts.utils.companion_file_utils import (
            find_consecutive_stage_groups,
            sort_image_files_by_timestamp_and_stage,
        )

        # Create test files with ONLY same stages
        test_files = [
            "20250705_214626_stage2_upscaled.png",
            "20250705_214953_stage2_upscaled.png",  # Same stage
            "20250705_215137_stage2_upscaled.png",  # Same stage
            "20250705_215319_stage2_upscaled.png",  # Same stage
        ]

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_paths = []

            for filename in test_files:
                file_path = temp_path / filename
                file_path.write_text("dummy content")
                file_paths.append(file_path)

            # Pre-sort by timestamp and stage
            sorted_paths = sort_image_files_by_timestamp_and_stage(file_paths)
            groups = find_consecutive_stage_groups(sorted_paths)

            print(f"  Found {len(groups)} groups")
            for i, group in enumerate(groups):
                filenames = [f.name for f in group]
                print(f"  Group {i+1}: {filenames}")

            # CRITICAL TEST: Same stages should NOT be grouped together
            # This test would have FAILED with the old broken logic!
            assert (
                len(groups) == 0
            ), f"Same stages should not be grouped, but got {len(groups)} groups"

        print("✅ Same stage not grouped test PASSED")

    except Exception as e:
        print(f"❌ Same stage not grouped test FAILED: {e}")
        assert False


def test_stage_progression_order():
    """Test that stages must be in correct progression order"""
    print("\n🧪 Testing Stage Progression Order...")

    try:
        from scripts.utils.companion_file_utils import (
            find_consecutive_stage_groups,
            sort_image_files_by_timestamp_and_stage,
        )

        # Create test files with correct progression
        test_files = [
            "20250705_214626_stage1_generated.png",
            "20250705_214953_stage1.5_face_swapped.png",
            "20250705_215137_stage2_upscaled.png",
            "20250705_215319_stage3_enhanced.png",
        ]

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_paths = []

            for filename in test_files:
                file_path = temp_path / filename
                file_path.write_text("dummy content")
                file_paths.append(file_path)

            # Pre-sort by timestamp and stage
            sorted_paths = sort_image_files_by_timestamp_and_stage(file_paths)
            groups = find_consecutive_stage_groups(sorted_paths)

            print(f"  Found {len(groups)} groups")
            for i, group in enumerate(groups):
                filenames = [f.name for f in group]
                print(f"  Group {i+1}: {filenames}")

            # Should have exactly 1 group with all 4 files
            assert len(groups) == 1, f"Expected 1 group, got {len(groups)}"
            assert (
                len(groups[0]) == 4
            ), f"Expected 4 files in group, got {len(groups[0])}"

            # Verify stage progression
            from scripts.utils.companion_file_utils import (
                detect_stage,
                get_stage_number,
            )

            stages = [get_stage_number(detect_stage(f.name)) for f in groups[0]]
            print(f"  Stages: {stages}")

            # Should be strictly increasing: 1.0 → 1.5 → 2.0 → 3.0
            expected_stages = [1.0, 1.5, 2.0, 3.0]
            assert (
                stages == expected_stages
            ), f"Expected {expected_stages}, got {stages}"

        print("✅ Stage progression order test PASSED")

    except Exception as e:
        print(f"❌ Stage progression order test FAILED: {e}")
        assert False


def test_backwards_stage_breaks_group():
    """Test that going backwards in stages breaks the group"""
    print("\n🧪 Testing Backwards Stage Breaks Group...")

    try:
        from scripts.utils.companion_file_utils import (
            detect_stage,
            find_consecutive_stage_groups,
            get_stage_number,
            sort_image_files_by_timestamp_and_stage,
        )

        # Create test files with backwards progression
        test_files = [
            "20250705_214626_stage2_upscaled.png",
            "20250705_214953_stage3_enhanced.png",
            "20250705_215137_stage1_generated.png",  # Goes backwards!
            "20250705_215319_stage2_upscaled.png",
        ]

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_paths = []

            for filename in test_files:
                file_path = temp_path / filename
                file_path.write_text("dummy content")
                file_paths.append(file_path)

            # Pre-sort by timestamp and stage
            sorted_paths = sort_image_files_by_timestamp_and_stage(file_paths)
            groups = find_consecutive_stage_groups(sorted_paths)

            print(f"  Found {len(groups)} groups")
            for i, group in enumerate(groups):
                filenames = [f.name for f in group]
                print(f"  Group {i+1}: {filenames}")

            # Should have 2 groups: [stage2, stage3] and [stage1, stage2]
            assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"
            assert (
                len(groups[0]) == 2
            ), f"First group should have 2 files, got {len(groups[0])}"
            assert (
                len(groups[1]) == 2
            ), f"Second group should have 2 files, got {len(groups[1])}"

            # Verify strict increase within each group
            for group in groups:
                stages = [get_stage_number(detect_stage(f.name)) for f in group]
                for i in range(len(stages) - 1):
                    assert (
                        stages[i + 1] > stages[i]
                    ), f"Stages should increase strictly within a group: {stages}"

        print("✅ Backwards stage breaks group test PASSED")

    except Exception as e:
        print(f"❌ Backwards stage breaks group test FAILED: {e}")
        assert False


def test_nearest_up_behavior_with_lookahead():
    """Test that 1,3,2 within lookahead yields 1→2→3 (nearest-up selection)."""
    print("\n🧪 Testing Nearest-Up Behavior with Lookahead...")

    try:
        from scripts.utils.companion_file_utils import (
            detect_stage,
            find_consecutive_stage_groups,
            get_stage_number,
            sort_image_files_by_timestamp_and_stage,
        )

        test_files = [
            "20250705_000000_stage1_generated.png",
            "20250705_000010_stage3_enhanced.png",  # earlier stage3 should be skipped in favor of 2
            "20250705_000020_stage2_upscaled.png",
            "20250705_000030_stage3_enhanced.png",  # later stage3 to complete [1,2,3]
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_paths = []
            for filename in test_files:
                p = temp_path / filename
                p.write_text("dummy")
                file_paths.append(p)

            sorted_paths = sort_image_files_by_timestamp_and_stage(file_paths)
            groups = find_consecutive_stage_groups(sorted_paths)

            assert len(groups) == 1, f"Expected 1 group, got {len(groups)}"
            stages = [get_stage_number(detect_stage(f.name)) for f in groups[0]]
            assert stages == [
                1.0,
                2.0,
                3.0,
            ], f"Expected nearest-up [1,2,3], got {stages}"

        print("✅ Nearest-up behavior test PASSED")
    except Exception as e:
        print(f"❌ Nearest-up behavior test FAILED: {e}")
        assert False


def test_time_gap_breaks_group():
    """Test that a large time gap enforces a group boundary when time_gap_minutes is set."""
    print("\n🧪 Testing Time Gap Breaks Group...")

    try:
        from scripts.utils.companion_file_utils import (
            find_consecutive_stage_groups,
            sort_image_files_by_timestamp_and_stage,
        )

        test_files = [
            "20250705_000000_stage1_generated.png",
            "20250705_000010_stage2_upscaled.png",
            "20250705_010000_stage3_enhanced.png",  # 1 hour later
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_paths = []
            for filename in test_files:
                p = temp_path / filename
                p.write_text("dummy")
                file_paths.append(p)

            sorted_paths = sort_image_files_by_timestamp_and_stage(file_paths)
            groups = find_consecutive_stage_groups(sorted_paths, time_gap_minutes=5)

            # Expect two groups: [1,2] then [3] (second one ignored since min_group_size=2 by default)
            # But since min_group_size=2, only the first group qualifies.
            assert (
                len(groups) == 1
            ), f"Expected 1 qualifying group due to time gap, got {len(groups)}"
            assert (
                len(groups[0]) == 2
            ), f"Expected first group size 2, got {len(groups[0])}"

        print("✅ Time gap breaks group test PASSED")
    except Exception as e:
        print(f"❌ Time gap breaks group test FAILED: {e}")
        assert False


def test_duplicate_stage_ends_run():
    """Test that encountering the same stage ends the current run and starts a new one."""
    print("\n🧪 Testing Duplicate Stage Ends Run...")

    try:
        from scripts.utils.companion_file_utils import (
            detect_stage,
            find_consecutive_stage_groups,
            get_stage_number,
            sort_image_files_by_timestamp_and_stage,
        )

        test_files = [
            "20250705_000000_stage1_generated.png",
            "20250705_000010_stage1.5_face_swapped.png",
            "20250705_000020_stage2_upscaled.png",
            "20250705_000030_stage2_upscaled.png",  # duplicate stage → ends run
            "20250705_000040_stage3_enhanced.png",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fps = []
            for fn in test_files:
                p = temp_path / fn
                p.write_text("dummy")
                fps.append(p)

            sorted_paths = sort_image_files_by_timestamp_and_stage(fps)
            groups = find_consecutive_stage_groups(sorted_paths)

            assert (
                len(groups) == 2
            ), f"Expected 2 groups split on duplicate stage, got {len(groups)}"
            stages0 = [get_stage_number(detect_stage(f.name)) for f in groups[0]]
            stages1 = [get_stage_number(detect_stage(f.name)) for f in groups[1]]
            assert stages0 == [1.0, 1.5, 2.0], f"First group stages mismatch: {stages0}"
            assert stages1 == [2.0, 3.0], f"Second group stages mismatch: {stages1}"

        print("✅ Duplicate stage ends run test PASSED")
    except Exception as e:
        print(f"❌ Duplicate stage ends run test FAILED: {e}")
        assert False


def test_mojo1_expected_pair_grouped():
    """Integration check: the known mojo1 pair should be grouped together if data exists."""
    print("\n🧪 Testing Mojo1 Expected Pair Grouped (integration)...")

    try:
        from scripts.utils.companion_file_utils import (
            find_all_image_files,
            find_consecutive_stage_groups,
            sort_image_files_by_timestamp_and_stage,
        )

        project_root = Path(__file__).parent.parent.parent
        mojo_dir = project_root / "mojo1"
        if not mojo_dir.exists():
            print("  ⚠️ mojo1 directory not found; skipping integration test.")
            return

        expected = {
            "20250705_231951_stage2_upscaled",
            "20250705_232142_stage3_enhanced",
        }

        files = sort_image_files_by_timestamp_and_stage(find_all_image_files(mojo_dir))
        if not files:
            print("  ⚠️ No files in mojo1; skipping integration test.")
            return

        groups = find_consecutive_stage_groups(files)
        name_groups = [[p.stem for p in g] for g in groups]
        in_any = any(expected.issubset(set(names)) for names in name_groups)
        assert in_any, "Expected mojo1 pair not found in any group"

        print("✅ Mojo1 expected pair grouped test PASSED")
    except Exception as e:
        print(f"❌ Mojo1 expected pair grouped test FAILED: {e}")
        assert False

        print("✅ Backwards stage breaks group test PASSED")

    except Exception as e:
        print(f"❌ Backwards stage breaks group test FAILED: {e}")
        assert False


def run_all_tests():
    """Run all triplet detection logic tests"""
    print("🧪 Triplet Detection Logic Test Suite")
    print("=" * 60)

    tests = [
        ("All Consecutive Combinations", test_all_consecutive_combinations),
        ("Same Stage NOT Grouped", test_same_stage_not_grouped),
        ("Stage Progression Order", test_stage_progression_order),
        ("Backwards Stage Breaks Group", test_backwards_stage_breaks_group),
        ("Nearest-Up Behavior", test_nearest_up_behavior_with_lookahead),
        ("Time Gap Breaks Group", test_time_gap_breaks_group),
        ("Duplicate Stage Ends Run", test_duplicate_stage_ends_run),
        ("Mojo1 Expected Pair Grouped", test_mojo1_expected_pair_grouped),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")

    if passed == total:
        print("\n🎉 ALL TRIPLET DETECTION LOGIC TESTS PASSED")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        assert False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
