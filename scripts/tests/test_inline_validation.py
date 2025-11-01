#!/usr/bin/env python3
"""
Test Inline Training Data Validation
=====================================
Tests that the inline validation in companion_file_utils catches bad data.

Run: python scripts/tests/test_inline_validation.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.companion_file_utils import (
    log_select_crop_entry,
    log_selection_only_entry,
)


def test_crop_validation_catches_zero_dimensions():
    """Test that 0x0 dimensions are caught immediately."""
    print("\n🧪 Test 1: Zero dimensions (THE BUG WE CAUGHT)")

    try:
        log_select_crop_entry(
            session_id="test_session",
            set_id="test_set",
            directory="/fake/dir",
            image_paths=["/fake/image1.png", "/fake/image2.png"],
            image_stages=["stage1", "stage2"],
            image_sizes=[(0, 0), (1920, 1080)],  # First image has 0x0!
            chosen_index=0,
            crop_norm=(0.1, 0.1, 0.9, 0.9),
        )
        print("   ❌ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        if "Invalid Image Dimensions" in str(e) and "0 x 0" in str(e):
            print("   ✅ PASSED: Caught 0x0 dimensions!")
            return True
        print(f"   ❌ FAILED: Wrong error: {e}")
        return False


def test_crop_validation_catches_invalid_coords():
    """Test that invalid crop coordinates are caught."""
    print("\n🧪 Test 2: Invalid crop coordinates")

    try:
        log_select_crop_entry(
            session_id="test_session",
            set_id="test_set",
            directory="/fake/dir",
            image_paths=["/fake/image1.png"],
            image_stages=["stage1"],
            image_sizes=[(1920, 1080)],
            chosen_index=0,
            crop_norm=(0.9, 0.1, 0.1, 0.9),  # x1 > x2 (invalid!)
        )
        print("   ❌ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        if "Invalid Crop Coordinates" in str(e):
            print("   ✅ PASSED: Caught invalid crop coordinates!")
            return True
        print(f"   ❌ FAILED: Wrong error: {e}")
        return False


def test_selection_validation_catches_empty_path():
    """Test that empty chosen path is caught."""
    print("\n🧪 Test 3: Empty chosen path")

    try:
        log_selection_only_entry(
            session_id="test_session",
            set_id="test_set",
            chosen_path="",  # Empty!
            negative_paths=["/fake/loser1.png", "/fake/loser2.png"],
        )
        print("   ❌ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        if "Empty Chosen Path" in str(e):
            print("   ✅ PASSED: Caught empty chosen path!")
            return True
        print(f"   ❌ FAILED: Wrong error: {e}")
        return False


def test_selection_validation_catches_invalid_neg_paths():
    """Test that non-list negative_paths is caught."""
    print("\n🧪 Test 4: Invalid negative paths type")

    try:
        log_selection_only_entry(
            session_id="test_session",
            set_id="test_set",
            chosen_path="/fake/winner.png",
            negative_paths="not_a_list",  # Wrong type!
        )
        print("   ❌ FAILED: Should have raised ValueError!")
        return False
    except ValueError as e:
        if "Invalid Negative Paths" in str(e):
            print("   ✅ PASSED: Caught invalid negative_paths type!")
            return True
        print(f"   ❌ FAILED: Wrong error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("INLINE VALIDATION TEST SUITE")
    print("=" * 70)
    print("\nTesting that bad training data is caught IMMEDIATELY...")

    results = [
        test_crop_validation_catches_zero_dimensions(),
        test_crop_validation_catches_invalid_coords(),
        test_selection_validation_catches_empty_path(),
        test_selection_validation_catches_invalid_neg_paths(),
    ]

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\nInline validation is working correctly!")
        print("Bad training data will be caught immediately during collection.")
        return 0
    print(f"❌ SOME TESTS FAILED ({passed}/{total} passed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
