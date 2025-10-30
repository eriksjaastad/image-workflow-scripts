#!/usr/bin/env python3
"""Create test data that reproduces the sequential timestamp + same stage problem"""

from pathlib import Path


def create_problematic_test_data():
    """Create test data with sequential timestamps but same stages (the problem pattern)"""

    # Use path relative to this script's location
    script_dir = Path(__file__).parent
    test_dir = script_dir / "data" / "problematic_sequential"
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating problematic test data in {test_dir}")

    # Problem Pattern 1: Many stage1 files (SCRAMBLED ORDER - should be singletons, filtered out)
    stage1_files = [
        "20250803_173341_stage1_generated.png",  # Scrambled!
        "20250803_172814_stage1_generated.png",
        "20250803_174338_stage1_generated.png",
        "20250803_172819_stage1_generated.png",
        "20250803_173528_stage1_generated.png",
        "20250803_173215_stage1_generated.png",
        "20250803_173117_stage1_generated.png",
    ]

    # Problem Pattern 2: Many stage2 files (SCRAMBLED ORDER - should be singletons, filtered out)
    stage2_files = [
        "20250803_180300_stage2_upscaled.png",  # Scrambled!
        "20250803_180000_stage2_upscaled.png",
        "20250803_180400_stage2_upscaled.png",
        "20250803_180100_stage2_upscaled.png",
        "20250803_180200_stage2_upscaled.png",
    ]

    # Good Pattern: Proper pairs/triplets (SCRAMBLED ORDER - should create groups after sorting)
    good_groups = [
        # Triplet - OUT OF ORDER
        "20250803_190000_stage2_upscaled.png",  # Scrambled!
        "20250803_190000_stage1_generated.png",
        "20250803_190000_stage1.5_face_swapped.png",
        # Pair - OUT OF ORDER
        "20250803_200000_stage2_upscaled.png",  # Scrambled!
        "20250803_200000_stage1_generated.png",
        # Another triplet - OUT OF ORDER
        "20250803_210000_stage1.5_face_swapped.png",  # Scrambled!
        "20250803_210000_stage2_upscaled.png",
        "20250803_210000_stage1_generated.png",
    ]

    all_files = stage1_files + stage2_files + good_groups

    for filename in all_files:
        filepath = test_dir / filename
        yaml_path = test_dir / (filepath.stem + ".yaml")

        # Create dummy image file
        with open(filepath, "w") as f:
            f.write(f"dummy image: {filename}")

        # Create matching YAML file
        with open(yaml_path, "w") as f:
            f.write(f"dummy yaml: {filename}")

    print(f"Created {len(all_files)} test files:")
    print(f"  - {len(stage1_files)} sequential stage1 files (should be filtered out)")
    print(f"  - {len(stage2_files)} sequential stage2 files (should be filtered out)")
    print(f"  - {len(good_groups)} files in proper groups (should create 3 groups)")
    print(
        f"  - Expected result: 3 groups total, {len(stage1_files) + len(stage2_files)} files filtered out"
    )

    return test_dir


if __name__ == "__main__":
    create_problematic_test_data()
