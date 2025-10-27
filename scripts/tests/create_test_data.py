#!/usr/bin/env python3
"""
Create Test Data for Image Selector Testing

This script creates realistic test data based on the XXX_CONTENT manifest,
allowing us to test all workflow scripts without using real image files.

Usage:
    python scripts/create_test_data.py --triplets 120 --output tests/test_images/
    python scripts/create_test_data.py --from-manifest tests/test_manifest.json --output tests/test_images/
"""

import argparse
import shutil
from pathlib import Path


def create_triplet_test_data(output_dir, num_triplets=120):
    """Create test data with proper triplet structure"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔧 Creating {num_triplets} test triplets in {output_path}")
    
    files_created = []
    
    for i in range(num_triplets):
        # Create realistic timestamp
        base_time = f"20250803_{i:06d}"
        
        # Create triplet files: stage1 -> stage1.5 -> stage2
        triplet_files = [
            f"{base_time}_stage1_generated.png",
            f"{base_time}_stage1_generated.yaml",
            f"{base_time}_stage1.5_face_swapped.png", 
            f"{base_time}_stage1.5_face_swapped.yaml",
            f"{base_time}_stage2_upscaled.png",
            f"{base_time}_stage2_upscaled.yaml"
        ]
        
        for filename in triplet_files:
            filepath = output_path / filename
            
            if filename.endswith('.png'):
                # Create dummy PNG content
                content = f"DUMMY_PNG_CONTENT_FOR_{filename}"
            else:
                # Create dummy YAML content
                content = f"# YAML metadata for {filename.replace('.yaml', '.png')}\nfilename: {filename.replace('.yaml', '.png')}\nstage: {extract_stage_from_filename(filename)}\n"
            
            filepath.write_text(content)
            files_created.append(filename)
    
    print(f"✅ Created {len(files_created)} test files")
    return files_created

def create_mixed_test_data(output_dir, triplets=100, pairs=10, singletons=5, with_subdirs=False):
    """Create mixed test data with triplets, pairs, and singletons"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subdirs_msg = " (with subdirectories)" if with_subdirs else ""
    print(f"🔧 Creating mixed test data: {triplets} triplets, {pairs} pairs, {singletons} singletons{subdirs_msg}")
    
    files_created = []
    file_counter = 0
    
    # Create triplets
    for i in range(triplets):
        base_time = f"20250803_{file_counter:06d}"
        file_counter += 10  # Space out timestamps
        
        # Determine if this triplet goes in a subdirectory
        target_dir = output_path
        if with_subdirs and i % 5 == 0:  # Every 5th triplet goes in a subdirectory
            subdir_name = f"subdir_{i // 5:02d}"
            target_dir = output_path / subdir_name
            target_dir.mkdir(exist_ok=True)
        
        triplet_files = [
            f"{base_time}_stage1_generated.png",
            f"{base_time}_stage1_generated.yaml",
            f"{base_time + '01'}_stage1.5_face_swapped.png", 
            f"{base_time + '01'}_stage1.5_face_swapped.yaml",
            f"{base_time + '02'}_stage2_upscaled.png",
            f"{base_time + '02'}_stage2_upscaled.yaml"
        ]
        
        for filename in triplet_files:
            create_test_file(target_dir / filename, filename)
            files_created.append(str(target_dir / filename))
    
    # Create pairs (missing one stage)
    for i in range(pairs):
        base_time = f"20250803_{file_counter:06d}"
        file_counter += 10
        
        # Randomly choose which stage to skip
        import random
        skip_stage = random.choice(['stage1.5', 'stage2'])
        
        if skip_stage == 'stage1.5':
            pair_files = [
                f"{base_time}_stage1_generated.png",
                f"{base_time}_stage1_generated.yaml",
                f"{base_time + '02'}_stage2_upscaled.png",
                f"{base_time + '02'}_stage2_upscaled.yaml"
            ]
        else:  # skip stage2
            pair_files = [
                f"{base_time}_stage1_generated.png",
                f"{base_time}_stage1_generated.yaml",
                f"{base_time + '01'}_stage1.5_face_swapped.png", 
                f"{base_time + '01'}_stage1.5_face_swapped.yaml"
            ]
        
        for filename in pair_files:
            create_test_file(output_path / filename, filename)
            files_created.append(filename)
    
    # Create singletons
    for i in range(singletons):
        base_time = f"20250803_{file_counter:06d}"
        file_counter += 10
        
        singleton_files = [
            f"{base_time}_stage1_generated.png",
            f"{base_time}_stage1_generated.yaml"
        ]
        
        for filename in singleton_files:
            create_test_file(output_path / filename, filename)
            files_created.append(filename)
    
    print(f"✅ Created {len(files_created)} test files")
    return files_created

def create_test_file(filepath, filename):
    """Create a single test file with appropriate content"""
    if filename.endswith('.png'):
        # Create dummy PNG content
        content = f"DUMMY_PNG_CONTENT_FOR_{filename}"
    else:
        # Create dummy YAML content
        stage = extract_stage_from_filename(filename)
        content = f"# YAML metadata for {filename.replace('.yaml', '.png')}\nfilename: {filename.replace('.yaml', '.png')}\nstage: {stage}\ntimestamp: generated\n"
    
    filepath.write_text(content)

def extract_stage_from_filename(filename):
    """Extract stage from filename"""
    if "stage1.5" in filename:
        return "stage1.5"
    elif "stage1" in filename:
        return "stage1"
    elif "stage2" in filename:
        return "stage2"
    elif "stage3" in filename:
        return "stage3"
    return "unknown"

def create_performance_test_data(output_dir, size="large", with_subdirs=False):
    """Create test data for performance testing"""
    output_path = Path(output_dir)
    
    if size == "small":
        triplets, pairs, singletons = 10, 2, 1
    elif size == "medium": 
        triplets, pairs, singletons = 50, 5, 2
    elif size == "large":
        triplets, pairs, singletons = 121, 10, 5  # Based on your XXX_CONTENT size
    elif size == "huge":
        triplets, pairs, singletons = 500, 50, 10  # Stress test
    else:
        raise ValueError(f"Unknown size: {size}")
    
    return create_mixed_test_data(output_path, triplets, pairs, singletons, with_subdirs)

def create_subdirectory_test_data(output_dir):
    """Create test data with complex subdirectory structure to test edge cases"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Creating subdirectory test data with complex structure")
    
    files_created = []
    
    # Root level triplets
    for i in range(5):
        base_time = f"20250803_{i:06d}"
        triplet_files = [
            f"{base_time}_stage1_generated.png",
            f"{base_time}_stage1_generated.yaml",
            f"{base_time}01_stage1.5_face_swapped.png", 
            f"{base_time}01_stage1.5_face_swapped.yaml",
            f"{base_time}02_stage2_upscaled.png",
            f"{base_time}02_stage2_upscaled.yaml"
        ]
        
        for filename in triplet_files:
            create_test_file(output_path / filename, filename)
            files_created.append(filename)
    
    # Subdirectory scenarios
    subdirs = ["folder_a", "nested/deep/path", "output_batch_01", "Downloads/extracted"]
    
    for subdir_idx, subdir in enumerate(subdirs):
        subdir_path = output_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        
        # Each subdirectory gets 2-3 triplets
        for i in range(2):
            base_time = f"20250803_{(subdir_idx + 1) * 10 + i:06d}"
            triplet_files = [
                f"{base_time}_stage1_generated.png",
                f"{base_time}_stage1_generated.yaml",
                f"{base_time}01_stage1.5_face_swapped.png", 
                f"{base_time}01_stage1.5_face_swapped.yaml",
                f"{base_time}02_stage2_upscaled.png",
                f"{base_time}02_stage2_upscaled.yaml"
            ]
            
            for filename in triplet_files:
                create_test_file(subdir_path / filename, filename)
                files_created.append(str(subdir_path / filename))
    
    # Mixed files in same subdirectory (some triplets, some pairs)
    mixed_subdir = output_path / "mixed_content"
    mixed_subdir.mkdir(exist_ok=True)
    
    # Add a triplet
    base_time = "20250803_500000"
    triplet_files = [
        f"{base_time}_stage1_generated.png",
        f"{base_time}_stage1_generated.yaml",
        f"{base_time}01_stage1.5_face_swapped.png", 
        f"{base_time}01_stage1.5_face_swapped.yaml",
        f"{base_time}02_stage2_upscaled.png",
        f"{base_time}02_stage2_upscaled.yaml"
    ]
    
    for filename in triplet_files:
        create_test_file(mixed_subdir / filename, filename)
        files_created.append(str(mixed_subdir / filename))
    
    # Add a pair (missing stage2)
    base_time = "20250803_500010"
    pair_files = [
        f"{base_time}_stage1_generated.png",
        f"{base_time}_stage1_generated.yaml",
        f"{base_time}01_stage1.5_face_swapped.png", 
        f"{base_time}01_stage1.5_face_swapped.yaml"
    ]
    
    for filename in pair_files:
        create_test_file(mixed_subdir / filename, filename)
        files_created.append(str(mixed_subdir / filename))
    
    # Create non-standard files subdirectory (should be ignored gracefully)
    random_subdir = output_path / "random_files"
    random_subdir.mkdir(exist_ok=True)
    
    # Add random PNG/YAML files that don't follow stage conventions
    random_files = [
        "IMG_1234.png",
        "IMG_1234.yaml",
        "photo_sunset.png", 
        "photo_sunset.yaml",
        "screenshot_2023.png",
        "config_backup.yaml",
        "thumbnail_abc123.png",
        "metadata_xyz.yaml",
        "random_image_001.png",
        "settings.yaml",
        "output_final.png",
        "data_export.yaml",
        "temp_file_999.png",
        "log_data.yaml"
    ]
    
    for filename in random_files:
        if filename.endswith('.png'):
            content = f"RANDOM_PNG_CONTENT_{filename}"
        else:
            content = f"# Random YAML file\nfilename: {filename}\ntype: random\nshould_be_ignored: true\n"
        
        filepath = random_subdir / filename
        filepath.write_text(content)
        files_created.append(str(filepath))
    
    # Add some files with partial stage names but wrong format (should be ignored)
    problematic_files = [
        "stage1_but_wrong_format.png",
        "stage1_but_wrong_format.yaml", 
        "not_stage2_at_all.png",
        "not_stage2_at_all.yaml",
        "stage1.5_missing_timestamp.png",
        "stage1.5_missing_timestamp.yaml",
        "20230101_stage99_invalid.png",  # Invalid stage number
        "20230101_stage99_invalid.yaml"
    ]
    
    for filename in problematic_files:
        if filename.endswith('.png'):
            content = f"PROBLEMATIC_PNG_{filename}"
        else:
            content = f"# Problematic YAML\nfilename: {filename}\ntype: problematic\n"
        
        filepath = random_subdir / filename
        filepath.write_text(content)
        files_created.append(str(filepath))
    
    print(f"✅ Created {len(files_created)} test files across multiple subdirectories")
    print(f"   📁 Including {len(random_files) + len(problematic_files)} non-standard files that should be ignored")
    return files_created

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test data for image selector testing")
    parser.add_argument("--output", "-o", default="tests/test_images/", 
                       help="Output directory for test files")
    parser.add_argument("--triplets", "-t", type=int, default=120,
                       help="Number of triplets to create")
    parser.add_argument("--pairs", "-p", type=int, default=0,
                       help="Number of pairs to create")
    parser.add_argument("--singletons", "-s", type=int, default=0,
                       help="Number of singletons to create")
    parser.add_argument("--size", choices=["small", "medium", "large", "huge"],
                       help="Preset size for performance testing")
    parser.add_argument("--with-subdirs", action="store_true",
                       help="Include subdirectories in test data")
    parser.add_argument("--subdirectory-test", action="store_true",
                       help="Create complex subdirectory test scenario")
    parser.add_argument("--from-manifest", 
                       help="Create test data based on existing manifest")
    
    args = parser.parse_args()
    
    # Clean output directory
    output_path = Path(args.output)
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"🧹 Cleaned existing directory: {output_path}")
    
    try:
        if args.from_manifest:
            # TODO: Implement manifest-based generation
            print("❌ Manifest-based generation not yet implemented")
        elif args.subdirectory_test:
            files = create_subdirectory_test_data(args.output)
        elif args.size:
            files = create_performance_test_data(args.output, args.size, args.with_subdirs)
        elif args.pairs > 0 or args.singletons > 0:
            files = create_mixed_test_data(args.output, args.triplets, args.pairs, args.singletons, args.with_subdirs)
        else:
            files = create_triplet_test_data(args.output, args.triplets)
        
        print("\n🎉 Test data created successfully!")
        print(f"📁 Location: {Path(args.output).resolve()}")
        print(f"📊 Files: {len(files)}")
        
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
        exit(1)
