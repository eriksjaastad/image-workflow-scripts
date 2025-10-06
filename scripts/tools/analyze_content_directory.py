#!/usr/bin/env python3
"""
Content Directory Analysis Tool
==============================
Analyzes content directories to understand file structures, patterns, and variations.
Helps identify what types of files and data we're working with before processing.

USAGE:
------
  python scripts/tools/analyze_content_directory.py mojo1/
  python scripts/tools/analyze_content_directory.py /path/to/content --detailed
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import re

# Add the scripts/ directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.companion_file_utils import extract_timestamp_from_filename


def analyze_file_extensions(directory: Path) -> Dict[str, int]:
    """Analyze all file extensions in the directory."""
    extensions = Counter()
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            extensions[file_path.suffix.lower()] += 1
    
    return dict(extensions)


def analyze_filename_patterns(directory: Path) -> Dict[str, int]:
    """Analyze filename patterns to understand naming conventions."""
    patterns = Counter()
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            filename = file_path.stem
            
            # Check for timestamp patterns
            if re.match(r'^\d{8}_\d{6}', filename):
                patterns['timestamp_prefix'] += 1
            
            # Check for stage patterns
            if 'stage1_generated' in filename:
                patterns['stage1_generated'] += 1
            elif 'stage1.5_face_swapped' in filename:
                patterns['stage1.5_face_swapped'] += 1
            elif 'stage2_upscaled' in filename:
                patterns['stage2_upscaled'] += 1
            
            # Check for other patterns
            if 'face_swapped' in filename:
                patterns['face_swapped'] += 1
            if 'upscaled' in filename:
                patterns['upscaled'] += 1
    
    return dict(patterns)


def analyze_companion_files(directory: Path) -> Dict[str, int]:
    """Analyze companion file relationships."""
    companion_stats = {
        'images_with_yaml': 0,
        'images_with_caption': 0,
        'images_with_both': 0,
        'images_with_neither': 0,
        'orphaned_yaml': 0,
        'orphaned_caption': 0,
        'total_images': 0
    }
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            companion_stats['total_images'] += 1
            
            yaml_path = file_path.parent / f"{file_path.stem}.yaml"
            caption_path = file_path.parent / f"{file_path.stem}.caption"
            
            has_yaml = yaml_path.exists()
            has_caption = caption_path.exists()
            
            if has_yaml and has_caption:
                companion_stats['images_with_both'] += 1
            elif has_yaml:
                companion_stats['images_with_yaml'] += 1
            elif has_caption:
                companion_stats['images_with_caption'] += 1
            else:
                companion_stats['images_with_neither'] += 1
    
    # Count orphaned files
    for file_path in directory.rglob('*.yaml'):
        image_path = file_path.parent / f"{file_path.stem}.png"
        if not image_path.exists():
            companion_stats['orphaned_yaml'] += 1
    
    for file_path in directory.rglob('*.caption'):
        image_path = file_path.parent / f"{file_path.stem}.png"
        if not image_path.exists():
            companion_stats['orphaned_caption'] += 1
    
    return companion_stats


def analyze_triplet_patterns(directory: Path) -> Dict[str, int]:
    """Analyze triplet patterns to understand image sequences."""
    triplet_stats = {
        'complete_triplets': 0,
        'incomplete_triplets': 0,
        'singles': 0,
        'pairs': 0,
        'quads': 0,
        'other_sequences': 0
    }
    
    # Group files by base timestamp
    timestamp_groups = defaultdict(list)
    
    for file_path in directory.rglob('*.png'):
        if file_path.is_file():
            filename = file_path.stem
            # Use centralized timestamp extraction
            timestamp = extract_timestamp_from_filename(filename)
            if timestamp:
                timestamp_groups[timestamp].append(file_path)
    
    for timestamp, files in timestamp_groups.items():
        if len(files) == 3:
            # Check if it's a complete triplet
            stages = set()
            for file_path in files:
                if 'stage1_generated' in file_path.name:
                    stages.add('stage1')
                elif 'stage1.5_face_swapped' in file_path.name:
                    stages.add('stage1.5')
                elif 'stage2_upscaled' in file_path.name:
                    stages.add('stage2')
            
            if len(stages) == 3:
                triplet_stats['complete_triplets'] += 1
            else:
                triplet_stats['incomplete_triplets'] += 1
        elif len(files) == 1:
            triplet_stats['singles'] += 1
        elif len(files) == 2:
            triplet_stats['pairs'] += 1
        elif len(files) == 4:
            triplet_stats['quads'] += 1
        else:
            triplet_stats['other_sequences'] += 1
    
    return triplet_stats


def analyze_prompt_files(directory: Path) -> Dict[str, int]:
    """Analyze prompt files if they exist."""
    prompt_stats = {
        'total_prompt_files': 0,
        'total_prompts': 0,
        'prompt_files': []
    }
    
    prompts_dir = directory / 'prompts'
    if prompts_dir.exists():
        for prompt_file in prompts_dir.glob('*.txt'):
            prompt_stats['total_prompt_files'] += 1
            prompt_stats['prompt_files'].append(prompt_file.name)
            
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    prompt_stats['total_prompts'] += len([line for line in lines if line.strip()])
            except Exception as e:
                print(f"Warning: Could not read {prompt_file}: {e}")
    
    return prompt_stats


def analyze_directory_structure(directory: Path) -> Dict[str, int]:
    """Analyze directory structure."""
    structure_stats = {
        'total_directories': 0,
        'total_files': 0,
        'max_depth': 0,
        'directory_names': []
    }
    
    for item in directory.rglob('*'):
        if item.is_dir():
            structure_stats['total_directories'] += 1
            structure_stats['directory_names'].append(item.name)
            
            # Calculate depth
            depth = len(item.relative_to(directory).parts)
            structure_stats['max_depth'] = max(structure_stats['max_depth'], depth)
        elif item.is_file():
            structure_stats['total_files'] += 1
    
    return structure_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze content directory structure and patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/tools/analyze_content_directory.py mojo1/
    python scripts/tools/analyze_content_directory.py /path/to/content --detailed
        """
    )
    
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed analysis including sample files")
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    print(f"🔍 Analyzing content directory: {directory}")
    print("=" * 60)
    
    # Run all analyses
    print("\n📊 FILE EXTENSIONS:")
    extensions = analyze_file_extensions(directory)
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext or '(no extension)'}: {count:,} files")
    
    print("\n📝 FILENAME PATTERNS:")
    patterns = analyze_filename_patterns(directory)
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count:,} files")
    
    print("\n🔗 COMPANION FILE ANALYSIS:")
    companions = analyze_companion_files(directory)
    print(f"  Total images: {companions['total_images']:,}")
    print(f"  Images with YAML only: {companions['images_with_yaml']:,}")
    print(f"  Images with caption only: {companions['images_with_caption']:,}")
    print(f"  Images with both: {companions['images_with_both']:,}")
    print(f"  Images with neither: {companions['images_with_neither']:,}")
    print(f"  Orphaned YAML files: {companions['orphaned_yaml']:,}")
    print(f"  Orphaned caption files: {companions['orphaned_caption']:,}")
    
    print("\n🎯 TRIPLET PATTERN ANALYSIS:")
    triplets = analyze_triplet_patterns(directory)
    print(f"  Complete triplets (3 files): {triplets['complete_triplets']:,}")
    print(f"  Incomplete triplets: {triplets['incomplete_triplets']:,}")
    print(f"  Singles: {triplets['singles']:,}")
    print(f"  Pairs: {triplets['pairs']:,}")
    print(f"  Quads (4 files): {triplets['quads']:,}")
    print(f"  Other sequences: {triplets['other_sequences']:,}")
    
    print("\n📁 DIRECTORY STRUCTURE:")
    structure = analyze_directory_structure(directory)
    print(f"  Total directories: {structure['total_directories']:,}")
    print(f"  Total files: {structure['total_files']:,}")
    print(f"  Maximum depth: {structure['max_depth']}")
    
    print("\n📄 PROMPT FILES:")
    prompts = analyze_prompt_files(directory)
    print(f"  Total prompt files: {prompts['total_prompt_files']:,}")
    print(f"  Total prompts: {prompts['total_prompts']:,}")
    if prompts['prompt_files']:
        print(f"  Prompt files: {', '.join(prompts['prompt_files'][:5])}")
        if len(prompts['prompt_files']) > 5:
            print(f"    ... and {len(prompts['prompt_files']) - 5} more")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if triplets['quads'] > 0:
        print(f"  ⚠️  Found {triplets['quads']} quad sequences - consider updating tools to handle 4-image sequences")
    if companions['orphaned_yaml'] > 0 or companions['orphaned_caption'] > 0:
        print(f"  ⚠️  Found orphaned metadata files - run cleanup script")
    if triplets['incomplete_triplets'] > 0:
        print(f"  ⚠️  Found {triplets['incomplete_triplets']} incomplete triplets - may need manual review")
    if prompts['total_prompt_files'] > 0:
        print(f"  📝 Found {prompts['total_prompt_files']} prompt files - consider analyzing for workflow insights")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
