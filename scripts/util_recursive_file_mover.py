#!/usr/bin/env python3
"""
Utility: Recursive File Mover
==============================
Recursively move all image/YAML pairs from source to destination directory.
Maintains file pair integrity with comprehensive logging and conflict handling.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Move files between directories:
  python scripts/util_recursive_file_mover.py {content} XXX_CONTENT
  python scripts/util_recursive_file_mover.py ~/Downloads/output 00_white
  python scripts/util_recursive_file_mover.py source_dir dest_dir --dry-run

FEATURES:
---------
• Recursively scans source directory for image/YAML pairs
• Moves files while maintaining pairs integrity
• Validates that both source and destination directories exist
• Provides detailed progress reporting with tqdm
• Uses FileTracker for comprehensive operation logging
• Handles name conflicts with automatic renaming
• Dry-run mode for safe preview of operations
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.file_tracker import FileTracker

def find_image_yaml_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all image/YAML pairs in source directory recursively."""
    pairs = []
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg'}
    
    # Find all image files recursively
    for image_file in source_dir.rglob('*'):
        if image_file.is_file() and image_file.suffix.lower() in image_extensions:
            # Look for corresponding YAML file
            yaml_file = image_file.with_suffix('.yaml')
            if yaml_file.exists():
                pairs.append((image_file, yaml_file))
            else:
                print(f"⚠️  WARNING: Image without YAML found: {image_file}")
    
    return pairs

def get_unique_filename(dest_file: Path) -> Path:
    """Generate unique filename if destination already exists."""
    if not dest_file.exists():
        return dest_file
    
    counter = 1
    stem = dest_file.stem
    suffix = dest_file.suffix
    parent = dest_file.parent
    
    while True:
        new_name = f"{stem}_{counter:03d}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def move_files(source_dir: Path, dest_dir: Path, dry_run: bool = False) -> None:
    """Move all image/YAML pairs from source to destination."""
    
    # Initialize FileTracker
    tracker = FileTracker("recursive_file_mover")
    
    # Destination directory should already exist (validated in main)
    # No auto-creation for safety
    
    # Find all image/YAML pairs
    print(f"🔍 Scanning {source_dir} for image/YAML pairs...")
    pairs = find_image_yaml_pairs(source_dir)
    
    if not pairs:
        print(f"❌ No image/YAML pairs found in {source_dir}")
        return
    
    print(f"📊 Found {len(pairs)} image/YAML pairs")
    print()
    
    moved_count = 0
    renamed_count = 0
    
    for i, (image_file, yaml_file) in enumerate(pairs, 1):
        # Determine destination paths
        dest_image = dest_dir / image_file.name
        dest_yaml = dest_dir / yaml_file.name
        
        # Handle name conflicts
        original_image_name = dest_image.name
        original_yaml_name = dest_yaml.name
        
        dest_image = get_unique_filename(dest_image)
        dest_yaml = get_unique_filename(dest_yaml)
        
        if dest_image.name != original_image_name or dest_yaml.name != original_yaml_name:
            renamed_count += 1
        
        print(f"[{i:3d}/{len(pairs)}] Moving: {image_file.name}")
        if dest_image.name != original_image_name:
            print(f"           → Renamed to: {dest_image.name}")
        
        if not dry_run:
            try:
                # Move files
                image_file.rename(dest_image)
                yaml_file.rename(dest_yaml)
                
                # Log the operations
                tracker.log_operation("move", str(image_file), str(dest_image))
                tracker.log_operation("move", str(yaml_file), str(dest_yaml))
                
                moved_count += 1
                
            except Exception as e:
                print(f"❌ ERROR moving {image_file.name}: {e}")
                tracker.log_operation("error", str(image_file), "", notes=str(e))
    
    print()
    print("📊 OPERATION COMPLETE:")
    print(f"   • Pairs processed: {len(pairs)}")
    print(f"   • Successfully moved: {moved_count}")
    print(f"   • Files renamed due to conflicts: {renamed_count}")
    if dry_run:
        print("   • DRY RUN - No files were actually moved")
    else:
        print(f"   • Files moved to: {dest_dir}")
        print(f"   • Operations logged by FileTracker")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively move all image/YAML pairs from source to destination directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/util_recursive_file_mover.py face_groups 00_white
    python scripts/util_recursive_file_mover.py ~/Downloads/output 00_white  
    python scripts/util_recursive_file_mover.py face_groups/person_0001 character_group_1
        """
    )
    
    parser.add_argument("source", help="Source directory to move files from")
    parser.add_argument("destination", help="Destination directory to move files to")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be moved without actually moving files")
    
    args = parser.parse_args()
    
    # Resolve paths
    source_dir = Path(args.source).expanduser().resolve()
    dest_dir = Path(args.destination).expanduser().resolve()
    
    # Validate source directory
    if not source_dir.exists():
        print(f"❌ ERROR: Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    if not source_dir.is_dir():
        print(f"❌ ERROR: Source is not a directory: {source_dir}")
        sys.exit(1)
    
    # Check if source and destination are the same
    if source_dir == dest_dir:
        print(f"❌ ERROR: Source and destination are the same directory")
        sys.exit(1)
    
    # Validate destination directory
    if not dest_dir.exists():
        print(f"❌ ERROR: Destination directory does not exist: {dest_dir}")
        sys.exit(1)
    
    if not dest_dir.is_dir():
        print(f"❌ ERROR: Destination is not a directory: {dest_dir}")
        sys.exit(1)
    
    # Check if destination is inside source (would cause issues)
    try:
        dest_dir.relative_to(source_dir)
        print(f"❌ ERROR: Destination directory is inside source directory")
        sys.exit(1)
    except ValueError:
        pass  # Good - destination is not inside source
    
    print("🚀 RECURSIVE FILE MOVER")
    print("=" * 50)
    print(f"📂 Source:      {source_dir}")
    print(f"📁 Destination: {dest_dir}")
    if args.dry_run:
        print("🧪 Mode:        DRY RUN (no files will be moved)")
    print()
    
    # Confirm operation unless it's a dry run
    if not args.dry_run:
        response = input("Continue with file move operation? [y/N]: ").strip().lower()
        if response != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        print()
    
    move_files(source_dir, dest_dir, args.dry_run)

if __name__ == "__main__":
    main()
