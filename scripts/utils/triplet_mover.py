#!/usr/bin/env python3
"""
Utility: Triplet Mover
=======================
Find and move complete triplets (stage1/stage1.5/stage2) to destination directory.
Maintains file integrity by moving complete sets with their companion files (same-stem metadata, e.g., .yaml/.caption).

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Move complete triplets to new directory:
  python scripts/utils/triplet_mover.py source_dir destination_dir
  python scripts/utils/triplet_mover.py ~/Downloads/raw_images triplets/

FEATURES:
---------
• Detects complete triplet sequences (stage1→stage1.5→stage2)
• Moves PNG files with corresponding companion metadata
• Creates destination directory if needed
• Reports incomplete triplets for manual review
• Safe file operations with progress tracking
• Preserves timestamp and metadata information
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Import standardized companion file utilities
sys.path.append(str(Path(__file__).parent))
from companion_file_utils import detect_stage, move_file_with_all_companions


def scan_images_recursive(folder: Path) -> List[Path]:
    """Scan for PNG files recursively in folder and subdirectories."""
    files = []
    for png_file in folder.rglob("*.png"):
        files.append(png_file)
    return sorted(files)


def find_triplets(files: List[Path]) -> List[Tuple[Path, Path, Path]]:
    """Find complete triplets in the sorted file list, following the logic from triplet_culler_v9."""
    triplets = []
    i = 0
    while i < len(files) - 2:  # Need at least 3 files remaining
        # Get stages for the next 3 files
        stages = [detect_stage(files[j].name) for j in range(i, i + 3)]
        
        # Check if we have stage1, stage1.5, stage2 in sequence (like triplet_culler_v9)
        if (stages[0] == "stage1_generated" and 
            stages[1] == "stage1.5_face_swapped" and 
            stages[2] == "stage2_upscaled"):
            # Found a triplet - timestamps don't need to match exactly
            triplets.append((files[i], files[i+1], files[i+2]))
            i += 3  # Skip past this triplet
        else:
            i += 1  # Move to next file
            
    return triplets


def check_conflicts(triplet: Tuple[Path, Path, Path], dest_dir: Path) -> List[str]:
    """Check for conflicting files that would be overwritten."""
    conflicts = []
    
    for png_path in triplet:
        # Check PNG conflict
        dest_png = dest_dir / png_path.name
        if dest_png.exists():
            conflicts.append(png_path.name)
        
        # Check YAML conflict
        yaml_path = png_path.parent / f"{png_path.stem}.yaml"
        if yaml_path.exists():
            dest_yaml = dest_dir / yaml_path.name
            if dest_yaml.exists():
                conflicts.append(yaml_path.name)
    
    return conflicts


def move_triplet_with_yamls(triplet: Tuple[Path, Path, Path], dest_dir: Path) -> bool:
    """Move a triplet of PNG files and ALL their corresponding companion files."""
    # First check for conflicts
    conflicts = check_conflicts(triplet, dest_dir)
    if conflicts:
        print("❌ CONFLICT DETECTED! Files would be overwritten:")
        for conflict in conflicts:
            print(f"  🚫 {conflict}")
        print("❌ STOPPING - triplet move cancelled to prevent overwriting")
        return False
    
    moved_files = []
    try:
        for png_path in triplet:
            # Use wildcard logic to move PNG and ALL companion files
            files_moved = move_file_with_all_companions(png_path, dest_dir, dry_run=False)
            moved_files.extend(files_moved)
            print(f"✓ Moved {len(files_moved)} files for: {png_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error moving triplet: {e}")
        # Try to move back any files we already moved
        for moved_file in moved_files:
            try:
                original_dir = triplet[0].parent  # Assume all from same dir
                shutil.move(str(dest_dir / moved_file), str(original_dir / moved_file))
            except:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(description="Find and move complete triplets to destination directory")
    parser.add_argument("source_dir", type=str, help="Source directory to search for triplets")
    parser.add_argument("dest_dir", type=str, help="Destination directory for triplets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without actually moving")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    dest_dir = Path(args.dest_dir).expanduser().resolve()
    
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"[!] Source directory not found: {source_dir}")
        sys.exit(1)
    
    if not dest_dir.exists():
        print(f"[!] Destination directory not found: {dest_dir}")
        sys.exit(1)

    print(f"🔍 Scanning for triplets in: {source_dir}")
    files = scan_images_recursive(source_dir)
    print(f"📁 Found {len(files)} PNG files total")
    
    triplets = find_triplets(files)
    print(f"🎯 Found {len(triplets)} complete triplets")
    
    if not triplets:
        print("No triplets found!")
        return
    
    if args.dry_run:
        print("\n📋 DRY RUN - Would move these triplets:")
        for i, triplet in enumerate(triplets, 1):
            print(f"\nTriplet {i}:")
            for png_path in triplet:
                print(f"  📄 {png_path.name}")
                yaml_path = png_path.parent / f"{png_path.stem}.yaml"
                if yaml_path.exists():
                    print(f"  📄 {yaml_path.name}")
        print(f"\nTotal: {len(triplets)} triplets would be moved")
        return
    
    # Confirm before moving
    response = input(f"\nMove {len(triplets)} triplets to {dest_dir}? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print(f"\n📦 Moving {len(triplets)} triplets to: {dest_dir}")
    moved_count = 0
    
    for i, triplet in enumerate(triplets, 1):
        print(f"\n--- Moving triplet {i}/{len(triplets)} ---")
        if move_triplet_with_yamls(triplet, dest_dir):
            moved_count += 1
        else:
            print(f"❌ Failed to move triplet {i} - STOPPING operation to prevent conflicts")
            break
    
    if moved_count == len(triplets):
        print(f"\n✅ Successfully moved {moved_count}/{len(triplets)} triplets")
    else:
        print(f"\n⚠️ Operation stopped after {moved_count}/{len(triplets)} triplets due to conflicts")
    
    # Clean up empty directories
    print("\n🧹 Cleaning up empty directories...")
    for root, dirs, files in source_dir.walk(top_down=False):
        if root != source_dir:  # Don't remove the source directory itself
            try:
                if not any(root.iterdir()):  # Directory is empty
                    root.rmdir()
                    print(f"🗑️ Removed empty directory: {root.name}")
            except:
                pass  # Directory not empty or permission error


if __name__ == "__main__":
    main()
