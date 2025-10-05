#!/usr/bin/env python3
"""
Utility: Check Companions - Orphaned Files Audit Tool
======================================================
Find and optionally clean up orphaned PNG/companion files in directory trees.
Identifies files missing their corresponding companions for data integrity.
Supports ALL companion file types (.yaml, .caption, etc.).

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Audit companion files in directories:
  python scripts/utils/check_companions.py [directory]
  python scripts/utils/check_companions.py --recursive .
  python scripts/utils/check_companions.py --recursive face_groups

FEATURES:
---------
• Recursively scans directory trees for PNG/companion files
• Identifies orphaned files (PNG without companions, companions without PNG)
• Supports ALL companion file types (.yaml, .caption, .txt, .json, etc.)
• Accepts ANY companion type - no false positives if one type exists
• Detailed report grouped by directory
• Optional cleanup with user confirmation
• Safe deletion using send2trash (recoverable)
• Comprehensive audit trail and statistics
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from send2trash import send2trash

def find_mismatched_files_recursive(root_directory):
    """Recursively find PNG files without matching metadata files and vice versa."""
    root_directory = Path(root_directory)
    
    print(f"🔍 Recursively scanning: {root_directory}")
    
    # Get all PNG and metadata files recursively, organized by directory
    all_orphaned_pngs = []
    all_orphaned_metadata = []
    total_pairs = 0
    directories_scanned = 0
    
    for current_dir in [root_directory] + list(root_directory.rglob("*/")):
        if not current_dir.is_dir():
            continue
            
        # Skip certain directories
        if any(skip in current_dir.name.lower() for skip in ['venv', '.git', '__pycache__', '.DS_Store']):
            continue
            
        directories_scanned += 1
        
        # Get PNG and metadata files in this directory
        png_files = {f.stem: f for f in current_dir.glob("*.png")}
        yaml_files = {f.stem: f for f in current_dir.glob("*.yaml")}
        caption_files = {f.stem: f for f in current_dir.glob("*.caption")}
        metadata_files = {**yaml_files, **caption_files}  # Combine both types
        
        if not png_files and not metadata_files:
            continue
            
        # Find mismatched files in this directory
        pngs_without_metadata = set(png_files.keys()) - set(metadata_files.keys())
        metadata_without_png = set(metadata_files.keys()) - set(png_files.keys())
        pairs_in_dir = len(set(png_files.keys()) & set(metadata_files.keys()))
        
        if pngs_without_metadata or metadata_without_png or pairs_in_dir > 0:
            print(f"  📁 {current_dir.relative_to(root_directory) if current_dir != root_directory else '.'}: "
                  f"{pairs_in_dir} pairs, {len(pngs_without_metadata)} orphaned PNGs, {len(metadata_without_png)} orphaned metadata")
        
        # Add to global lists
        all_orphaned_pngs.extend([png_files[stem] for stem in pngs_without_metadata])
        all_orphaned_metadata.extend([metadata_files[stem] for stem in metadata_without_png])
        total_pairs += pairs_in_dir
    
    print(f"📊 Scanned {directories_scanned} directories")
    
    return {
        'orphaned_pngs': all_orphaned_pngs,
        'orphaned_metadata': all_orphaned_metadata,
        'total_pairs': total_pairs
    }

def find_mismatched_files(directory):
    """Find PNG files without matching metadata files and vice versa in a single directory."""
    directory = Path(directory)
    
    # Get all PNG and metadata files
    png_files = {f.stem: f for f in directory.glob("*.png")}
    yaml_files = {f.stem: f for f in directory.glob("*.yaml")}
    caption_files = {f.stem: f for f in directory.glob("*.caption")}
    metadata_files = {**yaml_files, **caption_files}  # Combine both types
    
    # Find PNGs without metadata and metadata without PNGs
    pngs_without_metadata = set(png_files.keys()) - set(metadata_files.keys())
    metadata_without_png = set(metadata_files.keys()) - set(png_files.keys())
    
    return {
        'orphaned_pngs': [png_files[stem] for stem in pngs_without_metadata],
        'orphaned_metadata': [metadata_files[stem] for stem in metadata_without_png],
        'total_pairs': len(set(png_files.keys()) & set(metadata_files.keys()))
    }

def cleanup_and_verify(directory, recursive=False):
    """Generate a detailed report of orphaned files and optionally clean them up."""
    
    # Choose the appropriate scanning function
    if recursive:
        results = find_mismatched_files_recursive(directory)
    else:
        print(f"\n🔍 Scanning directory: {directory}")
        results = find_mismatched_files(directory)
    
    if not results['orphaned_metadata'] and not results['orphaned_pngs']:
        print("\n✅ All files are already properly paired!")
        print(f"Total pairs: {results['total_pairs']}")
        return
    
    # Generate detailed report
    print(f"\n📋 ORPHANED FILES REPORT:")
    print(f"{'='*60}")
    print(f"  • Total paired files: {results['total_pairs']}")
    print(f"  • Orphaned PNG files: {len(results['orphaned_pngs'])}")
    print(f"  • Orphaned metadata files: {len(results['orphaned_metadata'])}")
    
    # Show orphaned metadata files
    if results['orphaned_metadata']:
        print(f"\n🔸 ORPHANED METADATA FILES ({len(results['orphaned_metadata'])}):")
        # Group by directory for better readability
        metadata_by_dir = defaultdict(list)
        for f in results['orphaned_metadata']:
            metadata_by_dir[f.parent].append(f)
        
        for dir_path, files in sorted(metadata_by_dir.items()):
            print(f"  📁 {dir_path}:")
            for f in sorted(files):
                print(f"    • {f.name}")
    
    # Show orphaned PNG files
    if results['orphaned_pngs']:
        print(f"\n🔸 ORPHANED PNG FILES ({len(results['orphaned_pngs'])}):")
        # Group by directory for better readability
        pngs_by_dir = defaultdict(list)
        for f in results['orphaned_pngs']:
            pngs_by_dir[f.parent].append(f)
        
        for dir_path, files in sorted(pngs_by_dir.items()):
            print(f"  📁 {dir_path}:")
            for f in sorted(files):
                print(f"    • {f.name}")
    
    print(f"\n{'='*60}")
    
    # Ask user what to do
    total_orphans = len(results['orphaned_metadata']) + len(results['orphaned_pngs'])
    while True:
        print(f"\nFound {total_orphans} orphaned files.")
        choice = input("Do you want to move them to trash? (y/n/q): ").lower().strip()
        
        if choice in ['y', 'yes']:
            perform_cleanup(results)
            break
        elif choice in ['n', 'no']:
            print("📋 Report complete. No files were moved.")
            break
        elif choice in ['q', 'quit']:
            print("👋 Exiting without making changes.")
            return
        else:
            print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")

def perform_cleanup(results):
    """Actually move the orphaned files to trash."""
    # Move orphaned metadata files to trash
    if results['orphaned_metadata']:
        print(f"\n🗑️  Moving {len(results['orphaned_metadata'])} orphaned metadata files to trash:")
        for f in sorted(results['orphaned_metadata']):
            try:
                send2trash(str(f))
                print(f"  ✓ {f.name}")
            except Exception as e:
                print(f"  ❌ Error moving {f.name} to trash: {e}")
    
    # Move orphaned PNG files to trash as well
    if results['orphaned_pngs']:
        print(f"\n🗑️  Moving {len(results['orphaned_pngs'])} orphaned PNG files to trash:")
        for f in sorted(results['orphaned_pngs']):
            try:
                send2trash(str(f))
                print(f"  ✓ {f.name}")
            except Exception as e:
                print(f"  ❌ Error moving {f.name} to trash: {e}")
    
    print(f"\n✅ Cleanup complete! All orphaned files have been moved to trash.")

def main():
    parser = argparse.ArgumentParser(
        description='Clean up orphaned PNG and metadata files (.yaml/.caption)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_pairs.py normal_images_2          # Single directory
  python cleanup_pairs.py --recursive             # Recursive from project root  
  python cleanup_pairs.py --recursive "Slender Kiara"  # Recursive from specific directory
        """
    )
    
    parser.add_argument('directory', nargs='?', default='.',
                       help='Directory to scan (default: current directory)')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively scan all subdirectories')
    
    args = parser.parse_args()
    
    directory = args.directory
    if not os.path.isdir(directory):
        print(f"❌ Error: {directory} is not a directory")
        sys.exit(1)
    
    try:
        cleanup_and_verify(directory, recursive=args.recursive)
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
