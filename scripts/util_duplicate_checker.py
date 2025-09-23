#!/usr/bin/env python3
"""
Utility: Duplicate File Checker
================================
Find exact duplicate files across directories by comparing filenames.
Useful for verifying file integrity after operations like face grouping.

VIRTUAL ENVIRONMENT:
--------------------
Activate virtual environment first:
  source .venv311/bin/activate

USAGE:
------
Check for duplicates in project:
  python scripts/util_duplicate_checker.py

Check specific directory:
  python scripts/util_duplicate_checker.py --root-dir face_groups

FEATURES:
---------
• Scans all subdirectories recursively
• Finds files with identical names (not content comparison)
• Reports duplicate locations with full paths
• Supports PNG and YAML file filtering
• Clear summary statistics and actionable results
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


def find_all_files(root_dir: Path, extensions: List[str] = None) -> Dict[str, List[Path]]:
    """Find all files and group them by filename."""
    if extensions is None:
        extensions = ['.png', '.yaml']
    
    file_map = defaultdict(list)
    
    print(f"🔍 Scanning for files in: {root_dir}")
    
    for ext in extensions:
        pattern = f"**/*{ext}"
        for file_path in root_dir.rglob(pattern):
            if file_path.is_file():
                filename = file_path.name
                file_map[filename].append(file_path)
    
    return file_map


def find_duplicates(file_map: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
    """Find files that appear in multiple locations."""
    duplicates = {}
    
    for filename, paths in file_map.items():
        if len(paths) > 1:
            duplicates[filename] = paths
    
    return duplicates


def analyze_directories(root_dir: Path) -> None:
    """Analyze all directories for duplicate files."""
    print("🔍 DUPLICATE FILE CHECKER")
    print("=" * 50)
    print(f"📁 Root directory: {root_dir}")
    print()
    
    # Find all files
    file_map = find_all_files(root_dir)
    
    total_files = sum(len(paths) for paths in file_map.values())
    unique_filenames = len(file_map)
    
    print(f"📊 SUMMARY:")
    print(f"   • Total files found: {total_files}")
    print(f"   • Unique filenames: {unique_filenames}")
    print()
    
    # Find duplicates
    duplicates = find_duplicates(file_map)
    
    if not duplicates:
        print("✅ NO DUPLICATES FOUND!")
        print("   All files have unique names across all directories.")
        return
    
    print(f"⚠️  DUPLICATES FOUND: {len(duplicates)} filename(s) appear in multiple locations")
    print()
    
    # Group duplicates by directory pairs
    duplicate_pairs = defaultdict(list)
    
    for filename, paths in duplicates.items():
        # Sort paths to get consistent grouping
        sorted_paths = sorted(paths)
        for i in range(len(sorted_paths)):
            for j in range(i + 1, len(sorted_paths)):
                dir1 = sorted_paths[i].parent
                dir2 = sorted_paths[j].parent
                key = f"{dir1} ↔ {dir2}"
                duplicate_pairs[key].append((filename, sorted_paths[i], sorted_paths[j]))
    
    # Report duplicates by directory pairs
    for dir_pair, files in duplicate_pairs.items():
        print(f"📂 {dir_pair}")
        print(f"   {len(files)} duplicate file(s):")
        for filename, path1, path2 in sorted(files)[:10]:  # Show first 10
            print(f"   • {filename}")
            print(f"     - {path1}")
            print(f"     - {path2}")
        
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more")
        print()
    
    # Summary by directory
    print("📁 DUPLICATES BY DIRECTORY:")
    dir_counts = defaultdict(int)
    
    for filename, paths in duplicates.items():
        for path in paths:
            dir_counts[path.parent] += 1
    
    for directory, count in sorted(dir_counts.items()):
        print(f"   • {directory}: {count} duplicate files")


def main():
    parser = argparse.ArgumentParser(description="Find duplicate files across directories")
    parser.add_argument("root_dir", nargs="?", default=".", 
                       help="Root directory to scan (default: current directory)")
    parser.add_argument("--extensions", default="png,yaml",
                       help="Comma-separated file extensions to check (default: png,yaml)")
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[!] Directory not found: {root_dir}")
        sys.exit(1)
    
    extensions = [f".{ext.strip()}" for ext in args.extensions.split(",")]
    
    # Override the find_all_files function to use custom extensions
    global find_all_files
    original_find_all_files = find_all_files
    
    def find_all_files_custom(root_dir: Path, extensions_param=None):
        return original_find_all_files(root_dir, extensions)
    
    find_all_files = find_all_files_custom
    
    analyze_directories(root_dir)


if __name__ == "__main__":
    main()
