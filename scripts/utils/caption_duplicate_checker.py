#!/usr/bin/env python3
"""
Caption Duplicate Checker
=========================
Analyzes .caption files in a directory to find identical content.
Useful for identifying duplicate prompts or similar descriptions.

USAGE:
------
  # Basic duplicate analysis
  python scripts/utils/caption_duplicate_checker.py mixed-0919/black
  python scripts/utils/caption_duplicate_checker.py sorted/unknown --show-content
  
  # Move duplicate groups to subdirectories
  python scripts/utils/caption_duplicate_checker.py mixed-0919/black --move-groups
  python scripts/utils/caption_duplicate_checker.py sorted/unknown --move-groups --dry-run

FEATURES:
---------
• Finds all .caption files in directory
• Groups files by identical content
• Shows duplicate groups with file counts
• Optional content preview for each group
• Summary statistics of duplicates vs unique files
• Move duplicate groups to numbered subdirectories
• Moves both .caption and .png files together
• Dry-run mode for safe testing
"""

import argparse
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def analyze_caption_duplicates(directory: str, show_content: bool = False) -> Dict:
    """
    Analyze .caption files for duplicate content.
    
    Args:
        directory: Directory to scan for .caption files
        show_content: Whether to show the actual content of each group
        
    Returns:
        Dictionary with analysis results
    """
    directory_path = Path(directory).resolve()
    if not directory_path.exists() or not directory_path.is_dir():
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Find all .caption files
    caption_files = list(directory_path.rglob('*.caption'))
    total_files = len(caption_files)
    
    print(f"🔍 Analyzing caption files in: {directory_path}")
    print(f"[*] Found {total_files} caption files")
    
    if total_files == 0:
        print("[!] No caption files found")
        return {'total_files': 0, 'unique_contents': 0, 'duplicate_groups': []}
    
    # Group files by content
    content_groups = defaultdict(list)
    read_errors = []
    
    for caption_file in caption_files:
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            content_groups[content].append(caption_file)
        except Exception as e:
            read_errors.append(f"Error reading {caption_file}: {e}")
    
    if read_errors:
        print(f"[!] {len(read_errors)} files had read errors:")
        for error in read_errors:
            print(f"    {error}")
    
    # Analyze results
    unique_contents = len(content_groups)
    duplicate_groups = []
    total_duplicates = 0
    
    print(f"\n📊 Analysis Results:")
    print(f"[*] Total files: {total_files}")
    print(f"[*] Unique contents: {unique_contents}")
    
    # Find duplicate groups (content appearing in multiple files)
    for content, files in content_groups.items():
        if len(files) > 1:
            duplicate_groups.append({
                'content': content,
                'files': files,
                'count': len(files)
            })
            total_duplicates += len(files)
    
    if duplicate_groups:
        print(f"[*] Duplicate groups: {len(duplicate_groups)}")
        print(f"[*] Files with duplicates: {total_duplicates}")
        print(f"[*] Unique files: {total_files - total_duplicates}")
        
        print(f"\n🔄 Duplicate Groups:")
        print("=" * 60)
        
        # Sort by count (most duplicates first)
        duplicate_groups.sort(key=lambda x: x['count'], reverse=True)
        
        for i, group in enumerate(duplicate_groups, 1):
            print(f"\nGroup {i}: {group['count']} files with identical content")
            
            if show_content:
                content_preview = group['content'][:100] + "..." if len(group['content']) > 100 else group['content']
                print(f"Content: \"{content_preview}\"")
            
            print("Files:")
            for file_path in group['files']:
                print(f"  • {file_path.name}")
    else:
        print(f"[*] No duplicates found - all {total_files} files have unique content! ✅")
    
    return {
        'total_files': total_files,
        'unique_contents': unique_contents,
        'duplicate_groups': duplicate_groups,
        'total_duplicates': total_duplicates,
        'unique_files': total_files - total_duplicates,
        'directory_path': directory_path
    }


def move_duplicate_groups_to_subdirs(analysis_results: Dict, dry_run: bool = False) -> Dict:
    """
    Move duplicate groups into numbered subdirectories within the same parent directory.
    Moves both .caption and corresponding .png files together.
    
    Args:
        analysis_results: Results from analyze_caption_duplicates()
        dry_run: Preview mode - don't actually move files
        
    Returns:
        Dictionary with move operation results
    """
    duplicate_groups = analysis_results['duplicate_groups']
    directory_path = analysis_results['directory_path']
    
    if not duplicate_groups:
        print("[*] No duplicate groups to move")
        return {'groups_moved': 0, 'files_moved': 0, 'errors': []}
    
    print(f"\n🚚 Moving duplicate groups to subdirectories...")
    print(f"[*] Target directory: {directory_path}")
    print(f"[*] Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    
    groups_moved = 0
    files_moved = 0
    errors = []
    
    for i, group in enumerate(duplicate_groups, 1):
        group_dir_name = f"duplicate_group_{i:03d}"
        group_dir_path = directory_path / group_dir_name
        
        content_preview = group['content'][:50] + "..." if len(group['content']) > 50 else group['content']
        print(f"\nGroup {i}: {group['count']} files → {group_dir_name}/")
        print(f"Content: \"{content_preview}\"")
        
        if not dry_run:
            try:
                group_dir_path.mkdir(exist_ok=True)
            except Exception as e:
                error_msg = f"Failed to create directory {group_dir_name}: {e}"
                errors.append(error_msg)
                print(f"[!] {error_msg}")
                continue
        
        group_files_moved = 0
        
        for caption_file in group['files']:
            try:
                # Move caption file
                caption_dest = group_dir_path / caption_file.name
                if not dry_run:
                    shutil.move(str(caption_file), str(caption_dest))
                print(f"  • {caption_file.name}")
                group_files_moved += 1
                
                # Move corresponding PNG file if it exists
                png_file = caption_file.with_suffix('.png')
                if png_file.exists():
                    png_dest = group_dir_path / png_file.name
                    if not dry_run:
                        shutil.move(str(png_file), str(png_dest))
                    print(f"  • {png_file.name}")
                    group_files_moved += 1
                else:
                    print(f"  • {png_file.name} (not found)")
                
            except Exception as e:
                error_msg = f"Failed to move {caption_file.name}: {e}"
                errors.append(error_msg)
                print(f"[!] {error_msg}")
        
        if group_files_moved > 0:
            groups_moved += 1
            files_moved += group_files_moved
    
    print(f"\n📊 Move Summary:")
    print(f"[*] Groups moved: {groups_moved}")
    print(f"[*] Files moved: {files_moved}")
    if errors:
        print(f"[*] Errors: {len(errors)}")
        for error in errors:
            print(f"    {error}")
    
    return {
        'groups_moved': groups_moved,
        'files_moved': files_moved,
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check for identical content in .caption files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic duplicate check
  python scripts/utils/caption_duplicate_checker.py mixed-0919/black
  
  # Show content preview for each duplicate group
  python scripts/utils/caption_duplicate_checker.py sorted/unknown --show-content
  
  # Check multiple directories
  python scripts/utils/caption_duplicate_checker.py character_group_1 --show-content
        """
    )
    
    parser.add_argument("directory", type=str, help="Directory to scan for .caption files")
    parser.add_argument("--show-content", "-c", action="store_true", 
                       help="Show content preview for each duplicate group")
    parser.add_argument("--move-groups", "-m", action="store_true",
                       help="Move duplicate groups to numbered subdirectories")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Preview move operations without actually moving files")
    
    args = parser.parse_args()
    
    try:
        results = analyze_caption_duplicates(args.directory, args.show_content)
        
        # Move duplicate groups if requested
        if args.move_groups:
            if results['duplicate_groups']:
                move_results = move_duplicate_groups_to_subdirs(results, args.dry_run)
                print(f"\n✅ Move operation completed!")
            else:
                print(f"\n[*] No duplicate groups to move")
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"[!] Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
